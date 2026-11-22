"""Fixed data splits and task-local rewards."""

import json
import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .data_profile import training_profile


METRICS = {"regression": {"mse", "mae", "r2"}, "classification": {"accuracy", "f1_macro"}}


def reward_for(metric, value):
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {metric}: {value}")
    if metric in {"mse", "mae"}:
        return 1 / (1 + value)
    if metric == "r2":
        return 1 / (2 - value)
    return value


def prepare_task(task_file, output):
    task_file, output = Path(task_file).resolve(), Path(output).resolve()
    task = json.loads(task_file.read_text())
    if task["metric"] not in METRICS[task["problem_type"]]:
        raise ValueError("Metric does not match problem_type")
    frame = pd.read_csv(task_file.parent / task["data_path"])
    target = task["target"]
    seed = task["seed"]
    split = task["split"]
    if not (0 < split["validation"] < 1 and 0 < split["test"] < 1
            and split["validation"] + split["test"] < 1):
        raise ValueError("Validation and test fractions must be positive and sum to less than one")
    stratify = frame[target] if task["problem_type"] == "classification" else None
    training, test = train_test_split(frame, test_size=split["test"], random_state=seed, stratify=stratify)
    stratify = training[target] if task["problem_type"] == "classification" else None
    train, validation = train_test_split(
        training, test_size=split["validation"] / (1 - split["test"]),
        random_state=seed, stratify=stratify,
    )
    output.mkdir(parents=True)
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        part.to_csv(output / f"{name}.csv", index=False)
    (output / "split_indices.json").write_text(json.dumps({
        "train": train.index.tolist(), "validation": validation.index.tolist(), "test": test.index.tolist(),
    }, indent=2))
    # Only training-data metadata enters agent prompts; test data is evaluated after search.
    context = {k: task[k] for k in ["name", "problem_type", "target", "metric", "seed"]}
    if task.get("task_id"):
        context["task_id"] = task["task_id"]
    from .task_similarity import task_semantics
    context.update({k: v for k, v in task_semantics(task).items() if v is not None})
    context.update({
        "description": task.get("description", ""),
        "train_path": str(output / "train.csv"),
        "validation_path": str(output / "validation.csv"),
        "columns": {column: str(dtype) for column, dtype in train.dtypes.items()},
        "train_rows": len(train), "validation_rows": len(validation),
        "data_profile": training_profile(train, target, task["problem_type"]),
    })
    return context, output / "test.csv"
