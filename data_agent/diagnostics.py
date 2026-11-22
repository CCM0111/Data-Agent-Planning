"""Small, reproducible validation summaries; no model fitting or slice search."""

import hashlib
import importlib
from importlib import metadata
import platform
import inspect
import math
import os
from functools import partial
from pathlib import Path

import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_environment():
    """Reproducibility facts only; never dump environment variables or credentials."""
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "xgboost", "lightgbm", "catboost"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.system(),
            "machine": platform.machine(), "packages": versions}


def split_identity(task):
    return {"split_id": task.get("split_id"), "target": task["target"],
            "problem_type": task["problem_type"], "seed": task["seed"],
            "train_sha256": file_hash(task["train_path"]),
            "validation_sha256": file_hash(task["validation_path"])}


def callable_reference(value):
    """Identify importable functions without treating stateful callable objects as functions."""
    if not (inspect.isfunction(value) or inspect.isbuiltin(value) or isinstance(value, np.ufunc)):
        return None
    name = getattr(value, "__qualname__", getattr(value, "__name__", ""))
    module_name = getattr(value, "__module__", None)
    # NumPy ufuncs do not expose __module__. Verify the exported object by identity.
    if module_name is None and isinstance(value, np.ufunc):
        module_name = "numpy"
    if not module_name or module_name in {"__main__", "stage_code"} or not name or "<" in name:
        return None
    try:
        exported = importlib.import_module(module_name)
        for component in name.split("."):
            exported = getattr(exported, component)
    except (ImportError, AttributeError):
        return None
    if exported is value:
        return {"type": "importable_callable", "module": module_name, "qualname": name}
    return None


def json_value(value):
    if isinstance(value, np.random.RandomState):
        return {"type": "numpy.random.RandomState", "state": json_value(value.get_state())}
    if isinstance(value, np.random.Generator):
        return {"type": "numpy.random.Generator", "bit_generator": type(value.bit_generator).__name__,
                "state": json_value(value.bit_generator.state)}
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if hasattr(value, "get_params"):
        return model_configuration(value)
    if type(value) is partial:
        return {"type": "functools.partial", "function": json_value(value.func),
                "args": json_value(value.args), "keywords": json_value(value.keywords)}
    if callable(value):
        reference = callable_reference(value)
        if reference is not None:
            return reference
        # Unknown callables may carry state even when their repr has no address.
        # Preserve the observation, but never certify a cross-process replay from it.
        return {"type": "opaque_callable", "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "repr": repr(value), "process_id": os.getpid(), "object_id": id(value)}
    return repr(value)


def model_configuration(model):
    return {"class": f"{type(model).__module__}.{type(model).__qualname__}",
            "parameters": json_value(model.get_params(deep=False))}


def validation_diagnostics(task, training, validation, targets, predictions):
    """Use two prespecified groups, whose thresholds depend only on training."""
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    missing = validation.isna().any(axis=1).to_numpy()
    rare = np.zeros(len(validation), dtype=bool)
    categorical = training.select_dtypes(include=["object", "category", "bool"]).columns
    threshold = max(2, math.ceil(len(training) * .01))
    for column in categorical:
        frequencies = training[column].value_counts(dropna=True)
        counts = validation[column].map(frequencies).fillna(0).to_numpy()
        rare |= (counts <= threshold) & validation[column].notna().to_numpy()
    masks = {"any_missing": missing, "no_missing": ~missing,
             "rare_or_unseen_category": rare, "no_rare_or_unseen_category": ~rare}
    groups = {}
    for name, mask in masks.items():
        size = int(mask.sum())
        entry = {"rows": size, "small_group": size < 20}
        if size:
            if task["problem_type"] == "classification":
                entry["error_rate"] = float(np.mean(targets[mask] != predictions[mask]))
            else:
                residual = targets[mask].astype(float) - predictions[mask].astype(float)
                entry.update(mae=float(np.mean(np.abs(residual))), mse=float(np.mean(residual ** 2)))
        groups[name] = entry
    return {"split": "validation", "rows": len(targets), "groups": groups,
            "group_policy": {"version": 1, "rare_training_count_at_most": threshold,
                             "minimum_reliable_group_rows": 20},
            "limitations": "Descriptive validation observations; small groups are noisy. "
                            "Groups overlap and do not establish causes or generalization."}
