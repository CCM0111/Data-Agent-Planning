"""Execute generated stage functions in a child process and score centrally."""

import argparse
import importlib.util
import json
from pathlib import Path
import random
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

if __package__:
    from .diagnostics import file_hash, model_configuration, split_identity, validation_diagnostics, runtime_environment
    from .compatibility import feature_profile, detect_input_mismatch
else:
    from diagnostics import file_hash, model_configuration, split_identity, validation_diagnostics, runtime_environment
    from compatibility import feature_profile, detect_input_mismatch


def read_xy(path, target):
    frame = pd.read_csv(path)
    return frame.drop(columns=[target]), frame[target]


def score(problem_type, y, predictions):
    if problem_type == "classification":
        return {"accuracy": float(accuracy_score(y, predictions)),
                "f1_macro": float(f1_score(y, predictions, average="macro"))}
    return {"mse": float(mean_squared_error(y, predictions)),
            "mae": float(mean_absolute_error(y, predictions)),
            "r2": float(r2_score(y, predictions))}


def initialize_seed(seed):
    """Seed common RNGs; generated code and third-party estimators can still be nondeterministic."""
    random.seed(seed)
    np.random.seed(seed)
    return {"seed": seed, "seed_policy": "python.random and numpy global RNGs before module import"}


def validate_feature_outputs(train_features, validation_features, training, validation):
    """Check observable alignment before features reach a model.

    Arrays and sparse matrices have no row labels; their row-order contract
    still depends on the transformer. DataFrame indices/columns allow stricter
    checks but cannot detect a shuffle followed by a reset_index either.
    """
    for name, features, source in (("train", train_features, training),
                                   ("validation", validation_features, validation)):
        if len(getattr(features, "shape", ())) != 2:
            raise ValueError(f"FE {name} output must be a two-dimensional feature matrix")
        if features.shape[0] != len(source):
            raise ValueError(f"FE {name} output must preserve row count")
        if isinstance(features, pd.DataFrame) and not features.index.equals(source.index):
            raise ValueError(f"FE {name} output changed row index or order")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("FE train and validation outputs must have the same feature count")
    if (isinstance(train_features, pd.DataFrame) and isinstance(validation_features, pd.DataFrame)
            and not train_features.columns.equals(validation_features.columns)):
        raise ValueError("FE train and validation feature columns must match in name and order")


def execute(context, workspace):
    task, stage = context["task"], context["stage"]
    seed_state = initialize_seed(task["seed"])
    state_path = workspace / "execution_state.json"
    state = {**seed_state, "fit_started": False, "fit_completed": False}

    def phase(name, **updates):
        state.update(phase=name, **updates)
        state_path.write_text(json.dumps(state))

    phase("import")
    spec = importlib.util.spec_from_file_location("stage_code", workspace / "code.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    phase("read_data")
    X, y = read_xy(task["train_path"], task["target"])
    if stage == "EDA":
        phase("analyze")
        report = module.analyze(X, y, str(workspace))
        if not isinstance(report, dict):
            raise TypeError("EDA analyze must return a dict")
        phase("serialize")
        (workspace / "eda.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return {"report": report, "execution": state,
                "artifacts": {"eda": str(workspace / "eda.json")}}

    eda = json.loads(Path(context["eda_path"]).read_text())
    X_val, y_val = read_xy(task["validation_path"], task["target"])
    if stage == "FE":
        phase("build_transformer")
        transformer = module.build_transformer(task, eda)
        transformer_config = model_configuration(transformer)
        (workspace / "transformer_config.json").write_text(
            json.dumps(transformer_config, indent=2, allow_nan=False))
        phase("fit_transform")
        train_features = transformer.fit_transform(X, y)
        phase("transform")
        validation_features = transformer.transform(X_val)
        phase("validate_features")
        validate_feature_outputs(train_features, validation_features, X, X_val)
        phase("serialize")
        joblib.dump(transformer, workspace / "transformer.joblib")
        joblib.dump({"train": train_features, "validation": validation_features}, workspace / "features.joblib")
        return {"train_shape": list(train_features.shape), "validation_shape": list(validation_features.shape),
                "feature_profiles": {"train": feature_profile(train_features),
                                     "validation": feature_profile(validation_features)},
                "execution": state, "transformer_config": transformer_config,
                "artifacts": {"transformer": str(workspace / "transformer.joblib"),
                              "features": str(workspace / "features.joblib")}}

    phase("load_features")
    features = joblib.load(context["features_path"])
    phase("build_model")
    model = module.build_model(task, eda)
    configuration = model_configuration(model)
    (workspace / "model_config.json").write_text(json.dumps(configuration, indent=2, allow_nan=False))
    phase("verify_model_config")
    expected = context.get("expected_model_config")
    if expected is not None and configuration != expected:
        raise ValueError("Replay model configuration differs from the reference; refusing a changed algorithm")
    phase("fit", fit_started=True)
    started = time.monotonic()
    try:
        model.fit(features["train"], y)
    except (ValueError, TypeError) as error:
        save_compatibility_error(error, "fit", features["train"], workspace)
        raise
    phase("predict", fit_completed=True, fit_elapsed_seconds=time.monotonic() - started)
    try:
        predictions = np.asarray(model.predict(features["validation"]))
    except (ValueError, TypeError) as error:
        save_compatibility_error(error, "predict", features["validation"], workspace)
        raise
    phase("evaluate")
    if predictions.ndim != 1 or len(predictions) != len(y_val):
        raise ValueError("MT predictions must contain one value per validation row")
    metrics = score(task["problem_type"], y_val, predictions)
    phase("serialize")
    identity = split_identity(task)
    prediction_path = workspace / "validation_predictions.joblib"
    joblib.dump({"split_identity": identity, "row_id": np.arange(len(y_val)),
                 "target": y_val.to_numpy(), "prediction": predictions,
                 "node_id": context.get("node_id"), "attempt_id": context.get("attempt_id")}, prediction_path)
    diagnostics = validation_diagnostics(task, X, X_val, y_val, predictions)
    diagnostic_path = workspace / "validation_diagnostics.json"
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2, allow_nan=False))
    joblib.dump(model, workspace / "model.joblib")
    return {"metrics": metrics, "model_config": configuration, "split_identity": identity,
            "runtime_environment": runtime_environment(),
            "model_code_hash": file_hash(workspace / "code.py"), "diagnostics": diagnostics,
            "execution": state,
            "artifacts": {"model": str(workspace / "model.joblib"),
                          "predictions": str(prediction_path), "diagnostics": str(diagnostic_path)}}


def save_compatibility_error(error, phase, features, workspace):
    evidence = detect_input_mismatch(error, feature_profile(features), phase)
    if evidence:
        (workspace / "compatibility.json").write_text(json.dumps(evidence, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("context", type=Path)
    args = parser.parse_args()
    context = json.loads(args.context.read_text())
    workspace = args.context.resolve().parent
    try:
        if context["stage"] == "TEST":
            initialize_seed(context["task"]["seed"])
            X, y = read_xy(context["test_path"], context["task"]["target"])
            transformer = joblib.load(context["transformer_path"])
            model = joblib.load(context["model_path"])
            result = {"metrics": score(context["task"]["problem_type"], y, model.predict(transformer.transform(X)))}
        else:
            result = execute(context, workspace)
        (workspace / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except Exception as error:
        state_path = workspace / "execution_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        (workspace / "error.json").write_text(json.dumps({
            "kind": "execution_error", "phase": state.get("phase", "test_evaluation" if context["stage"] == "TEST" else "runtime"),
            "exception_type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()}, ensure_ascii=False, indent=2))
        raise


if __name__ == "__main__":
    main()
