"""Trace snapshots of observed errors and pipeline context, without causal labels."""

from copy import deepcopy
from pathlib import Path


def describe_error(error, kind, phase):
    return {"kind": kind, "phase": phase, "exception_type": type(error).__name__,
            "message": str(error)}


def path_errors(nodes, rollout_id=None):
    """Include ancestor repairs; repeated references keep the same per-run error ID."""
    errors = []
    for node in nodes:
        for index, attempt in enumerate(node.attempts):
            detail = attempt.get("error_details")
            message = attempt.get("error")
            if not detail and not message:
                continue
            if rollout_id is not None:
                attempt.setdefault("first_observed_rollout", rollout_id)
            attempt_id = attempt.get("attempt_id", index)
            later = node.attempts[index + 1:]
            recovery = next((a for a in later if a.get("status") == "success"), None)
            errors.append({
                "error_id": f"node_{node.id}:attempt_{attempt_id}",
                "node_id": node.id, "stage": node.stage, "attempt_id": attempt_id,
                "first_observed_rollout": attempt.get("first_observed_rollout"),
                **deepcopy(detail or {"kind": "execution_error", "phase": "unknown",
                                     "exception_type": None, "message": message}),
                "code_path": attempt.get("code_path"), "code_hash": attempt.get("code_hash"),
                "log_path": attempt.get("log_path"), "failure_path": attempt.get("failure_path"),
                "compatibility_kind": attempt.get("compatibility", {}).get("kind"),
                "outcome_after_error": "later_attempt_succeeded" if recovery else
                    "later_attempt_failed" if later and node.status == "failed" else
                    "interrupted" if node.status == "pending" else "not_recovered",
                "subsequent_attempts": [{k: a.get(k) for k in
                    ("attempt_id", "status", "code_path", "code_hash")} for a in later],
            })
    return errors


def pipeline_context(task, nodes, output):
    by_stage = {n.stage: n for n in nodes}
    eda, fe, mt = (by_stage.get(stage) for stage in ("EDA", "FE", "MT"))
    output = Path(output).resolve()
    return deepcopy({
        "run_dir": str(output),
        "task": {k: task[k] for k in ("name", "problem_type", "target", "metric", "seed",
                                     "train_rows", "validation_rows", "columns", "split_id") if k in task},
        "task_path": str(output / "task.json") if (output / "task.json").exists() else None,
        "config_path": str(output / "config.yaml") if (output / "config.yaml").exists() else None,
        "eda_node_id": eda.id if eda else None,
        "eda_report_path": eda.result.get("artifacts", {}).get("eda") if eda else None,
        "fe_node_id": fe.id if fe else None, "fe_code_hash": fe.code_hash if fe else None,
        "feature_profiles": fe.result.get("feature_profiles") if fe else None,
        "transformer_config": fe.result.get("transformer_config") if fe else None,
        "mt_node_id": mt.id if mt else None, "model_config": mt.result.get("model_config") if mt else None,
        "split_identity": mt.result.get("split_identity") if mt else None,
    })
