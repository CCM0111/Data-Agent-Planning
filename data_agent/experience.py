"""Deterministic, bounded records for later experience retrieval.

Read recorded facts only: no model loading, training, code analysis, or LLM calls.
The optional natural-language experience is deliberately left unset.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from .config_summary import compare_configs, summarize_config
from .ideas import compact_idea


DEFAULT_MAX_CHARS = 6000
PROFILE_FIELDS = (
    "schema_version", "scope", "train_rows", "feature_count", "numeric_features",
    "categorical_features", "other_features", "missing_cell_fraction",
    "rows_with_missing_fraction", "classes", "class_count_max_min_ratio",
)
COMPARISON_FIELDS = (
    "kind", "performance_comparable", "changed_scope", "fixed_model_configuration",
    "reference_status", "candidate_status", "metric", "reference_value", "candidate_value",
    "raw_delta", "improvement", "recovered", "reason", "limitations", "repaired", "damaged",
    "unchanged_correct", "unchanged_wrong", "count_interpretation", "improved_rows",
    "worsened_rows", "unchanged_rows", "mean_row_loss_reduction", "row_loss",
)


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _snapshot(node):
    return node if isinstance(node, dict) else vars(node)


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _select(source, keys, limit=160):
    """Only primitive whitelisted facts; never copy nested free-form reports."""
    result = {}
    for key in keys:
        value = _mapping(source).get(key)
        if key not in _mapping(source):
            continue
        if isinstance(value, str):
            result[key] = value if len(value) <= limit else value[:limit - 1] + "…"
        elif value is None or isinstance(value, bool) or _number(value):
            result[key] = value
        elif isinstance(value, float):
            result[key] = None
    return result


def _profile(task):
    profile = _mapping(task.get("data_profile"))
    if not profile:
        return None
    result = _select(profile, PROFILE_FIELDS)
    if isinstance(profile.get("target_summary"), dict):
        result["target_summary"] = _select(profile["target_summary"], (
            "finite_rows", "missing_rows", "nonfinite_rows", "mean", "std", "min", "max"))
    return result


def _stage(nodes, stage):
    return next((n for n in reversed(nodes) if n.get("stage") == stage), {})


def _config(node, kind):
    return _mapping(node.get("result")).get(kind)


def _representation(fe):
    result = _mapping(fe.get("result"))
    profile = _mapping(_mapping(result.get("feature_profiles")).get("train"))
    compact = _select(profile, (
        "dtype", "sparse", "format", "has_nan", "has_inf", "has_negative"))
    shape = profile.get("shape", result.get("train_shape"))
    if isinstance(shape, (list, tuple)) and len(shape) == 2 and all(_number(v) for v in shape):
        compact["shape"] = list(shape)
    return compact or None


def _attempts(node):
    return [item for item in node.get("attempts", []) if isinstance(item, dict)]


def _actual_attempt(node):
    for attempt in reversed(_attempts(node)):
        if ((node.get("code_path") and attempt.get("code_path") == node["code_path"])
                or (node.get("code_hash") and attempt.get("code_hash") == node["code_hash"]
                    and attempt.get("code_path"))):
            return attempt.get("attempt_id")
    # Historical trees lack attempts but retain the actual attempt's code path.
    match = re.search(r"(?:^|/)attempt_(\d+)/[^/]+$", node.get("code_path") or "")
    return int(match.group(1)) if match else None


def _error(node):
    if node.get("status") != "failed":
        return None
    attempts = _attempts(node)
    last = attempts[-1] if attempts else {}
    detail = _select(last.get("error_details"), ("kind", "phase", "exception_type", "message"), 240)
    if not detail:
        lines = str(last.get("error") or node.get("feedback") or "").strip().splitlines()
        detail = {"message": lines[-1][:240]} if lines else {}
    return detail or None


def _bound(record, max_chars):
    """Keep identity and measured outcome; explicitly mark every budget omission."""
    meta = record["summary_meta"]

    def omitted(path):
        meta["truncated"] = True
        if path not in meta["omitted_fields"]:
            meta["omitted_fields"].append(path)

    for scope in ("fe", "model"):
        delta = record["result"]["parameter_changes"][scope]
        if delta.get("truncated"):
            omitted(f"result.parameter_changes.{scope}.changes")
    for path, config in (("context.fe.configuration", record["context"]["fe"]["configuration"]),
                         ("context.model", record["context"]["model"])):
        if config and config.get("truncated"):
            omitted(path)
    # Full config differences are computed first, then shortened for retrieval.
    for scope in ("fe", "model"):
        delta = record["result"]["parameter_changes"][scope]
        changes = delta.get("changes", [])
        while len(changes) > 8:
            changes.pop()
            delta["truncated"] = True
            delta["omitted_changes"] = delta.get("omitted_changes", 0) + 1
            omitted(f"result.parameter_changes.{scope}.changes")
    for parent, key, path in (
        (record["context"]["fe"], "configuration", "context.fe.configuration"),
        (record["context"], "model", "context.model"),
    ):
        if len(_json(record)) <= max_chars:
            break
        if parent[key] is not None:
            parent[key] = {"class": str(parent[key].get("class", ""))[:120], "truncated": True}
            omitted(path)
    for scope in ("fe", "model"):
        delta = record["result"]["parameter_changes"][scope]
        changes = delta.get("changes", [])
        while len(_json(record)) > max_chars and len(changes) > 1:
            changes.pop()
            delta["truncated"] = True
            delta["omitted_changes"] = delta.get("omitted_changes", 0) + 1
            omitted(f"result.parameter_changes.{scope}.changes")
    # A particularly verbose idea must not displace validation evidence.
    for key in ("reason", "evidence", "method"):
        if len(_json(record)) <= max_chars:
            break
        value = record["idea"][key]
        if isinstance(value, str) and len(value) > 60:
            record["idea"][key] = value[:59] + "…"
            omitted(f"idea.{key}")
    for parent, key, path in (
        (record["context"]["task"], "description", "context.task.description"),
        (record["result"], "comparison", "result.comparison"),
        (record["context"], "data", "context.data"),
        (record["context"], "fe", "context.fe"),
        (record["context"], "model", "context.model"),
        (record["result"], "execution", "result.execution"),
    ):
        if len(_json(record)) <= max_chars:
            break
        if key in parent and parent[key] is not None:
            parent[key] = None
            omitted(path)
    if len(_json(record)) > max_chars:
        # Minimum-budget fallback retains outcome, source identity, and schema.
        record["context"]["task"] = _select(record["context"]["task"], ("problem_type", "metric"), 40)
        record["source_refs"] = {k: v for k, v in record["source_refs"].items()
                                 if k in {"run_id", "node_id", "rollout_id", "tree_path", "attempt_id"}}
        record["result"]["parameter_changes"] = {
            "fe": {"status": "omitted"}, "model": {"status": "omitted"}}
        meta["omitted_fields"] = ["budget_fallback:context,source_refs,parameter_changes"]
        meta["truncated"] = True
    if len(_json(record)) > max_chars:
        raise ValueError("Record identity and outcome exceed max_chars")
    return record


def build_record(task, node, ancestors, *, run_id, rollout_id=None, reference=None,
                 reference_ancestors=None, max_chars=DEFAULT_MAX_CHARS):
    """Build a code-free record for one executed stage, not one reused rollout."""
    if max_chars < 2000:
        raise ValueError("max_chars must be at least 2000")
    node = _snapshot(node)
    if node.get("stage") not in {"EDA", "FE", "MT"} or node.get("status") not in {"success", "failed"}:
        raise ValueError("Only completed EDA/FE/MT nodes have experience records")
    path = [_snapshot(n) for n in ancestors] + [node]
    reference = _snapshot(reference) if reference is not None else {}
    reference_path = [_snapshot(n) for n in (reference_ancestors or [])] + [reference]
    fe, ref_fe = _stage(path, "FE"), _stage(reference_path, "FE")
    mt, ref_mt = _stage(path, "MT"), _stage(reference_path, "MT")
    result = _mapping(node.get("result"))
    idea, omitted, normalization = compact_idea(node)
    metric = task.get("metric")
    value = _mapping(result.get("metrics")).get(metric)
    validation = None
    if node["stage"] == "MT" and node["status"] == "success" and _number(value):
        validation = {"split": "validation", "metric": metric,
                      "direction": "minimize" if metric in {"mse", "mae"} else "maximize", "value": value}
    comparison = (_select(node.get("comparison"), COMPARISON_FIELDS, 200) or None
                  if node["stage"] == "MT" else None)
    # Failed/incomparable trials may describe recovery but cannot imply score gains.
    if comparison and (node["status"] != "success" or comparison.get("performance_comparable") is not True):
        for key in ("raw_delta", "improvement", "reference_value", "candidate_value"):
            comparison.pop(key, None)
    attempts = _attempts(node)
    record = {
        "schema_version": 3, "stage": node["stage"], "idea": idea,
        "context": {
            "task": _select(task, ("name", "problem_type", "target", "metric", "description",
                                   "train_rows", "validation_rows")),
            "data": _profile(task),
            "fe": {"configuration": summarize_config(_config(fe, "transformer_config")),
                   "representation": _representation(fe)},
            "model": summarize_config(_config(mt, "model_config")),
        },
        "result": {
            "status": node["status"], "validation": validation,
            "execution": _select(result.get("execution"), (
                "attempt_count", "fit_attempt_count", "elapsed_seconds", "phase", "fit_started", "fit_completed")),
            "error": _error(node), "comparison": comparison,
            "parameter_changes": {
                "fe": compare_configs(_config(ref_fe, "transformer_config"), _config(fe, "transformer_config")),
                "model": compare_configs(_config(ref_mt, "model_config"), _config(mt, "model_config")),
            },
        },
        "experience": None,
        "source_refs": {
            "task_id": str(task.get("name", ""))[:160] or None, "run_id": str(run_id)[:160],
            "node_id": node["id"], "rollout_id": rollout_id,
            "ancestor_node_ids": [n["id"] for n in path[:-1] if n.get("stage") != "ROOT"],
            "reference_node_id": reference.get("id", node.get("based_on")),
            "reference_mt_id": node.get("reference_mt_id"),
            "reference_rollout_id": node.get("reference_rollout_id"),
            "split_id": str(task["split_id"])[:160] if task.get("split_id") is not None else None,
            "code_hash": str(node.get("code_hash") or "")[:80] or None,
            "attempt_id": _actual_attempt(node),
            "final_attempt_id": attempts[-1].get("attempt_id") if attempts else None,
            "tree_path": "tree.json", "node_path": f"node_{node['id']:04d}_{node['stage']}",
            "rollout_path": f"rollouts/rollout_{rollout_id:04d}.json" if rollout_id is not None else None,
        },
        "summary_meta": {"truncated": bool(omitted), "omitted_fields": omitted,
                         "max_chars": max_chars, "data_profile_available": bool(_profile(task))},
    }
    if normalization:
        record["summary_meta"]["idea_normalization"] = normalization
    # Whitelisted text is bounded independently of the overall record budget.
    for path_name, original, compact in (
        ("context.task", task, record["context"]["task"]),
        ("result.comparison", _mapping(node.get("comparison")), comparison or {}),
        ("result.error", _mapping(attempts[-1].get("error_details")) if attempts else {},
         record["result"]["error"] or {}),
    ):
        for key, value in compact.items():
            if isinstance(original.get(key), str) and value != original[key]:
                omitted.append(f"{path_name}.{key}")
    record["summary_meta"]["truncated"] = bool(omitted)
    return _bound(record, max_chars)


def _ancestors(node, by_id):
    ancestors, visited = [], {node["id"]}
    parent = node.get("parent")
    while parent is not None:
        parent_id = parent if isinstance(parent, int) else _snapshot(parent)["id"]
        if parent_id in visited:
            raise ValueError("Cycle in saved search tree")
        visited.add(parent_id)
        ancestor = by_id.get(parent_id)
        if ancestor is None:
            break
        ancestors.append(ancestor)
        parent = ancestor.get("parent")
    return list(reversed(ancestors))


def _run_id(directory):
    directory = Path(directory).resolve()
    return directory.name[:100] + ":" + hashlib.sha256(str(directory).encode()).hexdigest()[:12]


def save_records(run_dir, task, nodes, rollout_id=None, seen=None, *, output_dir=None,
                 rollout_ids=None, max_chars=DEFAULT_MAX_CHARS):
    """Persist each completed node once. Shared ancestors are not new evidence."""
    directory = Path(output_dir) if output_dir is not None else Path(run_dir) / "experience_records"
    snapshots = [_snapshot(n) for n in nodes]
    by_id = {n["id"]: n for n in snapshots}
    seen = set() if seen is None else seen
    paths = []
    for node in snapshots:
        if (node["id"] in seen or node.get("stage") not in {"EDA", "FE", "MT"}
                or node.get("status") not in {"success", "failed"}):
            continue
        reference = by_id.get(node.get("based_on"))
        record = build_record(task, node, _ancestors(node, by_id), run_id=_run_id(run_dir),
                              rollout_id=(rollout_ids or {}).get(node["id"], rollout_id), reference=reference,
                              reference_ancestors=_ancestors(reference, by_id) if reference else [],
                              max_chars=max_chars)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"node_{node['id']:04d}_{node['stage']}.json"
        path.write_text(_json(record) + "\n", encoding="utf-8")
        paths.append(path)
        seen.add(node["id"])
    return paths


def export_run(run_dir, output_dir, max_chars=DEFAULT_MAX_CHARS):
    """Backfill historical JSON artifacts without executing their code/models."""
    run_dir = Path(run_dir)
    # Node IDs are local to a run. Refuse cross-run filename collisions before
    # writing anything, including when the two trees currently have disjoint IDs.
    for path in Path(output_dir).glob("node_*_*.json"):
        existing = json.loads(path.read_text())
        if _mapping(_mapping(existing).get("source_refs")).get("run_id") != _run_id(run_dir):
            raise ValueError("Output contains records from a different run; use a separate output directory")
    task = json.loads((run_dir / "task.json").read_text())
    if not task.get("data_profile") and (run_dir / "data_profile.json").exists():
        task["data_profile"] = json.loads((run_dir / "data_profile.json").read_text())
    nodes = json.loads((run_dir / "tree.json").read_text())
    rollout_ids = {}
    for path in sorted((run_dir / "rollouts").glob("rollout_*.json")):
        rollout = json.loads(path.read_text())
        if rollout.get("status", "completed") != "completed":
            continue
        for node in rollout.get("path", []):
            rollout_ids.setdefault(node["node_id"], rollout.get("rollout_id"))
    return save_records(run_dir, task, nodes, output_dir=output_dir,
                        rollout_ids=rollout_ids, max_chars=max_chars)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    args = parser.parse_args()
    paths = export_run(args.run_dir, args.output_dir, args.max_chars)
    print(f"Wrote {len(paths)} deterministic records to {args.output_dir}")


if __name__ == "__main__":
    main()
