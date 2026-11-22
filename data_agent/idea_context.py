"""Deterministic, code-free views for planning; call once on raw observations.

These reducers never read artifacts or infer executed methods from an idea. They
select known fields, bound evidence details, and mark omissions. Issue identities
remain complete for the caller's review bookkeeping.
Coding and repair callers should continue to use their full implementation input.
"""

import json
import math
from numbers import Integral, Real
import re

from .config_summary import summarize_config
from .ideas import IDEA_FIELDS, IDEA_LIMITS


MAX_HISTORY = 12
MAX_PREFIX = 3
MAX_ISSUES = 8
MAX_FEATURES = 24
MAX_EDA_FEATURES = 8
MAX_GROUPS = 8
MAX_ERRORS = 3
MAX_CONFIG_COMPONENTS = 8
MAX_CONFIG_CHARACTERS = 3200
_SOURCE = re.compile(r"(?:^|[\n;])\s*(?:import\s|from\s+\S+\s+import\s|def\s+\w+\s*\(|class\s+\w+\s*[:(])|\blambda\b[^\n:]*:")
_EXCEPTION = re.compile(r"(?:^|\s)([\w.]*(?:Error|Exception|TimeoutExpired))\s*:")
_BLOCKED = frozenset({"code", "source", "source_code", "repr", "traceback", "artifacts",
                      "execution_feedback", "execution_log", "log", "code_path", "error_path"})
_CONFIG_PARAMETERS = 10


def _dict(value):
    return value if isinstance(value, dict) else {}


def _number(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        value = int(value)
        return value if value.bit_length() <= 256 else None
    if isinstance(value, Real):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def _text(value, limit=160):
    if not isinstance(value, str):
        return None
    if _SOURCE.search(value):
        return "[unknown: source text omitted]"
    value = " ".join(value.split())
    return value if len(value) <= limit else value[:limit - 1] + "…"


def _name(value, limit=64):
    if not isinstance(value, str) or value in _BLOCKED or _SOURCE.search(value):
        return None
    return _text(value, limit)


def _take_fields(source, numbers=(), texts=()):
    source = _dict(source)
    result = {key: _number(source[key]) for key in numbers if key in source}
    result.update({key: _text(source[key]) for key in texts if key in source})
    return result


def _meta(result, **values):
    result.setdefault("summary_meta", {}).update(values)
    return result


def _numeric_map(raw, limit, *, nested=(), omissions=None):
    if not isinstance(raw, dict):
        return {"status": "unknown"}
    entries = {}
    # Insertion order is the report's recorded order; no value-based slice search.
    for key, value in raw.items():
        name = _name(key)
        if name is None:
            continue
        if len(entries) == limit:
            break
        item = _take_fields(value, numbers=nested) if nested else _number(value)
        if nested and omissions is not None and isinstance(value, dict):
            omissions["fields"] = omissions.get("fields", 0) + len(value) - len(item)
        entries[name] = item
    return entries


def _names(raw, limit=MAX_FEATURES):
    if not isinstance(raw, (list, tuple)):
        return {"status": "unknown"}
    values = [_name(value) for value in raw[:limit]]
    values = [value for value in values if value is not None]
    return {"values": values, "omitted_entries": len(raw) - len(values)}


def _ids(raw, limit=MAX_HISTORY):
    if not isinstance(raw, (list, tuple)):
        return []
    return [int(value) for value in raw[:limit] if isinstance(value, Integral)
            and not isinstance(value, bool) and int(value).bit_length() <= 256]


def _idea(raw, method=None):
    raw = _dict(raw)
    result = {key: _text(raw.get(key), IDEA_LIMITS[key]) for key in IDEA_FIELDS}
    if result.get("method") is None and method is not None:
        result["method"] = _text(method, IDEA_LIMITS["method"])
    missing = [key for key in IDEA_FIELDS if result[key] is None]
    truncated = [key for key in IDEA_FIELDS if isinstance(raw.get(key), str)
                 and len(raw[key]) > IDEA_LIMITS[key]]
    if missing or truncated:
        _meta(result, unknown_fields=missing, truncated_fields=truncated)
    return result


def _value(value, depth=0):
    """Bound already summarized parameter values, without preserving raw objects."""
    if value is None or isinstance(value, (Real, bool)):
        return _number(value)
    if isinstance(value, str):
        return _text(value, 80)
    if depth >= 2:
        return {"status": "unknown", "reason": "parameter_depth_limit"}
    if isinstance(value, dict):
        if value.get("type") in {"opaque_callable", "numpy.random.RandomState", "numpy.random.Generator"}:
            return {"status": "unknown", "type": _text(value["type"], 48)}
        result = {}
        for key, item in value.items():
            name = _name(key, 48)
            if name is not None and len(result) < 4:
                result[name] = _value(item, depth + 1)
        if len(result) < len(value):
            result["omitted_items"] = len(value) - len(result)
        return result
    if isinstance(value, (list, tuple)):
        result = {"values": [_value(item, depth + 1) for item in value[:4]]}
        if len(value) > 4:
            result["omitted_items"] = len(value) - 4
        return result
    return {"status": "unknown", "reason": "unsupported_parameter_value"}


def _configuration(raw):
    try:
        summary = summarize_config(raw)
    except (TypeError, ValueError, OverflowError, RecursionError):
        summary = None
    if summary is None:
        return {"status": "unknown", "reason": "missing_or_unrecognized_configuration"}
    state = {"components": 0, "truncated": bool(summary.get("truncated"))}

    def component(value, depth=0):
        if not isinstance(value, dict) or "class" not in value:
            return _value(value)
        if depth >= 4 or state["components"] >= MAX_CONFIG_COMPONENTS:
            state["truncated"] = True
            return {"class": _text(value.get("class"), 120), "status": "unknown", "truncated": True}
        state["components"] += 1
        parameters = _dict(value.get("parameters"))
        result = {"class": _text(value.get("class"), 120),
                  "parameters": {key: _value(item) for key, item in list(parameters.items())[:_CONFIG_PARAMETERS]}}
        omitted = len(parameters) - len(result["parameters"]) + int(value.get("omitted_parameters", 0))
        if omitted:
            result["omitted_parameters"] = omitted
        for key in ("steps", "transformers", "transformer_list"):
            branches = value.get(key)
            if not isinstance(branches, list):
                continue
            result[key] = []
            for branch in branches:
                if state["components"] >= MAX_CONFIG_COMPONENTS or len(result[key]) >= MAX_CONFIG_COMPONENTS:
                    state["truncated"] = True
                    break
                branch = _dict(branch)
                entry = {"name": _text(branch.get("name"), 48),
                         "component": component(branch.get("component"), depth + 1)}
                if "columns" in branch:
                    entry["columns"] = _take_fields(branch["columns"], numbers=("count", "mask_length"),
                                                   texts=("selection",))
                result[key].append(entry)
            omitted = len(branches) - len(result[key]) + int(value.get(f"omitted_{key}", 0))
            if omitted:
                result[f"omitted_{key}"] = omitted
        if isinstance(value.get("components"), dict):
            result["components"] = {}
            for key, nested in value["components"].items():
                if state["components"] >= MAX_CONFIG_COMPONENTS:
                    state["truncated"] = True
                    break
                name = _name(key, 48)
                if name is not None:
                    result["components"][name] = component(nested, depth + 1)
            omitted = len(value["components"]) - len(result["components"]) + int(value.get("omitted_components", 0))
            if omitted:
                result["omitted_components"] = omitted
        return result

    result = component(summary)
    _meta(result, truncated=state["truncated"], component_count=state["components"])
    if len(json.dumps(result, ensure_ascii=False, allow_nan=False)) > MAX_CONFIG_CHARACTERS:
        # Keep the observed root class/parameters; don't invent missing components.
        result = {"class": result["class"], "parameters": result["parameters"],
                  "summary_meta": {"truncated": True, "unknown_components": True,
                                   "reason": "config_character_budget"}}
        while len(json.dumps(result, ensure_ascii=False, allow_nan=False)) > MAX_CONFIG_CHARACTERS:
            result["parameters"].pop(next(reversed(result["parameters"])))
    return result


def _feature_profile(raw):
    if not isinstance(raw, dict):
        return {"status": "unknown"}
    result = _take_fields(raw, numbers=("sparse", "has_nan", "has_inf", "has_negative", "estimated_dense_bytes"),
                          texts=("dtype", "format"))
    if isinstance(raw.get("shape"), (list, tuple)):
        result["shape"] = [_number(value) for value in raw["shape"][:2]]
    return result


def _profiles(raw):
    raw = _dict(raw)
    return {key: _feature_profile(raw.get(key)) for key in ("train", "validation")}


def _metrics(raw):
    return _take_fields(raw, numbers=("accuracy", "f1_macro", "mse", "mae", "r2"))


def _execution(raw):
    return _take_fields(raw, numbers=("attempt_count", "fit_attempt_count", "fit_started", "fit_completed",
                                      "elapsed_seconds", "fit_elapsed_seconds"), texts=("phase",))


def _error(raw):
    if isinstance(raw, dict):
        result = _take_fields(raw, numbers=("node_id", "attempt_id", "first_observed_rollout"),
                              texts=("stage", "kind", "phase", "exception_type", "error_type",
                                     "compatibility_kind", "outcome_after_error"))
        message = raw.get("message", raw.get("error"))
    else:
        result, message = {}, raw
    if isinstance(message, str):
        # The final exception line is evidence; stack frames and source are not.
        lines = message.splitlines()
        exceptions = [line.strip() for line in lines if _EXCEPTION.search(line)]
        if exceptions:
            message = exceptions[-1]
        elif "Traceback (most recent call last)" in message:
            message = "[unknown: traceback without a concise exception message]"
        else:
            message = next((line.strip() for line in reversed(lines) if line.strip()), "")
        result["message"] = _text(message, 200)
    if not result:
        result = {"status": "unknown"}
    return result


def _errors(raw):
    if not isinstance(raw, list):
        return []
    return [_error(item) for item in raw[-MAX_ERRORS:]]


def _comparison(raw):
    raw = _dict(raw)
    result = _take_fields(raw, numbers=("reference_mt_id", "candidate_mt_id", "performance_comparable",
        "fixed_model_configuration", "recovered", "reference_value", "candidate_value", "raw_delta", "improvement",
        "repaired", "damaged", "unchanged_correct", "unchanged_wrong", "improved_rows", "worsened_rows",
        "unchanged_rows", "mean_row_loss_reduction"), texts=("kind", "reference_status", "candidate_status",
        "changed_scope", "metric", "reason", "row_loss", "count_interpretation", "limitations"))
    return result


def _diagnostics(raw):
    raw = _dict(raw)
    result = _take_fields(raw, numbers=("rows",), texts=("split", "limitations"))
    groups = _dict(raw.get("groups"))
    result["groups"] = {name: _take_fields(value, numbers=("rows", "small_group", "error_rate", "mae", "mse"))
                        for key, value in list(groups.items())[:MAX_GROUPS] if (name := _name(key)) is not None}
    result["group_policy"] = _take_fields(raw.get("group_policy"), numbers=("version", "rare_training_count_at_most",
                                                                           "minimum_reliable_group_rows"))
    if len(groups) > len(result["groups"]):
        _meta(result, omitted_groups=len(groups) - len(result["groups"]))
    return result


def _eda(raw):
    if not isinstance(raw, dict):
        return {"status": "unknown", "reason": "missing_executed_eda_report"}
    result = _take_fields(raw, numbers=("n_rows", "rows", "n_features", "feature_count", "total_missing_cells",
        "target_n_classes", "target_missing", "numeric_correlation_max_abs"), texts=("task_type", "target_name"))
    handled = set(result)
    for key in ("numeric_features", "categorical_features", "numeric", "categorical", "constant_columns"):
        if key in raw:
            result[key] = _names(raw[key])
            handled.add(key)
    for key in ("categorical_cardinalities", "target_class_counts", "class_counts", "missing"):
        if key in raw:
            result[key] = _numeric_map(raw[key], MAX_EDA_FEATURES)
            handled.add(key)
            if isinstance(raw[key], dict) and len(raw[key]) > len(result[key]):
                _meta(result, **{f"omitted_{key}": len(raw[key]) - len(result[key])})
    for key, stats in (("missingness", ("missing", "count", "fraction")),
                       ("numeric_summary", ("mean", "std", "min", "max", "skew"))):
        if key in raw:
            if key == "missingness" and raw.get("total_missing_cells") == 0:
                continue  # The recorded global zero already conveys this fact.
            omissions = {}
            result[key] = _numeric_map(raw[key], MAX_EDA_FEATURES, nested=stats, omissions=omissions)
            handled.add(key)
            if isinstance(raw[key], dict) and len(raw[key]) > len(result[key]):
                _meta(result, **{f"omitted_{key}": len(raw[key]) - len(result[key])})
            if omissions.get("fields"):
                _meta(result, **{f"omitted_{key}_fields": omissions["fields"]})
    for key in ("target_summary", "target_distribution"):
        if key in raw:
            result[key] = _take_fields(raw[key], numbers=("count", "mean", "std", "min", "median", "max", "skew"))
            handled.add(key)
    pairs = raw.get("numeric_correlation_top_pairs")
    if isinstance(pairs, list):
        result["numeric_correlation_top_pairs"] = [_take_fields(pair, numbers=("corr",), texts=("a", "b"))
                                                    for pair in pairs[:4]]
        handled.add("numeric_correlation_top_pairs")
        if len(pairs) > 4:
            _meta(result, omitted_correlation_pairs=len(pairs) - 4)
    _meta(result, omitted_report_fields=len(raw) - len(handled))
    if not handled:
        result.update(status="unknown", reason="unrecognized_eda_report_schema",
                      limitations="No recognized executed facts were retained; the plan is not a substitute for observations.")
    return result


def compact_task(task):
    """Task facts and bounded feature names; never include data/artifact paths."""
    task = _dict(task)
    result = _take_fields(task, numbers=("seed", "train_rows", "validation_rows"),
                          texts=("name", "task_id", "problem_type", "target", "metric", "split_id"))
    if "description" in task:
        result["description"] = _text(task["description"], 240)
    for key in ("retrieval_description", "dataset_id", "task_family"):
        if key in task:
            result[key] = _text(task[key], 650 if key == "retrieval_description" else 160)
    profile = _dict(task.get("data_profile"))
    result["data_profile"] = _take_fields(profile, numbers=("schema_version", "train_rows", "feature_count",
        "numeric_features", "categorical_features", "other_features", "missing_cell_fraction",
        "rows_with_missing_fraction", "classes", "class_count_max_min_ratio"), texts=("scope",))
    if not result["data_profile"]:
        result["data_profile"] = {"status": "unknown"}
    if "target_summary" in profile:
        result["data_profile"]["target_summary"] = _take_fields(profile["target_summary"], numbers=(
            "finite_rows", "missing_rows", "nonfinite_rows", "mean", "std", "min", "max"))
    columns = _dict(task.get("columns"))
    result["columns"] = {name: _text(dtype, 32) for key, dtype in list(columns.items())[:MAX_FEATURES]
                         if key != task.get("target") and (name := _name(key)) is not None}
    omitted = len([key for key in columns if key != task.get("target")]) - len(result["columns"])
    _meta(result, omitted_columns=omitted, description_truncated=isinstance(task.get("description"), str)
          and len(task["description"]) > 240)
    return result


def _prefix_entry(raw):
    raw = _dict(raw)
    result = _take_fields(raw, numbers=("node_id",), texts=("stage", "status"))
    result.update(idea=_idea(raw.get("idea"), raw.get("method")), idea_is_plan=True,
                  actual_result_is_authoritative=True)
    observed = _dict(raw.get("result"))
    stage = raw.get("stage")
    actual = {}
    if stage == "EDA":
        actual["report"] = _eda(observed.get("report"))
    elif stage == "FE":
        actual["transformer_config"] = _configuration(observed.get("transformer_config"))
        actual["feature_profiles"] = _profiles(observed.get("feature_profiles"))
        for key in ("train_shape", "validation_shape"):
            if isinstance(observed.get(key), (list, tuple)):
                actual[key] = [_number(value) for value in observed[key][:2]]
    elif stage == "MT":
        actual["model_config"] = _configuration(observed.get("model_config"))
        actual["metrics"] = _metrics(observed.get("metrics"))
        if observed.get("diagnostics"):
            actual["diagnostics"] = _diagnostics(observed["diagnostics"])
    else:
        actual = {"status": "unknown", "reason": "unrecognized_stage"}
    if observed.get("execution"):
        actual["execution"] = _execution(observed["execution"])
    result["result"] = actual
    if raw.get("status") == "failed":
        result["error"] = _error(raw.get("error", raw.get("feedback", raw.get("execution_feedback"))))
    attempts = raw.get("attempts")
    if isinstance(attempts, list):
        errors = [_dict(a).get("error_details", _dict(a).get("error")) for a in attempts]
        errors = [error for error in errors if error]
        if errors:
            result["errors"] = _errors(errors)
            _meta(result, omitted_errors=max(0, len(errors) - MAX_ERRORS))
    if raw.get("comparison"):
        result["comparison"] = _comparison(raw["comparison"])
    return result


def compact_prefix(prefix):
    """At most three actual stage entries; source code and raw logs are omitted."""
    if not isinstance(prefix, list):
        return []
    result = [_prefix_entry(entry) for entry in prefix[-MAX_PREFIX:]]
    if result and len(prefix) > len(result):
        _meta(result[0], omitted_entries=len(prefix) - len(result))
    return result


def compact_candidate(candidate):
    """Reduce an executor summary; an optional raw prefix is compacted once."""
    candidate = _dict(candidate)
    result = _take_fields(candidate, numbers=("node_id", "candidate_id", "fe_node_id", "reward", "based_on",
        "reference_mt_id", "source_rollout"), texts=("stage", "status", "origin", "code_hash"))
    idea = candidate.get("idea") if isinstance(candidate.get("idea"), dict) else candidate
    result.update(idea=_idea(idea, candidate.get("method")), idea_is_plan=True,
                  actual_result_is_authoritative=True)
    observed = _dict(candidate.get("result"))
    result["model_config"] = _configuration(candidate.get("model_config", observed.get("model_config")))
    metrics = candidate.get("validation_metrics", observed.get("metrics"))
    result["validation_metrics"] = (_metrics(metrics) if isinstance(metrics, dict)
                                     and candidate.get("status") not in {"failed", "pending"} else None)
    for key, reducer in (("diagnostics", _diagnostics), ("comparison", _comparison)):
        value = candidate.get(key, observed.get(key))
        if value:
            result[key] = reducer(value)
    if candidate.get("error") or candidate.get("status") == "failed":
        result["error"] = _error(candidate.get("error_details") or candidate.get("error", candidate.get("feedback")))
    if "prefix" in candidate:
        result["prefix"] = compact_prefix(candidate["prefix"])
    if "feedback_expansions" in candidate:
        result["feedback_expansions"] = _take_fields(candidate["feedback_expansions"],
                                                     numbers=("completed", "total_reward"))
    if "reference_selection" in candidate:
        result["reference_selection"] = compact_selection(candidate["reference_selection"])
    return result


def compact_feedback(feedback):
    """Keep rollout outcome, compact errors and comparison; drop pipeline copies."""
    feedback = _dict(feedback)
    result = _take_fields(feedback, numbers=("rollout_id", "reward", "based_on", "reference_mt_id",
        "reference_rollout_id", "completed_rollouts", "attempted_rollout"), texts=("status", "origin", "repair_issue_id"))
    if isinstance(feedback.get("evaluation"), dict):
        result["evaluation"] = _take_fields(feedback["evaluation"], numbers=("value", "reward"),
                                             texts=("split", "metric", "direction"))
    else:
        result["evaluation"] = None
    for key in ("failure", "interruption"):
        if feedback.get(key):
            result[key] = _error(feedback[key])
    if feedback.get("error"):
        result["error"] = _error(feedback["error"])
    if feedback.get("comparison"):
        result["comparison"] = _comparison(feedback["comparison"])
    if isinstance(feedback.get("errors"), list):
        result["errors"] = _errors(feedback["errors"])
        _meta(result, omitted_errors=max(0, len(feedback["errors"]) - MAX_ERRORS))
    if isinstance(feedback.get("path"), list):
        result["path"] = [_take_fields(item, numbers=("node_id",), texts=("stage", "status", "origin"))
                          for item in feedback["path"][-MAX_PREFIX:]]
        if len(feedback["path"]) > MAX_PREFIX:
            _meta(result, omitted_path_entries=len(feedback["path"]) - MAX_PREFIX)
    if isinstance(feedback.get("compatibility_issue_ids"), list):
        result["compatibility_issue_ids"] = _names(feedback["compatibility_issue_ids"], MAX_ISSUES)
    return result


def compact_issues(issues):
    """Preserve actionable issue/reference IDs and observed input mismatch facts."""
    if not isinstance(issues, list):
        return []
    result = []
    # Every submitted issue can be marked reviewed by the caller. Preserve all
    # issue groups and active IDs; limit only each group's supporting history.
    for raw in issues:
        raw = _dict(raw)
        item = _take_fields(raw, numbers=("fe_node_id", "version", "reviewed_version", "last_review_rollout"),
                            texts=("issue_id", "kind", "status"))
        active_ids = raw.get("active_mt_ids") if isinstance(raw.get("active_mt_ids"), list) else []
        item["active_mt_ids"] = _ids(active_ids, len(active_ids))
        occurrences = raw.get("occurrences") if isinstance(raw.get("occurrences"), list) else []
        active = [entry for entry in occurrences if _dict(entry).get("mt_node_id") in item["active_mt_ids"]
                  and _dict(entry).get("status") == "unresolved"]
        chosen = (active or occurrences)[-MAX_ERRORS:]
        item["occurrences"] = []
        for occurrence in chosen:
            occurrence = _dict(occurrence)
            detail = _take_fields(occurrence, numbers=("mt_node_id", "rollout_id", "attempt_id"),
                                  texts=("status", "kind", "phase", "evidence_basis"))
            detail["error"] = _error(occurrence)
            detail["feature_profile"] = _feature_profile(occurrence.get("feature_profile"))
            item["occurrences"].append(detail)
        if "feature_profile" in raw:
            item["feature_profile"] = _feature_profile(raw["feature_profile"])
        if raw.get("error"):
            item["error"] = _error(raw)
        _meta(item, omitted_occurrences=len(occurrences) - len(chosen),
              omitted_active_mt_ids=len(active_ids) - len(item["active_mt_ids"]))
        result.append(item)
    return result


def compact_selection(selection):
    """Keep bounded search statistics, clearly separate from measured metrics."""
    selection = _dict(selection)
    result = _take_fields(selection, numbers=("fe_node_id", "fe_visits", "reference_mt_id"), texts=("policy",))
    candidates = selection.get("candidates") if isinstance(selection.get("candidates"), list) else []
    chosen = candidates[:MAX_HISTORY]
    reference = next((item for item in candidates if _dict(item).get("node_id") == selection.get("reference_mt_id")), None)
    if reference is not None and reference not in chosen and chosen:
        chosen = chosen[:-1] + [reference]
    result["candidates"] = [_take_fields(item, numbers=("node_id", "own_reward", "expansion_visits",
        "expansion_total_reward", "effective_count", "mean_reward", "exploration_bonus", "ucb"), texts=("status",))
        for item in chosen]
    result["statistics_are_search_history"] = True
    _meta(result, omitted_candidates=len(candidates) - len(chosen))
    return result
