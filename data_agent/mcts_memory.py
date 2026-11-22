"""Deterministic, scoped experiment evidence for MCTS proposal prompts.

The full experiment archive belongs to the caller. This module stores evidence
cards and an append-only revision journal; it neither calls an LLM nor reads
test-set results. Retrieval is a bounded selection of observed evidence, never
an assertion that a parameter change caused an improvement.
"""
from __future__ import annotations

import hashlib
import base64
from copy import deepcopy
from functools import lru_cache
import json
import math
import os
import re
from pathlib import Path
import tempfile
from typing import Any, Mapping


SCOPE_KEYS = ("task_id", "split_id", "arm", "seed", "run_id")


@lru_cache(maxsize=1)
def _local_tokenizer():
    """Load only an already cached tokenizer, with no possible HTTP fallback."""
    try:
        import tiktoken
        from tiktoken.registry import ENCODINGS
        if "cl100k_base" in ENCODINGS:
            return ENCODINGS["cl100k_base"]
        directory = os.environ.get("TIKTOKEN_CACHE_DIR", os.environ.get(
            "DATA_GYM_CACHE_DIR", str(Path(tempfile.gettempdir()) / "data-gym-cache")))
        if not directory:
            return None
        blob = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        path = Path(directory) / hashlib.sha1(blob.encode()).hexdigest()
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7":
            return None
        ranks = {base64.b64decode(token): int(rank) for token, rank in
                 (line.split() for line in content.splitlines() if line)}
        return tiktoken.Encoding(name="offline_cl100k_base", mergeable_ranks=ranks, special_tokens={},
            pat_str=r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s")
    except (ImportError, OSError, ValueError, KeyError):
        return None


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _clean(value):
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        return _clean(value.item())
    raise TypeError(f"Evidence must be JSON compatible, got {type(value).__name__}")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _short(value, maximum=240):
    value = str(value)
    return value if len(value) <= maximum else value[:maximum] + "…"


def repair_only_context(context, limit=2):
    """An allowlist for error repair; experiment scores and ranking evidence stay out."""
    cards = context.get("cards", []) if isinstance(context, dict) else context
    result = {"cards": [], "evidence_ids": []}
    for card in (cards or [])[:max(0, min(2, limit))]:
        if not isinstance(card, dict):
            continue
        row = {key: deepcopy(card[key]) for key in
               ("candidate_id", "evidence_id", "id", "error", "error_signatures") if key in card}
        config = card.get("configuration", {})
        row["configuration"] = {key: deepcopy(config[key]) for key in
            ("family", "model", "fixed_parameters", "stage_parameters") if key in config}
        row["repairs"] = [{key: _short(repair[key], 400) for key in
            ("error_type", "error", "error_signature", "fix", "status") if key in repair}
            for repair in card.get("repairs", [])[-2:] if isinstance(repair, dict)]
        result["cards"].append(row)
        identifier = row.get("evidence_id", row.get("candidate_id", row.get("id")))
        if identifier is not None:
            result["evidence_ids"].append(identifier)
    return result


def _fingerprint(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _idea(candidate):
    value = candidate.get("idea", candidate.get("strategy", {})) or {}
    nested = value.get("idea", {}) if isinstance(value.get("idea"), dict) else {}
    result = {}
    for key in ("name", "family", "fe_config", "model_config", "fixed_parameters",
                "stage_parameters", "scaling_rules", "stage_budgets", "feature_engineering", "model"):
        item = nested.get(key, value.get(key, candidate.get(key)))
        if item is not None:
            result[key] = _clean(item)
    return result


def _pipeline_values(spec, prefix=""):
    """Flatten the pre-fit runtime specification, retaining exact leaf values.

    Named pipeline steps have stable paths even when unrelated steps are added.
    These are configuration observations, not fitted statistics or FE benefits.
    """
    result = {}
    if isinstance(spec, dict):
        for key, value in sorted(spec.items()):
            path = f"{prefix}.{key}" if prefix else key
            if key in {"steps", "transformers"} and isinstance(value, list):
                for item in value:
                    if isinstance(item, list) and len(item) >= 2 and isinstance(item[0], str):
                        result.update(_pipeline_values(item[1], f"{path}.{item[0]}"))
                        if len(item) > 2:
                            result[f"{path}.{item[0]}.columns"] = _clean(item[2])
                    else:
                        result[path] = _clean(value)
                        break
            elif isinstance(value, dict):
                result.update(_pipeline_values(value, path))
            else:
                result[path] = _clean(value)
    return result


def _effective_configuration(report):
    """Read execution reports, never substitute declared defaults for actuals."""
    specification = report.get("pipeline_spec") or {}
    actual = report.get("actual_budget") or {}
    model = None
    for step in specification.get("parameters", {}).get("steps", []):
        if isinstance(step, (list, tuple)) and len(step) >= 2 and step[0] == "model":
            model = step[1]
    if isinstance(model, dict) and isinstance(model.get("parameters"), dict):
        parameters = model["parameters"]
        source = "pipeline_spec_after_stage_parameters_and_budget"
        model_class = model.get("class")
    elif actual.get("model_parameters"):
        parameters = actual["model_parameters"]
        source = "actual_budget.model_parameters (repr values)"
        model_class = actual.get("model_class")
    else:
        parameters, source, model_class = {}, "not_recorded", actual.get("model_class")
    flattened = _pipeline_values(specification)
    features = {path: value for path, value in flattened.items()
                if ".steps.model." not in path and path != "class"}
    return {"model_class": model_class, "model_parameters": _clean(parameters),
            "feature_configuration": features, "parameter_source": source}


def _report(candidate, stage):
    aliases = ("low", "low_report", "coreset") if stage == "coreset" else ("full", "full_report", "full_train")
    raw = next((candidate[key] for key in aliases if key in candidate), None)
    stage_error = candidate.get("low_error" if stage == "coreset" else "full_error")
    if raw is None:
        if stage_error:
            return {"status": "failed", "stage": stage, "error": _short(stage_error, 500)}
        return {"status": "not_evaluated"}
    if not isinstance(raw, dict):
        raise ValueError(f"{stage} report must be an object or null")
    declared_stage = str(raw.get("stage", stage)).lower()
    if declared_stage in {"test", "testing", "final_test", "test_set"}:
        raise ValueError("Test results must not enter search memory")
    status = raw.get("status")
    if status is None:
        status = "completed" if _number(raw.get("score")) else "not_evaluated"
    result = {"status": status, "stage": stage}
    for key in ("score", "metric", "direction", "fit_rows", "fit_seconds", "prediction_seconds",
                "validation_id", "evaluation_id", "evaluation_set", "requested_budget", "actual_budget",
                "stage_parameters", "error_type", "error", "error_signature", "mcs_survivor", "subgroup_errors",
                "coreset_version"):
        if key in raw:
            result[key] = _clean(raw[key])
    if not result.get("validation_id") and raw.get("evaluation_set"):
        # The containing store already fixes the exact task, split, arm and seed.
        result["validation_id"] = str(raw["evaluation_set"])
    context = raw.get("fit_context", {}) or {}
    if "fit_rows" not in result and context.get("fit_rows") is not None:
        result["fit_rows"] = _clean(context["fit_rows"])
    feedback = raw.get("training_feedback") or {}
    # Whitelisting deliberately excludes predictions, traces and held-out/test metrics.
    if feedback:
        result["training_feedback"] = {key: _clean(feedback[key]) for key in
            ("fit_rows", "training_metrics", "convergence", "transformed_feature_count", "metrics_source") if key in feedback}
        if isinstance(feedback.get("primary_metric"), dict):
            result["training_feedback"]["primary_metric"] = {key: _clean(feedback["primary_metric"][key])
                for key in ("metric", "metric_params", "source", "rows", "value", "direction", "status", "reason")
                if key in feedback["primary_metric"]}
    if raw.get("warnings"):
        result["warnings"] = [_short(item, 300) for item in raw["warnings"][:8]]
    result["effective_configuration"] = _effective_configuration(raw)
    return result


def _delta(report, parent):
    """Absent identity is not evidence that two evaluations are comparable."""
    identity = report.get("validation_id", report.get("evaluation_id"))
    parent_identity = parent.get("validation_id", parent.get("evaluation_id"))
    if (not identity or identity != parent_identity or not report.get("metric")
            or report.get("metric") != parent.get("metric")
            or report.get("stage") != parent.get("stage")
            or report.get("direction") not in {"maximize", "minimize"}
            or report.get("direction") != parent.get("direction")
            or not _number(report.get("score")) or not _number(parent.get("score"))):
        return None
    if report.get('stage')=='coreset' and report.get('coreset_version')!=parent.get('coreset_version'):
        return None
    return report["score"] - parent["score"]


def _changes(before, after, prefix=""):
    result = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(old, dict) and isinstance(new, dict):
            result.extend(_changes(old, new, path))
        elif old != new:
            result.append({"path": path, "before": old, "after": new})
    return result


# Runtime parameters that describe capacity, regularization and optimization.
# Administrative defaults are deliberately not used to fill a fixed quota.
_MODEL_KEYS = (
    (("linearregression",), ("fit_intercept", "positive", "tol")),
    (("gaussiannb", "bernoullinb", "multinomialnb", "complementnb", "categoricalnb"),
     ("var_smoothing", "alpha", "fit_prior", "binarize", "class_prior", "priors", "norm", "min_categories", "force_alpha")),
    (("histgradientboost",), ("max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "max_depth", "max_bins", "early_stopping", "max_features", "categorical_features")),
    (("randomforest", "extratrees"), ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "max_leaf_nodes", "min_samples_split", "bootstrap", "max_samples", "ccp_alpha", "criterion")),
    (("gradientboost", "xgb", "lgbm", "catboost"), ("n_estimators", "iterations", "learning_rate", "max_depth", "num_leaves", "min_samples_leaf", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha")),
    (("mlp",), ("hidden_layer_sizes", "alpha", "learning_rate_init", "max_iter", "batch_size", "activation", "solver", "early_stopping", "tol", "n_iter_no_change")),
    (("ridge", "lasso", "elasticnet"), ("alpha", "l1_ratio", "fit_intercept", "solver", "max_iter", "tol", "positive", "selection")),
    (("logistic", "svc", "svr"), ("C", "penalty", "l1_ratio", "kernel", "gamma", "class_weight", "solver", "max_iter", "tol", "epsilon")),
    (("kneighbors",), ("n_neighbors", "weights", "p", "metric", "algorithm", "leaf_size")),
    (("decisiontree",), ("max_depth", "min_samples_leaf", "max_leaf_nodes", "min_samples_split", "max_features", "ccp_alpha", "criterion", "class_weight")),
)
_GENERAL_KEYS = ("n_estimators", "max_iter", "max_depth", "min_samples_leaf", "max_leaf_nodes",
                 "learning_rate", "max_features", "C", "alpha", "l2_regularization", "hidden_layer_sizes")
_ADMIN_KEYS = {"n_jobs", "verbose", "random_state", "copy", "copy_X", "warm_start", "force_int_remainder_cols"}
_WRAPPERS = {"Pipeline", "ColumnTransformer", "FeatureUnion"}


def _parameter_keys(effective, changes=()):
    parameters = effective.get("model_parameters", {})
    name = str(effective.get("model_class") or "").lower()
    preferred = next((keys for names, keys in _MODEL_KEYS if any(n in name for n in names)), _GENERAL_KEYS)
    changed = {c["path"].removeprefix("model_parameters.") for c in changes
               if c["path"].startswith("model_parameters.")}
    # Relevant changed capacity parameters lead, not arbitrary alphabetic changes
    # introduced when an estimator class was replaced.
    ordered = [k for k in preferred if k in changed] + list(preferred)
    ordered += sorted(changed - set(preferred) - _ADMIN_KEYS)
    selected = [key for key in dict.fromkeys(ordered) if key in parameters
                and len(_json(parameters[key]).encode("utf-8")) <= 512]
    if not selected:
        selected = [key for key in sorted(parameters) if key not in _ADMIN_KEYS
                    and len(_json(parameters[key]).encode("utf-8")) <= 512][:6]
    return selected


def _meaningful_changes(changes):
    def priority(change):
        path = change["path"]
        leaf = path.rsplit(".", 1)[-1]
        if leaf in _ADMIN_KEYS or path == "parameter_source":
            return 9
        if path == "model_class":
            return 0
        if path.startswith("model_parameters."):
            return 1 if leaf in set(_GENERAL_KEYS).union(*(keys for _, keys in _MODEL_KEYS)) else 5
        if leaf == "class":
            names = {str(change.get(key) or "").rsplit(".", 1)[-1] for key in ("before", "after")}
            return 8 if names <= _WRAPPERS | {""} else 2
        if leaf in {"callable", "columns"}:
            return 2
        return 3
    return sorted((change for change in changes if priority(change) < 8), key=lambda c: (priority(c), c["path"]))


def _error_signature(error):
    if isinstance(error, dict):
        signature = error.get("error_signature")
        if signature:
            return str(signature)
        kind = error.get("error_type")
        message = str(error.get("error", error.get("message", "")))
        if kind and str(kind) not in message:
            message = str(kind) + ": " + message
        return _error_signature(message)
    value = str(error or "")
    matches = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b[^\n]*", value)
    if matches:
        # Type alone is too broad: unrelated ValueErrors must not share fixes.
        error_line = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\b[^\n]*", value)[-1]
        return re.sub(r"\b\d+\b", "#", error_line.strip())[:180]
    # Unclassified messages match only their normalized text, not all failures.
    return re.sub(r"\b\d+\b", "#", value.strip())[:180]


class ExperienceMemory:
    """Single-search-writer evidence store isolated by five scope identifiers.

    ``upsert`` is idempotent for unchanged observations. A new full report
    replaces the same card instead of counting a second independent example.
    Paths include a scope hash, so sharing a base directory never mixes arms.
    """

    def __init__(self, directory, scope, default_count=4, max_count=6, token_budget=3000, enabled=True):
        if type(enabled) is not bool:
            raise ValueError('memory.enabled must be boolean')
        self.enabled = enabled
        if set(scope) != set(SCOPE_KEYS) or any(scope[key] is None for key in SCOPE_KEYS):
            raise ValueError(f"scope requires exactly {SCOPE_KEYS}")
        if not 1 <= default_count <= max_count <= 6 or not isinstance(token_budget, int) or token_budget < 128:
            raise ValueError("Memory needs 1 <= default_count <= max_count <= 6 and token_budget >= 128")
        self.scope = _clean(dict(scope))
        self.default_count, self.max_count, self.token_budget = default_count, max_count, token_budget
        self.tokenizer = _local_tokenizer()
        self.directory = Path(directory) / _fingerprint(self.scope)[:24]
        self.directory.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.directory / "events.jsonl"
        self.cards_path = self.directory / "cards.json"
        self.cards: dict[str, dict[str, Any]] = {}
        # Journal is authoritative: a crash before snapshot replacement loses no evidence.
        if self.journal_path.exists():
            for line in self.journal_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event["scope"] != self.scope:
                    raise ValueError("Memory journal scope mismatch")
                self.cards[event["card"]["candidate_id"]] = event["card"]

    def upsert(self, candidate, parent=None):
        """Extract a current card from a candidate and optional parent record."""
        supplied_scope = candidate.get("scope")
        if supplied_scope is not None and supplied_scope != self.scope:
            raise ValueError("Candidate scope differs from memory scope")
        identifier = str(candidate.get("id", candidate.get("candidate_id", "")))
        if not identifier:
            raise ValueError("Candidate must have an id")
        parent_id = candidate.get("parent_id")
        parent_id = str(parent_id) if parent_id is not None else None
        parent_card = self.cards.get(parent_id)
        if parent is not None:
            if parent.get("scope") is not None and parent["scope"] != self.scope:
                raise ValueError("Parent scope differs from memory scope")
            parent_idea = _idea(parent)
            parent_stages = {stage: _report(parent, stage) for stage in ("coreset", "full_train")}
        else:
            parent_idea = parent_card.get("configuration", {}) if parent_card else {}
            parent_stages = parent_card.get("results", {}) if parent_card else {}
        configuration = _idea(candidate)
        results = {stage: _report(candidate, stage) for stage in ("coreset", "full_train")}
        # A partial callback must not erase full evidence already recorded.
        old = self.cards.get(identifier)
        if old:
            for stage, aliases in (("coreset", ("low", "low_report", "coreset")),
                                   ("full_train", ("full", "full_report", "full_train"))):
                error_key = "low_error" if stage == "coreset" else "full_error"
                if not any(candidate.get(key) is not None for key in aliases) and not candidate.get(error_key):
                    results[stage] = old["results"][stage]
        mutation = candidate.get("mutation", {}) or {}
        if isinstance(mutation, str):
            mutation = {"direction": mutation}
        card = {
            "candidate_id": identifier,
            "observation_order": old.get("observation_order", 0) if old else
                max((c.get("observation_order", 0) for c in self.cards.values()), default=0) + 1,
            "evidence_id": "exp:" + _fingerprint({"scope": self.scope, "candidate": identifier})[:20],
            "parent_id": parent_id,
            "branch_id": str(candidate.get("branch_id", identifier)),
            "method_summary": _short((candidate.get("idea") or {}).get("description", ""), 240),
            "configuration": configuration,
            "mutation": {"direction": mutation.get("direction", candidate.get("mutation_direction", candidate.get("modification_direction"))),
                         "reason": _short(mutation.get("reason", (candidate.get("idea") or {}).get("change_reason", candidate.get("reason", ""))), 500)},
            "changes": _changes(parent_idea, configuration),
            "actual_changes": {
                stage: _changes(parent_stages[stage]["effective_configuration"], report["effective_configuration"])
                if parent_stages.get(stage, {}).get("effective_configuration") and report.get("effective_configuration")
                else []
                for stage, report in results.items()
            },
            "results": results,
            "parent_deltas": {stage: _delta(report, parent_stages.get(stage, {})) for stage, report in results.items()},
            "source_candidate_ids": list(dict.fromkeys([identifier] + ([parent_id] if parent_id else [])
                                         + [str(value) for value in candidate.get("source_candidate_ids", [])])),
            "independent_evidence_count": 1,
            "interpretation": "Observed configuration comparison; individual changes are not causal findings.",
        }
        for key in ("mcs_survivor", "admitted", "decision"):
            if key in candidate:
                card[key] = _clean(candidate[key])
            elif old and key in old:
                card[key] = old[key]
        if "experience_use" in candidate:
            card["experience_use"] = [{key: _clean(record[key]) for key in
                ("evidence_id", "adoption_reason", "applicability_difference") if key in record}
                for record in candidate["experience_use"] if isinstance(record, dict)]
        elif old and "experience_use" in old:
            card["experience_use"] = old["experience_use"]
        signatures = []
        repairs = candidate.get("repairs", candidate.get("repair_history", [])) or []
        card["repairs"] = []
        for repair in repairs:
            if not isinstance(repair, dict):
                continue
            signature = _error_signature(repair)
            if signature:
                signatures.append(signature)
            card["repairs"].append({key: _clean(repair[key]) if key == "changes" else _short(repair[key], 400)
                                    for key in ("error_type", "error", "error_signature", "fix", "changes", "status") if key in repair})
        if candidate.get("error"):
            card["error"] = _short(candidate["error"], 500)
            signatures.append(_error_signature(candidate))
        for report in results.values():
            if report.get("error") or report.get("error_type"):
                signatures.append(_error_signature(report))
        for error in candidate.get("execution_errors", []):
            if isinstance(error, dict) and error.get("error"):
                signatures.append(_error_signature(error))
        card["error_signatures"] = sorted(set(value for value in signatures if value))
        card["failed"] = candidate.get("status") in {"failed", "error"} or any(
            result["status"] in {"failed", "error", "timeout"} for result in results.values())
        low_delta, full_delta = (card["parent_deltas"][stage] for stage in ("coreset", "full_train"))
        tolerance = candidate.get("sampling_practical_tolerance",
                                  (old or {}).get("sampling_practical_tolerance"))
        if tolerance is not None:
            if not _number(tolerance) or tolerance < 0:
                raise ValueError("Sampling practical tolerance must be finite and nonnegative")
            card["sampling_practical_tolerance"] = tolerance
        else:
            tolerance = 0.  # Preserve historical behavior outside the new skill workflow.
        card["ranking_reversal"] = (low_delta is not None and full_delta is not None
                                     and abs(low_delta) > tolerance and abs(full_delta) > tolerance
                                     and low_delta * full_delta < 0)
        card["evidence_stage"] = "full_train" if _number(results["full_train"].get("score")) else "coreset_only"
        if candidate.get("screening") is not None:
            card["screening"] = candidate["screening"]
        elif old and "screening" in old:
            card["screening"] = old["screening"]
        if not parent_id and not card["mutation"]["reason"]:
            idea = candidate.get("idea") or {}
            card["mutation"]["reason"] = _short(idea.get("rationale") or idea.get("description") or "", 180)
        card["method_path"] = self._method_path(card)
        card = _clean(card)
        if old == card:
            return json.loads(_json(card))
        with self.journal_path.open("a", encoding="utf-8") as stream:
            stream.write(_json({"scope": self.scope, "event": "upsert", "card": card}) + "\n")
            stream.flush()
        self.cards[identifier] = card
        temporary = self.cards_path.with_suffix(".tmp")
        temporary.write_text(_json({"scope": self.scope, "cards": self.cards}), encoding="utf-8")
        temporary.replace(self.cards_path)
        return json.loads(_json(card))

    @staticmethod
    def _relevance(card, query):
        config = card["configuration"]
        value = 0
        if query.get("family") and config.get("family") == query["family"]:
            value += 4
        if query.get("direction") and card["mutation"].get("direction") == query["direction"]:
            value += 3
        requested = set(query.get("parameter_paths", []))
        if requested:
            changes = card.get("actual_changes", {})
            paths = {change["path"] for records in changes.values() for change in records}
            paths.update(change["path"] for change in card["changes"])
            value += 2 * len(requested.intersection(paths))
        if query.get("fit_rows"):
            for report in card["results"].values():
                rows = report.get("fit_rows")
                if _number(rows) and .5 <= rows / query["fit_rows"] <= 2:
                    value += 1
                    break
        return value

    @staticmethod
    def _successful_full(card):
        report = card.get("results", {}).get("full_train", {})
        return report.get("status") not in {"failed", "error", "timeout"} and _number(report.get("score"))

    @staticmethod
    def _experiment_row(card, max_parameters=8):
        result = {key: card.get(key) for key in ("candidate_id", "parent_id", "evidence_id", "parent_deltas")}
        result["family"] = card["configuration"].get("family")
        result["name"] = card["configuration"].get("name")
        result["feature_mechanism"] = []
        result["results"] = {}
        result["actual_changes"] = {}
        for stage, report in card["results"].items():
            row = {key: report[key] for key in ("status", "score", "metric", "direction", "validation_id",
                    "evaluation_id", "evaluation_set", "fit_rows", "fit_seconds", "coreset_version") if key in report}
            feedback = report.get("training_feedback", {})
            training = {key: feedback[key] for key in ("primary_metric", "metrics_source") if key in feedback}
            if training:
                row["training"] = training
            effective = report.get("effective_configuration") or _effective_configuration(report)
            parameters = effective.get("model_parameters", {})
            changes = _meaningful_changes(card.get("actual_changes", {}).get(stage, []))
            keys = _parameter_keys(effective, changes)
            chosen = {key: parameters[key] for key in keys if key in parameters
                      and len(_json(parameters[key]).encode("utf-8")) <= 512}
            chosen = dict(list(chosen.items())[:max_parameters])
            row.update(model_class=effective.get("model_class"), actual_key_parameters=chosen,
                       parameter_source=effective["parameter_source"],
                       omitted_parameter_count=len(parameters) - len(chosen))
            model_name = str(effective.get("model_class") or "").lower()
            if chosen and not any(any(name in model_name for name in names) for names, _ in _MODEL_KEYS):
                row["parameter_selection"] = "generic_runtime_keys; model_specific_priority_unknown"
            result["results"][stage] = row
            result["actual_changes"][stage] = [change if len(_json(change).encode("utf-8")) <= 768 else
                {"path": change["path"], "values_omitted": "large_exact_values_in_archive"}
                for change in changes[:max_parameters]]
            if len(changes) > max_parameters:
                row["omitted_change_count"] = len(changes) - max_parameters
            for path, value in effective.get("feature_configuration", {}).items():
                if (path.endswith((".class", ".callable")) and value not in result["feature_mechanism"]
                        and str(value).rsplit(".", 1)[-1] not in _WRAPPERS
                        and not str(value).startswith("numpy.")):
                    result["feature_mechanism"].append(value)
        # The runtime identifies components/callables, not their semantic benefits.
        if not result["feature_mechanism"]:
            result["feature_mechanism_source"] = "not_recorded"
        else:
            result["feature_mechanism_source"] = "runtime_pipeline_components_not_causal_explanation"
            count = len(result["feature_mechanism"])
            result["feature_mechanism"] = result["feature_mechanism"][:8]
            if count > 8:
                result["omitted_feature_component_count"] = count - 8
        return result

    def experiment_row(self, candidate_id, max_parameters=8):
        """Programmatic, test-free row for the global-best or experiment context."""
        card = self.cards.get(str(candidate_id))
        return None if card is None else self._experiment_row(card, max_parameters=max_parameters)

    @staticmethod
    def _compact(card):
        config = card["configuration"]
        result = ExperienceMemory._experiment_row(card)
        result.update({key: card[key] for key in ("branch_id", "evidence_stage", "failed",
                       "ranking_reversal", "independent_evidence_count")})
        for key in ("mcs_survivor", "admitted", "decision"):
            if key in card:
                result[key] = card[key]
        if card.get("experience_use"):
            result["experience_use"] = [{key: value if key == "evidence_id" else _short(value, 120)
                                          for key, value in record.items()}
                                         for record in card["experience_use"][:3]]
        result["mutation"] = {"direction": card["mutation"].get("direction"),
                              "reason": _short(card["mutation"].get("reason", ""), 120)}
        # Declared changes remain explicitly separate from runtime observations.
        # Individual parameter values are exact; whole fields can be omitted.
        result["declared_changes"] = card["changes"][:4]
        result["omitted_declared_change_count"] = max(0, len(card["changes"]) - 4)
        for stage, report in card["results"].items():
            compact = result["results"][stage]
            feedback = report.get("training_feedback", {})
            convergence = feedback.get("convergence", {}) or {}
            diagnostics = {key: convergence[key] for key in ("n_iter", "iteration_limit", "iteration_limit_reached", "loss") if key in convergence}
            curve = convergence.get("loss_curve") or {}
            if curve:
                diagnostics["loss_curve"] = {key: curve[key] for key in ("length", "first", "last") if key in curve}
            if feedback.get("transformed_feature_count") is not None:
                diagnostics["transformed_feature_count"] = feedback["transformed_feature_count"]
            if diagnostics:
                compact["training"] = {**compact.get("training", {}), **diagnostics}
            if report.get("warnings"):
                compact["warnings"] = [_short(value, 160) for value in report["warnings"][:2]]
            if report.get("subgroup_errors"):
                # Inputs are predefined-group aggregate summaries, never row predictions.
                compact["subgroup_errors"] = report["subgroup_errors"]
            result["results"][stage] = compact
        if card.get("repairs"):
            result["repairs"] = card["repairs"][-2:]
        if card.get("error"):
            result["error"] = _short(card["error"], 200)
        return result

    @staticmethod
    def _minimum(card):
        """A concrete runtime setting/change paired with its measured results.

        Missing runtime metadata stays missing. Declared-only changes are
        explicitly labelled and never qualify as useful measured full evidence.
        """
        result = {key: deepcopy(card[key]) for key in ("evidence_id", "candidate_id", "parent_id", "evidence_stage",
                  "parent_deltas", "failed", "ranking_reversal")}
        result["results"] = {stage: {key: report[key] for key in
            ("status", "score", "metric", "direction", "validation_id", "evaluation_id", "evaluation_set", "fit_rows", "fit_seconds", "error", "error_type", "coreset_version")
            if key in report} for stage, report in card["results"].items()}
        row = ExperienceMemory._experiment_row(card, max_parameters=2)
        result["family"] = row["family"]
        result["feature_mechanism"] = row["feature_mechanism"][:2]
        result["actual_changes"] = {}
        for stage, observed in row["results"].items():
            parameters = {key: value for key, value in observed["actual_key_parameters"].items()
                          if len(_json(value).encode("utf-8")) <= 256}
            if parameters:
                source = ("runtime.pipeline_spec" if observed["parameter_source"].startswith("pipeline_spec")
                          else "runtime.actual_budget.model_parameters_repr")
                result["results"][stage].update(
                    actual_key_parameters=dict(list(parameters.items())[:1]),
                    parameter_source=source, model_class=observed["model_class"])
                if observed.get("parameter_selection"):
                    result["results"][stage]["parameter_selection"] = observed["parameter_selection"]
            changes = [change for change in _meaningful_changes(card.get("actual_changes", {}).get(stage, []))
                       if change["path"] != "parameter_source" and
                       len(_json(change).encode("utf-8")) <= 256]
            if changes:
                result["actual_changes"][stage] = changes[:1]
        if not result["actual_changes"] and not any(
                report.get("actual_key_parameters") for report in result["results"].values()):
            declared = [change for change in card["changes"]
                        if change["path"] not in {"name", "family"} and
                        len(_json(change).encode("utf-8")) <= 256]
            result["declared_changes"] = declared[:1]
            result["configuration_status"] = "runtime_parameters_not_recorded; declaration_is_not_execution_evidence"
        rules = card["configuration"].get("scaling_rules", {})
        if isinstance(rules, dict):
            compact_rules = {key: value for key, value in rules.items()
                             if len(_json({key: value}).encode("utf-8")) <= 256}
            if compact_rules:
                result["declared_scaling_rules"] = dict(list(compact_rules.items())[:1])
        if card.get("repairs"):
            result["repairs"] = card["repairs"][-1:]
        elif card.get("error"):
            result["error"] = _short(card["error"], 160)
        result["detail_level"] = "minimum_configuration_and_outcome"
        return result

    @staticmethod
    def _useful_full(card):
        if not ExperienceMemory._successful_full(card):
            return False
        minimum = ExperienceMemory._minimum(card)
        return bool(minimum["results"]["full_train"].get("actual_key_parameters") or
                    minimum["actual_changes"].get("full_train"))

    @staticmethod
    def _parent_diagnostics(card):
        """Small training observations needed to interpret the parent outcome."""
        result = {}
        warned = grouped = False
        for stage in ("full_train", "coreset"):
            report = card["results"].get(stage, {})
            feedback = report.get("training_feedback", {})
            if not feedback and not _number(report.get("score")):
                continue
            row, training = {}, {}
            primary = feedback.get("primary_metric")
            source = feedback.get("metrics_source") or {"fit_stage": stage, "evaluation_split": stage,
                                                        "training_rows": report.get("fit_rows")}
            if not primary and report.get("metric") in feedback.get("training_metrics", {}):
                primary = {"metric": report["metric"], "value": feedback["training_metrics"][report["metric"]],
                           "source": source, "status": "success"}
            if primary:
                training["primary_metric"] = {key: deepcopy(primary[key]) for key in
                    ("metric", "value", "direction", "status", "rows", "source") if key in primary}
                training["primary_metric"].setdefault("source", source)
                if primary.get("reason"):
                    training["primary_metric"]["reason"] = _short(primary["reason"], 160)
            convergence = feedback.get("convergence") or {}
            for key in ("n_iter", "iteration_limit", "iteration_limit_reached", "loss"):
                value = convergence.get(key)
                if value is None:
                    continue
                if key == "n_iter" and isinstance(value, list):
                    values = [item for item in value if _number(item)]
                    if values:
                        training["n_iter"] = {"minimum": min(values), "maximum": max(values), "count": len(values)}
                elif _number(value) or isinstance(value, bool):
                    training[key] = value
            training["transformed_feature_count"] = feedback.get("transformed_feature_count")
            if training["transformed_feature_count"] is None:
                training["transformed_feature_count"] = "not_recorded"
            curve = convergence.get("loss_curve")
            if isinstance(curve, dict) and curve:
                summary = {key: curve[key] for key in ("length", "first", "last") if key in curve}
                if isinstance(curve.get("tail"), list):
                    summary["tail"] = curve["tail"][-3:]
                training["loss_curve"] = summary
            else:
                training["loss_curve"] = "not_recorded"
            if training:
                if "primary_metric" not in training:
                    training["metrics_source"] = source
                row["training"] = training
            if not warned and report.get("warnings"):
                row["warnings"] = [_short(report["warnings"][0], 160)]
                warned = True
            groups = [group for group in report.get("subgroup_errors", [])
                      if isinstance(group, dict) and _number(group.get("score"))]
            if not grouped and groups and report.get("direction") in {"minimize", "maximize"}:
                key = lambda group: group["score"]
                worst = (max if report["direction"] == "minimize" else min)(groups, key=key)
                compact = {field: worst[field] for field in ("name", "group_id", "rows", "count", "metric", "score", "small_sample")
                           if field in worst}
                compact["selection"] = "worst_measured_predefined_group; descriptive_only"
                row["subgroup_errors"] = [compact]
                grouped = True
            if row:
                result[stage] = row
        return result

    @staticmethod
    def _feature_summary(card):
        """Observed transforms/column groups; declared function meaning is labelled."""
        stages = {stage: report.get("effective_configuration", {}).get("feature_configuration", {})
                  for stage, report in card["results"].items()}
        source = "full_train" if stages.get("full_train") else "coreset"
        flat = stages.get(source) or {}
        if not flat:
            return {"status": "not_recorded"}
        transforms, groups = [], []
        for path, value in flat.items():
            if path.endswith(".columns"):
                item = {"group": path.removesuffix(".columns").replace("parameters.", "")}
                if isinstance(value, list):
                    item.update(count=len(value), examples=value[:4])
                else:
                    item["selector"] = _short(value, 100)
                groups.append(item)
            if not path.endswith(".class") or str(value).rsplit(".", 1)[-1] in _WRAPPERS or str(value).startswith("numpy."):
                continue
            prefix = path.removesuffix(".class")
            params = {}
            for key, observed in flat.items():
                stem = prefix + ".parameters."
                if not key.startswith(stem):
                    continue
                name = key[len(stem):]
                if name in _ADMIN_KEYS | {"accept_sparse", "check_inverse", "validate", "feature_names_out"} or "." in name or isinstance(observed, (dict, list)):
                    continue
                if len(_json(observed)) <= 120:
                    params[name] = observed
            functions = {key[len(prefix + ".parameters."):]: observed for key, observed in flat.items()
                         if key.startswith(prefix + ".parameters.") and key.endswith(".callable")}
            kwargs = {key[len(prefix + ".parameters.kw_args."):]: observed for key, observed in flat.items()
                      if key.startswith(prefix + ".parameters.kw_args.") and
                      "." not in key[len(prefix + ".parameters.kw_args."):] and len(_json(observed)) <= 100}
            if kwargs:
                params["kw_args"] = dict(list(kwargs.items())[:3])
            transform = {"path": prefix.replace("parameters.", ""), "class": value}
            if params:
                transform["parameters"] = dict(list(params.items())[:6])
            if functions:
                transform["functions"] = functions
            transforms.append(transform)
        def importance(transform):
            name = transform["class"].rsplit(".", 1)[-1]
            if transform.get("functions") or name in {"PolynomialFeatures", "SplineTransformer", "KBinsDiscretizer", "PCA"}:
                return 0
            return 2 if name in {"SimpleImputer", "StandardScaler"} else 1
        transforms.sort(key=importance)
        result = {"source_stage": source, "source": "runtime.pipeline_spec",
                  "transforms": transforms[:2], "column_groups": groups[:2],
                  "omitted_transform_count": max(0, len(transforms) - 2),
                  "omitted_column_group_count": max(0, len(groups) - 2)}
        other = "coreset" if source == "full_train" else "full_train"
        if stages.get(other):
            if stages[other] == flat:
                result["also_applies_to"] = other
            else:
                result["other_stage_configuration"] = "different; not summarized here"
        declared = card["configuration"].get("feature_engineering")
        if declared:
            values = declared if isinstance(declared, list) else [declared]
            result["declared_semantics_unverified"] = [_short(value, 100) for value in values[:1]]
        return result

    @staticmethod
    def _coalesce_parameters(row):
        """Serialize equal observed stage values once, without filling missing stages."""
        values, sources, classes, omitted, selections = {}, {}, {}, {}, {}
        for stage, report in row["results"].items():
            parameters = report.pop("actual_key_parameters", {})
            source = report.pop("parameter_source", "not_recorded")
            klass = report.pop("model_class", None)
            count = report.pop("omitted_parameter_count", None)
            selection = report.pop("parameter_selection", None)
            if parameters or (source != "not_recorded" and klass is not None):
                values[stage], sources[stage], classes[stage] = parameters, source, klass
                if selection:
                    selections[stage] = selection
            if count:
                omitted[stage] = count
        if not values:
            row["effective_parameters"] = {"status": "not_recorded"}
            return
        shared = {}
        if len(values) == 2 and len(set(classes.values())) == 1:
            for key, value in values["coreset"].items():
                if key in values["full_train"] and values["full_train"][key] == value:
                    shared[key] = value
        row["effective_parameters"] = {"shared": shared,
            "by_stage": {stage: {key: value for key, value in params.items() if key not in shared}
                         for stage, params in values.items()}, "source_by_stage": sources}
        if len(set(classes.values())) == 1:
            row["effective_parameters"]["model_class"] = next(iter(classes.values()))
        else:
            row["effective_parameters"]["model_class_by_stage"] = classes
        if omitted:
            row["effective_parameters"]["omitted_count_by_stage"] = omitted
        if selections:
            row["effective_parameters"]["selection_by_stage"] = selections

    def _pack(self, selected, limit, *, roles=None, history_plan=None, relations=None):
        """Reserve concrete evidence, then enrich roles in a budget-independent order.

        Parameters precede optional cards. A fixed prefix of enrichment operations
        makes increasing the budget unable to remove an already visible role
        parameter. Incomplete critical coverage is explicitly reported.
        """
        role_ids = {str(value) for value in (roles or {}).values() if value is not None}
        seen, unique = {}, []
        for card in selected:
            evidence = {stage: {key: report.get(key) for key in
                       ("status", "score", "metric", "direction", "validation_id", "fit_rows", "stage_parameters", "effective_configuration", "coreset_version")}
                        for stage, report in card["results"].items()}
            has_actuals = any(report.get("effective_configuration", {}).get("parameter_source") not in (None, "not_recorded")
                              for report in card["results"].values())
            if has_actuals:
                for report in evidence.values():
                    report.pop("stage_parameters", None)
            declared = {key: value for key, value in card["configuration"].items() if key != "name"}
            signature = _fingerprint({"configuration": None if has_actuals else declared, "results": evidence,
                                      "errors": card["error_signatures"]})
            if signature in seen and card["candidate_id"] not in role_ids:
                continue
            if any(existing["candidate_id"] == card["candidate_id"] for existing in unique):
                continue
            unique.append(card)
            seen.setdefault(signature, card["candidate_id"])
        available = {card["candidate_id"]: card for card in unique}
        plan = history_plan or {"recent_full": [], "counterexample": None}
        plan = {"recent_full": [i for i in plan["recent_full"] if i in available],
                "counterexample": plan.get("counterexample") if plan.get("counterexample") in available else None}
        historical = [card for card in unique if card["candidate_id"] not in role_ids and self._useful_full(card)]
        required = [card for card in unique if card["candidate_id"] in role_ids]
        if roles is not None and historical:
            required += historical[:1]
        if not limit:
            required = []
        details = {}
        def minimum(card):
            row = self._minimum(card)
            for key in ("failed", "evidence_stage", "feature_mechanism", "family"):
                row.pop(key, None)
            if not row.get("ranking_reversal"):
                row.pop("ranking_reversal", None)
            if not any(value is not None for value in row.get("parent_deltas", {}).values()):
                row.pop("parent_deltas", None)
            if not row.get("actual_changes"):
                row.pop("actual_changes", None)
            if row.get("configuration_status"):
                row["configuration_status"] = "declared_only; runtime_not_recorded"
            if card["candidate_id"] not in role_ids and self._successful_full(card):
                if not card.get("ranking_reversal"):
                    row["results"].pop("coreset", None)
                    row.get("actual_changes", {}).pop("coreset", None)
                    row.get("parent_deltas", {}).pop("coreset", None)
                for key in ("failed", "evidence_stage", "declared_scaling_rules", "feature_mechanism"):
                    row.pop(key, None)
                if not row.get("ranking_reversal"):
                    row.pop("ranking_reversal", None)
            row.pop("detail_level", None)
            return row
        for card in required:
            details[card["candidate_id"]] = minimum(card)
        targets = {card["candidate_id"]: self._experiment_row(card, max_parameters=8)
                   for card in required if card["candidate_id"] in role_ids}
        diagnostics = {ident: self._parent_diagnostics(available[ident]) for ident in targets}
        features = {ident: self._feature_summary(available[ident]) for ident in targets}
        desired_history = list(dict.fromkeys(plan["recent_full"][:2] +
            ([plan["counterexample"]] if plan["counterexample"] else []) + plan["recent_full"][2:3]))

        def render(rows):
            context = {"cards": deepcopy(list(rows.values())), "estimated_tokens": self.token_budget,
                "token_count_method": self.token_count_method}
            scopes, scope_ids = {}, {}
            missing_parameters, missing_features = {}, []
            for row in context["cards"]:
                ident = row["candidate_id"]
                if ident in targets:
                    for stage, report in row["results"].items():
                        total = len(available[ident]["results"].get(stage, {}).get("effective_configuration", {}).get("model_parameters", {}))
                        if total:
                            report["omitted_parameter_count"] = total - len(report.get("actual_key_parameters", {}))
                    missing = sum(len(set(report.get("actual_key_parameters", {})) -
                        set(rows[ident]["results"].get(stage, {}).get("actual_key_parameters", {})))
                        for stage, report in targets[ident]["results"].items())
                    if missing:
                        missing_parameters[ident] = missing
                    fields = []
                    for stage, diag in diagnostics[ident].items():
                        current = rows[ident]["results"].get(stage, {})
                        missing_training = sum(key not in current.get("training", {}) for key in diag.get("training", {}))
                        if missing_training:
                            fields.append(stage + ".training")
                        fields.extend(stage + "." + key for key in ("warnings", "subgroup_errors")
                                      if key in diag and key not in current)
                    if fields:
                        row["omitted_parent_diagnostics"] = fields
                    if "feature_configuration" not in row:
                        missing_features.append(ident)
                if relations and ident in relations:
                    row["history_relation"] = relations[ident]
                self._coalesce_parameters(row)
                for report in row["results"].values():
                    scope = {key: report.pop(key) for key in ("metric", "direction", "validation_id", "evaluation_id", "coreset_version") if key in report}
                    if scope:
                        identifier = scope_ids.setdefault(_json(scope), f"v{len(scope_ids)}")
                        scopes[identifier] = scope
                        report["evaluation_scope"] = identifier
            context["evidence_ids"] = [row["evidence_id"] for row in context["cards"]]
            if scopes:
                context["evaluation_scopes"] = scopes
            if roles is not None:
                context["roles"] = {role: str(ident) if ident is not None and str(ident) in rows else None for role, ident in roles.items()}
                context["history_evidence_ids"] = [available[i]["evidence_id"] for i in rows if i not in role_ids and self._useful_full(available[i])]
                context["history_status"] = ("included_useful_full_history" if context["history_evidence_ids"] else
                    "no_distinct_useful_full_history" if not historical else "excluded_by_requested_zero_limit")
                context["history_selection"] = {"recent_full": [i for i in plan["recent_full"] if i in rows],
                    "counterexample": plan["counterexample"] if plan["counterexample"] in rows else None}
                omitted = {}
                if missing_parameters:
                    omitted["key_parameter_stage_values"] = missing_parameters
                if missing_features:
                    omitted["feature_details"] = missing_features
                absent_history = [i for i in desired_history if i not in rows]
                if absent_history:
                    omitted["history_ids"] = absent_history
                if omitted:
                    context["coverage_omissions"] = omitted
                context["budget_scope"] = "all_candidate_feedback"
                context["omitted_settings"] = "Preserve omitted parent settings/rules; they are not defaults."
            return context

        context = render(details)
        if self.count_tokens(context) > self.token_budget:
            return self._budget_fallback(details, roles, role_ids, historical)

        def commit(trial):
            nonlocal details, context
            proposal = render(trial)
            if self.count_tokens(proposal) > self.token_budget:
                return False
            details, context = trial, proposal
            return True

        def finish():
            context["estimated_tokens"] = self.count_tokens(context)
            return context

        # This prefix never depends on the requested budget. Optional history
        # and bulk diagnostics cannot displace already-visible role parameters.
        role_keys = {ident: list(dict.fromkeys(key for report in target["results"].values()
                                              for key in report.get("actual_key_parameters", {})))
                     for ident, target in targets.items()}
        for index in range(max((len(keys) for keys in role_keys.values()), default=0)):
            for ident, target in targets.items():
                if index >= len(role_keys[ident]):
                    continue
                key = role_keys[ident][index]
                trial = deepcopy(details)
                for stage, report in target["results"].items():
                    if key in report.get("actual_key_parameters", {}):
                        current = trial[ident]["results"][stage]
                        current.setdefault("actual_key_parameters", {})[key] = report["actual_key_parameters"][key]
                        current.update(parameter_source=report["parameter_source"], model_class=report["model_class"])
                if not commit(trial):
                    return finish()
        for ident, stages in diagnostics.items():
            for stage in ("full_train", "coreset"):
                for key, value in stages.get(stage, {}).get("training", {}).items():
                    trial = deepcopy(details)
                    trial[ident]["results"][stage].setdefault("training", {})[key] = value
                    if not commit(trial):
                        continue
                for key in ("warnings", "subgroup_errors"):
                    if key in stages.get(stage, {}):
                        trial = deepcopy(details)
                        trial[ident]["results"][stage][key] = stages[stage][key]
                        if not commit(trial):
                            continue
        for ident, feature in features.items():
            trial = deepcopy(details)
            trial[ident]["feature_configuration"] = feature
            trial[ident].pop("feature_mechanism", None)
            if not commit(trial):
                continue
        capacity = min(self.max_count, max(limit, len(role_ids) + len(desired_history))) if limit else 0
        for ident in desired_history:
            if ident in details or len(details) >= capacity:
                continue
            trial = deepcopy(details)
            trial[ident] = minimum(available[ident])
            if not commit(trial):
                return finish()
        for card in unique:
            ident = card["candidate_id"]
            if ident not in details and len(details) >= capacity:
                continue
            if ident in role_ids:
                continue
            trial = deepcopy(details)
            trial[ident] = self._compact(card)
            if not commit(trial) and ident not in details:
                trial[ident] = minimum(card)
                commit(trial)
        return finish()

    @property
    def token_count_method(self):
        return ("cl100k_base_times_1.2_approximate_not_deepseek" if self.tokenizer else
                "offline_chars_ascii_div3_unicode_times1.5_estimate_not_tokenizer")

    def count_tokens(self, value):
        """Estimate feedback tokens without requiring a tokenizer download.

        The offline fallback charges one token per three ASCII characters and
        1.5 per non-ASCII character. This is a budgeting heuristic, not an upper
        bound or the provider's exact usage. UTF-8 bytes are unsuitable here:
        equating bytes with tokens silently shrank a 3,000-token budget to a
        fraction of its intended size, especially for Chinese feedback.
        """
        encoded = _json(value)
        if self.tokenizer:
            return math.ceil(len(self.tokenizer.encode_ordinary(encoded)) * 1.2)
        ascii_count = sum(ord(char) < 128 for char in encoded)
        return math.ceil(ascii_count / 3 + (len(encoded) - ascii_count) * 1.5)

    def _budget_fallback(self, details, roles, role_ids, historical):
        """Keep measured evidence when even the normal minimum is oversized.

        Large warnings, repair transcripts and declared rules are dropped as
        whole fields. Exact parameter values, score identities and IDs are
        never truncated into different facts. If a genuinely tiny budget
        cannot hold all required cards, omissions are explicit instead of
        aborting the search. The complete archive is unaffected.
        """
        reduced = {}
        priority = list(dict.fromkeys(str((roles or {}).get(role)) for role in ("parent", "global_best")
                                     if (roles or {}).get(role) is not None))
        priority.extend(ident for ident in details if ident not in priority)
        for ident in priority:
            if ident not in details:
                continue
            source = details[ident]
            row = {key: deepcopy(source[key]) for key in
                   ("candidate_id", "parent_id", "evidence_id", "actual_changes") if key in source}
            row["results"] = {}
            for stage, report in source["results"].items():
                row["results"][stage] = {key: deepcopy(report[key]) for key in
                    ("status", "score", "metric", "direction", "validation_id", "evaluation_id",
                     "actual_key_parameters", "parameter_source", "model_class", "coreset_version") if key in report}
            self._coalesce_parameters(row)
            reduced[ident] = row

        def render(rows):
            result = {"cards": list(rows.values()), "evidence_ids": [row["evidence_id"] for row in rows.values()],
                "estimated_tokens": self.token_budget, "token_count_method": self.token_count_method,
                "packing_status": "reduced_to_fit_budget", "omitted_card_count": len(details) - len(rows),
                "omitted_settings": "Preserve omitted settings; complete records remain archived."}
            if roles is not None:
                result["roles"] = {role: str(ident) if ident is not None and str(ident) in rows else None
                                   for role, ident in roles.items()}
                result["history_evidence_ids"] = [row["evidence_id"] for ident, row in rows.items()
                    if ident not in role_ids and ident in {card["candidate_id"] for card in historical}]
                result["history_status"] = ("included_useful_full_history" if result["history_evidence_ids"] else
                    "excluded_by_budget" if historical else "no_distinct_useful_full_history")
            return result

        result = render(reduced)
        if self.count_tokens(result) > self.token_budget:
            # Retain the strongest exact evidence per card before dropping any
            # role/history card. Full-only here is marked by the reduced schema.
            for row in reduced.values():
                full = row["results"].get("full_train", {})
                if _number(full.get("score")):
                    row["results"].pop("coreset", None)
                    row.get("actual_changes", {}).pop("coreset", None)
                    for key in ("by_stage", "source_by_stage", "model_class_by_stage",
                                "omitted_count_by_stage", "selection_by_stage"):
                        row.get("effective_parameters", {}).get(key, {}).pop("coreset", None)
                row.pop("parent_id", None)
            result = render(reduced)
        if self.count_tokens(result) > self.token_budget:
            kept = {}
            for ident, row in reduced.items():
                trial = {**kept, ident: row}
                if self.count_tokens(render(trial)) <= self.token_budget:
                    kept = trial
            result = render(kept)
        if self.count_tokens(result) > self.token_budget:
            # Metadata itself can exceed a caller's minimum 128-token budget.
            result = {"cards": [], "evidence_ids": [], "estimated_tokens": self.token_budget,
                "token_count_method": self.token_count_method, "packing_status": "budget_too_small_for_evidence"}
        result["estimated_tokens"] = self.count_tokens(result)
        return result

    def retrieve(self, parent_id, ancestor_ids=(), query=None, limit=None, stalled=False, global_best_id=None):
        """Recent route evidence and a counterexample, with explicit relationships.

        Ancestors and siblings are never labelled as the same sequential path.
        Relevance chooses the pool; explicit observation chronology breaks ties.
        """
        if not self.enabled:
            return {**self._pack([], 0), 'disabled': True}
        limit = min(self.max_count, self.default_count if limit is None else max(0, int(limit)))
        parent_id = str(parent_id)
        parent = self.cards.get(parent_id)
        best = self.cards.get(str(global_best_id)) if global_best_id is not None else None
        if not parent and not best:
            return self._pack([], limit, roles={"parent": None, "global_best": None})
        query = dict(query or {})
        query_idea = _idea(query)
        query.setdefault("family", query_idea.get("family") or (parent or {}).get("configuration", {}).get("family"))
        query.setdefault("direction", query.get("modification_direction", query.get("mutation_direction")) or
                         (parent or {}).get("mutation", {}).get("direction"))
        query.setdefault("fit_rows", (query.get("low_report") or {}).get("fit_rows"))
        ancestors = {str(value) for value in ancestor_ids}
        current, visited = parent, set()
        while current and current.get("parent_id") and current["parent_id"] not in visited:
            ident = current["parent_id"]
            ancestors.add(ident); visited.add(ident)
            current = self.cards.get(ident)
        role_ids = {card["candidate_id"] for card in (parent, best) if card}
        relations = {}
        for ident, card in self.cards.items():
            if ident == parent_id:
                relation = "parent"
            elif ident in ancestors:
                relation = "ancestor"
            elif card["parent_id"] == parent_id:
                relation = "child"
            elif parent and parent.get("parent_id") is not None and card["parent_id"] == parent["parent_id"]:
                relation = "sibling"
            elif parent and card["branch_id"] == parent["branch_id"]:
                relation = "same_branch"
            else:
                relation = "other_branch"
            relations[ident] = relation
        insertion_order = {ident: index for index, ident in enumerate(self.cards)}
        def chronology(card):
            # Existing archives predate observation_order. Numeric expansion
            # rounds, unlike lexical IDs, still give their actual search order.
            match = re.match(r"expansion_(\d+)_", card["candidate_id"])
            if match:
                return int(match.group(1)), card.get("observation_order", insertion_order[card["candidate_id"]])
            if card["candidate_id"].startswith("initial_"):
                return -1, card.get("observation_order", insertion_order[card["candidate_id"]])
            return 0, card.get("observation_order", insertion_order[card["candidate_id"]])
        available = [c for c in self.cards.values() if c["candidate_id"] not in role_ids]
        related = [c for c in available if relations[c["candidate_id"]] != "other_branch" or self._relevance(c, query) >= 3]
        full = [c for c in related if self._useful_full(c)]
        # First preserve actual ancestor continuity, then recent comparable
        # children/siblings on the route; relations remain visible in the prompt.
        chain = sorted([c for c in full if relations[c["candidate_id"]] == "ancestor"], key=chronology, reverse=True)
        route = sorted([c for c in full if relations[c["candidate_id"]] in {"child", "sibling", "same_branch"}], key=chronology, reverse=True)
        recent = (chain + route)[:3]
        def counterexample(card):
            delta = card.get("parent_deltas", {}).get("full_train")
            direction = card["results"].get("full_train", {}).get("direction")
            return card["ranking_reversal"] or (delta is not None and
                ((direction == "maximize" and delta < 0) or (direction == "minimize" and delta > 0)))
        counter = sorted([c for c in full if counterexample(c)],
                         key=lambda c: (relations[c["candidate_id"]] != "other_branch", self._relevance(c, query), chronology(c)), reverse=True)
        role_counter = next((c for c in (parent, best) if c and self._successful_full(c) and counterexample(c)), None)
        counter = role_counter or (counter[0] if counter else None)
        remaining = sorted(related, key=lambda c: (self._relevance(c, query), self._successful_full(c), chronology(c)), reverse=True)
        fallback = sorted([c for c in available if self._useful_full(c)], key=chronology, reverse=True)
        selected = [c for c in (parent, best) if c] + recent[:2] + ([counter] if counter else []) + recent[2:] + remaining + fallback
        return self._pack(selected, limit, roles={"parent": parent_id if parent else None,
                          "global_best": str(global_best_id) if best else None},
                          history_plan={"recent_full": [c["candidate_id"] for c in recent],
                                        "counterexample": counter["candidate_id"] if counter else None}, relations=relations)

    @staticmethod
    def _method_path(card):
        row = ExperienceMemory._experiment_row(card, max_parameters=3)
        full = row["results"]["full_train"]
        return dict(candidate_id=row["candidate_id"], parent_id=row["parent_id"],
            branch_id=card["branch_id"], evidence_id=row["evidence_id"], name=row["name"],
            family=row["family"], model_class=full.get("model_class") or row["results"]["coreset"].get("model_class"),
            method_summary=card.get("method_summary"), change_reason=card["mutation"].get("reason"), screening=card.get("screening"),
            change_direction=card["mutation"].get("direction"),
            feature_engineering=_short(card["configuration"].get("feature_engineering"), 180),
            parameter_stage=("full_train" if full["actual_key_parameters"] else "coreset"
                             if row["results"]["coreset"]["actual_key_parameters"] else None),
            parameters=full["actual_key_parameters"] or row["results"]["coreset"]["actual_key_parameters"], full_changes=row["actual_changes"]["full_train"],
            full_score=full.get("score"), metric=full.get("metric"), direction=full.get("direction"),
            parent_full_delta=card["parent_deltas"].get("full_train"),
            coreset_score=row["results"]["coreset"].get("score"),
            coreset_version=row["results"]["coreset"].get("coreset_version"))

    def generation_evidence(self, parent_id, **kwargs):
        """Add the complete measured search tree to the bounded detailed cards."""
        context = self.retrieve(parent_id, **kwargs)
        if not self.enabled:
            return context
        paths = []
        for card in self.cards.values():
            if not self._successful_full(card):
                continue
            paths.append(card.get("method_path") or self._method_path(card))
        context["full_method_paths"] = paths
        parent = self.cards.get(str(parent_id), {})
        sibling_rows = [c["method_path"] for c in self.cards.values()
                        if c.get("screening") and "method_path" in c and not self._successful_full(c)
                        and c["parent_id"] in (parent_id, parent.get("parent_id"))]
        if sibling_rows:
            context["screened_only_methods"] = sibling_rows
        context["evidence_ids"] = list(dict.fromkeys(context["evidence_ids"] + [r["evidence_id"] for r in paths + sibling_rows]))
        return context

    def retrieve_repairs(self, error, limit=2):
        if not self.enabled:
            return repair_only_context([])
        signature = _error_signature(error)
        candidates = [card for card in self.cards.values() if signature and signature in card["error_signatures"]]
        candidates.sort(key=lambda card: (card["failed"], card["candidate_id"]))
        result = repair_only_context(candidates, min(2, max(0, int(limit))))
        # Repair evidence has a separate schema: the normal experiment packer
        # includes scores, deltas and selection decisions, none relevant here.
        while result["cards"] and self.count_tokens(result) > self.token_budget:
            result["cards"].pop()
            result["evidence_ids"] = [card.get("evidence_id", card.get("candidate_id", card.get("id")))
                                      for card in result["cards"]]
        return result
