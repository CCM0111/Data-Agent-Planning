"""Bounded, descriptive views of serialized estimator configurations; no inference.

Diffs compare the original serialized values, not the selective summaries. Paths
are JSON Pointers. Hashes describe the supplied serialization, including any
process-specific identifiers already present in an opaque callable's snapshot.
"""

import hashlib
import json
import re


MAX_DEPTH = 5
MAX_COMPONENTS = 24
MAX_PARAMETERS = 16
MAX_STRING_LENGTH = 120
MAX_VALUE_ITEMS = 6
MAX_CHANGES = 64
MAX_PATH_LENGTH = 512
_MISSING = object()
_RANDOM_KEYS = frozenset({"random_state", "random_seed", "seed"})
_SOURCE_KEYS = frozenset({"code", "source", "source_code", "repr"})
_KEY_PARAMETERS = (
    "strategy", "fill_value", "add_indicator", "keep_empty_features",
    "handle_unknown", "drop", "min_frequency", "max_categories", "sparse_output",
    "sparse", "dtype", "with_mean", "with_std", "with_centering", "with_scaling",
    "quantile_range", "norm", "feature_range", "clip", "n_quantiles",
    "output_distribution", "standardize", "degree", "include_bias", "interaction_only",
    "threshold", "k", "percentile", "n_components", "whiten", "func", "inverse_func",
    "kw_args", "inv_kw_args", "remainder", "sparse_threshold", "transformer_weights",
    "loss", "penalty", "solver", "C", "alpha", "l1_ratio", "class_weight",
    "fit_intercept", "positive", "max_iter", "n_estimators", "learning_rate",
    "max_depth", "max_leaf_nodes", "l2_regularization", "min_samples_split", "min_samples_leaf", "max_features", "criterion",
    "bootstrap", "subsample", "colsample_bytree", "gamma", "reg_lambda", "reg_alpha",
    "objective", "boosting_type", "num_leaves", "n_neighbors", "weights", "algorithm",
    "leaf_size", "p", "metric", "kernel", "coef0", "shrinking", "nu", "epsilon",
    "hidden_layer_sizes", "activation", "batch_size", "early_stopping",
    "validation_fraction", "n_iter_no_change", "momentum", "beta_1", "beta_2",
    "warm_start", "tol",
)


def _serialized(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _hash(value):
    return hashlib.sha256(_serialized(value).encode("utf-8")).hexdigest()


def _is_config(value):
    return (isinstance(value, dict) and isinstance(value.get("class"), str)
            and bool(value["class"]) and isinstance(value.get("parameters"), dict))


def _descriptor(value):
    """An opaque but change-sensitive description; never contains code or arrays."""
    kind = type(value).__name__
    if isinstance(value, dict) and isinstance(value.get("type"), str):
        label = value["type"]
        if len(label) <= MAX_STRING_LENGTH and re.fullmatch(r"[\w.]+", label):
            kind = label
    result = {"type": kind, "sha256": _hash(value), "summarized": True}
    if isinstance(value, (dict, list, str)):
        result["length"] = len(value)
    return result


def _source_text(value):
    return ("\n" in value or "\r" in value
            or re.search(r"(?:^|;)\s*(?:import |from \S+ import |def \w+\(|class \w+[:(])", value)
            or re.search(r"\blambda\b[^:]*:", value))


def _compact(value, depth=0):
    if isinstance(value, str):
        return _descriptor(value) if len(value) > MAX_STRING_LENGTH or _source_text(value) else value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict) and value.get("type") == "importable_callable":
        return {key: _compact(value[key], depth + 1)
                for key in ("type", "module", "qualname") if key in value}
    if isinstance(value, dict) and value.get("type") in {
            "opaque_callable", "functools.partial", "numpy.random.RandomState", "numpy.random.Generator"}:
        return _descriptor(value)
    if depth >= 2 or len(value) > MAX_VALUE_ITEMS:
        return _descriptor(value)
    if isinstance(value, dict):
        # Arbitrary mapping keys may contain source or long user-provided names.
        if any(not isinstance(key, str) or len(key) > MAX_STRING_LENGTH or _source_text(key)
               or key in _SOURCE_KEYS or key in _RANDOM_KEYS for key in value):
            return _descriptor(value)
        return {key: _compact(item, depth + 1) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_compact(item, depth + 1) for item in value]
    return _descriptor(value)


def _selector(value):
    """Record selector form and known size without exposing feature-name lists."""
    if isinstance(value, list):
        if not value:
            return {"selection": "empty_sequence", "count": 0}
        if all(isinstance(item, bool) for item in value):
            return {"selection": "boolean_mask", "count": sum(value), "mask_length": len(value)}
        if all(isinstance(item, str) for item in value):
            return {"selection": "column_names", "count": len(value)}
        if all(type(item) is int for item in value):
            return {"selection": "column_positions", "count": len(value)}
        return {"selection": "sequence", "count": len(value)}
    if isinstance(value, dict) and value.get("type") in {
            "importable_callable", "opaque_callable", "functools.partial"}:
        return {"selection": "callable", "count": None}
    if isinstance(value, str):
        if value.startswith("slice(") and value.endswith(")"):
            return {"selection": "serialized_slice", "count": None}
        return {"selection": "column_name", "count": 1}
    if type(value) is int:
        return {"selection": "column_position", "count": 1}
    return {"selection": "unknown", "count": None}


def summarize_config(config):
    """Summarize observed components and selected parameters, or return None.

    Missing options stay absent; this function does not fill estimator defaults.
    ``omitted_parameters`` includes parameters outside the allowlist. ``truncated``
    marks component/depth/parameter budgets, while opaque values mark themselves
    with ``summarized`` and a digest. Column selectors intentionally omit names.
    """
    if not _is_config(config):
        return None
    state = {"components": 0, "truncated": False}

    def component(value, depth):
        state["components"] += 1
        if not _is_config(value):
            return _compact(value)
        if depth >= MAX_DEPTH:
            state["truncated"] = True
            return {"class": _compact(value["class"]), "truncated": True,
                    "reason": "component_or_depth_limit", "sha256": _hash(value)}
        parameters = value["parameters"]
        result = {"class": _compact(value["class"]), "parameters": {}}
        handled = set()
        for key in ("steps", "transformers", "transformer_list"):
            branches = parameters.get(key)
            if not isinstance(branches, list):
                continue
            handled.add(key)
            entries = []
            for index, branch in enumerate(branches):
                if len(entries) >= MAX_COMPONENTS or state["components"] >= MAX_COMPONENTS:
                    state["truncated"] = True
                    result[f"omitted_{key}"] = len(branches) - index
                    break
                if not isinstance(branch, list) or len(branch) not in (2, 3):
                    state["components"] += 1
                    entries.append({"structure": _descriptor(branch)})
                    continue
                entry = {"name": _compact(branch[0]), "component": component(branch[1], depth + 1)}
                if len(branch) == 3:
                    entry["columns"] = _selector(branch[2])
                entries.append(entry)
            result[key] = entries
        nested = [(key, item) for key, item in sorted(parameters.items())
                  if key not in handled and _is_config(item)]
        if nested:
            result["components"] = {}
        for index, (key, item) in enumerate(nested):
            if state["components"] >= MAX_COMPONENTS or index >= MAX_PARAMETERS:
                state["truncated"] = True
                result["omitted_components"] = len(nested) - index
                break
            if len(key) > MAX_STRING_LENGTH or _source_text(key):
                state["truncated"] = True
                continue
            result["components"][key] = component(item, depth + 1)
            handled.add(key)
        selected = [key for key in _KEY_PARAMETERS if key in parameters and key not in handled]
        if len(selected) > MAX_PARAMETERS:
            state["truncated"] = True
        for key in selected[:MAX_PARAMETERS]:
            result["parameters"][key] = _compact(parameters[key])
            handled.add(key)
        omitted = len(parameters) - len(handled)
        if omitted:
            result["omitted_parameters"] = omitted
        return result

    result = component(config, 0)
    result["truncated"] = state["truncated"]
    result["component_count"] = state["components"]
    return result


def compare_configs(before, after):
    """Compare raw snapshots, keeping at most MAX_CHANGES descriptive changes.

    Arrays exceeding MAX_VALUE_ITEMS, selectors, random state and source text are
    compared in full but displayed as digest summaries. Additions/removals use an
    explicit missing marker, distinct from a present null. Traversal continues
    beyond the display budget so ``change_count`` also counts omitted changes.
    """
    if not _is_config(before) or not _is_config(after):
        return {"status": "unavailable", "changes": [], "truncated": False,
                "reason": "missing_configuration"}
    changes = []
    state = {"count": 0, "truncated": False}

    def display(value, opaque=False, selector=False):
        if value is _MISSING:
            return {"missing": True}
        if selector:
            return {**_selector(value), **_descriptor(value)}
        if isinstance(value, dict) or (isinstance(value, list) and any(
                item is not None and not isinstance(item, (int, float, bool)) for item in value)):
            return _descriptor(value)
        return _descriptor(value) if opaque else _compact(value)

    def record(left, right, parts, opaque=False, selector=False):
        state["count"] += 1
        if len(changes) >= MAX_CHANGES:
            state["truncated"] = True
            return
        path = "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in parts)
        item = {"path": path, "before": display(left, opaque, selector),
                "after": display(right, opaque, selector)}
        if len(path) > MAX_PATH_LENGTH or _source_text(path):
            item["path"] = "/[summarized-path]"
            item["path_sha256"] = _hash(parts)
            item["path_truncated"] = True
            state["truncated"] = True
        if left is _MISSING or right is _MISSING:
            item["kind"] = "added" if left is _MISSING else "removed"
        changes.append(item)

    def walk(left, right, parts, depth=0):
        selector = (len(parts) >= 3 and parts[-3] == "transformers"
                    and isinstance(parts[-2], int) and parts[-1] == 2)
        key = parts[-1] if parts else None
        opaque = key in _RANDOM_KEYS or key in _SOURCE_KEYS
        encoded = any(isinstance(value, dict) and value.get("type") in {
            "opaque_callable", "functools.partial", "numpy.random.RandomState", "numpy.random.Generator"}
            for value in (left, right))
        if left is _MISSING or right is _MISSING:
            record(left, right, parts, opaque=opaque, selector=selector)
        elif opaque or encoded or selector or depth >= 64:
            if _serialized(left) != _serialized(right):
                record(left, right, parts, opaque=opaque or encoded or depth >= 64, selector=selector)
        elif isinstance(left, dict) and isinstance(right, dict):
            for name in sorted(left.keys() | right.keys()):
                walk(left.get(name, _MISSING), right.get(name, _MISSING), parts + [name], depth + 1)
        elif isinstance(left, list) and isinstance(right, list):
            if max(len(left), len(right)) > MAX_VALUE_ITEMS and key not in {
                    "steps", "transformers", "transformer_list"}:
                if _serialized(left) != _serialized(right):
                    record(left, right, parts, opaque=True)
            else:
                for index in range(max(len(left), len(right))):
                    walk(left[index] if index < len(left) else _MISSING,
                         right[index] if index < len(right) else _MISSING, parts + [index], depth + 1)
        elif _serialized(left) != _serialized(right):
            record(left, right, parts)

    walk(before, after, [])
    return {"status": "available", "changes": changes, "truncated": state["truncated"],
            "change_count": state["count"], "omitted_changes": state["count"] - len(changes)}
