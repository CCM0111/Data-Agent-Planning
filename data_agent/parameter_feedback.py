"""Validate and freeze one feedback-based parameter plan before V_core scoring.

Rules are data, never Python expressions. They modify a small whitelist of main
model parameters on a clone of the original unfitted proxy pipeline; feature
engineering and all other method choices stay fixed.
"""

from copy import deepcopy
import json
import math
from numbers import Integral

from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import FeatureUnion, Pipeline

if __package__:
    from .model_registry import MODEL_REGISTRY
else:
    from model_registry import MODEL_REGISTRY


ALLOWED_PARAMETERS = frozenset({
    'max_depth', 'max_leaf_nodes', 'min_samples_leaf', 'min_samples_split',
    'max_features', 'alpha', 'C', 'l1_ratio', 'n_neighbors', 'weights',
    'learning_rate', 'learning_rate_init', 'n_estimators', 'max_iter', 'tol',
    'l2_regularization',
})

PARAMETER_FEEDBACK_INSTRUCTION = '''Use the supplied P-training/Q-prediction feedback, training sizes, training-only EDA, original unfitted methods and stage budget caps to make exactly ONE parameter revision. Do not use V_core or external validation/test observations. Preserve each exact model class, feature engineering, loss, solver, architecture, seeds and parallelism. Every coreset arm and seed must use the same frozen coreset parameters for each method. No code, expressions, extra methods or tuning trials.
P/Q performance gaps describe observations at the supplied P training size, not evidence that full training also needs stronger regularization. Separate observed facts from full-stage hypotheses in reason. Use the task primary metric and its source; do not infer AUC generalization from an accuracy gap. Do not automatically grow min_samples_leaf or regularization with training rows. Any scaling rule needs a model-specific justification. With insufficient evidence, preserve the supplied original stage baseline: rules may be empty.
Use actual P fit timing and runtime_risk estimates to avoid exceeding stage wall time. Caps are maxima, not targets. For high-risk stages, explicitly predeclare smaller resource counts and/or bounded tree depth/larger leaves, using only permitted rules, with substantial prediction-time headroom. Timing estimates may underestimate nonlinear costs. Do not change the estimator or loss to hide an infeasible design.
Return JSON only: {"methods":[{"method_id":0,"reason":"feedback-based rationale","rules":{"model__min_samples_leaf":{"kind":"power","value":10,"exponent":0.5,"rounding":"ceil","min":1,"max":100},"model__max_depth":{"kind":"constant","value":8},"model__n_estimators":{"kind":"stages","coreset":80,"full_train":300}}}]}.
Include every supplied method_id exactly once. An empty rules object preserves the validated original coreset/full_train stage baselines, including their declared parameter differences. Rules override only their named parameters. Only existing main-model parameters in this whitelist may change: max_depth, max_leaf_nodes, min_samples_leaf, min_samples_split, max_features, alpha, C, l1_ratio, n_neighbors, weights, learning_rate, learning_rate_init, n_estimators, max_iter, tol, l2_regularization. HGB l2_regularization must be finite and nonnegative; zero is valid. Other pipeline parameters remain exactly as in the supplied original stage baselines.
There are three rule kinds. constant has exactly kind/value and applies the same JSON scalar to both stages. power has kind/value/exponent/rounding and optional numeric min/max: value * (stage_rows / coreset_rows) ** exponent, rounded using ceil, floor, nearest (half upward), or none, then explicitly bounded by min/max if supplied. The anchor is always coreset_rows, exponent must lie in [-1,1], and all numbers must be finite. stages has exactly kind/coreset/full_train and is allowed only for n_estimators or max_iter. Give integer values for integer parameters. Resource values must be positive and must not exceed that stage's cap. Omitting a resource rule preserves its supplied stage baseline, not the low-cost proxy budget; it does not increase a declared below-cap count to the cap. Respect n_neighbors <= stage_rows and choose tree leaf/split sizes that permit a split for each stage. There is no silent clipping or fallback for an invalid plan. Explain regularization scaling using that model's loss normalization; no universal alpha scaling is assumed. These are prospective parameter choices, not measured performance improvements.'''


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def normalize_parameter_names(response, pipelines):
    """Audited syntax repair only: alpha -> model__alpha on that exact main model."""
    normalized = deepcopy(response)
    changes = []
    for item in normalized.get('methods', []) if isinstance(normalized, dict) else []:
        if not isinstance(item, dict) or type(item.get('method_id')) is not int:
            continue
        i = item['method_id']
        if i not in range(len(pipelines)) or not isinstance(item.get('rules'), dict):
            continue
        available = pipelines[i]['model'].get_params(deep=False)
        rules = item['rules']
        for name in list(rules):
            if name in ALLOWED_PARAMETERS and name in available:
                canonical = 'model__' + name
                if canonical in rules:
                    raise ValueError(f'Duplicate feedback parameter aliases: {name}, {canonical}')
                rules[canonical] = rules.pop(name)
                changes.append({'method_id': i, 'from': name, 'to': canonical})
    return normalized, changes


def _number(value, name):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f'{name} must be a finite number, not an expression')
    return value


def _scalar(value, name):
    if value is None or type(value) is str:
        return value
    return _number(value, name)


def _rule_value(rule, parameter, stage, rows, anchor):
    if not isinstance(rule, dict):
        raise ValueError(f'{parameter}: rule must be an object')
    kind = rule.get('kind')
    if kind == 'constant':
        if set(rule) != {'kind', 'value'}:
            raise ValueError(f'{parameter}: constant requires exactly kind/value')
        return _scalar(rule['value'], parameter)
    if kind == 'stages':
        if parameter not in ('n_estimators', 'max_iter'):
            raise ValueError(f'{parameter}: stages is restricted to training resource controls')
        if set(rule) != {'kind', 'coreset', 'full_train'}:
            raise ValueError(f'{parameter}: stages requires both stages and no extra fields')
        return _positive_integer(rule[stage], parameter)
    if kind != 'power':
        raise ValueError(f'{parameter}: unknown rule kind {kind!r}; expressions are forbidden')
    required = {'kind', 'value', 'exponent', 'rounding'}
    if not required <= set(rule) or set(rule) - required - {'min', 'max'}:
        raise ValueError(f'{parameter}: power requires value/exponent/rounding and optional min/max')
    base = _number(rule['value'], parameter + '.value')
    exponent = _number(rule['exponent'], parameter + '.exponent')
    if not -1 <= exponent <= 1:
        raise ValueError(f'{parameter}: power exponent must be in [-1, 1]')
    rounding = rule['rounding']
    if rounding not in ('none', 'ceil', 'floor', 'nearest'):
        raise ValueError(f'{parameter}: unsupported rounding')
    lower = _number(rule['min'], parameter + '.min') if 'min' in rule else None
    upper = _number(rule['max'], parameter + '.max') if 'max' in rule else None
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(f'{parameter}: min exceeds max')
    try:
        value = _number(base * (rows / anchor) ** exponent, parameter)
    except OverflowError as error:
        raise ValueError(f'{parameter}: nonfinite scaled parameter') from error
    if rounding == 'ceil':
        value = math.ceil(value)
    elif rounding == 'floor':
        value = math.floor(value)
    elif rounding == 'nearest':
        value = math.floor(value + 0.5)
    if lower is not None:
        value = max(lower, value)
    if upper is not None:
        value = min(upper, value)
    return value


def _resource_control(model, stage_budgets):
    name = type(model).__name__
    registry = MODEL_REGISTRY.get(name)
    if registry is None:
        raise ValueError(f'No supported model budget rule for {name}')
    if registry.parameter is None:
        return None, {}
    caps = {}
    for stage, budget in stage_budgets.items():
        caps[stage] = _positive_integer(budget.get(registry.budget_key),
                                        f'{stage}.{registry.budget_key}')
    return registry.parameter, caps


def validate_estimators(value, seen=None):
    seen = seen if seen is not None else set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, BaseEstimator):
        validator = getattr(value, '_validate_params', None)
        # Some sklearn composites inherit the generic validator without the
        # constraint table it requires. Their children still need validation.
        if validator and (getattr(validator, '__func__', None) is not BaseEstimator._validate_params
                          or hasattr(value, '_parameter_constraints')):
            validator()
        if isinstance(value, FeatureUnion):
            value._validate_transformers()
            value._validate_transformer_weights()
        for child in value.get_params(deep=False).values():
            validate_estimators(child, seen)
    elif isinstance(value, dict):
        for child in value.values():
            validate_estimators(child, seen)
    elif isinstance(value, (list, tuple)):
        for child in value:
            validate_estimators(child, seen)


def _validate_stage(model, updates, rows, stage, resource, cap):
    candidate = clone(model).set_params(**{key[7:]: value for key, value in updates.items()})
    # sklearn's own estimator domains distinguish integer counts, fractional
    # leaf sizes, per-model learning-rate enums, and nullable parameters.
    try:
        validate_estimators(candidate)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{stage}: invalid {type(model).__name__} parameters: {error}') from error
    params = candidate.get_params(deep=False)
    if resource:
        value = _positive_integer(params[resource], f'{stage}.{resource}')
        if value > cap:
            raise ValueError(f'{stage}.{resource}={value} exceeds budget cap {cap}')
    if 'n_neighbors' in params and params['n_neighbors'] > rows:
        raise ValueError(f'{stage}.n_neighbors exceeds training rows {rows}')
    if rows > 1:
        if 'min_samples_leaf' in params:
            leaf = params['min_samples_leaf']
            count = leaf if isinstance(leaf, Integral) else math.ceil(leaf * rows)
            if 2 * count > rows:
                raise ValueError(f'{stage}.min_samples_leaf prevents any split with {rows} rows')
        if 'min_samples_split' in params:
            split = params['min_samples_split']
            count = split if isinstance(split, Integral) else math.ceil(split * rows)
            if count > rows:
                raise ValueError(f'{stage}.min_samples_split exceeds training rows {rows}')


def complete_stage_parameters(stage_parameters, stage_pipelines):
    """Fill a one-stage override's other-stage value from its unfitted baseline.

    Inputs have coreset/full_train keys. Values already declared remain exact;
    omitted keys use the corresponding actual pipeline, never the P defaults or
    the other stage's override. No fitting or mutation is performed.
    """
    phases = {'coreset', 'full_train'}
    if (not isinstance(stage_parameters, dict) or set(stage_parameters) != phases
            or not isinstance(stage_pipelines, dict) or set(stage_pipelines) != phases
            or any(not isinstance(value, dict) for value in stage_parameters.values())):
        raise ValueError('Stage completion requires coreset/full_train parameters and pipelines')
    result = deepcopy(stage_parameters)
    keys = set().union(*(value.keys() for value in result.values()))
    for phase in sorted(phases):
        available = stage_pipelines[phase].get_params(deep=True)
        for key in sorted(keys):
            if key not in available:
                raise ValueError(f'{phase}: parameter absent from stage baseline: {key}')
            if key not in result[phase]:
                result[phase][key] = deepcopy(available[key])
    return result


def freeze_parameter_plan(response, pipelines, n_core, n_full, budgets, stage_baselines=None,
                          stage_pipelines=None):
    """Return audited concrete coreset/full overrides without fitting or mutation.

    ``pipelines`` are the original unfitted proxy pipelines in method-ID order.
    ``stage_pipelines`` optionally supplies the actual unfitted formal-stage
    baselines in that same order. New feedback uses these instead of carrying P
    safeguards into later stages; old callers without them keep their contract.
    Applying the returned overrides to clones freezes the entire method because
    no stage calls the generated factory again. ``stage_baselines``, when
    supplied, is an ordered list of each method's original coreset/full_train
    overrides. Revisions overlay this baseline instead of erasing its declared
    stage differences. Missing registered resources retain the original common
    formal-stage budget; explicit resource counts stay exact. Invalid feedback raises rather
    than silently reverting to the previous parameter scheme.
    """
    _positive_integer(n_core, 'n_core')
    _positive_integer(n_full, 'n_full')
    if not isinstance(budgets, dict) or set(budgets) != {'coreset', 'full_train'}:
        raise ValueError('budgets must contain exactly coreset and full_train')
    if any(not isinstance(budget, dict) for budget in budgets.values()):
        raise ValueError('stage budgets must be objects')
    if stage_baselines is not None and (not isinstance(stage_baselines, (list, tuple))
                                        or len(stage_baselines) != len(pipelines)):
        raise ValueError('stage_baselines must include each method in method-ID order')
    if stage_pipelines is not None and (not isinstance(stage_pipelines, (list, tuple))
                                       or len(stage_pipelines) != len(pipelines)):
        raise ValueError('stage_pipelines must include each method in method-ID order')
    if not isinstance(response, dict) or set(response) != {'methods'}:
        raise ValueError('response must contain exactly methods')
    methods = response['methods']
    if not isinstance(methods, list) or not pipelines or len(methods) != len(pipelines):
        raise ValueError('methods must include every method_id exactly once')
    by_id = {}
    for item in methods:
        if not isinstance(item, dict) or set(item) != {'method_id', 'reason', 'rules'}:
            raise ValueError('Each method requires exactly method_id/reason/rules')
        method_id = item['method_id']
        if type(method_id) is not int or method_id in by_id or method_id not in range(len(pipelines)):
            raise ValueError('methods must include every method_id exactly once')
        if not isinstance(item['reason'], str) or not item['reason'].strip():
            raise ValueError('Every method needs a feedback-based reason')
        if not isinstance(item['rules'], dict):
            raise ValueError('rules must be an object')
        by_id[method_id] = item

    rows_by_stage = {'coreset': n_core, 'full_train': n_full}
    frozen = []
    for method_id, pipeline in enumerate(pipelines):
        if not isinstance(pipeline, Pipeline) or not pipeline.steps or pipeline.steps[-1][0] != 'model':
            raise ValueError('Each pipeline must end with the main model step')
        model = pipeline.steps[-1][1]
        available = model.get_params(deep=False)
        item = by_id[method_id]
        rules = deepcopy(item['rules'])
        for key in rules:
            if not isinstance(key, str) or not key.startswith('model__'):
                raise ValueError(f'{key!r}: only whitelisted model__ parameters may change')
            parameter = key[7:]
            if parameter not in ALLOWED_PARAMETERS or parameter not in available:
                raise ValueError(f'{key}: parameter is forbidden or absent from this model')
        resource, caps = _resource_control(model, budgets)
        originals = (stage_pipelines[method_id] if stage_pipelines is not None
                     else {phase: pipeline for phase in rows_by_stage})
        if not isinstance(originals, dict) or set(originals) != set(rows_by_stage):
            raise ValueError('Each formal baseline requires coreset and full_train pipelines')
        for phase, original in originals.items():
            if (not isinstance(original, Pipeline) or not original.steps or original.steps[-1][0] != 'model'
                    or type(original['model']) is not type(model)):
                raise ValueError(f'{phase}: original stage pipeline must preserve the proxy model class')
        baseline = deepcopy(stage_baselines[method_id]) if stage_baselines is not None else {
            'coreset': {}, 'full_train': {}}
        if not isinstance(baseline, dict) or set(baseline) != set(rows_by_stage):
            raise ValueError('Each stage baseline requires coreset and full_train')
        for stage, parameters in baseline.items():
            pipeline_parameters = originals[stage].get_params(deep=True)
            if not isinstance(parameters, dict):
                raise ValueError(f'{stage}: baseline must be a parameter object')
            for key, value in parameters.items():
                if (not isinstance(key, str) or '__' not in key or key not in pipeline_parameters
                        or key.split('__')[-1] in {'steps', 'transformers', 'estimator', 'base_estimator',
                                                   'random_state', 'n_jobs', 'warm_start', 'early_stopping'}):
                    raise ValueError(f'{stage}: forbidden or unknown original stage parameter {key!r}')
                try:
                    json.dumps(value, allow_nan=False)
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError(f'{stage}.{key}: baseline must contain finite JSON values') from error
        # A key overridden only in coreset must explicitly reset to its original
        # value in full_train; otherwise the coreset reference comparison would
        # accidentally carry that override forward.
        missing_resources = resource and all('model__' + resource not in parameters
                                             for parameters in baseline.values())
        if resource and stage_pipelines is None:
            for stage in rows_by_stage:
                baseline[stage].setdefault('model__' + resource, caps[stage])
        baseline = complete_stage_parameters(baseline, originals)
        defaults = []
        if resource and 'model__' + resource not in rules and missing_resources and stage_pipelines is None:
            key = 'model__' + resource
            rules[key] = {'kind': 'stages', **caps}
            defaults.append(key)
        stage_parameters = {}
        for stage, rows in rows_by_stage.items():
            if resource:
                baseline[stage].setdefault('model__' + resource,
                                           originals[stage]['model'].get_params()[resource])
            # Validate the original plan independently: a revision must not conceal
            # an invalid or unknown baseline that the prompt claimed was valid.
            baseline_pipeline = clone(originals[stage]).set_params(**baseline[stage])
            validate_estimators(baseline_pipeline)
            _validate_stage(baseline_pipeline['model'], {}, rows, stage, resource, caps.get(stage))
            updates = {**baseline[stage], **{
                key: _rule_value(rule, key[7:], stage, rows, n_core)
                for key, rule in sorted(rules.items())}}
            _validate_stage(originals[stage]['model'], {key: value for key, value in updates.items()
                                    if key.startswith('model__')}, rows, stage, resource, caps.get(stage))
            stage_parameters[stage] = updates
        frozen.append({
            'method_id': method_id,
            'model_class': type(model).__name__,
            'reason': item['reason'],
            'rules': rules,
            'budget_default_parameters': defaults,
            'stage_baseline': baseline,
            'stage_parameters': stage_parameters,
        })
    return {
        'schema_version': 1,
        'anchor_rows': n_core,
        'training_rows': rows_by_stage,
        'methods': frozen,
    }
