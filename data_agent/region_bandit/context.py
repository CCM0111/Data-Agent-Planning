"""Versioned, training-only contextual features for region actions."""
from __future__ import annotations

import hashlib
import json
import math

from data_agent.stage1_metrics import _metric, _parameters

FEATURE_SCHEMA_VERSION = 'region_features_v1'
METRICS = ('accuracy', 'f1', 'auc', 'log_loss', 'mae', 'mse', 'rmse', 'r2')
ACTION_TYPES = ('baseline', 'cover', 'focus')
RULE_TYPES = ('numeric_le', 'numeric_gt', 'category_in', 'is_missing', 'gmm_component_in', 'AND')
TASK_STATS = ('log_rows', 'log_features', 'categorical_fraction', 'missing_fraction',
              'p_fraction', 'coreset_fraction', 'target_imbalance', 'target_tail_fraction',
              'proxy_disagreement')
REGION_STATS = ('q_fraction', 'c0_fraction', 'mean_density', 'mean_hardness',
                'mean_disagreement', 'discover_score', 'check_score',
                'direction_consistent', 'coverage_gap')
INTERVENTIONS = ('actual_increase_fraction', 'group_moved_fraction', 'class_moved_fraction')


def admission_evidence(method_ids, scores, slots, direction, source):
    """Describe the provisional cutoff only when every competing score is known."""
    from .reward import rank_method_ids
    ranked = rank_method_ids(scores, direction=direction) if scores else []
    boundary = None
    if set(method_ids) == set(scores) and 0 < slots < len(ranked):
        inside, outside = ranked[slots-1:slots+1]
        boundary = dict(method_ids=[inside, outside], score_gap=abs(scores[inside]-scores[outside]))
    return dict(source=source, direction=direction, slots=slots, candidate_ids=method_ids,
                observed_ranking=[dict(method_id=i, score=scores[i]) for i in ranked], boundary=boundary)


def _pair_loss_evidence(region, admission):
    """Describe the actual within-region order, not the inside/outside contrast."""
    pair = region.get('primary_pair')
    if not admission or not pair or len(pair) != 2:
        return {}
    ids = region.get('primary_pair_method_ids')
    if ids is None and admission.get('source') == 'P-trained proxy scores on Q':
        ids = pair  # Initialization uses proxy bank indices as admission IDs.
    if not ids or len(ids) != 2:
        return {}
    summary = dict(method_ids=ids)
    for split in ('discover', 'check'):
        part = region.get(split, {})
        losses = part.get('inside_losses')
        if not losses or not part.get('inside_count') or not all(type(i) is int and 0 <= i < len(losses) for i in pair):
            return {}
        summary[split] = dict(n=part['inside_count'], loss_A_minus_B=losses[pair[0]] - losses[pair[1]])
    return {'pair_loss_evidence': summary}


def canonical_metric(task):
    """Use the project's registry; explicit metric options remain compatibility keys."""
    name = _metric(task)
    params = _parameters(task, name)
    name = 'auc' if name == 'roc_auc' else ('f1' if name == 'f1_macro' else name)
    if name == 'auc':
        params.setdefault('average', 'macro')
        # The shared evaluator defaults multiclass AUC to OVR. For binary AUC
        # this option does not affect the score; use the same canonical default
        # even before the saved probability class order is available.
        params.setdefault('multi_class', 'ovr')
    signature = json.dumps({'metric': name, 'params': params}, sort_keys=True, separators=(',', ':'))
    return name, params, signature


def default_feature_schema():
    names = ['constant', 'task.classification', 'task.regression']
    names += ['task.' + key for key in TASK_STATS]
    names += ['metric.' + key for key in METRICS]
    names += ['metric.average.' + key for key in ('binary', 'macro', 'weighted', 'micro', 'none')]
    names += ['metric.multi_class.' + key for key in ('raise', 'ovr', 'ovo', 'none')]
    names += ['metric.max_fpr', 'metric.max_fpr_specified', 'metric.labels_specified',
              'metric.pos_label_specified', 'metric.zero_division_zero',
              'metric.zero_division_one', 'metric.force_finite']
    names += ['models.' + key for key in ('linear', 'tree', 'other')]
    names += ['action.' + key for key in ACTION_TYPES]
    names += ['rule.' + key for key in RULE_TYPES]
    names += ['region.' + key for key in REGION_STATS]
    names += ['intervention.' + key for key in INTERVENTIONS]
    names += ['interaction.missing_rule', 'interaction.disagreement', 'interaction.tree_disagreement',
              'interaction.core_increase', 'interaction.increase_difference',
              'interaction.increase_gap', 'interaction.increase_hardness']
    names += ['interaction.metric.' + metric + '.' + action for metric in METRICS for action in ACTION_TYPES]
    names += ['api_available']
    names += ['missing.' + name for name in names if name.startswith(('task.', 'region.', 'intervention.'))
              and name not in ('task.classification', 'task.regression')]
    entries = [{'name': name, 'scaling': ('clip(log10(x)/7,0,1)' if name == 'task.log_rows' else
                'clip(log10(x)/4,0,1)' if name == 'task.log_features' else 'clip(x,0,1)'),
                'missing': 0} for name in names]
    payload = {'version': FEATURE_SCHEMA_VERSION, 'features': entries}
    payload['schema_id'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _rules(rule):
    if not isinstance(rule, dict):
        return []
    kind = rule.get('op', rule.get('type', ''))
    result = [kind]
    for child in rule.get('args', rule.get('rules', rule.get('children', rule.get('conditions', [])))):
        result.extend(_rules(child))
    return result


def build_context(task_summary, method_bank, region, action, *, api_available=True, schema=None):
    """Return a frozen feature vector; do not read evaluation or posterior fields."""
    schema = schema or default_feature_schema()
    if schema != default_feature_schema():
        raise ValueError('Unsupported or modified region feature schema')
    metric, params, metric_signature = canonical_metric(task_summary)
    values, missing = {}, {}

    def put(name, value, *, baseline_zero=False):
        parsed = _number(value)
        values[name] = 0.0 if parsed is None else min(1.0, max(0.0, parsed))
        missing['missing.' + name] = float(parsed is None and not baseline_zero)

    values['constant'] = 1.0
    for kind in ('classification', 'regression'):
        values['task.' + kind] = float(task_summary.get('problem_type') == kind)
    for key in TASK_STATS:
        value = task_summary.get(key)
        if key in ('log_rows', 'log_features'):
            raw = _number(task_summary.get('n_rows' if key == 'log_rows' else 'n_features'))
            value = math.log10(max(1.0, raw)) / (7 if key == 'log_rows' else 4) if raw is not None else None
        put('task.' + key, value)
    for name in METRICS:
        values['metric.' + name] = float(metric == name)
    for name in ('binary', 'macro', 'weighted', 'micro', 'none'):
        values['metric.average.' + name] = float(params.get('average', 'none') == name)
    for name in ('raise', 'ovr', 'ovo', 'none'):
        values['metric.multi_class.' + name] = float(params.get('multi_class', 'none') == name)
    values.update({'metric.max_fpr': float(params.get('max_fpr') or 0),
                   'metric.max_fpr_specified': float(params.get('max_fpr') is not None),
                   'metric.labels_specified': float('labels' in params),
                   'metric.pos_label_specified': float('pos_label' in params),
                   'metric.zero_division_zero': float(params.get('zero_division') == 0),
                   'metric.zero_division_one': float(params.get('zero_division') == 1),
                   'metric.force_finite': float(params.get('force_finite', True))})
    methods = method_bank.get('methods', []) if isinstance(method_bank, dict) else method_bank
    counts = {'linear': 0, 'tree': 0, 'other': 0}
    summaries = []
    for method in methods:
        family = method.get('family', 'other')
        group = 'linear' if family == 'linear' else 'tree' if family in ('tree', 'bagging', 'boosting') else 'other'
        counts[group] += 1
        summaries.append({key: method[key] for key in ('id', 'method_id', 'name', 'family', 'stage_parameters', 'idea') if key in method})
    for key, count in counts.items():
        values['models.' + key] = count / max(1, len(methods))
    action_type = action.get('type', action.get('action_type'))
    if action_type not in ACTION_TYPES:
        raise ValueError('Unknown region action type')
    baseline = action_type == 'baseline'
    region = {} if baseline else (region or {})
    if not baseline:
        region = {**region.get('statistics', {}), **region}
        region.setdefault('c0_fraction', region.get('reference_fraction'))
        region.setdefault('region_id', region.get('id'))
        if 'protected_q_indices' in action:
            # The solver budgets the editable rows. Discovery statistics still
            # describe all rows, so derive the decision gap on the solver's scope.
            region['q_fraction'] = action['region_fraction']
            count = action.get('current_parent_region_count', action['reference_region_count'])
            region['c0_fraction'] = count / action['size']
            region['coverage_gap'] = max(0., region['q_fraction'] - region['c0_fraction'])
    for name in ACTION_TYPES:
        values['action.' + name] = float(name == action_type)
    kinds = _rules(region.get('rule', region.get('rule_ast', {})))
    for name in RULE_TYPES:
        values['rule.' + name] = float(name in kinds)
    for key in REGION_STATS:
        put('region.' + key, 0 if baseline else region.get(key), baseline_zero=baseline)
    size = _number(action.get('coreset_size', action.get('size')))
    for key in INTERVENTIONS:
        raw = action.get(key)
        if key == 'actual_increase_fraction' and raw is None:
            raw = action.get('actual_region_increase_fraction')
        if raw is None and size and size > 0:
            if key == 'actual_increase_fraction':
                actual = _number(action.get('feasible_region_count'))
                original = _number(action.get('baseline_region_count', action.get('reference_region_count')))
                raw = (actual - original) / size if actual is not None and original is not None else None
            else:
                count = _number(action.get(key.replace('_fraction', '_count')))
                raw = count / size if count is not None else None
        put('intervention.' + key, 0 if baseline else raw, baseline_zero=baseline)
    increase = values['intervention.actual_increase_fraction']
    values.update({'interaction.missing_rule': values['task.missing_fraction'] * values['rule.is_missing'],
                   'interaction.disagreement': values['task.proxy_disagreement'] * values['region.mean_disagreement'],
                   'interaction.tree_disagreement': values['models.tree'] * values['region.mean_disagreement'],
                   'interaction.core_increase': values['task.coreset_fraction'] * increase,
                   'interaction.increase_difference': increase * min(values['region.discover_score'], values['region.check_score']),
                   'interaction.increase_gap': increase * values['region.coverage_gap'],
                   'interaction.increase_hardness': increase * values['region.mean_hardness'],
                   'api_available': float(bool(api_available))})
    for name in METRICS:
        for kind in ACTION_TYPES:
            values['interaction.metric.' + name + '.' + kind] = float(metric == name and action_type == kind)
    values.update(missing)
    names = [entry['name'] for entry in schema['features']]
    vector = [float(values.get(name, 0)) for name in names]
    retrieval_names = [name for name in names if name.startswith(('task.', 'metric.', 'models.', 'region.', 'rule.'))]
    return {'features': vector, 'feature_names': names, 'schema_id': schema['schema_id'],
            'feature_schema_version': schema['version'], 'metric_signature': metric_signature,
            'retrieval_features': {name: values[name] for name in retrieval_names},
            'action_id': action.get('action_id'), 'action_type': action_type,
            'rule_types': sorted(set(kinds)),
            'problem_type': task_summary['problem_type'], 'metric': metric, 'metric_params': params,
            **({'editable_sample_scope': 'Action quotas apply only to the editable 85%; the MMD and per-method error rows are protected.'} if 'protected_q_indices' in action else {}),
            'task_summary': {key: task_summary[key] for key in ('problem_type', 'metric', 'metric_params',
               'n_rows', 'n_features', *TASK_STATS) if key in task_summary},
            'method_summary': summaries,
            'budget_summary': method_bank.get('formal_budgets', {}) if isinstance(method_bank, dict) else {},
            'region_summary': {key: region[key] for key in ('region_id', 'rule', 'rule_ast', 'discovery_kind', 'primary_pair', 'primary_pair_method_ids', 'single_pair_score', *REGION_STATS) if key in region}
                | _pair_loss_evidence(region, method_bank.get('formal_budgets', {}).get('admission') if isinstance(method_bank, dict) else None),
            'action_summary': {key: action[key] for key in ('action_id', 'type', 'action_type', 'region_id',
               'requested_region_count', 'feasible_region_count', 'baseline_region_count', 'coreset_size',
               'size', 'variant', 'full_cover_target', 'reference_region_count', 'actual_region_increase_fraction',
               'quota_table', 'group_moved_count', 'class_moved_count', 'solver_status',
               'group_values', 'reference_group_counts', 'group_bounds', 'class_bounds', 'group_quota_move_budget',
               'sampling_mode', 'deletion_rule', 'max_donor_remove_fraction', 'allocation', 'agent_adjustment_fraction', 'positive_increase_reference', 'repeat_effects', 'max_replaced_rows') if key in action}}


def pair_at_boundary(context):
    boundary = (context.get('budget_summary', {}).get('admission') or {}).get('boundary') or {}
    pair = (context.get('region_summary') or {}).get('primary_pair_method_ids')
    ids = boundary.get('method_ids')
    return set(map(str, pair)) == set(map(str, ids)) if pair and ids and len(pair) == len(ids) == 2 else None
