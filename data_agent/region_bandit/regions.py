"""Bounded, reproducible region discovery using only frozen training evidence."""
from itertools import combinations
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .diagnostics import common_losses, regional_contrast, problem_type

DEFAULTS = {
    'diagnostic_max_rows': 100000, 'discover_fraction': .5, 'max_features': 8,
    'numeric_quantiles': [.2, .4, .6, .8], 'max_basic_rules': 90,
    'max_conjunction_rules': 30, 'max_rule_conditions': 2, 'min_rows_per_half': 30,
    'min_expected_coreset_rows': 5, 'max_region_fraction': .9,
    'max_behavior_regions': 4, 'max_single_pair_regions': 0, 'max_coverage_regions': 2, 'jaccard_dedup_threshold': .8,
    'numeric_tolerance': 1e-12,
}


def _scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError('Rules and row IDs must contain finite JSON scalar values')


def _canonical(rule):
    return json.dumps(rule, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def rule_id(rule):
    return hashlib.sha256(_canonical(rule).encode()).hexdigest()[:20]


def validate_rule(rule):
    if not isinstance(rule, dict):
        raise ValueError('Region rule must be an AST object')
    op = rule.get('op')
    keys = {'numeric_le': {'op', 'feature', 'threshold'}, 'numeric_gt': {'op', 'feature', 'threshold'},
            'category_in': {'op', 'feature', 'values'}, 'is_missing': {'op', 'feature'},
            'gmm_component_in': {'op', 'component_ids'}, 'AND': {'op', 'args'}}
    if op not in keys or set(rule) != keys[op]:
        raise ValueError('Unknown rule operation or unexpected AST fields')
    if 'feature' in rule and (not isinstance(rule['feature'], str) or not rule['feature']):
        raise ValueError('Rule feature must be a nonempty column name')
    if op.startswith('numeric_'):
        value = rule['threshold']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError('Numeric threshold must be finite')
    elif op == 'category_in':
        values = rule['values']
        if not isinstance(values, list) or not values or any(value is None for value in values):
            raise ValueError('Category rules need nonmissing JSON values')
        for value in values:
            _scalar(value)
    elif op == 'gmm_component_in':
        values = rule['component_ids']
        if (not isinstance(values, list) or not values or any(isinstance(v, bool)
                or not isinstance(v, int) or v < 0 for v in values)):
            raise ValueError('GMM components must be nonnegative integers')
    elif op == 'AND':
        args = rule['args']
        if not isinstance(args, list) or len(args) != 2 or any(not isinstance(a, dict) or a.get('op') == 'AND' for a in args):
            raise ValueError('Exactly two basic conditions are allowed in an AND rule')
        for arg in args:
            validate_rule(arg)
    return rule


def apply_rule(rule, frame, components):
    """Interpret a validated frozen AST; never evaluate code or refit thresholds."""
    validate_rule(rule)
    op = rule['op']
    if op == 'AND':
        return apply_rule(rule['args'][0], frame, components) & apply_rule(rule['args'][1], frame, components)
    if op == 'gmm_component_in':
        components = np.asarray(components)
        if components.shape != (len(frame),):
            raise ValueError('GMM labels must match the supplied rows')
        return np.isin(components, rule['component_ids'])
    feature = rule['feature']
    if feature not in frame.columns:
        raise ValueError(f'Frozen region column is missing: {feature}')
    values = frame[feature]
    # Match fit/predict preprocessing without converting textual categories.
    if pd.api.types.is_numeric_dtype(values.dtype):
        values = values.replace([np.inf, -np.inf], np.nan)
    if op == 'is_missing':
        return values.isna().to_numpy(dtype=bool)
    if op == 'category_in':
        return (values.notna() & values.isin(rule['values'])).to_numpy(dtype=bool)
    values = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    compare = values <= rule['threshold'] if op == 'numeric_le' else values > rule['threshold']
    return np.isfinite(values) & compare


def _config(config):
    values = config or {}
    values = values.get('region_bandit', values)
    values = values.get('regions', values)
    cfg = {**DEFAULTS, **values}
    upper = {'diagnostic_max_rows': 100000, 'max_features': 8, 'max_basic_rules': 90,
             'max_conjunction_rules': 30, 'max_rule_conditions': 2,
             'max_behavior_regions': 4, 'max_single_pair_regions': 2, 'max_coverage_regions': 2}
    for key, bound in upper.items():
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value <= bound:
            raise ValueError(f'{key} must be an integer in [0, {bound}]')
    if cfg['max_behavior_regions'] + cfg['max_single_pair_regions'] > 4:
        raise ValueError('At most four model-difference regions are supported')
    for key in ('min_rows_per_half', 'min_expected_coreset_rows'):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, np.integer)) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['diagnostic_max_rows'] < 2 or cfg['max_rule_conditions'] != 2 or cfg['discover_fraction'] != .5:
        raise ValueError('Use at least two diagnostic rows, two-condition rules, and equal discovery/check halves')
    for key in ('max_region_fraction', 'jaccard_dedup_threshold'):
        if not 0 < cfg[key] < 1:
            raise ValueError(f'{key} must lie strictly between zero and one')
    if not np.isfinite(cfg['numeric_tolerance']) or cfg['numeric_tolerance'] <= 0:
        raise ValueError('numeric_tolerance must be positive')
    quantiles = cfg['numeric_quantiles']
    if not isinstance(quantiles, (list, tuple)) or any(not 0 < q < 1 for q in quantiles):
        raise ValueError('numeric_quantiles must lie strictly inside (0, 1)')
    return cfg


def _excluded(task):
    from ..coreset import TASK_ID_COLUMNS
    excluded = {task.get('target', task.get('target_column'))}
    excluded.update(TASK_ID_COLUMNS.get(task.get('name'), ()))
    keys = ('id_column', 'id_columns', 'row_id_column', 'row_id_columns', 'leakage_columns',
            'exclude_columns', 'drop_columns', 'alignment_key')
    for source in (task, task.get('metadata', {}), task.get('coreset', {}), task.get('split_manifest', {})):
        if isinstance(source, dict):
            for key in keys:
                values = source.get(key) or []
                excluded.update([values] if isinstance(values, str) else values)
    return excluded


def _choose_features(frame, task, importances, seed, maximum):
    eligible = [name for name in frame.columns if isinstance(name, str) and name not in _excluded(task)]
    if task.get('feature_columns') is not None:
        eligible = [name for name in eligible if name in task['feature_columns']]
    # Only explicitly mapped original-column names are accepted. In particular,
    # transformed names are never stripped or guessed back to their origin.
    sets = [importances] if isinstance(importances, dict) else list(importances or [])
    rankings = {}
    accepted_sources = 0
    for values in sets:
        if not isinstance(values, dict):
            continue
        available = [(key, abs(float(value))) for key, value in values.items() if key in eligible
                     and isinstance(value, (int, float, np.number)) and np.isfinite(value)]
        if not available:
            continue
        accepted_sources += 1
        ranks = rankdata([value for _, value in available], method='average') / len(available)
        for (key, _), rank in zip(available, ranks):
            rankings[key] = rankings.get(key, 0.) + float(rank)
    ranked = sorted(rankings, key=lambda key: (-rankings[key], key))[:maximum]
    remaining = sorted(set(eligible) - set(ranked))
    rng = np.random.default_rng(seed)
    if remaining:
        remaining = [remaining[index] for index in rng.permutation(len(remaining))]
    fallback = remaining[:max(0, maximum - len(ranked))]
    return ranked + fallback, {'importance_sources_used': accepted_sources, 'importance_selected': ranked,
                               'fixed_seed_fallback_fields': fallback, 'excluded_fields': sorted(str(x) for x in _excluded(task) if x is not None)}


def _basic_rules(frame, components, fields, task, cfg):
    rules = []
    categorical = set(task.get('categorical_columns', []))
    for name in fields:
        column = frame[name]
        if pd.api.types.is_numeric_dtype(column.dtype):
            column = column.replace([np.inf, -np.inf], np.nan)
        if pd.api.types.is_numeric_dtype(column) and name not in categorical and not pd.api.types.is_bool_dtype(column):
            values = pd.to_numeric(column, errors='coerce').to_numpy(dtype=float, na_value=np.nan)
            values = values[np.isfinite(values)]
            if len(values):
                for threshold in np.unique(np.quantile(values, cfg['numeric_quantiles'])):
                    for op in ('numeric_le', 'numeric_gt'):
                        rules.append({'op': op, 'feature': name, 'threshold': float(threshold)})
        else:
            counts = column.dropna().value_counts()
            ordered = sorted(((_scalar(value), int(count)) for value, count in counts.items()),
                             key=lambda item: (-item[1], _canonical(item[0])))
            for value, _ in ordered[:4]:
                rules.append({'op': 'category_in', 'feature': name, 'values': [value]})
            # Bounded serialized tail group, selected only on discovery rows.
            rare = [value for value, count in ordered if count / max(1, len(column)) <= .05]
            if rare:
                rare = sorted(rare, key=_canonical)[:128]
                rules.append({'op': 'category_in', 'feature': name, 'values': rare})
        if column.isna().any():
            rules.append({'op': 'is_missing', 'feature': name})
    for component in np.unique(components):
        rules.append({'op': 'gmm_component_in', 'component_ids': [int(component)]})
    unique = {}
    for rule in rules:
        unique.setdefault(rule_id(rule), rule)
    return list(unique.values())[:cfg['max_basic_rules']]


def _conjunction_rules(ranked_basic, frame, components, losses, maximum):
    """Rank every top-16 pair on discovery rows before spending the rule budget."""
    audit = {'scope': 'discovery half only', 'pairs_evaluated': 0,
             'empty_or_full_removed': 0, 'equivalent_removed': 0, 'unique_rule_count': 0, 'shortlist': []}
    if not maximum:
        return [], audit
    masks = {rule_id(rule): apply_rule(rule, frame, components) for rule in ranked_basic}
    # A conjunction identical to a basic rule adds no new discovery evidence.
    seen = {np.packbits(mask).tobytes() for mask in masks.values()}
    scored = []
    for first, second in combinations(ranked_basic[:16], 2):
        audit['pairs_evaluated'] += 1
        mask = masks[rule_id(first)] & masks[rule_id(second)]
        if not mask.any() or mask.all():
            audit['empty_or_full_removed'] += 1
            continue
        rule = {'op': 'AND', 'args': sorted([first, second], key=rule_id)}
        score = regional_contrast(losses, mask)['score']
        scored.append((score, rule_id(rule), rule, np.packbits(mask).tobytes()))
    unique = []
    for score, identifier, rule, signature in sorted(scored, key=lambda item: (-item[0], item[1])):
        if signature in seen:
            audit['equivalent_removed'] += 1
            continue
        seen.add(signature)
        unique.append((score, identifier, rule))
    audit['unique_rule_count'] = len(unique)
    audit['shortlist'] = [{'id': identifier, 'discover_score': score}
                          for score, identifier, _ in unique[:maximum]]
    return [rule for _, _, rule in unique[:maximum]], audit


def build_regions(frame, task, proxy_predictions, reference_indices, components, density, hardness,
                  *, config=None, seed=2026, feature_importances=None, row_ids=None,
                  candidate_rules=None, evidence_indices=None):
    """Select behavior, strongest-pair and coverage regions with exact Q masks.

    Initial rule thresholds are frozen on the discovery half. An expansion
    rescores that rule pool on held-out Q rows using its current method bank.
    Regional population and coreset coverage use all Q and reference rows.
    """
    cfg = _config(config)
    n = len(frame)
    if n < 2 or not frame.columns.is_unique:
        raise ValueError('Region discovery requires at least two rows and unique column names')
    components = np.asarray(components)
    density, hardness = np.asarray(density, dtype=float), np.asarray(hardness, dtype=float)
    if (components.shape != (n,) or components.dtype.kind not in 'iu' or np.any(components < 0)
            or density.shape != (n,) or hardness.shape != (n,) or not np.isfinite(density).all()
            or not np.isfinite(hardness).all() or np.any((density < 0) | (density > 1))
            or np.any((hardness < 0) | (hardness > 1))):
        raise ValueError('Aligned GMM labels and bounded density/hardness ranks are required')
    reference = np.asarray(reference_indices)
    if (reference.ndim != 1 or reference.dtype.kind not in 'iu' or not len(reference)
            or len(np.unique(reference)) != len(reference) or np.any(reference < 0) or np.any(reference >= n)):
        raise ValueError('Reference coreset must contain unique valid Q positions')
    if row_ids is None:
        row_ids = np.arange(n)
    if len(row_ids) != n:
        raise ValueError('Diagnostic row IDs must align with Q')
    row_ids = [_scalar(value) for value in row_ids]
    if len(set(_canonical(value) for value in row_ids)) != n:
        raise ValueError('Diagnostic row IDs must be unique')
    target = task.get('target', task.get('target_column'))
    classes = task.get('classes')
    evidence = np.arange(n) if evidence_indices is None else np.asarray(evidence_indices)
    if (evidence.ndim != 1 or evidence.dtype.kind not in 'iu' or len(evidence) < 2
            or len(np.unique(evidence)) != len(evidence) or np.any((evidence < 0) | (evidence >= n))):
        raise ValueError('Expected at least two distinct diagnostic rows in Q')
    if evidence_indices is not None and np.intersect1d(evidence, reference).size:
        raise ValueError('Node diagnostic rows must exclude the coreset used for training')
    count = len(proxy_predictions)
    pairs = tuple(combinations(range(count), 2))
    heldout = []
    for result in proxy_predictions:
        selected = dict(result)
        for key in ('predictions', 'probabilities'):
            if result.get(key) is None:
                continue
            values = np.asarray(result[key])
            if key == 'probabilities' and not values.size:
                selected[key] = None
            elif values.ndim == 0 or len(values) != n:
                raise ValueError('Model predictions must align with the complete Q population')
            elif evidence_indices is not None:
                selected[key] = values[evidence]
        heldout.append(selected)
    target_values = frame[target].to_numpy()[evidence]
    held_losses, scale = common_losses(task, target_values, heldout, classes,
                                       method_count=5 if candidate_rules is None else count)
    fallback = False
    if problem_type(task) == 'regression':
        predictions = np.asarray([p['predictions'] for p in heldout], dtype=float)
        fallback = bool(np.quantile(np.abs(predictions-target_values), .9) == 0)
        disagreement = np.minimum(np.std(predictions, axis=0) / scale, 1.)
    else:
        disagreement = np.std(held_losses, axis=0)
    rng = np.random.default_rng(seed)
    diagnostic = rng.choice(len(evidence), size=min(len(evidence), cfg['diagnostic_max_rows']), replace=False)
    midpoint = len(diagnostic) // 2
    discover, check = evidence[diagnostic[:midpoint]], evidence[diagnostic[midpoint:]]
    discover_losses, check_losses = held_losses[:, diagnostic[:midpoint]], held_losses[:, diagnostic[midpoint:]]
    if candidate_rules is None:
        fields, feature_selection = _choose_features(frame, task, feature_importances, seed, cfg['max_features'])
        basic = _basic_rules(frame.iloc[discover], components[discover], fields, task, cfg)
        ranked_basic = sorted(basic, key=lambda rule: (
            -regional_contrast(discover_losses, apply_rule(rule, frame.iloc[discover], components[discover]))['score'], rule_id(rule)))
        conjunctions, conjunction_audit = _conjunction_rules(
            ranked_basic, frame.iloc[discover], components[discover], discover_losses,
            cfg['max_conjunction_rules'])
    else:
        fields, feature_selection = [], 'frozen_initial_rules'
        basic = [r for r in candidate_rules if r['op'] != 'AND']
        conjunctions = [r for r in candidate_rules if r['op'] == 'AND']
        conjunction_audit = {'scope': 'frozen_initial_conjunctions'}
    candidates, rejected = [], []
    for rule in basic + conjunctions:
        mask = apply_rule(rule, frame, components)
        q_count, ref_count = int(mask.sum()), int(mask[reference].sum())
        fraction, ref_fraction = q_count / n, ref_count / len(reference)
        first = regional_contrast(discover_losses, mask[discover])
        second = regional_contrast(check_losses, mask[check])
        support = (all(part['inside_count'] >= cfg['min_rows_per_half'] and part['outside_count'] >= cfg['min_rows_per_half']
                       and part['fraction'] <= cfg['max_region_fraction'] for part in (first, second))
                   and fraction <= cfg['max_region_fraction']
                   and fraction * len(reference) >= cfg['min_expected_coreset_rows'])
        diffs = np.asarray(first['pair_contrasts'])
        pair_index = int(np.argmax(np.abs(diffs)))
        primary = list(pairs[pair_index]) if abs(diffs[pair_index]) > cfg['numeric_tolerance'] else None
        direction = bool(primary is not None and diffs[pair_index] * second['pair_contrasts'][pair_index] > 0
                         and abs(second['pair_contrasts'][pair_index]) > cfg['numeric_tolerance'])
        stable = float(min(first['score'], second['score']))
        gap = max(0., fraction - ref_fraction)
        behavior = support and direction and stable > cfg['numeric_tolerance']
        coverage = support and gap > cfg['numeric_tolerance']
        single_pair = None
        pair_score = 0.
        if cfg['max_single_pair_regions'] and support:
            for i, (a, b) in enumerate(zip(diffs, second['pair_contrasts'])):
                if a * b <= 0 or min(abs(a), abs(b)) <= cfg['numeric_tolerance']:
                    continue
                score = min(part['fraction'] * (1 - part['fraction']) * contrast ** 2
                            for part, contrast in ((first, a), (second, b)))
                if score > max(pair_score, cfg['numeric_tolerance']):
                    single_pair, pair_score = list(pairs[i]), float(score)
        if not behavior and not coverage and single_pair is None:
            rejected.append({'id': rule_id(rule), 'support': bool(support), 'direction_consistent': direction,
                             'stable_score': stable, 'coverage_gap': gap})
            continue
        stats = {'discover': first, 'check': second, 'stable_score': stable,
                 'discover_score': first['score'], 'check_score': second['score'],
                 'direction_consistent': direction, 'coverage_gap': gap,
                 'mean_density': float(density[mask].mean()), 'mean_hardness': float(hardness[mask].mean()),
                 'mean_disagreement': float(disagreement[mask[evidence]].mean()),
                 'disagreement_definition': 'prediction_std_div_evidence_loss_scale_clipped' if problem_type(task) == 'regression' else 'bounded_probability_loss_std',
                 'expected_coreset_count': fraction * len(reference)}
        stats['full_q' if evidence_indices is None else 'heldout_q'] = regional_contrast(held_losses, mask[evidence])
        candidates.append({'id': rule_id(rule), 'rule': rule, 'mask': mask, 'indices': np.flatnonzero(mask).tolist(),
                           'q_count': q_count, 'q_fraction': fraction, 'reference_count': ref_count,
                           'reference_fraction': ref_fraction, 'statistics': stats, 'primary_pair': primary,
                           'strongest_stable_pair': single_pair, 'single_pair_score': pair_score,
                           'behavior_eligible': bool(behavior), 'coverage_eligible': bool(coverage)})
    # Dedup before ranking. Simpler rules win overlap conflicts, then stronger
    # replicated behavior evidence, then larger exact coverage gap, then hash.
    preference = lambda r: (2 if r['rule']['op'] == 'AND' else 1, -r['statistics']['stable_score'],
                            -r['statistics']['coverage_gap'], r['id'])
    unique, duplicates = [], []
    for region in sorted(candidates, key=preference):
        duplicate = None
        for prior in unique:
            union = int(np.count_nonzero(region['mask'] | prior['mask']))
            jaccard = int(np.count_nonzero(region['mask'] & prior['mask'])) / union
            if jaccard > cfg['jaccard_dedup_threshold']:
                duplicate = {'id': region['id'], 'kept_id': prior['id'], 'jaccard': jaccard}
                break
        if duplicate:
            duplicates.append(duplicate)
        else:
            unique.append(region)
    behavior = sorted([r for r in unique if r['behavior_eligible']], key=lambda r: (-r['statistics']['stable_score'], r['id']))[:cfg['max_behavior_regions']]
    chosen_ids = {r['id'] for r in behavior}
    single_pairs = sorted((r for r in unique if r['strongest_stable_pair'] and r['id'] not in chosen_ids),
                          key=lambda r: (-r['single_pair_score'], r['id']))[:cfg['max_single_pair_regions']]
    for region in single_pairs:
        region['discovery_kind'] = 'single_pair'
        region['primary_pair'] = region['strongest_stable_pair']
        region['statistics']['direction_consistent'] = True
        chosen_ids.add(region['id'])
    coverage = sorted([r for r in unique if r['coverage_eligible'] and r['id'] not in chosen_ids],
                      key=lambda r: (-r['statistics']['coverage_gap'], r['id']))[:cfg['max_coverage_regions']]
    for region in behavior:
        region['discovery_kind'] = 'behavior'
    for region in coverage:
        region['discovery_kind'] = 'coverage'
    selected = behavior + single_pairs + coverage
    return {'regions': selected, 'loss_scale': scale,
            'discovery': {'scope': 'Q only; no V_core, external validation, or test evidence',
                          'seed': int(seed), 'discover_q_positions': discover.tolist(), 'check_q_positions': check.tolist(),
                          'discover_row_ids': [row_ids[i] for i in discover], 'check_row_ids': [row_ids[i] for i in check],
                          'loss_scale': scale, 'loss_scale_zero_fallback': fallback,
                          'loss_scale_source': 'diagnostic population pooled absolute-error p90' if problem_type(task) == 'regression' else 'classification bounded probability distance',
                          'fields': fields, 'feature_selection': feature_selection,
                          'basic_rule_count': len(basic), 'conjunction_rule_count': len(conjunctions),
                          'conjunction_screening': conjunction_audit,
                          'candidate_rules': basic + conjunctions, 'candidate_rule_count': len(basic) + len(conjunctions), 'eligible_rule_count': len(candidates),
                          'selected_region_count': len(selected), 'rejected': rejected,
                          'jaccard_removed': duplicates, 'config': cfg}}
