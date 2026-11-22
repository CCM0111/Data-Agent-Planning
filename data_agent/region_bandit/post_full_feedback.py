"""Sample-outside-training diagnostics for every predeclared region.

These observations describe fitted model behavior; they are not counterfactual
labels for unexecuted region actions and must not enter the original decision.
"""
import numpy as np
from itertools import combinations

from .diagnostics import PAIRS, common_losses, regional_contrast, paired_region_bootstrap
from .regions import apply_rule


def _dependence(task, dependence):
    from ..stage1_selection import _dependency_evidence
    evidence = _dependency_evidence(task, dependence if isinstance(dependence, dict) else {})
    if dependence not in (None, False, '', [], {}):
        if isinstance(dependence, (str, bool)):
            if str(dependence).lower() not in {'iid', 'independent', 'independent_rows', 'row', 'rows', 'false'}:
                evidence.append({'field': 'dependence', 'value': dependence})
        elif isinstance(dependence, (list, tuple)):
            evidence.extend(dependence)
    return evidence


def _ci_direction(interval, atol):
    return 1 if interval[0] > atol else (-1 if interval[1] < -atol else 0)


def compute_region_feedback(regions, frame, task, core_predictions, full_predictions, components,
                            *, loss_scale, config=None, seed=2026, dependence=None, method_ids=None):
    """Diagnose frozen rules on the same external-validation rows, without fit.

    Inputs must be aligned to ``frame``. GMM component labels must be obtained
    using the original frozen GMM/representation; this function never fits it.
    The caller adds executed action, sampling seed, and configuration identities
    to the returned record. Bootstrap CIs describe fixed-model sampling noise.
    """
    config = config or {}
    config = config.get('region_bandit', config)
    cfg = {'min_region_rows': 30, 'bootstrap_repeats': 300, **config.get('post_full_feedback', config)}
    for key in ('min_region_rows', 'bootstrap_repeats'):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, np.integer)) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if loss_scale is None or isinstance(loss_scale, bool) or not np.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError('A positive loss scale frozen from Q is required')
    method_ids = list(range(5)) if method_ids is None else list(method_ids)
    if (len(method_ids) not in (3, 5) or len(set(method_ids)) != len(method_ids)
            or any(type(m) is not int or m not in range(5) for m in method_ids)):
        raise ValueError('Feedback needs three or five distinct original method IDs')
    if len(core_predictions) != len(method_ids) or len(full_predictions) != len(method_ids):
        raise ValueError('Prediction banks must align with observed method IDs')
    order = sorted(range(len(method_ids)), key=lambda i: method_ids[i])
    core_predictions = [core_predictions[i] for i in order]
    full_predictions = [full_predictions[i] for i in order]
    method_ids = [method_ids[i] for i in order]
    observed_pairs = tuple(combinations(method_ids, 2))
    target = task.get('target', task.get('target_column'))
    classes = task.get('classes')
    if classes is None:
        classes = full_predictions[0].get('classes')
    core, used_scale = common_losses(task, frame[target].to_numpy(), core_predictions, classes, loss_scale,
                                    method_count=len(method_ids))
    full, _ = common_losses(task, frame[target].to_numpy(), full_predictions, classes, loss_scale,
                            method_count=len(method_ids))
    dependence_evidence = _dependence(task, dependence)
    results = []
    for index, region in enumerate(regions):
        mask = apply_rule(region['rule'], frame, components)
        core_stats, full_stats = regional_contrast(core, mask), regional_contrast(full, mask)
        pair = region.get('primary_pair')
        if pair is not None and tuple(pair) not in PAIRS:
            raise ValueError('Frozen primary_pair must identify one of the five-method pairs')
        record = {'region_id': region['id'], 'rule': region['rule'], 'primary_pair': pair,
                  'observed_method_ids': method_ids,
                  'inside_count': int(mask.sum()), 'outside_count': int((~mask).sum()),
                  'core': core_stats, 'full': full_stats, 'loss_scale': used_scale,
                  'evidence_kind': 'core_full_external_regional_diagnostic',
                  'counterfactual_sampling_gain': None, 'bootstrap': None,
                  'primary_direction_reversed': None, 'q_to_full_direction_reversed': None}
        if min(record['inside_count'], record['outside_count']) < cfg['min_region_rows']:
            record['status'] = 'insufficient_support'
        elif dependence_evidence:
            record['status'] = 'unsupported_dependence'
            record['dependence_evidence'] = dependence_evidence
        else:
            # Stable per-region hash seed avoids order-dependent diagnostics.
            import hashlib
            region_seed = (int(seed) + int(hashlib.sha256(str(region['id']).encode()).hexdigest()[:8], 16)) % (2 ** 63)
            intervals = paired_region_bootstrap(core, full, mask, repeats=cfg['bootstrap_repeats'], seed=region_seed)
            record['bootstrap'] = intervals
            record['status'] = 'not_confirmed'
            if pair is not None and tuple(pair) not in observed_pairs:
                record['status'] = 'primary_pair_unobserved'
            if pair is not None and tuple(pair) in observed_pairs:
                pair_index = observed_pairs.index(tuple(pair))
                core_direction = _ci_direction(intervals['core_ci'][pair_index], 1e-12)
                full_direction = _ci_direction(intervals['full_ci'][pair_index], 1e-12)
                if full_direction and core_direction:
                    record['status'] = 'consistent_direction' if core_direction == full_direction else 'reversed_direction'
                elif full_direction:
                    record['status'] = 'full_signal_only'
                core_value = core_stats['pair_contrasts'][pair_index]
                full_value = full_stats['pair_contrasts'][pair_index]
                record['primary_direction_reversed'] = bool(core_value * full_value < 0)
                discovery_contrasts = region.get('statistics', {}).get('discover', {}).get('pair_contrasts', [])
                if len(discovery_contrasts) == len(PAIRS):
                    record['q_to_full_direction_reversed'] = bool(discovery_contrasts[PAIRS.index(tuple(pair))] * full_value < 0)
        record['pairs'] = [
            {'pair': list(pair_indices), 'core_contrast': core_stats['pair_contrasts'][j],
             'full_contrast': full_stats['pair_contrasts'][j],
             'core_minus_full': core_stats['pair_contrasts'][j] - full_stats['pair_contrasts'][j],
             **({key: record['bootstrap'][key][j] for key in ('core_ci', 'full_ci', 'core_minus_full_ci')}
                if record['bootstrap'] else {})}
            for j, pair_indices in enumerate(observed_pairs)]
        results.append(record)
    return {'regions': results, 'loss_scale': used_scale,
            'observed_method_ids': method_ids, 'observed_pair_count': len(observed_pairs),
            'evaluation_scope': 'external_validation_only',
            'region_definition_source': 'predecision frozen Q discovery',
            'bootstrap': {'repeats': int(cfg['bootstrap_repeats']), 'seed': int(seed), 'paired': True,
                          'stratification': 'inside/outside each frozen region',
                          'simultaneous_guarantee': False, 'extra_model_fits': 0},
            'warning': 'Regional contrasts are descriptive fitted-model diagnostics, not gains from unexecuted sampling actions'}
