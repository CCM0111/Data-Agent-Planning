"""Probability sampling under frozen regional counts and shared group coverage."""
from __future__ import annotations

import hashlib

import numpy as np

from ..marginal_coverage import GroupCoverageState, prepare_coverage_graph
from .action_planner import SAMPLER_VERSION, canonical_sampler_config


def sample_from_action(action, reference_indices, density, hardness, graph, groups,
                       region_mask=None, *, seed, config=None, decision_seed=2026):
    """Return Q positions in draw order and an execution audit.

    A0 returns this seed's original reference selection, untouched. For regional
    actions the cell schedule uses decision_seed, identical across repeats; only
    within-cell sampling uses seed. Coverage is shared by both cells of each
    original group, while different original groups remain independent.
    """
    if 'protected_q_indices' in action:
        from ..guided_coreset import coordinates
        action=dict(action);fixed=np.asarray(action.pop('protected_q_indices'),int)
        pool,ref=coordinates(reference_indices,groups,fixed)
        mutable,audit=sample_from_action(action,ref,np.asarray(density)[pool],np.asarray(hardness)[pool],
            graph.tocsr()[pool][:,pool],np.asarray(groups)[pool],None if region_mask is None else region_mask[pool],
            seed=seed,config=config,decision_seed=decision_seed)
        order=np.r_[fixed,pool[mutable]]
        if len(np.unique(order))!=len(reference_indices):raise ValueError('Protected sample overlap')
        return order,{**audit,'protected_q_indices':fixed.tolist(),'final_size':len(order),'selection_order_q_indices':order.tolist()}
    if action.get('variant') == 'gentle_cover':
        from .local_edit import sample_local_action, reference_hash
        reference_indices = np.sort(reference_indices)
        # Paired validation can redraw on the same frozen parent with a new seed.
        if str(seed) not in action['per_seed_plans']:
            parent_hash = reference_hash(reference_indices)
            plan = next((p for p in action['per_seed_plans'].values() if p['reference_hash'] == parent_hash), None)
            if plan is None:
                raise ValueError('Gentle cover replay requires the frozen parent sample')
            action = {**action, 'per_seed_plans': {str(seed): plan}}
        return sample_local_action(action, reference_indices, density, hardness, graph, groups,
                                   region_mask, seed=seed, config=config, decision_seed=decision_seed)
    d, h = np.asarray(density, float), np.asarray(hardness, float)
    original_groups = np.asarray(groups)
    reference = np.asarray(reference_indices)
    if original_groups.ndim != 1 or original_groups.dtype.kind not in 'iu' or np.any(original_groups < 0):
        raise ValueError('Invalid original groups')
    groups = original_groups.astype(int, copy=False)
    if reference.ndim != 1 or reference.dtype.kind not in 'iu' or np.any(reference < 0) or np.any(reference >= len(groups)):
        raise ValueError('Invalid reference coreset indices')
    reference = reference.astype(int, copy=False)
    if len(np.unique(reference)) != len(reference) or not len(reference) or action.get('size', len(reference)) != len(reference):
        raise ValueError('Invalid reference coreset size or uniqueness')
    if d.shape != groups.shape or h.shape != d.shape or not np.isfinite([d, h]).all() or np.any((d < 0)|(d > 1)|(h < 0)|(h > 1)):
        raise ValueError('Invalid percentiles')
    coefficients = canonical_sampler_config(action.get('sampler_config', config))
    if config is not None and canonical_sampler_config(config) != coefficients:
        raise ValueError('Sampler configuration differs from frozen action')
    if action.get('sampler_version', SAMPLER_VERSION) != SAMPLER_VERSION:
        raise ValueError('Incompatible regional sampler version')
    if action.get('type') == 'baseline':
        return reference.copy(), {'action_id': action['action_id'], 'mechanism': 'reference_reuse',
                                  'sampling_seed': seed, 'final_size': len(reference),
                                  'selection_order_q_indices': reference.tolist(),
                                  'sampler_version': SAMPLER_VERSION, 'sampler_config': coefficients}
    if action.get('type') not in ('cover', 'focus', 'resample_control'):
        raise ValueError('Unsupported regional action')
    mask = np.asarray(region_mask)
    if mask.shape != groups.shape or mask.dtype.kind != 'b':
        raise ValueError('Frozen region mask must cover the complete Q')
    mask_hash = hashlib.sha256(np.packbits(mask).tobytes()+str(len(mask)).encode()).hexdigest()
    if action.get('region_mask_hash', mask_hash) != mask_hash:
        raise ValueError('Region mask differs from frozen action')
    if action.get('quota_columns') != ['outside', 'inside']:
        raise ValueError('Unrecognized quota column order')
    values = np.asarray(action['group_values'])
    if values.dtype.kind not in 'iu' or not np.array_equal(values, np.unique(groups)):
        raise ValueError('Original group schema differs from frozen action')
    raw = np.asarray(action['quota_table'])
    if raw.dtype.kind not in 'iu' or raw.shape != (len(values), 2) or np.any(raw < 0) or raw.sum() != len(reference):
        raise ValueError('Invalid frozen quota table')
    table = raw.astype(int, copy=True)
    ids_by_group = [np.flatnonzero(groups == value) for value in values]
    capacities = np.array([[int((~mask[ids]).sum()), int(mask[ids].sum())] for ids in ids_by_group])
    if np.any(table > capacities) or ('cell_capacities' in action and not np.array_equal(capacities, action['cell_capacities'])):
        raise ValueError('Frozen quota exceeds or differs from cell capacity')
    if int(table[:, 1].sum()) != action['feasible_region_count']:
        raise ValueError('Frozen regional count does not match quota table')
    if action['type'] == 'resample_control':
        from .local_edit import cell_counts
        if not np.array_equal(table, cell_counts(reference, groups, mask, values)) or action['reference_region_count'] != int(mask[reference].sum()):
            raise ValueError('Resampling control must preserve every original cell count')
    elif action['feasible_region_count'] <= action['reference_region_count']:
        raise ValueError('Frozen action does not increase its reference region')
    graph = prepare_coverage_graph(graph, len(d))
    states = {}
    for row, ids in enumerate(ids_by_group):
        if table[row].sum():
            states[row] = GroupCoverageState(ids, d, h, graph, gamma=coefficients['gamma'],
                                             rho=coefficients['rho'], beta=coefficients['beta'],
                                             density_coefficient=coefficients['a'])
    cell_rng = np.random.default_rng(decision_seed)
    sample_rng = np.random.default_rng(seed)
    remaining = table.copy()
    chosen, cells, gains, bonuses = [], [], [], []
    for _ in range(len(reference)):
        cumulative_cells = np.cumsum(remaining.ravel())
        cell = min(int(np.searchsorted(cumulative_cells,
                                      cell_rng.random()*cumulative_cells[-1], side='right')),
                   remaining.size-1)
        row, side = divmod(cell, 2)
        state = states[row]
        weights, bonus = state.weights()
        weights[mask[state.ids] != bool(side)] = 0.
        cumulative = np.cumsum(weights)
        if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0:
            raise RuntimeError('Quota cell has no positive-weight available rows')
        i = min(int(np.searchsorted(cumulative, sample_rng.random()*cumulative[-1], side='right')),
                len(state.ids)-1)
        if state.used[i] or mask[state.ids[i]] != bool(side) or remaining[row, side] <= 0:
            raise RuntimeError('Regional selection invariant failed')
        chosen.append(int(state.ids[i]));cells.append([int(values[row]), side])
        gains.append(float(state.gain[i]));bonuses.append(float(bonus[i]))
        state.select(i)
        remaining[row, side] -= 1
    indices = np.asarray(chosen, int)
    actual = np.array([[int(np.sum((groups[indices] == value)&(~mask[indices]))),
                        int(np.sum((groups[indices] == value)&mask[indices]))] for value in values])
    if len(np.unique(indices)) != len(reference) or np.any(remaining) or not np.array_equal(actual, table):
        raise RuntimeError('Final regional quota validation failed')
    return indices, {
        'action_id': action['action_id'], 'mechanism': ('unchanged_quota_resampling_control'
            if action['type'] == 'resample_control' else 'regional_marginal_coverage_probability'),
        'sampling_seed': seed, 'decision_seed': decision_seed, 'sampler_version': SAMPLER_VERSION,
        'sampler_config': coefficients, 'final_size': len(indices),
        'selection_order_q_indices': indices.tolist(), 'cell_order': cells,
        'quota_columns': ['outside', 'inside'], 'group_values': values.tolist(),
        'quota_table': table.tolist(), 'gain_at_draw': gains, 'bonus_at_draw': bonuses,
        'group_scales': [{'group': int(values[row]), 'scale': state.scale} for row, state in states.items()],
        'final_coverage': [state.summary(values[row]) for row, state in states.items()],
        'coverage_scope': 'original group; inside/outside share weights, scale and coverage; no cross-group edges',
        'cell_schedule_randomness': 'decision_seed; identical cell schedule across sampling repeats',
        'positive_increase_reference': ('not_applicable_unchanged_cell_quotas'
            if action['type'] == 'resample_control' else 'reference_seed_only'),
    }
