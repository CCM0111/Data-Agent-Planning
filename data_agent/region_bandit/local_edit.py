"""Minimal within-group exchanges of frozen historical coresets.

No validation/full outcomes enter planning or sampling. Each repeat receives
its own nonnegative quota increase; retained rows seed marginal coverage.
"""
import hashlib
import json
import math
import numpy as np

from .action_planner import _inputs, canonical_sampler_config, round_half_up, SAMPLER_VERSION
from ..marginal_coverage import GroupCoverageState, prepare_coverage_graph

LOCAL_SAMPLER_VERSION = 'region-local-edit-v1'
SCORED_SAMPLER_VERSION = 'region-local-scored-exchange-v2'
PROTECTED_SAMPLER_VERSION = 'region-local-protected-v3'

def local_sampler_version(config):
    if config['sampling']['max_donor_remove_fraction'] != 1 or config['sampling']['allocation'] != 'capacity':
        return PROTECTED_SAMPLER_VERSION
    return SCORED_SAMPLER_VERSION if config['sampling']['deletion_rule'] == 'retention_score' else LOCAL_SAMPLER_VERSION


def reference_hash(reference):
    return hashlib.sha256(np.asarray(reference, dtype='<i8').tobytes()).hexdigest()


def mask_hash(mask):
    return hashlib.sha256(np.packbits(mask).tobytes()+str(len(mask)).encode()).hexdigest()


def cell_counts(reference, groups, mask, values):
    return np.array([[int(np.sum((groups[reference] == g) & ~mask[reference])),
                      int(np.sum((groups[reference] == g) & mask[reference]))] for g in values])


def plan_local_actions(regions, references, groups, class_ids=None, *, size, config, sampler_config=None):
    from ..multi_proxy import split_quotas
    if not references:
        raise ValueError('Missing repeat references')
    references = {int(s): np.asarray(r) for s, r in references.items()}
    for reference in references.values():
        _inputs(reference, groups, class_ids, size)
    groups = np.asarray(groups)
    values = np.unique(groups)
    coefficients = canonical_sampler_config(sampler_config)
    budget = math.floor(size*config['sampling']['max_replace_fraction'])
    common = dict(size=size, group_values=values.tolist(), quota_columns=['outside', 'inside'],
        sampler_version=local_sampler_version(config), sampler_config=coefficients,
        deletion_rule=config['sampling']['deletion_rule'],
        max_donor_remove_fraction=config['sampling']['max_donor_remove_fraction'],
        allocation=config['sampling']['allocation'],
        solver_version='within_group_minimum_exchanges_v1', sampling_mode='local_edit',
        positive_increase_reference='each_seed_independently; already sufficient means unchanged',
        group_moved_count=0, class_moved_count=0, max_replaced_rows=budget)
    actions = [{**common, 'action_id': 'baseline', 'type': 'baseline', 'region_id': None,
                'solver_status': 'reference_reuse'}]
    rejected, seen = [], set()
    for region in regions:
        mask = np.asarray(region['mask'])
        if mask.shape != groups.shape or mask.dtype.kind != 'b':
            raise ValueError('Invalid complete-Q region mask')
        rid = str(region.get('region_id', region.get('id')))
        fraction = float(mask.mean())
        for kind in ('cover', 'focus'):
            if kind not in config['actions']['types']:
                continue
            target_fraction = fraction if kind == 'cover' else min(
                fraction*(1+config['actions']['focus_relative_increase']),
                fraction+config['actions']['focus_max_percentage_points'], 1.)
            target = round_half_up(size*target_fraction)
            plans, effects = {}, []
            for seed, reference in references.items():
                table = cell_counts(reference, groups, mask, values)
                original_count = int(table[:, 1].sum())
                original_outside = table[:, 0].copy()
                available = mask.copy(); available[reference] = False
                capacities = np.minimum(table[:, 0], [np.sum(available & (groups == g)) for g in values])
                donor_limits = np.floor(original_outside*common['max_donor_remove_fraction']).astype(int)
                capacities = np.minimum(capacities, donor_limits)
                increment = min(max(0, target-original_count), budget, int(capacities.sum()))
                swaps = np.zeros(len(values), int)
                if increment:
                    eligible = capacities > 0
                    counts = original_outside if common['allocation'] == 'original_outside' else capacities
                    swaps[eligible] = split_quotas(counts[eligible], increment, config['decision_seed'],
                        lower=np.zeros(int(eligible.sum()), int), upper=capacities[eligible])
                table[:, 0] -= swaps; table[:, 1] += swaps
                plans[str(seed)] = dict(reference_hash=reference_hash(reference), quota_table=table.tolist(),
                    reference_region_count=original_count, feasible_region_count=original_count+increment,
                    actual_region_increase=increment, original_outside_counts=original_outside.tolist(),
                    swap_capacities=capacities.tolist(), allocation_seed=config['decision_seed'])
                effects.append(dict(seed=seed, before=original_count, after=original_count+increment,
                                    replaced_rows=increment, retained_fraction=1-increment/size,
                                    donor_protection=donor_summary(values, original_outside, swaps, donor_limits)))
            if not any(p['actual_region_increase'] for p in plans.values()):
                rejected.append(dict(action_id=f'{kind}:{rid}', reason='No repeat has a feasible positive increase'))
                continue
            signature = json.dumps([mask_hash(mask), [p['quota_table'] for p in plans.values()]])
            if signature in seen:
                rejected.append(dict(action_id=f'{kind}:{rid}', reason='Duplicate executable quotas across repeats'))
                continue
            seen.add(signature)
            first = next(iter(plans.values()))
            actions.append({**common, **first, 'action_id': f'{kind}:{rid}', 'type': kind, 'region_id': rid,
                'region_mask_hash': mask_hash(mask), 'region_fraction': fraction, 'requested_region_count': target,
                'actual_region_increase_fraction': float(np.mean([p['actual_region_increase']/size for p in plans.values()])),
                'per_seed_plans': plans, 'repeat_effects': effects, 'rule': region.get('rule'),
                'solver_status': 'exact_minimum_replacements'})
    return dict(actions=actions, rejected=rejected)


def sample_local_action(action, reference_indices, density, hardness, graph, groups,
                        region_mask=None, *, seed, config=None, decision_seed=2026):
    reference, groups, values, *_ = _inputs(reference_indices, groups, None, action['size'])
    d, h = np.asarray(density, float), np.asarray(hardness, float)
    if d.shape != groups.shape or h.shape != d.shape or not np.isfinite([d, h]).all() or np.any((d < 0)|(d > 1)|(h < 0)|(h > 1)):
        raise ValueError('Invalid percentiles')
    coefficients = canonical_sampler_config(action['sampler_config'])
    expected_version = SCORED_SAMPLER_VERSION if action.get('deletion_rule') == 'retention_score' else LOCAL_SAMPLER_VERSION
    if action.get('max_donor_remove_fraction', 1) != 1 or action.get('allocation', 'capacity') != 'capacity':
        expected_version = PROTECTED_SAMPLER_VERSION
    if action['sampler_version'] != expected_version or (config is not None and canonical_sampler_config(config) != coefficients):
        raise ValueError('Local sampler configuration/version changed')
    audit = dict(action_id=action['action_id'], sampler_version=expected_version,
                 mechanism='minimal_within_group_exchanges', sampling_seed=seed,
                 sampler_config=coefficients, final_size=len(reference))
    if action['type'] == 'baseline':
        return reference.copy(), {**audit, 'retained_fraction': 1., 'replaced_rows': 0,
                                  'selection_order_q_indices': reference.tolist()}
    if action['type'] not in ('cover', 'focus'):
        raise ValueError('Unsupported local action')
    mask = np.asarray(region_mask)
    if mask.dtype.kind != 'b' or mask.shape != groups.shape or mask_hash(mask) != action['region_mask_hash']:
        raise ValueError('Frozen local region changed')
    plan = action['per_seed_plans'][str(seed)]
    if reference_hash(reference) != plan['reference_hash'] or values.tolist() != action['group_values']:
        raise ValueError('Frozen local reference changed')
    before = cell_counts(reference, groups, mask, values)
    target = np.asarray(plan['quota_table'])
    if target.dtype.kind not in 'iu' or target.shape != before.shape or np.any(target < 0):
        raise ValueError('Invalid local quota table')
    swaps = target[:, 1]-before[:, 1]
    if (np.any(swaps < 0) or not np.array_equal(target.sum(1), before.sum(1)) or
            swaps.sum() != plan['actual_region_increase'] or swaps.sum() > action['max_replaced_rows'] or
            before[:, 1].sum() != plan['reference_region_count'] or target[:, 1].sum() != plan['feasible_region_count']):
        raise ValueError('Local exchange invariants violated')
    donor_limit = np.floor(before[:, 0]*action.get('max_donor_remove_fraction', 1.)).astype(int)
    if np.any(swaps > donor_limit):
        raise ValueError('Frozen exchange exceeds original outside-region donor limits')
    exchanges = []
    if action.get('deletion_rule') == 'retention_score':
        removed, added, exchanges = scored_exchanges(reference, groups, values, swaps, mask, d, h, graph, coefficients, seed)
        retained = reference[~np.isin(reference, removed)]
    else:
        rng = np.random.default_rng(seed)
        removed = []
        for g, count in zip(values, swaps):
            if count:
                removed.extend(rng.choice(reference[(groups[reference] == g) & ~mask[reference]], int(count), replace=False).tolist())
        retained = reference[~np.isin(reference, removed)]
        added = []
        if swaps.sum():
            graph = prepare_coverage_graph(graph, len(groups))
            for g, count in zip(values, swaps):
                if not count:
                    continue
                ids = np.flatnonzero(groups == g)
                state = GroupCoverageState(ids, d, h, graph, gamma=coefficients['gamma'], rho=coefficients['rho'],
                                           beta=coefficients['beta'], density_coefficient=coefficients['a'])
                for i in np.searchsorted(ids, retained[groups[retained] == g]):
                    state.select(int(i))
                for _ in range(int(count)):
                    weights, _ = state.weights(); weights[~mask[ids]] = 0.
                    cumulative = np.cumsum(weights)
                    if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0:
                        raise RuntimeError('No feasible positive local sampling weight')
                    i = min(int(np.searchsorted(cumulative, rng.random()*cumulative[-1], side='right')), len(ids)-1)
                    added.append(int(ids[i])); state.select(i)
    result = np.r_[retained, np.asarray(added, int)]
    if (len(result) != len(reference) or len(np.unique(result)) != len(reference) or
            not np.array_equal(cell_counts(result, groups, mask, values), target) or
            len(np.intersect1d(result, reference)) != len(reference)-int(swaps.sum())):
        raise RuntimeError('Final local sample validation failed')
    return result, {**audit, 'removed_q_indices': removed, 'added_q_indices': added,
        'retained_fraction': len(retained)/len(reference), 'replaced_rows': len(added),
        'reference_region_count': plan['reference_region_count'], 'feasible_region_count': plan['feasible_region_count'],
        'quota_table': target.tolist(), 'selection_order_q_indices': result.tolist(),
        'donor_protection': donor_summary(values, before[:, 0], swaps, donor_limit),
        'coverage_initialization': ('all original rows, then sequential remove/add' if exchanges
                                    else 'all retained original rows in each group'),
        'deletion_rule': action.get('deletion_rule', 'uniform'), 'exchanges': exchanges,
        'exchange_order': 'delete minimum current retention score, add by probability, update, repeat'
            if exchanges else 'original local sampler'}


def resample_control_action(reference, groups, mask, sampler_config):
    reference, groups = np.asarray(reference, int), np.asarray(groups, int)
    values = np.unique(groups)
    count = int(mask[reference].sum())
    return dict(action_id='resample_control', type='resample_control', size=len(reference),
        sampler_version=SAMPLER_VERSION, sampler_config=canonical_sampler_config(sampler_config),
        group_values=values.tolist(), quota_columns=['outside', 'inside'],
        quota_table=cell_counts(reference, groups, mask, values).tolist(),
        region_mask_hash=mask_hash(mask), reference_region_count=count, feasible_region_count=count)


def scored_exchanges(reference, groups, values, swaps, mask, d, h, graph, coefficients, seed):
    from .exchange_coverage import ExchangeCoverageState
    rng = np.random.default_rng(seed)
    removed, added, audit = [], [], []
    if not swaps.sum():
        return removed, added, audit
    graph = prepare_coverage_graph(graph, len(groups))
    for g, count in zip(values, swaps):
        if not count:
            continue
        ids = np.flatnonzero(groups == g)
        state = ExchangeCoverageState(ids, d, h, graph, gamma=coefficients['gamma'], rho=coefficients['rho'],
            beta=coefficients['beta'], density_coefficient=coefficients['a'])
        for i in np.searchsorted(ids, reference[groups[reference] == g]):
            state.select(int(i))
        for _ in range(int(count)):
            losses = state.removal_losses()
            retention = state.base+state.gamma*losses/(losses+state.scale)
            candidates = np.flatnonzero(state.used & ~mask[ids])
            if not len(candidates):
                raise RuntimeError('No eligible original outside-region row to remove')
            minimum = retention[candidates].min()
            victim = int(rng.choice(candidates[retention[candidates] == minimum]))
            loss = float(losses[victim]); score = float(retention[victim])
            state.remove(victim)
            weights, _ = state.weights(); weights[~mask[ids]] = 0.
            cumulative = np.cumsum(weights)
            if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0:
                raise RuntimeError('No positive feasible addition weight')
            newcomer = min(int(np.searchsorted(cumulative, rng.random()*cumulative[-1], side='right')), len(ids)-1)
            gain = float(state.gain[newcomer])
            audit.append(dict(group=int(g), removed=int(ids[victim]), added=int(ids[newcomer]),
                removal_coverage_loss=loss, retention_score=score, addition_coverage_gain=gain,
                coverage_scale=float(state.scale)))
            removed.append(int(ids[victim])); added.append(int(ids[newcomer]))
            state.select(newcomer)
    return removed, added, audit


def donor_summary(values, outside, swaps, limits):
    ratios = np.divide(swaps, outside, out=np.zeros(len(outside), float), where=outside > 0)
    groups = [dict(group=int(g), original_outside=int(n), removed=int(k), remaining=int(n-k),
                   removal_fraction=float(r), protected_limit=int(limit))
              for g,n,k,r,limit in zip(values, outside, swaps, ratios, limits) if n > 0]
    groups.sort(key=lambda row: (-row['removal_fraction'], row['original_outside'], row['group']))
    return dict(original_outside=int(outside.sum()), removed=int(swaps.sum()),
        outside_removal_fraction=float(swaps.sum()/outside.sum()) if outside.sum() else 0.,
        max_group_removal_fraction=float(ratios.max()) if len(ratios) else 0.,
        protected_small_group_count=int(np.sum((outside > 0) & (limits == 0))),
        most_affected_groups=groups[:5])


def scale_local_action(action, fraction):
    """Materialize the Agent's chosen fraction of an already feasible maximum."""
    from copy import deepcopy
    from ..multi_proxy import split_quotas
    if type(fraction) not in (int, float) or not np.isfinite(fraction):
        raise ValueError('Invalid adjustment fraction')
    if action['type'] == 'baseline':
        if fraction != 0:
            raise ValueError('Baseline adjustment fraction must be zero')
        return {**deepcopy(action), 'agent_adjustment_fraction': 0.}
    if not 0 < fraction <= 1:
        raise ValueError('Regional adjustment fraction must be in (0,1]')
    result = deepcopy(action)
    values = np.asarray(action['group_values']);size=action['size'];effects=[]
    for seed, plan in result['per_seed_plans'].items():
        outside = np.asarray(plan['original_outside_counts'], int)
        capacity = np.asarray(plan['swap_capacities'], int)
        old_table = np.asarray(plan['quota_table'], int)
        old_swaps = outside-old_table[:, 0]
        original_inside = old_table[:, 1]-old_swaps
        maximum = plan['actual_region_increase']
        count = min(maximum, max(1, round_half_up(maximum*fraction))) if maximum else 0
        swaps = np.zeros(len(values), int)
        if count:
            eligible=capacity > 0
            weights=outside if action['allocation']=='original_outside' else capacity
            swaps[eligible]=split_quotas(weights[eligible], count, plan['allocation_seed'],
                lower=np.zeros(int(eligible.sum()),int), upper=capacity[eligible])
        plan.update(quota_table=np.column_stack([outside-swaps,original_inside+swaps]).tolist(),
                    maximum_feasible_region_increase=maximum, actual_region_increase=count,
                    feasible_region_count=plan['reference_region_count']+count)
        effects.append(dict(seed=int(seed),before=plan['reference_region_count'],after=plan['feasible_region_count'],
            replaced_rows=count,retained_fraction=1-count/size,
            donor_protection=donor_summary(values,outside,swaps,
                np.floor(outside*action['max_donor_remove_fraction']).astype(int))))
    first=next(iter(result['per_seed_plans'].values()))
    for key in ('quota_table','actual_region_increase','feasible_region_count'):
        result[key]=first[key]
    result.update(agent_adjustment_fraction=float(fraction), repeat_effects=effects,
        actual_region_increase_fraction=float(np.mean([p['actual_region_increase']/size for p in result['per_seed_plans'].values()])))
    return result
