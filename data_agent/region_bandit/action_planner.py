"""Frozen, executable regional quotas; no model scores enter this module."""
from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp


SOLVER_VERSION = 'regional_quota_lexicographic_v1'
SAMPLER_VERSION = 'region-coverage-v1'


def round_half_up(value):
    if not np.isfinite(value) or value < 0:
        raise ValueError('round_half_up requires a finite nonnegative value')
    return int(math.floor(float(value)+.5))


def canonical_sampler_config(config=None):
    config = dict(config or {})
    result = {
        'a': float(config.get('a', config.get('density_coefficient', .3))),
        'beta': float(config.get('beta', .6)),
        'gamma': float(config.get('gamma', config.get('lambda', 2.))),
        'rho': float(config.get('rho', 1.)),
    }
    if not np.isfinite(list(result.values())).all() or not 0 <= result['a'] < 1 or any(
            result[key] < 0 for key in ('beta', 'gamma', 'rho')):
        raise ValueError('Invalid marginal coverage coefficients')
    if config.get('mechanism', 'probability') != 'probability':
        raise ValueError('Regional sampling requires probability sampling')
    return result


def _integers(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f'{name} must be a one-dimensional integer array')
    return array.astype(int, copy=False)


def _inputs(reference_indices, groups, class_ids, size):
    groups = _integers(groups, 'groups')
    reference = _integers(reference_indices, 'reference_indices')
    if type(size) is not int or size < 1 or size > len(groups) or len(reference) != size:
        raise ValueError('Invalid coreset size/reference length')
    if np.any(groups < 0) or np.any(reference < 0) or np.any(reference >= len(groups)):
        raise ValueError('Negative group or invalid reference index')
    if len(np.unique(reference)) != size:
        raise ValueError('Reference coreset contains duplicate rows')
    values, group_rows = np.unique(groups, return_inverse=True)
    capacities = np.bincount(group_rows, minlength=len(values))
    baseline = np.bincount(group_rows[reference], minlength=len(values))
    classes = class_values = class_for_group = None
    if class_ids is not None:
        classes = _integers(class_ids, 'class_ids')
        if classes.shape != groups.shape or np.any(classes < 0):
            raise ValueError('Invalid class_ids')
        class_values = np.unique(classes)
        class_for_group = []
        for group in values:
            labels = np.unique(classes[groups == group])
            if len(labels) != 1:
                raise ValueError('An original sampling group crosses class boundaries')
            class_for_group.append(int(labels[0]))
        class_for_group = np.asarray(class_for_group)
    return reference, groups, values, group_rows, capacities, baseline, classes, class_values, class_for_group


def _solve(capacity, baseline, lower, upper, class_constraints, size, t0, target,
           max_move, seconds):
    """Three MILPs share one deadline. Columns are outside, inside."""
    started = time.monotonic()
    deadline = started+seconds
    count = len(baseline)
    # x[2g:2g+2], u[g] >= |sum(x[g])-b[g]|, z[cell] >= proportional deviation.
    dimensions = 5*count
    ustart, zstart = 2*count, 3*count
    variable_upper = np.full(dimensions, np.inf)
    variable_upper[:2*count] = capacity.ravel()
    integrality = np.zeros(dimensions, int)
    integrality[:2*count] = 1
    rows, lo, hi = [], [], []

    def constrain(row, low=-np.inf, high=np.inf):
        rows.append(row);lo.append(low);hi.append(high)

    total = np.zeros(dimensions);total[:2*count] = 1
    constrain(total, size, size)
    inside = np.zeros(dimensions);inside[1:2*count:2] = 1
    constrain(inside, t0+1, target)
    for group in range(count):
        row = np.zeros(dimensions);row[2*group:2*group+2] = 1
        constrain(row, lower[group], upper[group])
        positive = row.copy();positive[ustart+group] = -1
        negative = -row;negative[ustart+group] = -1
        constrain(positive, high=baseline[group])
        constrain(negative, high=-baseline[group])
    movement = np.zeros(dimensions);movement[ustart:zstart] = 1
    constrain(movement, high=2*max_move)
    for item in class_constraints:
        row = np.zeros(dimensions)
        for group in item['group_rows']:
            row[2*group:2*group+2] = 1
        constrain(row, item['lower'], item['upper'])
    phases = []

    def solve(objective, phase):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f'Solver total deadline exhausted before {phase}')
        result = milp(objective, integrality=integrality,
                      bounds=Bounds(np.zeros(dimensions), variable_upper),
                      constraints=LinearConstraint(np.asarray(rows), np.asarray(lo), np.asarray(hi)),
                      options={'time_limit': remaining, 'mip_rel_gap': 0.})
        phases.append({'phase': phase, 'status': int(result.status),
                       'message': str(result.message), 'success': bool(result.success)})
        if not result.success or result.status != 0 or result.x is None:
            raise RuntimeError(f'{phase} failed: {result.message}')
        if time.monotonic() > deadline:
            raise RuntimeError(f'Solver total deadline exhausted during {phase}')
        if not np.isfinite(result.x).all() or np.max(np.abs(result.x[:2*count]-np.rint(result.x[:2*count]))) > 1e-5:
            raise RuntimeError(f'{phase} returned nonfinite or noninteger counts')
        return result

    try:
        first = solve(-inside, 'maximize_region_count')
        feasible = int(np.rint(first.x[:2*count].reshape(count, 2)[:, 1]).sum())
        constrain(inside, feasible, feasible)
        second = solve(movement, 'minimize_group_movement')
        table = np.rint(second.x[:2*count]).astype(int).reshape(count, 2)
        minimum_l1 = int(np.abs(table.sum(1)-baseline).sum())
        constrain(movement, minimum_l1, minimum_l1)
        objective = np.zeros(dimensions)
        desired = np.zeros((count, 2))
        for side, total_side in enumerate((size-feasible, feasible)):
            capacity_side = int(capacity[:, side].sum())
            if total_side == 0 or capacity_side == 0:
                continue
            desired[:, side] = total_side*capacity[:, side]/capacity_side
            for group in range(count):
                position = 2*group+side
                row = np.zeros(dimensions);row[position] = 1;row[zstart+position] = -1
                constrain(row, high=desired[group, side])
                row = np.zeros(dimensions);row[position] = -1;row[zstart+position] = -1
                constrain(row, high=-desired[group, side])
                objective[zstart+position] = 1./total_side
        third = solve(objective, 'match_conditional_group_proportions')
        table = np.rint(third.x[:2*count]).astype(int).reshape(count, 2)
        totals = table.sum(1)
        if (np.any(table < 0) or np.any(table > capacity) or table.sum() != size
                or not t0 < table[:, 1].sum() <= target or table[:, 1].sum() != feasible
                or np.any(totals < lower) or np.any(totals > upper)
                or np.abs(totals-baseline).sum() != minimum_l1 or minimum_l1 > 2*max_move
                or any(not item['lower'] <= totals[item['group_rows']].sum() <= item['upper']
                       for item in class_constraints)):
            raise RuntimeError('Final integer quota validation failed')
        deviation = sum(float(np.abs(table[:, side]-desired[:, side]).sum())/total_side
                        for side, total_side in enumerate((size-feasible, feasible)) if total_side)
        return {'quota_table': table.tolist(), 'feasible_region_count': feasible,
                'group_moved_count': minimum_l1//2, 'solver_status': 'optimal',
                'solver_phases': phases, 'solver_elapsed_seconds': time.monotonic()-started,
                'proportional_target_counts': desired.tolist(),
                'proportional_absolute_deviation': deviation,
                'proportional_objective': 'sum_side sum_group abs(x[group,side]/selected_side - Q[group,side]/Q_side); omit empty selected side'}, None
    except (RuntimeError, ValueError) as exc:
        return None, {'reason': str(exc), 'solver_phases': phases,
                      'solver_elapsed_seconds': time.monotonic()-started}


def plan_actions(regions, reference_indices, groups, class_ids=None, *, size,
                 config=None, quota_config=None, sampler_config=None):
    """Return ``{actions, rejected}`` with A0 first and only executable plans.

    ``quota_table`` rows follow ``group_values``; columns are outside then inside.
    Positive regional increases are relative only to the reference seed's C0.
    The movement budget constrains original group counts, not selected row overlap.
    """
    config = config or {}
    action_cfg = config.get('actions', config)
    quota_cfg = {**config.get('quotas', {}), **(quota_config or {})}
    sampler = canonical_sampler_config(sampler_config)
    reference, groups, values, group_rows, capacities, baseline, classes, class_values, class_for_group = _inputs(
        reference_indices, groups, class_ids, size)
    slack = float(quota_cfg.get('subgroup_relative_slack', .30))
    class_slack = float(quota_cfg.get('class_relative_slack', .10))
    move_fraction = float(quota_cfg.get('total_move_fraction', .10))
    seconds = float(quota_cfg.get('solver_time_limit_per_action_seconds', 2.))
    relative = float(action_cfg.get('focus_relative_increase', .25))
    points = float(action_cfg.get('focus_max_percentage_points', .05))
    if not np.isfinite([slack, class_slack, move_fraction, seconds, relative, points]).all() or min(slack, class_slack, move_fraction, relative, points) < 0 or seconds <= 0:
        raise ValueError('Invalid action/quota configuration')
    if move_fraction > 1:
        raise ValueError('Group movement fraction must not exceed one')
    action_types = action_cfg.get('types', ['baseline', 'cover', 'focus'])
    if set(action_types)-{'baseline', 'cover', 'focus'}:
        raise ValueError('Unsupported action type')
    delta = np.array([max(1, round_half_up(slack*x)) for x in baseline])
    lower = np.maximum(0, baseline-delta)
    upper = np.minimum(capacities, baseline+delta)
    max_move = int(math.floor(move_fraction*size))
    class_constraints = []
    if classes is not None:
        minima = quota_cfg.get('class_minimum_counts', {})
        for label in class_values:
            rows = np.flatnonzero(class_for_group == label)
            base = int(baseline[rows].sum())
            minimum = minima.get(str(int(label)), minima.get(int(label), 1 if base else 0))
            if type(minimum) is not int or not 0 <= minimum <= base:
                raise ValueError('Class minimum must be a nonnegative integer no larger than its reference quota')
            change = max(1, round_half_up(class_slack*base))
            class_constraints.append({'class_id': int(label), 'group_rows': rows.tolist(),
                                      'reference_count': base, 'minimum': minimum,
                                      'lower': max(minimum, base-change),
                                      'upper': min(int((classes == label).sum()), base+change)})
    common = {'size': size, 'group_values': values.tolist(), 'quota_columns': ['outside', 'inside'],
              'reference_group_counts': baseline.tolist(), 'group_capacities': capacities.tolist(),
              'group_bounds': np.column_stack([lower, upper]).tolist(), 'class_bounds': class_constraints,
              'group_quota_move_budget': max_move, 'sampler_config': sampler,
              'sampler_version': SAMPLER_VERSION, 'solver_version': SOLVER_VERSION,
              'scipy_version': scipy.__version__,
              'positive_increase_reference': 'reference_seed_only'}
    baseline_action = {**common, 'action_id': 'baseline', 'type': 'baseline', 'region_id': None,
                       'requested_region_count': None, 'feasible_region_count': None,
                       'group_moved_count': 0, 'class_moved_count': 0,
                       'solver_status': 'reference_reuse'}
    actions, rejected, seen = [baseline_action], [], {}
    for region in regions:
        mask = np.asarray(region['mask'])
        if mask.shape != groups.shape or mask.dtype.kind != 'b':
            raise ValueError('Region mask must be a boolean vector over complete Q')
        mask_hash = hashlib.sha256(np.packbits(mask).tobytes()+str(len(mask)).encode()).hexdigest()
        region_id = str(region.get('id', region.get('region_id', mask_hash[:16])))
        fraction = float(mask.mean())
        t0 = int(mask[reference].sum())
        capacity = np.column_stack([np.bincount(group_rows[~mask], minlength=len(values)),
                                    np.bincount(group_rows[mask], minlength=len(values))])
        for kind in ('cover', 'focus'):
            if kind not in action_types:
                continue
            target_fraction = fraction if kind == 'cover' else min((1+relative)*fraction, fraction+points, 1.)
            target = round_half_up(size*target_fraction)
            action_id = f'{kind}:{region_id}'
            header = {'action_id': action_id, 'type': kind, 'region_id': region_id,
                      'region_mask_hash': mask_hash, 'reference_region_count': t0,
                      'requested_region_count': target, 'region_fraction': fraction}
            if target <= t0:
                rejected.append({**header, 'reason': 'Intent does not positively increase reference regional count'})
                continue
            solution, failure = _solve(capacity, baseline, lower, upper, class_constraints,
                                       size, t0, target, max_move, seconds)
            if solution is None:
                rejected.append({**header, 'solver_version': SOLVER_VERSION, **failure})
                continue
            table = np.asarray(solution['quota_table'])
            totals = table.sum(1)
            class_moved = sum(abs(int(totals[item['group_rows']].sum())-item['reference_count'])
                              for item in class_constraints)//2
            action = {**common, **header, **solution, 'cell_capacities': capacity.tolist(),
                      'class_moved_count': class_moved,
                      'actual_region_increase': solution['feasible_region_count']-t0,
                      'actual_region_increase_fraction': (solution['feasible_region_count']-t0)/size,
                      'rule': region.get('rule')}
            signature = json.dumps([mask_hash, action['quota_table'], sampler], sort_keys=True)
            if signature in seen:
                incumbent = seen[signature]
                if kind == 'cover' and incumbent['type'] == 'focus':
                    actions.remove(incumbent)
                    rejected.append({**{key: incumbent[key] for key in ('action_id', 'type', 'region_id')},
                                     'reason': 'Duplicate executable action; cover takes precedence',
                                     'duplicate_of': action_id})
                else:
                    rejected.append({**header, 'reason': 'Duplicate executable action',
                                     'duplicate_of': incumbent['action_id']})
                    continue
            seen[signature] = action
            actions.append(action)
    return {'actions': actions, 'rejected': rejected}
