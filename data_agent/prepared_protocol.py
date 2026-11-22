"""Honor explicitly frozen preparation metadata before fitting any model."""
from dataclasses import replace

import numpy as np
import pandas as pd

from .coreset import file_hash


def preparation_config(task, cfg):
    overrides = task.get('recommended_config_overrides', {})
    allowed = {'proxy_train_min_rows', 'core_validation_min_rows'}
    if not isinstance(overrides, dict) or set(overrides) - allowed:
        raise ValueError('Unsupported dataset preparation overrides')
    if any(type(v) is not int or v < 2 for v in overrides.values()):
        raise ValueError('Preparation minima must be integers >= 2')
    result = replace(cfg, **overrides) if overrides else cfg
    result.validate()
    return result


def internal_split_mode(task, split):
    """Resolve old and new frozen dependency declarations without an IID fallback."""
    protocol = task.get('screening_protocol', {})
    aliases = {'time': 'time', 'temporal': 'time', 'chronological': 'time',
               'group': 'group', 'grouped': 'group'}
    declared = protocol.get('internal_split')
    if declared is not None and declared not in {'stratified', *aliases}:
        raise ValueError('Unsupported internal preparation protocol')
    outer = task.get('split', {})
    modes = {aliases[value] for value in
             (protocol.get('outer_split'), outer.get('origin'), outer.get('strategy'),
              split.get('strategy'), split.get('original_strategy'), declared)
             if isinstance(value, str) and value in aliases}
    if len(modes) > 1 or (modes and declared == 'stratified'):
        raise ValueError('Conflicting time/group and internal preparation protocols')
    return next(iter(modes)) if modes else 'stratified'


def _protocol_values(task, task_path, split, rows):
    mode = internal_split_mode(task, split)
    if mode == 'stratified':
        return mode, None, None
    filename = task.get('protocol_files', {}).get('train')
    if not filename:
        raise ValueError(f'{mode} split requires a frozen training protocol sidecar; '
                         'regenerate the prepared task with time/group metadata')
    path = task_path.parent / filename
    if not path.is_file():
        raise ValueError('Missing frozen training protocol sidecar: ' + str(path))
    if split['sha256'].get(path.name) != file_hash(path):
        raise ValueError('Protocol sidecar changed after freezing')
    sidecar = pd.read_csv(path)
    if (len(sidecar) != rows or not {'source_position', mode} <= set(sidecar)
            or sidecar[mode].isna().any()):
        raise ValueError('Protocol sidecar is incomplete or misaligned')
    if sidecar.source_position.tolist() != split['indices']['train']:
        raise ValueError('Protocol source positions are misaligned')
    return mode, sidecar[mode].to_numpy(), path


def validate_prepared_partition(task, task_path, split, p, v, q):
    """Reject dependent legacy source runs whose cached internal split leaks units."""
    mode, values, _ = _protocol_values(task, task_path, split, split['rows']['train'])
    if mode == 'stratified':
        return
    parts = [np.asarray(part) for part in (p, v, q)]
    if any(part.ndim != 1 or not part.size or part.dtype.kind not in 'iu' for part in parts):
        raise ValueError('Dependent source requires complete P/V_core/Q positions')
    positions = np.concatenate(parts)
    if not np.array_equal(np.sort(positions), np.arange(len(values))):
        raise ValueError('Protocol P/V_core/Q must partition the frozen training rows')
    p, v, q = parts
    if set(values[v]) & set(values[np.r_[p, q]]):
        raise ValueError('Validation protocol units overlap fitting units')
    if mode == 'group' and set(values[p]) & set(values[q]):
        raise ValueError('P and Q groups overlap')
    if mode == 'time' and values[np.r_[p, q]].max() >= values[v].min():
        raise ValueError('Internal validation must be strictly later than P and Q')


def partition_coreset_prepared(y, task, task_path, split, p, v, cfg):
    """Split the standalone selection pool and V_core using frozen units.

    Whole units may round validation size up, but must retain v < p, enough
    selection rows, and (for classification) every class in both partitions.
    The ordinary IID path keeps its original stratified sampling behavior.
    """
    from .coreset import internal_split
    mode, values, path = _protocol_values(task, task_path, split, len(y))
    if mode == 'stratified':
        return internal_split(y, task['problem_type'], p, v, cfg)
    unique, encoded = np.unique(values, return_inverse=True)
    counts = np.bincount(encoded)
    classification = task['problem_type'] == 'classification'
    labels = set(y) if classification else set()
    rng = np.random.default_rng(cfg.core_validation_seed)
    attempts = 64 if mode == 'group' else 1
    for attempt in range(attempts):
        order = rng.permutation(len(unique)) if mode == 'group' else np.arange(len(unique))[::-1]
        unit_count = int(np.searchsorted(np.cumsum(counts[order]), v)) + 1
        mask = np.isin(encoded, order[:unit_count])
        validation, pool = np.flatnonzero(mask), np.flatnonzero(~mask)
        if not 0 < len(validation) < p or len(pool) < p:
            continue
        if classification and (set(y.iloc[pool]) != labels or set(y.iloc[validation]) != labels):
            continue
        break
    else:
        raise ValueError(f'Cannot construct {mode} protocol split in {attempts} attempt(s): '
                         'whole validation units must preserve v < p, coreset capacity '
                         'and classification coverage in both partitions')
    if set(values[pool]) & set(values[validation]):
        raise ValueError('Validation protocol units overlap selection pool units')
    if mode == 'time' and values[pool].max() >= values[validation].min():
        raise ValueError('Internal validation must be strictly later than the selection pool')
    info = {'internal_protocol': mode, 'protocol_sha256': file_hash(path),
            'protocol_units': {'selection_pool': len(np.unique(values[pool])),
                               'V_core': len(np.unique(values[validation]))},
            'requested_validation_rows': v, 'validation_rows': len(validation),
            'pool_rows': len(pool), 'unit_rounding_extra_rows': len(validation) - v,
            'validation_requires_class_coverage': classification, 'attempts': attempt + 1,
            'selection_policy': ('latest_whole_time_units' if mode == 'time' else 'seeded_whole_groups')}
    return pool, validation, info


def partition_prepared(y, task, task_path, split, cfg):
    from .multi_proxy import partition, split_quotas
    from .coreset import strata, stratified_sample
    mode, values, path = _protocol_values(task, task_path, split, len(y))
    p, v, q, info = partition(y, task['problem_type'], cfg)
    if mode == 'stratified':
        return p, v, q, info
    unique, encoded = np.unique(values, return_inverse=True)
    counts = np.bincount(encoded)
    classification = task['problem_type'] == 'classification'
    labels = set(y) if classification else set()
    goal_p, goal_v, core = len(p), len(v), info['actual_coreset_rows']
    rng = np.random.default_rng(cfg.split_seed)

    def take_units(order, goal):
        size = min(len(order), int(np.searchsorted(np.cumsum(counts[order]), goal))+1)
        return order[:size]

    # Group splits can require additional units for class coverage. Attempts use
    # only labels and identities; no fitted score or held-out prediction is read.
    for attempt in range(64 if mode == 'group' else 1):
        order = rng.permutation(len(unique)) if mode == 'group' else np.arange(len(unique))[::-1]
        vu = take_units(order, goal_v)
        v = np.flatnonzero(np.isin(encoded, vu))
        remaining = np.flatnonzero(~np.isin(encoded, vu))
        if mode == 'time':
            groups, _ = strata(y.iloc[remaining], task['problem_type'], cfg.regression_bins,
                               max_strata=min(goal_p, core))
            available = np.bincount(groups)
            if len(remaining) < goal_p+core or (classification and set(y.iloc[remaining]) != labels):
                raise ValueError('Chronological P/Q history is too small or lacks a later class; cannot repartition through future rows')
            quota = split_quotas(available, goal_p, cfg.split_seed, upper=available-1)
            p = remaining[stratified_sample(groups, quota, cfg.split_seed)]
        else:
            order = order[len(vu):]
            pu = take_units(order, goal_p).tolist()
            while pu:
                p = np.flatnonzero(np.isin(encoded, pu))
                if not classification or set(y.iloc[p]) == labels:
                    break
                if len(pu) >= len(order)-1:
                    break
                pu.append(int(order[len(pu)]))
        q = np.setdiff1d(remaining, p)
        if not len(v) or len(q) < core or not len(p):
            continue
        if classification and (set(y.iloc[p]) != labels or set(y.iloc[q]) != labels):
            continue
        if task['metric'] in ('roc_auc', 'auc') and classification and set(y.iloc[v]) != labels:
            continue
        break
    else:
        raise ValueError('No feasible group-disjoint P/V_core/Q with training class coverage')
    if not len(p) or not len(v) or len(q) < core or len(np.unique(np.r_[p,v,q])) != len(y):
        raise ValueError('Infeasible protocol-aware internal split')
    if classification and (set(y.iloc[p]) != labels or set(y.iloc[q]) != labels):
        raise ValueError('Protocol split cannot represent all training classes without leakage')
    if set(values[v]) & set(values[np.r_[p,q]]):
        raise ValueError('Validation protocol units overlap fitting units')
    if mode == 'group' and set(values[p]) & set(values[q]):
        raise ValueError('P and Q groups overlap')
    if mode == 'time' and values[np.r_[p,q]].max() >= values[v].min():
        raise ValueError('Internal validation must be strictly later than P and Q')
    qgroups, qmeta = strata(y.iloc[q], task['problem_type'], cfg.regression_bins, max_strata=core)
    qcounts = np.bincount(qgroups)
    info.pop('stratum_counts', None)
    info.update(actual_proxy_rows=len(p), validation_rows=len(v), q_rows=len(q),
        v_over_p=len(v)/core, v_over_train=len(v)/len(y),
        validation_requires_class_coverage=task['metric'] in ('roc_auc','auc'),
        internal_protocol=mode, protocol_sha256=file_hash(path),
        protocol_units={name:len(np.unique(values[pos])) for name,pos in [('P',p),('V_core',v),('Q',q)]},
        coreset_quotas={'strata':qmeta,'counts':qcounts.tolist(),
                        'allocation':split_quotas(qcounts,core,cfg.split_seed+3).tolist()})
    info['adjustments'].append('Whole validation units; group P/Q also disjoint. Row ratios may increase to preserve units and training class coverage.')
    return np.sort(p), np.sort(v), np.sort(q), info
