"""Offline-capable, record-only comparison of methods on fixed coresets.

No model loading, fitting, API calls, or full-stage results enter analyze_run.
Bootstrap resamples validation rows jointly across all saved predictions.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .coreset import file_hash
from .multi_proxy_io import atomic_json, read_raw
from .mcs_range import mcs_range
from .mcts_admission import normalize_admission_config, build_admission
from .stage1_metrics import (comparison_backend, metric_direction, primary_score,
                             requires_stratified_bootstrap, sample_losses)

DEFAULT_SELECTION_CONFIG = {
    'enabled': True, 'mode': 'record_only', 'bootstrap_replicates': 2000,
    'bootstrap_seed': 2026, 'alpha': .05, 'coreset_weights': [.25] * 4,
    'mcts_admission': normalize_admission_config(),
}


def normalize_selection_config(value=None):
    if value is not None and not isinstance(value, dict):
        raise ValueError('stage1_selection must be an object')
    unknown = set(value or {}) - set(DEFAULT_SELECTION_CONFIG)
    if unknown:
        raise ValueError(f'Unknown stage1_selection settings: {sorted(unknown)}')
    config = {**copy.deepcopy(DEFAULT_SELECTION_CONFIG), **(value or {})}
    if not isinstance(config['enabled'], bool) or config['mode'] != 'record_only':
        raise ValueError('stage1_selection supports boolean enabled and record_only mode only')
    for name, minimum, maximum in [('bootstrap_replicates', 2, None), ('bootstrap_seed', 0, 2**32)]:
        v = config[name]
        if isinstance(v, bool) or not isinstance(v, int) or v < minimum or (maximum and v >= maximum):
            raise ValueError(f'Invalid stage1_selection {name}')
    if isinstance(config['alpha'], bool) or not isinstance(config['alpha'], (int, float)) or not 0 < config['alpha'] < 1:
        raise ValueError('stage1_selection alpha must be in (0,1)')
    weights = config['coreset_weights']
    if not isinstance(weights, list) or not 1 <= len(weights) <= 9 or any(not isinstance(w, (int, float)) or isinstance(w, bool) or not np.isclose(w, 1/len(weights), rtol=1e-12, atol=0) for w in weights):
        raise ValueError('Coreset weights must be equal and sum to one')
    if config['mcts_admission'] is not False:
        config['mcts_admission'] = normalize_admission_config(config['mcts_admission'])
    return config


def _read(path):
    return json.loads(Path(path).read_text())


def _finite(value):
    return value is not None and isinstance(value, (int, float, np.number)) and np.isfinite(value)


def _id_key(value):
    return str(value)


def _dependency_evidence(task, split_metadata):
    """Explicit dependent sampling metadata is fail-closed; never infer from features."""
    found = []
    keys = {'group_column', 'group_columns', 'group_col', 'group_id_column', 'groups', 'group_by', 'group_key',
            'time_column', 'time_columns', 'time_col', 'timestamp_column', 'date_column', 'time_index',
            'time_series', 'is_time_series', 'temporal', 'grouped', 'panel',
            'data_dependency', 'data_dependencies', 'dependency', 'dependencies',
            'bootstrap_scheme', 'sampling_unit'}
    markers = ('group', 'time', 'temporal', 'chronolog', 'block', 'panel', 'cluster')
    def walk(node, prefix):
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            path = f'{prefix}.{key}'
            if key in keys and value not in (None, False, '', [], {}):
                if str(value).lower() not in {'iid', 'independent', 'independent_rows', 'row', 'rows'}:
                    found.append({'field': path, 'value': value})
            elif key in {'strategy', 'split_strategy', 'original_strategy', 'type', 'kind'} and isinstance(value, str):
                if any(marker in value.lower() for marker in markers):
                    found.append({'field': path, 'value': value})
            if isinstance(value, dict):
                walk(value, path)
    walk(task, 'task'); walk(split_metadata, 'split_metadata')
    return found


def bootstrap_indices(y, config, scheme):
    """Reconstructible generator: PCG64, ordered strata, one draw per stratum/replicate."""
    rng = np.random.Generator(np.random.PCG64(config['bootstrap_seed']))
    n = len(y)
    strata = [np.flatnonzero(y == label) for label in np.unique(y)] if scheme == 'stratified_by_true_class' else None
    for _ in range(config['bootstrap_replicates']):
        if strata is None:
            yield rng.integers(0, n, size=n, dtype=np.int64)
        else:
            yield np.concatenate([group[rng.integers(0, len(group), size=len(group), dtype=np.int64)] for group in strata])


def _aligned_predictions(folder, rows, classes):
    report = _read(folder / 'evaluation.json')
    if report.get('evaluation_set') != 'V_core' or report.get('evaluation_indices') != 'V_core_indices.json':
        raise ValueError(f'{folder.name}: predictions are not certified as V_core evaluations')
    if report.get('prediction_rows') != rows:
        raise ValueError(f'{folder.name}: prediction row count mismatch')
    with np.load(folder / 'predictions.npz', allow_pickle=False) as result:
        positions = result['row_positions']
        if positions.shape != (rows,) or positions.dtype.kind not in 'iu' or not np.array_equal(np.sort(positions), np.arange(rows)):
            raise ValueError(f'{folder.name}: row_positions must be a permutation of V_core row positions')
        order = np.argsort(positions)
        predictions = result['predictions']
        if predictions.shape != (rows,):
            raise ValueError(f'{folder.name}: invalid prediction shape')
        predictions = predictions[order]
        probabilities = result['probabilities'] if 'probabilities' in result.files else np.empty((0, 0))
        if probabilities.size:
            if probabilities.ndim != 2 or probabilities.shape[0] != rows:
                raise ValueError(f'{folder.name}: invalid probability shape')
            probabilities = probabilities[order]
            observed = report.get('classes')
            if observed is None or len(set(observed)) != len(observed) or set(observed) != set(classes):
                raise ValueError(f'{folder.name}: probability classes disagree with shared class order')
            probabilities = probabilities[:, [observed.index(c) for c in classes]]
        else:
            probabilities = None
    return report, predictions, probabilities


def analyze_run(run_dir, config=None, manifest=None, *, include_gmm=True, coreset_count=None):
    """Analyze saved V_core predictions, replacing only analysis output artifacts."""
    root = Path(run_dir).resolve()
    manifest = manifest if manifest is not None else _read(root / 'manifest.json')
    cfg = normalize_selection_config(config if config is not None else manifest.get('configuration', {}).get('stage1_selection'))
    task = manifest['task']
    reports = _read(root / 'evaluations.json')
    m = manifest.get('configuration', {}).get('n_strategies', 3)
    count = m + 1 if include_gmm else m
    if coreset_count is not None:
        if not include_gmm or type(coreset_count) is not int or coreset_count!=1 or cfg['coreset_weights']!=[1.]:
            raise ValueError('Explicit single-coreset analysis requires one weight and the standard output folder')
        count=coreset_count
    if not include_gmm:
        reports = [r for r in reports if r['coreset_id'] in range(count)]
    method_ids = sorted({r['method_id'] for r in reports}, key=_id_key)
    if len(method_ids) != m or len(reports) != m*count or {(r['method_id'], r['coreset_id']) for r in reports} != {(i, j) for i in method_ids for j in range(count)}:
        raise ValueError(f'Stage-one analysis requires exactly {m} methods and {count} complete coreset evaluations each')
    folder = root / ('stage1_selection' if include_gmm else 'stage1_selection_without_gmm')
    folder.mkdir(exist_ok=True)
    # Obsolete comparison must not outlive a recomputed first-stage conclusion.
    (folder / 'full_train_comparison.json').unlink(missing_ok=True)
    metric = task['metric']
    direction = metric_direction(task)
    backend = comparison_backend(task)
    sign = 1 if direction == 'maximize' else -1
    values = np.full((m, count), np.nan)
    for r in reports:
        score = r.get('metrics', {}).get(metric)
        if _finite(score):
            values[method_ids.index(r['method_id']), r['coreset_id']] = float(score)
    rewards = sign * values
    means = values.mean(axis=1)
    initial = rewards.mean(axis=1)
    order = sorted(range(m), key=lambda i: (not np.isfinite(initial[i]), -initial[i] if np.isfinite(initial[i]) else 0, _id_key(method_ids[i])))
    ranks = {i: rank for rank, i in enumerate(order, 1) if np.isfinite(initial[i])}
    n = manifest.get('evaluation_set', {}).get('rows', 0)
    scheme = 'stratified_by_true_class' if requires_stratified_bootstrap(task) else 'iid_rows'
    summary = {
        'analysis_version': '1.0', 'selection_mode': cfg['mode'], 'primary_metric': metric,
        'metric_parameters': task.get('metric_params', {}), 'metric_direction': direction,
        'comparison_backend': backend, 'bootstrap_scheme': scheme,
        'bootstrap_seed': cfg['bootstrap_seed'], 'bootstrap_replicates': cfg['bootstrap_replicates'],
        'alpha': cfg['alpha'], 'coreset_weights': [1/count]*count, 'validation_rows': n,
        'include_gmm': include_gmm and coreset_count is None, 'included_coreset_ids': list(range(count)),
        'fidelity': {'level': 'low', 'coreset_rows': manifest.get('sizes', {}).get('actual_coreset_rows'),
                     'evaluation_budget': manifest.get('configuration', {}).get('evaluation_budget'),
                     'actual_configurations': 'method_configurations.json'},
        'method_ids': method_ids, 'survivor_method_ids': method_ids, 'inactive_method_ids': [],
        'selection_status': 'pending', 'selection_reason': None,
        'initial_ranking': [method_ids[i] for i in order if i in ranks],
        'scope': f'Conditional on {count} fixed equally weighted training subsets and fitted methods; validation-sample uncertainty only. Not IID replicates or an ensemble. No MCTS state.',
        'uncertainty_conditioning': 'observed class counts' if scheme == 'stratified_by_true_class' else 'observed validation sampling distribution',
        'range_rule': 'MCS range elimination; metric-level centered-bootstrap adaptation' if backend == 'metric_recompute' else 'MCS range elimination on paired mean sample losses',
        'reference': 'https://sylvain-barde.github.io/projects/fast_mcs/',
        'second_stage_policy': 'train all methods regardless of recorded inactivity',
    }
    bootstrap = np.empty((0, m)); losses = np.empty((0, m)); trace = []
    difference = -initial[:, None] + initial[None, :]
    se = np.full((m, m), np.nan); statistic = np.full((m, m), np.nan)
    boot_se = np.full(m, np.nan); pvalues = {i: None for i in method_ids}
    vpositions = np.empty(0, dtype=np.int64); y = np.empty(0)
    configurations = [{'method_id': r['method_id'], 'coreset_id': r['coreset_id'],
        'pipeline_spec': r.get('pipeline_spec'), 'stage_parameters': r.get('stage_parameters'),
        'actual_budget': r.get('actual_budget'),
        'report': f"evaluations/method_{r['method_id']}_coreset_{r['coreset_id']}/evaluation.json"}
        for r in reports]
    try:
        indices = _read(root / 'V_core_indices.json')
        vpositions = np.asarray(indices['train_positions'])
        frame = read_raw(root / 'core_validation.csv')
        if frame[task['target']].isna().any():
            raise ValueError('V_core target contains missing labels')
        y = frame[task['target']].to_numpy().astype(float if task['problem_type'] == 'regression' else str)
        if len(y) == 0 or len(y) != n or vpositions.shape != (n,) or vpositions.dtype.kind not in 'iu' or len(np.unique(vpositions)) != n or np.any(vpositions < 0):
            raise ValueError('V_core labels, saved row indices and evaluation row count disagree')
        if indices.get('internal_split_id') != manifest.get('internal_split_id'):
            raise ValueError('V_core split identity mismatch')
        summary['validation_identity'] = {'labels_sha256': file_hash(root / 'core_validation.csv'),
            'indices_sha256': file_hash(root / 'V_core_indices.json'), 'internal_split_id': indices.get('internal_split_id'),
            'train_positions': vpositions.tolist(), 'source_positions': indices.get('source_positions')}
        saved_set = manifest.get('evaluation_set', {})
        for saved_key, filename in [('labels_sha256', 'core_validation.csv'), ('indices_sha256', 'V_core_indices.json')]:
            if saved_set.get(saved_key) is not None and saved_set[saved_key] != file_hash(root / filename):
                raise ValueError(f'V_core {filename} changed since evaluation')
        context = _read(root / 'task_context.json') if (root / 'task_context.json').exists() else {}
        classes = context.get('classes') or (sorted(np.unique(y).tolist()) if task['problem_type'] == 'classification' else [])
        predictions = {}; sample_columns = np.empty((n, m, count)) if backend == 'sample_loss' else None
        for i, method in enumerate(method_ids):
            for j in range(count):
                run = root / 'evaluations' / f'method_{method}_coreset_{j}'
                report, pred, proba = _aligned_predictions(run, n, classes)
                if report.get('method_id') != method or report.get('coreset_id') != j or report.get('metric') != metric:
                    raise ValueError(f'{run.name}: evaluation identity or metric mismatch')
                next(c for c in configurations if c['method_id'] == method and c['coreset_id'] == j)['prediction_sha256'] = file_hash(run / 'predictions.npz')
                for report_key, identity_key in [('evaluation_labels_sha256', 'labels_sha256'), ('evaluation_indices_sha256', 'indices_sha256')]:
                    if report.get(report_key) is not None and report[report_key] != summary['validation_identity'][identity_key]:
                        raise ValueError(f'{run.name}: validation identity changed since prediction')
                recorded = report.get('metrics', {}).get(metric)
                if _finite(recorded) != np.isfinite(values[i, j]) or (_finite(recorded) and not np.isclose(recorded, values[i, j], rtol=1e-10, atol=1e-12)):
                    raise ValueError(f'{run.name}: per-fit and aggregate primary metrics disagree')
                computed = primary_score(task, y, pred, proba, classes)
                if not np.isfinite(values[i, j]) or not np.isclose(computed, values[i, j], rtol=1e-10, atol=1e-12):
                    raise ValueError(f'{run.name}: saved primary score disagrees with aligned predictions or is undefined')
                predictions[i, j] = pred, proba
                if sample_columns is not None:
                    sample_columns[:, i, j] = sample_losses(task, y, pred, proba, classes)
        dependency = _dependency_evidence(task, manifest.get('split_metadata', {}))
        summary['dependency_evidence'] = dependency
        if not cfg['enabled']:
            summary.update(selection_status='disabled', selection_reason='stage1_selection.enabled is false; all candidates retained')
        elif dependency:
            summary.update(selection_status='unsupported_dependency', bootstrap_scheme='unsupported',
                           selection_reason='Explicit temporal/group dependence requires block/group bootstrap, which is not implemented; all candidates retained')
        else:
            bootstrap = np.empty((cfg['bootstrap_replicates'], m))
            if sample_columns is not None:
                losses = sample_columns.mean(axis=2)
                # Accuracy error = 1 - accuracy, while theta = -accuracy.
                # A method-common offset restores theta exactly and cancels in pair differences.
                offsets = np.full(m, -1.0 if metric == 'accuracy' else 0.0)
                if not np.allclose(losses.mean(axis=0) + offsets, -initial, rtol=1e-10, atol=1e-12):
                    raise ValueError('Sample losses do not match the declared primary metric aggregation')
                summary['sample_loss_theta_offset'] = float(offsets[0])
            for b, rows in enumerate(bootstrap_indices(y, cfg, scheme)):
                if sample_columns is not None:
                    bootstrap[b] = losses[rows].mean(axis=0) + offsets[0]
                else:
                    for i in range(m):
                        bootstrap[b, i] = -sign * np.mean([primary_score(task, y[rows], predictions[i, j][0][rows],
                            None if predictions[i, j][1] is None else predictions[i, j][1][rows], classes) for j in range(count)])
            result = mcs_range(-initial, bootstrap, method_ids, cfg['alpha'])
            for key in ('selection_status', 'selection_reason', 'survivor_method_ids', 'inactive_method_ids'):
                summary[key] = result[key]
            summary['degenerate_pairs'] = [{k: (v if not isinstance(v, float) or np.isfinite(v) else None) for k, v in pair.items()} for pair in result['degenerate_pairs']]
            difference = result['mean_loss_difference']; se = result['standard_error']; statistic = result['t_statistic']
            boot_se = result['bootstrap_standard_error']; pvalues = result['mcs_pvalues']; trace = result['elimination_trace']
    except (ValueError, KeyError, OSError, TypeError) as error:
        # Data/metric failures invalidate screening only. Never substitute another metric.
        summary.update(selection_status='unavailable', selection_reason=f'{type(error).__name__}: {error}',
                       survivor_method_ids=method_ids, inactive_method_ids=[])
        bootstrap = np.empty((0, m))
    summary['bootstrap_replicates_completed'] = len(bootstrap)
    summary['bootstrap_reconstruction'] = {
        'generator': 'numpy.random.Generator(PCG64)', 'numpy_version': np.__version__,
        'initial_state': np.random.PCG64(cfg['bootstrap_seed']).state,
        'sampler': 'data_agent.stage1_selection.bootstrap_indices v1',
        'draws': 'IID: integers(0,N,N,int64). Stratified: sorted true classes, concatenate per-class draws retaining each class count.',
        'validation_order': 'bootstrap_statistics.npz: validation_train_positions and validation_labels',
        'resampled_coresets': False,
    }
    summary['method_mcs_pvalues'] = [{'method_id': m, 'mcs_pvalue': pvalues[m]} for m in method_ids]
    admission = None
    if cfg['mcts_admission'] is not False:
        admission = build_admission(summary, dict(zip(method_ids, initial)), cfg['mcts_admission'])
        summary['mcts_admission'] = admission
    admission_rows = {item['method_id']: item for item in admission['methods']} if admission else {}
    rows = []
    for i, method in enumerate(method_ids):
        row = {'method_id': method, **{f'coreset_{j}_score': values[i, j] for j in range(count)},
               **{f'coreset_{j}_reward': rewards[i, j] for j in range(count)},
               'mean_score': means[i], 'initial_value': initial[i], 'bootstrap_standard_error': boot_se[i],
               'initial_rank': ranks.get(i), 'mcs_pvalue': pvalues[method], 'survivor': method in summary['survivor_method_ids']}
        row['mcs_survivor'] = row['survivor']
        if admission:
            row.update(mcts_active=admission_rows[method]['mcts_active'],
                       admission_reason=admission_rows[method]['admission_reason'],
                       minimum_expansions=admission_rows[method]['minimum_expansions'])
        rows.append(row)
    pd.DataFrame(rows).to_csv(folder / 'method_scores.csv', index=False)
    pd.DataFrame([{'method_i': a, 'method_j': b, 'mean_loss_difference': difference[i, j],
                   'standard_error': se[i, j], 't_statistic': statistic[i, j]}
                  for i, a in enumerate(method_ids) for j, b in enumerate(method_ids) if i != j]).to_csv(folder / 'pairwise_statistics.csv', index=False)
    np.savez_compressed(folder / 'bootstrap_statistics.npz', theta=-initial, bootstrap_theta=bootstrap,
                        bootstrap_rewards=-bootstrap, sample_losses=losses, method_ids=np.asarray(method_ids),
                        validation_train_positions=vpositions, validation_labels=y)
    atomic_json(folder / 'method_configurations.json', configurations)
    atomic_json(folder / 'elimination_trace.json', trace)
    if admission:
        atomic_json(folder / 'mcts_admission.json', admission)
    else:
        (folder / 'mcts_admission.json').unlink(missing_ok=True)
    atomic_json(folder / 'summary.json', summary)
    return summary


def compare_full_stage(run_dir, selection_summary=None, *, include_gmm=True):
    """Post-hoc audit only: never writes first-stage scores, weights or conclusions."""
    root = Path(run_dir)
    folder = root / ('stage1_selection' if include_gmm else 'stage1_selection_without_gmm')
    selection = selection_summary if selection_summary is not None else _read(folder / 'summary.json')
    full = _read(root / 'full_train_ranking.json')
    methods = full['methods']
    ids = selection['method_ids']; survivors = set(selection['survivor_method_ids'])
    comparison = {'selection_mode': 'record_only', 'initial_ranking': selection['initial_ranking'],
        'survivor_method_ids': selection['survivor_method_ids'], 'inactive_method_ids': selection['inactive_method_ids'],
        'full_train_ranking': methods, 'primary_metric': selection['primary_metric'],
        'metric_direction': selection['metric_direction'], 'selection_status': selection['selection_status'],
        'scope': 'Post-hoc screening audit on external validation; does not update stage-one conclusions',
        'tie_rule': 'exact equality of saved primary metric values'}
    valid = (len(methods) == len(ids) and {m['method_id'] for m in methods} == set(ids)
             and all(_finite(m.get('score')) and m.get('metric') == selection['primary_metric'] for m in methods)
             and full.get('direction') == selection['metric_direction'])
    if not valid:
        comparison.update(status='unavailable', reason='Full-stage primary scores are incomplete, undefined or inconsistent')
    else:
        maximize = selection['metric_direction'] == 'maximize'
        best = (max if maximize else min)(m['score'] for m in methods)
        winners = [m['method_id'] for m in methods if m['score'] == best]
        removed = [m for m in winners if m not in survivors]
        retained_scores = [m['score'] for m in methods if m['method_id'] in survivors]
        retained_best = (max if maximize else min)(retained_scores) if retained_scores else None
        comparison.update(status='completed', full_best_method_ids=winners,
            inactive_full_best_method_ids=removed, any_full_best_inactive=bool(removed),
            all_full_best_inactive=len(removed) == len(winners),
            some_but_not_all_tied_best_inactive=0 < len(removed) < len(winners),
            all_methods_best_score=best, survivor_methods_best_score=retained_best,
            screening_regret=None if retained_best is None else (best-retained_best if maximize else retained_best-best))
    admission = selection.get('mcts_admission')
    if admission:
        active = set(admission['active_method_ids'])
        audit = {'active_method_ids': admission['active_method_ids'],
                 'inactive_method_ids': admission['inactive_method_ids'],
                 'backfilled_method_ids': admission['backfilled_method_ids']}
        if valid:
            missed = [m for m in winners if m not in active]
            active_best = (max if maximize else min)(m['score'] for m in methods if m['method_id'] in active)
            audit.update(full_best_method_ids=winners, inactive_full_best_method_ids=missed,
                any_full_best_inactive=bool(missed), all_full_best_inactive=len(missed)==len(winners),
                some_but_not_all_tied_best_inactive=0 < len(missed) < len(winners),
                all_methods_best_score=best, active_methods_best_score=active_best,
                screening_regret=best-active_best if maximize else active_best-best)
        comparison['mcts_admission_audit'] = audit
    atomic_json(folder / 'full_train_comparison.json', comparison)
    return comparison
