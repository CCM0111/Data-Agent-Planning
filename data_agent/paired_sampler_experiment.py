"""Paired sampler comparison; shared initialization, isolated search feedback.

Stage one uses a single five-method full reference. Stage two imports only the
unfitted method definitions/P-Q plan, never stage-one validation scores.
"""
from copy import deepcopy
from pathlib import Path
import math
import shutil

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .coreset import export_rows, file_hash, strata, stratified_sample
from .mcts_policy import MCTSConfig
from .mcts_search import compare_candidates
from .multi_proxy import load_fixed, split_quotas
from .multi_proxy_io import read_typed
from .region_bandit.config import normalize_config
from .region_bandit.mcts_rounds import RoundBandit
from .region_bandit.reward import compute_action_reward
from .region_bandit.runtime import run_region_task
from .region_bandit.schemas import read_json, write_json, stable_hash
from .stage1_metrics import metric_direction
from .workflow import WorkflowBackend, run_search, source_view, finish_region_feedback

ARMS = ('random', 'coverage', 'bandit')


def freeze_json(path, value):
    path = Path(path)
    if path.exists() and read_json(path) != value:
        raise ValueError('Frozen experiment inputs changed: ' + str(path))
    write_json(path, value)


def base_samples(source, output, seed=20261026):
    """Reproduce the repository's class/target-quantile stratified random baseline."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(source/'manifest.json')
    task, split, paths, _, _ = load_fixed(Path(manifest['task_path']))
    context = read_json(source/'task_context.json')
    qpos = np.asarray(read_json(source/'Q_indices.json')['train_positions'], dtype=int)
    reference = read_json(source/'coreset_0_indices.json')['train_positions']
    size = len(reference)
    cfg = manifest['configuration']
    train = read_typed(paths['train'], context['csv_schema'])
    groups, _ = strata(train.iloc[qpos][task['target']], task['problem_type'],
                       cfg['regression_bins'], max_strata=size)
    quota = split_quotas(np.bincount(groups), size, cfg['split_seed']+3)
    chosen = stratified_sample(groups, quota, seed)
    for arm, positions in [('coverage', np.asarray(reference)), ('random', np.sort(qpos[chosen]))]:
        folder = output/arm
        folder.mkdir(exist_ok=True)
        indices = {'train_positions': positions.tolist(), 'sampler': arm, 'seed': seed}
        freeze_json(folder/'coreset_indices.json', indices)
        if not (folder/'coreset.csv').exists():
            if arm == 'coverage':
                shutil.copyfile(source/'coreset_0.csv', folder/'coreset.csv')
            else:
                export_rows(paths['train'], [(folder/'coreset.csv', positions)])
    return {arm: output/arm for arm in ('random', 'coverage')}


def ranking_metrics(core_scores, full_scores, direction, *, epsilon_fraction=.05):
    """Epsilon is predeclared as 5% of the full-bank score range, not tuned per arm."""
    reward = compute_action_reward(core_scores, full_scores, direction=direction)
    ids = sorted(core_scores)
    core = np.asarray([core_scores[i] for i in ids], dtype=float)
    full = np.asarray([full_scores[i] for i in ids], dtype=float)
    sign = 1 if direction == 'maximize' else -1
    order = sorted(ids, key=lambda i: (-sign*core_scores[i], str(i)))
    utility = sign*full
    tolerance = 1e-12+1e-8*np.max(np.abs(full))
    regret = float(utility.max()-sign*full_scores[order[0]])
    epsilon = float(epsilon_fraction*np.ptp(full))
    near = {i for i in ids if utility.max()-sign*full_scores[i] <= epsilon+tolerance}
    def finite(value):
        return float(value) if math.isfinite(value) else None
    tau = kendalltau(core, full).statistic if np.ptp(core) and np.ptp(full) else float('nan')
    rho = spearmanr(core, full).statistic if np.ptp(core) and np.ptp(full) else float('nan')
    return {'weighted_accuracy': reward['weighted_accuracy'], 'gap_weighted_inversion_loss': reward['order_loss'],
        'kendall_tau': finite(tau), 'spearman_rho': finite(rho),
        'top1_hit': regret <= tolerance, 'top1_selection_loss': regret,
        'normalized_top1_loss': regret/float(np.ptp(full)) if np.ptp(full) > tolerance else 0.,
        'epsilon': epsilon, 'epsilon_fraction_of_full_range': epsilon_fraction,
        'epsilon_near_optimal_retention_top3': len(near.intersection(order[:3]))/len(near),
        'full_best_retained_top3': any(utility.max()-sign*full_scores[i] <= tolerance for i in order[:3]),
        'core_ranking': order, 'full_ranking': reward['full_ranking'],
        'core_scores': core_scores, 'full_scores': full_scores,
        'comparable_pair_count': reward['comparable_pair_count']}


async def low_only(source, output, *, model, model_identity, pricing, search):
    """No full fit or API generation; recover only locally verified low evaluations."""
    output = Path(output)
    backend = WorkflowBackend(source, 0, output/'execution', model=model, pricing=pricing)
    backend.identity['generation_model'] = model_identity
    freeze_json(output/'identity.json', backend.identity)
    paths = [backend.output/'candidates'/f'initial_{i}'/'candidate.json' for i in range(5)]
    if all(p.exists() for p in paths):
        candidates = [read_json(p) for p in paths]
    else:
        candidates = backend.initial_candidates()
    for i, candidate in enumerate(candidates):
        definition = backend.manifest['methods'][i]
        if candidate['code_sha256'] != definition['code_sha256'] or candidate['source_method_id'] != i:
            raise ValueError('Paired method bank identity mismatch')
        recovered = backend.recover_evaluation(candidate, 'low')
        if recovered is None:
            candidate.pop('low_report', None)
            await backend.evaluate(candidate, 'low')
    mcs = compare_candidates(backend, candidates, MCTSConfig.from_dict(search), search['bootstrap_seed'])
    result = {'candidates': candidates, 'mcs': mcs, 'test_access': False}
    write_json(output/'result.json', result)
    return result


async def screen(task, root, *, config, search, memory, model, model_identity, pricing, group, policy_path):
    root = Path(root)
    cfg = deepcopy(config)
    cfg['region_bandit'] = normalize_config({**cfg['region_bandit'], 'full_train_methods': 'all', 'baseline_audit': False})
    sampling = root/'sampling'
    prep = await run_region_task(task, sampling, config=cfg, model=model, model_identity=model_identity,
        pricing=pricing, memory_path=policy_path, dataset_group_id=group,
        resume=(sampling/'state.json').exists(), sampling_only=True)
    source = Path(prep['source'])
    bank = read_json(sampling/'method_bank.json')
    samples = base_samples(source, root/'samples')
    samples['bandit'] = Path(prep['arm'])
    views = {arm: source_view(source, path, root/'sources'/arm) for arm, path in samples.items()}
    initial_cfg = {**search, 'enabled': False}
    summary, engine = await run_search(views['bandit'], root/'bandit', model=model,
        model_identity=model_identity, pricing=pricing, search=initial_cfg, memory=memory,
        frozen_initial=True, resume=(root/'bandit/search_state.json').exists())
    if not summary['initialization_complete']:
        raise RuntimeError('Shared five-method bank did not finish all initial evaluations; see bandit search state')
    finish_region_feedback(sampling, engine, config=cfg, memory_path=policy_path)
    candidates = [engine.state['candidates'][i] for i in engine.state['initial_ids']]
    full_scores = {c['source_method_id']: c['full_score'] for c in candidates}
    full_path = root/'full_reference.json'
    write_json(full_path, {'method_bank_hash': bank['hash'], 'candidates': candidates,
                          'scope': 'Stage-one ranking reference only; never imported into MCTS', 'test_access': False})
    results = []
    decision = read_json(sampling/'decision.json')
    chosen_action = next(a for a in decision['actions'] if a['action_id'] == decision['chosen_action_id'])
    reference_rows = set(read_json(samples['coverage']/'coreset_indices.json')['train_positions'])
    for arm in ARMS:
        if arm == 'bandit':
            rows, mcs = candidates, engine.state['initial_mcs']
        else:
            low = await low_only(views[arm], root/arm, model=model, model_identity=model_identity,
                                 pricing=pricing, search=initial_cfg)
            rows, mcs = low['candidates'], low['mcs']
        if any(c['code_sha256'] != candidates[c['source_method_id']]['code_sha256'] or
               c['contract_sha256'] != candidates[c['source_method_id']]['contract_sha256'] for c in rows):
            raise ValueError('Coreset arms did not share exactly the same code and stage plans')
        metrics = ranking_metrics({c['source_method_id']: c['low_score'] for c in rows}, full_scores,
                                  metric_direction(engine.backend.task))
        sampled_rows = set(read_json(samples[arm]/'coreset_indices.json')['train_positions'])
        results.append({'arm': arm, **metrics, 'mcs': mcs, 'shared_full_reference': str(full_path),
            'coreset_rows': len(sampled_rows),
            'overlap_with_original_coverage': len(sampled_rows & reference_rows)/len(reference_rows),
            'chosen_action': chosen_action['type'] if arm == 'bandit' else arm})
        write_json(root/'comparison_partial.json', {'arms': results, 'test_access': False})
    result = {'status': 'completed', 'task': str(task), 'source': str(source),
        'method_bank_hash': bank['hash'], 'metric': engine.backend.metric,
        'direction': metric_direction(engine.backend.task), 'arms': results,
        'full_reference_fits': 5, 'coreset_fits': 15, 'test_access': False}
    write_json(root/'comparison.json', result)
    return result


def search_comparison(engine, arm):
    state, backend = engine.state, engine.backend
    initial = [state['candidates'][i] for i in state['initial_selected'] if state['candidates'][i].get('full_report')]
    sign = 1 if backend.maximize else -1
    first = max((c['full_score']*sign for c in initial), default=None)
    best = state['candidates'].get(engine.tree.global_best_id, {})
    rounds = []
    for batch in state['expansions']:
        parent = state['candidates'][batch['parent_id']]
        anchor = batch.get('region_anchor', parent)
        children = [state['candidates'][i] for i in batch['candidate_ids'] if state['candidates'][i].get('full_report')]
        if not anchor.get('low_report') or not children:
            rounds.append({'iteration': batch['iteration'], 'status': 'no_comparable_full_pair'})
            continue
        core = {parent['id']: anchor['low_score'], **{c['id']: c['low_score'] for c in children}}
        full = {parent['id']: parent['full_score'], **{c['id']: c['full_score'] for c in children}}
        reward = compute_action_reward(core, full, direction=metric_direction(backend.task),
                                       comparison_scope='mcts_parent_anchor', anchor_method_id=parent['id'])
        rounds.append({'iteration': batch['iteration'], **reward})
    return {'arm': arm, 'best_id': best.get('id'), 'best_validation_score': best.get('full_score'),
        'initial_best_validation_score': first*sign if first is not None else None,
        'validation_improvement': sign*best['full_score']-first if best.get('full_report') and first is not None else None,
        'full_successes': state['full_fit_successes'], 'full_attempts': state['full_fit_attempts'],
        'low_attempts': state['low_fit_attempts'], 'api_calls': state['api_calls'],
        'search_wall_seconds': state.get('elapsed_seconds'),
        'completed_expansions': len(state['expansions']), 'rounds': rounds,
        'ranking_scope': 'Observed parent plus one/two full-evaluated children; selection-biased, not all proposals',
        'initial_selected': state['initial_selected'], 'test_access': False}


async def search_arm(task, root, stage_one, arm, *, config, search, memory, model, model_identity, pricing, group, policy_path):
    root, stage_one = Path(root), Path(stage_one)
    previous = read_json(stage_one/'comparison.json')
    source = Path(previous['source'])
    controller, callback = None, None
    if arm == 'bandit':
        cfg = deepcopy(config)
        cfg['region_bandit'] = normalize_config({**cfg['region_bandit'], 'full_train_methods': 'top3', 'baseline_audit': False})
        sampling = root/'sampling'
        prep = await run_region_task(task, sampling, config=cfg, model=model, model_identity=model_identity,
            pricing=pricing, memory_path=policy_path, dataset_group_id=group,
            resume=(sampling/'state.json').exists(), sampling_only=True, bootstrap_source=source)
        if read_json(sampling/'method_bank.json')['hash'] != previous['method_bank_hash']:
            raise ValueError('MCTS did not reuse the shared five-method bank')
        view = source_view(source, prep['arm'], root/'source')
        controller = RoundBandit(sampling, config=cfg, memory_path=policy_path,
                                 model=model, model_identity=model_identity, pricing=pricing)
        def callback(engine):
            return finish_region_feedback(sampling, engine, config=cfg, memory_path=policy_path)
    else:
        view = stage_one/'sources'/arm
    summary, engine = await run_search(view, root/'search', model=model, model_identity=model_identity,
        pricing=pricing, search={**search, 'enabled': True}, memory=memory, frozen_initial=True,
        region_controller=controller, initial_feedback=callback, resume=(root/'search/search_state.json').exists())
    result = {'status': summary['status'], 'initialization_complete': summary['initialization_complete'],
              'task': str(task), 'method_bank_hash': previous['method_bank_hash'], **search_comparison(engine, arm)}
    write_json(root/'comparison.json', result)
    if not summary['initialization_complete']:
        raise RuntimeError('MCTS initialization failed: ' + arm)
    return result
