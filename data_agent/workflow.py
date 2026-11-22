"""Compose optional sampling and search stages without repeating initial fits."""
from copy import deepcopy
from dataclasses import asdict
import fcntl
from pathlib import Path
import shutil

from .coreset import file_hash
from .mcts_backend import MCTSBackend
from .mcts_policy import MCTSConfig
from .mcts_search import MCTSSearch
from .region_bandit.runtime import bootstrap_config
from .region_bandit.schemas import read_json, write_json, stable_hash


MODULE_DEFAULTS = {
    'region_selection': True, 'region_advisor': True, 'parameter_feedback': True,
    'mcts': True, 'mcs': True, 'search_memory': True, 'post_full_feedback': True,
}


def module_settings(values=None):
    values = {} if values is None else values
    if not isinstance(values, dict) or set(values) - set(MODULE_DEFAULTS):
        raise ValueError('Unknown workflow modules; expected ' + ', '.join(MODULE_DEFAULTS))
    if any(type(value) is not bool for value in values.values()):
        raise ValueError('Module switches must be YAML true/false booleans')
    return {**MODULE_DEFAULTS, **values}


def source_view(source, arm, output):
    """Point the existing search adapter at the one selected, immutable coreset."""
    source, arm, output = Path(source), Path(arm), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {**read_json(source/'manifest.json'), 'source': str(source.resolve())}
    artifacts = {'coreset_0.csv': arm/'coreset.csv',
                 'coreset_0_indices.json': arm/'coreset_indices.json'}
    for name, path in artifacts.items():
        destination = output/name
        if destination.exists():
            if file_hash(destination) != file_hash(path):
                raise ValueError('Selected search coreset changed: ' + name)
        else:
            shutil.copyfile(path, destination)
    target = output/'manifest.json'
    if target.exists() and read_json(target) != manifest:
        raise ValueError('Search source manifest changed')
    write_json(target, manifest)
    return output


class WorkflowBackend(MCTSBackend):
    async def _repair(self, candidate, error, stage):
        # The regional decision was based on exactly this frozen five-method bank.
        if candidate.get('origin') == 'source_initial':
            raise RuntimeError('Frozen initial methods cannot change after sampling') from error
        return await super()._repair(candidate, error, stage)


async def run_search(source, output, *, model, model_identity, pricing=None,
                     search=None, memory=None, subgroups=None, resume=False,
                     frozen_initial=False,region_controller=None,initial_feedback=None,
                     full_success_target=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = MCTSConfig.from_dict(search)
    with (output/'.search.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another search or final evaluation owns this directory') from error
        if (output/'final_test/selection_manifest.json').exists():
            raise ValueError('Final test selection is already frozen; search cannot resume')
        backend_class = WorkflowBackend if frozen_initial else MCTSBackend
        backend = backend_class(source, 0, output/'execution', model=model, pricing=pricing,
                                max_repairs=cfg.repair_attempts, subgroup_rules=subgroups)
        backend.identity['generation_model'] = deepcopy(model_identity)
        engine = MCTSSearch(backend, output, cfg, resume=resume, memory_options=memory,
                            region_controller=region_controller,initial_feedback=initial_feedback)
        summary = await engine.run(full_success_target=full_success_target) if full_success_target is not None else await engine.run()
    return summary, engine


async def run_plain_task(task_path, output, *, config, model, model_identity,
                         pricing=None, resume=False, search=None, memory=None, subgroups=None):
    """Base coverage: five full evaluations without MCTS, initial top3 with it."""
    from run_mcts_data_agent import bootstrap_with_repairs
    output = Path(output).resolve()
    cfg = bootstrap_config(config.get('multi_proxy'))
    source = await bootstrap_with_repairs(Path(task_path).resolve(), output/'bootstrap', cfg,
                                         model, model_identity, pricing, resume=resume)
    if search is None:
        search = {'enabled': False, 'mcs_enabled': config.get('multi_proxy', {}).get(
            'stage1_selection', {}).get('enabled', True)}
    summary, _ = await run_search(source, output/'search', model=model,
        model_identity=model_identity, pricing=pricing, search=search, memory=memory,
        subgroups=subgroups, resume=resume)
    result = {**summary, 'region_selection_enabled': False, 'policy_updated': False,
              'search_run': str(output/'search'), 'test_access': False}
    write_json(output/'result.json', result)
    return result


def finish_region_feedback(root, engine, *, config, memory_path):
    """Learn only from the frozen initial bank, never adaptively generated children."""
    import numpy as np
    from .region_bandit.config import normalize_config
    from .region_bandit.context import default_feature_schema
    from .region_bandit.memory_store import MemoryStore
    from .region_bandit.reward import compute_action_reward, aggregate_seed_rewards
    from .region_bandit.runtime import (_recover_feedback, _freeze_artifacts, _export_policy)
    from .stage1_metrics import metric_direction

    root = Path(root)
    cfg = normalize_config(config['region_bandit'])
    decision = read_json(root/'decision.json')
    frozen = cfg['mode'] == 'evaluate_frozen'
    initial = [engine.state['candidates'][i] for i in engine.state['initial_ids']]
    chosen = next(a for a in decision['actions'] if a['action_id'] == decision['chosen_action_id'])
    if not engine.state['initialization_complete']:
        return {'status': 'pending_initial_results', 'policy_updated': False}
    observed = [engine.state['candidates'][i] for i in engine.state['initial_selected']]
    expected = engine.cfg.initial_full_count
    if len(initial) != 5 or len(observed) != expected or cfg['full_train_methods'] != engine.cfg.full_train_methods:
        raise ValueError('Initial full evaluations do not match the frozen MCTS/all-five mode')
    if any(not c.get('full_report') for c in observed):
        return {'status': 'pending_initial_results', 'policy_updated': False,
                'expected_method_count': expected}
    core_scores = {c['source_method_id']: c['low_score'] for c in initial}
    full_scores = {c['source_method_id']: c['full_score'] for c in observed}
    if not all(np.isfinite(value) for value in [*core_scores.values(), *full_scores.values()]):
        raise ValueError('Nonfinite initial ranking feedback')
    # The sampling decision must still refer to the same implementations.
    bank = {m['method_id']: m for m in decision['frozen_config']['methods']}
    if any(c['code_sha256'] != bank[c['source_method_id']]['code_sha256'] for c in initial):
        raise ValueError('Initial method bank changed after the regional decision')
    with MemoryStore(memory_path, read_only=frozen) as store:
        saved, committed = _recover_feedback(root, store, decision, frozen)
        if saved is not None and set(saved['reward']['method_ids']) != set(full_scores):
            raise ValueError('Saved bandit feedback uses a different observed method set')
        if saved is None:
            reward = compute_action_reward(core_scores, full_scores, direction=metric_direction(engine.backend.task),
                                           config=cfg['reward'])
            seed = cfg['sampling_seeds'][0]
            feedback = {'action_id': chosen['action_id'], 'observed_method_count': expected,
                'method_ids': sorted(full_scores), **aggregate_seed_rewards([{'seed': seed, **reward}]),
                'seed_rewards': [{'seed': seed, **reward}],
                'scope': 'Frozen initial bank only; excludes every MCTS-generated candidate'}
            diagnostics = search_region_diagnostics(root, engine, observed, cfg, decision)
            write_json(root/'region_feedback.json', diagnostics)
            saved = {'decision_id': decision['decision_id'], 'reward': feedback,
                     'regions': diagnostics['regions']}
            write_json(root/'feedback_pending.json', saved)
        _freeze_artifacts(root, 'feedback', [root/'feedback_pending.json'])
        before = committed is not None
        committed = committed or store.commit_feedback(decision, saved['reward'], saved['regions'],
            mode=cfg['mode'], schema=default_feature_schema(), config=cfg)
        if not frozen:
            _export_policy(memory_path, default_feature_schema(), committed)
        write_json(root/'action_reward.json', saved['reward'])
        write_json(root/'action_feedback.json', saved['reward'])
        write_json(root/'feedback_commit.json', {'mode': cfg['mode'],
            'decision_id': decision['decision_id'], 'snapshot_id': committed.get('snapshot_id'),
            'updated': not frozen and not before and saved['reward']['update_eligible']})
        state = read_json(root/'state.json')
        state['phase'] = 'feedback_committed'
        write_json(root/'state.json', state)
        return {'status': 'completed', 'reward': saved['reward']['reward'],
                'reward_status': saved['reward']['status'],
                'policy_updated': not frozen and saved['reward']['update_eligible'],
                'observed_method_count': expected, 'chosen_action_type': chosen['type']}


def search_region_diagnostics(root, engine, observed, cfg, decision):
    """Reuse initial fitted models; optional diagnostics never add training fits."""
    result = {'status': 'disabled', 'regions': []}
    if not cfg['post_full_feedback']['enabled']:
        return result
    if not cfg['post_full_feedback']['predict_core_on_external_validation']:
        return {**result, 'status': 'prediction_disabled'}
    regions = read_json(root/'regions.json')['regions']
    if not regions:
        return {**result, 'status': 'no_regions'}
    import joblib
    import numpy as np
    from .multi_proxy_io import read_typed
    from .region_bandit.execution import predict_cached
    from .region_bandit.post_full_feedback import compute_region_feedback
    from .region_bandit.runtime import _load_predictions
    backend = engine.backend
    try:
        source = Path(read_json(root/'sampling_preparation.json')['source'])
        task = read_json(backend.task_path)
        external = read_typed(backend.task_path.parent/task['files']['validation'], backend.schema)
        transform = joblib.load(source/'similarity_transform.joblib')
        gmm = joblib.load(source/'density_gmm/fitted.joblib')
        components = np.concatenate([gmm.predict(transform.transform(external.iloc[i:i+backend.batch_size]))
                                     for i in range(0, len(external), backend.batch_size)])
        low, full, ids = [], [], []
        for candidate in observed:
            ident = candidate['source_method_id']
            folder = root/'core_external_predictions'/f'method_{ident}'
            outcome = predict_cached(Path(candidate['code_path']), candidate['low_report']['fitted_pipeline_path'],
                backend.external_features, backend.task_context, backend.classes, folder,
                timeout_seconds=backend.evaluation_budget['timeout_seconds'], batch_size=backend.batch_size,
                identity=decision['identity'])
            if outcome['status'] != 'ok':
                raise RuntimeError('Initial core prediction diagnostic failed: '+str(outcome.get('error')))
            low.append(_load_predictions(folder/'predictions.npz', backend.classes))
            full.append(backend.load_prediction(candidate['full_report']))
            ids.append(ident)
        diagnostics = compute_region_feedback(regions, external, backend.task_context, low, full, components,
            loss_scale=read_json(root/'regions.json')['loss_scale'], config=cfg, seed=cfg['decision_seed'],
            dependence=backend.dependency_evidence, method_ids=ids)
        originals = {r.get('region_id', r.get('id')): r for r in regions}
        for row in diagnostics['regions']:
            rid = row.get('region_id', row.get('id'))
            ctx = decision['region_contexts'][rid]
            result['regions'].append({**row, 'seed': cfg['sampling_seeds'][0], 'context': ctx,
                'region': {k: v for k, v in originals[rid].items() if k not in ('mask', 'indices')},
                'rule_types': ctx.get('rule_types'),
                'signal_type': 'specific_coreset_reversal_diagnostic_not_action_reward'})
        result['status'] = 'completed'
    except Exception as error:
        result.update(status='partial_or_failed', error=f'{type(error).__name__}: {error}')
    return result


async def run_workflow(task_path, output, *, config, search, memory, modules, model,
                       model_identity, memory_path, pricing=None, subgroups=None, resume=False,
                       dataset_group_id=None, snapshot_id=None, evaluation_manifest=None,
                       extend_wall_seconds=0):
    from .region_bandit.runtime import run_region_task
    modules = module_settings(modules)
    if (type(extend_wall_seconds) is not int or extend_wall_seconds < 0 or
            (extend_wall_seconds and not resume)):
        raise ValueError('A nonnegative execution budget extension requires --resume')
    if extend_wall_seconds and not modules['region_selection']:
        raise ValueError('Wall extensions are only supported by the regional sampling stage')
    search = MCTSConfig.from_dict(search)
    if (modules['region_selection'] and search.enabled
            and config['region_bandit'].get('mcts_rounds',{}).get('enabled',True)
            and config['region_bandit']['mode']=='evaluate_frozen'):
        raise ValueError('Existing frozen policies cover initialization only; set '
                         'region_bandit.mcts_rounds.enabled=false for evaluate_frozen')
    if (config['region_bandit']['enabled'] != modules['region_selection'] or
            config['region_bandit']['full_train_methods'] != search.full_train_methods or
            config['multi_proxy']['parameter_feedback'] != modules['parameter_feedback'] or
            search.enabled != modules['mcts'] or search.mcs_enabled != modules['mcs'] or
            memory.get('enabled', True) != modules['search_memory']):
        raise ValueError('Module switches do not match effective execution configuration')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = stable_hash({'task': file_hash(task_path), 'config': config,
        'search': MCTSConfig.from_dict(search).to_dict(), 'memory': memory, 'modules': modules,
        'model': model_identity, 'memory_path': str(Path(memory_path).resolve()) if modules['region_selection'] else None,
        'dataset_group_id': dataset_group_id, 'snapshot_id': snapshot_id,
        'evaluation_manifest': evaluation_manifest, 'subgroups': subgroups,
        'workflow_code': file_hash(Path(__file__))})
    path = root/'workflow_manifest.json'
    if path.exists():
        if not resume or read_json(path)['fingerprint'] != fingerprint:
            raise ValueError('Workflow requires --resume with unchanged modules, configuration and inputs')
    else:
        if (root/'search/search_state.json').exists() or (root/'sampling/state.json').exists():
            raise ValueError('Existing artifacts have no matching workflow manifest; choose a new output directory')
        write_json(path, {'fingerprint': fingerprint, 'modules': modules, 'test_access': False})
    if not modules['region_selection']:
        result = await run_plain_task(task_path, root, config=config, model=model,
            model_identity=model_identity, pricing=pricing, resume=resume,
            search=search, memory=memory, subgroups=subgroups)
    else:
        region_root = root/'sampling'
        region_root.mkdir(exist_ok=True)
        with (region_root/'.region_bandit.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError('Another process owns the regional sampling stage') from error
            preparation = await run_region_task(task_path, region_root, config=config, model=model,
                model_identity=model_identity, pricing=pricing, memory_path=memory_path,
                dataset_group_id=dataset_group_id, snapshot_id=snapshot_id,
                evaluation_manifest=evaluation_manifest, resume=resume, sampling_only=True,
                extend_wall_seconds=extend_wall_seconds)
            if preparation.get('status') != 'sampled':
                return preparation
            source = source_view(preparation['source'], preparation['arm'], root/'selected_source')
            controller=None
            if search.enabled and config['region_bandit'].get('mcts_rounds',{}).get('enabled',True):
                from .region_bandit.mcts_rounds import RoundBandit
                controller=RoundBandit(region_root,config=config,memory_path=memory_path,
                    model=model,model_identity=model_identity,pricing=pricing)
            initial_feedback=(lambda engine: finish_region_feedback(
                region_root,engine,config=config,memory_path=memory_path)) if controller else None
            summary, engine = await run_search(source, root/'search', model=model,
                model_identity=model_identity, pricing=pricing, search=search, memory=memory,
                subgroups=subgroups, resume=resume, frozen_initial=True,
                region_controller=controller,initial_feedback=initial_feedback)
            feedback = finish_region_feedback(region_root, engine, config=config, memory_path=memory_path)
            result = {**summary, 'region_selection_enabled': True, 'bandit_feedback': feedback,
                      'search_run': str(root/'search'), 'test_access': False}
    result['modules'] = modules
    write_json(root/'result.json', result)
    return result
