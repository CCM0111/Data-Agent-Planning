"""Independent regional sampling experiment; delayed feedback, no search or test access."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
from scipy import sparse

from ..coreset import file_hash, export_rows
from ..multi_proxy import MultiProxyConfig, load_fixed, evaluation_metrics
from ..multi_proxy_io import read_typed
from ..multi_proxy_feedback import frozen_fit_context
from ..execution_budget import (TaskDeadlineExceeded, deadline_scope, check_deadline,
                                await_with_budget, call_with_budget, start_wall_accounting, account_wall_time)
from ..stage1_metrics import metric_direction
from ..prepared_protocol import validate_prepared_partition
from ..stage1_selection import analyze_run, compare_full_stage, _dependency_evidence
from ..marginal_coverage import coverage_sample
from .config import normalize_config, reward_protocol_hash, full_train_count
from .schemas import (VERSIONS, clean, stable_hash, read_json, write_json,
                      environment_identity, dataset_group, checkpoint)
from .context import default_feature_schema, canonical_metric, build_context
from .regions import build_regions, apply_rule
from .action_planner import plan_actions
from .regional_sampler import sample_from_action
from .memory_store import MemoryStore
from .memory_retrieval import retrieve_memory
from .policy import select_action, derive_decision_seed
from .llm_advisor import score_actions_with_api
from .reward import compute_action_reward, rank_method_ids, aggregate_seed_rewards
from .post_full_feedback import compute_region_feedback

SAMPLER = {'a': .3, 'beta': .6, 'gamma': 2., 'rho': 1.}


def _verify_artifacts(root):
    for manifest in (root/'integrity').glob('*.json'):
        for filename, expected in read_json(manifest).items():
            if not Path(filename).is_file() or file_hash(filename) != expected:
                raise ValueError('Frozen experiment artifact changed: '+filename)


def _freeze_artifacts(root, name, files):
    path = root/'integrity'/f'{name}.json'
    hashes = {str(Path(f).resolve()): file_hash(f) for f in files}
    if path.exists() and read_json(path) != hashes:
        raise ValueError('Frozen '+name+' artifacts changed')
    write_json(path, hashes)


def _export_policy(memory_path, schema, snapshot):
    """Readable mirrors of transactional records; serialize concurrent exporters."""
    directory = Path(memory_path).resolve().parent
    with (directory/'.policy_export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory/'feature_schema.json'
        if path.exists() and read_json(path) != schema:
            raise ValueError('Global policy directory has a different feature schema; choose a new directory')
        if not path.exists():
            write_json(path, schema)
        write_json(directory/'policy_snapshots'/f"{snapshot['snapshot_id']}.json", snapshot)


def bootstrap_config(values):
    values = deepcopy(values or {})
    if 'lambda' in values:
        values['similarity_lambda'] = values.pop('lambda')
    values.setdefault('parameter_feedback', True)
    values.setdefault('method_preflight', True)
    values.update(n_strategies=5, n_coresets=1, gmm_components=5,
                  full_train_enabled=True, defer_full_train=True, mcts_bootstrap_only=True,
                  proxy_selection_scheme='marginal_coverage_only')
    values['stage1_selection'] = {**values.get('stage1_selection', {}),
        'enabled': False, 'mode': 'record_only', 'mcts_admission': False, 'coreset_weights': [1.]}
    result = MultiProxyConfig(**values)
    result.validate()
    from ..guarded_proxy_feedback import effective_config
    return effective_config(result)


def evaluation_protocol(config, model_identity):
    base = Path(__file__).parents[1]
    return {'multi_proxy': asdict(bootstrap_config(config.get('multi_proxy'))),
            'region_bandit': normalize_config(config.get('region_bandit')),
            'model_identity': model_identity,
            'algorithm_hashes': {name: file_hash(base/name) for name in
                ('region_bandit/regions.py', 'region_bandit/action_planner.py',
                 'region_bandit/regional_sampler.py', 'region_bandit/gentle_cover.py', 'region_bandit/local_edit.py', 'region_bandit/exchange_coverage.py', 'region_bandit/agent_policy.py', 'region_bandit/advisor_repair.py', 'region_bandit/diagnostics.py',
                 'region_bandit/post_full_feedback.py', 'region_bandit/reward.py',
                 'region_bandit/policy.py', 'region_bandit/memory_retrieval.py',
                 'region_bandit/runtime.py', 'marginal_coverage.py')},
            'prompt_hashes': {name: file_hash(base/name) for name in
                ('multi_proxy.py', 'parameter_feedback.py', 'multi_proxy_feedback.py', 'guarded_proxy_feedback.py',
                 'region_bandit/llm_advisor.py', 'region_bandit/context.py')}}


def validate_evaluation_manifest(value, group):
    manifest = read_json(value) if isinstance(value, (str, Path)) else deepcopy(value)
    if not isinstance(manifest, dict):
        raise ValueError('evaluate_frozen requires a predeclared evaluation split manifest')
    development = manifest.get('development_dataset_groups')
    evaluation = manifest.get('evaluation_dataset_groups')
    if (not isinstance(development, list) or not isinstance(evaluation, list)
            or any(not isinstance(v, str) or not v for v in development + evaluation)
            or len(set(development)) != len(development) or len(set(evaluation)) != len(evaluation)):
        raise ValueError('Evaluation manifest requires distinct source dataset group lists')
    if set(development) & set(evaluation) or group not in evaluation:
        raise ValueError('Evaluation groups must be held out and include the current source group')
    snapshots = manifest.get('snapshot_ids')
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError('Frozen evaluation requires snapshot_ids from the exported evaluation manifest')
    ids = [item.get('snapshot_id') if isinstance(item, dict) else item for item in snapshots]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError('Frozen evaluation snapshot_ids must be distinct nonempty IDs')
    protocol = manifest.get('evaluation_protocol')
    if not isinstance(protocol, dict) or not {'multi_proxy', 'region_bandit', 'model_identity', 'prompt_hashes', 'algorithm_hashes'} <= set(protocol):
        raise ValueError('Frozen evaluation requires the complete exported evaluation_protocol')
    return manifest


def _reward_resolved(feedback):
    return (feedback.get('reward') is not None or
            (feedback.get('status') == 'reward_uninformative' and not feedback.get('update_eligible', False)))


def _required_results_complete(full_complete, rewards, baseline_audit):
    return bool(full_complete and _reward_resolved(rewards['chosen']) and
                (not baseline_audit or _reward_resolved(rewards.get('baseline', {}))))


def _full_evaluation_plan(methods, core_records, task, cfg):
    count = full_train_count(cfg)
    if count == 5:
        ids, ranking = [m['method_id'] for m in methods], None
        reason = ('explicit_all' if cfg['full_train_methods'] == 'all' else
                  'baseline_audit' if cfg['baseline_audit'] else 'shared_reference_for_repeats')
    else:
        scores = {r['method_id']: r.get('metrics', {}).get(task['metric']) for r in core_records}
        if (len(scores) != 5 or set(scores) != {m['method_id'] for m in methods}
                or any(r['status'] != 'ok' for r in core_records)
                or any(v is None or not np.isfinite(v) for v in scores.values())):
            return None
        ranking = rank_method_ids(scores, direction=metric_direction(task), config=cfg['reward'])
        ids, reason = sorted(ranking[:3]), 'coreset_top3'
    return {'requested': cfg['full_train_methods'], 'method_ids': ids, 'method_count': len(ids),
            'reason': reason, 'core_ranking': ranking,
            'selection_source': 'V_core only' if count == 3 else 'predeclared complete method bank'}


def _task_decision_seed(base_seed, group, task, source):
    # General experiment identities also contain held-out file hashes. They are
    # suitable for integrity checks, never for determining a sampling decision.
    return derive_decision_seed(base_seed, {'dataset_group_id': group, 'target': task['target'],
        'P_sha256': file_hash(source/'proxy_train.csv'), 'Q_features_sha256': file_hash(source/'Q_features.csv')})


def _recover_feedback(root, store, decision, frozen):
    """Prefer the committed event, then the immutable write-ahead payload."""
    committed = None if frozen else store.get_committed_feedback(decision['decision_id'], decision=decision)
    path = root/'feedback_pending.json'
    pending = read_json(path) if path.exists() else None
    if pending and pending.get('decision_id') != decision['decision_id']:
        raise ValueError('Pending feedback belongs to a different frozen decision')
    if committed:
        event = committed['event']
        return {'decision_id': decision['decision_id'], 'reward': event['feedback'],
                'regions': event['region_feedback']}, committed['snapshot']
    return pending, None


def _load_predictions(path, classes):
    with np.load(path, allow_pickle=False) as data:
        pred = data['predictions'].copy()
        probabilities = data['probabilities'].copy() if 'probabilities' in data else None
    if classes:
        mapping = {str(v): v for v in classes}
        pred = np.asarray([mapping[str(v)] for v in pred])
    return {'predictions': pred, 'probabilities': probabilities, 'classes': classes}


def _evaluation(result, folder, task, labels, classes, method, *, core=False, manifest=None):
    record = {'method_id': method['method_id'], 'method_name': method['name'],
              'metric': task['metric'], 'status': result['status'], 'error': result.get('error'),
              'execution': {k: result.get(k) for k in ('cache_key', 'cache_hit', 'attempts')}}
    if core:
        record['coreset_id'] = 0
    if result['status'] == 'ok':
        record.update(result['report'])
        record['fit_report_status'] = record.get('status')
        record['status'] = result['status']
        pred = _load_predictions(folder / 'predictions.npz', classes)
        if len(pred['predictions']) != len(labels):
            raise ValueError('Prediction row alignment failed')
        metrics, reason = evaluation_metrics(task, labels, pred['predictions'], pred['probabilities'], classes)
        record.update(metrics=metrics, primary_metric_reason=reason,
                      primary_metric_status='undefined' if reason else 'defined')
    record['evaluation_set'] = 'V_core' if core else 'external_validation'
    if core:
        record.update(evaluation_indices='V_core_indices.json',
            evaluation_labels_sha256=manifest['evaluation_set']['labels_sha256'],
            evaluation_indices_sha256=manifest['evaluation_set']['indices_sha256'])
    write_json(folder / 'evaluation.json', record)
    return record


def _summary(task, context, frame, p_rows, core_rows, proxy_predictions):
    columns = context['feature_columns']
    target = frame[task['target']]
    classification = task['problem_type'] == 'classification'
    imbalance = float(target.value_counts(normalize=True).min()) if classification else None
    # Label imbalance is 1-minority/majority, not a validation statistic.
    if classification:
        counts = target.value_counts()
        imbalance = float(1 - counts.min()/counts.max())
        disagreement = float(np.mean(np.std(np.stack([p['probabilities'] for p in proxy_predictions]), axis=0)))
    else:
        stack = np.stack([p['predictions'].astype(float) for p in proxy_predictions])
        scale = max(float(np.std(target.astype(float))), 1e-12)
        disagreement = float(np.clip(np.mean(np.std(stack, axis=0))/scale, 0, 1))
    tail = None
    if not classification:
        y = target.to_numpy(float)
        q1, q3 = np.quantile(y, [.25, .75])
        tail = float(np.mean((y < q1-1.5*(q3-q1)) | (y > q3+1.5*(q3-q1))))
    return {k: v for k, v in {
        'problem_type': task['problem_type'], 'metric': task['metric'],
        'metric_params': task.get('metric_params', {}), 'n_rows': len(frame),
        'n_features': len(columns), 'categorical_fraction': len(context['categorical_columns'])/max(1,len(columns)),
        'missing_fraction': float(frame[columns].isna().to_numpy().mean()),
        'p_fraction': p_rows/len(frame), 'coreset_fraction': core_rows/len(frame),
        'target_imbalance': imbalance, 'target_tail_fraction': tail,
        'proxy_disagreement': disagreement, 'scope': 'P + Q only; never V_core or external validation',
    }.items() if v is not None}


def _collect_costs(root, repeats, audited, full_count=5):
    boot_manifests = [read_json(p) for p in (root/'bootstrap').rglob('manifest.json')]
    boot_manifests = [m for m in boot_manifests if m.get('method') == 'multi_proxy_data_agent']
    calls = [c for m in boot_manifests for c in m.get('llm_calls', [])]
    advisor_usage = root/'advisor/response.usage.json'
    if advisor_usage.exists():
        calls.append(read_json(advisor_usage))
    # Every execution attempt has exactly one durable context. Preflight has no fit.
    formal_attempts = list(root.glob('arms/**/attempts/*/context.json')) + list(root.glob('full_train/**/attempts/*/context.json'))
    bootstrap_attempts = list((root/'bootstrap').glob('**/proxies/*/context.json'))
    completed = [p for p in formal_attempts + bootstrap_attempts if (p.parent/'fit_report.json').exists()]
    evaluations = list(root.glob('arms/**/evaluation.json')) + list(root.glob('full_train/**/evaluation.json'))
    completed_reports = [read_json(p.parent/'fit_report.json') for p in completed]
    return {'theoretical_predictive_fits': 5+full_count+5*repeats+(5*repeats if audited else 0),
        'fit_attempts': len(formal_attempts)+len(bootstrap_attempts), 'successful_predictive_fits': len(completed),
        'proxy_fit_attempts': len(bootstrap_attempts), 'formal_fit_attempts': len(formal_attempts),
        'gmm_fits': sum(m.get('gmm_fits_completed', 0) for m in boot_manifests),
        'cache_hits_in_last_evaluation_pass': sum(bool(read_json(p).get('execution', {}).get('cache_hit')) for p in evaluations),
        'formal_retry_attempts': len(formal_attempts)-len({str(p.parent.parent) for p in formal_attempts}),
        'failed_or_interrupted_fit_attempts': len(formal_attempts)+len(bootstrap_attempts)-len(completed),
        'completed_fit_process_wall_seconds': sum(r.get('process_wall_seconds', 0) for r in completed_reports),
        'bootstrap_wall_seconds': sum(m.get('timings_seconds', {}).get('total_wall', 0) for m in boot_manifests),
        'api_calls': len(calls), 'prompt_tokens': sum(c.get('usage', {}).get('prompt_tokens', 0) for c in calls),
        'completion_tokens': sum(c.get('usage', {}).get('completion_tokens', 0) for c in calls),
        'estimated_cost': sum(c['estimated_cost'] for c in calls) if calls and all(c.get('estimated_cost') is not None for c in calls) else None,
        'prediction_only_calls': len(list((root/'core_external_predictions').glob('**/worker_context.json'))),
        'scope': 'Actual saved execution attempts, including failed bootstrap repair attempts; no test evaluation'}


async def run_region_task(task_path, output, *, config, model, model_identity, pricing=None,
                          memory_path, dataset_group_id=None, snapshot_id=None,
                          evaluation_manifest=None, resume=False, extend_wall_seconds=0,
                          sampling_only=False, bootstrap_source=None, historical_references=None,
                          procedural=None):
    """Run one immutable decision. Caller serializes use of the output directory."""
    from run_mcts_data_agent import bootstrap_with_repairs  # initialization only; never constructs a search
    from .execution import fit_cached, predict_cached
    from ..multi_proxy import _llm_json_once

    started = time.monotonic()
    task_path, root = Path(task_path).resolve(), Path(output).resolve()
    cfg = normalize_config(config.get('region_bandit'))
    if procedural is not None:
        from .procedural_skills import validate_bundle
        validate_bundle(procedural['bundle'])
        if (not sampling_only or cfg['mode'] != 'develop_online'
                or cfg['sampling']['mode'] != 'resample' or cfg['sampling']['resample_control']
                or cfg['advisor']['decision_mode'] != 'agent'):
            raise ValueError('Procedural selection requires sampling-only Agent resampling')
    if type(sampling_only) is not bool:
        raise ValueError('sampling_only must be boolean')
    if historical_references is not None and (not sampling_only or bootstrap_source is None):
        raise ValueError('Historical reuse requires sampling-only and an explicit frozen source')
    if cfg['reward']['objective'] == 'paired_delta' and historical_references is None:
        raise ValueError('Paired delta rewards require historical reference coreset evaluations')
    if cfg['sampling']['resample_control'] and historical_references is None:
        raise ValueError('Resampling control requires historical replay')
    if sampling_only and ((cfg['coreset_repeats'] != 1 and historical_references is None) or cfg['baseline_audit']):
        raise ValueError('Sampling handoff requires one coreset and no baseline audit')
    if not cfg['enabled']:
        if (root/'state.json').exists():
            raise ValueError('Regional module changed for an existing run; use a new output directory')
        if snapshot_id or evaluation_manifest is not None or cfg['mode'] == 'evaluate_frozen':
            raise ValueError('A disabled region module cannot use a frozen bandit evaluation protocol')
        if extend_wall_seconds:
            raise ValueError('Wall extensions are only supported by the regional execution path')
        from ..workflow import run_plain_task
        return await run_plain_task(task_path, root, config=config, model=model,
            model_identity=model_identity, pricing=pricing, resume=resume)
    if (root/'search/search_state.json').exists():
        raise ValueError('Existing base workflow cannot become a region run; use a new output directory')
    if type(extend_wall_seconds) is not int or extend_wall_seconds < 0 or (extend_wall_seconds and not resume):
        raise ValueError('A nonnegative execution budget extension requires --resume')
    bootcfg = bootstrap_config(config.get('multi_proxy'))
    # Paired experiments may reuse definitions/P-Q feedback, never validation scores.
    shared_bootstrap = None
    if bootstrap_source is not None:
        from run_mcts_data_agent import validate_bootstrap_resume
        shared_bootstrap = Path(bootstrap_source).resolve()
        shared_manifest = read_json(shared_bootstrap/'manifest.json')
        if (shared_manifest.get('status') != 'success' or
                Path(shared_manifest['task_path']).resolve() != task_path):
            raise ValueError('Shared bootstrap must be successful and belong to this task')
        if historical_references is None:
            validate_bootstrap_resume(task_path, bootcfg, model_identity, shared_manifest)
        else:
            from .historical_replay import validate_historical_source
            validate_historical_source(task_path, shared_bootstrap, historical_references, cfg['sampling_seeds'])
    task, split, paths, hashes, _ = load_fixed(task_path)
    del _
    external_path = (task_path.parent/task['files']['validation']).resolve()
    hashes['validation'] = file_hash(external_path)
    if split.get('sha256', {}).get(external_path.name) != hashes['validation']:
        raise ValueError('Frozen external validation file changed or lacks a fingerprint')
    group = dataset_group(task, dataset_group_id)
    schema = default_feature_schema()
    metric, params, signature = canonical_metric(task)
    identity = {**VERSIONS, 'feature_schema_version': schema['version'],
        'dataset_group_id': group, 'dataset_version': stable_hash(hashes),
        'split_fingerprint': split['split_id'], 'problem_type': task['problem_type'],
        'metric': metric, 'metric_params': params, 'metric_signature': signature,
        'reward_protocol_hash': reward_protocol_hash(cfg)}
    if cfg['sampling']['mode'] == 'local_edit':
        from .local_edit import local_sampler_version
        if historical_references is None or not sampling_only:
            raise ValueError('Local editing currently requires historical sampling-only replay')
        identity['sampler_version'] = local_sampler_version(cfg)
    if cfg['advisor']['decision_mode'] == 'agent':
        from .agent_policy import AGENT_POLICY_VERSION
        identity['policy_version'] = AGENT_POLICY_VERSION
    if procedural is not None:
        from .procedural_skills import VERSION
        identity['policy_version'] = VERSION
    frozen = cfg['mode'] == 'evaluate_frozen'
    heldout = validate_evaluation_manifest(evaluation_manifest, group) if frozen else None
    if frozen and not snapshot_id:
        raise ValueError('evaluate_frozen requires an explicit frozen snapshot ID')
    if frozen:
        allowed = [v.get('snapshot_id') if isinstance(v, dict) else v for v in heldout['snapshot_ids']]
        if snapshot_id not in allowed:
            raise ValueError('Snapshot ID is not bound to the predeclared evaluation manifest')
    protocol = evaluation_protocol(config, model_identity)
    if frozen and heldout['evaluation_protocol'] != protocol:
        raise ValueError('Frozen evaluation model, prompts or configuration changed from the declared protocol')
    fingerprint = stable_hash({'task': file_hash(task_path), 'inputs': hashes, 'split': split,
        'config': cfg, 'bootstrap': asdict(bootcfg), 'model': model_identity,
        'group': group, 'snapshot': snapshot_id, 'heldout': heldout, 'protocol': protocol,
        'environment': environment_identity(), 'sampling_only': sampling_only})
    if procedural is not None:
        fingerprint = stable_hash({'base': fingerprint, 'procedural': procedural})
    if shared_bootstrap is not None:
        fingerprint = stable_hash({'base': fingerprint, 'bootstrap_source': str(shared_bootstrap),
                                   'manifest_sha256': file_hash(shared_bootstrap/'manifest.json')})
    root.mkdir(parents=True, exist_ok=True)
    if historical_references is not None:
        fingerprint = stable_hash({'base': fingerprint, 'historical_references': historical_references})
    state = read_json(root/'state.json') if (root/'state.json').exists() else {}
    if state and (not resume or state.get('fingerprint') != fingerprint):
        raise ValueError('Existing region run requires --resume with unchanged data, configuration, model and identity')
    _verify_artifacts(root)
    if state.get('wall_session'):
        account_wall_time(state, recover=True)
        write_json(root/'state.json', state)
    store = MemoryStore(memory_path, read_only=frozen)
    audited = False
    wall_limit = float(state.get('wall_budget_seconds', cfg['runtime']['max_wall_seconds']))
    if extend_wall_seconds:
        if not state:
            store.close()
            raise ValueError('Execution budget can only be extended for an existing run')
        wall_limit += extend_wall_seconds
        state.setdefault('budget_extensions', []).append({'seconds': extend_wall_seconds,
            'at': datetime.now(timezone.utc).isoformat(), 'wall_budget_seconds': wall_limit})
    if state:
        state['wall_budget_seconds'] = wall_limit
    deadline = started + max(0., wall_limit - float(state.get('wall_seconds', 0)))
    scope = deadline_scope(deadline)
    scope.__enter__()
    def budget_available():
        return time.monotonic() < deadline
    try:
        if state:
            snapshot = read_json(root/'policy_snapshot.json')
        else:
            snapshot = store.get_snapshot(snapshot_id) if snapshot_id else store.latest_snapshot(
                identity, schema=schema, config={**cfg['policy'], **cfg['memory']})
        if historical_references is not None and not state and not frozen:
            from .policy import rebuild_snapshot
            events = [e for e in store.events(identity) if e['dataset_group_id'] != group]
            from .memory_retrieval import compatibility
            snapshot = rebuild_snapshot(events, schema=schema, config={**cfg['policy'], **cfg['memory']},
                                        identity=compatibility(identity))
            store.save_snapshot(snapshot, identity)
        if frozen:
            store.validate_frozen(snapshot, identity, heldout['evaluation_dataset_groups'])
            events = store.events(identity, snapshot=snapshot)
            if (set(snapshot.get('training_dataset_groups', [])) - set(heldout['development_dataset_groups'])
                    or any(e['dataset_group_id'] not in heldout['development_dataset_groups'] for e in events)):
                raise ValueError('Frozen snapshot or memory contains a source outside the declared development split')
        else:
            _export_policy(memory_path, schema, snapshot)
        if state.get('phase') == 'feedback_committed':
            if sampling_only:
                return read_json(root/'sampling_preparation.json')
            result = read_json(root/'result.json')
            # Older runs could consume chosen feedback before a required audit finished.
            if (not cfg['baseline_audit'] or result.get('baseline_reward') is not None
                    or result.get('baseline_reward_status') == 'reward_uninformative'):
                return result
        if not state:
            state = {'fingerprint': fingerprint, 'mode': cfg['mode'], 'created_at': datetime.now(timezone.utc).isoformat(), 'wall_seconds': 0.}
            state['wall_budget_seconds'] = wall_limit
            start_wall_accounting(state, started)
            write_json(root/'policy_snapshot.json', snapshot)
            write_json(root/'config.json', {'region_bandit': cfg, 'multi_proxy': asdict(bootcfg), 'model': model_identity})
            write_json(root/'feature_schema.json', schema)
            if heldout:
                write_json(root/'evaluation_split_manifest.json', heldout)
            _freeze_artifacts(root, 'preparation', [root/'policy_snapshot.json', root/'config.json',
                root/'feature_schema.json'] + ([root/'evaluation_split_manifest.json'] if heldout else []))
            checkpoint(root, state, 'prepared')
        state['wall_budget_seconds'] = wall_limit
        if not state.get('wall_session'):
            start_wall_accounting(state, started)
        write_json(root/'state.json', state)
        check_deadline()
        source = shared_bootstrap
        if source is None:
            source = await await_with_budget(bootstrap_with_repairs(task_path, root/'bootstrap', bootcfg, model,
                                                  model_identity, pricing, resume=resume))
        source = Path(source).resolve()
        manifest, context = read_json(source/'manifest.json'), read_json(source/'task_context.json')
        effective_cfg = MultiProxyConfig(**manifest['configuration'])
        methods = read_json(source/'methods/methods.json')
        if isinstance(methods, dict):
            methods = methods['methods']
        if len(methods) != 5:
            raise ValueError('Region mode requires exactly five frozen methods')
        plan_path = source/'parameter_feedback/frozen_plan.json'
        plan = read_json(plan_path) if effective_cfg.parameter_feedback else None
        bank = []
        for i, method in enumerate(methods):
            method['method_id'] = i
            item = (next(p for p in plan['methods'] if p['method_id'] == i) if plan else
                    {'stage_parameters': method['idea']['stage_parameters'], 'pipeline_artifacts': {}})
            bank.append(deepcopy({k: method[k] for k in ('method_id', 'name', 'family', 'idea')}))
            bank[-1]['idea']['stage_parameters'] = deepcopy(item['stage_parameters'])
            if item.get('reason'):
                bank[-1]['idea']['adaptation_reason'] = item['reason']
            bank[-1].update(code_sha256=file_hash(method['code_path']), stage_parameters=item['stage_parameters'],
                frozen_pipelines={p: a['sha256'] for p, a in item['pipeline_artifacts'].items()})
        identity['method_bank_hash'] = stable_hash(bank)
        bank_context = {'methods': bank, 'formal_budgets': {
            'coreset': effective_cfg.evaluation_budget, 'full_train': effective_cfg.full_train_budget}}
        identity['internal_split_fingerprint'] = manifest['internal_split_id']
        if (root/'task_identity.json').exists() and read_json(root/'task_identity.json') != identity:
            raise ValueError('Frozen methods or training identity changed on resume')
        write_json(root/'task_identity.json', identity)
        write_json(root/'method_bank.json', {'methods': bank, 'hash': identity['method_bank_hash'],
            'source': str(source), 'formal_budgets': {'coreset': effective_cfg.evaluation_budget, 'full_train': effective_cfg.full_train_budget}})
        write_json(root/'frozen_methods.json', read_json(root/'method_bank.json'))
        _freeze_artifacts(root, 'methods', [root/'task_identity.json', root/'method_bank.json',
            *([plan_path] if plan else []), source/'task_context.json',
            source/'coverage_selection_inputs.npz', source/'similarity_graph.npz',
            source/'similarity_transform.joblib', source/'density_gmm/fitted.joblib',
            source/'core_validation.csv', source/'external_validation_features.csv',
            *[source/f'{part}_indices.json' for part in ('P','Q','V_core')],
            source/'coreset_0_selection_order.json', *[Path(m['code_path']) for m in methods],
            *[Path(a['path']) for item in (plan['methods'] if plan else []) for a in item['pipeline_artifacts'].values()],
            *[source/f'proxies/method_{i}/{name}' for i in range(5) for name in ('predictions.npz','fitted.joblib')]])
        checkpoint(root, state, 'methods_frozen', source=str(source))
        train = call_with_budget(read_typed, paths['train'], context['csv_schema'])
        qpos = np.asarray(read_json(source/'Q_indices.json')['train_positions'], int)
        ppos = np.asarray(read_json(source/'P_indices.json')['train_positions'], int)
        validate_prepared_partition(task, task_path, split, ppos,
            read_json(source/'V_core_indices.json')['train_positions'], qpos)
        qframe = train.iloc[qpos].reset_index(drop=True)
        classes = context['classes']
        with np.load(source/'coverage_selection_inputs.npz', allow_pickle=False) as data:
            inputs = {k: data[k].copy() for k in data.files}
        if not np.array_equal(inputs['train_positions'], qpos):
            raise ValueError('Coverage input row IDs do not match Q')
        graph = sparse.load_npz(source/'similarity_graph.npz')
        references = {}
        for seed in cfg['sampling_seeds']:
            if historical_references is not None:
                item = historical_references[str(seed)]
                refs = np.asarray(read_json(source/f"coreset_{item['source_coreset_id']}_selection_order.json")['q_indices'], int)
            elif seed == 20261026:
                refs = np.asarray(read_json(source/'coreset_0_selection_order.json')['q_indices'], int)
            else:
                refs, _ = call_with_budget(coverage_sample, inputs['density'], inputs['hardness'], graph,
                    inputs['groups'], inputs['quota'], seed, 'probability',
                    gamma=2., rho=1., beta=.6, density_coefficient=.3)
            references[seed] = refs
        reference_seed = cfg['sampling_seeds'][0] if historical_references is not None else 20261026
        size = len(references[reference_seed])
        write_json(root/'reference_coresets.json', {'sampler': SAMPLER, 'reference_seed': reference_seed,
            'seeds': {str(s): {'q_indices': r.tolist(), 'train_positions': qpos[r].tolist(),
                'group_counts': np.bincount(inputs['groups'][r], minlength=len(inputs['quota'])).tolist()}
                for s, r in references.items()}})
        write_json(root/'reference_coreset_indices.json', read_json(root/'reference_coresets.json'))
        proxies = [_load_predictions(source/f'proxies/method_{i}/predictions.npz', classes) for i in range(5)]
        if (root/'regions.json').exists():
            discovered = read_json(root/'regions.json')
            regions = discovered['regions']
            for region in regions:
                region['mask'] = apply_rule(region['rule'], qframe, inputs['components'])
                if np.flatnonzero(region['mask']).tolist() != region['indices']:
                    raise ValueError('Frozen region mask changed')
        else:
            importances = []
            try:
                from .execution import inspect_proxy_importance
                for i, method in enumerate(methods):
                    measured = inspect_proxy_importance(Path(method['code_path']), source/f'proxies/method_{i}/fitted.joblib',
                        context['feature_columns'], out=root/'proxy_importance'/f'method_{i}')
                    if measured:
                        importances.append({v['feature']: v['importance'] for v in measured})
            except (ImportError, AttributeError):
                pass
            discovered = call_with_budget(build_regions, qframe, context, proxies, references[reference_seed],
                inputs['components'], inputs['density'], inputs['hardness'], config=cfg,
                seed=cfg['decision_seed'], feature_importances=importances or None,
                row_ids=qpos.tolist())
            regions = discovered['regions']
            write_json(root/'regions.json', {**discovered, 'regions': [{k: v for k, v in r.items() if k != 'mask'} for r in regions]})
        write_json(root/'region_discovery.json', {**discovered['discovery'], 'loss_scale': discovered['loss_scale']})
        write_json(root/'region_specs.json', [{k: v for k, v in r.items() if k != 'mask'} for r in regions])
        region_by_id = {r.get('region_id', r.get('id')): r for r in regions}
        if (root/'action_plans.json').exists():
            action_plans = read_json(root/'action_plans.json')
        else:
            class_ids = None
            if classes:
                class_ids = np.asarray([{str(c): j for j,c in enumerate(classes)}[str(v)] for v in qframe[task['target']]])
            if (source/'validation_guidance.json').exists() and procedural is not None:
                from ..guided_coreset import protected_plans
                fixed=read_json(source/'validation_guidance.json')['protected']
                action_plans=protected_plans(regions,references[reference_seed],inputs['groups'],class_ids,fixed,config=cfg,sampler_config=SAMPLER)
            elif cfg['sampling']['mode'] == 'local_edit':
                from .local_edit import plan_local_actions
                action_plans = call_with_budget(plan_local_actions, regions, references, inputs['groups'], class_ids,
                                               size=size, config=cfg, sampler_config=SAMPLER)
            else:
                action_plans = call_with_budget(plan_actions, regions, references[reference_seed], inputs['groups'], class_ids,
                                               size=size, config=cfg, sampler_config=SAMPLER)
            write_json(root/'action_plans.json', action_plans)
        summary = _summary(task, context, train.iloc[np.sort(np.r_[ppos,qpos])], len(ppos), size, proxies)
        write_json(root/'task_summary.json', summary)
        _freeze_artifacts(root, 'region_design', [root/'regions.json', root/'action_plans.json',
            root/'reference_coresets.json', root/'task_summary.json'])
        if procedural and procedural.get('initial_admission_slots'):
            from ..stage1_metrics import primary_score, UndefinedMetricError
            from .context import admission_evidence
            proxy_scores = {}
            for i, pred in enumerate(proxies):
                try:
                    proxy_scores[i] = float(primary_score(task, qframe[task['target']], pred['predictions'],
                                                         pred['probabilities'], classes))
                except UndefinedMetricError:
                    continue
            bank_context['formal_budgets']['admission'] = admission_evidence(
                list(range(len(proxies))), proxy_scores, procedural['initial_admission_slots'],
                metric_direction(task), 'P-trained proxy scores on Q')
        contexts = {a['action_id']: build_context(summary, bank_context, region_by_id.get(a.get('region_id')), a,
                                                 schema=schema) for a in action_plans['actions']}
        if procedural is not None:
            for ctx in contexts.values():
                ctx['search_features'] = dict(is_initialization=1., progress=0., parent_depth=0.,
                    prior_conflict_fraction=0., current_replacement_fraction=0., model_diversity=1.)
            # Exact candidate sets use the same sampler and seeds as execution.
            # No fitting or held-out labels enter these cross-region previews.
            previews = {}
            # Append eligible variants to this decision only; later nodes replan on their parent.
            decision_actions = list(action_plans['actions'])
            for action in decision_actions:
                rows = {}
                for seed in cfg['sampling_seeds']:
                    order, _ = call_with_budget(sample_from_action, action, references[seed],
                        inputs['density'], inputs['hardness'], graph, inputs['groups'],
                        region_by_id.get(action.get('region_id'), {}).get('mask'),
                        seed=seed, config=SAMPLER, decision_seed=cfg['decision_seed'])
                    rows[str(seed)] = {'sample_hash': stable_hash(sorted(order.tolist())),
                        'regions': {rid: {'before': int(r['mask'][references[seed]].sum()),
                            'after': int(r['mask'][order].sum()),
                            'change': int(r['mask'][order].sum()-r['mask'][references[seed]].sum())}
                            for rid, r in region_by_id.items()}}
                    if procedural.get('sampling_policy', {}).get('version') == 'adaptive-coreset-v3':
                        replaced = len(order)-len(np.intersect1d(order, references[seed]))
                        rows[str(seed)].update(sample_replaced_count=replaced,
                                              sample_replaced_fraction=replaced/len(order))
                previews[action['action_id']] = rows
                contexts[action['action_id']]['action_summary']['cross_region_preview'] = rows
                if procedural.get('sampling_policy', {}).get('version') == 'adaptive-coreset-v3':
                    contexts[action['action_id']]['action_summary']['sampling_contract'] = dict(
                        mechanism='reference_reuse' if action['type'] == 'baseline' else 'quota_constrained_resampling',
                        size=action.get('size',size), total_coreset_rows=size, region_fraction_of_Q=action.get('region_fraction'),
                        quota_scope='editable_rows' if 'protected_q_indices' in action else 'all_rows',
                        requested_region_count=action.get('requested_region_count'),
                        feasible_region_count=action.get('feasible_region_count'),
                        focus_relative_increase=cfg['actions']['focus_relative_increase'],
                        focus_max_percentage_points=cfg['actions']['focus_max_percentage_points'],
                        small_quota_change_does_not_guarantee_small_sample_change=True)
                if action.get('variant') == 'gentle_cover':
                    contract = contexts[action['action_id']]['action_summary']['sampling_contract']
                    contract.update(mechanism='minimal_within_group_exchanges',
                                    small_quota_change_does_not_guarantee_small_sample_change=False)
                elif (action['type'] == 'cover' and
                      procedural.get('sampling_policy', {}).get('enable_gentle_cover')):
                    from .gentle_cover import plan_gentle_cover
                    region = region_by_id[action['region_id']]
                    gentle = plan_gentle_cover(action, references, rows, inputs['groups'],
                                               region['mask'], cfg['decision_seed'])
                    if gentle:
                        decision_actions.append(gentle)
                        ctx = build_context(summary, bank_context, region, gentle, schema=schema)
                        ctx['search_features'] = contexts[action['action_id']]['search_features']
                        contexts[gentle['action_id']] = ctx
            write_json(root/'skill_preview.json', previews)
        if (root/'decision.json').exists():
            decision = read_json(root/'decision.json')
            if decision['identity'] != identity or decision['policy_snapshot_id'] != snapshot['snapshot_id']:
                raise ValueError('Decision identity or policy snapshot changed')
        else:
            cases = store.cases(identity, snapshot=snapshot)
            retrieved = []
            excluded = heldout['evaluation_dataset_groups'] if heldout else []
            for ctx in contexts.values():
                retrieved.extend(retrieve_memory(ctx, cases, identity=identity, excluded_dataset_groups=excluded,
                    max_cases=cfg['advisor']['max_history_datasets'], max_distance=cfg['advisor']['max_retrieval_distance']))
            unique, seen = [], set()
            for case in sorted(retrieved, key=lambda c: (c['distance'], c['record_id'])):
                if case['dataset_group_id'] not in seen:
                    unique.append(case); seen.add(case['dataset_group_id'])
                if len(unique) >= cfg['advisor']['max_history_datasets']:
                    break
            write_json(root/'retrieved_cases.json', unique)
            from .experience import build_skill_cards
            skills = build_skill_cards(unique) if cfg['advisor']['experience_skills'] else []
            write_json(root/'retrieved_skills.json', skills)
            advisor_dir = root/'advisor'
            advisor_dir.mkdir(exist_ok=True)
            previous_advisor = None
            advisor = None
            if (advisor_dir/'result.json').exists():
                advisor = read_json(advisor_dir/'result.json')
                if (procedural is None and not advisor.get('api_available') and advisor.get('error') and advisor.get('calls') == 1
                        and not advisor.get('repair_attempted') and not advisor.get('skipped_reason')
                        and not (advisor_dir/'repair_1/started.json').exists()):
                    previous_advisor = advisor
            elif (advisor_dir/'started.json').exists():
                if procedural is not None:
                    raise ValueError('Interrupted skill request: result unknown; use a new decision directory')
                advisor = {'scores': {k: 0. for k in contexts}, 'api_available': 0, 'calls': 0,
                           'error': 'Interrupted advisor attempt; not repeated', 'request': None, 'response': None}
                write_json(advisor_dir/'result.json', advisor)
            if previous_advisor is not None or advisor is None:
                costs = []
                request_count = previous_advisor.get('calls', 0) if previous_advisor else 0
                async def request(prompt):
                    nonlocal request_count
                    check_deadline()
                    request_count += 1
                    if request_count > 2:
                        raise RuntimeError('Advisor initial call plus one repair budget exhausted')
                    attempt_dir = advisor_dir if request_count == 1 else advisor_dir/'repair_1'
                    attempt_dir.mkdir(exist_ok=True)
                    write_json(attempt_dir/'started.json', {'attempt': request_count, 'prompt_hash': stable_hash(prompt)})
                    return await await_with_budget(_llm_json_once(model, prompt, attempt_dir/'response.json', costs, pricing or {}))
                advisor_fn = score_actions_with_api
                if cfg['advisor']['decision_mode'] == 'agent':
                    from .agent_policy import choose_action_with_api
                    advisor_fn = choose_action_with_api
                advisor_cfg = dict(cfg['advisor'])
                if cfg['advisor']['decision_mode'] == 'agent':
                    advisor_cfg.update(sampling_mode=cfg['sampling']['mode'],
                        reward_objective=cfg['reward']['objective'])
                if procedural is not None:
                    from .procedural_skills import choose_with_skill_repair
                    advisor = await await_with_budget(choose_with_skill_repair(list(contexts.values()),
                        procedural['bundle'], request, history=procedural.get('history', []),
                        sampling=procedural.get('sampling_policy')))
                else:
                    advisor = await await_with_budget(advisor_fn(list(contexts.values()), unique, request,
                        config=advisor_cfg, skills=skills, previous_result=previous_advisor))
                write_json(advisor_dir/'result.json', advisor)
            if procedural is None and historical_references is not None and not advisor['api_available'] and advisor.get('skipped_reason') != 'baseline_only':
                raise RuntimeError('Historical replay requires real API advice; no fallback decision: '+str(advisor.get('error')))
            executable_actions = decision_actions if procedural is not None else action_plans['actions']
            if cfg['advisor']['decision_mode'] == 'agent' and cfg['sampling']['mode'] == 'local_edit':
                from .local_edit import scale_local_action
                executable_actions = [scale_local_action(a, advisor['adjustment_fraction'])
                    if a['action_id'] == advisor['chosen_action_id'] else a for a in executable_actions]
            actions = [{**a, 'context': build_context(summary, bank_context, region_by_id.get(a.get('region_id')), a,
                api_available=bool(advisor['api_available']), schema=schema)} for a in executable_actions]
            task_seed = _task_decision_seed(cfg['decision_seed'], group, task, source)
            if procedural is not None:
                from .procedural_skills import skill_decision
                for action in actions:
                    action['context']['search_features'] = contexts[action['action_id']]['search_features']
                    action['context']['action_summary']['cross_region_preview'] = previews[action['action_id']]
                    if 'sampling_contract' in contexts[action['action_id']]['action_summary']:
                        action['context']['action_summary']['sampling_contract'] = contexts[action['action_id']]['action_summary']['sampling_contract']
                decision = skill_decision(snapshot, actions, advisor, task_seed)
            elif cfg['advisor']['decision_mode'] == 'agent':
                from .agent_policy import materialize_agent_decision
                decision = materialize_agent_decision(snapshot, actions, advisor, task_seed, mode=cfg['mode'])
            else:
                decision = select_action(snapshot, actions, advisor['scores'], task_seed,
                                         mode=cfg['mode'], config=cfg['policy'])
            decision.update(identity=identity, decision_id=stable_hash({'run': str(root), 'identity': identity,
                'config': cfg, 'snapshot': snapshot['snapshot_id']}),
                frozen_config={'methods': bank, 'sampler': SAMPLER, 'evaluation_budget': effective_cfg.evaluation_budget,
                               'full_train_budget': effective_cfg.full_train_budget},
                api_available=advisor['api_available'], sampling_seeds=cfg['sampling_seeds'])
            # Even unexecutable diagnostic regions get a frozen training-side context;
            # this never represents an executed cover action or supplies a reward label.
            diagnostic_contexts = {}
            for rid, region in region_by_id.items():
                related = next((a for a in decision['actions'] if a.get('region_id') == rid), None)
                action = related or {'action_id': 'diagnostic:'+rid, 'type': 'cover', 'region_id': rid,
                    'size': size, 'reference_region_count': region['reference_count'],
                    'feasible_region_count': region['reference_count'], 'group_moved_count': 0, 'class_moved_count': 0}
                diagnostic_contexts[rid] = related['context'] if related else build_context(
                    summary, bank_context, region, action, api_available=bool(advisor['api_available']), schema=schema)
            decision['region_contexts'] = diagnostic_contexts
            if procedural is not None and procedural.get('sampling_policy') is not None:
                from .skill_sampling import plan_sampling
                decision['sampling_plan'] = plan_sampling(decision, procedural['sampling_policy'],
                    local_support=procedural.get('local_supported_actions', []),
                    exploration=procedural.get('exploration'))
            write_json(root/'decision.json', decision)
        _freeze_artifacts(root, 'decision', [root/'decision.json', root/'retrieved_cases.json', root/'advisor/result.json'])
        advisor_record = read_json(root/'advisor/result.json')
        write_json(root/'advisor_request.json', advisor_record.get('request'))
        write_json(root/'advisor_response.json', {k: advisor_record.get(k) for k in
            ('response','scores','api_available','error','calls','skipped_reason')})
        write_json(root/'decision_contexts.json', {'actions': [a['context'] for a in decision['actions']],
                                                 'diagnostic_regions': decision['region_contexts']})
        if not frozen:
            store.freeze_decision(decision)
        checkpoint(root, state, 'decision_frozen', decision_id=decision['decision_id'])
        chosen = next(a for a in decision['actions'] if a['action_id'] == decision['chosen_action_id'])
        baseline = next(a for a in decision['actions'] if a['type'] == 'baseline')
        audited = cfg['baseline_audit'] and chosen['type'] != 'baseline'
        saved_feedback, committed_snapshot = _recover_feedback(root, store, decision, frozen)
        arms = [('chosen', chosen)] + ([('baseline', baseline)] if audited else [])
        alternative_id = decision.get('sampling_plan', {}).get('alternative_action_id')
        if alternative_id:
            # Materializing a backup is cheap; model fits remain conditional on uncertainty.
            arms.append(('alternative', next(a for a in decision['actions'] if a['action_id'] == alternative_id)))
        if cfg['sampling']['resample_control']:
            arms.append(('resample_control', chosen))
        core_labels = read_typed(source/'core_validation.csv', context['csv_schema'])
        if not (source/'core_validation_features.csv').exists():
            # This projection is deliberately made after decision freeze.
            core_labels.drop(columns=[task['target']]).to_csv(root/'core_validation_features.csv', index=False)
            core_features = root/'core_validation_features.csv'
        else:
            core_features = source/'core_validation_features.csv'
        selected, core_results, selections = {}, {}, {}
        for arm_name, action in arms:
            for seed in cfg['sampling_seeds']:
                arm = root/'arms'/arm_name/f'seed_{seed}'
                arm.mkdir(parents=True, exist_ok=True)
                sampler = sample_from_action
                execution_action = action
                region_mask = region_by_id.get(action.get('region_id'), {}).get('mask')
                if arm_name == 'resample_control':
                    from .local_edit import resample_control_action
                    if region_mask is None:
                        region_mask = np.zeros(len(inputs['groups']), bool)
                    execution_action = resample_control_action(references[seed], inputs['groups'], region_mask, SAMPLER)
                elif cfg['sampling']['mode'] == 'local_edit' and 'protected_q_indices' not in action:
                    from .local_edit import sample_local_action
                    sampler = sample_local_action
                order, sampling = call_with_budget(sampler, execution_action, references[seed], inputs['density'], inputs['hardness'],
                    graph, inputs['groups'], region_mask,
                    seed=seed, config=SAMPLER, decision_seed=cfg['decision_seed'])
                selected[(arm_name, seed)] = order
                if procedural is not None and stable_hash(sorted(order.tolist())) != previews[action['action_id']][str(seed)]['sample_hash']:
                    raise ValueError('Executed sample differs from skill preview')
                positions = np.sort(qpos[order])
                row_ids = split.get('indices', {}).get('train')
                write_json(arm/'coreset_indices.json', {'train_positions': positions.tolist(),
                    'row_ids': [row_ids[i] for i in positions] if row_ids else positions.tolist(),
                    'selection_order_q_indices': order.tolist(), 'sampling': sampling})
                if not (arm/'coreset.csv').exists():
                    export_rows(paths['train'], [(arm/'coreset.csv', positions)])
                for filename in ('V_core_indices.json', 'core_validation.csv', 'task_context.json'):
                    if not (arm/filename).exists():
                        shutil.copyfile(source/filename, arm/filename)
                write_json(arm/'manifest.json', manifest)
        _freeze_artifacts(root, 'samples', [root/'arms'/name/f'seed_{seed}'/filename
            for name, _ in arms for seed in cfg['sampling_seeds']
            for filename in ('coreset.csv', 'coreset_indices.json', 'core_validation.csv', 'V_core_indices.json')])
        write_json(root/'selected_coreset_indices.json', {f'{name}:{seed}': read_json(
            root/'arms'/name/f'seed_{seed}'/'coreset_indices.json') for name, _ in arms for seed in cfg['sampling_seeds']})
        checkpoint(root, state, 'sampled')
        if sampling_only:
            preparation = {'status': 'sampled', 'source': str(source),
                'arm': str(root/'arms/chosen'/f'seed_{cfg["sampling_seeds"][0]}'),
                'decision_id': decision['decision_id'], 'test_access': False}
            write_json(root/'sampling_preparation.json', preparation)
            return preparation
        for arm_name, action in arms:
            for seed in cfg['sampling_seeds']:
                arm = root/'arms'/arm_name/f'seed_{seed}'
                records = []
                for method in methods:
                    i = method['method_id']; folder = arm/'evaluations'/f'method_{i}_coreset_0'
                    if not budget_available() and not (folder/'cache.json').exists():
                        outcome = {'status': 'infrastructure_failed', 'error': 'Task wall budget exhausted; no fit attempted'}
                    else:
                        outcome = fit_cached(Path(method['code_path']), arm/'coreset.csv', core_features,
                            {**context, 'expected_family': method['family'], 'fit_stage': 'evaluation',
                             **(frozen_fit_context(plan, i, 'coreset') if plan else {
                                 'stage_parameters': method['idea']['stage_parameters']['coreset']})}, classes, effective_cfg.evaluation_budget,
                            effective_cfg.model_seed, folder, effective_cfg.batch_size,
                            identity={**identity, 'training_row_ids': read_json(arm/'coreset_indices.json')['row_ids']},
                            infrastructure_retries=cfg['runtime']['infrastructure_retries'])
                    records.append(_evaluation(outcome, folder, task, core_labels[task['target']], classes,
                                                method, core=True, manifest=manifest))
                write_json(arm/'evaluations.json', records)
                core_results[f'{arm_name}:{seed}'] = records
                mcs_enabled = config.get('multi_proxy', {}).get('stage1_selection', {}).get('enabled', True)
                if mcs_enabled and all(r['status'] == 'ok' for r in records):
                    mcs_config = {**effective_cfg.stage1_selection, 'enabled': True, 'mode': 'record_only',
                                  'mcts_admission': False, 'coreset_weights': [1.]}
                    selection = call_with_budget(analyze_run, arm, mcs_config, manifest=manifest, coreset_count=1)
                else:
                    selection = {'selection_status': 'unavailable' if mcs_enabled else 'disabled',
                                 'selection_reason': 'Incomplete coreset results' if mcs_enabled else 'MCS disabled',
                                 'selection_mode': 'record_only'}
                selections[f'{arm_name}:{seed}'] = selection
        write_json(root/'core_results.json', core_results)
        write_json(root/'stage1_selection.json', selections)
        checkpoint(root, state, 'core_evaluated')
        # Freeze the evaluated identities before reading any full-stage outcome.
        full_plan = _full_evaluation_plan(methods, core_results[f'chosen:{cfg["sampling_seeds"][0]}'], task, cfg)
        if full_plan is None:
            result = {'status': 'incomplete', 'reason': 'incomplete_coreset_ranking', 'test_access': False}
            write_json(root/'result.json', result)
            return result
        plan_path = root/'full_evaluation_plan.json'
        if plan_path.exists() and read_json(plan_path) != full_plan:
            raise ValueError('Frozen full evaluation method identities changed')
        write_json(plan_path, full_plan)
        _freeze_artifacts(root, 'full_plan', [plan_path])
        full_ids = full_plan['method_ids']
        full_methods = [method for method in methods if method['method_id'] in full_ids]
        external_path = (task_path.parent/task['files']['validation']).resolve()
        if file_hash(external_path) != manifest['input_sha256']['validation']:
            raise ValueError('External validation changed after preparation')
        external = read_typed(external_path, context['csv_schema'])
        full_results = []
        for method in full_methods:
            i = method['method_id']; folder = root/'full_train'/f'method_{i}'
            if not budget_available() and not (folder/'cache.json').exists():
                outcome = {'status': 'infrastructure_failed', 'error': 'Task wall budget exhausted; no fit attempted'}
            else:
                outcome = fit_cached(Path(method['code_path']), paths['train'], source/'external_validation_features.csv',
                    {**context, 'expected_family': method['family'], 'fit_stage': 'full_train',
                     'reference_pipeline_path': (frozen_fit_context(plan, i, 'coreset')['initial_pipeline_path'] if plan else
                         str(root/'arms/chosen'/f'seed_{cfg["sampling_seeds"][0]}'/'evaluations'/f'method_{i}_coreset_0/unfitted.joblib')),
                     **(frozen_fit_context(plan, i, 'full_train') if plan else {
                         'stage_parameters': method['idea']['stage_parameters']['full_train']})}, classes, effective_cfg.full_train_budget,
                    effective_cfg.model_seed, folder, effective_cfg.batch_size,
                    identity={**identity, 'training_row_ids': split.get('indices', {}).get('train', list(range(len(train))))},
                    infrastructure_retries=cfg['runtime']['infrastructure_retries'])
            full_results.append(_evaluation(outcome, folder, task, external[task['target']], classes, method))
            write_json(root/'full_results.json', full_results)
        full_scores = {r['method_id']: r.get('metrics', {}).get(task['metric']) for r in full_results}
        full_complete = (set(full_scores) == set(full_ids)
                         and all(v is not None and np.isfinite(v) for v in full_scores.values()))
        full_ranking = {'direction': metric_direction(task), 'methods': [
            {'method_id': r['method_id'], 'method_name': r['method_name'], 'metric': task['metric'],
             'score': full_scores[r['method_id']]} for r in full_results]}
        write_json(root/'full_train_ranking.json', full_ranking)
        for key, selection in selections.items():
            if selection.get('method_ids'):
                arm_name, seed = key.split(':'); arm = root/'arms'/arm_name/f'seed_{seed}'
                write_json(arm/'full_train_ranking.json', full_ranking)
                compare_full_stage(arm, selection)
        checkpoint(root, state, 'full_evaluated')
        rewards = {}
        for arm_name, _ in arms:
            seed_rewards = []
            for seed in cfg['sampling_seeds']:
                records = core_results[f'{arm_name}:{seed}']
                scores = {r['method_id']: r.get('metrics', {}).get(task['metric']) for r in records}
                observed_records = [r for r in records if r['method_id'] in full_ids]
                if not full_complete or any(r['status'] != 'ok' for r in observed_records):
                    reward = {'status': 'reward_missing', 'reward': None, 'update_eligible': False}
                elif any(scores.get(i) is None or not np.isfinite(scores[i]) for i in full_ids):
                    reward = {'status': 'reward_missing', 'reward': None, 'update_eligible': False}
                else:
                    reward = compute_action_reward(scores, full_scores, direction=metric_direction(task),
                                                   method_ids=full_ids, config=cfg['reward'])
                seed_rewards.append({'seed': seed, **reward})
            rewards[arm_name] = {'action_id': chosen['action_id'] if arm_name == 'chosen' else baseline['action_id'],
                'observed_method_count': len(full_ids), 'method_ids': full_ids,
                **aggregate_seed_rewards(seed_rewards), 'seed_rewards': seed_rewards}
        if chosen['type'] == 'baseline' and cfg['baseline_audit']:
            rewards['baseline'] = {**deepcopy(rewards['chosen']), 'reused_chosen': True}
        if saved_feedback:
            # The action label/diagnostics have already been frozen for this decision.
            rewards['chosen'] = saved_feedback['reward']
        write_json(root/'action_reward.json', rewards['chosen'])
        write_json(root/'action_feedback.json', rewards['chosen'])
        write_json(root/'baseline_audit.json', {'enabled': cfg['baseline_audit'],
            'extra_core_fits_expected': 5*cfg['coreset_repeats'] if audited else 0,
            'chosen': rewards['chosen'], 'baseline': rewards.get('baseline'),
            'full_reference_shared': True, 'policy_label': 'chosen only'})
        region_feedback = []
        diagnostics = {'status': 'disabled', 'seeds': [], 'regions': []}
        if saved_feedback:
            region_feedback = saved_feedback['regions']
            diagnostics = (read_json(root/'region_feedback.json') if (root/'region_feedback.json').exists()
                           else {'status': 'recovered_frozen_feedback', 'seeds': []})
        elif not regions:
            diagnostics['status'] = 'no_regions'
        elif cfg['post_full_feedback']['enabled'] and cfg['post_full_feedback']['predict_core_on_external_validation'] and full_complete:
            diagnostics['status'] = 'completed'
            try:
                representation = joblib.load(source/'similarity_transform.joblib')
                gmm = joblib.load(source/'density_gmm/fitted.joblib')
                components = np.concatenate([call_with_budget(lambda batch: gmm.predict(representation.transform(batch)),
                    external.iloc[i:i+effective_cfg.batch_size])
                    for i in range(0,len(external),effective_cfg.batch_size)])
                full_predictions = [_load_predictions(root/f'full_train/method_{i}/predictions.npz', classes) for i in full_ids]
                for seed in cfg['sampling_seeds']:
                    if not all(r['status'] == 'ok' for r in core_results[f'chosen:{seed}']):
                        diagnostics['seeds'].append({'seed': seed, 'status': 'unavailable_core_execution'})
                        continue
                    core_predictions = []
                    for method in full_methods:
                        i = method['method_id']; folder = root/'core_external_predictions'/f'seed_{seed}'/f'method_{i}'
                        fitted = root/'arms/chosen'/f'seed_{seed}'/'evaluations'/f'method_{i}_coreset_0/fitted.joblib'
                        predicted = predict_cached(Path(method['code_path']), fitted, source/'external_validation_features.csv',
                            context, classes, folder, timeout_seconds=effective_cfg.evaluation_budget['timeout_seconds'],
                            batch_size=effective_cfg.batch_size, identity=identity)
                        if predicted['status'] != 'ok':
                            raise RuntimeError('Prediction-only diagnostic failed: '+str(predicted.get('error')))
                        core_predictions.append(_load_predictions(folder/'predictions.npz', classes))
                    feedback = call_with_budget(compute_region_feedback, regions, external, context, core_predictions, full_predictions,
                        components, loss_scale=discovered['loss_scale'], config=cfg,
                        seed=cfg['decision_seed'], dependence=_dependency_evidence(task, manifest.get('split_metadata', {})),
                        method_ids=full_ids)
                    diagnostics['seeds'].append({'seed': seed, 'status': 'ok', **feedback})
                    for row in feedback['regions']:
                        rid = row.get('region_id', row.get('id'))
                        original = region_by_id.get(rid)
                        if original is None:
                            continue
                        ctx = decision['region_contexts'][rid]
                        region_feedback.append({**row, 'seed': seed, 'context': ctx,
                            'region': {k:v for k,v in original.items() if k not in ('mask','indices')},
                            'rule_types': ctx.get('rule_types'),
                            'signal_type': 'specific_coreset_reversal_diagnostic_not_action_reward'})
            except Exception as error:
                diagnostics.update(status='partial_or_failed', error=type(error).__name__+': '+str(error))
        elif cfg['post_full_feedback']['enabled']:
            diagnostics['status'] = 'unavailable_full_reference' if not full_complete else 'prediction_disabled'
        diagnostics['regions'] = region_feedback
        write_json(root/'region_feedback.json', diagnostics)
        chosen_complete = full_complete and _reward_resolved(rewards['chosen'])
        if chosen_complete and saved_feedback is None:
            saved_feedback = {'decision_id': decision['decision_id'], 'reward': rewards['chosen'], 'regions': region_feedback}
            write_json(root/'feedback_pending.json', saved_feedback)
        if (root/'feedback_pending.json').exists():
            _freeze_artifacts(root, 'feedback', [root/'feedback_pending.json'])
        checkpoint(root, state, 'feedback_ready')
        complete = _required_results_complete(full_complete, rewards, cfg['baseline_audit'])
        # A missing infrastructure result remains resumable. Never consume an
        # immutable decision event with a placeholder label that would later change.
        committed = (committed_snapshot or store.commit_feedback(decision, saved_feedback['reward'],
            saved_feedback['regions'], mode=cfg['mode'], schema=schema, config=cfg)) if chosen_complete else {
                'snapshot_id': decision['policy_snapshot_id'], 'status': 'pending_missing_results'}
        write_json(root/'feedback_commit.json', {'mode': cfg['mode'], 'snapshot_id': committed.get('snapshot_id'),
            'decision_id': decision['decision_id'], 'updated': chosen_complete and not frozen
                and committed_snapshot is None and rewards['chosen']['update_eligible']})
        if chosen_complete and not frozen:
            _export_policy(memory_path, schema, committed)
        costs = _collect_costs(root, cfg['coreset_repeats'], audited, len(full_ids))
        account_wall_time(state)
        costs['elapsed_wall_seconds'] = state.get('wall_seconds', 0)
        write_json(root/'cost_report.json', costs)
        result = {'status': 'completed' if complete else 'incomplete',
            'mode': cfg['mode'], 'dataset_group_id': group, 'chosen_action_id': chosen['action_id'],
            'chosen_action_type': chosen['type'], 'reward': rewards['chosen']['reward'],
            'full_results': full_ranking, 'baseline_reward': rewards.get('baseline', {}).get('reward'),
            'reward_status': rewards['chosen']['status'], 'full_evaluation_plan': full_plan,
            'baseline_reward_status': rewards.get('baseline', {}).get('status'),
            'region_feedback_status': diagnostics['status'], 'costs': costs, 'test_access': False,
            'wall_budget_seconds': wall_limit,
            'policy_updated': not frozen and chosen_complete and rewards['chosen']['update_eligible']}
        write_json(root/'result.json', result)
        if complete:
            checkpoint(root, state, 'feedback_committed')
        return result
    except TaskDeadlineExceeded as error:
        state['last_error'] = type(error).__name__ + ': ' + str(error)
        result = {'status': 'incomplete', 'reason': 'task_wall_budget_exhausted',
                  'phase': state.get('phase'), 'wall_budget_seconds': wall_limit,
                  'resume_hint': 'Use --resume --extend-wall-seconds SECONDS', 'test_access': False}
        write_json(root/'result.json', result)
        return result
    except Exception as error:
        if state.get('phase'):
            state['last_error'] = type(error).__name__+': '+str(error)
        raise
    finally:
        scope.__exit__(None, None, None)
        account_wall_time(state, finish=True)
        if state.get('phase'):
            write_json(root/'state.json', state)
            if not (root/'cost_report.json').exists() or state['phase'] != 'feedback_committed':
                costs = _collect_costs(root, cfg['coreset_repeats'], audited)
                if sampling_only:
                    costs.update(theoretical_predictive_fits=5,
                        scope='Preparation only: P proxy fits; formal evaluations are accounted in the search run')
                write_json(root/'cost_report.json', {**costs, 'elapsed_wall_seconds': state['wall_seconds']})
        store.close()
