"""Read-only historical bank adapter and paired coreset evaluation.

Selection sees P/Q artifacts and frozen definitions only. Full scores and old
low scores are opened by the evaluator strictly after the decision is frozen.
"""
from pathlib import Path
import math
import numpy as np

from ..coreset import file_hash
from ..multi_proxy import MultiProxyConfig, load_fixed
from ..multi_proxy_io import read_typed
from ..multi_proxy_feedback import frozen_fit_context
from ..stage1_metrics import metric_direction
from .schemas import read_json, write_json, clean


def validate_historical_source(task_path, source, references, seeds):
    source, task_path = Path(source), Path(task_path).resolve()
    manifest = read_json(source/'manifest.json')
    if manifest['status'] != 'success' or Path(manifest['task_path']).resolve() != task_path:
        raise ValueError('Historical bank must be successful and match this task')
    cfg = MultiProxyConfig(**manifest['configuration'])
    if not cfg.parameter_feedback or cfg.n_strategies != 5:
        raise ValueError('Historical bank must have five methods and frozen stage plans')
    task, split, paths, hashes, _ = load_fixed(task_path)
    hashes['validation'] = file_hash(task_path.parent/task['files']['validation'])
    if any(hashes[k] != manifest['input_sha256'][k] for k in ('train', 'validation')):
        raise ValueError('Historical dataset contents changed')
    for key in ('target', 'problem_type', 'metric', 'metric_params'):
        if task.get(key) != manifest['task'].get(key):
            raise ValueError('Historical task semantics changed: '+key)
    # Outer metadata may have acquired a separate test set since the old run.
    # Exact train/validation byte hashes above, not metadata IDs, establish reuse.
    for method in manifest['methods']:
        if file_hash(method['code_path']) != method['code_sha256']:
            raise ValueError('Historical method code changed')
    plan = read_json(source/'parameter_feedback/frozen_plan.json')
    if sorted(m['method_id'] for m in plan['methods']) != list(range(5)):
        raise ValueError('Historical stage plan lacks five methods')
    for method in plan['methods']:
        for artifact in method['pipeline_artifacts'].values():
            if file_hash(artifact['path']) != artifact['sha256']:
                raise ValueError('Historical unfitted pipeline changed')
    for filename, key in [('core_validation.csv', 'labels_sha256'), ('V_core_indices.json', 'indices_sha256')]:
        if file_hash(source/filename) != manifest['evaluation_set'][key]:
            raise ValueError('Historical V_core changed')
    qpos = np.asarray(read_json(source/'Q_indices.json')['train_positions'], int)
    design = read_json(source/'coverage_comparison_design.json')['arms']
    if set(references) != {str(seed) for seed in seeds}:
        raise ValueError('Historical references must match all requested seeds')
    for seed in seeds:
        item = references[str(seed)]
        cid = item['source_coreset_id']
        if not any(a['coreset_id'] == cid and a['seed'] == seed and
                   a['mechanism'] == 'coverage_a30_b60_l200' for a in design):
            raise ValueError('Historical reference is not the requested original coverage arm')
        order = np.asarray(read_json(source/f'coreset_{cid}_selection_order.json')['q_indices'], int)
        if order.ndim != 1 or np.any(order < 0) or np.any(order >= len(qpos)) or len(np.unique(order)) != len(order):
            raise ValueError('Invalid historical Q selection')
        arm = Path(item['arm'])
        original = sorted(read_json(arm/'coreset_0_indices.json')['train_positions'])
        if sorted(qpos[order].tolist()) != original:
            raise ValueError('Saved coreset row IDs differ from original selection order')
        if file_hash(arm/'coreset_0.csv') != file_hash(source/f'coreset_{cid}.csv'):
            raise ValueError('Historical reference CSV does not match source')
    return manifest


def scores_from_records(records, metric):
    scores = {}
    for row in records:
        value = row.get('metrics', {}).get(metric)
        if row.get('status') not in ('ok', 'success') or value is None or not math.isfinite(value):
            raise ValueError('Missing or failed five-method evaluation')
        identifier = row['method_id']
        if identifier in scores:
            raise ValueError('Duplicate method evaluation')
        scores[identifier] = float(value)
    if set(scores) != set(range(5)):
        raise ValueError('Exactly five method results are required')
    return scores


def historical_records(arm):
    records = read_json(Path(arm)/'evaluations.json')
    if not isinstance(records, list):
        raise ValueError('Unsupported historical evaluation format')
    return records


def bind_replica_memory(config, memory_path):
    """Prevent another seed or sampling protocol from accidentally sharing this store."""
    from ..paired_sampler_experiment import freeze_json
    from .config import normalize_config
    from .local_edit import local_sampler_version
    from .action_planner import SAMPLER_VERSION
    from .config import reward_protocol_hash
    cfg = normalize_config(config['region_bandit'])
    if not cfg['sampling']['independent_bandit']:
        if Path(memory_path).with_suffix('.replica.json').exists():
            raise ValueError('Independent replica memory cannot be used by a pooled run')
        return
    marker = Path(memory_path).with_suffix('.replica.json')
    if Path(memory_path).exists() and not marker.exists():
        raise ValueError('Independent replica requires a new or already bound memory store')
    freeze_json(marker, dict(sampling_seed=cfg['sampling_seeds'][0], decision_seed=cfg['decision_seed'],
        sampler_version=(local_sampler_version(cfg) if cfg['sampling']['mode']=='local_edit' else SAMPLER_VERSION),
        max_replace_fraction=cfg['sampling']['max_replace_fraction'],
        donor_remove_fraction=cfg['sampling']['max_donor_remove_fraction'], allocation=cfg['sampling']['allocation'],
        decision_mode=cfg['advisor']['decision_mode'],
        reward_protocol_hash=reward_protocol_hash(cfg),
        isolation='one sampling seed per memory, API history, policy and reward stream'))


async def run_historical_task(row, root, *, source, references, random_references,
                              config, model, model_identity, pricing, memory_path):
    from .runtime import run_region_task, _evaluation, _export_policy
    from .execution import fit_cached
    from .memory_store import MemoryStore
    from .reward import compute_action_reward, aggregate_seed_rewards, paired_delta_reward
    from .experience import build_skill_cards
    from ..paired_sampler_experiment import ranking_metrics, freeze_json
    root, source = Path(root), Path(source)
    sampling = root/'sampling'
    cfg = config['region_bandit']
    objective = cfg['reward'].get('objective', 'absolute')
    bind_replica_memory(config, memory_path)
    prep = await run_region_task(Path(row['task_path']), sampling, config=config, model=model,
        model_identity=model_identity, pricing=pricing, memory_path=memory_path,
        dataset_group_id=row['collection_dataset'], resume=(sampling/'state.json').exists(),
        sampling_only=True, bootstrap_source=source, historical_references=references)
    decision = read_json(sampling/'decision.json')
    advisor = read_json(sampling/'advisor/result.json')
    if not advisor['api_available'] and advisor.get('skipped_reason') != 'baseline_only':
        raise RuntimeError('Real API advisor unavailable; task stopped, no fallback training: '+str(advisor.get('error')))
    chosen = next(a for a in decision['actions'] if a['action_id'] == decision['chosen_action_id'])
    manifest = read_json(source/'manifest.json')
    effective = MultiProxyConfig(**manifest['configuration'])
    task = read_json(row['task_path'])
    context = read_json(source/'task_context.json')
    plan = read_json(source/'parameter_feedback/frozen_plan.json')
    labels = read_typed(source/'core_validation.csv', context['csv_schema'])[task['target']]
    core_features = source/'core_validation_features.csv'
    if not core_features.exists():
        core_features = sampling/'core_validation_features.csv'
    methods = [dict(m, method_id=i) for i, m in enumerate(manifest['methods'])]
    direction = metric_direction(task)
    # Boundary: only here, after immutable decision+API success, open full outcomes.
    full_files = [source/f'full_train/method_{i}/evaluation.json' for i in range(5)]
    full = scores_from_records([read_json(p) for p in full_files], task['metric'])
    freeze_json(root/'full_reference.json', {'scores': {str(k): v for k, v in full.items()}, 'artifacts': {str(p): file_hash(p) for p in full_files},
        'decision_id': decision['decision_id'], 'new_full_fits': 0})
    comparisons, seed_rewards, before_rewards, after_rewards = [], [], [], []
    new_core_fits_expected = 0
    control_enabled = cfg.get('sampling', {}).get('resample_control', False)
    control_rewards = []
    for seed in cfg['sampling_seeds']:
        old_arm = Path(references[str(seed)]['arm'])
        old_records = historical_records(old_arm)
        old_scores = scores_from_records(old_records, task['metric'])
        old_by_id = {r['method_id']: r for r in old_records}
        old_rows = read_json(old_arm/'coreset_0_indices.json')['train_positions']
        evaluated = {}
        for arm_name in ['chosen'] + (['resample_control'] if control_enabled else []):
            arm = sampling/'arms'/arm_name/f'seed_{seed}'
            rows = read_json(arm/'coreset_indices.json')['train_positions']
            reuse = sorted(rows) == sorted(old_rows)
            if arm_name == 'chosen' and chosen['type'] == 'baseline' and not reuse:
                raise ValueError('Baseline must preserve exact historical rows')
            if reuse:
                records = old_records
            else:
                records = []
                new_core_fits_expected += len(methods)
                for method in methods:
                    i = method['method_id']
                    folder = arm/'evaluations'/f'method_{i}_coreset_0'
                    print(f"FIT {row['task']} arm={arm_name} seed={seed} method={i}", flush=True)
                    result = fit_cached(Path(method['code_path']), arm/'coreset.csv', core_features,
                        {**context, 'expected_family': method['family'], 'fit_stage': 'evaluation',
                         **frozen_fit_context(plan, i, 'coreset')}, context['classes'], effective.evaluation_budget,
                        effective.model_seed, folder, effective.batch_size,
                        identity={'historical_source': str(source), 'decision_id': decision['decision_id'],
                                  'training_row_ids': rows},
                        infrastructure_retries=cfg['runtime']['infrastructure_retries'])
                    record = _evaluation(result, folder, task, labels, context['classes'], method, core=True, manifest=manifest)
                    if result['status'] == 'ok' and record['pipeline_spec'] != old_by_id[i]['pipeline_spec']:
                        raise ValueError(f'Effective pipeline changed for method {i}; cannot compare as sampling-only')
                    records.append(record)
                    write_json(arm/'evaluations_partial.json', records)
            write_json(arm/'evaluations.json', records)
            evaluated[arm_name] = dict(scores=scores_from_records(records, task['metric']), reuse=reuse,
                                      overlap=len(set(rows)&set(old_rows))/len(old_rows))
        new_scores = evaluated['chosen']['scores']
        random_scores = scores_from_records(historical_records(random_references[str(seed)]['arm']), task['metric'])
        new_reward = compute_action_reward(new_scores, full, direction=direction, config=cfg['reward'])
        old_reward = compute_action_reward(old_scores, full, direction=direction, config=cfg['reward'])
        after_rewards.append({'seed': seed, **new_reward})
        seed_rewards.append({'seed': seed, **(paired_delta_reward(new_reward, old_reward)
            if objective == 'paired_delta' else new_reward)})
        before_rewards.append({'seed': seed, **old_reward})
        for name, scores in [('random', random_scores), ('coverage', old_scores)]:
            comparisons.append({'seed': seed, 'arm': name, **ranking_metrics(scores, full, direction),
                'reused_history': True, 'overlap_with_original_coverage': 1. if name == 'coverage' else None,
                'chosen_action_type': name})
        for name, result in evaluated.items():
            comparisons.append({'seed': seed, 'arm': 'bandit' if name == 'chosen' else name,
                **ranking_metrics(result['scores'], full, direction), 'reused_history': result['reuse'],
                'overlap_with_original_coverage': result['overlap'],
                'chosen_action_type': chosen['type'] if name == 'chosen' else name})
        if control_enabled:
            control_rewards.append({'seed': seed, **compute_action_reward(
                evaluated['resample_control']['scores'], full, direction=direction, config=cfg['reward'])})
        write_json(root/'comparison_partial.json', comparisons)
    before = aggregate_seed_rewards(before_rewards)['reward']
    feedback = {'action_id': chosen['action_id'], **aggregate_seed_rewards(seed_rewards, objective=objective),
        'reward_objective': objective,
        'seed_rewards': seed_rewards, 'method_ids': list(range(5)), 'observed_method_count': 5}
    after = aggregate_seed_rewards(after_rewards)['reward']
    feedback['paired_comparison'] = {'before_weighted_accuracy': before, 'after_weighted_accuracy': after,
        'delta_weighted_accuracy': after-before if after is not None and before is not None else None,
        'sampling_repeats': len(seed_rewards), 'independent_source_count': 1}
    diagnostic_control = aggregate_seed_rewards(control_rewards) if control_enabled else None
    freeze_json(root/'action_feedback.json', clean(feedback))
    schema = read_json(sampling/'feature_schema.json')
    with MemoryStore(memory_path) as store:
        snapshot = store.commit_feedback(decision, feedback, [], schema=schema, config=cfg)
        _export_policy(memory_path, schema, snapshot)
        # Archive the new evidence card; it is not used to decide this task.
        own = [c for c in store.cases(decision['identity']) if c['record_id'] == 'action:'+decision['decision_id']]
        skills = build_skill_cards(own)
        write_json(root/'experience_cards.json', skills)
        write_json(Path(memory_path).parent/'skills'/f"{decision['decision_id']}.json", skills)
    result = {'status': 'completed', 'task': row['task'], 'dataset': row['collection_dataset'],
        'problem_type': row['problem_type'], 'metric': task['metric'], 'direction': direction,
        'chosen_action_type': chosen['type'], 'advisor_api_available': advisor['api_available'],
        'decision_id': decision['decision_id'], 'arms': comparisons, 'paired_comparison': feedback['paired_comparison'],
        'new_full_fits': 0, 'new_proxy_fits': 0, 'new_core_fits_expected': new_core_fits_expected,
        'resample_control_reward': diagnostic_control,
        'control_used_for_policy_update': False,
        'independent_bandit': cfg.get('sampling', {}).get('independent_bandit', False),
        'sampling_seeds': cfg['sampling_seeds'],
        'test_access': False, 'policy_snapshot_id': snapshot['snapshot_id']}
    write_json(root/'comparison.json', result)
    return result
