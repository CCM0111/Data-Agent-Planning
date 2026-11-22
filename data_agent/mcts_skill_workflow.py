"""One skill library from initialization through staged MCTS development.

Initialization and expansions use one screening branch. Stage
boundaries never alter an already frozen decision.
"""
from copy import deepcopy
import fcntl
from pathlib import Path

from .coreset import file_hash
from .mcts_policy import MCTSConfig
from .paired_sampler_experiment import freeze_json
from .region_bandit.config import normalize_config
from .region_bandit.mcts_skill_reward import checkpoint_reward, reward_settings, RELATIVE_PROTOCOLS, INITIAL_TOP3_VERSION, relative_fact
from .region_bandit.mcts_skill_initial import freeze_initial_reference, save_initial_checkpoint
from .region_bandit.mcts_rounds import _method_bank
from .region_bandit.runtime import run_region_task
from .region_bandit.schemas import read_json, write_json, stable_hash
from .region_bandit.skill_sampling import mcts_sampling_policy
from .region_bandit.skill_store import SkillStore, sampling_path
from .stage1_metrics import metric_direction
from .workflow import run_search, source_view


def skill_workflow_config(config):
    result = deepcopy(config)
    result.setdefault('multi_proxy', {})['validation_guided_mix']=True
    cfg = result['region_bandit']
    cfg.update(enabled=True, mode='develop_online', full_train_methods='top3',
               baseline_audit=False, coreset_repeats=1, sampling_seeds=[20261026])
    cfg['regions'].update(max_behavior_regions=2, max_single_pair_regions=2, max_coverage_regions=2)
    cfg['advisor'].update(enabled=True, decision_mode='agent', experience_skills=False)
    cfg['sampling'].update(mode='resample', resample_control=False)
    cfg['reward']['objective'] = 'absolute'
    cfg['post_full_feedback']['enabled'] = False
    cfg['mcts_rounds']['enabled'] = True
    result['region_bandit'] = normalize_config(cfg)
    return result


def initial_fact(engine, root, *, store_path, source, task, role, initial, settings, controller=None):
    if not engine.state['initialization_complete']:
        return {'status': 'pending_initial_results'}
    path = root/'initial_skill_fact.json'
    reference = read_json(root/'initial_reward_reference.json')
    decision = read_json(root/'sampling/decision.json')
    bank = [engine.state['candidates'][i] for i in engine.state['initial_ids']]
    selected = engine.state['initial_selected']
    observed = {c['id'] for c in bank if c.get('full_report')}
    if (len(bank) != 5 or len({c['id'] for c in bank}) != 5
            or len(selected) != 3 or len(set(selected)) != 3 or observed != set(selected)
            or any(not c.get('low_report') for c in bank)):
        raise ValueError('Initial skill feedback requires the frozen five low and selected three full evaluations')
    frozen_methods = decision['frozen_config']['methods']
    methods = {m['method_id']: m for m in frozen_methods}
    if (len(frozen_methods) != 5 or len(methods) != 5
            or {c.get('source_method_id') for c in bank} != set(methods)
            or any(c.get('code_sha256') != methods[c['source_method_id']].get('code_sha256') for c in bank)):
        raise ValueError('Initial method bank changed after the skill decision')
    if (stable_hash(initial['procedural']['bundle']) != initial['bundle_id']
            or decision['skill_execution'].get('skill_bundle_hash') != initial['bundle_id']):
        raise ValueError('Initial skill decision and pinned bundle differ')
    for candidate in bank:
        for stage in ('low', 'full'):
            report = candidate.get(stage+'_report')
            if report and candidate.get(stage+'_score') != report.get('score'):
                raise ValueError('Initial score differs from its committed evaluation report')
    core = {c['id']: c['low_score'] for c in bank if c.get('low_report')}
    full = {c['id']: c['full_score'] for c in bank if c.get('full_report')}
    screening = engine.state.get('initial_skill_screening')
    if screening:
        if len(screening['branches']) != 1:
            raise ValueError('Skill MCTS requires single-branch screening')
        if set(screening['combined']['scores']) != set(core):
            raise ValueError('Initial branch aggregation changed the frozen method bank')
        core = screening['combined']['scores']
    reward = checkpoint_reward(core, full, expected_ids=[c['id'] for c in bank],
        direction=metric_direction(engine.backend.task), metric=engine.backend.task['metric'], settings=settings,
        reference_scale=reference['reference_scale'])
    action = next(a for a in decision['actions'] if a['action_id'] == decision['chosen_action_id'])
    fact = dict(task=task, source=source, role=role, bundle_hash=initial['bundle_id'],
        bank_hash=stable_hash(_method_bank(bank)), seed=20261026,
        checkpoint_id=stable_hash({'run': str(root), 'stage': 'initialization'}),
        decision_id=decision['decision_id'], skill_execution=decision['skill_execution'],
        action={k: action[k] for k in ('action_id', 'type', 'region_id') if k in action},
        sample_hash=file_hash(root/'selected_source/coreset_0_indices.json'),
        decision_protocol_hash=stable_hash(initial), sampling_policy_hash=stable_hash(initial['procedural']['sampling_policy']),
        core_scores=core, full_scores=full, reward=reward, branch_count=1,
        sampling_path=sampling_path(action['context'], _method_bank(bank), actions=decision['actions'],
            regions=read_json(root/'sampling/regions.json')['regions'] if (root/'sampling/regions.json').exists() else ()),
        decision_context=action['context'].get('region_summary', {}),
        search_stage='initialization', search_features=dict(is_initialization=1., progress=0., parent_depth=0.),
        stage='initialization', trajectory_id=str(root), execution_kind='online',
        nominated_method_ids=list(selected), nominated_method_id=selected[0],
        selected_method_id=selected[0], successful_full_method_ids=list(selected),
        method_search_gain=None, scope='Observed initial three of five; no complete-bank retention claim')
    if screening:
        branches = deepcopy(screening['branches'])
        for branch in branches:
            branch['reward'] = checkpoint_reward(branch['scores'], full, expected_ids=[c['id'] for c in bank],
                direction=metric_direction(engine.backend.task), metric=engine.backend.task['metric'], settings=settings,
                reference_scale=reference['reference_scale'])
        primary_reward = branches[0]['reward']['reward']
        fact.update(branches=branches, branch_count=len(branches), uncertainty=screening['trigger'],
            sample_hash=screening['combined']['sample_hash'], combination=screening['combined'],
            branch_reward_delta=None,
            combination_gain=None if primary_reward is None or reward['reward'] is None else reward['reward']-primary_reward)
    if 'baseline_scores' in reference:
        baseline = {c['id']: reference['baseline_scores'][str(c['source_method_id'])] for c in bank}
        relative_fact(fact, baseline, 'random_proxy')
        reward = fact['reward']
    with SkillStore(store_path) as store:
        if path.exists():
            saved = read_json(path)
            recorded = store.fact(saved['fact_id'])
            if 'sampling_path' not in recorded:
                # Older sealed checkpoints predate the compact path field.
                legacy_context = fact.pop('sampling_path')
            else:
                # The method bank, decision and sample hashes below bind the inputs.
                # Preserve the summary format originally sealed with this checkpoint.
                fact['sampling_path'] = recorded['sampling_path']
            if recorded['fact_id'] != stable_hash(fact):
                raise ValueError('Cached initial skill fact differs from the frozen decision and observations')
            ident = recorded['fact_id']
            if 'sampling_path' not in store.history_observation(recorded):
                store.attach_historical_context(ident, legacy_context)
        else:
            ident = store.record_fact(fact)
    result = dict(status='completed', fact_id=ident, reward=reward['reward'], observed_methods=len(full),
                  checkpoint_folder=str(root/'sampling/mcts_skill_initial'))
    if 'baseline_kind' in fact:
        result.update(baseline_kind=fact['baseline_kind'], reward_status=reward['status'],
                      order_gain=reward['order_gain'])
    if controller is not None:
        result['checkpoint_folder'] = save_initial_checkpoint(controller, engine, initial, fact, reference)
    if path.exists():
        if saved != result:
            raise ValueError('Cached initial skill feedback summary differs from its fact')
        return saved
    write_json(path, result)
    return result


async def run_skill_task(row, output, *, config, search, memory, model, model_identity,
                         store_path, role, bundle_id=None, pricing=None, subgroups=None,
                         sampling=None, reward=None, full_success_target=None):
    from .region_bandit.mcts_skills import RoundSkills
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=True)
    source, task = row['collection_dataset'], row['task']
    cfg = skill_workflow_config(config); search = MCTSConfig.from_dict(search)
    if not search.enabled or search.initial_full_count != 3:
        raise ValueError('Skill MCTS requires initial three full evaluations and enabled search')
    existing = read_json(root/'skill_workflow.json') if (root/'skill_workflow.json').exists() else {}
    policy = mcts_sampling_policy(sampling or existing.get('sampling'))
    reward_cfg = reward_settings(reward or existing.get('reward') or {'version': INITIAL_TOP3_VERSION})
    manifest = dict(protocol='unified-skill-mcts-v1', task_path=str(Path(row['task_path']).resolve()),
        task_sha256=file_hash(row['task_path']), source=source, role=role,
        config=cfg, search=search.to_dict(), memory=memory, model=model_identity,
        store_path=str(Path(store_path).resolve()), sampling=policy, reward=reward_cfg,
        initialization_sampling=existing.get('initialization_sampling', policy['mode']), test_access=False)
    with (root/'.skill_workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_json(root/'skill_workflow.json', manifest)
        with SkillStore(store_path) as store:
            store.register_sources([source], role)
            active = bundle_id or store.active
            store.bundle(active)
            if (root/'initial_skill_input.json').exists():
                initial = read_json(root/'initial_skill_input.json')
            else:
                initial = dict(bundle_id=active, procedural=dict(bundle=store.bundle(active),
                    initial_admission_slots=search.initial_full_count,
                    history=store.decision_history(source, allow_local=False, limit=None),
                    sampling_policy=policy))
                freeze_json(root/'initial_skill_input.json', initial)
        preparation = await run_region_task(row['task_path'], root/'sampling', config=cfg, model=model,
            model_identity=model_identity, pricing=pricing, memory_path=root/'sampling_ledger.sqlite',
            dataset_group_id=source, resume=(root/'sampling/state.json').exists(), sampling_only=True,
            procedural=initial['procedural'])
        if preparation.get('status') != 'sampled':
            raise RuntimeError('Initial sampling did not complete: '+str(preparation))
        reference = freeze_initial_reference(root, preparation['source'], relative=reward_cfg['version'] in RELATIVE_PROTOCOLS)
        selected = source_view(preparation['source'], preparation['arm'], root/'selected_source')
        controller = RoundSkills(root/'sampling', config=cfg, memory_path=store_path,
            source=source, task=task, role=role, bundle_id=active, model_identity=model_identity,
            sampling_policy=policy, reward_settings=reward_cfg)
        controller.initial_reference_scale = reference['reference_scale']
        feedback = lambda engine: initial_fact(engine, root, store_path=store_path, source=source,
            task=task, role=role, initial=initial, settings=reward_cfg, controller=controller)
        summary, engine = await run_search(selected, root/'search', model=model, model_identity=model_identity,
            pricing=pricing, search=search, memory=memory, subgroups=subgroups,
            resume=(root/'search/search_state.json').exists(), frozen_initial=True,
            region_controller=controller, initial_feedback=feedback, full_success_target=full_success_target)
        result = {**summary, 'skill_bundle': active, 'search_run': str(root/'search'),
                  'source_role': role, 'test_access': False}
        write_json(root/'result.json', result)
        return result, engine, controller
