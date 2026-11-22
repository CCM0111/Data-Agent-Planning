"""Fixed-five-method skill experiments. Selection and outcome access are separate."""
from copy import deepcopy
from pathlib import Path

import time

from ..coreset import file_hash
from ..multi_proxy import MultiProxyConfig, load_fixed
from ..multi_proxy_feedback import frozen_fit_context
from ..multi_proxy_io import read_typed
from ..paired_sampler_experiment import freeze_json
from ..stage1_metrics import metric_direction
from .config import normalize_config
from .historical_replay import validate_historical_source, scores_from_records, historical_records
from .schemas import read_json, write_json, stable_hash, environment_identity
from .skill_store import sampling_path
from .skill_reward import screening_reward, reward_settings
from .skill_sampling import sampling_policy, decide_second_branch, combine_scores


def frozen_bank(row, seeds):
    """Validate definitions and bind all reusable artifacts; expose only hashes."""
    source = Path(row['source']).resolve()
    task_path = Path(row['task_path']).resolve()
    manifest = validate_historical_source(task_path, source, row['coverage_references'], seeds)
    task, split, paths, _, _ = load_fixed(task_path)
    from ..prepared_protocol import validate_prepared_partition
    validate_prepared_partition(task, task_path, split,
        *[read_json(source/f'{part}_indices.json')['train_positions'] for part in ('P', 'V_core', 'Q')])
    # Outcome bytes are fingerprinted, never exposed to the selection API.
    files = [source/'manifest.json', source/'parameter_feedback/frozen_plan.json',
             source/'core_validation.csv', source/'V_core_indices.json',
             source/'Q_indices.json', paths['train'], task_path.parent/task['files']['validation']]
    files += [source/f'full_train/method_{i}/evaluation.json' for i in range(5)]
    files += [Path(m['code_path']) for m in manifest['methods']]
    for reference in row['coverage_references'].values():
        files += [Path(reference['arm'])/name for name in
                  ('evaluations.json', 'coreset_0_indices.json', 'coreset_0.csv')]
    binding = {str(p.resolve()): file_hash(p) for p in files}
    return dict(bank_hash=stable_hash(binding), artifacts=binding,
                source=str(source), task_path=str(task_path))


def verify_bank(bank):
    if any(file_hash(path) != digest for path, digest in bank['artifacts'].items()):
        raise ValueError('Frozen skill experiment bank changed')


def skill_config(config, seeds):
    result = deepcopy(config)
    cfg = result['region_bandit']
    cfg.update(mode='develop_online', full_train_methods='all', baseline_audit=False,
               coreset_repeats=len(seeds), sampling_seeds=list(seeds))
    cfg.setdefault('advisor', {}).update(enabled=True, decision_mode='agent', experience_skills=False)
    cfg.setdefault('sampling', {}).update(mode='resample', resample_control=False,
        independent_bandit=len(seeds) == 1 and seeds[0] != 20261026,
        deletion_rule='uniform', max_donor_remove_fraction=1., allocation='capacity')
    cfg.setdefault('reward', {})['objective'] = 'absolute'
    cfg.setdefault('post_full_feedback', {})['enabled'] = False
    cfg.setdefault('mcts_rounds', {})['enabled'] = False
    result['region_bandit'] = normalize_config(cfg)
    return result


async def prepare_decision(row, folder, *, bank, bundle, history, config, model, model_identity, pricing,
                           sampling=None, exploration=None):
    from .runtime import run_region_task
    folder = Path(folder)
    verify_bank(bank)
    procedural = dict(bundle=bundle, history=history)
    if exploration is not None:
        procedural['exploration'] = exploration
    if sampling is not None:
        procedural['sampling_policy'] = sampling_policy(sampling)
        # Prompt history is a bounded retrieval window, not a complete validation ledger:
        # older counterexamples and protocol differences may be absent. Keep using local
        # history for skill reasoning, but do not derive control exemptions from it.
        procedural['local_supported_actions'] = []
    freeze_json(folder/'skill_input.json', dict(bank=bank, procedural=procedural,
                                             model=model_identity, config=config))
    # Dedicated accounting database: learned numeric scores never enter skill selection.
    result = await run_region_task(Path(row['task_path']), folder/'sampling', config=config,
        model=model, model_identity=model_identity, pricing=pricing,
        memory_path=folder/'accounting.sqlite3', dataset_group_id=row['collection_dataset'],
        sampling_only=True, bootstrap_source=Path(row['source']),
        historical_references=row['coverage_references'], procedural=procedural,
        resume=(folder/'sampling/state.json').exists())
    if result.get('status') != 'sampled':
        raise RuntimeError('Skill decision did not produce a complete sample')
    return read_json(folder/'sampling/decision.json')


def evaluate_decision(row, folder, *, bank, cache, store, role, settings=None, execution_budget=None):
    """Bound each evaluation invocation without altering historical training parameters."""
    from ..execution_budget import deadline_scope
    from .skill_budget import budget_settings, evaluation_allowance
    if execution_budget is None:
        return _evaluate_decision(row, folder, bank=bank, cache=cache, store=store, role=role, settings=settings)
    cfg = budget_settings(execution_budget)
    decision = read_json(Path(folder)/'sampling/decision.json')
    branch_count = 2 if decision.get('sampling_plan', {}).get('alternative_action_id') else 1
    manifest = read_json(Path(row['source'])/'manifest.json')
    allowance = evaluation_allowance(manifest['configuration']['evaluation_budget'],
        seed_count=len(decision['sampling_seeds']), branch_count=branch_count, settings=cfg)
    freeze_json(Path(folder)/'execution_budget.json', dict(settings=cfg, allowance=allowance))
    with deadline_scope(time.monotonic()+allowance['evaluation_wall_seconds']):
        return _evaluate_decision(row, folder, bank=bank, cache=cache, store=store, role=role,
            settings=settings, infrastructure_retries=cfg['infrastructure_retries'])


def _evaluate_decision(row, folder, *, bank, cache, store, role, settings=None, infrastructure_retries=1):
    """No new full fits. Cache by actual sample, fixed method bank and fit contract."""
    from ..execution_budget import check_deadline, TaskDeadlineExceeded
    from .execution import fit_cached
    from .runtime import _evaluation, _verify_artifacts
    folder, cache = Path(folder), Path(cache)
    verify_bank(bank)
    _verify_artifacts(folder/'sampling')
    decision = read_json(folder/'sampling/decision.json')
    execution = decision['skill_execution']
    selection_input = read_json(folder/'skill_input.json')
    decision_protocol_hash = stable_hash(dict(model=selection_input['model'], config=selection_input['config'],
        history=selection_input['procedural']['history'],
        sampling_policy=selection_input['procedural'].get('sampling_policy')))
    source = Path(row['source'])
    manifest = read_json(source/'manifest.json')
    task = read_json(row['task_path'])
    context = read_json(source/'task_context.json')
    plan = read_json(source/'parameter_feedback/frozen_plan.json')
    effective = MultiProxyConfig(**manifest['configuration'])
    labels = read_typed(source/'core_validation.csv', context['csv_schema'])[task['target']]
    core_features = source/'core_validation_features.csv'
    if not core_features.exists():
        core_features = folder/'sampling/core_validation_features.csv'
    cfg = reward_settings(settings)
    freeze_json(folder/'evaluation_protocol.json', cfg)
    actions = {a['action_id']: a for a in decision['actions']}
    chosen = actions[decision['chosen_action_id']]
    plan_sampling = decision.get('sampling_plan')
    direction = metric_direction(task)
    output, measured = [], []
    new_fits = 0

    def fit_branch(seed, arm_name, action):
        nonlocal new_fits
        check_deadline()
        arm = folder/'sampling/arms'/arm_name/f'seed_{seed}'
        indices = read_json(arm/'coreset_indices.json')
        old = Path(row['coverage_references'][str(seed)]['arm'])
        old_rows = read_json(old/'coreset_0_indices.json')['train_positions']
        positions = indices['train_positions']
        sample_hash = stable_hash(dict(bank=bank['bank_hash'], positions=positions,
                                       csv_sha256=file_hash(arm/'coreset.csv')))
        binding = dict(bank_hash=bank['bank_hash'], sample_hash=sample_hash,
            feature_hash=file_hash(core_features), model_seed=effective.model_seed,
            evaluation_budget=effective.evaluation_budget, stage='coreset',
            environment=environment_identity(), implementation={name: file_hash(Path(__file__).parents[1]/name)
                for name in ('region_bandit/skill_experiment.py', 'region_bandit/execution.py',
                    'region_bandit/prediction_worker.py', 'multi_proxy_runtime.py', 'multi_proxy_io.py',
                    'multi_proxy.py', 'model_registry.py', 'stage1_metrics.py')})
        directory = cache/stable_hash(binding)
        freeze_json(directory/'binding.json', binding)
        reused = False
        if (directory/'result.json').exists():
            saved = read_json(directory/'result.json')
            if any(file_hash(p) != h for p, h in saved['artifacts'].items()):
                raise ValueError('Cached fit evidence changed')
            scores = {int(k): v for k, v in saved['scores'].items()}
            reused = True
        else:
            records = []
            if positions == sorted(old_rows) and file_hash(arm/'coreset.csv') == file_hash(old/'coreset_0.csv'):
                records = historical_records(old)
                artifacts = {str(old/'evaluations.json'): file_hash(old/'evaluations.json')}
                reused = True
            else:
                artifacts = {}
                for i, method in enumerate(manifest['methods']):
                    out = directory/f'method_{i}'
                    outcome = fit_cached(Path(method['code_path']), arm/'coreset.csv', core_features,
                        {**context, 'expected_family': method['family'], 'fit_stage': 'evaluation',
                         **frozen_fit_context(plan, i, 'coreset')}, context['classes'],
                        effective.evaluation_budget, effective.model_seed, out, effective.batch_size,
                        identity=binding, infrastructure_retries=infrastructure_retries)
                    new_fits += outcome.get('fit_calls_this_call', int(not outcome.get('cache_hit', False)))
                    if outcome.get('budget_exhausted'):
                        raise TaskDeadlineExceeded('Skill evaluation allowance exhausted; completed fits remain cached')
                    record = _evaluation(outcome, out, task, labels, context['classes'],
                                         {**method, 'method_id': i}, core=True, manifest=manifest)
                    if outcome['status'] != 'ok':
                        raise RuntimeError('Incomplete paired coreset fit; no promotion is allowed')
                    original = next(r for r in historical_records(old) if r['method_id'] == i)
                    if record['pipeline_spec'] != original['pipeline_spec']:
                        raise ValueError('Method changed during sampling-only validation')
                    records.append(record)
                    for name in ('evaluation.json', 'predictions.npz'):
                        if (out/name).exists():
                            artifacts[str(out/name)] = file_hash(out/name)
            scores = scores_from_records(records, task['metric'])
            write_json(directory/'records.json', records)
            artifacts[str(directory/'records.json')] = file_hash(directory/'records.json')
            write_json(directory/'result.json', dict(scores=scores, artifacts=artifacts))
        check_deadline()
        return dict(arm=arm_name, action={k: action.get(k) for k in ('action_id', 'type', 'region_id',
            'quota_table', 'actual_region_increase_fraction')}, scores=scores, sample_hash=sample_hash,
            cross_region_preview=action['context']['action_summary'].get('cross_region_preview'),
            reused=reused, artifact_directory=str(directory.resolve()),
            sample_indices_path=str((arm/'coreset_indices.json').resolve()))

    # Finish all conditional selection/coreset fits before opening any full scores.
    for seed in decision['sampling_seeds']:
        branches = [fit_branch(seed, 'chosen', chosen)]
        trigger = dict(run_alternative=False, reasons=['single_policy'])
        if plan_sampling is not None:
            trigger = decide_second_branch(plan_sampling, seed, branches[0]['scores'],
                keep_k=cfg['keep_k'], direction=direction, ranking_settings=cfg)
        freeze_json(folder/f'trigger_{seed}.json', trigger)
        if trigger['run_alternative']:
            branches.append(fit_branch(seed, 'alternative', actions[plan_sampling['alternative_action_id']]))
        combined = combine_scores(branches, direction=direction, ranking_settings=cfg,
                                  aggregation=trigger.get('aggregation', 'equal_metric_mean_v1'))
        measured.append((seed, branches, combined, trigger))

    full = scores_from_records([read_json(source/f'full_train/method_{i}/evaluation.json')
                               for i in range(5)], task['metric'])
    for seed, branches, combined, trigger in measured:
        check_deadline()
        scores, sample_hash = combined['scores'], combined['sample_hash']
        old = Path(row['coverage_references'][str(seed)]['arm'])
        before = scores_from_records(historical_records(old), task['metric'])
        for branch in branches:
            branch['reward'] = screening_reward(branch['scores'], full, expected_ids=list(range(5)),
                                                direction=direction, settings=cfg)
        reward = screening_reward(scores, full, expected_ids=list(range(5)), direction=metric_direction(task), settings=cfg)
        baseline = screening_reward(before, full, expected_ids=list(range(5)), direction=metric_direction(task), settings=cfg)
        paired = dict(before=baseline['reward'], after=reward['reward'], delta=None if
            baseline['reward'] is None or reward['reward'] is None else reward['reward']-baseline['reward'])
        fact = dict(task=row['task'], source=row['collection_dataset'], role=role,
            bundle_hash=execution['skill_bundle_hash'], bank_hash=bank['bank_hash'], seed=seed,
            sample_hash=sample_hash, reward=reward, paired_baseline=paired,
            decision_id=decision['decision_id'], skill_execution=execution,
            decision_protocol_hash=decision_protocol_hash,
            action=(branches[0]['action'] if len(branches) == 1 else dict(type='dual',
                action_ids=[b['action']['action_id'] for b in branches], aggregation=combined['aggregation'])),
            cross_region_preview=chosen['context']['action_summary'].get('cross_region_preview'),
            sampling_path=sampling_path(chosen['context'], actions=decision['actions']),
            decision_context={k: chosen['context'].get('region_summary', {}).get(k) for k in
                ('coverage_gap', 'q_fraction', 'c0_fraction', 'mean_density', 'mean_hardness',
                 'mean_disagreement', 'discover_score', 'check_score', 'direction_consistent')},
            core_scores=scores, full_scores=full,
            branches=[{k: v for k, v in b.items() if k != 'reused'} for b in branches], branch_count=len(branches),
            combination=combined, uncertainty=trigger,
            sampling_policy_hash=plan_sampling['protocol_hash'] if plan_sampling else None,
            branch_reward_delta=(branches[1]['reward']['reward']-branches[0]['reward']['reward']
                if len(branches) == 2 and all(b['reward']['reward'] is not None for b in branches) else None),
            combination_gain=(reward['reward']-branches[0]['reward']['reward']
                if reward['reward'] is not None else None),
            artifact_directory=branches[0]['artifact_directory'],
            sample_indices_path=branches[0]['sample_indices_path'] if len(branches) == 1 else None)
        check_deadline()
        output.append(dict(seed=seed, fact_id=store.record_fact(fact), sample_hash=sample_hash,
                           reused=all(b['reused'] for b in branches), reward=reward,
                           branch_count=len(branches), selected_method_ids=reward['retained_ids']))
    write_json(folder/'skill_result.json', dict(facts=output, new_full_fits=0,
        new_core_fits_this_invocation=new_fits,
        logical_core_fits=sum(r['branch_count']*5 for r in output), test_access=False))
    return output


async def propose_edit(store, request_fn, output, *, min_sources=5):
    """One proposal plus one bounded revision; invalid rules never block collection."""
    import inspect
    import json
    from .procedural_skills import FIELDS
    output = Path(output)
    if (output/'proposal.json').exists():
        return read_json(output/'proposal.json')
    context = (read_json(output/'request.json')['context'] if (output/'request.json').exists()
               else store.optimization_context(min_sources))
    request = dict(context=context, allowed_activation_fields=sorted(FIELDS),
        allowed_skill_ids=[s['id'] for s in context['bundle']['skills']],
        evidence_contract=dict(historical='All context.facts are completed DEVELOPMENT experiments.',
            current_inputs=['coverage gaps', 'replicated proxy differences', 'sample/region fractions',
                'feasible action increments', 'quota conflicts', 'cross-region sample coverage previews',
                'search stage', 'normalized search progress/depth', 'previous completed ranking conflicts'],
            forbidden_current_inputs=['branch rewards', 'combination_gain', 'full scores',
                'champion retention', 'validation scores', 'paired_baseline', 'branch_count']),
        activation_contract={'operators': ['gt', 'ge', 'eq'], 'value_range': [0, 1]}, instruction=(
        'Propose exactly ONE bounded edit to one skill activation clause, step or exception. '
        'Use multiple independent development sources and inspect supplied negative evidence. '
        'An association is not causation. Do not repeat rejected edits without new evidence. '
        'A rejection_reason beginning validation_budget_exhausted means insufficient evidence '
        'within the declared validation budget, not demonstrated harm. Preserve that distinction. '
        'Return JSON {skill_id, section: activation|steps|exceptions, operation: ADD|REPLACE|DELETE, '
        'index: zero-based integer, value: clause object or text (null for DELETE), reason: nonempty string '
        'of at most 1000 characters (one short evidence-based sentence; omit generic caveats and restated instructions), '
        'evidence_ids: [supplied development fact_id]}. skill_id MUST be one of allowed_skill_ids, '
        'without an @vN suffix. Activation predicates use only supported '
        'fields in the existing skill format; text edits <=1000 characters AND <=1200 JSON serialized characters. '
        'Do not change actions, reward, validation thresholds, data access, or the model. '
        'Facts may contain two separately evaluated sampling branches and a combined score ranking. '
        'Use branch rewards to study the action contrast, and the combined reward to judge the '
        'executed screening policy. A combined success does not prove either branch alone is good. '
        'A sampling_purpose of bounded_development_exploration labels a program-selected focus contrast, '
        'not a skill recommendation; its primary_only aggregation keeps the primary ranking. '
        'Use the separate branch evidence to form hypotheses, never claim its final ranking tested averaging. '
        'Inspect uncertainty reasons, branch_reward_delta, combination_gain and branch_count. '
        'These quantities exist ONLY in completed historical facts, not in the new task at action-selection '
        'time. Discuss them in reason and cite evidence_ids. The executable value must instead name '
        'conditions measurable before the new action, such as coverage gaps or cross-region conflicts. '
        'Do NOT put reward, branch_reward_delta, combination_gain, branch_count, paired_baseline, full scores '
        'or champion retention into the executable value, even with a historical qualifier. '
        'Avoid rules that require training an unchosen action to decide whether to choose it. '
        'Edits can improve primary/alternative proposals, but cannot change the frozen trigger '
        'thresholds or aggregation weights. Repeated seeds and branches are not independent sources. '
        'For MCTS facts, distinguish model_search_gain from sampling rank calibration: a child can '
        'perform worse while its sampling ranking is correct. Observed parent/child evidence says '
        'nothing about untrained candidates. A null champion_miss is unknown, never zero. '
        'The same three skills cover initialization and expansion; condition rules on supplied '
        'search_features when supported, rather than inventing a separate local skill library. '
        'This is a hypothesis; only a separate paired validation gate may accept it.'))
    if any(f.get('baseline_kind') for f in context['facts']):
        request['instruction'] += (
            ' For relative-protocol facts, reward is the primary branch gain over random_proxy '
            'at initialization or parent_coreset at expansion. A positive branch_reward_delta '
            'favors adjusted sampling over original sampling. Partial references report order_gain '
            'and branch_order_gain separately. Use paired_method_ids and comparable_pair_count to '
            'describe their support. For observed_initial_champion_order_gain, judge both rankings within the same full-observed initial methods using the 35% champion, '
            '65% order composite. For mean_parent_child_direction_gain the expansion reward is '
            'the mean change in parent-child sign accuracy over comparable full-trained children. '
            'For fixed_parent_direction_reward the parent score stays on the inherited sample. '
            'Correcting a direction earns +1, preserving a correct direction +0.25, preserving an '
            'incorrect direction -0.25, and breaking a correct direction -1. '
            'composite_gain is a separate complete-reference diagnostic. A single observed child '
            'provides one pair and cannot establish the best child. combined_reward reports the '
            'aggregated-score evidence separately.')
    freeze_json(output/'request.json', request)
    failure = None
    for attempt in range(2):
        folder = output if attempt == 0 else output/'revision_1'
        payload = dict(request)
        if failure:
            payload['revise_invalid_proposal'] = dict(previous_response=response, error=failure['error'],
                instruction='Produce one corrected bounded hypothesis from the SAME historical evidence. '
                    'Translate outcome observations into executable pre-decision conditions. '
                    'Do not weaken the acceptance gate or change data access. Return only the edit object.')
        freeze_json(folder/'request.json', payload)
        if (folder/'response.json').exists():
            response = read_json(folder/'response.json')
        else:
            if (folder/'started.json').exists():
                raise ValueError('Interrupted optimizer request; inspect the attempt before retrying')
            write_json(folder/'started.json', {'request_hash': stable_hash(payload)})
            response = request_fn(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
            if inspect.isawaitable(response):
                response = await response
            write_json(folder/'response.json', response)
        try:
            response = json.loads(response) if isinstance(response, str) else response
            if not isinstance(response, dict):
                raise ValueError('Proposal must be a structured edit object')
            proposal = store.propose(response, context)
        except ValueError as error:
            failure = dict(parent_hash=context['parent_hash'], new_sources=context['new_sources'],
                attempt=attempt+1, error=str(error), response=response, round_exhausted=attempt == 1)
            if 'development_window' in context:
                failure['development_window'] = context['development_window']
            failure_id = store.record_proposal_failure(failure)
            write_json(folder/'rejection.json', dict(failure_id=failure_id, **failure))
            continue
        write_json(output/'proposal.json', proposal)
        return proposal
    result = dict(status='rejected_invalid', reason=failure['error'], attempts=2,
                  active_bundle=store.active, failure_id=failure_id)
    write_json(output/'proposal.json', result)
    return result
