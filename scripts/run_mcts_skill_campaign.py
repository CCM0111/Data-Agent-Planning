#!/usr/bin/env python3
"""Stage MCTS trajectories, propose bounded skills, replay frozen checkpoints.

--prepare-only validates/freezes the plan without a model or any fits/API calls.
Executing the same command resumes committed stages; never use legacy outputs.
"""
import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_agent.coreset import file_hash
from data_agent.mcts_search import SearchBudgetExceeded
from data_agent.mcts_skill_workflow import run_skill_task
from data_agent.method_regeneration import RegenerationExhausted
from data_agent.paired_sampler_experiment import freeze_json
from data_agent.region_bandit.mcts_skill_reward import bind_validation, checkpoint_gate, gate_settings, reward_settings, INITIAL_TOP3_VERSION
from data_agent.region_bandit.mcts_skill_validation import ValidationRunner, IncompleteValidation
from data_agent.region_bandit.mcts_skill_initial import replay_initial_checkpoint
from data_agent.region_bandit.mcts_skills import RoundSkills, prepare_replay_inputs
from data_agent.region_bandit.matched_checkpoints import matched_cases
from data_agent.workflow import WorkflowBackend
from data_agent.execution_budget import TaskDeadlineExceeded
from data_agent.region_bandit.schemas import read_json, write_json, stable_hash
from data_agent.region_bandit.skill_experiment import propose_edit
from data_agent.region_bandit.skill_sampling import mcts_sampling_policy
from data_agent.region_bandit.skill_store import SkillStore, InsufficientDevelopmentEvidence
from run_data_agent_workflow import settings_for
from run_region_bandit import _public_model
from scripts.run_mcts_collection import freeze_config
from scripts.run_region_skill_campaign import source_split, event
from scripts.run_region_skill_memory import code_identity, create_model


def stage_plan(sources, *, full_target=11, batch_size=5, max_proposals=15, direct_tuning=True):
    if (type(batch_size) is not int or batch_size < 5 or type(max_proposals) is not int
            or not 1 <= max_proposals <= 15):
        raise ValueError('Use batches of at least five sources and at most fifteen proposals')
    batches = [sources[i:i+batch_size] for i in range(0, len(sources), batch_size)]
    rows = [dict(target=target, sources=batch) for target in (3, 6, 9, full_target) for batch in batches]
    eligible = [i for i, row in enumerate(rows) if len(row['sources']) >= 5 and (not direct_tuning or row['target'] <= 9)]
    count = min(max_proposals, len(eligible))
    opportunities = ({eligible[-1]} if count == 1 else
        {eligible[round(i*(len(eligible)-1)/(count-1))] for i in range(count)}) if count else set()
    return [dict(index=i, propose=i in opportunities, **row) for i, row in enumerate(rows)]


def matched_stage_plan(sources, *, full_target, max_proposals):
    """Finish each dataset once; allow one edit at source 6 and every five thereafter."""
    if type(max_proposals) is not int or not 1 <= max_proposals <= 15:
        raise ValueError('Use between one and fifteen proposal opportunities')
    return [dict(index=i, target=full_target, sources=[source],
                 propose=i >= 5 and i % 5 == 0 and i // 5 <= max_proposals)
            for i, source in enumerate(dict.fromkeys(sources))]


def initialize(args):
    rows = read_json(args.tasks)
    if not isinstance(rows, list) or not rows or len({r['task'] for r in rows}) != len(rows):
        raise ValueError('Distinct named task rows required')
    pilot = getattr(args, 'pilot_first_seven', False)
    if pilot:
        rows = rows[:7]
        if len(rows) != 7 or len({r['collection_dataset'] for r in rows}) != 7:
            raise ValueError('The seven-task pilot requires seven distinct dataset sources')
    rows = [dict(task=r['task'], task_path=str(Path(r['task_path']).resolve()),
        collection_dataset=r['collection_dataset'],
        directory_key=stable_hash(dict(task=r['task'], source=r['collection_dataset']))[:24]) for r in rows]
    matched = getattr(args, "validation_mode", "matched") == "matched"
    roles = (dict(development=list(dict.fromkeys(r["collection_dataset"] for r in rows)), validation=[], test=[])
             if matched else dict(development=[r['collection_dataset'] for r in rows[:5]],
                  validation=[r['collection_dataset'] for r in rows[5:]], test=[]) if pilot else
             source_split(rows, args.split_seed, development=args.development_sources,
                          validation=args.validation_sources))
    gate = gate_settings(dict(first_sources=2, max_sources=2) if pilot else None)
    validation_seeds = getattr(args, 'validation_seeds', None)
    if validation_seeds is not None and (not validation_seeds or
            len(validation_seeds) != len(set(validation_seeds)) or
            any(type(s) is not int or not 0 <= s < 2**32 for s in validation_seeds)):
        raise ValueError('Validation seeds must be distinct nonnegative uint32 integers')
    seed_count = len(validation_seeds) if validation_seeds is not None else 1
    validation_budget = dict(max_low_fits=gate['max_sources']*9*4*seed_count,
        max_api_calls=gate['max_sources']*2*4*seed_count, max_wall_seconds=86400)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    config_path = freeze_config(output, args.config, 'unified')
    config, search, memory, modules, model, pricing, subgroups = settings_for(config_path)
    full_target = search.max_rollout_successes
    if full_target not in (11, 12) or search.base_expansions != (full_target - 4 if search.root_trials else 6) or search.max_expansions != full_target - 3:
        raise ValueError('Unexpected primary expansion reservations for the frozen full target')
    if matched:
        stages = matched_stage_plan(roles['development'], full_target=full_target, max_proposals=args.max_proposals)
    else:
        stages = stage_plan(roles['development'], full_target=full_target, max_proposals=args.max_proposals,
                            direct_tuning=search.root_trials)
    if model.get('base_url', '').rstrip('/') != 'https://api.deepseek.com':
        raise ValueError('This experiment uses the official direct DeepSeek endpoint')
    policy = mcts_sampling_policy()
    public = _public_model(model)
    code = code_identity()
    code.update({str(p.relative_to(ROOT)): file_hash(p) for p in (Path(__file__), ROOT/'run_data_agent_workflow.py')})
    manifest = dict(protocol='staged-unified-skill-mcts-v3', rows=rows, roles=roles,
        config=config, search=search.to_dict(), memory=memory, model=public,
        pricing=pricing, subgroups=subgroups, sampling=policy, reward=reward_settings({'version': INITIAL_TOP3_VERSION}), gate=gate,
        stages=stages, pilot_first_seven=pilot, validation_sampling_seeds=validation_seeds,
        evaluation_scope=('Seven-task workflow pilot: five development, two validation, no heldout; '
            'promotions apply only to this isolated pilot library, not evidence of generalization'
            if pilot else 'Source-isolated development, paired validation and heldout evaluation'),
        validation_round_numbers=[0, 4],  # 0 is initialization, not another expansion.
        validation_budget=validation_budget,
        validation_task_by_source={s: sorted((r['task'] for r in rows if r['collection_dataset'] == s))[0]
                                   for s in roles['validation']},
        task_hashes={r['task']: file_hash(r['task_path']) for r in rows}, code=code,
        max_successful_full_fits=len(rows)*full_target,
        max_external_validation_low_fits=sum(s['propose'] for s in stages)*validation_budget['max_low_fits'],
        max_optimizer_calls=sum(s['propose'] for s in stages)*2,
        initialization_sampling=policy['mode'],
        replicate_scope='One MCTS trajectory per task; base sampling seed 20261026, model seed inherited. '
            'Validation changes regional sampling seeds only: fixed baseline, methods, model seed, splits and full scores. '
            'Derived round seeds and multiple checkpoints are not independent experiment replicates.',
        validation_scope='Initialization and fourth expansion; shared frozen development/local history; '
            'paired ranking of observed full methods only',
        cohort_limitation='Previously investigated cohort; test sources held out from THIS skill optimization only',
        test_data_access=False)
    if matched:
        history_runs = [Path(path).resolve() for path in getattr(args, 'history_run', [])]
        if any(not path.is_dir() for path in history_runs):
            raise ValueError('Historical replay directory does not exist')
        validation_budget.update(max_low_fits=4*2*2*5*seed_count, max_api_calls=4*2*2*seed_count)
        manifest.update(protocol='matched-skill-mcts-v4', validation_mode='matched',
            proposal_schedule=dict(unit='dataset_source', first_after=6, interval=5),
            history_runs=list(map(str, history_runs)), validation_round_numbers=[],
            max_external_validation_low_fits=sum(s['propose'] for s in stages)*validation_budget['max_low_fits'],
            validation_scope='Four matched frozen decisions, prefer initialization, three sampling wins',
            evaluation_scope='Paired low-only historical replay in a shared task pool',
            cohort_limitation='Development and validation share datasets; engineering promotion screen')
    freeze_json(output/'campaign.json', manifest)
    with SkillStore(output/'skills.sqlite') as store:
        for role, sources in roles.items():
            if sources:
                store.register_sources(sources, role)
        path = output/'initial_bundle.json'
        if not path.exists():
            freeze_json(path, dict(bundle_id=store.active, bundle=store.bundle()))
    return manifest, model


def task_folder(root, row):
    return root/'tasks'/row['directory_key']


def validation_cases(root, manifest, proposal_number):
    """Fixed rotation and round numbers, independent of any replay reward."""
    sources = manifest['roles']['validation']
    offset = proposal_number % len(sources)
    sources = (sources[offset:]+sources[:offset])[:manifest['gate']['max_sources']]
    cases = []
    for source in sources:
        row = next(r for r in manifest['rows'] if r['task'] == manifest['validation_task_by_source'][source])
        folder = task_folder(root, row)
        result_path = folder/'result.json'
        failed = result_path.is_file() and read_json(result_path).get('status') == 'method_repair_exhausted'
        state = {} if failed else read_json(folder/'search/search_state.json')
        for number in manifest['validation_round_numbers']:
            if failed:
                checkpoint_folder = None
            elif number == 0:
                commit = folder/'initial_skill_fact.json'
                checkpoint_folder = read_json(commit).get('checkpoint_folder') if commit.exists() else None
            else:
                batch = next((b for b in state['expansions'] if b['round_number'] == number), None)
                checkpoint_folder = (batch or {}).get('region_feedback', {}).get('checkpoint_folder')
            frozen = (read_json(Path(checkpoint_folder)/'checkpoint.json')['frozen']
                      if checkpoint_folder and (Path(checkpoint_folder)/'checkpoint.json').exists() else None)
            seeds = manifest.get('validation_sampling_seeds') or [frozen['seed'] if frozen else 20261026]
            for seed in seeds:
                cases.append(dict(source=source, task=row['task'], round_number=number,
                    stage='initialization' if number == 0 else 'mcts_expansion',
                    checkpoint_id=frozen['checkpoint_id'] if frozen else stable_hash(dict(source=source, missing_round=number)),
                    seed=seed, checkpoint_folder=checkpoint_folder))
    return cases


async def validate_candidate(root, manifest, store, edit, number, run_task, *, model=None):
    directory = root/'proposals'/f'{number:03d}'
    plan_path = directory/'validation_plan.json'
    matched = manifest.get('validation_mode') == 'matched'
    if plan_path.exists():
        cases = read_json(plan_path)
    elif matched:
        cases = matched_cases([root, *manifest['history_runs']], edit, store, manifest['validation_sampling_seeds'])
    else:
        cases = validation_cases(root, manifest, number)
    freeze_json(plan_path, cases)
    bind_validation(store, edit['edit_id'], cases, settings=manifest['gate'], matched=matched)
    if not matched:
        history_path = directory/'development_history.json'
        remote_history = (read_json(history_path) if history_path.exists() else
                          store.decision_history('__paired_validation__', allow_local=False, limit=12))
        freeze_json(history_path, remote_history)
    pairs_path = directory/'pairs.json'
    pairs = read_json(pairs_path) if pairs_path.exists() else []
    completed = {(f['checkpoint_id'], f['seed']) for f in (store.fact(a) for a, _ in pairs)}
    failures_path = directory/'incomplete_cases.json'
    failures = read_json(failures_path) if failures_path.exists() else {}
    sources = list(dict.fromkeys(c['source'] for c in cases))
    exhausted = False
    if matched:
        limits = (len(sources),)
        if len({c['checkpoint_id'] for c in cases}) < 4:
            gate = checkpoint_gate(store, edit['edit_id'], [], final=True)
            write_json(directory/'gate.json', gate)
            return gate
    else:
        first_sources = set(sources[:manifest['gate']['first_sources']])
        already_expanding = any(store.fact(a)['source'] not in first_sources for a, _ in pairs)
        limits = ((manifest['gate']['max_sources'],) if already_expanding else
                  (manifest['gate']['first_sources'], manifest['gate']['max_sources']))
    loaded_task, loaded_execution = None, None
    for limit in limits:
        for case in cases:
            case_key = f"{case['checkpoint_id']}:{case['seed']}"
            if (case['source'] not in sources[:limit] or (case['checkpoint_id'], case['seed']) in completed
                    or case_key in failures):
                continue
            if not case['checkpoint_folder']:
                continue  # The fixed missing case stays incomplete; never replace it using its reward.
            if matched:
                trajectory = Path(case['trajectory'])
                if loaded_task != trajectory:
                    cfg = read_json(trajectory/'skill_workflow.json')
                    backend = WorkflowBackend(trajectory/'selected_source', 0, directory/'execution'/stable_hash(str(trajectory)),
                        model=model, pricing=manifest['pricing'], subgroup_rules=manifest['subgroups'])
                    controller = RoundSkills(trajectory/'sampling', config=cfg['config'],
                        memory_path=root/'skills.sqlite', source=case['source'], task=case['task'],
                        role='validation', shared_pool_replay=True, bundle_id=edit['parent_hash'], model_identity=cfg['model'],
                        sampling_policy=cfg['sampling'], reward_settings=cfg['reward'])
                    loaded_task = trajectory
            else:
                row = next(r for r in manifest['rows'] if r['task'] == case['task'])
                if loaded_task != row['task']:
                    result, engine, controller = await run_task(row, 'validation',
                        read_json(root/'initial_bundle.json')['bundle_id'], manifest['search']['max_rollout_successes'])
                    if result.get('status') == 'method_repair_exhausted':
                        failures[case_key] = dict(reason=result['stop_reason'])
                        write_json(failures_path, failures)
                        continue
                    loaded_task, loaded_execution = row['task'], (engine, controller)
                else:
                    engine, controller = loaded_execution
                backend = engine.backend
            checkpoint = read_json(Path(case['checkpoint_folder'])/'checkpoint.json')
            history = checkpoint['frozen']['history']
            if not matched:
                local = [f for f in history if f.get('scope') in ('previous_local_checkpoint', 'task_local')]
                history = [*remote_history, *local]
            effective_frozen = prepare_replay_inputs(controller, checkpoint, history,
                                                     sampling_seed=case['seed'])
            try:
                with ValidationRunner(backend, directory/'validation_budget.json',
                                      **manifest['validation_budget']) as runner:
                    outputs = [directory/'replays'/case['checkpoint_id']/f'seed_{case["seed"]}'/b
                               for b in (edit['parent_hash'], edit['candidate_hash'])]
                    # Freeze BOTH action decisions before evaluating either replay.
                    for bundle, output in zip((edit['parent_hash'], edit['candidate_hash']), outputs):
                        await controller._choose(runner, output, effective_frozen, bundle)
                    values = []
                    for bundle, output in zip((edit['parent_hash'], edit['candidate_hash']), outputs):
                        if checkpoint['frozen'].get('stage') == 'initialization':
                            value = await replay_initial_checkpoint(controller, runner,
                                case['checkpoint_folder'], bundle, output, history=history,
                                sampling_seed=case['seed'])
                        else:
                            value = await controller.replay_checkpoint(runner,
                                case['checkpoint_folder'], bundle, output, history=history,
                                sampling_seed=case['seed'])
                        values.append(value)
                pairs.append([v['fact_id'] for v in values]); write_json(pairs_path, pairs)
                completed.add((case['checkpoint_id'], case['seed']))
            except IncompleteValidation as error:
                failures[case_key] = dict(reason=str(error),
                    conclusion='Missing paired evidence; no claim of harm or benefit')
                write_json(failures_path, failures)
            except (SearchBudgetExceeded, TaskDeadlineExceeded) as error:
                write_json(directory/'validation_budget_exhausted.json', dict(reason=str(error)))
                exhausted = True; break
        gate = checkpoint_gate(store, edit['edit_id'], pairs,
                               final=matched or exhausted or limit == manifest['gate']['max_sources'])
        write_json(directory/'gate.json', gate)
        if gate['status'] != 'pending':
            return gate
    return gate


async def execute(args, manifest, settings):
    root = args.output.resolve(); store_path = root/'skills.sqlite'
    model = await create_model(settings, args.env_file)
    try:
        async def run_task(row, role, bundle, target):
            folder = task_folder(root, row)
            path = folder/'result.json'
            previous = read_json(path) if path.exists() else {}
            if previous.get('status') == 'method_repair_exhausted':
                return previous, None, None
            try:
                return await run_skill_task(row, folder, config=manifest['config'],
                    search=manifest['search'], memory=manifest['memory'], model=model,
                    model_identity=manifest['model'], store_path=store_path, role=role,
                    bundle_id=bundle, pricing=manifest['pricing'], subgroups=manifest['subgroups'],
                    sampling=manifest['sampling'], reward=manifest['reward'], full_success_target=target)
            except RegenerationExhausted as error:
                result = {**previous, 'status': 'method_repair_exhausted', 'stop_reason': str(error),
                          'full_fit_successes': previous.get('full_fit_successes', 0)}
                write_json(path, result)
                event(root, 'task_method_repair_exhausted', task=row['task'], role=role, reason=str(error))
                return result, None, None

        initial = read_json(root/'initial_bundle.json')['bundle_id']
        full_target = manifest['search']['max_rollout_successes']
        for source in manifest['roles']['validation']:
            row = next(r for r in manifest['rows'] if r['task'] == manifest['validation_task_by_source'][source])
            event(root, 'validation_trajectory', task=row['task'], target=full_target)
            await run_task(row, 'validation', initial, full_target)
        proposal_number = 0
        with SkillStore(store_path) as store:
            for stage in manifest['stages']:
                stage_root = root/'stages'/f'{stage["index"]:03d}'
                if (stage_root/'completed.json').exists():
                    if stage['propose']: proposal_number += 1
                    continue
                pinned = (read_json(stage_root/'bundle.json')['bundle_id']
                          if (stage_root/'bundle.json').exists() else store.active)
                freeze_json(stage_root/'bundle.json', dict(bundle_id=pinned))
                results = []
                for row in manifest['rows']:
                    if row['collection_dataset'] not in stage['sources']:
                        continue
                    event(root, 'development', task=row['task'], target=stage['target'], stage=stage['index'])
                    result, _, _ = await run_task(row, 'development', pinned, stage['target'])
                    results.append(dict(task=row['task'], status=result['status'],
                        full_successes=result['full_fit_successes'], stop_reason=result.get('stop_reason')))
                proposal = None
                if stage['propose']:
                    directory = root/'proposals'/f'{proposal_number:03d}'
                    from data_agent.multi_proxy import _llm_json_once
                    async def request(prompt):
                        name = 'optimizer_revision.json' if 'revise_invalid_proposal' in json.loads(prompt) else 'optimizer.json'
                        return await _llm_json_once(model, prompt, directory/name, [], manifest['pricing'] or {})
                    try:
                        proposal = await propose_edit(store, request, directory, min_sources=5)
                        if proposal.get('edit_id'):
                            current = store.candidate(proposal['edit_id'])
                            if current['status'] in ('candidate', 'pending'):
                                event(root, 'paired_skill_validation', proposal=proposal_number)
                                await validate_candidate(root, manifest, store, current, proposal_number, run_task, model=model)
                    except InsufficientDevelopmentEvidence as error:
                        proposal = dict(status='insufficient_fresh_evidence', reason=str(error))
                        write_json(directory/'skipped.json', proposal)
                    proposal_number += 1
                freeze_json(stage_root/'completed.json', dict(results=results, proposal=proposal, active=store.active))
            # Freeze once; no further promotion after heldout sources start.
            final_path = root/'final_bundle.json'
            final = read_json(final_path)['bundle_id'] if final_path.exists() else store.active
            freeze_json(final_path, dict(bundle_id=final, bundle=store.bundle(final)))
            for row in manifest['rows']:
                if row['collection_dataset'] in manifest['roles']['test']:
                    event(root, 'heldout_dataset', task=row['task'], target=full_target)
                    await run_task(row, 'test', final, full_target)
                elif (row['collection_dataset'] in manifest['roles']['validation']
                      and row['task'] != manifest['validation_task_by_source'][row['collection_dataset']]):
                    # Additional targets of a validation source share its role,
                    # but cannot replace the predeclared representative checkpoints.
                    event(root, 'additional_validation_target', task=row['task'], target=full_target)
                    await run_task(row, 'validation', final, full_target)
            statuses = [dict(edit_id=r[0], status=r[1]) for r in store.db.execute('SELECT id,status FROM edits')]
        incomplete, task_results = [], []
        for row in manifest['rows']:
            path = task_folder(root, row)/'result.json'
            summary = read_json(path) if path.exists() else {}
            if summary.get('status') == 'method_repair_exhausted' or summary.get('full_fit_successes', 0) != full_target:
                incomplete.append(row['task'])
            state_path = task_folder(root, row)/'search/search_state.json'
            state = read_json(state_path) if state_path.exists() else {}
            initial_scores = [state['candidates'][i]['full_report']['score']
                for i in state.get('initial_selected', [])
                if state.get('candidates', {}).get(i, {}).get('full_report')]
            sign = 1 if summary.get('direction') == 'maximize' else -1
            initial_best = (max(initial_scores) if sign == 1 else min(initial_scores)) if initial_scores else None
            final_best = summary.get('best_full_score')
            task_results.append(dict(task=row['task'], source=row['collection_dataset'],
                role=next(role for role, sources in manifest['roles'].items() if row['collection_dataset'] in sources),
                status=summary.get('status', 'missing'), stop_reason=summary.get('stop_reason'),
                full_fit_successes=summary.get('full_fit_successes', 0),
                completed_expansions=summary.get('completed_expansions', 0), metric=summary.get('metric'),
                initial_best_full_score=initial_best, final_best_full_score=final_best,
                search_gain=None if initial_best is None or final_best is None else sign*(final_best-initial_best)))
        status = 'completed' if not incomplete else 'completed_with_resource_limits'
        if any(t['status'] == 'method_repair_exhausted' for t in task_results):
            status = 'completed_with_task_failures'
        result = dict(status=status,
            active_bundle=final, edits=statuses, accepted=sum(r['status'] == 'accepted' for r in statuses),
            incomplete_tasks=incomplete, test_data_access=False,
            evaluation_scope=manifest['evaluation_scope'], tasks=task_results,
            result_scope='Search validation improvement over the initial three full methods; '
                'not a paired baseline-sampling trajectory comparison or final test estimate')
        write_json(root/'campaign_result.json', result); event(root, **result)
        return result
    finally:
        await model.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'config/workflow_skill_mcts.yaml')
    parser.add_argument('--env-file', type=Path, default=Path('/home/cmcui/InternAgent/.env'))
    parser.add_argument('--split-seed', type=int, default=20261116)
    parser.add_argument('--development-sources', type=int, default=40)
    parser.add_argument('--validation-sources', type=int, default=8)
    parser.add_argument('--max-proposals', type=int, default=15)
    parser.add_argument('--pilot-first-seven', action='store_true',
                        help='First seven manifest tasks only (legacy mode splits five development and two validation)')
    parser.add_argument('--validation-seeds', type=int, nargs='+',
                        help='Regional sampling seeds for paired validation only; online search remains one trajectory')
    parser.add_argument('--validation-mode', choices=('matched', 'legacy'), default='matched')
    parser.add_argument('--history-run', type=Path, default=[], action='append',
                        help='Additional frozen skill campaign or task trajectory for matched replay')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, settings = initialize(args)
        if args.prepare_only:
            result = dict(status='prepared', tasks=len(manifest['rows']), source_roles={k:len(v) for k,v in manifest['roles'].items()},
                full_fits_cap=manifest['max_successful_full_fits'],
                validation_low_fits_cap=manifest['max_external_validation_low_fits'],
                proposal_opportunities=sum(s['propose'] for s in manifest['stages']), api_calls=0, fits=0)
        else:
            result = asyncio.run(execute(args, manifest, settings))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
