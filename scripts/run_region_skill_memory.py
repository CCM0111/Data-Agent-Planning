#!/usr/bin/env python3
"""Versioned regional skills: init, run, propose, validate, reject, status.

Run/validate reuse historical five-method full references; never fit full models.
Use --prepare-only to freeze inputs and check source separation without API/fits.
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
from data_agent.paired_sampler_experiment import freeze_json
from data_agent.region_bandit.schemas import read_json, write_json, stable_hash
from data_agent.region_bandit.skill_store import SkillStore, gate_settings
from data_agent.region_bandit.skill_reward import reward_settings
from data_agent.region_bandit.skill_sampling import sampling_policy
from data_agent.region_bandit.skill_budget import budget_settings, batch_budget_plan
from data_agent.region_bandit.skill_experiment import (
    frozen_bank, skill_config, prepare_decision, evaluate_decision, propose_edit)


def task_rows(args):
    rows = read_json(args.tasks)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Task manifest must be a nonempty list')
    inventory = read_json(args.inventory) if args.inventory else None
    result = []
    for original in rows:
        row = dict(original)
        if inventory:
            row['source'] = inventory['sources'][row['task']]
            row['coverage_references'] = {str(a['seed']): dict(arm=a['arm'], source_coreset_id=a['source_coreset_id'])
                for a in inventory['arms'] if a['task'] == row['task'] and a['mode'] == 'coverage_a30_b60_l200'
                and int(a['seed']) in args.seeds}
        if not {'task', 'task_path', 'source', 'collection_dataset', 'coverage_references'} <= set(row):
            raise ValueError('Each task needs task, task_path, source, collection_dataset, coverage_references')
        # Keep user labels out of filesystem path construction.
        row['directory_key'] = stable_hash({'task': row['task'], 'source': row['collection_dataset']})[:24]
        result.append(row)
    if len({r['task'] for r in result}) != len(result):
        raise ValueError('Task names must be distinct')
    return result


def code_identity():
    files = [*sorted((ROOT/'data_agent').rglob('*.py')), Path(__file__), ROOT/'run_region_bandit.py']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files}


def prepare(args, store):
    from run_region_bandit import settings_for, _public_model
    config, model_settings, pricing = settings_for(args.config)
    config = skill_config(config, args.seeds)
    execution = budget_settings(read_json(args.budget_config) if args.budget_config else None)
    config['region_bandit']['runtime']['max_wall_seconds'] = max(
        config['region_bandit']['runtime']['max_wall_seconds'],
        execution['selection_wall_seconds_per_seed']*len(args.seeds))
    config['region_bandit']['runtime']['infrastructure_retries'] = execution['infrastructure_retries']
    model_settings['temperature'] = 0.
    public = {**_public_model(model_settings), 'request_options': model_settings.get('request_options', {})}
    role = 'validation' if args.command == 'validate' else args.role
    rows = task_rows(args)
    store.register_sources(sorted({r['collection_dataset'] for r in rows}), role)
    reward = reward_settings(read_json(args.reward_config) if args.reward_config else None)
    gate = gate_settings(read_json(args.gate_config) if args.gate_config else None)
    sampling = sampling_policy(read_json(args.sampling_policy_config) if args.sampling_policy_config else None)
    if args.command == 'validate':
        candidate = store.candidate(args.edit)
        store.bind_gate(args.edit, gate)
        bundles = [candidate['parent_hash'], candidate['candidate_hash']]
        if len({r['collection_dataset'] for r in rows}) < gate['min_sources']:
            raise ValueError('Validation manifest has too few independent sources')
    else:
        if args.bundle == 'initial':
            from data_agent.region_bandit.procedural_skills import initial_bundle
            bundles = [stable_hash(initial_bundle())]
        elif args.bundle == 'active' and (args.output/'experiment.json').exists():
            bundles = read_json(args.output/'experiment.json')['bundles']
        else:
            bundles = [store.active if args.bundle == 'active' else args.bundle]
    for b in bundles:
        store.bundle(b)
    banks = {r['task']: frozen_bank(r, args.seeds) for r in rows}
    budget_plan = batch_budget_plan({r['task']: read_json(Path(r['source'])/'manifest.json')
        ['configuration']['evaluation_budget'] for r in rows}, seed_count=len(args.seeds),
        policy_count=len(bundles), mode=sampling['mode'], model=public,
        selection_wall_seconds=config['region_bandit']['runtime']['max_wall_seconds'], settings=execution)
    experiment = dict(command=args.command, role=role, rows=rows, banks=banks, bundles=bundles,
        config=config, model=public, pricing=pricing, reward=reward, gate=gate, sampling_policy=sampling,
        execution_budget=execution, budget_plan=budget_plan,
        edit_id=args.edit, require_valid_advice=args.require_valid_advice,
        code=code_identity(), library=str(args.library.resolve()),
        heldout_test_policy='No updates or acceptance decisions from role=test', test_data_access=False)
    freeze_json(args.output/'experiment.json', experiment)
    freeze_json(args.output/'budget_plan.json', budget_plan)
    return experiment, model_settings


async def create_model(settings, env_file):
    from dotenv import load_dotenv
    from internagent.mas.models.openai_model import OpenAIModel
    load_dotenv(env_file)
    settings = dict(settings)
    options = settings.pop('request_options', {})
    model = OpenAIModel(**settings)
    model.request_options = options
    return model


async def execute_batch(args, store):
    experiment, model_settings = prepare(args, store)
    if args.prepare_only:
        return {'status': 'prepared', 'tasks': len(experiment['rows']), 'bundles': experiment['bundles'],
                'api_calls': 0, 'fits': 0, 'budget_plan': experiment['budget_plan']}
    model = await create_model(model_settings, args.env_file)
    pairs, completed = [], []
    try:
        for row in experiment['rows']:
            bank = experiment['banks'][row['task']]
            task_root = args.output/'tasks'/row['directory_key']
            history_path = task_root/'history.json'
            if history_path.exists():
                history = read_json(history_path)
            else:
                history = store.decision_history(row['collection_dataset'], bank['bank_hash'],
                    allow_local=experiment['role'] == 'development',
                    limit=experiment['execution_budget']['history_limit'])
                freeze_json(history_path, history)
            # Both decisions are frozen before either arm's validation outcomes are opened.
            exploration = None
            if experiment['sampling_policy']['version'] == 'adaptive-coreset-v3':
                exploration_path = task_root/'exploration_context.json'
                exploration = (read_json(exploration_path) if exploration_path.exists() else
                    store.exploration_context(row['collection_dataset'],
                        experiment['sampling_policy']['focus_exploration_max_sources']))
                freeze_json(exploration_path, exploration)
            for bundle_hash in experiment['bundles']:
                decision = await prepare_decision(row, task_root/bundle_hash, bank=bank,
                    bundle=store.bundle(bundle_hash), history=history,
                    config=experiment['config'], model=model, model_identity=experiment['model'],
                    pricing=experiment['pricing'], sampling=experiment['sampling_policy'], exploration=exploration)
                if experiment['require_valid_advice']:
                    require_valid_advice(decision)
                if decision.get('sampling_plan', {}).get('purpose') == 'exploration_control':
                    store.reserve_focus_exploration(row['collection_dataset'],
                        experiment['sampling_policy']['focus_exploration_max_sources'])
            results = [evaluate_decision(row, task_root/b, bank=bank,
                cache=args.library.parent/'skill_fit_cache', store=store, role=experiment['role'],
                settings=experiment['reward'], execution_budget=experiment['execution_budget'])
                for b in experiment['bundles']]
            if len(results) == 2:
                if [r['seed'] for r in results[0]] != [r['seed'] for r in results[1]]:
                    raise ValueError('Paired seed mismatch')
                pairs.extend([[a['fact_id'], b['fact_id']] for a, b in zip(*results)])
            completed.append(row['task'])
            write_json(args.output/'progress.json', {'completed': completed, 'total': len(experiment['rows'])})
        if args.command == 'validate':
            # No failed or omitted task can be silently dropped from the gate.
            if len(pairs) != len(experiment['rows'])*len(args.seeds):
                raise ValueError('Incomplete predeclared validation matrix')
            result = store.gate(args.edit, pairs, settings=experiment['gate'])
            write_json(args.output/'gate.json', result)
        else:
            result = dict(status='completed', role=experiment['role'], tasks=len(completed),
                          bundle=experiment['bundles'][0], active_bundle=store.active,
                          next_step='propose after five independent development sources have fresh decisions')
        write_json(args.output/'result.json', result)
        return result
    finally:
        await model.client.close()


async def optimize(args, store):
    from run_region_bandit import settings_for
    if args.patch:
        freeze_json(args.output/'patch.json', read_json(args.patch))
        if (args.output/'proposal.json').exists():
            return read_json(args.output/'proposal.json')
        context = (read_json(args.output/'development_context.json')
            if (args.output/'development_context.json').exists() else store.optimization_context(args.min_new_sources))
        freeze_json(args.output/'development_context.json', context)
        result = store.propose(read_json(args.patch), context)
        write_json(args.output/'proposal.json', result)
        return result
    _, settings, pricing = settings_for(args.config)
    settings['temperature'] = 0.
    model = await create_model(settings, args.env_file)
    from data_agent.multi_proxy import _llm_json_once
    try:
        async def request(prompt):
            payload = json.loads(prompt)
            name = 'optimizer_revision_1.json' if 'revise_invalid_proposal' in payload else 'optimizer_call.json'
            return await _llm_json_once(model, prompt, args.output/name, [], pricing or {})
        return await propose_edit(store, request, args.output, min_sources=args.min_new_sources)
    finally:
        await model.client.close()


def require_valid_advice(decision):
    """Scientific runs must not count an API/schema failure as a policy result."""
    execution = decision.get('skill_execution', {})
    if execution.get('decision_status') not in ('selected', 'abstain', 'baseline_only'):
        raise RuntimeError('Valid skill advice required before fitting: '+str(execution.get('error')))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['init', 'run', 'propose', 'validate', 'reject', 'status'])
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--tasks', type=Path)
    p.add_argument('--inventory', type=Path)
    p.add_argument('--config', type=Path, default=ROOT/'config/region_skill_memory.yaml')
    p.add_argument('--env-file', type=Path, default=Path('/home/cmcui/InternAgent/.env'))
    p.add_argument('--role', choices=['development', 'validation', 'test'], default='development')
    p.add_argument('--bundle', default='active', help='active, initial, or immutable bundle hash')
    p.add_argument('--seeds', nargs='+', type=int, default=[20261026])
    p.add_argument('--reward-config', type=Path, default=ROOT/'config/region_skill_reward.json')
    p.add_argument('--gate-config', type=Path, default=ROOT/'config/region_skill_gate.json')
    p.add_argument('--budget-config', type=Path, default=ROOT/'config/region_skill_budget.json',
                   help='Execution time, infrastructure retries and history allowance; frozen fit budgets are preserved')
    p.add_argument('--sampling-policy-config', type=Path, default=ROOT/'config/region_skill_sampling.json',
                   help='Frozen single/adaptive/dual policy, uncertainty triggers and thresholds')
    p.add_argument('--edit', help='Candidate edit ID returned by propose')
    p.add_argument('--patch', type=Path, help='Explicit structured patch instead of an optimizer API call')
    p.add_argument('--min-new-sources', type=int, default=5,
                   help='Distinct development sources with fresh decisions; previously used sources may recur')
    p.add_argument('--reason', help='Explicit rejection reason')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--require-valid-advice', action='store_true',
                   help='Stop before fitting if the real skill API failed or returned invalid advice')
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.command in ('run', 'validate', 'propose') and args.output is None:
        p.error('--output is required')
    if args.command in ('run', 'validate') and args.tasks is None:
        p.error('--tasks is required')
    if args.command in ('validate', 'reject') and not args.edit:
        p.error('--edit is required')
    if args.min_new_sources < 5:
        p.error('At least five independent development sources with fresh decisions are required')
    args.library = args.library.resolve()
    args.library.parent.mkdir(parents=True, exist_ok=True)
    if args.output:
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=True)
    # Serializes promotion, source registration, decisions and shared fit caches.
    with args.library.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with SkillStore(args.library) as store:
            if args.command in ('init', 'status'):
                result = dict(active=store.active, bundle=store.bundle(), development_facts=len(store.development_facts()),
                    candidates=[dict(edit_id=r[0], status=r[1]) for r in store.db.execute('SELECT id,status FROM edits')])
            elif args.command == 'reject':
                store.reject_pending(args.edit, args.reason)
                result = store.candidate(args.edit)
            elif args.command == 'propose':
                result = asyncio.run(optimize(args, store))
            else:
                result = asyncio.run(execute_batch(args, store))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
