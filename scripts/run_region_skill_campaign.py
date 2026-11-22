#!/usr/bin/env python3
"""Resumable source-isolated development, paired skill gates and heldout evaluation.

All source roles and repair requirements are fixed before calls or training.
Failed stages stop the campaign; restarting the same command resumes artifacts.
"""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_agent.coreset import file_hash
from data_agent.paired_sampler_experiment import freeze_json
from data_agent.region_bandit.schemas import read_json, write_json, stable_hash
from data_agent.region_bandit.skill_store import SkillStore, InsufficientDevelopmentEvidence
from data_agent.region_bandit.skill_experiment import frozen_bank
from scripts.run_region_skill_memory import code_identity, create_model


def source_split(rows, seed, development=40, validation=8):
    sources = sorted({r['collection_dataset'] for r in rows})
    if development < 5 or validation < 6 or len(sources) <= development+validation:
        raise ValueError('Need development, at least six validation sources, and a heldout group')
    random.Random(seed).shuffle(sources)
    return dict(development=sources[:development], validation=sources[development:development+validation],
                test=sources[development+validation:])


def export_rebuilt_references(row, source, output, seeds):
    """Adapt coverage replicas without changing their samples or frozen methods."""
    from data_agent.region_bandit.historical_replay import scores_from_records
    source, output = Path(source), Path(output)
    manifest = read_json(source/'manifest.json')
    if manifest['status'] != 'success':
        raise ValueError('Rebuilt source is incomplete')
    metric = manifest['task']['metric']
    scores_from_records([read_json(source/f'full_train/method_{i}/evaluation.json') for i in range(5)], metric)
    design = read_json(source/'coverage_comparison_design.json')['arms']
    evaluations = read_json(source/'evaluations.json')
    references = {}
    for seed in seeds:
        matches = [a for a in design if a['seed'] == seed and a['mechanism'] == 'coverage_a30_b60_l200']
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous rebuilt coverage seed')
        cid = matches[0]['coreset_id']
        records = [dict(r, coreset_id=0) for r in evaluations if r['coreset_id'] == cid]
        scores_from_records(records, metric)
        arm = output/str(seed)
        arm.mkdir(parents=True, exist_ok=True)
        for suffix in ('.csv', '_indices.json'):
            src, dest = source/f'coreset_{cid}{suffix}', arm/f'coreset_0{suffix}'
            if dest.exists() and file_hash(src) != file_hash(dest):
                raise ValueError('Rebuilt reference export changed')
            if not dest.exists():
                shutil.copyfile(src, dest)
        freeze_json(arm/'evaluations.json', records)
        references[str(seed)] = dict(arm=str(arm.resolve()), source_coreset_id=cid)
    return dict(row, source=str(source.resolve()), coverage_references=references,
                reference_origin='rebuilt_protocol_safe')


def event(root, status, **details):
    value = dict(status=status, time=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), **details)
    write_json(root/'campaign_progress.json', value)
    with (root/'events.jsonl').open('a') as handle:
        handle.write(json.dumps(value, ensure_ascii=False)+'\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def initialize(args):
    rows = read_json(args.tasks)
    audit = read_json(args.preflight)
    if (audit['checked'] != len(rows) or audit['total_tasks'] != len(rows)
            or {r['task'] for r in audit['audit']} != {r['task'] for r in rows}):
        raise ValueError('A complete source preflight is required')
    if any(r['status'] not in ('ready', 'requires_rebuild') for r in audit['audit']):
        raise ValueError('Unknown preflight state')
    roles = source_split(rows, args.split_seed)
    config_names = ['region_skill_memory.yaml', 'region_skill_reward.json', 'region_skill_gate.json',
                    'region_skill_sampling.json', 'region_skill_budget.json', 'coverage_expanded.yaml',
                    'coverage_regression30.yaml']
    configs = {}
    config_dir = Path(getattr(args, 'config_dir', None) or ROOT/'config')
    for name in config_names:
        source, dest = config_dir/name, args.output/'configs'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and file_hash(dest) != file_hash(source):
            raise ValueError('Campaign configuration changed: '+name)
        if not dest.exists():
            shutil.copyfile(source, dest)
        configs[name] = file_hash(dest)
    code = code_identity()
    for path in (Path(__file__), ROOT/'run_mcts_data_agent.py'):
        code[str(path.relative_to(ROOT))] = file_hash(path)
    manifest = dict(rows=rows, roles=roles, split_seed=args.split_seed, seeds=args.seeds,
        rebuild_tasks=[r['task'] for r in audit['audit'] if r['status'] == 'requires_rebuild'],
        preflight_sha256=file_hash(args.preflight), configs=configs, code=code,
        development_batch_sources=5, max_proposal_rounds=8,
        proposal_trigger='fresh decisions from five distinct development sources; sources may recur',
        max_proposal_calls_per_round=2,
        invalid_proposal_policy='Record both failures, retain active bundle, continue development',
        pending_policy='After one complete inconclusive validation, archive candidate, retain parent, allow later proposals',
        max_validation_passes_per_candidate=1,
        final_validation_policy='Evaluate final active bundle without further promotion',
        sampling_seeds_only=True, outer_split='existing frozen task split', model_seed=101,
        require_valid_advice=True, new_full_fits_per_rebuilt_source=5,
        reuse_original_fit_budgets=True, rebuilt_profile='coverage_expanded.yaml',
        test_data_access=False,
        cohort_limitation='Previously selected suitable58 cohort; heldout only from this skill optimization')
    initial = getattr(args, 'initial_skills', None)
    evidence = getattr(args, 'initial_skill_evidence', None)
    if bool(initial) != bool(evidence):
        raise ValueError('Initial skills and development-only evidence must be supplied together')
    if initial:
        from data_agent.region_bandit.procedural_skills import validate_bundle, validate_decision_rule
        bundle, provenance = read_json(initial), read_json(evidence)
        validate_bundle(bundle)
        for skill in bundle['skills']:
            for rule in skill['steps']+skill['exceptions']:
                validate_decision_rule(rule)
        if (provenance.get('evidence_role') != 'development' or not provenance.get('facts')
                or not set(provenance.get('sources', [])) <= set(roles['development'])
                or any(f.get('role') != 'development' or f.get('source') not in provenance['sources']
                       for f in provenance['facts'])):
            raise ValueError('Initial skill evidence must come only from current development sources')
        evidence_ids = {f['fact_id'] for f in provenance['facts']}
        if any(not set(s['evidence_ids']) <= evidence_ids for s in bundle['skills']):
            raise ValueError('Initial skill citations missing from provenance')
        freeze_json(args.output/'configs/initial_skills.json', bundle)
        freeze_json(args.output/'configs/initial_skill_evidence.json', provenance)
        manifest['initial_guidance'] = dict(bundle_hash=stable_hash(bundle), evidence_hash=stable_hash(provenance),
            interpretation='Development-derived warm start, not validated as a new bundle; repeated cohort evaluation')
    reference_run = getattr(args, 'reference_run', None)
    if reference_run:
        reference_run = Path(reference_run).resolve()
        parent = read_json(reference_run/'campaign.json')
        if parent['roles'] != roles or parent['seeds'] != args.seeds or parent['rows'] != rows:
            raise ValueError('Reference campaign cohort, source roles or seeds differ')
        reused = []
        for task in manifest['rebuild_tasks']:
            key = stable_hash(task)[:24]
            source = reference_run/'rebuilt_banks'/key/'resolved_row.json'
            resolved = read_json(source)
            if resolved['task'] != task:
                raise ValueError('Reused reference task mismatch')
            freeze_json(args.output/'rebuilt_banks'/key/'resolved_row.json', resolved)
            reused.append(task)
        manifest['reference_reuse'] = dict(parent=str(reference_run), tasks=reused,
                                          note='Each bank is reverified before use; outcomes never select an action')
    freeze_json(args.output/'campaign.json', manifest)
    with SkillStore(args.output/'library.sqlite3') as store:
        if initial:
            store.initialize_guidance(bundle, provenance)
        for role, sources in roles.items():
            store.register_sources(sources, role)
    return manifest


async def rebuild(row, args):
    import yaml
    from data_agent.multi_proxy import MultiProxyConfig
    from run_mcts_data_agent import bootstrap_with_repairs
    from run_region_bandit import _public_model
    config = yaml.safe_load((args.output/'configs/coverage_expanded.yaml').read_text())
    values = dict(config['workflow'])
    values['similarity_lambda'] = values.pop('lambda', 1.)
    cfg = MultiProxyConfig(**values)
    cfg.validate()
    settings = config['model']
    model = await create_model(settings, args.env_file)
    key = stable_hash(row['task'])[:24]
    folder = args.output/'rebuilt_banks'/key
    try:
        source = await bootstrap_with_repairs(Path(row['task_path']), folder/'source', cfg,
            model, _public_model(settings), config.get('pricing'), resume=True)
        result = export_rebuilt_references(row, source, folder/'coverage_references', args.seeds)
        frozen_bank(result, args.seeds)
        freeze_json(folder/'resolved_row.json', result)
        return result
    finally:
        await model.client.close()


def resolved_rows(rows, manifest, args):
    result = []
    for row in rows:
        if row['task'] in manifest['rebuild_tasks']:
            path = args.output/'rebuilt_banks'/stable_hash(row['task'])[:24]/'resolved_row.json'
            if path.exists():
                row = read_json(path)
            else:
                event(args.output, 'rebuilding_reference', task=row['task'])
                row = asyncio.run(rebuild(row, args))
        result.append(row)
    return result


def invoke(args, name, command, *, rows=None, role='development', edit=None):
    folder = args.output/'stages'/name
    marker = folder/'stage_completed.json'
    if marker.exists():
        return read_json(marker)['result']
    folder.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT/'scripts/run_region_skill_memory.py'), command,
        '--library', str(args.output/'library.sqlite3'), '--output', str(folder),
        '--config', str(args.output/'configs/region_skill_memory.yaml'),
        '--env-file', str(args.env_file), '--require-valid-advice',
        '--seeds', *map(str, args.seeds)]
    for flag, name_ in [('reward-config', 'reward'), ('gate-config', 'gate'),
                       ('sampling-policy-config', 'sampling'), ('budget-config', 'budget')]:
        cmd += ['--'+flag, str(args.output/f'configs/region_skill_{name_}.json')]
    if rows is not None:
        freeze_json(folder/'tasks.json', rows)
        cmd += ['--tasks', str(folder/'tasks.json'), '--role', role]
    if edit:
        cmd += ['--edit', edit]
    freeze_json(folder/'command.json', cmd)
    event(args.output, 'running', stage=name, command=command)
    with (folder/'run.log').open('a') as log:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    path = folder/('proposal.json' if command == 'propose' else 'result.json')
    result = read_json(path)
    freeze_json(marker, dict(result=result))
    return result


def write_report(root, manifest, store):
    """Use final stage fact IDs, not all repeated gates, as the reporting matrix."""
    from collections import defaultdict
    from statistics import mean
    report = {}
    rebuilds = set(manifest['rebuild_tasks'])
    for role in ('development', 'validation', 'test'):
        pattern = {'development': 'development_*', 'validation': 'validation_final', 'test': 'heldout'}[role]
        facts = []
        for stage in sorted((root/'stages').glob(pattern)):
            experiment = read_json(stage/'experiment.json')
            for row in experiment['rows']:
                result = read_json(stage/'tasks'/row['directory_key']/experiment['bundles'][0]/'skill_result.json')
                facts.extend(store.fact(item['fact_id']) for item in result['facts'])
        sources = defaultdict(list)
        for fact in facts:
            sources[fact['source']].append(fact)
        def source_mean(field):
            values = []
            for records in sources.values():
                # Average seeds within tasks, then targets within the source.
                tasks = defaultdict(list)
                for fact in records:
                    value = field(fact)
                    if value is not None:
                        tasks[fact['task']].append(float(value))
                if tasks:
                    values.append(mean(mean(v) for v in tasks.values()))
            return mean(values) if values else None
        report[role] = dict(sources=len(sources), tasks=len({f['task'] for f in facts}),
            task_seed_observations=len(facts), informative_observations=sum(f['reward']['update_eligible'] for f in facts),
            source_mean_reward=source_mean(lambda f: f['reward']['reward']),
            source_mean_baseline_reward=source_mean(lambda f: f['paired_baseline']['before']),
            source_mean_paired_delta=source_mean(lambda f: f['paired_baseline']['delta']),
            source_mean_champion_miss=source_mean(lambda f: f['reward']['champion_miss']),
            source_mean_dual_rate=source_mean(lambda f: f['branch_count'] == 2),
            rebuilt_tasks=sorted({f['task'] for f in facts if f['task'] in rebuilds}),
            fact_ids=[f['fact_id'] for f in facts])
    write_json(root/'metrics.json', report)
    lines = ['# 区域采样技能实验结果', '',
        '58 个来源、59 个任务；种子为 20261026、20261027、20261028，仅改变采样。',
        '按来源等权汇总：先平均任务内种子，再平均同源目标。开发阶段使用随实验更新的技能；验证和留出使用最终技能。',
        '8 个任务使用重建参考，其余使用原冻结参考；本队列此前经过筛选，留出仅针对本轮技能优化。', '',
        '| 分组 | 来源 / 任务 | 平均奖励 | 相对覆盖基线差值 | 冠军漏选率 | 双分支比例 |',
        '| --- | --- | --- | --- | --- | --- |']
    def fmt(value):
        return 'NA' if value is None else f'{value:.4f}'
    for role, row in report.items():
        values = [fmt(row[k]) for k in ('source_mean_reward', 'source_mean_paired_delta',
                                      'source_mean_champion_miss', 'source_mean_dual_rate')]
        lines.append(f"| {role} | {row['sources']} / {row['tasks']} | "+' | '.join(values)+' |')
    (root/'summary.md').write_text('\n'.join(lines)+'\n')


def run(args, manifest):
    roles, rows = manifest['roles'], manifest['rows']
    def select(sources):
        return [r for source in sources for r in rows if r['collection_dataset'] == source]
    for offset in range(0, len(roles['development']), 5):
        round_id = offset//5+1
        name = f'development_{round_id:02d}'
        dev = resolved_rows(select(roles['development'][offset:offset+5]), manifest, args)
        invoke(args, name, 'run', rows=dev)
        plan_path = args.output/'stages'/name/'optimization_plan.json'
        if plan_path.exists():
            plan = read_json(plan_path)
        else:
            with SkillStore(args.output/'library.sqlite3') as store:
                unresolved = store.db.execute('SELECT id FROM edits WHERE status IN ("candidate","pending")').fetchone()
                if unresolved:
                    raise RuntimeError('Unresolved candidate '+unresolved[0]+
                        '; resume its proposal/validation stage before continuing development')
                else:
                    try:
                        store.optimization_context(5)
                        plan = dict(propose=True)
                    except InsufficientDevelopmentEvidence as exc:
                        plan = dict(propose=False, reason=str(exc))
            freeze_json(plan_path, plan)
        if plan['propose']:
            proposal = invoke(args, f'proposal_{round_id:02d}', 'propose')
            if proposal.get('status') == 'rejected_invalid':
                event(args.output, 'proposal_rejected', stage=f'proposal_{round_id:02d}',
                      reason=proposal['reason'])
                continue
            val = resolved_rows(select(roles['validation']), manifest, args)
            stage = f'validation_{round_id:02d}'
            gate = invoke(args, stage, 'validate', rows=val,
                          role='validation', edit=proposal['edit_id'])
            if gate['status'] == 'pending':
                with SkillStore(args.output/'library.sqlite3') as store:
                    resolution = store.close_inconclusive(proposal['edit_id'], gate,
                        expected_cases=[(r['collection_dataset'], r['task'], seed)
                                        for r in val for seed in args.seeds])
                resolution_path = args.output/'stages'/stage/'pending_resolution.json'
                already_logged = resolution_path.exists()
                freeze_json(resolution_path, resolution)
                if not already_logged:
                    event(args.output, 'candidate_archived', stage=stage, resolution=resolution)
    # Always cover validation sources even if no edit was proposed or accepted.
    invoke(args, 'validation_final', 'run',
           rows=resolved_rows(select(roles['validation']), manifest, args), role='validation')
    invoke(args, 'heldout', 'run', rows=resolved_rows(select(roles['test']), manifest, args), role='test')
    with SkillStore(args.output/'library.sqlite3') as store:
        write_report(args.output, manifest, store)
        summary = dict(status='completed', tasks=len(rows), source_datasets=sum(map(len, roles.values())),
            sampling_seeds=args.seeds, active_bundle=store.active,
            edits=[dict(edit_id=i, status=s) for i, s in store.db.execute('SELECT id,status FROM edits')],
            facts_by_role=dict(store.db.execute('SELECT role,count(*) FROM facts GROUP BY role')),
            new_reference_banks=len(manifest['rebuild_tasks'])-len(manifest.get('reference_reuse', {}).get('tasks', [])),
            test_data_access=False)
    write_json(args.output/'campaign_result.json', summary)
    event(args.output, 'completed', result=summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[20261026, 20261027, 20261028])
    p.add_argument('--split-seed', type=int, default=20261116)
    p.add_argument('--env-file', type=Path, default=Path('/home/cmcui/InternAgent/.env'))
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--config-dir', type=Path, help='Independent campaign configuration directory')
    p.add_argument('--initial-skills', type=Path, help='Explicit initial three-skill bundle for a fresh library')
    p.add_argument('--initial-skill-evidence', type=Path, help='Development-only provenance for the initial bundle')
    p.add_argument('--reference-run', type=Path, help='Reuse resolved reference banks from an identical prior campaign')
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            manifest = initialize(args)
            if args.prepare_only:
                event(args.output, 'prepared', tasks=len(manifest['rows']), roles=manifest['roles'])
            else:
                run(args, manifest)
        except Exception as exc:
            event(args.output, 'failed', error=type(exc).__name__+': '+str(exc))
            raise


if __name__ == '__main__':
    main()
