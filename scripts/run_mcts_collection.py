#!/usr/bin/env python3
"""Persistent optional-workflow batch with post-search test diagnostics.

New batches use the unified workflow; existing legacy MCTS batches keep their
frozen entry/configuration. No existing task definition is modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from data_agent.multi_proxy_io import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def search_directory(folder, engine):
    return folder/'search' if engine == 'unified' else folder


def source_group(row):
    """Use the collection's source identity, shared by multiple target tasks."""
    from data_agent.region_bandit.schemas import dataset_group
    value = row.get('collection_dataset')
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Each collection row requires a nonempty collection_dataset source ID')
    task = read(row['task_path'])
    group = dataset_group(task, value)
    declared = task.get('dataset_group_id') or task.get('metadata', {}).get('dataset_group_id')
    if declared is not None and dataset_group(task) != group:
        raise ValueError(f'Collection and task source IDs differ: {row["task"]}')
    return group


def resolve_engine(output, requested='auto', config=None):
    frozen = [name for name, filename in (('unified', 'workflow.yaml'), ('legacy', 'mcts.yaml'))
              if (output/filename).exists()]
    if len(frozen) > 1:
        raise ValueError('Output contains both workflow and legacy batch configurations')
    supplied = None
    if config is not None:
        values = yaml.safe_load(Path(config).read_text())
        if not isinstance(values, dict):
            raise ValueError('Batch configuration must be a YAML object')
        if set(values) & {'modules', 'region_config', 'search_config', 'region_bandit', 'multi_proxy'}:
            supplied = 'unified'
        elif 'initialization_config' in values:
            supplied = 'legacy'
        # search/memory-only overrides are valid in either format; let the
        # frozen or explicit engine decide rather than silently assuming legacy.
    engine = frozen[0] if frozen else (supplied or ('unified' if requested == 'auto' else requested))
    if (requested != 'auto' and requested != engine) or (supplied and supplied != engine):
        raise ValueError('Engine/configuration differs from the frozen batch; use a new output directory')
    return engine


def freeze_config(output, source, engine):
    """Freeze the referenced initialization/region/search files, not just their paths."""
    destination = output/('workflow.yaml' if engine == 'unified' else 'mcts.yaml')
    if destination.exists():
        return destination
    settings = yaml.safe_load(Path(source).read_text())

    def dependency(value):
        path = Path(value)
        return path if path.is_absolute() else ROOT/path

    def freeze_initial(values, filename):
        initial = dependency(values.get('initialization_config', 'config/coverage_regression30.yaml'))
        target = output/filename
        target.write_text(initial.read_text())
        return {**values, 'initialization_config': str(target)}

    if engine == 'legacy':
        settings = freeze_initial(settings, 'initialization.yaml')
    else:
        for key, default, prefix in (('region_config', 'config/region_bandit.yaml', 'region'),
                                     ('search_config', 'config/mcts.yaml', 'search')):
            values = yaml.safe_load(dependency(settings.get(key, default)).read_text())
            values = freeze_initial(values, prefix+'_initialization.yaml')
            target = output/(prefix+'.yaml')
            target.write_text(yaml.safe_dump(values, sort_keys=False))
            settings[key] = str(target)
    temporary = destination.with_suffix('.yaml.tmp')
    temporary.write_text(yaml.safe_dump(settings, sort_keys=False))
    temporary.replace(destination)
    return destination


def should_run(status, retry_failed=False):
    return status not in {'completed', 'completed_with_node_test_failures'} and (
        status != 'failed' or retry_failed)


def has_resume_state(folder, engine):
    names = ('workflow_manifest.json', 'sampling/state.json', 'bootstrap/attempt_state.json',
             'bootstrap/manifest.json', 'search/search_state.json') if engine == 'unified' else (
             'search_state.json', 'bootstrap/manifest.json', 'bootstrap/attempt_state.json')
    return any((folder/name).exists() for name in names)


def search_command(row, folder, config, engine, memory_path=None, extend_wall_seconds=0):
    executable = 'run_data_agent_workflow.py' if engine == 'unified' else 'run_mcts_data_agent.py'
    command = [str(ROOT/'.venv/bin/python'), str(ROOT/executable), '--task', row['task_path'],
               '--output', str(folder), '--config', str(config)]
    resume = has_resume_state(folder, engine)
    if resume:
        command.append('--resume')
    if engine == 'unified':
        command += ['--dataset-group-id', source_group(row)]
        if memory_path is not None:
            command += ['--memory', str(memory_path)]
        # Only sampling runs with saved state have a wall budget to extend.
        if extend_wall_seconds and (folder/'sampling/state.json').exists():
            command += ['--extend-wall-seconds', str(extend_wall_seconds)]
    return command


def validate_batch_task(row, folder, engine='legacy'):
    """Check current inputs even when a completed task would otherwise be skipped."""
    from data_agent.multi_proxy import load_fixed
    from data_agent.coreset import file_hash
    from data_agent.prepared_protocol import _protocol_values
    task_path = Path(row['task_path'])
    task, split, _, _, train = load_fixed(task_path)
    _protocol_values(task, task_path, split, len(train))
    for key in ('validation', 'test_features', 'test_labels'):
        path = (task_path.parent / task['files'][key]).resolve()
        if not path.is_file() or file_hash(path) != split['sha256'].get(path.name):
            raise ValueError(f'Changed or missing frozen {key}: {row["task"]}')
    if engine == 'unified':
        source_group(row)
    candidates = [folder/'bootstrap/manifest.json', folder/'sampling/bootstrap/manifest.json',
                  folder/'selected_source/manifest.json']
    # Interrupted/repaired bootstrap roots may only contain an attempt pointer.
    for base in (folder/'bootstrap', folder/'sampling/bootstrap'):
        if (base/'attempt_state.json').exists():
            effective = read(base/'attempt_state.json').get('effective_run')
            if effective:
                candidates.append(Path(effective)/'manifest.json')
    found = [path for path in candidates if path.exists()]
    for bootstrap in found:
        previous = read(bootstrap)
        if previous.get('split_id') != split['split_id']:
            raise ValueError(f'Frozen task/protocol changed for {row["task"]}; '
                             'rerun in a new output directory instead of reusing previous results')
    search = search_directory(folder, engine)
    if not found and ((search/'search_state.json').exists() or (search/'final_test/selection_manifest.json').exists()):
        raise ValueError(f'Cannot verify previous results without bootstrap metadata: {row["task"]}')


def worker(row, output, config, engine='legacy', memory_path=None, extend_wall_seconds=0):
    name = row['task']
    folder = output/'runs'/name
    folder.mkdir(parents=True, exist_ok=True)
    status_path = folder/'batch_task.json'
    search = search_directory(folder, engine)
    status = {**row, 'engine': engine, 'search_run': str(search),
              'status':'running', 'started_at':stamp(), 'pid':None}
    atomic_json(status_path,status)
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    try:
        frozen = (search/'final_test/selection_manifest.json').exists()
        if not frozen:
            command = search_command(row, folder, config, engine, memory_path, extend_wall_seconds)
            with (folder/'search.log').open('a') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
                status.update(pid=process.pid,phase='search');atomic_json(status_path,status)
                code=process.wait()
            if code:
                raise RuntimeError(f'Search exited {code}; see search.log and saved stage state')
        summary=read(search/'search_summary.json')
        if not summary['initialization_complete']:
            raise RuntimeError('Initial full training incomplete')
        command=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/evaluate_mcts_final.py'),
                 '--search-run',str(search),'--all-nodes']
        with (folder/'posthoc.log').open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
            status.update(pid=process.pid,phase='posthoc_test');atomic_json(status_path,status)
            code=process.wait()
        if code:
            raise RuntimeError(f'Post-search testing exited {code}; see posthoc.log')
        tested=read(search/'final_test/all_nodes_results.json')
        winner=next(r for r in tested['nodes'] if r['id']==summary['best_id'])
        if winner['status']!='success':
            raise RuntimeError('Frozen winner test failed: '+winner.get('error','unknown'))
        status.update(status='completed' if tested['status']=='success' else 'completed_with_node_test_failures',
                      metric=summary['metric'],direction=summary['direction'],winner_id=summary['best_id'],
                      validation_score=summary['best_full_score'],test_score=winner['test_score'],
                      completed_expansions=summary['completed_expansions'],full_fit_attempts=summary['full_fit_attempts'],
                      full_fit_successes=summary['full_fit_successes'],search_api_calls=summary['api_calls'],
                      rollout_budget_basis=summary.get('rollout_budget_basis','attempts_and_successes'),
                      tested_nodes=sum(r['status']=='success' for r in tested['nodes']),
                      search_stop_reason=summary.get('stop_reason'))
        # Provider response IDs and token usage are evidence of real API calls;
        # never include credentials or prompts in the batch summary.
        evidence=[]
        for path in folder.rglob('*.usage.json'):
            usage=read(path)
            if usage.get('response_id') and usage.get('status')=='success':
                evidence.append({k:usage.get(k) for k in ('response_id','response_model','usage')})
        atomic_json(folder/'provider_evidence.json',{'successful_responses':len(evidence),'calls':evidence})
        status['verified_provider_responses']=len(evidence)
        if not evidence:
            raise RuntimeError('No successful provider response evidence found')
    except Exception as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}')
    status.update(finished_at=stamp(),pid=None)
    atomic_json(status_path,status)
    return status


def aggregate(output, rows, engine='legacy'):
    records=[]
    for row in rows:
        path=output/'runs'/row['task']/'batch_task.json'
        item=read(path) if path.exists() else {**row,'status':'pending'}
        summary_path=search_directory(path.parent, engine)/'search_summary.json'
        if item['status']=='running' and summary_path.exists():
            summary=read(summary_path)
            item.update({key:summary.get(key) for key in ('completed_expansions','full_fit_attempts','full_fit_successes','api_calls','best_full_score','rollout_budget_basis')})
        records.append(item)
    counts={key:sum(r['status']==key for r in records) for key in sorted({r['status'] for r in records})}
    summary={'updated_at':stamp(),'engine':engine,'tasks':len(rows),'datasets':len({r['collection_dataset'] for r in rows}),
             'counts':counts,'random_baseline':False,'selection_basis':'external_validation',
             'parameters':{'a':.3,'beta':.6,'lambda':2.,'sampling_seed':20261026},'results':records}
    atomic_json(output/'batch_summary.json',summary)
    columns=['task','collection_dataset','problem_type','status','metric','direction','winner_id','validation_score','test_score',
             'completed_expansions','full_fit_attempts','full_fit_successes','rollout_budget_basis','tested_nodes','search_api_calls','verified_provider_responses','error']
    temporary=output/'results.csv.tmp'
    with temporary.open('w') as file:
        writer=csv.DictWriter(file,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    temporary.replace(output/'results.csv')
    lines=['# 适合方法的数据集：采样与搜索实验','',f'更新时间：{summary["updated_at"]}',
           f'范围：{summary["datasets"]} 个数据集，{len(rows)} 个任务；状态：{counts}。','',
           f'执行入口：{engine}；模块和预算以本次冻结配置为准。','',
           '参数 (0.3, 0.6, 2)，采样种子 20261026；真实 DeepSeek；无随机基线。',
           '最佳方法按外部验证集选择；下表测试成绩来自冻结后的最佳方法。其他节点的测试结果仅作事后分析。',
           '这是先前根据结果整理的适用数据集集合，结论限于该集合；不同任务的原始指标不直接平均。','',
           '成功 rollout 指全量训练完成且得到有效评分；尝试次数另列，包含失败和进行中的尝试。','',
           '| 任务 | 状态 | 指标 | 最佳验证成绩 | 对应测试成绩 | 成功 rollout | 全量尝试次数 |',
           '|---|---|---|---:|---:|---:|---:|']
    for row in records:
        def number(key):
            value=row.get(key);return f'{value:.6g}' if isinstance(value,(float,int)) else '—'
        lines.append(f'| {row["task"]} | {row["status"]} | {row.get("metric","—")} | {number("validation_score")} | {number("test_score")} | {number("full_fit_successes")} | {number("full_fit_attempts")} |')
    temporary=output/'report.md.tmp';temporary.write_text('\n'.join(lines)+'\n');temporary.replace(output/'report.md')
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks',type=Path,default=ROOT/'datasets/method_suitable58_20261104/tasks.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,help='Default: workflow.yaml for new runs; frozen config on resume')
    parser.add_argument('--engine',choices=('auto','unified','legacy'),default='auto',
                        help='auto keeps existing legacy batches and uses the unified workflow for new batches')
    parser.add_argument('--memory',type=Path,
                        help='Shared bandit memory; defaults to the saved batch path or results/region_bandit_policy/memory.sqlite3')
    parser.add_argument('--retry-failed',action='store_true',help='Resume failed tasks as well as unfinished tasks')
    parser.add_argument('--extend-wall-seconds',type=int,default=0,
                        help='Extend existing unified sampling budgets when resuming tasks; does not change frozen configuration')
    parser.add_argument('--workers',type=int,default=4)
    arguments=parser.parse_args(argv)
    if not 1<=arguments.workers<=16:raise ValueError('workers must be between 1 and 16')
    if arguments.extend_wall_seconds < 0:raise ValueError('extend-wall-seconds must be nonnegative')
    output=arguments.output.resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/'.batch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        engine=resolve_engine(output,arguments.engine,arguments.config)
        previous=read(output/'run_manifest.json') if (output/'run_manifest.json').exists() else {}
        saved_memory=previous.get('bandit_memory')
        memory_path=(arguments.memory or (Path(saved_memory) if saved_memory else
                     ROOT/'results/region_bandit_policy/memory.sqlite3')).resolve()
        if engine=='unified' and saved_memory and str(memory_path)!=saved_memory:
            raise ValueError('Bandit memory differs from the frozen batch; use the saved memory path')
        if arguments.extend_wall_seconds and engine!='unified':
            raise ValueError('Wall extensions require the unified regional sampling workflow')
        rows=read(arguments.tasks)
        names=[row['task'] for row in rows]
        if len(set(names))!=len(names) or any(Path(name).name!=name or name in {'.','..'} for name in names):
            raise ValueError('Task IDs must be unique safe directory names')
        for row in rows:
            validate_batch_task(row, output / 'runs' / row['task'], engine)
        manifest=output/'tasks.json'
        if manifest.exists() and read(manifest)!=rows:raise ValueError('Frozen collection differs')
        atomic_json(manifest,rows)
        source_config=arguments.config or ROOT/('config/workflow.yaml' if engine=='unified' else 'config/mcts.yaml')
        frozen_config=freeze_config(output,source_config,engine)
        if engine=='unified':
            from run_data_agent_workflow import settings_for
            effective=settings_for(frozen_config)
            if arguments.extend_wall_seconds and not effective[3]['region_selection']:
                raise ValueError('Wall extensions require enabled regional sampling')
        else:
            from run_mcts_data_agent import settings_for
            settings_for(frozen_config)
        atomic_json(output/'run_manifest.json',{'started_at':stamp(),'pid':os.getpid(),'workers':arguments.workers,
                    'engine':engine,'bandit_memory':str(memory_path) if engine=='unified' else None,
                    'tasks_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'real_api':True,'random_baseline':False,
                    'config_sha256':hashlib.sha256(frozen_config.read_bytes()).hexdigest()})
        todo=[]
        for row in rows:
            state=output/'runs'/row['task']/'batch_task.json'
            if state.exists() and not should_run(read(state)['status'],arguments.retry_failed):continue
            todo.append(row)
        aggregate(output,rows,engine)
        with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
            pending={pool.submit(worker,row,output,frozen_config,engine,memory_path,
                                 arguments.extend_wall_seconds):row['task'] for row in todo}
            while pending:
                completed,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                for future in completed:
                    name=pending.pop(future);result=future.result()
                    print(stamp(),name,result['status'],result.get('error',''),flush=True)
                aggregate(output,rows,engine)
        result=aggregate(output,rows,engine)
        atomic_json(output/'batch_complete.json',{'finished_at':stamp(),'counts':result['counts']})
        print(json.dumps(result['counts']),flush=True)


if __name__=='__main__':
    main()
