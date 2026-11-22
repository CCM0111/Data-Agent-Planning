"""Verified sharing and a full-training barrier for paired coreset experiments."""
import json
import shutil
import time
from pathlib import Path
import numpy as np
from .coreset import file_hash, write_json
from .multi_proxy_io import read_typed
from .stage1_selection import compare_full_stage
from .multi_proxy_trace import save_trace


def read(path):
    return json.loads(Path(path).read_text())


def validate_shared_source(source, destination, current):
    previous = read(source/'manifest.json')
    for key in ('method_set_id', 'internal_split_id', 'input_sha256'):
        if not current.get(key) or current[key] != previous.get(key):
            raise ValueError('Shared fit identity mismatch: '+key)
    a, b = dict(previous['configuration']), dict(current['configuration'])
    for config in (a,b):
        for key in ('proxy_selection_scheme','defer_full_train','method_preflight','stage1_selection'):
            config.pop(key,None)
        # Manifests predating the opt-in feedback workflow have no such field.
        # Only an explicit false is equivalent; enabled feedback changes fits.
        if config.get('parameter_feedback', False) is False:
            config.pop('parameter_feedback', None)
            config.pop('parameter_feedback_mode', None)
    if a != b:
        raise ValueError('Shared fit training configuration mismatch')
    for filename in ('P_indices.json','Q_indices.json','V_core_indices.json','csv_schema.json',
                     'task_context.json','core_validation.csv','Q_features.csv'):
        if file_hash(source/filename)!=file_hash(destination/filename):
            raise ValueError('Shared fit input mismatch: '+filename)
    for method in current['methods']:
        i=method['method_id'] if 'method_id' in method else current['methods'].index(method)
        if file_hash(source/f'methods/method_{i}/method.py')!=method['code_sha256']:
            raise ValueError('Shared source method changed')


def copy_directory(source, destination):
    # The data volume does not support all chmod/utime metadata operations.
    destination.mkdir(parents=True,exist_ok=False)
    for item in source.rglob('*'):
        target=destination/item.relative_to(source)
        if item.is_dir():
            target.mkdir(parents=True,exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(item,target)


def copy_fit(source, destination):
    """Copy immutable outputs; preserve source context and hashes for the audit."""
    destination.mkdir(parents=True,exist_ok=False)
    files=('fit_report.json','predictions.npz','unfitted.joblib')
    hashes={}
    for name in files:
        if not (source/name).is_file():
            raise ValueError('Missing reusable fit artifact: '+str(source/name))
        shutil.copyfile(source/name,destination/name)
        hashes[name]=file_hash(source/name)
    if (source/'context.json').exists():
        shutil.copyfile(source/'context.json',destination/'source_context.json')
    write_json(destination/'reuse.json',{'source':str(source.resolve()),'sha256':hashes,
                                        'new_fit':False})
    return read(destination/'fit_report.json')


def finish_shared_full(first, second):
    """Run full models only after BOTH independent screening/MCS outputs exist."""
    from .multi_proxy import (run_fit, evaluation_metrics, rank_methods, metric_direction)
    first,second=Path(first).resolve(),Path(second).resolve()
    manifests=[read(p/'manifest.json') for p in (first,second)]
    a,b=manifests
    validate_shared_source(first,second,b)
    for path,manifest in zip((first,second),manifests):
        if manifest['status']!='success':
            raise ValueError('Both screening runs must finish before full training')
        if not (path/'ranking.json').is_file() or not (path/'stage1_selection/summary.json').is_file():
            raise ValueError('Both cross evaluations and MCS decisions must precede full training')
    if not a.get('full_train_pending') and not b.get('full_train_pending'):
        return
    started=time.perf_counter()
    # Hash the decisions before full scores become available; verify they stay unchanged.
    frozen={str(p/f):file_hash(p/f) for p in (first,second)
            for f in ('ranking.json','stage1_selection/summary.json','stage1_selection/elimination_trace.json')}
    cfg=a['configuration'];methods=a['methods'];n=len(methods)
    task=a['task'];context=read(first/'task_context.json');classes=context['classes']
    external=read_typed(a['full_train_evaluation_set']['path'],context['csv_schema'])
    if file_hash(a['full_train_evaluation_set']['path'])!=a['input_sha256']['validation']:
        raise ValueError('External validation changed')
    taskpath=Path(a['task_path'])
    train=(taskpath.parent/task['files']['train']).resolve()
    if file_hash(train)!=a['input_sha256']['train']:
        raise ValueError('Training file changed')
    a['status']='running';a['paired_stage']='shared_full_training'
    write_json(first/'manifest.json',a)
    reports=[]
    try:
        for i,method in enumerate(methods):
            folder=first/'full_train'/f'method_{i}'
            a['active_execution']=str(folder.relative_to(first))
            write_json(first/'manifest.json',a)
            if file_hash(method['code_path'])!=method['code_sha256']:
                raise ValueError('Full training method changed')
            if (folder/'evaluation.json').exists():
                report=read(folder/'evaluation.json')
            else:
                if folder.exists():
                    raise ValueError('Interrupted full fit requires explicit reconciliation: '+str(folder))
                proxy=read(first/'evaluations'/f'method_{i}_coreset_0/fit_report.json')
                report=run_fit(method['code_path'],train,first/'external_validation_features.csv',
                    {**context,'expected_family':method['family'],
                     'expected_model_class':proxy['actual_budget']['model_class'],
                     'reference_pipeline_path':str(first/'evaluations'/f'method_{i}_coreset_0/unfitted.joblib'),
                     'fit_stage':'full_train','stage_parameters':method['idea']['stage_parameters']['full_train']},
                    classes,cfg['full_train_budget'],cfg['model_seed'],folder,cfg['batch_size'])
                with np.load(folder/'predictions.npz',allow_pickle=False) as saved:
                    predictions=saved['predictions']; probabilities=saved['probabilities']
                if task['problem_type']=='classification':
                    mapping={str(c):c for c in classes}
                    predictions=np.asarray([mapping[v] for v in predictions])
                if len(predictions)!=len(external):
                    raise ValueError('Full prediction count mismatch')
                metrics,reason=evaluation_metrics(task,external[task['target']],predictions,probabilities,classes)
                report.update(method_id=i,method_name=method['name'],metric=task['metric'],metrics=metrics,
                    primary_metric_status='undefined' if reason else 'defined',primary_metric_reason=reason,
                    evaluation_set='external_validation',training_indices='full_train_indices.json')
                write_json(folder/'evaluation.json',report)
                a['fits_completed']+=1;a['full_train_fits_completed']+=1
            reports.append(report)
            write_json(first/'full_train_evaluations.json',reports)
            a.pop('active_execution',None);write_json(first/'manifest.json',a)
        ranking=[{'method_id':r['method_id'],'method_name':r['method_name'],
                  'metric':task['metric'],'score':r['metrics'][task['metric']]} for r in reports]
        maximize=metric_direction(task)=='maximize'
        rank_methods(ranking,'score',maximize)
        write_json(first/'full_train_ranking.json',{'evaluation_set':a['full_train_evaluation_set'],
            'direction':'maximize' if maximize else 'minimize','methods':ranking,
            'scope':'One shared full-training reference, run after both screening decisions'})
        for filename in ('full_train_ranking.json','full_train_evaluations.json'):
            shutil.copyfile(first/filename,second/filename)
        for i, report in enumerate(reports):
            target=second/'full_train'/f'method_{i}'
            if not target.exists():
                copy_fit(first/'full_train'/f'method_{i}',target)
                write_json(target/'evaluation.json',report)
        b['shared_full_train_source']=str(first)
        b['full_train_fits_reused']=n
        b['reused_predictive_fits']=2*n+n
        a['timings_seconds']['shared_full_training']=time.perf_counter()-started
        a['timings_seconds']['total_wall']+=a['timings_seconds']['shared_full_training']
        expected=(n*(2+cfg['n_coresets']), n*(cfg['n_coresets']-1))
        if (a['fits_completed'],b['fits_completed'])!=expected:
            raise ValueError('Shared predictive fit count mismatch')
        for path,manifest in zip((first,second),(a,b)):
            compare_full_stage(path)
            manifest.update(status='success',full_train_pending=False,paired_stage='complete')
            manifest['expected_predictive_fits']=manifest['fits_completed']
            manifest['logical_predictive_fits']=n*(2+cfg['n_coresets'])
            manifest['stage1_selection']['full_train_comparison_path']='stage1_selection/full_train_comparison.json'
            write_json(path/'manifest.json',manifest)
            save_trace(path,manifest)
        if any(file_hash(Path(p))!=value for p,value in frozen.items()):
            raise ValueError('Screening decisions changed after full training')
        write_json(first.parent.parent/('shared_full_audit_'+first.name+'.json'),
                   {'first':str(first),'second':str(second),'screening_sha256':frozen,
                    'new_predictive_fits':a['fits_completed']+b['fits_completed'],
                    'planned_predictive_fits':n*(1+2*cfg['n_coresets']),
                    'order':['scheme1_screening','scheme2_screening','shared_full_training']})
    except BaseException as error:
        a.update(status='failed',failure_stage='shared_full_training',error=f'{type(error).__name__}: {error}')
        write_json(first/'manifest.json',a)
        raise


def attach_completed_full(source, destination):
    """Attach an unchanged reference only after the new screening is committed."""
    source,destination=Path(source).resolve(),Path(destination).resolve()
    previous=read(source/'manifest.json');current=read(destination/'manifest.json')
    validate_shared_source(source,destination,current)
    if previous['status']!='success' or previous.get('full_train_pending'):
        raise ValueError('Full reference is not complete')
    if current['status']!='success' or not (destination/'stage1_selection/summary.json').exists():
        raise ValueError('New screening must be complete before attaching full reference')
    if not current.get('full_train_pending'):
        if current.get('shared_full_train_source')!=str(source):
            raise ValueError('Existing full reference differs')
        return
    n=len(current['methods'])
    if current['fits_completed']!=n*(current['configuration']['n_coresets']-1):
        raise ValueError('Unexpected new cross-fit count')
    frozen={f:file_hash(destination/f) for f in ('ranking.json','stage1_selection/summary.json','stage1_selection/elimination_trace.json')}
    for i in range(n):
        # Pipeline configuration must match, not just estimator names.
        a=read(source/f'evaluations/method_{i}_coreset_0/fit_report.json')
        b=read(destination/f'evaluations/method_{i}_coreset_0/fit_report.json')
        if a['pipeline_spec']!=b['pipeline_spec']:
            raise ValueError('Full reference pipeline differs')
        folder=destination/'full_train'/f'method_{i}'
        copy_fit(source/'full_train'/f'method_{i}',folder)
        shutil.copyfile(source/'full_train'/f'method_{i}/evaluation.json',folder/'evaluation.json')
    for filename in ('full_train_ranking.json','full_train_evaluations.json'):
        shutil.copyfile(source/filename,destination/filename)
    compare_full_stage(destination)
    if any(file_hash(destination/f)!=h for f,h in frozen.items()):
        raise ValueError('Reference attachment changed screening decisions')
    current.update(full_train_pending=False,shared_full_train_source=str(source),
                   full_train_fits_reused=n,reused_predictive_fits=3*n,
                   logical_predictive_fits=n*(2+current['configuration']['n_coresets']),
                   paired_stage='complete')
    current['stage1_selection']['full_train_comparison_path']='stage1_selection/full_train_comparison.json'
    write_json(destination/'full_reference_audit.json',{'source':str(source),'screening_sha256':frozen,
                'full_ranking_sha256':file_hash(source/'full_train_ranking.json'),'new_full_fits':0})
    write_json(destination/'manifest.json',current)
    save_trace(destination,current)
