"""One conservative P/Q-only intervention before coreset selection.

Full parameters and preprocessing are frozen first. A weak P fit plus weak Q
performance and a specific capacity/optimization symptom can authorize ONE
parameter proposal and at most ONE additional fit. Never consumes V_core.
"""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil

import joblib
import numpy as np
from sklearn.base import clone

from .coreset import file_hash, digest
from .model_registry import MODEL_REGISTRY
from .multi_proxy_feedback import load_method_module, finite_json, stage_profile
from .multi_proxy_io import atomic_json
from .multi_proxy_runtime import prepare_pipeline
from .parameter_feedback import ALLOWED_PARAMETERS

# Heuristic screening thresholds, not calibrated estimates of adjustment benefit.
# Persist the policy in the resume identity so old decisions cannot be reused.
SCREENING_POLICY = {
    'version': 'guarded_once_v2',
    'severe_gain_max': .10,
    'moderate_gain_max': .70,
    'max_generalization_gap': .15,
    'moderate_min_rows': 200,
    'capacity_fraction': .0025,
}

INSTRUCTION = '''You may propose at most ONE mild parameter change for this method, using only supplied P/Q evidence.
The full-data pipeline and parameters are immutable. Preserve all features, model class, loss, architecture, seeds and other settings.
Choose either no change, or exactly one of the supplied allowed_changes, with its exact value. Do not invent parameters or values.
Address the observed capacity/optimization symptom, not merely a low Q score. Nonconvergence alone does not prove underfitting.
The screening thresholds only admit a hypothesis, not proof that a change will help. For moderate_fit candidates, examine P/Q gains and the capacity restriction together. Prefer no change when noisy targets, weak features or missing evidence are equally plausible. A large leaf/neighbor setting is a possible restriction, not proof it is binding.
There is one proposal opportunity and at most one refit; no repair/tuning loop. Return JSON only:
{"reason":"one observed P/Q fact supporting the update or abstention; omit generic caveats", "updates":{"model__parameter":value}}
Use an empty updates object if evidence does not justify changing the model. This changes only the P proxy and coreset stages, never full training.'''


def enabled(cfg):
    return cfg.parameter_feedback and cfg.parameter_feedback_mode == 'guarded_once'


def effective_config(cfg):
    if not enabled(cfg):
        return cfg
    # Full settings must not silently be truncated by legacy small-model caps.
    # Stage wall-clock limits remain explicit, unchanged execution constraints.
    caps = {k:cfg.full_train_budget[k] for k in ('max_trees', 'max_epochs')}
    return replace(cfg, scoring_budget={**cfg.scoring_budget, **caps},
                   evaluation_budget={**cfg.evaluation_budget, **caps})


def weak_fit_evidence(task, report, y_p, y_q, predictions, probabilities, classes):
    from .multi_proxy import evaluation_metrics
    from .stage1_metrics import metric_direction
    metric = task['metric']
    observed = report.get('training_feedback', {}).get('training_metrics', {}).get(metric)
    if observed is None:
        return {'eligible':False, 'reason':'missing_training_primary_metric'}
    if task['problem_type'] == 'classification':
        prior = np.asarray([np.mean(np.asarray(y_p)==c) for c in classes])
        label = classes[int(np.argmax(prior))]
        dummy_p = np.full(len(y_p), label);dummy_q = np.full(len(y_q), label)
        probs_p = np.tile(prior, (len(y_p),1));probs_q = np.tile(prior, (len(y_q),1))
    else:
        value = float(np.mean(y_p))
        dummy_p = np.full(len(y_p),value);dummy_q = np.full(len(y_q),value)
        probs_p = probs_q = None
    base_p, reason_p = evaluation_metrics(task,y_p,dummy_p,probs_p,classes)
    base_q, reason_q = evaluation_metrics(task,y_q,dummy_q,probs_q,classes)
    q, reason = evaluation_metrics(task,y_q,predictions,probabilities,classes)
    values = [observed, base_p.get(metric),base_q.get(metric),q.get(metric)]
    if reason_p or reason_q or reason or any(v is None or not math.isfinite(float(v)) for v in values):
        return {'eligible':False,'reason':'undefined_primary_metric'}
    maximize = metric_direction(task) == 'maximize'
    def gain(score, baseline):
        # Bounded higher-is-better metrics use remaining headroom (also valid for R2).
        return ((score-baseline)/max(1-baseline,1e-8) if maximize else
                (baseline-score)/max(abs(baseline),1e-8))
    p_gain,q_gain = gain(observed,base_p[metric]),gain(q[metric],base_q[metric])
    gap = p_gain - q_gain
    severe = max(p_gain, q_gain) <= SCREENING_POLICY['severe_gain_max']
    moderate = (min(len(y_p), len(y_q)) >= SCREENING_POLICY['moderate_min_rows']
                and max(p_gain, q_gain) <= SCREENING_POLICY['moderate_gain_max'])
    gap_ok = gap <= SCREENING_POLICY['max_generalization_gap']
    eligible = (severe or moderate) and gap_ok
    return {'eligible': eligible,
            'reason': ('large_generalization_gap' if not gap_ok else
                       'severe_fit' if severe else 'moderate_fit' if moderate else
                       'strong_fit_or_insufficient_rows'),
            'screening_policy': dict(SCREENING_POLICY),
            'generalization_gap': gap,
            'metric':metric,'p_score':observed,'q_score':q[metric],
            'p_baseline':base_p[metric],'q_baseline':base_q[metric],
            'p_gain':p_gain,'q_gain':q_gain,
            'scope':'P/Q only; not a proof of underfitting'}


def allowed_changes(parameters, model_name, report, rows, budget):
    choices = {}
    convergence = report.get('training_feedback',{}).get('convergence',{})
    curve = convergence.get('loss_curve') or {}
    tail = curve.get('tail') or []
    improving = (len(tail)>=3 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in tail)
                 and tail[-1] < tail[0] - max(abs(tail[0])*.01,1e-10))
    rule = MODEL_REGISTRY.get(model_name)
    # Iteration budget is adjusted only with a measured falling training loss.
    if rule and rule.parameter and convergence.get('iteration_limit_reached') and improving:
        key='model__'+rule.parameter;old=parameters.get(key)
        if type(old) is int:
            value=min(math.floor(old*1.25),int(budget[rule.budget_key]))
            if value>old:choices[key]=value
    # A sample-count restriction is a hypothesis, not measured leaf occupancy.
    for name in ('min_samples_leaf','min_samples_split','n_neighbors'):
        key='model__'+name;old=parameters.get(key)
        if type(old) is int and old>=max(3,math.ceil(rows*SCREENING_POLICY['capacity_fraction'])):
            value=max(2 if name=='min_samples_split' else 1,math.ceil(old*.8))
            if value<old:choices[key]=value
    # Do not relax arbitrary alpha/C/depth merely because Q is poor: the gate
    # requires a measured optimization or sample-count capacity symptom.
    return choices


def validate_revision(response, choices):
    if not isinstance(response,dict) or set(response)!={'reason','updates'} or not isinstance(response['reason'],str):
        raise ValueError('Expected reason and updates only')
    updates=response['updates']
    if not isinstance(updates,dict) or len(updates)>1:
        raise ValueError('At most one parameter may change')
    for key,value in updates.items():
        if key not in choices or type(value) is not type(choices[key]) or value!=choices[key]:
            raise ValueError('Change is outside the frozen mild adjustment choices')
    return updates


def _artifact(pipeline, path, context, budget, seed):
    from .method_preflight import validate_estimators
    _,spec=prepare_pipeline(pipeline,context,budget,seed)
    validate_estimators(pipeline)
    joblib.dump(pipeline,path)
    return {'path':str(path),'sha256':file_hash(path),'spec':spec}


async def fit_guarded_proxy(out, method, i, task, pframe, qframe, cfg, model,
                            costs, pricing, request, *, interrupted=None):
    from .multi_proxy import run_fit
    from .method_repair import reuse_interrupted_proxy
    out=Path(out);root=out/'guarded_proxy'/f'method_{i}';root.mkdir(parents=True,exist_ok=True)
    previous=Path(interrupted)/'guarded_proxy'/f'method_{i}' if interrupted else None
    module=load_method_module(method['code_path'])
    full_parameters=deepcopy(method['idea']['stage_parameters']['full_train'])
    full_context={**task,'fit_stage':'full_train','fit_rows':task['training_context']['fixed_train_rows'],
                  'stage_parameters':full_parameters,'parameter_budget_mode':'planned'}
    pipeline=module.build_method(full_context,cfg.full_train_budget,cfg.model_seed)
    prepare_pipeline(pipeline,full_context,cfg.full_train_budget,cfg.model_seed)
    # Record all adjustable values in BOTH stages so full-stage reference checks
    # reset an adjusted coreset parameter to its original value explicitly.
    for key,value in pipeline['model'].get_params(deep=False).items():
        if key in ALLOWED_PARAMETERS:full_parameters['model__'+key]=value
    identity={'code':method['code_sha256'],'train':file_hash(out/'proxy_train.csv'),
              'Q':file_hash(out/'Q_features.csv'),'P_target':digest(pframe[task['target']].tolist()),
              'Q_target':digest(qframe[task['target']].tolist()),
              'task':task,'cfg':{'proxy':cfg.scoring_budget,'core':cfg.evaluation_budget,
                               'full':cfg.full_train_budget,'seed':cfg.model_seed},
              'full_parameters':full_parameters,'policy':dict(SCREENING_POLICY)}
    identity=finite_json(identity)
    if previous and (previous/'identity.json').exists():
        if json.loads((previous/'identity.json').read_text())!=identity:
            raise ValueError('Guarded proxy resume inputs changed')
    else:previous=None
    atomic_json(root/'identity.json',identity)
    # Reuse original immutable artifact paths on resume: worker cache contexts
    # remain identical while method code and training bytes are verified again.
    if previous and (previous/'full_artifact.json').exists():
        full_artifact=json.loads((previous/'full_artifact.json').read_text())
        if file_hash(full_artifact['path'])!=full_artifact['sha256']:
            raise ValueError('Frozen full pipeline changed')
    else:
        full_artifact=_artifact(pipeline,root/'full_unfitted.joblib',
            {'fit_stage':'full_train','parameter_budget_mode':'planned','stage_parameters':full_parameters},
            cfg.full_train_budget,cfg.model_seed)
    atomic_json(root/'full_artifact.json',full_artifact)
    new_fits=reused=0
    def fit(name,parameters,artifact):
        nonlocal new_fits,reused
        context={**task,'expected_family':method['family'],'fit_stage':'proxy',
                 'collect_training_feedback':True,'parameter_budget_mode':'planned',
                 'stage_parameters':parameters,'initial_pipeline_path':artifact['path'],
                 'initial_pipeline_sha256':artifact['sha256'],'expected_pipeline_spec':artifact['spec']}
        folder=root/name
        report=(reuse_interrupted_proxy(previous/name,folder,method['code_path'],out/'proxy_train.csv',
                 out/'Q_features.csv',context,task['classes'],cfg.scoring_budget,cfg.model_seed,cfg.batch_size)
                if previous else None)
        if report is not None:reused+=1
        else:
            report=run_fit(method['code_path'],out/'proxy_train.csv',out/'Q_features.csv',context,
                           task['classes'],cfg.scoring_budget,cfg.model_seed,folder,cfg.batch_size)
            new_fits+=1
        return folder,report
    folder,report=fit('initial',full_parameters,full_artifact)
    with np.load(folder/'predictions.npz',allow_pickle=False) as values:
        evidence=weak_fit_evidence(task,report,pframe[task['target']],qframe[task['target']],
                                  values['predictions'],values['probabilities'],task['classes'])
    choices=allowed_changes(full_parameters,type(pipeline['model']).__name__,report,len(pframe),cfg.scoring_budget)
    decision={'status':'not_triggered','evidence':evidence,'allowed_changes':choices,'updates':{},
              'proposal_limit':1,'refit_limit':1,'full_parameters_immutable':True}
    decision_path=root/'decision.json'
    if previous and (previous/'decision.json').exists():
        decision=json.loads((previous/'decision.json').read_text())
        # A lost API response never grants a second proposal opportunity.
        if decision['status']=='requesting':
            saved=previous/'revision.json'
            if saved.exists():
                try:
                    response=json.loads(saved.read_text());decision.update(updates=validate_revision(response,choices),reason=response['reason'],status='accepted')
                except (ValueError,TypeError):decision.update(status='invalid',updates={})
            else:decision.update(status='interrupted_proposal_kept_baseline',updates={})
    elif evidence['eligible'] and choices:
        decision['status']='requesting';atomic_json(decision_path,finite_json(decision))
        payload={'method_id':i,'name':method['name'],'task':{k:task[k] for k in ('problem_type','metric')},
                 'evidence':evidence,'training_feedback':report['training_feedback'],
                 'training_sizes':task['training_context'],
                 'P_eda':stage_profile(pframe,task,'P only'),
                 'parameters':full_parameters,'allowed_changes':choices}
        atomic_json(root/'input.json',finite_json(payload))
        try:
            response=await request(model,INSTRUCTION+'\n'+json.dumps(finite_json(payload),ensure_ascii=False),
                                   root/'revision.json',costs,pricing)
            atomic_json(root/'revision.json',finite_json(response))
            decision.update(updates=validate_revision(response,choices),reason=response['reason'],status='accepted')
        except (ValueError,TypeError) as error:decision.update(status='invalid_kept_baseline',error=str(error),updates={})
    atomic_json(decision_path,finite_json(decision))
    core_parameters={**full_parameters,**decision['updates']}
    final_artifact=full_artifact
    if decision['updates']:
        validate_revision({'reason':'resume validation','updates':decision['updates']},choices)
        if previous and (previous/'adjusted_artifact.json').exists():
            final_artifact=json.loads((previous/'adjusted_artifact.json').read_text())
            if file_hash(final_artifact['path'])!=final_artifact['sha256']:raise ValueError('Adjusted pipeline changed')
        else:
            final_artifact=_artifact(clone(pipeline),root/'adjusted_unfitted.joblib',
                {'fit_stage':'evaluation','parameter_budget_mode':'planned','stage_parameters':core_parameters},
                cfg.evaluation_budget,cfg.model_seed)
        atomic_json(root/'adjusted_artifact.json',final_artifact)
        folder,report=fit('refit',core_parameters,final_artifact)
    destination=out/'proxies'/f'method_{i}';destination.mkdir(parents=True,exist_ok=True)
    for filename in ('fit_report.json','predictions.npz','fitted.joblib','unfitted.joblib'):
        shutil.copyfile(folder/filename,destination/filename)
    for filename in ('context.json','source_context.json'):
        if (folder/filename).exists():shutil.copyfile(folder/filename,destination/filename)
    atomic_json(destination/'guarded_source.json',{'final_fit':str(folder),'decision':str(decision_path)})
    item={'method_id':i,'model_class':type(pipeline['model']).__name__,
          'reason':decision.get('reason',decision['status']),'rules':{},
          'stage_parameters':{'coreset':core_parameters,'full_train':full_parameters},
          'pipeline_artifacts':{'coreset':final_artifact,'full_train':full_artifact},
          'source_code_sha256':method['code_sha256'],'proxy_refit':bool(decision['updates']),
          'decision_path':str(decision_path)}
    atomic_json(root/'result.json',finite_json(item))
    return report,item,new_fits,reused


def freeze_guarded_plan(out,items):
    plan={'schema_version':1,'status':'frozen','mode':'guarded_once','methods':items,
          'proxy_refit':any(i['proxy_refit'] for i in items),'shared_across_all_coresets':True,
          'validation_used':False,'freeze_point':'before coreset selection',
          'full_parameters_immutable':True,'max_adjustments_per_method':1}
    folder=Path(out)/'parameter_feedback';folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'frozen_plan.json',finite_json(plan))
    return plan
