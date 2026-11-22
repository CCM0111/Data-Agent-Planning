"""One training-only parameter revision, frozen before any screening scores."""
import json
import importlib.util
import math
from copy import deepcopy
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from .coreset import file_hash
from .data_profile import (training_profile, feature_structure_profile,
                           feature_distribution_difference, feature_metadata_for_task)
from .multi_proxy_io import atomic_json


def load_method_module(code_path):
    """Restore top-level FE functions referenced by existing joblib pipelines."""
    from .multi_proxy_runtime import validate_code
    path = Path(code_path)
    validate_code(path.read_text())
    spec = importlib.util.spec_from_file_location('generated_method', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def stage_profile(frame, task, scope):
    """Only actual training features and aggregate targets, never validation data."""
    columns = list(task['feature_columns']) + [task['target']]
    frame = frame[columns]
    result = training_profile(frame, task['target'], task['problem_type'])
    result['scope'] = scope
    result['feature_structure'] = feature_structure_profile(
        frame, feature_columns=task['feature_columns'],
        categorical_columns=task['categorical_columns'], scope=scope,
        metadata=feature_metadata_for_task(task, task['feature_columns']))
    result['columns'] = {}
    selected = {entry['feature'] for entry in result['feature_structure']['feature_details']}
    for name in task['feature_columns']:
        if str(name) not in selected:
            continue
        values = frame[name]
        distinct = int(values.nunique(dropna=True))
        result['columns'][name] = {
            'missing': int(values.isna().sum()),
            'distinct': distinct,
            'constant': distinct <= 1,
            'kind': 'categorical' if name in task['categorical_columns'] else 'numeric',
        }
    result['column_summary_scope'] = 'Full supplied rows for the bounded feature_details selection; omitted columns are unreported'
    result['omitted_column_summaries'] = len(task['feature_columns']) - len(result['columns'])
    if task['problem_type'] == 'classification':
        result['class_counts'] = {str(k): int(v) for k, v in frame[task['target']].value_counts().items()}
    return finite_json(result)


def prediction_feedback(y, predictions, probabilities, classes, problem, *, task=None, source=None):
    y = np.asarray(y)
    predictions = np.asarray(predictions)
    if problem == 'classification':
        lookup = {str(label): i for i, label in enumerate(classes)}
        encoded = np.array([lookup[str(label)] for label in y])
        values = np.asarray(probabilities, dtype=float)
        losses = -np.log(np.clip(values[np.arange(len(y)), encoded], np.finfo(float).eps, 1. - np.finfo(float).eps))
        incorrect = y.astype(str) != predictions.astype(str)
        metrics = {'accuracy': float(np.mean(~incorrect)), 'log_loss': float(np.mean(losses))}
        group_labels = y.astype(str)
        groups = sorted(set(group_labels), key=lambda s: (int(np.sum(group_labels == s)), s))[:12]
    else:
        residual = predictions.astype(float) - y.astype(float)
        with np.errstate(over='ignore', invalid='ignore'):
            losses = residual ** 2
        incorrect = np.abs(residual)
        metrics = {'mse': float(np.mean(losses)), 'mae': float(np.mean(incorrect))}
        boundaries = np.unique(np.quantile(y.astype(float), np.linspace(0, 1, 6)))
        group_labels = np.searchsorted(boundaries[1:-1], y.astype(float), side='right')
        groups = sorted(set(group_labels))
    summaries = []
    for group in groups:
        selected = group_labels == group
        row = {'group': str(group), 'rows': int(np.sum(selected)),
               'mean_loss': float(np.mean(losses[selected]))}
        row['error_rate' if problem == 'classification' else 'mae'] = float(np.mean(incorrect[selected]))
        summaries.append(row)
    result = {'rows': len(y), 'metrics': metrics, 'groups': summaries,
              'grouping': 'up to 12 smallest classes' if problem == 'classification' else 'target quantile bins'}
    if source is not None:
        result['metrics_source'] = dict(source)
    if task is not None:
        from .multi_proxy_runtime import task_metric_feedback
        # Prediction artifacts store class labels as strings. Restore the actual
        # class types so F1 pos_label/labels and AUC parameters retain their meaning.
        if problem == 'classification':
            labels = {str(label): label for label in classes}
            predictions = np.asarray([labels[str(value)] for value in predictions])
        result['primary_metric'] = task_metric_feedback(
            task, y, predictions, probabilities, classes, source or {})
        if task.get('metric'):
            metrics[task['metric']] = result['primary_metric']['value']
    return finite_json(result)


def runtime_risk(pipeline, report, sizes, cfg):
    """Prospective timing evidence only; never silently changes frozen parameters."""
    from .model_registry import MODEL_REGISTRY
    model = pipeline['model']
    rule = MODEL_REGISTRY.get(type(model).__name__)
    seconds = report.get('fit_seconds')
    proxy_rows = report.get('training_feedback', {}).get('fit_rows')
    result = {'scope':'Heuristic from P fit only; excludes prediction and may underestimate nonlinear scaling. '
                       'Leave substantial wall-time headroom. No guarantee or automatic parameter change.', 'stages':{}}
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not proxy_rows:
        result['status'] = 'timing_unavailable'
        return result
    for stage, rows, budget in [('coreset', sizes['coreset_rows'], cfg.evaluation_budget),
                                ('full_train', sizes['fixed_train_rows'], cfg.full_train_budget)]:
        scale = rows / proxy_rows
        if rule and rule.family in ('bagging', 'boosting', 'tree'):
            scale *= math.log2(max(2, rows)) / math.log2(max(2, proxy_rows))
        resource_scale = 1
        if rule and rule.parameter:
            resource_scale = budget[rule.budget_key] / model.get_params()[rule.parameter]
        estimate = seconds * scale * resource_scale
        result['stages'][stage] = {'fit_seconds_at_resource_cap_estimate':estimate,
            'wall_budget_seconds':budget['timeout_seconds'],
            'risk_with_twofold_headroom':2*estimate > budget['timeout_seconds']}
    result['status'] = 'estimated'
    return result


def formal_stage_baselines(method, module, task, cfg, folder, method_id):
    """Construct the two original formal stages without fitting or using labels.

    P-only safeguards are deliberately reevaluated at the actual formal row
    counts. The original budget mode is respected before freezing planned
    overrides; this records effective values rather than ignored declarations.
    """
    from .multi_proxy_runtime import prepare_pipeline
    from .method_preflight import validate_estimators
    from .parameter_feedback import complete_stage_parameters
    from .model_registry import MODEL_REGISTRY
    sizes = task['training_context']
    originals, parameters, artifacts = {}, {}, {}
    sys.modules['generated_method'] = module
    for phase, stage, rows, budget in [
        ('coreset', 'evaluation', sizes['coreset_rows'], cfg.evaluation_budget),
        ('full_train', 'full_train', sizes['fixed_train_rows'], cfg.full_train_budget),
    ]:
        declared = method['idea']['stage_parameters'][phase]
        context = {**task, 'fit_stage': stage, 'fit_rows': rows,
                   'fit_class_counts': sizes.get('proxy_class_counts', []),
                   'stage_parameters': declared}
        pipeline = module.build_method(context, budget, cfg.model_seed)
        prepare_pipeline(pipeline, context, budget, cfg.model_seed)
        validate_estimators(pipeline)
        originals[phase] = pipeline
        available = pipeline.get_params(deep=True)
        parameters[phase] = {key: deepcopy(available[key]) for key in declared}
        rule = MODEL_REGISTRY.get(type(pipeline['model']).__name__)
        if rule and rule.parameter:
            key = 'model__' + rule.parameter
            parameters[phase][key] = deepcopy(available[key])
    parameters = complete_stage_parameters(parameters, originals)
    for phase, stage, budget in [('coreset', 'evaluation', cfg.evaluation_budget),
                                  ('full_train', 'full_train', cfg.full_train_budget)]:
        context = {'fit_stage': stage, 'parameter_budget_mode': 'planned',
                   'stage_parameters': parameters[phase]}
        if phase == 'full_train':
            context['reference_pipeline_path'] = artifacts['coreset']['path']
            context['expected_model_class'] = type(originals['coreset']['model']).__name__
        actual, specification = prepare_pipeline(originals[phase], context, budget, cfg.model_seed)
        path = Path(folder) / f'method_{method_id}_{phase}_baseline.joblib'
        joblib.dump(originals[phase], path)
        artifacts[phase] = {'path': str(path), 'sha256': file_hash(path), 'spec': specification,
                            'actual_budget': actual}
    return originals, parameters, artifacts


async def refine_parameters(out, methods, task, eda, pframe, qframe, selections,
                            proxy_reports, cfg, model, costs, pricing, request):
    """No V_core/external validation objects enter this function."""
    from .parameter_feedback import PARAMETER_FEEDBACK_INSTRUCTION, freeze_parameter_plan, normalize_parameter_names
    from .multi_proxy_runtime import prepare_pipeline
    from .method_preflight import validate_estimators
    from .mcts_changes import differences

    out = Path(out)
    folder = out / 'parameter_feedback'
    folder.mkdir()
    sizes = task['training_context']
    coresets = []
    allowed_features = pd.concat([pframe[task['feature_columns']], qframe[task['feature_columns']]])
    # qframe retains fixed-train indices; every selection must belong to Q.
    for selected in selections:
        if not set(map(int, selected)) <= set(map(int, qframe.index)):
            raise ValueError('Feedback coreset EDA must use Q rows only')
        coreset = stage_profile(qframe.loc[selected], task, 'actual coreset; subset of Q')
        coreset['difference_from_allowed_training'] = feature_distribution_difference(
            allowed_features, qframe.loc[selected], feature_columns=task['feature_columns'],
            categorical_columns=task['categorical_columns'],
            reference_scope='P + Q only; V_core, external validation and test excluded',
            candidate_scope='actual coreset; subset of Q')
        coresets.append(coreset)
    # Canonical order removes sampler names and sampling-seed identities.
    coresets.sort(key=lambda item: json.dumps(item, sort_keys=True))
    payload = {
        'task': {k: task[k] for k in ('name', 'target', 'problem_type', 'metric', 'metric_params') if k in task},
        'training_sizes': sizes,
        'stage_eda': {
            'proxy': stage_profile(pframe, task, 'P only'),
            'coresets': coresets,
            'full_train': {
                'planned_train_rows': sizes['fixed_train_rows'],
                'eda_source_rows': int(len(pframe) + len(qframe)),
                'scope': 'P + Q only; V_core excluded even though final training includes it',
                'eda': eda,
            },
        },
        'budgets': {'proxy': cfg.scoring_budget, 'coreset': cfg.evaluation_budget,
                    'full_train': cfg.full_train_budget},
        'methods': [],
        'policy': 'One shared plan per method across all coresets. No validation scores, no per-arm tuning. '
                  'The initial P proxy and its Q sampling scores are not refitted in this version.',
    }
    pipelines, stage_pipelines, stage_baselines, baseline_artifacts = [], [], [], []
    modules = []
    for i, method in enumerate(methods):
        source = out / 'proxies' / f'method_{i}'
        modules.append(load_method_module(method['code_path']))
        pipeline = joblib.load(source / 'unfitted.joblib')
        pipelines.append(pipeline)
        originals, baseline, artifacts = formal_stage_baselines(method, modules[-1], task, cfg, folder, i)
        stage_pipelines.append(originals)
        stage_baselines.append(baseline)
        baseline_artifacts.append(artifacts)
        report = proxy_reports[i]
        if not report.get('training_feedback'):
            raise ValueError('Parameter feedback requires recorded P training diagnostics')
        with np.load(source / 'predictions.npz', allow_pickle=False) as predictions:
            q_feedback = prediction_feedback(qframe[task['target']], predictions['predictions'],
                                             predictions['probabilities'], task['classes'], task['problem_type'],
                                             task=task, source={'fit_stage': 'P', 'evaluation_split': 'Q',
                                                               'training_rows': len(pframe)})
        payload['methods'].append({
            'method_id': i, 'name': method['name'], 'model_class': type(pipeline['model']).__name__,
            'proxy_feedback': report['training_feedback'], 'q_feedback': q_feedback,
            'proxy_fit_summary': {k: report.get(k) for k in ('warnings', 'fit_seconds', 'actual_budget')},
            'runtime_risk': runtime_risk(pipeline, report, sizes, cfg),
            'proxy_pipeline_spec': report['pipeline_spec'],
            'initial_stage_parameters': method['idea']['stage_parameters'],
            'formal_stage_changes_from_proxy': {
                phase: differences(report['pipeline_spec'], artifact['spec'])
                for phase, artifact in artifacts.items()},
        })
    baseline_plan = freeze_parameter_plan(
        {'methods': [{'method_id': i, 'reason': 'Validate original formal-stage baseline', 'rules': {}}
                     for i in range(len(methods))]},
        pipelines, sizes['coreset_rows'], sizes['fixed_train_rows'],
        {'coreset': cfg.evaluation_budget, 'full_train': cfg.full_train_budget},
        stage_baselines=stage_baselines, stage_pipelines=stage_pipelines)
    for item, baseline in zip(payload['methods'], baseline_plan['methods']):
        item['validated_stage_baseline'] = baseline['stage_parameters']
    payload = finite_json(payload)
    atomic_json(folder / 'input.json', payload)
    prompt = PARAMETER_FEEDBACK_INSTRUCTION + '\n' + json.dumps(payload, ensure_ascii=False, allow_nan=False)
    ledger = {'max_repairs': 2, 'attempts': [], 'status': 'running', 'additional_fit_calls': 0}
    for attempt in range(ledger['max_repairs'] + 1):
        revision_path = folder / ('revision.json' if attempt == 0 else f'revision_repair_{attempt}.json')
        record = {'attempt': attempt, 'path': str(revision_path), 'status': 'requesting'}
        ledger['attempts'].append(record)
        atomic_json(folder / 'revision_attempts.json', ledger)
        # The normal adapter records each actual request, token usage and cost.
        # Transport errors propagate; only a returned invalid plan is corrected.
        try:
            response = await request(model, prompt, revision_path, costs, pricing)
        except Exception as error:
            record.update(status='request_failed', error=f'{type(error).__name__}: {error}')
            ledger['status'] = 'request_failed'
            atomic_json(folder / 'revision_attempts.json', ledger)
            raise
        atomic_json(revision_path, finite_json(response))
        try:
            normalized, changes = normalize_parameter_names(response, pipelines)
            atomic_json(revision_path.with_suffix('.normalization.json'),
                        finite_json({'changes': changes, 'response': normalized}))
            plan = freeze_parameter_plan(normalized, pipelines, sizes['coreset_rows'], sizes['fixed_train_rows'],
                                         {'coreset': cfg.evaluation_budget, 'full_train': cfg.full_train_budget},
                                         stage_baselines=[item['stage_parameters'] for item in baseline_plan['methods']],
                                         stage_pipelines=stage_pipelines)
        except (ValueError, TypeError, KeyError) as error:
            record.update(status='invalid', error=f'{type(error).__name__}: {error}')
            ledger['status'] = 'exhausted' if attempt == ledger['max_repairs'] else 'correcting'
            atomic_json(folder / 'revision_attempts.json', ledger)
            if attempt == ledger['max_repairs']:
                raise ValueError('Parameter feedback exhausted two semantic correction attempts: ' + str(error)) from error
            prompt = PARAMETER_FEEDBACK_INSTRUCTION + '''\nCorrect only the invalid parameter rules described below.
Use the same original P/Q evidence and validated formal-stage baselines. No new fits, tuning trials,
V_core, external validation or test feedback are available. Preserve valid unrelated choices and return
the complete methods JSON using the existing schema. This is a validity repair, not another performance revision.
''' + json.dumps({'original_input': payload, 'invalid_revision': finite_json(response),
                  'validation_error': record['error']}, ensure_ascii=False, allow_nan=False)
        else:
            record['status'] = 'accepted'
            ledger.update(status='accepted', accepted_revision_path=str(revision_path))
            atomic_json(folder / 'revision_attempts.json', ledger)
            break
    for item in plan['methods']:
        i = item['method_id']
        # Each generated source uses the same module name in child processes.
        # Restore its own function objects before serializing that method.
        sys.modules['generated_method'] = modules[i]
        artifacts = {}
        for phase, fit_stage, budget in [('coreset', 'evaluation', cfg.evaluation_budget),
                                         ('full_train', 'full_train', cfg.full_train_budget)]:
            pipeline = clone(stage_pipelines[i][phase])
            context = {'fit_stage': fit_stage, 'parameter_budget_mode': 'planned',
                       'stage_parameters': item['stage_parameters'][phase]}
            if phase == 'full_train':
                context['reference_pipeline_path'] = artifacts['coreset']['path']
            _, spec = prepare_pipeline(pipeline, context, budget, cfg.model_seed)
            validate_estimators(pipeline)
            path = folder / f'method_{i}_{phase}.joblib'
            joblib.dump(pipeline, path)
            artifacts[phase] = {'path': str(path), 'sha256': file_hash(path), 'spec': spec}
        item['pipeline_artifacts'] = artifacts
        item['baseline_pipeline_artifacts'] = baseline_artifacts[i]
        item['source_code_sha256'] = methods[i]['code_sha256']
    plan.update(status='frozen', input_sha256=file_hash(folder / 'input.json'),
                revision_sha256=file_hash(revision_path), revision_path=str(revision_path),
                semantic_repairs_used=attempt,
                proxy_refit=False, shared_across_all_coresets=True,
                validation_used=False, freeze_point='after selection, before formal_evaluation')
    atomic_json(folder / 'frozen_plan.json', plan)
    return plan


def frozen_fit_context(plan, method_id, phase):
    item = next(item for item in plan['methods'] if item['method_id'] == method_id)
    artifact = item['pipeline_artifacts'][phase]
    return {'stage_parameters': item['stage_parameters'][phase],
            'parameter_budget_mode': 'planned',
            'initial_pipeline_path': artifact['path'],
            'initial_pipeline_sha256': artifact['sha256'],
            'expected_pipeline_spec': artifact['spec']}
