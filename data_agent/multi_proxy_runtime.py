"""One fresh, budget-enforced FE+MT fit in a child process; no search or retrieval."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, is_classifier, is_regressor, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_limits

if __package__:
    from .multi_proxy_io import read_typed, atomic_json
    from .model_registry import MODEL_REGISTRY
    from .stage1_metrics import primary_score, metric_direction, UndefinedMetricError
else:
    from multi_proxy_io import read_typed, atomic_json
    from model_registry import MODEL_REGISTRY
    from stage1_metrics import primary_score, metric_direction, UndefinedMetricError


def validate_code(code):
    tree = ast.parse(code)
    allowed = {'numpy', 'pandas', 'scipy', 'sklearn', 'math'}
    forbidden = {'open', 'exec', 'eval', 'compile', '__import__', 'getattr', 'setattr',
                 'fit', 'fit_transform', 'partial_fit', 'load', 'dump', 'save', 'savez'}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            raise ValueError('Use standard sklearn transformers and estimators, not custom estimator classes')
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or '']
            if any(m.split('.')[0] not in allowed for m in modules):
                raise ValueError('Generated code imports a module outside the numerical/sklearn interface')
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
            if name in forbidden or name.startswith(('read_', 'to_csv', 'to_pickle', 'to_json', '__')):
                raise ValueError(f'Generated factory may not fit models or perform IO: {name}')
    if not any(isinstance(n, ast.FunctionDef) and n.name == 'build_method' for n in tree.body):
        raise ValueError('Generated code must define build_method(task, budget, seed)')


def enforce_budget(pipeline, budget, seed, exact=False, planned=False):
    if not isinstance(pipeline, Pipeline) or pipeline.steps[-1][0] != 'model':
        raise TypeError('build_method must return a sklearn Pipeline ending with a model step')
    estimator = pipeline.steps[-1][1]
    name = type(estimator).__name__
    requested_model_parameters = {k: repr(v) for k, v in estimator.get_params().items()}
    params = pipeline.get_params(deep=True)
    # A weak learner is part of AdaBoost itself, not an independent predictor.
    from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    allowed_child = (type(estimator) is AdaBoostClassifier and type(params.get('model__estimator')) is DecisionTreeClassifier
                     or type(estimator) is AdaBoostRegressor and type(params.get('model__estimator')) is DecisionTreeRegressor)
    for key, value in params.items():
        if isinstance(value, BaseEstimator) and value is not estimator and (is_classifier(value) or is_regressor(value)):
            if allowed_child and key == 'model__estimator':
                try:
                    check_is_fitted(value)
                except NotFittedError:
                    continue
                raise ValueError('AdaBoost weak learner must be unfitted')
            raise ValueError(f'Additional fitted predictor outside the main model is not allowed: {key}')
    updates = {}
    for key, value in params.items():
        if key.endswith('random_state'):
            updates[key] = seed
        elif key.endswith('n_jobs'):
            updates[key] = 1
        elif key.endswith('warm_start'):
            updates[key] = False
    rule = MODEL_REGISTRY.get(name)
    if rule is not None:
        key = rule.parameter
        limit = budget[rule.budget_key] if rule.budget_key else None
        if rule.family == 'neural':
            updates['model__early_stopping'] = False
    else:
        controls = budget.get('other_controls', {}).get(name)
        if not controls:
            raise ValueError(f'No explicit budget controller for {name}')
        key, limit = controls['parameter'], controls['maximum']
    if key:
        requested = estimator.get_params()[key]
        if planned:
            if isinstance(requested, (bool, np.bool_)) or not isinstance(requested, (int, np.integer)):
                raise ValueError(f'Planned training budget {key} must be a positive integer')
            if requested > limit:
                raise ValueError(f'Planned training budget {key}={requested} exceeds cap {limit}')
            updates['model__'+key] = int(requested)
        else:
            updates['model__'+key] = int(limit) if exact else min(int(requested), int(limit))
        if updates['model__'+key] < 1:
            raise ValueError('Effective training budget must be positive')
    pipeline.set_params(**updates)
    # A factory returns a fresh unfitted instance, never a persisted proxy model.
    try:
        check_is_fitted(estimator)
    except NotFittedError:
        pass
    else:
        raise ValueError('Generated factory returned an already fitted model')
    return {'model_class': name, 'budget': dict(budget),
            'requested_model_parameters': requested_model_parameters, 'enforced_parameters': updates,
            'model_parameters': {k: repr(v) for k, v in estimator.get_params().items()}}


def model_family(name):
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name].family
    if name.startswith(('RandomForest', 'ExtraTrees')): return 'bagging'
    if name.startswith(('GradientBoosting', 'HistGradientBoosting', 'AdaBoost')): return 'boosting'
    if name.startswith('MLP'): return 'neural'
    if name.startswith('KNeighbors'): return 'neighbors'
    if name == 'GaussianNB': return 'naive_bayes'
    if name.startswith('DecisionTree'): return 'tree'
    return 'linear'


def align_probabilities(probabilities, model_classes, classes):
    values = np.asarray(probabilities, dtype=float)
    order = np.asarray(model_classes)
    if values.ndim != 2 or values.shape[1] != len(order) or set(order.tolist()) != set(classes):
        raise ValueError('Probability classes do not match shared training class order')
    values = values[:, [order.tolist().index(c) for c in classes]]
    if not np.isfinite(values).all() or (values < -1e-9).any() or (values > 1+1e-9).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-5):
        raise ValueError('Invalid predicted probabilities')
    return values


def pipeline_spec(value):
    """Stable pre-fit configuration including nested FE and estimator parameters."""
    if isinstance(value, BaseEstimator):
        return {'class':type(value).__module__+'.'+type(value).__qualname__,
                'parameters':{k:pipeline_spec(v) for k,v in sorted(value.get_params(deep=False).items())}}
    if isinstance(value, dict):
        return {str(k):pipeline_spec(v) for k,v in sorted(value.items(),key=lambda item:str(item[0]))}
    if isinstance(value,(list,tuple)):
        return [pipeline_spec(v) for v in value]
    if isinstance(value,np.ndarray):
        return {'dtype':str(value.dtype),'values':pipeline_spec(value.tolist())}
    if isinstance(value,np.generic):
        return pipeline_spec(value.item())
    if callable(value):
        return {'callable':value.__module__+'.'+value.__qualname__}
    if value is None or isinstance(value,(str,bool,int)):
        return value
    if isinstance(value,float):
        return value if np.isfinite(value) else str(value)
    raise ValueError('Unsupported parameter for reproducible pipeline comparison: '+str(type(value)))


class ModelOutsideSearch(ValueError):
    """A valid estimator outside the chosen search scope needs no code repair."""


def check_search_model(model_class, allowed):
    if allowed is not None and model_class not in allowed:
        raise ModelOutsideSearch(f"Estimator {model_class} is outside the parameter-focus shortlist {allowed}")


def prepare_pipeline(pipeline, task, budget, seed):
    parameters=task.get('stage_parameters')
    if parameters is not None:
        pipeline.set_params(**parameters)
    planned = task.get('parameter_budget_mode') == 'planned'
    actual=enforce_budget(pipeline,budget,seed,exact=task.get('fit_stage') in ('evaluation','full_train'),
                          planned=planned)
    check_search_model(actual['model_class'], task.get('allowed_model_classes'))
    if task.get('expected_model_class') and actual['model_class']!=task['expected_model_class']:
        raise ValueError('Full-train model class differs from coreset method')
    specification=pipeline_spec(pipeline)
    if task.get('expected_pipeline_spec') is not None and specification!=task['expected_pipeline_spec']:
        raise ValueError('Coreset fairness violation: pipeline parameters differ between coresets')
    if task.get('reference_pipeline_path'):
        reference=joblib.load(task['reference_pipeline_path'])
        reference.set_params(**(parameters or {}))
        enforce_budget(reference,budget,seed,exact=True,planned=planned)
        if specification!=pipeline_spec(reference):
            raise ValueError('Full-train configuration changed outside predeclared stage parameters')
    return actual,specification


def normalize_numeric_infinities(frame):
    """Keep categorical dtypes stable, including entirely missing CSV batches."""
    result = frame.copy()
    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan)
    return result


def finite_feedback(value):
    """Training diagnostics must be strict JSON, including failed numeric diagnostics."""
    if isinstance(value, np.ndarray):
        return finite_feedback(value.tolist())
    if isinstance(value, np.generic):
        return finite_feedback(value.item())
    if isinstance(value, dict):
        return {str(key): finite_feedback(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_feedback(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    return str(value)


def feedback_source(task, *, evaluation_split=None, training_rows=None):
    """Name both fit and evaluation populations without conflating their scores."""
    stage = {'proxy': 'P', 'evaluation': 'coreset', 'low': 'coreset',
             'full': 'full_train'}.get(task.get('fit_stage'), task.get('fit_stage', 'P'))
    return {'fit_stage': stage, 'evaluation_split': evaluation_split or stage,
            'training_rows': training_rows if training_rows is not None else task.get('fit_rows')}


def task_metric_feedback(task, y, predictions, probabilities, classes, source, *, measured_value=None):
    """Canonical whole-population metric; undefined values retain their reason."""
    result = {'metric': task.get('metric'), 'metric_params': task.get('metric_params') or {},
              'source': dict(source), 'rows': len(y), 'value': None}
    if not task.get('metric'):
        return {**result, 'status': 'unavailable', 'reason': 'Task primary metric was not supplied'}
    result['direction'] = metric_direction(task)
    try:
        result['value'] = (primary_score(task, y, predictions, probabilities, classes)
                           if measured_value is None else float(measured_value))
        if not np.isfinite(result['value']):
            raise UndefinedMetricError('Task primary metric is nonfinite')
        result['status'] = 'success'
    except UndefinedMetricError as error:
        result.update(status='undefined', reason=str(error))
    return finite_feedback(result)


def collect_training_feedback(pipeline, x, y, problem_type, classes, batch_size, *, task=None):
    """Reuse fitted predictions, aggregating rows before nonadditive task metrics."""
    started = time.perf_counter()
    totals = {'accuracy': 0.0, 'log_loss': 0.0} if problem_type == 'classification' else {'mse': 0.0, 'mae': 0.0}
    predictions, probabilities = [], []
    for offset in range(0, len(x), batch_size):
        batch = x.iloc[offset:offset + batch_size]
        target = y.iloc[offset:offset + batch_size].to_numpy()
        predicted = np.asarray(pipeline.predict(batch))
        if predicted.shape != (len(batch),):
            raise ValueError('Training feedback predictions must preserve input row count and order')
        predictions.append(predicted)
        if problem_type == 'classification':
            probs = align_probabilities(pipeline.predict_proba(batch), pipeline.classes_, classes)
            probabilities.append(probs)
            totals['accuracy'] += float(np.sum(predicted == target))
            totals['log_loss'] += primary_score(
                {'problem_type': 'classification', 'metric': 'log_loss'},
                target, predicted, probs, classes) * len(batch)
        else:
            residual = predicted.astype(float) - target.astype(float)
            totals['mse'] += float(np.sum(np.square(residual)))
            totals['mae'] += float(np.sum(np.abs(residual)))
    model = pipeline.steps[-1][1]
    iterations = getattr(model, 'n_iter_', None)
    limit = model.get_params(deep=False).get('max_iter')
    reached = None
    if iterations is not None and isinstance(limit, (int, np.integer)):
        try:
            reached = bool(np.any(np.asarray(iterations, dtype=float) >= limit))
        except (TypeError, ValueError):
            pass
    curve = getattr(model, 'loss_curve_', None)
    curve_summary = None
    if curve is not None and len(curve):
        curve_summary = {'length': len(curve), 'first': curve[0], 'last': curve[-1], 'tail': list(curve[-5:])}
    result = {
        'fit_rows': len(x),
        'training_metrics': {key: value / len(x) for key, value in totals.items()},
        'convergence': {'n_iter': iterations, 'iteration_limit': limit, 'iteration_limit_reached': reached,
                        'loss': getattr(model, 'loss_', None), 'loss_curve': curve_summary},
        'transformed_feature_count': None,
    }
    source = feedback_source(task or {'fit_stage': 'proxy'}, training_rows=len(x))
    result['metrics_source'] = source
    if task is not None:
        result['primary_metric'] = task_metric_feedback(
            task, np.asarray(y), np.concatenate(predictions),
            np.concatenate(probabilities) if probabilities else None, classes, source)
        if task.get('metric'):
            result['training_metrics'][task['metric']] = result['primary_metric']['value']
    # Estimator metadata avoids even a transform for most sklearn estimators.
    feature_count = getattr(model, 'n_features_in_', None)
    if feature_count is not None:
        result['transformed_feature_count'] = int(feature_count)
    else:
        try:
            sample = x.iloc[:min(128, len(x))]
            transformed = pipeline[:-1].transform(sample) if len(pipeline.steps) > 1 else sample
            result['transformed_feature_count'] = int(transformed.shape[1])
        except Exception as error:
            result['feature_count_error'] = f'{type(error).__name__}: {error}'
    result['feedback_seconds'] = time.perf_counter() - started
    return finite_feedback(result)


def load_serialized_pipeline(fitted_path, code_path):
    """Restore custom functions from a fresh module, as final inference does."""
    spec = importlib.util.spec_from_file_location('generated_method', code_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['generated_method'] = module
    spec.loader.exec_module(module)
    return joblib.load(fitted_path)


def check_serialized_predictions(pipeline, fitted_path, code_path, x, problem_type, classes):
    """Check a fresh-import load on training rows, without refitting or test access.

    A same-module joblib load misses feature functions that depend on mutable
    globals initialized only by build_method. Reimporting the source reproduces
    the final inference lifecycle and detects that state loss while repair is
    still possible.
    """
    started = time.perf_counter()
    positions = np.linspace(0, len(x) - 1, min(64, len(x)), dtype=int)
    sample = x.iloc[positions]
    expected = np.asarray(pipeline.predict(sample))
    expected_proba = (align_probabilities(pipeline.predict_proba(sample), pipeline.classes_, classes)
                      if problem_type == 'classification' else None)
    original = sys.modules.get('generated_method')
    try:
        restored = load_serialized_pipeline(fitted_path, code_path)
        actual = np.asarray(restored.predict(sample))
        if expected.shape != actual.shape:
            raise ValueError('prediction shape changed after reload')
        if problem_type == 'classification':
            if not np.array_equal(expected, actual):
                raise ValueError('class predictions changed after reload')
            actual_proba = align_probabilities(restored.predict_proba(sample), restored.classes_, classes)
            if not np.allclose(expected_proba, actual_proba, rtol=1e-10, atol=1e-12):
                raise ValueError('class probabilities changed after reload')
        elif not np.allclose(expected.astype(float), actual.astype(float), rtol=1e-10, atol=1e-12):
            raise ValueError('regression predictions changed after reload')
        # Discard any state changed by the probe. Keep this fresh module bound
        # for the restored function references used in all subsequent scoring.
        del restored
        restored = load_serialized_pipeline(fitted_path, code_path)
    except Exception as error:
        if original is None:
            sys.modules.pop('generated_method', None)
        else:
            sys.modules['generated_method'] = original
        raise ValueError('Serialized pipeline roundtrip failed: fitted feature behavior must survive a fresh '
                         'module import without calling build_method. Store required feature state in '
                         f'serialized transformer parameters, not factory-initialized globals. {type(error).__name__}: {error}') from error
    return restored, {'status': 'success', 'rows': len(sample), 'source': 'training_features_only',
                      'sample_policy': 'evenly_spaced_training_rows',
                      'fresh_module_import': True, 'seconds': time.perf_counter() - started}


def execute(context):
    worker_started = time.perf_counter()
    out = Path(context['output'])
    code = Path(context['code']).read_text()
    validate_code(code)
    np.random.seed(context['seed'])
    spec = importlib.util.spec_from_file_location('generated_method', context['code'])
    module = importlib.util.module_from_spec(spec)
    sys.modules['generated_method'] = module
    spec.loader.exec_module(module)
    task = dict(context['task'])
    schema = task.get('csv_schema')
    train = read_typed(context['train'], schema, low_memory=False) if schema else pd.read_csv(context['train'], low_memory=False)
    features = task['feature_columns']
    x, y = normalize_numeric_infinities(train[features]), train[task['target']]
    task['fit_rows'] = len(train)
    task['fit_class_counts'] = [
        {'label': label.item() if isinstance(label, np.generic) else label, 'count': int(count)}
        for label, count in y.value_counts(sort=False).items()
    ] if task['problem_type'] == 'classification' else []
    fit_context = {key: task.get(key) for key in ('fit_rows', 'fit_class_counts', 'fit_stage')}
    if task.get('initial_pipeline_path'):
        initial=Path(task['initial_pipeline_path'])
        if not task.get('expected_pipeline_spec') or hashlib.sha256(initial.read_bytes()).hexdigest()!=task.get('initial_pipeline_sha256'):
            raise ValueError('Frozen pipeline requires verified hash and expected pre-fit specification')
        # Clone discards all learned state even if an artifact were accidentally fitted.
        pipeline=clone(joblib.load(initial))
    else:
        pipeline = module.build_method(task, context['budget'], context['seed'])
    actual, specification = prepare_pipeline(pipeline, task, context['budget'], context['seed'])
    if task.get('expected_family') and model_family(actual['model_class']) != task['expected_family']:
        raise ValueError('Generated implementation differs from the planned modeling mechanism')
    joblib.dump(pipeline, out/'unfitted.joblib')
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always')
        started = time.perf_counter()
        pipeline.fit(x, y)
        fit_seconds = time.perf_counter()-started
        joblib.dump(pipeline, out/'fitted.joblib')
        pipeline, serialization_check = check_serialized_predictions(
            pipeline, out/'fitted.joblib', context['code'], x, task['problem_type'], context['classes'])
        training_feedback = None
        if task.get('collect_training_feedback'):
            training_feedback = collect_training_feedback(
                pipeline, x, y, task['problem_type'], context['classes'], context['batch_size'], task=task)
        prediction_started = time.perf_counter()
        prediction_seconds = 0.0
        predictions, probabilities = [], []
        chunks = read_typed(context['predict'], schema, low_memory=False, chunksize=context['batch_size']) if schema else pd.read_csv(context['predict'], low_memory=False, chunksize=context['batch_size'])
        for chunk in chunks:
            started = time.perf_counter()
            x = normalize_numeric_infinities(chunk[features])
            pred = np.asarray(pipeline.predict(x))
            if pred.shape != (len(x),):
                raise ValueError('predict must return a one-dimensional array in input row order')
            if task['problem_type'] == 'classification':
                if not np.isin(pred, context['classes']).all():
                    raise ValueError('Unknown predicted class')
                probs = align_probabilities(pipeline.predict_proba(x), pipeline.classes_, context['classes'])
                probabilities.append(probs)
            elif not np.isfinite(pred.astype(float)).all():
                raise ValueError('Nonfinite regression prediction')
            predictions.append(pred)
            prediction_seconds += time.perf_counter()-started
        prediction_wall_seconds = time.perf_counter()-prediction_started
    predictions = np.concatenate(predictions)
    result = pd.DataFrame({'row_position': np.arange(len(predictions)), 'prediction': predictions})
    result.to_csv(out/'predictions.csv', index=False)
    np.savez_compressed(out/'predictions.npz', predictions=predictions.astype(str) if task['problem_type']=='classification' else predictions,
                        probabilities=np.concatenate(probabilities) if probabilities else np.empty((0,0)),
                        row_positions=np.arange(len(predictions)))
    report = {'status': 'success', 'fit_calls': 1, 'fit_rows': len(train), 'prediction_rows': len(predictions),
              'fit_context': fit_context, 'pipeline_spec': specification, 'stage_parameters': task.get('stage_parameters'), 'requested_budget': dict(context['budget']),
              'fit_seconds': fit_seconds, 'prediction_seconds': prediction_seconds,
              'prediction_wall_seconds': prediction_wall_seconds,
              'worker_wall_seconds': time.perf_counter()-worker_started, 'actual_budget': actual,
              'serialization_check': serialization_check,
              'prediction_source': 'serialized_fitted_pipeline',
              'classes': context['classes'], 'warnings': [str(w.message) for w in caught]}
    if training_feedback is not None:
        report['training_feedback'] = training_feedback
    atomic_json(out/'fit_report.json', report)


if __name__ == '__main__':
    context = json.loads(Path(sys.argv[1]).read_text())
    try:
        execute(context)
    except Exception as error:
        atomic_json(Path(context['output'])/'failure.json', {'error': f'{type(error).__name__}: {error}'})
        traceback.print_exc()
        sys.exit(1)
