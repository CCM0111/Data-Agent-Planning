"""Fresh-process inference on a frozen fitted pipeline; never calls its factory."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from data_agent.coreset import file_hash
from data_agent.multi_proxy_io import atomic_json, read_typed
from data_agent.multi_proxy_runtime import (align_probabilities, load_serialized_pipeline,
                                            normalize_numeric_infinities, validate_code)


def _mapped_columns(transformer, columns, issues=None, path='features'):
    """Map each fitted output position to its verified original input column.

    Unknown ColumnTransformer branches keep their exact output width as None,
    so a custom feature branch cannot invalidate or shift a reliable sibling.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (StandardScaler, MinMaxScaler, MaxAbsScaler,
                                       RobustScaler, Normalizer, PowerTransformer,
                                       QuantileTransformer, OneHotEncoder, OrdinalEncoder, FunctionTransformer)
    issues = [] if issues is None else issues
    if isinstance(transformer, str):
        if transformer == 'passthrough':
            return list(columns)
        if transformer == 'drop':
            return []
        raise ValueError('Unknown feature operation')
    if isinstance(transformer, Pipeline):
        names = list(columns)
        for name, step in transformer.steps:
            names = _mapped_columns(step, names, issues, f'{path}.{name}')
        return names
    if getattr(transformer, 'n_features_in_', len(columns)) != len(columns):
        raise ValueError('Fitted input width differs from the mapped input width')
    input_names = list(getattr(transformer, 'feature_names_in_',
                               [f'x{i}' for i in range(len(columns))]))
    if isinstance(transformer, ColumnTransformer):
        output = [None] * max((part.stop for part in transformer.output_indices_.values()), default=0)
        for name, step, selector in transformer.transformers_:
            positions = transformer.output_indices_[name]
            width = positions.stop - positions.start
            if not width:
                continue
            try:
                if isinstance(selector, slice):
                    if isinstance(selector.start, str) or isinstance(selector.stop, str):
                        selector = pd.Index(input_names).slice_indexer(selector.start, selector.stop, selector.step)
                    selected = list(columns)[selector]
                else:
                    values = [selector] if isinstance(selector, str) else np.atleast_1d(selector).tolist()
                    if values and all(isinstance(value, bool) for value in values):
                        if len(values) != len(columns):
                            raise ValueError('Invalid boolean feature selector')
                        selected = [column for column, keep in zip(columns, values) if keep]
                    else:
                        indices = [int(value) if isinstance(value, (int, np.integer)) else input_names.index(value)
                                   for value in values]
                        selected = [columns[index] for index in indices]
                mapped = _mapped_columns(step, selected, issues, f'{path}.{name}')
                if len(mapped) != width:
                    raise ValueError('Mapped branch width differs from fitted ColumnTransformer output')
                output[positions] = mapped
            except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
                issues.append({'path': f'{path}.{name}', 'transformer': type(step).__name__,
                               'unmapped_output_count': width, 'reason': str(error)})
        return output
    if type(transformer) is FunctionTransformer and transformer.func is None:
        return list(columns)
    if type(transformer) is OneHotEncoder:
        if len(transformer.categories_) != len(columns):
            raise ValueError('Encoder categories do not match input columns')
        infrequent = getattr(transformer, 'infrequent_categories_', [None] * len(columns))
        dropped = transformer.drop_idx_
        result = []
        for index, (column, categories) in enumerate(zip(columns, transformer.categories_)):
            count = len(categories)
            if infrequent[index] is not None:
                count -= len(infrequent[index]) - 1
            if dropped is not None and dropped[index] is not None:
                count -= 1
            if count < 0:
                raise ValueError('Invalid fitted encoder output width')
            result.extend([column] * count)
        if len(result) != len(transformer.get_feature_names_out()):
            raise ValueError('Encoder output width does not match fitted category groups')
        return result
    accepted = (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer,
                PowerTransformer, QuantileTransformer, SimpleImputer, OrdinalEncoder)
    if type(transformer) not in accepted or (isinstance(transformer, SimpleImputer) and transformer.add_indicator):
        raise ValueError('Feature transform has no verified original-column mapping')
    result = transformer.get_feature_names_out().tolist()
    if type(transformer) is SimpleImputer:
        if len(result) != len(set(result)) or any(name not in input_names for name in result):
            raise ValueError('Imputer introduced unmapped names')
        return [columns[input_names.index(name)] for name in result]
    if len(result) != len(columns):
        raise ValueError('One-to-one transformer changed output width')
    return list(columns)


def _importance_details(pipeline, original_columns):
    audit = {'status': 'unavailable', 'mapped_output_count': 0, 'unmapped_output_count': 0, 'issues': []}
    try:
        model = pipeline.steps[-1][1]
        if hasattr(model, 'feature_importances_'):
            values = np.asarray(model.feature_importances_, float)
        elif hasattr(model, 'coef_'):
            coefficients = np.asarray(model.coef_, float)
            values = np.abs(coefficients) if coefficients.ndim == 1 else np.abs(coefficients).mean(axis=0)
        else:
            raise ValueError('Fitted model does not expose feature importances or coefficients')
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Model importance values must be a finite nonnegative vector')
        audit['unmapped_output_count'] = len(values)
        columns = (_mapped_columns(pipeline[:-1], list(original_columns), audit['issues'])
                   if len(pipeline.steps) > 1 else list(original_columns))
        if values.shape != (len(columns),):
            raise ValueError('Model importance width differs from mapped feature width')
        aggregated = {}
        for column, value in zip(columns, values):
            if column is None:
                continue
            if column not in original_columns:
                raise ValueError('Mapped feature is not an original column')
            aggregated[column] = aggregated.get(column, 0.) + float(value)
        mapped = sum(column is not None for column in columns)
        audit.update(mapped_output_count=mapped, unmapped_output_count=len(columns)-mapped,
                     status='complete' if mapped == len(columns) else 'partial' if mapped else 'unavailable')
        importance = [{'feature': column, 'importance': aggregated[column],
                       'source': 'fitted_model_verified_original_column_mapping'}
                      for column in original_columns if column in aggregated]
        return importance, audit
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
        audit['issues'].append({'path': 'pipeline', 'reason': str(error)})
        return [], audit


def _importance(pipeline, original_columns):
    return _importance_details(pipeline, original_columns)[0]


def inspect_proxy_importance(code, fitted, original_columns):
    """Load only; absence of a safe mapping returns an empty list."""
    validate_code(Path(code).read_text())
    return _importance(load_serialized_pipeline(fitted, code), original_columns)


def execute(request):
    started = time.monotonic()
    out = Path(request['output'])
    inspecting = request.get('operation') == 'inspect_importance'
    for name in ('code', 'fitted') if inspecting else ('code', 'fitted', 'predict'):
        if file_hash(request[name]) != request['input_hashes'][name]:
            raise OSError(f'Frozen {name} input changed before prediction')
    validate_code(Path(request['code']).read_text())
    pipeline = load_serialized_pipeline(request['fitted'], request['code'])
    if inspecting:
        importance, mapping = _importance_details(pipeline, request['original_columns'])
        result = {'status': 'success', 'fit_calls': 0,
                  'importance': importance, 'importance_mapping': mapping,
                  'worker_wall_seconds': time.monotonic()-started}
        atomic_json(out/'importance.json', result)
        return result
    task, classes = request['task'], request['classes']
    columns = task['feature_columns']
    schema = task.get('csv_schema')
    kwargs = {'low_memory': False, 'chunksize': request['batch_size']}
    chunks = read_typed(request['predict'], schema, **kwargs) if schema else pd.read_csv(request['predict'], **kwargs)
    predictions, probabilities, prediction_seconds = [], [], 0.
    with threadpool_limits(limits=1):
        for chunk in chunks:
            x = normalize_numeric_infinities(chunk[columns])
            tick = time.monotonic()
            predicted = np.asarray(pipeline.predict(x))
            if predicted.shape != (len(x),):
                raise ValueError('Prediction must preserve one-dimensional input row order')
            if task['problem_type'] == 'classification':
                if not np.isin(predicted, classes).all():
                    raise ValueError('Unknown predicted class')
                probabilities.append(align_probabilities(pipeline.predict_proba(x), pipeline.classes_, classes))
            elif not np.isfinite(predicted.astype(float)).all():
                raise ValueError('Nonfinite regression prediction')
            predictions.append(predicted)
            prediction_seconds += time.monotonic()-tick
    if not predictions or sum(len(part) for part in predictions) == 0:
        raise ValueError('Prediction input has no rows')
    predictions = np.concatenate(predictions)
    temporary = out/'predictions.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, predictions=predictions.astype(str) if task['problem_type'] == 'classification' else predictions,
                            probabilities=np.concatenate(probabilities) if probabilities else np.empty((0, 0)),
                            row_positions=np.arange(len(predictions)))
    temporary.replace(out/'predictions.npz')
    importance, mapping = _importance_details(pipeline, columns)
    report = {'status': 'success', 'fit_calls': 0, 'prediction_rows': len(predictions),
              'prediction_seconds': prediction_seconds, 'worker_wall_seconds': time.monotonic()-started,
              'prediction_source': 'serialized_fitted_pipeline', 'classes': classes,
              'original_feature_importance': importance, 'importance_mapping': mapping}
    atomic_json(out/'prediction_report.json', report)
    return report


if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text())
    try:
        execute(request)
    except Exception as error:
        atomic_json(Path(request['output'])/'failure.json',
                    {'error_type': type(error).__name__, 'error': f'{type(error).__name__}: {error}'})
        traceback.print_exc()
        sys.exit(1)
