"""Canonical task metrics shared by evaluation and paired stage-one analysis.

Probability columns must follow ``classes``; label encoding here preserves that
order even when it is not alphabetical. ``metric_params`` is deliberately
restricted to scalar, unweighted metrics so bootstrap resampling cannot silently
change the estimand. F1 keeps sklearn's default union-of-observed-labels behavior
unless ``labels`` is explicitly supplied.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)


SUPPORTED_METRICS = {
    'classification': {'accuracy', 'f1_macro', 'f1', 'auc', 'roc_auc', 'log_loss'},
    'regression': {'mae', 'mse', 'rmse', 'r2'},
}
_MAXIMIZE = {'accuracy', 'f1_macro', 'f1', 'auc', 'roc_auc', 'r2'}
_ADDITIVE = {'accuracy', 'mae', 'mse', 'log_loss'}


class UndefinedMetricError(ValueError):
    """Valid evaluation inputs cannot define a finite task primary metric."""


def _metric(task):
    metric = task.get('metric')
    if metric not in set().union(*SUPPORTED_METRICS.values()):
        raise ValueError('Unsupported primary metric: ' + str(metric))
    problem = task.get('problem_type')
    if problem is not None and metric not in SUPPORTED_METRICS.get(problem, set()):
        raise ValueError(f'Metric {metric} does not match problem_type {problem}')
    return metric


def metric_direction(task):
    """Return the task metric's fixed direction, independent of observed scores."""
    return 'maximize' if _metric(task) in _MAXIMIZE else 'minimize'


def comparison_backend(task):
    return 'sample_loss' if _metric(task) in _ADDITIVE else 'metric_recompute'


def requires_stratified_bootstrap(task):
    """AUC requires every original class to remain present in each resample."""
    return _metric(task) in {'auc', 'roc_auc'}


def _parameters(task, metric):
    params = dict(task.get('metric_params') or {})
    for alias, canonical in [('positive_label', 'pos_label'), ('class_labels', 'labels')]:
        if alias in params:
            if canonical in params:
                raise ValueError(f'Specify only one of {alias} and {canonical}')
            params[canonical] = params.pop(alias)
    allowed = {
        'f1': {'average', 'pos_label', 'labels', 'zero_division'},
        'f1_macro': {'average', 'labels', 'zero_division'},
        'auc': {'average', 'pos_label', 'labels', 'multi_class', 'max_fpr'},
        'roc_auc': {'average', 'pos_label', 'labels', 'multi_class', 'max_fpr'},
        'log_loss': {'labels'}, 'r2': {'force_finite'},
    }.get(metric, set())
    unexpected = set(params) - allowed
    if unexpected:
        raise ValueError(f'Unsupported {metric} metric_params: {sorted(unexpected)}')
    if metric in {'f1', 'f1_macro'}:
        params.setdefault('average', 'macro' if metric == 'f1_macro' else 'binary')
        if metric == 'f1_macro' and params['average'] != 'macro':
            raise ValueError('f1_macro requires average="macro"')
        if params['average'] not in {'binary', 'macro', 'weighted', 'micro'}:
            raise ValueError('F1 must use a scalar average')
    return params


def _rows(y, predictions):
    actual, predicted = np.asarray(y), np.asarray(predictions)
    if actual.ndim != 1 or predicted.shape != actual.shape or not len(actual):
        raise ValueError('Labels and predictions must be matching nonempty one-dimensional arrays')
    return actual, predicted


def _label_parameters(params, labels):
    """Match CSV text labels without parsing or merging distinct category tokens."""
    if len(labels) and all(isinstance(label, str) for label in labels):
        if 'pos_label' in params:
            params['pos_label'] = str(params['pos_label'])
        if params.get('labels') is not None:
            params['labels'] = [str(label) for label in params['labels']]


def _probabilities(y, probabilities, classes, params):
    _label_parameters(params, classes if classes is not None else y)
    explicit = params.pop('labels', None)
    if classes is None:
        classes = explicit
    elif explicit is not None and list(explicit) != list(classes):
        raise ValueError('metric_params.labels must match the saved probability class order')
    if classes is None:
        raise ValueError('Probability metrics require an explicit saved class order')
    order = list(classes)
    try:
        indices = {label: i for i, label in enumerate(order)}
    except TypeError as error:
        raise ValueError('Probability classes must be scalar labels') from error
    if len(indices) != len(order) or any(label is None or label != label for label in order):
        raise ValueError('Probability classes must be distinct and nonmissing')
    if len(order) < 2:
        raise UndefinedMetricError('Probability metrics require at least two classes')
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(y), len(order)):
        raise ValueError('Probability shape does not match validation rows and saved class order')
    if (not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
            or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError('Invalid class probabilities')
    try:
        encoded = np.asarray([indices[label] for label in y], dtype=int)
    except (KeyError, TypeError) as error:
        raise ValueError('Validation label is absent from the saved probability class order') from error
    return encoded, values, order


def sample_losses(task, y, predictions, probabilities=None, classes=None):
    """Per-row loss for additive metrics, or None for metric-level recomputation.

    Accuracy's loss mean is ``1 - accuracy``; its pairwise differences equal
    those of ``-accuracy``. The analysis layer must preserve that constant when
    storing absolute method-level comparison quantities.
    """
    metric = _metric(task)
    params = _parameters(task, metric)
    if metric not in _ADDITIVE:
        return None
    actual, predicted = _rows(y, predictions)
    if metric == 'accuracy':
        # sklearn also validates target types (e.g. disallow continuous targets).
        accuracy_score(actual, predicted)
        result = (actual != predicted).astype(float)
    elif metric in {'mae', 'mse'}:
        actual, predicted = actual.astype(float), predicted.astype(float)
        if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise ValueError('Regression labels and predictions must be finite')
        with np.errstate(over='ignore', invalid='ignore'):
            residual = actual - predicted
            result = np.abs(residual) if metric == 'mae' else residual ** 2
    else:
        encoded, values, _ = _probabilities(actual, probabilities, classes, params)
        # Matches sklearn log_loss's float64 clipping, with no prediction pooling.
        clipped = np.clip(values, np.finfo(values.dtype).eps, 1 - np.finfo(values.dtype).eps)
        result = -np.log(clipped[np.arange(len(actual)), encoded])
    if not np.isfinite(result).all():
        raise UndefinedMetricError(f'{metric} produced nonfinite per-sample losses')
    return result


def primary_score(task, y, predictions, probabilities=None, classes=None):
    """Compute exactly one scalar primary metric for online/offline/bootstrap use.

    Binary AUC defaults to the greatest class label as positive, matching
    sklearn, or accepts an explicit ``metric_params.pos_label``. Multiclass AUC
    defaults to one-vs-rest macro and accepts explicit ``multi_class``/``average``.
    R2 preserves sklearn's existing ``force_finite=True`` default; fewer than two
    observations remain undefined. Invalid inputs raise ValueError, while an
    undefined/nonfinite metric raises its subclass UndefinedMetricError.
    """
    metric = _metric(task)
    params = _parameters(task, metric)
    actual, predicted = _rows(y, predictions)
    if metric == 'accuracy':
        result = accuracy_score(actual, predicted)
    elif metric in {'f1', 'f1_macro'}:
        if params['average'] == 'binary':
            params.setdefault('pos_label', 1)
        _label_parameters(params, actual)
        result = f1_score(actual, predicted, **params)
    elif metric in {'mae', 'mse', 'rmse', 'r2'}:
        actual, predicted = actual.astype(float), predicted.astype(float)
        if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise ValueError('Regression labels and predictions must be finite')
        if metric == 'r2':
            if len(actual) < 2:
                raise UndefinedMetricError('R2 is undefined with fewer than two observations')
            result = r2_score(actual, predicted, **params)
        elif metric == 'mae':
            result = mean_absolute_error(actual, predicted)
        else:
            result = mean_squared_error(actual, predicted)
            if metric == 'rmse':
                result = np.sqrt(result)
    elif metric == 'log_loss':
        encoded, values, order = _probabilities(actual, probabilities, classes, params)
        result = log_loss(encoded, values, labels=np.arange(len(order)))
    else:
        encoded, values, order = _probabilities(actual, probabilities, classes, params)
        if len(np.unique(encoded)) != len(order):
            raise UndefinedMetricError('AUC is undefined because validation is missing required classes')
        if len(order) == 2:
            try:
                positive = params.pop('pos_label') if 'pos_label' in params else sorted(order)[-1]
                column = order.index(positive)
            except (TypeError, ValueError) as error:
                raise ValueError('Binary AUC requires a valid, unambiguous positive class') from error
            # average/multi_class do not affect a binary AUC; validate rather than
            # silently accepting options intended for a different task.
            average = params.pop('average', 'macro')
            multi_class = params.pop('multi_class', 'raise')
            if average not in {'macro', 'weighted', 'micro', None} or multi_class not in {'raise', 'ovr', 'ovo'}:
                raise ValueError('Invalid binary AUC averaging or multiclass setting')
            result = roc_auc_score(encoded == column, values[:, column], **params)
        else:
            if 'pos_label' in params:
                raise ValueError('pos_label applies only to binary AUC')
            params.setdefault('average', 'macro')
            params.setdefault('multi_class', 'ovr')
            if params['average'] not in {'macro', 'weighted', 'micro'}:
                raise ValueError('Multiclass AUC must use a scalar average')
            result = roc_auc_score(encoded, values, labels=np.arange(len(order)), **params)
    result = float(result)
    if not np.isfinite(result):
        raise UndefinedMetricError(f'{metric} is undefined or nonfinite on this validation sample')
    return result
