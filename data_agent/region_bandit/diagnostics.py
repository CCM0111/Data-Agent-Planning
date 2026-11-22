"""Training-scale losses and paired regional diagnostics; no fitting or API calls."""
from itertools import combinations

import numpy as np

PAIRS = tuple(combinations(range(5), 2))


def problem_type(task):
    value = task.get('problem_type', task.get('task_type', task.get('type')))
    if value not in ('classification', 'regression'):
        raise ValueError('Regional diagnostics require classification or regression')
    return value


def common_losses(task, y, predictions, classes=None, scale=None, *, method_count=5):
    """Return bounded per-row losses for the declared method bank.

    With ``scale=None``, use only the diagnostic population, which excludes
    the parent coreset for a node refresh. Subsequent diagnostic calls on
    external predictions reuse the fitted scale.
    Classification probabilities are aligned by each prediction's optional
    ``classes`` field; missing probabilities use the hard prediction's one-hot.
    """
    y = np.asarray(y)
    if method_count < 2 or y.ndim != 1 or len(y) == 0 or len(predictions) != method_count:
        raise ValueError('Predictions must match the declared method bank and nonempty target')
    if problem_type(task) == 'regression':
        target = np.asarray(y, dtype=float)
        errors = []
        for result in predictions:
            values = np.asarray(result['predictions'], dtype=float)
            if values.shape != y.shape or not np.isfinite(values).all():
                raise ValueError('Regression prediction shape or finite-value mismatch')
            errors.append(np.abs(target - values))
        errors = np.asarray(errors)
        if not np.isfinite(errors).all():
            raise ValueError('Regression targets and errors must be finite')
        if scale is None:
            scale = float(np.quantile(errors, .9))
            if scale == 0:
                scale = 1.0
        if isinstance(scale, bool) or not np.isfinite(scale) or scale <= 0:
            raise ValueError('Frozen loss scale must be finite and positive')
        return np.minimum(errors / float(scale), 1.), float(scale)
    if classes is None:
        classes = task.get('classes')
    if classes is None:
        classes = predictions[0].get('classes')
    classes = list(classes if classes is not None else np.unique(y))
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError('Classification classes must contain unique labels')
    lookup = {label: index for index, label in enumerate(classes)}
    try:
        encoded = np.asarray([lookup[label] for label in y], dtype=int)
    except KeyError as exc:
        raise ValueError('Unknown classification target label') from exc
    truth = np.eye(len(classes))[encoded]
    losses = []
    for result in predictions:
        values = result.get('probabilities')
        if values is None:
            hard = np.asarray(result['predictions'])
            if hard.shape != y.shape:
                raise ValueError('Classification prediction row count mismatch')
            try:
                values = np.eye(len(classes))[[lookup[label] for label in hard]]
            except KeyError as exc:
                raise ValueError('Unknown predicted class') from exc
        else:
            values = np.asarray(values, dtype=float)
            declared_classes = result.get('classes')
            source_classes = list(declared_classes if declared_classes is not None else classes)
            if values.shape != (len(y), len(source_classes)) or len(set(source_classes)) != len(source_classes):
                raise ValueError('Classification probability shape or class mismatch')
            if not set(source_classes).issubset(lookup):
                raise ValueError('Unknown probability class')
            aligned = np.zeros((len(y), len(classes)))
            for column, label in enumerate(source_classes):
                aligned[:, lookup[label]] = values[:, column]
            values = aligned
        if (not np.isfinite(values).all() or np.any(values < -1e-12)
                or np.any(values > 1 + 1e-12)
                or not np.allclose(values.sum(axis=1), 1, atol=1e-8, rtol=1e-8)):
            raise ValueError('Probabilities must be finite, bounded, and sum to one')
        losses.append(np.linalg.norm(np.clip(values, 0, 1) - truth, axis=1) / np.sqrt(2))
    return np.asarray(losses), 1.0


def regional_contrast(losses, mask):
    """Contrasts for every pair in the supplied method bank."""
    losses = np.asarray(losses, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if losses.ndim != 2 or losses.shape[0] < 2 or losses.shape[1] != len(mask) or not np.isfinite(losses).all():
        raise ValueError('Expected finite aligned losses for at least two methods')
    pairs = tuple(combinations(range(len(losses)), 2))
    inside, outside = int(mask.sum()), int((~mask).sum())
    if not inside or not outside:
        return {'score': 0., 'pair_contrasts': [0.] * len(pairs),
                'inside_count': inside, 'outside_count': outside,
                'fraction': float(mask.mean()) if len(mask) else 0.,
                'inside_losses': None, 'outside_losses': None}
    inner, outer = losses[:, mask].mean(axis=1), losses[:, ~mask].mean(axis=1)
    contrasts = np.array([(inner[a] - inner[b]) - (outer[a] - outer[b]) for a, b in pairs])
    fraction = float(mask.mean())
    return {'score': float(fraction * (1 - fraction) * np.mean(contrasts ** 2)),
            'pair_contrasts': contrasts.tolist(), 'inside_count': inside, 'outside_count': outside,
            'fraction': fraction, 'inside_losses': inner.tolist(), 'outside_losses': outer.tolist()}


def paired_region_bootstrap(core_losses, full_losses, mask, *, repeats=300, seed=2026):
    """Stratify inside/outside; share rows across observed core/full methods."""
    core_losses, full_losses = np.asarray(core_losses), np.asarray(full_losses)
    mask = np.asarray(mask, dtype=bool)
    if (core_losses.ndim != 2 or core_losses.shape[0] not in (3, 5)
            or core_losses.shape[1] != len(mask) or full_losses.shape != core_losses.shape):
        raise ValueError('Paired bootstrap prediction dimensions do not agree')
    count = len(core_losses)
    pairs = tuple(combinations(range(count), 2))
    if isinstance(repeats, bool) or not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError('bootstrap_repeats must be a positive integer')
    inner, outer = np.flatnonzero(mask), np.flatnonzero(~mask)
    if not len(inner) or not len(outer):
        raise ValueError('Both regional partitions must be nonempty')
    all_losses = np.concatenate([core_losses, full_losses], axis=0)
    rng = np.random.default_rng(seed)
    values = np.empty((repeats, 2, len(pairs)))
    for b in range(repeats):
        means = (all_losses[:, rng.choice(inner, len(inner), replace=True)].mean(axis=1)
                 - all_losses[:, rng.choice(outer, len(outer), replace=True)].mean(axis=1))
        for j, (a, c) in enumerate(pairs):
            values[b, 0, j] = means[a] - means[c]
            values[b, 1, j] = means[count + a] - means[count + c]
    return {'core_ci': np.quantile(values[:, 0], [.025, .975], axis=0).T.tolist(),
            'full_ci': np.quantile(values[:, 1], [.025, .975], axis=0).T.tolist(),
            'core_minus_full_ci': np.quantile(values[:, 0] - values[:, 1], [.025, .975], axis=0).T.tolist(),
            'repeats': int(repeats), 'seed': int(seed), 'confidence': .95,
            'scope': 'Fixed fitted models; validation-row variation, not simultaneous regional inference'}
