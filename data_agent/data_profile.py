"""Compact, deterministic metadata computed from training data only."""

import math
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype


def _finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _target_summary(target):
    """Describe finite regression targets; retain counts of excluded values."""
    values = pd.to_numeric(target, errors="raise").to_numpy(dtype=float, na_value=np.nan)
    finite = values[np.isfinite(values)]
    summary = {
        "finite_rows": int(len(finite)),
        "missing_rows": int(np.isnan(values).sum()),
        "nonfinite_rows": int(np.isinf(values).sum()),
        "mean": None, "std": None, "min": None, "max": None,
    }
    if len(finite):
        # Scaling avoids overflowing the mean/variance of otherwise finite data.
        scale = float(np.max(np.abs(finite)))
        scaled = finite / scale if scale else finite
        summary.update(
            mean=_finite_number(np.mean(scaled) * scale),
            std=_finite_number(np.std(scaled, ddof=0) * scale),
            min=float(np.min(finite)), max=float(np.max(finite)),
        )
    return summary


def training_profile(frame, target, problem_type):
    """Summarize only the supplied training frame, without per-column or row data.

    Feature missingness excludes the target and means pandas ``isna``; infinities
    are not missing cells. Classification labels are interpreted from the task's
    problem type, never inferred from their dtype. Regression statistics use
    finite targets only and population standard deviation (ddof=0).
    """
    if problem_type not in {"classification", "regression"}:
        raise ValueError("problem_type must be classification or regression")
    labels = frame[target]
    rows = len(frame)
    columns = numeric = categorical = missing_cells = 0
    row_missing = np.zeros(rows, dtype=bool)
    # A column mask and one row mask avoid copying the feature table or allocating
    # its full rows-by-features missingness matrix.
    for name, feature in frame.items():
        if name == target:
            continue
        columns += 1
        dtype = feature.dtype
        if (is_bool_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype)
                or is_object_dtype(dtype) or is_string_dtype(dtype)):
            categorical += 1
        elif is_numeric_dtype(dtype):
            numeric += 1
        missing = feature.isna().to_numpy(dtype=bool, copy=False)
        missing_cells += int(missing.sum())
        row_missing |= missing
    profile = {
        "schema_version": 1,
        "scope": "train",
        "train_rows": int(rows),
        "feature_count": int(columns),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "other_features": int(columns - numeric - categorical),
        "missing_cell_fraction": float(missing_cells / (rows * columns))
        if rows and columns else None,
        "rows_with_missing_fraction": float(row_missing.sum() / rows) if rows else None,
    }
    if problem_type == "classification":
        counts = labels.value_counts(dropna=True)
        # Categorical dtype can include unused categories with zero observations.
        counts = counts[counts > 0]
        profile.update(
            classes=int(len(counts)),
            class_count_max_min_ratio=float(counts.max() / counts.min()) if len(counts) > 1 else None,
        )
    else:
        profile["target_summary"] = _target_summary(labels)
    return profile


def _profile_fingerprint(profile):
    profile['fingerprint'] = hashlib.sha256(json.dumps(
        profile, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return profile


def feature_metadata_for_task(task, feature_columns, *, task_path=None):
    """Read explicit feature declarations and one verified legacy preparation record.

    Only the declared preparation audit is read, never any data split. Legacy
    conversion recognition requires the recorded audit hash and the exact recipe
    hash whose conversion has been checked; description/column-name heuristics
    are deliberately not used. No preparation is executed.
    """
    declared = task.get('feature_metadata')
    metadata = dict(declared) if isinstance(declared, dict) else {}
    path = task.get('preparation_audit')
    expected = task.get('preparation_audit_sha256')
    if path and expected:
        try:
            audit_path = Path(path)
            if not audit_path.is_absolute():
                declared_task_path = task_path or task.get('task_path')
                if not declared_task_path:
                    raise ValueError('Relative preparation audit requires the task file path')
                audit_path = Path(declared_task_path).resolve().parent / audit_path
            raw = audit_path.read_bytes() if audit_path.stat().st_size <= 1_000_000 else b''
            if hashlib.sha256(raw).hexdigest() == expected:
                audit = json.loads(raw)
                if not isinstance(audit, dict):
                    raise ValueError('Preparation audit must be an object')
                audit_metadata = audit.get('feature_metadata')
                if isinstance(audit_metadata, dict):
                    metadata = {**audit_metadata, **metadata}
                action = 'Age days converted to years; impossible anthropometry/BP set missing by fixed bounds.'
                if ('age_years' in feature_columns and 'age_years' not in metadata
                        and audit.get('dataset') == 'cardiovascular'
                        and audit.get('candidate_id') == 6
                        and audit.get('recipe_sha256') == 'b82bd139150997cbd4da7b26e3cbc08b1e8835b9af204fa477314eff2c42c43c'
                        and action in audit.get('actions', [])
                        and 'age_years' in audit.get('columns', [])
                        and 'age' not in audit.get('columns', [])):
                    metadata['age_years'] = {
                        'unit': 'years', 'preprocessing': 'age_days / 365.25; already applied; do not repeat',
                        'source': f'{audit_path}#actions; sha256={expected}',
                    }
        except (OSError, ValueError, TypeError):
            # An unverifiable optional audit supplies no declarations.
            pass
    allowed = {'unit', 'group', 'preprocessing', 'source'}
    return dict(list({str(name): {key: str(value)[:500] for key, value in declaration.items()
                        if key in allowed and isinstance(value, (str, int, float))}
            for name, declaration in metadata.items()
            if name in feature_columns and isinstance(declaration, dict)}.items())[:128])


def _feature_sample(frame, feature_columns, max_rows, max_columns, seed):
    if max_rows < 1 or max_columns < 1:
        raise ValueError('Feature profile limits must be positive')
    columns = list(frame.columns if feature_columns is None else feature_columns)
    if len(columns) != len(set(columns)):
        raise ValueError('Feature profile columns must be unique')
    if not set(columns) <= set(frame.columns):
        raise ValueError('Feature profile received unavailable columns')
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(frame), min(len(frame), max_rows), replace=False))
    selected = columns
    if len(columns) > max_columns:
        selected = [columns[i] for i in sorted(rng.choice(len(columns), max_columns, replace=False))]
    # Slice one column at a time: avoid an intermediate sample containing every
    # feature of a very wide source table before applying the column cap.
    sample = pd.DataFrame({name:frame[name].iloc[positions].reset_index(drop=True) for name in selected})
    if not selected:
        sample = pd.DataFrame(index=np.arange(len(positions)))
    sampling = {'source_rows': int(len(frame)), 'sample_rows': int(len(sample)),
                'source_features': len(columns), 'inspected_features': len(selected),
                'row_limit': max_rows, 'column_limit': max_columns, 'seed': seed,
                'row_sampling': 'all' if len(sample) == len(frame) else 'fixed-seed uniform without replacement',
                'column_sampling': 'all' if len(selected) == len(columns) else 'fixed-seed uniform subset',
                'position_fingerprint': hashlib.sha256(positions.tobytes()).hexdigest()}
    return sample, sampling


def _column_structure(values, categorical=False):
    count = len(values)
    observed = values.dropna()
    if isinstance(observed.dtype, pd.CategoricalDtype):
        observed = observed.cat.remove_unused_categories()
    distinct = int(observed.nunique())
    stats = {'missing_fraction': float(values.isna().mean()) if count else None,
             'distinct': distinct,
             'distinct_fraction': float(distinct / len(observed)) if len(observed) else None}
    numeric = (is_numeric_dtype(values.dtype) and not is_bool_dtype(values.dtype)
               and not categorical)
    if numeric:
        numbers = values.to_numpy(dtype=float, na_value=np.nan)
        finite = numbers[np.isfinite(numbers)]
        quantiles = [None] * 5
        skew = None
        if len(finite):
            scale = np.max(np.abs(finite)) or 1.
            scaled = finite / scale
            quantiles = [_finite_number(v * scale) for v in np.quantile(scaled, [.05, .25, .5, .75, .95])]
            centered = scaled - np.mean(scaled)
            std = np.std(scaled)
            skew = _finite_number(np.mean((centered / std) ** 3)) if std else 0.
        stats.update(kind='numeric', quantiles=dict(zip(('p05', 'p25', 'p50', 'p75', 'p95'), quantiles)),
                     zero_fraction=float(np.mean(finite == 0)) if len(finite) else None,
                     skewness=skew, finite_rows=int(len(finite)),
                     nonfinite_fraction=float(np.isinf(numbers).mean()) if count else None)
    else:
        counts = observed.value_counts()
        counts = counts[counts > 0]
        rare_limit = max(1, int(len(observed) * .01))
        stats.update(kind='categorical',
                     head_frequencies=[{'value': str(value)[:80], 'fraction': float(n / len(observed))}
                                       for value, n in counts.iloc[:3].items()],
                     rare_row_fraction=float(counts[counts <= rare_limit].sum() / len(observed)) if len(observed) else None,
                     rare_count_threshold=rare_limit)
    return stats


def feature_structure_profile(frame, *, feature_columns=None, categorical_columns=(), scope,
                              metadata=None, max_rows=2048, max_columns=128,
                              max_details=12, max_pairs=6, seed=0):
    """Bounded descriptive feature evidence from the explicitly supplied frame.

    Callers must pass feature-only data or feature_columns that exclude the
    target. No labels, predictive errors, semantic units or physical invariances
    are inferred. Sample duplicates are reported only as sample observations.
    """
    if max_details < 0 or max_pairs < 0:
        raise ValueError('Feature profile output limits must be nonnegative')
    sample, sampling = _feature_sample(frame, feature_columns, max_rows, max_columns, seed)
    categories = set(categorical_columns)
    statistics = {str(name): _column_structure(values, name in categories)
                  for name, values in sample.items()}
    def salience(item):
        name, stats = item
        value = max(stats.get('missing_fraction') or 0., stats.get('nonfinite_fraction') or 0.)
        if stats['distinct'] <= 1:
            value += 2.
        if stats['kind'] == 'numeric':
            value += min(2., abs(stats['skewness'] or 0.) / 3) + (stats['zero_fraction'] or 0.)
        else:
            value += stats['rare_row_fraction'] or 0.
            value += (stats['head_frequencies'] or [{'fraction': 0.}])[0]['fraction']
        return -value, name
    ranked = sorted(statistics.items(), key=salience)
    details = [{'feature': name, **stats} for name, stats in ranked[:max_details]]
    # Hash first, then compare actual sample columns: no hash-only duplicate claim.
    groups, hashes = [], {}
    for name in sample:
        key = (str(sample[name].dtype), hashlib.sha256(
            pd.util.hash_pandas_object(sample[name], index=False).to_numpy().tobytes()).hexdigest())
        matches = hashes.setdefault(key, [])
        if matches and sample[matches[0]].equals(sample[name]):
            matches.append(name)
        elif not matches:
            matches.append(name)
    for names in hashes.values():
        if len(names) > 1:
            groups.append({'features': [str(name) for name in names[:8]], 'group_size': len(names),
                           'evidence': 'equal values and dtype on inspected rows only'})
    numeric = [name for name in sample if statistics[str(name)]['kind'] == 'numeric'][:64]
    correlations = []
    if len(numeric) >= 2 and len(sample) >= 3 and max_pairs:
        numbers = sample[numeric].astype(float).replace([np.inf, -np.inf], np.nan)
        scales = numbers.abs().max().replace(0., 1.)
        correlations_frame = numbers.div(scales).corr(min_periods=3)
        for i, left in enumerate(numeric):
            for right in numeric[i + 1:]:
                correlation = correlations_frame.loc[left, right]
                if np.isfinite(correlation) and abs(correlation) >= .9:
                    correlations.append({'features': [str(left), str(right)],
                                         'pearson_r': float(correlation),
                                         'paired_rows': int(numbers[[left, right]].notna().all(axis=1).sum())})
        correlations.sort(key=lambda row: (-abs(row['pearson_r']), row['features']))
    declarations = feature_metadata_for_task({'feature_metadata': metadata or {}},
                                              list(frame.columns if feature_columns is None else feature_columns))
    return _profile_fingerprint({
        'schema_version': 1, 'scope': scope, 'sampling': sampling,
        'feature_details': details, 'omitted_feature_details': len(statistics) - len(details),
        'detail_selection': 'largest missingness, skewness, zero mass, rare-category or constant-column signals; ties by name',
        'high_correlation_pairs': correlations[:max_pairs], 'correlation_feature_limit': 64,
        'duplicate_column_groups': groups[:max_pairs],
        'confirmed_feature_metadata': dict(list(declarations.items())[:max_details]),
        'interpretation': 'Descriptive training-feature evidence only. Sample cardinalities depend on sample size; '
                          'correlation does not establish causality, target relevance or invariance. Unreported features were not certified normal.',
    })


def feature_distribution_difference(reference_frame, candidate_frame, *, feature_columns=None,
                                    categorical_columns=(), reference_scope='P + Q',
                                    candidate_scope='coreset subset of Q', max_rows=2048,
                                    max_columns=128, max_details=10, seed=0):
    """Largest observed feature-distribution changes; not a generalization test."""
    if max_details < 0:
        raise ValueError('Feature difference output limit must be nonnegative')
    columns = list(reference_frame.columns if feature_columns is None else feature_columns)
    reference, reference_sampling = _feature_sample(reference_frame, columns, max_rows, max_columns, seed)
    # Compare the same feature subset even when the frames have different sizes.
    candidate, candidate_sampling = _feature_sample(candidate_frame, list(reference.columns), max_rows, max_columns, seed)
    shifts = []
    for name in reference:
        left = _column_structure(reference[name], name in categorical_columns)
        right = _column_structure(candidate[name], name in categorical_columns)
        row = {'feature': str(name), 'kind': left['kind'],
               'reference_distinct': left['distinct'], 'candidate_distinct': right['distinct']}
        score = 0.
        for key in ('missing_fraction', 'distinct_fraction', 'zero_fraction'):
            if left.get(key) is not None and right.get(key) is not None:
                delta = right[key] - left[key]
                row[key + '_delta'] = delta
                score = max(score, abs(delta))
        if left['kind'] == right['kind'] == 'numeric':
            a, b = left['quantiles'], right['quantiles']
            if all(value is not None for value in [*a.values(), *b.values()]):
                # Scale in normalized space to avoid overflow with extreme floats.
                magnitude = max(max(abs(value) for value in a.values()), max(abs(value) for value in b.values()), 1.)
                denominator = max(abs(a['p75'] / magnitude - a['p25'] / magnitude),
                                  abs(a['p95'] / magnitude - a['p05'] / magnitude), 1. / magnitude)
                deltas = {key: _finite_number((b[key] / magnitude - a[key] / magnitude) / denominator)
                          for key in a}
                row['quantile_shift_reference_scale'] = deltas
                row['quantile_scale'] = 'max(reference p95-p05, reference IQR, 1); descriptive only'
                finite_shifts = [abs(value) for value in deltas.values() if value is not None]
                if len(finite_shifts) != len(deltas):
                    row['quantile_shift_status'] = 'partial' if finite_shifts else 'unavailable'
                    row['quantile_shift_reason'] = 'Some normalized quantile differences exceed finite float range; null shifts are not zero'
                # Legal finite features may still have an unrepresentable
                # difference, e.g. opposite signs near the float range limit.
                # Optional EDA must report that limitation without aborting.
                score = max(score, max(finite_shifts, default=0.))
        elif left['kind'] == right['kind'] == 'categorical':
            # Use one shared observed support, so an unchanged cardinality does
            # not conceal a large shift in category frequencies. Bounded row
            # sampling bounds this union to at most 2 * max_rows categories.
            reference_counts = reference[name].dropna().astype(object).value_counts(normalize=True)
            candidate_counts = candidate[name].dropna().astype(object).value_counts(normalize=True)
            support = reference_counts.index.union(candidate_counts.index, sort=False)
            if len(reference_counts) and len(candidate_counts):
                frequency_delta = candidate_counts.reindex(support, fill_value=0.) - reference_counts.reindex(support, fill_value=0.)
                total_variation = float(frequency_delta.abs().sum() / 2.)
                row['sample_category_total_variation'] = total_variation
                row['largest_category_frequency_changes'] = [
                    {'value':str(value)[:80], 'frequency_delta':float(delta)}
                    for value, delta in sorted(frequency_delta.items(), key=lambda item: -abs(item[1]))[:3]
                    if abs(delta) > 1e-12]
                score = max(score, total_variation)
        if score > 1e-12 or row.get('quantile_shift_status'):
            shifts.append((score, row))
    shifts.sort(key=lambda item: (-item[0], item[1]['feature']))
    return _profile_fingerprint({
        'schema_version': 1, 'reference_scope': reference_scope, 'candidate_scope': candidate_scope,
        'reference_sampling': reference_sampling, 'candidate_sampling': candidate_sampling,
        'largest_changes': [row for _, row in shifts[:max_details]],
        'omitted_changed_features': max(0, len(shifts) - max_details),
        'interpretation': 'Observed feature-distribution differences on supplied training frames only, not generalization error '
                          'or evidence that a model improves. Cardinality and rare-category estimates depend on sample size.',
    })
