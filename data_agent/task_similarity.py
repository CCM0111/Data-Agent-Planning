"""Local MiniLM semantics plus comparable, training-only structural evidence.

Thresholds are a configurable initial policy, not calibrated transfer probabilities.
No experience text, performance, validation rows, or test artifacts enter ranking.
"""
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = 'c9745ed1d9f207416be6d2e6f8de32d1f16199bf'
# A scale denotes a substantive difference, rather than an arbitrary dataset maximum.
SCALES = {'log_rows': math.log(10), 'log_features': math.log(10),
          'numeric_fraction': 1., 'categorical_fraction': 1., 'missing_fraction': 1.,
          'log_classes': math.log(10), 'log_imbalance': math.log(10),
          'log_cardinality': math.log(10), 'log_abs_skew': math.log(10)}


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


@lru_cache(maxsize=1)
def semantic_catalog():
    path = ROOT / 'datasets/tabular_tasks/task_semantics.json'
    return json.loads(path.read_text()) if path.exists() else {}


def task_semantics(task):
    """Metadata overlay leaves frozen datasets and their manifest hashes untouched."""
    known = semantic_catalog().get(task.get('name'), {})
    return {key: task.get(key) or known.get(key) for key in
            ('retrieval_description', 'dataset_id', 'task_family')}


def _names(summary, key):
    value = summary.get(key, {})
    if isinstance(value, dict):
        return value.get('values', [])
    return value if isinstance(value, list) else []


def semantic_text(task, metadata):
    facts = task_semantics(task)
    description = facts['retrieval_description']
    if not description:
        raw = str(task.get('description') or '')
        # Legacy packaging instructions describe this framework, not the task.
        sentences = re.split(r'(?<=[.!?])\s+', raw)
        boilerplate = ('only input packaging', 'all original feature', 'analyze them in eda',
                       'this is an independent', 'original metric:', 'use the provided frozen',
                       'do not repartition', 'test labels are', 'no final refit')
        description = ' '.join(s for s in sentences if not s.lower().startswith(boilerplate))
    name = re.sub(r'^(dsbench|datascibench|imcts|mlebench)[_ :]+', '', str(task.get('name') or ''))
    summary = metadata.get('summary', {})
    features = sorted(set(_names(summary, 'numeric_features') + _names(summary, 'categorical_features')
                          + _names(summary, 'numeric') + _names(summary, 'categorical')))
    if not description.strip() and not features:
        return ""  # A task ID plus a generic target is not a semantic description.
    # A bounded descriptor fits the model's 256 word-piece limit in ordinary tasks;
    # the encoder applies its documented token limit as a final bound.
    return (f"{description[:650] or name[:120]}. "
            f"Predict {str(task.get('target') or '')[:80]}. {task.get('problem_type', '')}. "
            + ('Features: ' + ', '.join(features[:12]) if features else ''))


def structure_from_report(report):
    """Aggregate the full recorded EDA before bounded per-column views truncate it."""
    if not isinstance(report, dict):
        return {}
    result = {}
    cards = report.get('categorical_cardinalities', {})
    if isinstance(cards, dict):
        values = [math.log1p(v) for v in cards.values() if number(v) and v >= 0]
        if values:
            result['log_cardinality'] = statistics.median(values)
    summaries = report.get('numeric_summary', {})
    if isinstance(summaries, dict):
        values = [math.log1p(abs(s['skew'])) for s in summaries.values()
                  if isinstance(s, dict) and number(s.get('skew'))]
        if values:
            result['log_abs_skew'] = statistics.median(values)
    return result


def structural_features(task, metadata):
    if metadata.get('scope') != 'train':
        return {}
    summary = metadata.get('summary', {})
    profile = task.get('data_profile', {})
    if profile.get('scope') != 'train':
        profile = {}
    values = {}
    rows = profile.get('train_rows', summary.get('n_rows', summary.get('rows')))
    features = profile.get('feature_count', summary.get('n_features', summary.get('feature_count')))
    for key, val in (('log_rows', rows), ('log_features', features)):
        if number(val) and val > 0:
            values[key] = math.log1p(val)
    for kind in ('numeric', 'categorical'):
        count = profile.get(kind + '_features')
        if count is None:
            raw = summary.get(kind + '_features', summary.get(kind))
            if isinstance(raw, dict) and isinstance(raw.get('values'), list):
                count = len(raw['values']) + raw.get('omitted_entries', 0)
            elif isinstance(raw, list):
                count = len(raw)
        if number(features) and features > 0 and number(count) and 0 <= count <= features:
            values[kind + '_fraction'] = count / features
    missing = profile.get('missing_cell_fraction')
    if missing is None and number(summary.get('total_missing_cells')):
        if number(rows) and rows > 0 and number(features) and features > 0:
            missing = summary['total_missing_cells'] / (rows * features)
    if number(missing) and 0 <= missing <= 1:
        values['missing_fraction'] = missing
    if task.get('problem_type') == 'classification':
        balance = metadata.get('class_balance', {})
        classes = profile.get('classes', balance.get('observed_classes', summary.get('target_n_classes')))
        ratio = profile.get('class_count_max_min_ratio', balance.get('max_min_ratio'))
        if number(classes) and classes >= 2:
            values['log_classes'] = math.log1p(classes)
        if number(ratio) and ratio >= 1:
            values['log_imbalance'] = math.log(ratio)
    # Never compute a global statistic from the first few displayed columns.
    values.update({k: v for k, v in metadata.get('structure', {}).items()
                   if k in ('log_cardinality', 'log_abs_skew') and number(v) and v >= 0})
    return values


def eda_similarity(left, right):
    shared = sorted(set(left) & set(right) & set(SCALES))
    if not shared:
        return None, shared
    # Bounded, scaled L1 distance; absent measurements are not zeros or matches.
    distance = sum(min(abs(left[k] - right[k]) / SCALES[k], 1.) for k in shared) / len(shared)
    return 1. - distance, shared


@lru_cache(maxsize=2)
def _model(name, revision, cache_folder, local_files_only):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, revision=revision, cache_folder=cache_folder,
                               device='cpu', local_files_only=local_files_only, trust_remote_code=False)


class MiniLMEncoder:
    def __init__(self, *, model_name=MODEL_NAME, revision=MODEL_REVISION,
                 cache_folder='models/sentence_transformers', local_files_only=True):
        self.model_name, self.revision = model_name, revision
        path = Path(cache_folder)
        self.cache_folder = str(path if path.is_absolute() else ROOT / path)
        self.local_files_only = local_files_only
        self.cache = {}

    def encode(self, texts):
        unseen = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if unseen:
            model = _model(self.model_name, self.revision, self.cache_folder, self.local_files_only)
            vectors = model.encode(unseen, batch_size=32, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
            self.cache.update(zip(unseen, vectors))
        return np.asarray([self.cache[t] for t in texts])


class SimilaritySelector:
    def __init__(self, *, semantic_weight=.7, min_similarity=.60, min_semantic=.35,
                 min_eda=.45, min_shared_features=4, min_coverage=.5, enabled=True,
                 encoder=None, **encoder_config):
        if type(enabled) is not bool:
            raise ValueError('similarity.enabled must be boolean')
        for key, value in dict(semantic_weight=semantic_weight, min_similarity=min_similarity,
                               min_semantic=min_semantic, min_eda=min_eda, min_coverage=min_coverage).items():
            if not number(value) or not 0 <= value <= 1:
                raise ValueError(f'similarity.{key} must be between 0 and 1')
        if type(min_shared_features) is not int or not 1 <= min_shared_features <= len(SCALES):
            raise ValueError('similarity.min_shared_features must be between 1 and 9')
        self.enabled, self.encoder = enabled, encoder or MiniLMEncoder(**encoder_config)
        self.weight, self.min_similarity = semantic_weight, min_similarity
        self.min_semantic, self.min_eda = min_semantic, min_eda
        self.min_shared, self.min_coverage = min_shared_features, min_coverage
        self.last_report = None

    def __call__(self, query, candidates):
        report = self.rank(query, candidates)
        self.last_report = report
        return report['selected_task_key']

    def rank(self, query, candidates):
        task, meta = query.get('task', {}), query.get('eda_metadata', {})
        text = semantic_text(task, meta)
        features = structural_features(task, meta)
        report = {'policy_version': 1, 'model': getattr(self.encoder, 'model_name', MODEL_NAME),
                  'revision': getattr(self.encoder, 'revision', MODEL_REVISION),
                  'formula': 'semantic_weight * max(0, cosine) + (1-semantic_weight) * eda_similarity',
                  'parameters': {'semantic_weight': self.weight, 'min_similarity': self.min_similarity,
                                 'min_semantic': self.min_semantic, 'min_eda': self.min_eda,
                                 'min_shared_features': self.min_shared, 'min_coverage': self.min_coverage},
                  'query_text': text, 'query_structure': features, 'candidates': [],
                  'selected_task_key': None, 'reason': 'no_eligible_source'}
        if not self.enabled:
            return {**report, 'reason': 'disabled'}
        eligible = []
        own = task_semantics(task)
        for candidate in sorted(candidates, key=lambda c: c['task_key']):
            other, other_meta = candidate.get('task', {}), candidate.get('eda_metadata', {})
            other_features = structural_features(other, other_meta)
            eda, shared = eda_similarity(features, other_features)
            coverage = len(shared) / max(1, len(set(features) | set(other_features)))
            other_text = semantic_text(other, other_meta)
            row = {'task_key': candidate['task_key'], 'task_name': other.get('name'),
                   'semantic_text': other_text, 'structure': other_features,
                   'eda': eda, 'shared_features': shared, 'coverage': coverage,
                   'semantic': None, 'score': None, 'accepted': False}
            other_facts = task_semantics(other)
            if task.get('problem_type') not in {'classification', 'regression'} or other.get('problem_type') != task.get('problem_type'):
                reason = 'different_problem_type'
            elif ((task.get('task_id') or task.get('name')) == (other.get('task_id') or other.get('name'))
                  or (own['dataset_id'] and own['dataset_id'] == other_facts['dataset_id'])):
                reason = 'same_task_or_dataset'
            elif not text or not other_text:
                reason = 'insufficient_task_semantics'
            elif meta.get('scope') != 'train' or other_meta.get('scope') != 'train':
                reason = 'non_training_metadata'
            elif len(shared) < self.min_shared or coverage < self.min_coverage:
                reason = 'insufficient_eda_evidence'
            elif eda < self.min_eda:
                reason = 'eda_below_threshold'
            else:
                reason = 'pending_semantics'
                eligible.append((row, other_text))
            row['reason'] = reason
            report['candidates'].append(row)
        if not eligible:
            return report
        try:
            vectors = np.asarray(self.encoder.encode([text] + [t for _, t in eligible]), dtype=float)
            norms = np.linalg.norm(vectors, axis=1)
            if vectors.ndim != 2 or len(vectors) != len(eligible) + 1 or not np.isfinite(vectors).all() or np.any(norms <= 0):
                raise ValueError('Invalid embedding vectors')
            vectors = vectors / norms[:, None]
        except Exception as error:
            for row, _ in eligible:
                row['reason'] = 'embedding_unavailable'
            return {**report, 'reason': 'embedding_unavailable', 'error_type': type(error).__name__,
                    'error': str(error)[:400]}
        for (row, _), vector in zip(eligible, vectors[1:]):
            row['semantic'] = float(np.clip(vectors[0] @ vector, -1., 1.))
            row['score'] = self.weight * max(0., row['semantic']) + (1 - self.weight) * row['eda']
            if row['semantic'] < self.min_semantic:
                row['reason'] = 'semantic_below_threshold'
            elif row['score'] < self.min_similarity:
                row['reason'] = 'combined_below_threshold'
            else:
                row.update(accepted=True, reason='eligible')
        admitted = [row for row, _ in eligible if row['accepted']]
        if admitted:
            chosen = min(admitted, key=lambda r: (-r['score'], r['task_key']))
            report.update(selected_task_key=chosen['task_key'], reason='selected')
        return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Fetch the pinned public model before running tasks offline.')
    args = parser.parse_args()
    encoder = MiniLMEncoder(local_files_only=not args.download)
    vectors = encoder.encode(['Predict diamond prices from carat and clarity.',
                              'Estimate diamond value using weight, clarity and cut.'])
    print(json.dumps({'model': MODEL_NAME, 'revision': MODEL_REVISION, 'dimensions': int(vectors.shape[1]),
                      'example_cosine': float(vectors[0] @ vectors[1]), 'device': 'cpu'}))


if __name__ == '__main__':
    main()
