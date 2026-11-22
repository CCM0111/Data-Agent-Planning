"""Bounded ridge API fusion and reproducible type-balanced exploration."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import numpy as np
from scipy.optimize import lsq_linear

from .context import default_feature_schema

POLICY_VERSION = 'region-linear-api-v1'


def _settings(config):
    config = config or {}
    config = config.get('region_bandit', config)
    if 'policy' in config:
        return {**config['policy'], 'dataset_total_weight': config.get('memory', {}).get('dataset_total_weight', 1.)}
    return config


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def derive_decision_seed(base_seed, decision_identity):
    """Use distinct deterministic draws per task, independent of process/order.

    Callers freeze the resulting seed in the decision. The identity must be a
    stable task/decision key, not a mutable policy snapshot or an output path.
    Sampling-repeat seeds are deliberately unrelated to this one-time draw.
    """
    if type(base_seed) is not int or not 0 <= base_seed < 2**32:
        raise ValueError('Decision base seed must fit uint32')
    if not isinstance(decision_identity, (str, dict)) or not decision_identity:
        raise ValueError('A stable nonempty decision identity is required')
    value = {'scheme': 'region-task-decision-seed-v1', 'base_seed': base_seed,
             'identity': decision_identity}
    return int(_hash(value)[:8], 16)


def initial_snapshot(schema=None, config=None):
    schema, config = schema or default_feature_schema(), _settings(config)
    result = {'policy_version': POLICY_VERSION, 'schema_id': schema['schema_id'],
              'feature_schema_version': schema['version'], 'theta': [0.] * len(schema['features']),
              'gamma': float(config.get('api_coefficient_prior', .1)), 'training_event_ids': [],
              'training_decision_ids': [], 'training_dataset_groups': [], 'training_data_hash': _hash([]),
              'memory_event_ids': [], 'memory_data_hash': _hash([]), 'dataset_count': 0}
    bounds = config.get('api_coefficient_bounds', [0, .2])
    if not float(bounds[0]) <= result['gamma'] <= float(bounds[1]):
        raise ValueError('API prior is outside coefficient bounds')
    result['snapshot_id'] = _hash(result)
    return result


def rebuild_snapshot(events, *, schema=None, config=None, identity=None):
    schema, config = schema or default_feature_schema(), _settings(config)
    result = initial_snapshot(schema, config)
    events = list(events)
    result['memory_event_ids'] = sorted(e['event_id'] for e in events)
    result['memory_data_hash'] = _hash(sorted(events, key=lambda e: e['event_id']))
    legal = [e for e in events if e.get('update_eligible', True) and e.get('reward') is not None]
    if not legal:
        if identity:
            result['compatibility'] = identity
        result['snapshot_id'] = _hash({k: v for k, v in result.items() if k != 'snapshot_id'})
        return result
    legal.sort(key=lambda e: e['event_id'])
    if len({e['decision_id'] for e in legal}) != len(legal):
        raise ValueError('A decision may contribute only one aggregated reward')
    dim = len(schema['features'])
    groups = Counter(e['dataset_group_id'] for e in legal)
    rows, targets, fitted_weights = [], [], []
    dataset_weight = float(config.get('dataset_total_weight', 1.0))
    if not np.isfinite(dataset_weight) or not 0 < dataset_weight <= 1:
        raise ValueError('Dataset feedback weight must be finite and in (0,1]')
    for event in legal:
        x = np.asarray(event['features'], float)
        if x.shape != (dim,) or not np.isfinite(x).all() or event.get('schema_id') != schema['schema_id']:
            raise ValueError('Incompatible event features')
        reward, score = float(event['reward']), float(event.get('api_score', 0))
        objective = event.get('feedback', {}).get('reward_objective', 'absolute')
        if objective not in ('absolute', 'paired_delta'):
            raise ValueError('Unknown event reward objective')
        if not (-1 if objective == 'paired_delta' else 0) <= reward <= 1 or score not in (-1, -.5, 0, .5, 1):
            raise ValueError('Invalid event reward or advisor score')
        confidence = event.get('sample_weight', 1.)
        if (type(confidence) not in (int, float) or not np.isfinite(confidence)
                or not 0 < confidence <= 1):
            raise ValueError('Event sample weight must be finite and in (0,1]')
        # Normalize by event count, not sum(confidence): otherwise a source
        # containing only single-pair observations would regain full weight.
        weight = dataset_weight * float(confidence) / groups[event['dataset_group_id']]
        fitted_weights.append(weight)
        rows.append(np.r_[x, score] * np.sqrt(weight))
        targets.append((reward - (0 if objective == 'paired_delta' else .5)) * np.sqrt(weight))
    regularization = np.r_[np.full(dim, float(config.get('ridge_theta', 1.0))), float(config.get('ridge_api', 1.0))]
    if (regularization <= 0).any():
        raise ValueError('Ridge strengths must be positive')
    matrix = np.vstack([rows, np.diag(np.sqrt(regularization))])
    prior = np.zeros(dim + 1)
    prior[-1] = float(config.get('api_coefficient_prior', .1))
    target = np.r_[targets, np.sqrt(regularization) * prior]
    bounds = config.get('api_coefficient_bounds', [0, .2])
    solved = lsq_linear(matrix, target, bounds=(np.r_[np.full(dim, -np.inf), bounds[0]],
                                              np.r_[np.full(dim, np.inf), bounds[1]]), tol=1e-10)
    if not solved.success or not np.isfinite(solved.x).all():
        raise RuntimeError('Bounded policy rebuild did not converge: ' + str(solved.message))
    result.update(theta=solved.x[:-1].tolist(), gamma=float(solved.x[-1]),
                  training_event_ids=[e['event_id'] for e in legal],
                  training_decision_ids=[e['decision_id'] for e in legal],
                  training_dataset_groups=sorted(groups), dataset_count=len(groups),
                  training_data_hash=_hash(legal))
    if any('sample_weight' in event for event in legal):
        result['training_event_weights'] = dict(zip(result['training_event_ids'], fitted_weights))
    if identity:
        result['compatibility'] = identity
    result['snapshot_id'] = _hash({k: v for k, v in result.items() if k != 'snapshot_id'})
    return result


def select_action(snapshot, actions_with_features, api_scores, seed, *, mode='develop_online', config=None):
    config = _settings(config)
    if mode not in ('develop_online', 'evaluate_frozen'):
        raise ValueError('Unknown region policy mode')
    actions = list(actions_with_features)
    if not actions or len({a['action_id'] for a in actions}) != len(actions):
        raise ValueError('Actions must have distinct IDs')
    levels = {'baseline': 0, 'cover': 1, 'focus': 2}
    actions.sort(key=lambda a: (levels[a.get('type', a.get('action_type'))], a['action_id']))
    theta, gamma = np.asarray(snapshot['theta'], float), float(snapshot['gamma'])
    bounds = config.get('api_coefficient_bounds', [0, .2])
    if not np.isfinite(theta).all() or not bounds[0] <= gamma <= bounds[1]:
        raise ValueError('Invalid policy snapshot parameters')
    if isinstance(api_scores, dict) and 'scores' in api_scores:
        api_scores = api_scores['scores']
    records = []
    for action in actions:
        context = action.get('context', action)
        vector = np.asarray(context['features'], float)
        if context.get('schema_id') != snapshot['schema_id'] or vector.shape != theta.shape or not np.isfinite(vector).all():
            raise ValueError('Action features do not match frozen policy schema')
        kind = action.get('type', action.get('action_type'))
        score = float(api_scores.get(action['action_id'], 0))
        if score not in (-1, -.5, 0, .5, 1) or (kind == 'baseline' and score != 0):
            raise ValueError('Invalid API recommendation')
        component = float(theta @ vector)
        records.append({**action, 'context': context, 'action_features': vector.tolist(),
                        'bandit_component': component, 'api_score': score, 'gamma': gamma,
                        'joint_score': .5 + component + gamma * score})
    best = max(range(len(records)), key=lambda i: records[i]['joint_score'])
    count = snapshot.get('dataset_count', len(snapshot.get('training_dataset_groups', [])))
    epsilon = 0.0 if mode == 'evaluate_frozen' else float(config.get(
        'cold_start_epsilon' if count < int(config.get('cold_start_distinct_tasks', 12)) else 'epsilon',
        .5 if count < int(config.get('cold_start_distinct_tasks', 12)) else .2))
    if not 0 <= epsilon <= 1:
        raise ValueError('Exploration probability must be in [0,1]')
    type_counts = Counter(a.get('type', a.get('action_type')) for a in records)
    probabilities = []
    for i, record in enumerate(records):
        kind = record.get('type', record.get('action_type'))
        q = 1.0 / (len(type_counts) * type_counts[kind])
        probability = (1 - epsilon) * (i == best) + epsilon * q
        record.update(exploration_probability=q, selection_probability=float(probability))
        probabilities.append(probability)
    chosen = records[int(np.random.default_rng(seed).choice(len(records), p=probabilities))]
    return {'policy_version': POLICY_VERSION, 'policy_snapshot_id': snapshot['snapshot_id'],
            'schema_id': snapshot['schema_id'], 'actions': records, 'chosen_action': chosen['action_id'],
            'chosen_action_id': chosen['action_id'], 'decision_seed': int(seed), 'epsilon': epsilon,
            'mode': mode, 'selection_probability': chosen['selection_probability']}
