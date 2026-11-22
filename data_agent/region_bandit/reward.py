"""Gap-weighted ranking accuracy for initial banks and explicit parent anchors."""
from itertools import combinations
import math

import numpy as np

REWARD_VERSION = 'region-pairwise-accuracy-v2'


def aggregate_seed_rewards(seed_rewards, *, objective='absolute'):
    """Keep resolved-but-uninformative evaluations distinct from missing results."""
    rows = [row if isinstance(row, dict) else {'reward': row} for row in seed_rewards]
    missing = not rows or any(row.get('reward') is None and row.get('status') != 'reward_uninformative'
                              for row in rows)
    values = [float(row['reward']) for row in rows if row.get('reward') is not None]
    if objective not in ('absolute', 'paired_delta'):
        raise ValueError('Unknown reward objective')
    lower = -1 if objective == 'paired_delta' else 0
    if any(not math.isfinite(value) or not lower <= value <= 1 for value in values):
        raise ValueError('Seed rewards outside objective range')
    reward = float(np.mean(values)) if values and not missing else None
    eligible = reward is not None and all(row.get('update_eligible', True)
        for row in rows if row.get('reward') is not None)
    return {'reward': reward, 'update_eligible': eligible,
            'status': 'reward_missing' if missing else 'ok' if eligible else 'reward_uninformative',
            'informative_seed_count': len(values) if not missing else 0}


def _settings(config):
    config = config or {}
    config = config.get('region_bandit', config)
    config = config.get('reward', {}) if ('reward' in config or 'mcts_rounds' in config) else config
    config = dict(config)
    if config.pop('objective', 'absolute') not in ('absolute', 'paired_delta'):
        raise ValueError('Unknown reward objective')
    if set(config) - {'atol', 'rtol', 'practical_tolerance'}:
        raise ValueError('Unsupported ranking-accuracy reward settings')
    result = {'atol': 1e-12, 'rtol': 1e-8, 'practical_tolerance': 0., **config}
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(key + ' must be finite and nonnegative')
    return result


def paired_delta_reward(after, before):
    """Signed improvement over the same seed's original coreset, never clipped."""
    a, b = after.get('reward'), before.get('reward')
    if a is None or b is None:
        status = ('reward_uninformative' if
            all(x.get('reward') is not None or x.get('status') == 'reward_uninformative'
                for x in (after, before)) else 'reward_missing')
        return {'reward': None, 'status': status, 'update_eligible': False,
                'reward_objective': 'paired_delta',
                'before_weighted_accuracy': b, 'after_weighted_accuracy': a}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (a, b)):
        raise ValueError('Paired accuracies must lie in [0,1]')
    return {**after, 'reward': a-b, 'reward_objective': 'paired_delta',
            'before_weighted_accuracy': b, 'after_weighted_accuracy': a,
            'update_eligible': after.get('update_eligible', True) and before.get('update_eligible', True)}


def anchored_sample_weight(comparable_pair_count, config=None):
    """Confidence is a training weight, not another multiplier on accuracy.

    With one informative pair the full-score gap cancels from normalized
    accuracy. Give such evidence less influence without changing its meaning.
    """
    config = config or {}
    section = config.get('region_bandit', config)
    weight = section.get('mcts_rounds', {}).get('single_pair_weight', .5)
    if type(weight) not in (int, float) or not math.isfinite(weight) or not 0 < weight <= 1:
        raise ValueError('mcts_rounds.single_pair_weight must be finite and in (0,1]')
    if type(comparable_pair_count) is not int or comparable_pair_count < 0:
        raise ValueError('Comparable pair count must be a nonnegative integer')
    return float(weight) if comparable_pair_count <= 1 else 1.


def _tied(a, b, cfg):
    return abs(a-b) <= cfg['atol'] + cfg['rtol'] * max(abs(a), abs(b))


def rank_method_ids(scores, *, direction, config=None):
    """Stable primary-metric ranking; ties never use full-stage information."""
    cfg = _settings(config)
    if direction not in ('maximize', 'minimize'):
        raise ValueError('Unknown metric direction')
    if not scores or any(v is None or not np.isfinite(v) for v in scores.values()):
        raise ValueError('Ranking requires finite method scores')
    utility = {m: float(v) if direction == 'maximize' else -float(v) for m, v in scores.items()}
    ordered = sorted(scores, key=lambda m: (-utility[m], str(m)))
    result, cursor = [], 0
    while cursor < len(ordered):
        stop = cursor + 1
        while stop < len(ordered) and _tied(utility[ordered[cursor]], utility[ordered[stop]], cfg):
            stop += 1
        result.extend(sorted(ordered[cursor:stop], key=str))
        cursor = stop
    return result


def compute_action_reward(core_scores, full_scores, *, direction, method_ids=None, config=None,
                          comparison_scope='initial', anchor_method_id=None):
    """Compare matching observed methods, never impute an unobserved full score.

    A dict of core scores may contain all five methods and full_scores only
    three. None-valued full placeholders are unobserved. An explicit method_ids
    list must match the observed set exactly; callers separately enforce their
    frozen execution plan so failed five-method runs cannot masquerade as top3.
    Parent-anchored MCTS feedback is an explicit, separate learning protocol:
    exactly one parent and one or two newly evaluated children. All low scores
    must come from the same coreset version (enforced by the caller).
    """
    cfg = _settings(config)
    if direction not in ('maximize', 'minimize'):
        raise ValueError('Unknown metric direction')
    if isinstance(full_scores, dict):
        if not isinstance(core_scores, dict):
            raise ValueError('Core and full scores must use matching containers')
        full_map, core_map = dict(full_scores), dict(core_scores)
    else:
        if isinstance(core_scores, dict) or len(core_scores) != len(full_scores):
            raise ValueError('Sequence scores must be aligned')
        ids = list(method_ids) if method_ids is not None else list(range(len(full_scores)))
        if len(ids) != len(full_scores) or len(set(ids)) != len(ids):
            raise ValueError('Method identities must be distinct and aligned')
        full_map, core_map = dict(zip(ids, full_scores)), dict(zip(ids, core_scores))
    observed = {m: v for m, v in full_map.items() if v is not None}
    ids = list(method_ids) if method_ids is not None else sorted(observed, key=str)
    if comparison_scope not in ('initial', 'mcts_parent_anchor'):
        raise ValueError('Unknown reward comparison scope')
    counts = (3, 5) if comparison_scope == 'initial' else (2, 3)
    if len(ids) not in counts or len(set(ids)) != len(ids) or set(ids) != set(observed):
        raise ValueError('Reward requires exactly three or five observed full method scores'
                         if comparison_scope == 'initial' else
                         'Anchored reward requires exactly two or three observed full method scores')
    if comparison_scope == 'mcts_parent_anchor':
        if anchor_method_id is None or anchor_method_id not in ids:
            raise ValueError('Anchored reward requires its observed parent method identity')
    elif anchor_method_id is not None:
        raise ValueError('Parent anchors require the explicit MCTS comparison scope')
    if any(m not in core_map or core_map[m] is None for m in ids):
        raise ValueError('Every observed full method needs its matching core score')
    full = np.asarray([observed[m] for m in ids], float)
    core = np.asarray([core_map[m] for m in ids], float)
    if full.shape != (len(ids),) or core.shape != full.shape or not np.isfinite([full, core]).all():
        raise ValueError('Observed core/full scores must be finite scalars')
    utility = full if direction == 'maximize' else -full
    proxy = core if direction == 'maximize' else -core
    pairs, numerator, denominator = [], 0., 0.
    for i, j in combinations(range(len(ids)), 2):
        gap = float(abs(utility[i]-utility[j]))
        excluded = _tied(utility[i], utility[j], cfg) or gap <= cfg['practical_tolerance']
        credit = None if excluded else (.5 if _tied(proxy[i], proxy[j], cfg)
            else float((proxy[i]-proxy[j]) * (utility[i]-utility[j]) > 0))
        weight = 0. if excluded else gap
        if not excluded:
            numerator += weight * credit
            denominator += weight
        pairs.append({'method_ids': [ids[i], ids[j]], 'full_gap': gap, 'weight': weight,
                      'accuracy': credit, 'status': 'full_tied_or_near_tied' if excluded else 'comparable'})
    informative = denominator > 0
    reward = float(np.clip(numerator/denominator, 0, 1)) if informative else None
    core_map = {m: float(core_map[m]) for m in ids}
    full_map = {m: float(observed[m]) for m in ids}
    result = {'reward_version': REWARD_VERSION, 'reward': reward, 'weighted_accuracy': reward,
            'status': 'ok' if informative else 'reward_uninformative', 'update_eligible': informative,
            'scope': 'all_five_methods' if len(ids) == 5 else 'observed_three_methods',
            'observed_method_count': len(ids), 'method_ids': ids, 'pair_count': len(pairs),
            'comparable_pair_count': sum(p['status'] == 'comparable' for p in pairs),
            'pairs': pairs, 'correct_weight_sum': numerator, 'order_weight_sum': denominator,
            'order_loss': 1-reward if informative else None,
            'raw_order_loss': denominator-numerator, 'full_spread': float(np.ptp(utility)),
            'all_full_tied': _tied(float(max(utility)), float(min(utility)), cfg),
            'core_ranking': rank_method_ids(core_map, direction=direction, config=cfg),
            'full_ranking': rank_method_ids(full_map, direction=direction, config=cfg),
            'core_scores': core_map, 'full_scores': full_map}
    if comparison_scope == 'mcts_parent_anchor':
        result.update(scope=comparison_scope, comparison_scope=comparison_scope,
                      anchor_method_id=anchor_method_id,
                      sample_weight=anchored_sample_weight(result['comparable_pair_count'], config))
    return result
