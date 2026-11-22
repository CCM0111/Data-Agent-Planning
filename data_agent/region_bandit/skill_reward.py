"""Versioned screening utility; incomplete full references never imply retention."""
from itertools import combinations
import math

from .reward import rank_method_ids
from .schemas import stable_hash

DEFAULT_REWARD = dict(version='skill-screening-v1', keep_k=3, keep_weight=.8,
                      atol=1e-12, rtol=1e-8, practical_tolerance=0.)


def reward_settings(settings=None):
    if set(settings or {}) - set(DEFAULT_REWARD):
        raise ValueError('Unknown skill reward setting')
    cfg = {**DEFAULT_REWARD, **(settings or {})}
    if cfg['version'] != DEFAULT_REWARD['version'] or type(cfg['keep_k']) is not int or cfg['keep_k'] < 1:
        raise ValueError('Invalid skill reward version/keep count')
    for key in ('keep_weight', 'atol', 'rtol', 'practical_tolerance'):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError('Invalid reward coefficient: ' + key)
    if cfg['keep_weight'] > 1:
        raise ValueError('keep_weight must be in [0,1]')
    return cfg


def screening_reward(core, full, *, expected_ids, direction, settings=None):
    """Top-k retains ANY full-best tied method; order is gap-weighted accuracy.

    Complete references produce .8*retention + .2*order_accuracy by default.
    Partial observations expose order diagnostics only, never a composite label.
    All-full-tied banks have descriptive retention but no informative reward.
    """
    cfg = reward_settings(settings)
    ids = list(expected_ids)
    if len(ids) < 2 or len(set(ids)) != len(ids) or set(core) != set(ids) or set(full) - set(ids):
        raise ValueError('Explicit distinct bank and aligned coreset scores required')
    if direction not in ('maximize', 'minimize') or cfg['keep_k'] > len(ids):
        raise ValueError('Invalid direction/keep count')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in core.values()):
        raise ValueError('Finite core scores required')
    observed = {k: v for k, v in full.items() if v is not None}
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in observed.values()):
        raise ValueError('Finite observed full scores required')
    sign = 1 if direction == 'maximize' else -1
    tied = lambda a, b: abs(a-b) <= cfg['atol'] + cfg['rtol'] * max(abs(a), abs(b))
    ranking_cfg = {k: cfg[k] for k in ('atol', 'rtol', 'practical_tolerance')}
    kept = rank_method_ids(core, direction=direction, config=ranking_cfg)[:cfg['keep_k']]
    numerator = denominator = 0.
    pairs = []
    for a, b in combinations(sorted(observed, key=str), 2):
        gap = abs(observed[a]-observed[b])
        comparable = not tied(observed[a], observed[b]) and gap > cfg['practical_tolerance']
        credit = None if not comparable else (.5 if tied(core[a], core[b]) else
                   float((core[a]-core[b])*(observed[a]-observed[b]) > 0))
        if comparable:
            numerator += gap*credit
            denominator += gap
        pairs.append(dict(method_ids=[a, b], full_gap=gap, accuracy=credit,
                          reversed=credit == 0 if comparable else None))
    accuracy = numerator/denominator if denominator else None
    complete = set(observed) == set(ids)
    champions = []
    if complete:
        best = max(sign*v for v in observed.values())
        champions = [k for k, v in observed.items() if tied(sign*v, best)]
    retained = bool(set(champions) & set(kept)) if complete else None
    reward = (cfg['keep_weight']*float(retained)+(1-cfg['keep_weight'])*accuracy
              if complete and accuracy is not None else None)
    return dict(reward=reward, status='ok' if reward is not None else
                'partial_full_reference' if not complete else 'reward_uninformative',
                update_eligible=reward is not None, full_reference_complete=complete,
                protocol=cfg, protocol_hash=stable_hash(cfg), retained_ids=kept,
                champion_ids=champions if complete else None, champion_retained=retained,
                champion_miss=None if retained is None else int(not retained),
                keep_loss=None if retained is None else int(not retained),
                order_accuracy=accuracy, order_loss=None if accuracy is None else 1-accuracy,
                observed_method_ids=sorted(observed, key=str), expected_method_ids=ids, pairs=pairs,
                evidence_scope='fixed_bank' if complete else 'observed_methods_only')
