"""Observed-method calibration and bounded, source-level skill promotion.

This is an engineering acceptance screen. Parent/child pairs, sampling seeds and
successive search nodes never become independent validation datasets.
"""
from collections import defaultdict
from itertools import combinations
import math

import numpy as np

from .reward import rank_method_ids
from .schemas import stable_hash
from .skill_store import encoded

REWARD_DEFAULTS = dict(version='mcts-observed-rank-v1', bounded_tolerance=.002,
    relative_tolerance=.01, absolute_floor=1e-8, severe_multiplier=2.)
RELATIVE_VERSION = 'mcts-relative-champion-order-v2'
PARENT_PAIR_VERSION = 'mcts-relative-parent-pair-v3'
FIXED_PARENT_VERSION = 'mcts-fixed-parent-direction-v4'
INITIAL_TOP3_VERSION = 'mcts-observed-initial-top3-v5'
FIXED_PARENT_PROTOCOLS = (FIXED_PARENT_VERSION, INITIAL_TOP3_VERSION)
RELATIVE_PROTOCOLS = (RELATIVE_VERSION, PARENT_PAIR_VERSION, *FIXED_PARENT_PROTOCOLS)
GATE_DEFAULTS = dict(version='mcts-paired-checkpoints-v1', first_sources=4,
    max_sources=6, checkpoints_per_source=2, min_delta=.01,
    extra_branch_penalty=.01, max_source_regression=.25,
    bootstrap_repeats=2000, bootstrap_seed=2026)


def reward_settings(settings=None):
    cfg = {**REWARD_DEFAULTS, **(settings or {})}
    if set(cfg) != set(REWARD_DEFAULTS) or cfg['version'] not in (REWARD_DEFAULTS['version'], *RELATIVE_PROTOCOLS):
        raise ValueError('Unknown observed-rank reward protocol')
    if any(type(cfg[k]) not in (int, float) or not math.isfinite(cfg[k]) or cfg[k] <= 0
           for k in set(cfg)-{'version'}):
        raise ValueError('Reward tolerances must be finite and positive')
    return cfg


ABSOLUTE_METRICS = {'accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'auc', 'roc_auc', 'r2'}


def practical_tolerance(metric, reference_scale=None, *, settings=None):
    """One practical-tie threshold for sampling rewards and search feedback.

    Loss scales must be frozen before the action: the full parent for an
    expansion, or an initialization proxy scale. Missing scales retain the
    legacy numerical floor for archived observations only.
    """
    cfg = reward_settings(settings)
    if reference_scale is not None and (type(reference_scale) not in (int, float)
            or not math.isfinite(reference_scale) or reference_scale < 0):
        raise ValueError('Reference scale must be finite and nonnegative')
    if metric in ABSOLUTE_METRICS:
        return cfg['bounded_tolerance']
    return max(cfg['absolute_floor'], cfg['relative_tolerance']*(reference_scale or 0.))


def checkpoint_reward(core, full, *, expected_ids, direction, metric, anchor_id=None,
                      reference_scale=None, settings=None):
    cfg = reward_settings(settings)
    expected = list(expected_ids)
    observed = {k: v for k, v in full.items() if v is not None}
    if (len(expected) < 2 or len(set(expected)) != len(expected)
            or set(core)-set(expected) or set(observed)-set(expected)
            or direction not in ('maximize', 'minimize')):
        raise ValueError('Aligned explicit method bank and observed scores required')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in [*core.values(), *observed.values()]):
        raise ValueError('Scores must be finite; unobserved full scores must be absent or null')
    # R2 is unbounded below but has an interpretable absolute score scale.
    if metric in ABSOLUTE_METRICS:
        scale = 1.
    elif anchor_id is not None:
        if anchor_id not in observed:
            raise ValueError('Relative-loss tolerance requires the frozen full parent')
        scale = abs(observed[anchor_id])
        if reference_scale is not None and reference_scale != scale:
            raise ValueError('Expansion reference scale differs from the frozen full parent')
    else:
        # Initialization passes a proxy scale frozen before sampling. Never
        # estimate it from the candidate outcomes that this reward judges.
        scale = reference_scale
    tolerance = practical_tolerance(metric, scale, settings=cfg)
    numerator = denominator = 0.
    pairs = []
    anchor_missing = anchor_id is not None and anchor_id not in core
    for a, b in combinations(sorted(set(observed)&set(core), key=str), 2):
        full_gap, core_gap = observed[a]-observed[b], core[a]-core[b]
        comparable = not anchor_missing and abs(full_gap) > tolerance
        credit = (None if not comparable else .5 if abs(core_gap) <= tolerance
                  else float(full_gap*core_gap > 0))
        severe = bool(comparable and credit == 0 and abs(full_gap) >= cfg['severe_multiplier']*tolerance)
        if comparable:
            numerator += abs(full_gap)*credit
            denominator += abs(full_gap)
        pairs.append(dict(method_ids=[a, b], full_gap=abs(full_gap), accuracy=credit,
                          reversed=credit == 0 if comparable else None, severe=severe))
    comparable = [p for p in pairs if p['accuracy'] is not None]
    accuracy = numerator/denominator if denominator else None
    complete = set(observed) == set(expected)
    return dict(reward=accuracy, order_accuracy=accuracy, order_loss=None if accuracy is None else 1-accuracy,
        status='parent_anchor_unavailable' if anchor_missing else 'ok' if accuracy is not None else 'reward_uninformative', update_eligible=accuracy is not None,
        protocol=cfg, protocol_hash=stable_hash(cfg), practical_tolerance=tolerance, reference_scale=scale,
        metric=metric, direction=direction, anchor_id=anchor_id,
        expected_method_ids=expected, observed_method_ids=sorted(observed, key=str),
        full_reference_complete=complete, champion_miss=None, champion_retained=None,
        evidence_scope='observed_parent_children' if anchor_id is not None else 'observed_initial_methods',
        pairs=pairs, comparable_pair_count=len(comparable),
        severe_error_rate=(sum(p['severe'] for p in comparable)/len(comparable) if comparable else None),
        sample_weight=(.5 if len(comparable) == 1 else 1. if comparable else 0.),
        selection_bias='Only full-observed methods are judged; no claim about untrained candidates')


def difference(after, before):
    return None if after is None or before is None else after-before


def relative_reward(scores, full, baseline, measured, *, nominated_method_id=None):
    """Compare paired rankings; the combined policy may supply its actual nominee."""
    expected = measured['expected_method_ids']
    if set(baseline)-set(expected):
        raise ValueError('Unknown method in relative baseline')
    scores = {i: score for i, score in scores.items() if i in baseline}
    parent = measured['anchor_id']
    fixed_parent = parent is not None and measured['protocol']['version'] in FIXED_PARENT_PROTOCOLS
    if fixed_parent and parent in baseline:
        scores[parent] = baseline[parent]
    kwargs = dict(expected_ids=expected, direction=measured['direction'], metric=measured['metric'],
        anchor_id=measured['anchor_id'], reference_scale=measured['reference_scale'], settings=measured['protocol'])
    measured = checkpoint_reward(scores, full, **kwargs)
    before = checkpoint_reward({i: baseline[i] for i in scores}, full, **kwargs)
    gain = difference(measured['order_accuracy'], before['order_accuracy'])
    nominees = [i for i in expected if i != measured['anchor_id']]
    if nominated_method_id is not None and nominated_method_id not in nominees:
        raise ValueError('Nominated method outside the frozen nomination bank')
    observed_initial = parent is None and measured['protocol']['version'] == INITIAL_TOP3_VERSION
    if observed_initial:
        nominees = measured['observed_method_ids']
        nominated_method_id = None  # Replay can nominate an unobserved method outside the frozen three.
    def winner(values):
        if not nominees or not all(i in values for i in nominees):
            return None
        return rank_method_ids({i: values[i] for i in nominees}, direction=measured['direction'])[0]
    action_winner = nominated_method_id if nominated_method_id is not None else winner(scores)
    baseline_winner = winner(baseline)
    complete = bool(nominees) and all(i in scores and full.get(i) is not None for i in nominees)
    def champion(first):
        if not complete:
            return None
        sign = 1 if measured['direction'] == 'maximize' else -1
        return max(sign*full[i] for i in nominees)-sign*full[first] <= measured['practical_tolerance']
    hit, base_hit = champion(action_winner), champion(baseline_winner)
    absolute = .35*hit + .65*measured['order_accuracy'] if hit is not None and gain is not None else None
    base = .35*base_hit + .65*before['order_accuracy'] if absolute is not None else None
    result = {**measured, 'reward': difference(absolute, base), 'absolute_score': absolute,
        'baseline_absolute_score': base, 'champion_retained': hit,
        'predicted_champion_id': action_winner, 'baseline_predicted_champion_id': baseline_winner,
        'nomination_source': 'policy_nominee' if nominated_method_id is not None else 'score_ranking',
        'champion_miss': None if hit is None else not hit, 'baseline_champion_hit': base_hit,
        'paired_method_ids': sorted(scores, key=str), 'order_gain': gain, 'baseline_order_accuracy': before['order_accuracy'],
        'status': 'reward_uninformative' if gain is None else 'complete' if complete else 'partial_observed_order',
        'update_eligible': gain is not None}

    if observed_initial:
        result.update(reward_basis='observed_initial_champion_order_gain',
            paired_method_ids=sorted(set(nominees) & scores.keys(), key=str),
            update_eligible=result['reward'] is not None,
            status='observed_initial_gain' if result['reward'] is not None else 'reward_uninformative')

    if parent is not None and measured['protocol']['version'] in (PARENT_PAIR_VERSION, *FIXED_PARENT_PROTOCOLS):
        comparisons = []
        for after, prior in zip(measured['pairs'], before['pairs']):
            pair = after['method_ids']
            if parent not in pair or after['accuracy'] is None:
                continue
            child = next(i for i in pair if i != parent)
            gain = after['accuracy'] - prior['accuracy']
            if fixed_parent:
                gain = .625 * (2 * after['accuracy'] - 1) - .375 * (2 * prior['accuracy'] - 1)
            comparisons.append(dict(child_id=child, full_gap=full[child]-full[parent],
                parent_sample_gap=baseline[child]-baseline[parent], action_sample_gap=scores[child]-scores[parent],
                before=prior['accuracy'], after=after['accuracy'], gain=gain,
                severe=after['severe']))
        count = len(comparisons)
        result.update(reward=float(np.mean([c['gain'] for c in comparisons])) if count else None,
            reward_basis='fixed_parent_direction_reward' if fixed_parent else 'mean_parent_child_direction_gain', parent_child_comparisons=comparisons,
            composite_gain=difference(absolute, base), update_eligible=bool(count),
            sample_weight=.5 if count == 1 else float(bool(count)),
            severe_error_rate=sum(c['severe'] for c in comparisons)/count if count else None,
            status='observed_parent_child_gain' if count else 'reward_uninformative')
    return result


def relative_fact(fact, baseline, kind):
    """Score primary, combined and paired actions on the same observed methods."""
    if baseline is None:
        return fact
    branches = fact.get('branches', [])
    for branch in branches:
        branch['reward'] = relative_reward(branch['scores'], fact['full_scores'], baseline, branch['reward'])
    primary = branches[0]['reward'] if branches else relative_reward(
        fact['core_scores'], fact['full_scores'], baseline, fact['reward'])
    combined = relative_reward(fact['core_scores'], fact['full_scores'], baseline, fact['reward'],
                               nominated_method_id=fact.get('nominated_method_id'))
    fact.update(baseline_scores=baseline, baseline_kind=kind, combined_reward=combined)
    fact['reward'] = primary
    if branches:
        fact['combination_gain'] = difference(combined['reward'], fact['reward']['reward'])
    if len(branches) == 2:
        control = next((b for b in branches if b['action']['type'] == 'baseline'), None)
        adjusted = next((b for b in branches if b['action']['type'] != 'baseline'), None)
        fact['branch_reward_delta'] = None
        if control and adjusted:
            paired = relative_reward(adjusted['scores'], fact['full_scores'], control['scores'],
                                     adjusted['reward'])
            fact['branch_reward_delta'] = paired['reward']
            fact['branch_order_gain'] = paired['order_gain']
    return fact


def gate_settings(settings=None):
    cfg = {**GATE_DEFAULTS, **(settings or {})}
    if set(cfg) != set(GATE_DEFAULTS) or cfg['version'] != GATE_DEFAULTS['version']:
        raise ValueError('Unknown checkpoint validation protocol')
    if (any(type(cfg[k]) is not int for k in ('first_sources', 'max_sources', 'checkpoints_per_source',
            'bootstrap_repeats', 'bootstrap_seed')) or not 2 <= cfg['first_sources'] <= cfg['max_sources']
            or cfg['checkpoints_per_source'] < 2 or cfg['bootstrap_repeats'] < 100 or cfg['bootstrap_seed'] < 0):
        raise ValueError('Invalid checkpoint validation sizes')
    if any(type(cfg[k]) not in (int, float) or not math.isfinite(cfg[k]) or cfg[k] < 0
           for k in ('min_delta', 'extra_branch_penalty', 'max_source_regression')):
        raise ValueError('Invalid checkpoint validation thresholds')
    return cfg


def bind_validation(store, edit_id, cases, *, settings=None, matched=False):
    """Freeze all six source/checkpoint identities BEFORE reading paired outcomes."""
    cfg = gate_settings(settings)
    edit = store.candidate(edit_id)
    sources = list(dict.fromkeys(c['source'] for c in cases))
    keys = [(c['source'], c['checkpoint_id'], c['seed']) for c in cases]
    if len(keys) != len(set(keys)) or (not matched and len(sources) != cfg['max_sources']):
        raise ValueError('Validation plan needs distinct cases from the declared source count')
    if matched and len({c['checkpoint_id'] for c in cases}) > 4:
        raise ValueError('At most four matched checkpoints')
    for source in ([] if matched else sources):
        role = store.db.execute('SELECT role FROM sources WHERE id=?', (source,)).fetchone()
        if not role or role[0] != 'validation':
            raise ValueError('Only validation sources can enter the frozen plan')
        if len({c['checkpoint_id'] for c in cases if c['source'] == source}) != cfg['checkpoints_per_source']:
            raise ValueError('Predeclare exactly two checkpoints per source (or configured count)')
    plan = dict(settings=cfg, cases=cases, sources=sources)
    if matched:
        plan.update(vote_protocol='matched-four-v1', win_margin=.05,
                    bundle_hashes=[edit['parent_hash'], edit['candidate_hash']])
    key = 'mcts_gate:'+edit_id
    with store.transaction():
        row = store.db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        if row and row[0] != encoded(plan):
            raise ValueError('Checkpoint validation plan already frozen')
        store.db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', (key, encoded(plan)))
    return plan


def nomination_stage_evidence(comparisons):
    """Average seeds within checkpoint, then checkpoints within source, then sources."""
    stages = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    votes = {'improved': 1, 'practical_tie': 0, 'regressed': -1}
    for case in comparisons:
        stages[case['stage']][case['source']][case['checkpoint_id']].append(
            votes.get(case['outcome']))
    evidence = {}
    for stage, sources in stages.items():
        coverage = []
        balances = {}
        for source, checkpoints in sources.items():
            checkpoint_coverage = []
            checkpoint_balance = []
            for seeds in checkpoints.values():
                observed = [vote for vote in seeds if vote is not None]
                checkpoint_coverage.append(len(observed) / len(seeds))
                if observed:
                    checkpoint_balance.append(float(np.mean(observed)))
            coverage.append(float(np.mean(checkpoint_coverage)))
            if checkpoint_balance:
                balances[source] = float(np.mean(checkpoint_balance))
        evidence[stage] = dict(observed_sources=len(balances), coverage=float(np.mean(coverage)),
            source_balances=balances, balance=float(np.mean(list(balances.values()))) if balances else None)
    return evidence


def checkpoint_gate(store, edit_id, pairs, *, final=False):
    """Compare persisted old/new observations; archive an exhausted pending edit."""
    import json
    row = store.db.execute('SELECT value FROM meta WHERE key=?', ('mcts_gate:'+edit_id,)).fetchone()
    if not row:
        raise ValueError('Freeze validation cases before replaying the policies')
    plan = json.loads(row[0]); cfg = plan['settings']; edit = store.candidate(edit_id)
    planned_stages = {(c['source'], c['checkpoint_id'], c['seed']): c.get('stage') for c in plan['cases']}
    allowed = set(planned_stages)
    seen = set(); values = defaultdict(lambda: defaultdict(list)); changed = 0; incomplete = False
    nomination_comparisons = []; checkpoint_stages = {}
    nomination_set_counts = {side: dict(observed=0, nominated=0) for side in ('old', 'new')}
    # When any relative checkpoint is partial, compare order gain consistently
    # across the entire gate. Preserve complete-reference champion regressions.
    paired_facts = [(store.fact(old), store.fact(new)) for old, new in pairs]
    relative_metrics = [f['reward'] for pair in paired_facts for f in pair
                        if f['reward'].get('protocol', {}).get('version') in RELATIVE_PROTOCOLS]
    comparison_metric = 'order_gain' if any(r['reward'] is None for r in relative_metrics) else 'reward'
    if relative_metrics and relative_metrics[0]['protocol']['version'] in (PARENT_PAIR_VERSION, *FIXED_PARENT_PROTOCOLS):
        comparison_metric = 'stage_relative_gain'
    vote_deltas = {}
    champion_regressions = 0
    for old, new in paired_facts:
        if (old['role'] != 'validation' or new['role'] != 'validation'
                or old['bundle_hash'] != edit['parent_hash'] or new['bundle_hash'] != edit['candidate_hash']):
            raise ValueError('Validation requires paired parent/candidate bundle facts')
        for key in ('source', 'task', 'bank_hash', 'checkpoint_id', 'seed', 'decision_protocol_hash',
                    'sampling_policy_hash', 'full_scores'):
            if key not in old or old[key] != new.get(key):
                raise ValueError('Frozen validation differs: '+key)
        key = (old['source'], old['checkpoint_id'], old['seed'])
        if key not in allowed or key in seen:
            raise ValueError('Duplicate or undeclared checkpoint')
        seen.add(key)
        stage = old.get('search_stage') or 'unknown'
        if plan.get('vote_protocol') == 'matched-four-v1' and any(
                f['reward']['protocol']['version'] != INITIAL_TOP3_VERSION or
                (f['reward']['anchor_id'] is None) != (stage == 'initialization') for f in (old, new)):
            raise ValueError('Matched validation requires the stage-specific v5 sampling reward')
        if stage != (new.get('search_stage') or 'unknown'):
            raise ValueError('Frozen validation differs: search_stage')
        if planned_stages[key] is not None and stage != planned_stages[key]:
            raise ValueError('Checkpoint stage differs from frozen validation plan')
        stage_key = key[:2]
        if stage_key in checkpoint_stages and checkpoint_stages[stage_key] != stage:
            raise ValueError('Checkpoint stage differs between seeds')
        checkpoint_stages[stage_key] = stage
        for field in ('protocol_hash', 'observed_method_ids', 'expected_method_ids', 'practical_tolerance',
                      'reference_scale', 'metric', 'direction', 'anchor_id', 'paired_method_ids'):
            if old['reward'].get(field) != new['reward'].get(field):
                raise ValueError('Validation rewards have different observed references: '+field)
        # Recompute the primary decision's reward from the frozen scores.
        if relative_metrics and any(f['reward']['protocol']['version'] != relative_metrics[0]['protocol']['version'] for f in (old, new)):
            raise ValueError('Cannot mix legacy and relative validation protocols')
        if relative_metrics and (not all(isinstance(f.get('baseline_scores'), dict) for f in (old, new))
                                 or old['baseline_scores'] != new['baseline_scores']
                                 or old.get('baseline_kind') != new.get('baseline_kind')):
            raise ValueError('Validation baseline differs or is missing')
        samples = []
        for fact in (old, new):
            primary = fact['branches'][0] if relative_metrics and fact.get('branches') else None
            scores = primary['scores'] if primary else fact['core_scores']
            samples.append((primary['sample_hash'] if primary else fact['sample_hash'], scores))
            r = fact['reward']
            if relative_metrics:
                calculated = relative_reward(scores, fact['full_scores'], fact['baseline_scores'], r)
                combined = relative_reward(fact['core_scores'], fact['full_scores'], fact['baseline_scores'],
                                           r, nominated_method_id=fact.get('nominated_method_id'))
                if combined != fact.get('combined_reward'):
                    raise ValueError('Stored combined policy reward is inconsistent with scores or nomination')
            else:
                calculated = checkpoint_reward(scores, fact['full_scores'], expected_ids=r['expected_method_ids'],
                    direction=r['direction'], metric=r['metric'], anchor_id=r.get('anchor_id'),
                    reference_scale=r.get('reference_scale'), settings=r['protocol'])
            if calculated != r:
                raise ValueError('Stored checkpoint reward is inconsistent with scores')
        if samples[0][0] == samples[1][0] and samples[0][1] != samples[1][1]:
            raise ValueError('Same frozen samples produced inconsistent replay results')
        changed += samples[0][0] != samples[1][0]
        # The sampling score is distinct from model improvement. Nomination
        # utility is reported only for two actually full-evaluated nominees.
        noms = [f.get('nominated_method_id', f.get('selected_method_id')) for f in (old, new)]
        utility = None
        # Archived expansion replays nominated the raw ranking leader. Their
        # scores remain valid, but cannot establish a scheduled nominee.
        comparable_policy = all(
            f.get('search_stage') != 'mcts_expansion' or
            f.get('execution_kind') != 'checkpoint_replay' or
            f.get('nomination_protocol') == 'scheduled-primary-v1'
            for f in (old, new))
        if comparable_policy and all(n in old['full_scores'] and old['full_scores'][n] is not None for n in noms):
            sign = 1 if old['reward']['direction'] == 'maximize' else -1
            utility = sign*(old['full_scores'][noms[1]]-old['full_scores'][noms[0]])
        set_coverage = {}
        for side, fact in (('old', old), ('new', new)):
            method_ids = fact.get('nominated_method_ids')
            if method_ids:
                observed = sum(fact['full_scores'].get(method) is not None for method in method_ids)
                set_coverage[side] = dict(method_ids=method_ids, observed_count=observed,
                    observed_fraction=observed/len(method_ids))
                nomination_set_counts[side]['observed'] += observed
                nomination_set_counts[side]['nominated'] += len(method_ids)
        nomination_comparisons.append(dict(stage=stage, source=key[0], checkpoint_id=key[1], seed=key[2],
            old_method=noms[0], new_method=noms[1], observed_both=utility is not None,
            signed_metric_delta=utility, metric=old['reward']['metric'],
            nomination_set_coverage=set_coverage,
            outcome='unobserved' if utility is None else
                'regressed' if utility < -old['reward']['practical_tolerance'] else
                'improved' if utility > old['reward']['practical_tolerance'] else 'practical_tie'))
        # Use one common ranking metric across partially observed validation cases.
        if relative_metrics:
            champion_regressions += (old['reward']['champion_retained'] is True and
                                     new['reward']['champion_retained'] is False)
        field = comparison_metric
        if field == 'stage_relative_gain' and old['reward']['protocol']['version'] == INITIAL_TOP3_VERSION:
            field = 'reward'
        elif field == 'stage_relative_gain':
            field = 'reward' if old['reward']['anchor_id'] is not None or old['reward']['full_reference_complete'] else 'order_gain'
        if any(f['reward'][field] is None or f['skill_execution'].get('decision_status') == 'fallback'
               for f in (old, new)):
            incomplete = True; continue
        delta = new['reward'][field]-old['reward'][field]
        vote_deltas[key] = float(delta)
        branch_delta = new.get('branch_count', 1)-old.get('branch_count', 1)
        severe_delta = new['reward']['severe_error_rate']-old['reward']['severe_error_rate']
        # A single observed parent/child pair is useful but weaker than a bank
        # with multiple pairs. Use the same conservative weight on both sides.
        weight = min(old['reward']['sample_weight'], new['reward']['sample_weight'])
        values[key[0]][key[1]].append((
            (delta, delta-cfg['extra_branch_penalty']*branch_delta, severe_delta), weight))
    expected_first = {c for c in allowed if c[0] in plan['sources'][:cfg['first_sources']]}
    if not final and not seen <= expected_first:
        raise ValueError('First-stage validation must use the predeclared source prefix')
    required = allowed if final else expected_first
    complete = required <= seen and not incomplete
    source_means = {}; checkpoint_weights = {}
    stage_values = defaultdict(lambda: defaultdict(list))
    for source, checkpoints in values.items():
        means = []; weights = []
        checkpoint_weights[source] = {}
        for checkpoint, rows in checkpoints.items():
            outcomes, row_weights = zip(*rows)
            checkpoint_mean = np.average(outcomes, axis=0, weights=row_weights)
            means.append(checkpoint_mean)
            # Repeating a seed cannot increase a checkpoint's evidence weight.
            weight = float(np.mean(row_weights))
            weights.append(weight); checkpoint_weights[source][checkpoint] = weight
            stage_values[checkpoint_stages[(source, checkpoint)]][source].append((checkpoint_mean, weight))
        source_means[source] = np.average(means, axis=0, weights=weights)
    stage_means = {}
    for stage, sources in stage_values.items():
        # Repeated checkpoints/seeds are aggregated within source first, just
        # like the overall screen; they never become independent datasets.
        per_source = []
        for rows in sources.values():
            outcomes, weights = zip(*rows)
            per_source.append(np.average(outcomes, axis=0, weights=weights))
        stage_means[stage] = np.mean(per_source, axis=0)
    mean = np.mean(list(source_means.values()), axis=0) if source_means else None
    result = dict(protocol=cfg['version'], edit_id=edit_id, parent_hash=edit['parent_hash'],
        candidate_hash=edit['candidate_hash'], status='pending', reason='incomplete_validation',
        pairs=pairs, changed_samples=changed, source_count=len(source_means), final=final,
        mean_delta=None if mean is None else float(mean[0]),
        mean_net_delta=None if mean is None else float(mean[1]),
        mean_severe_error_delta=None if mean is None else float(mean[2]),
        source_deltas={s: float(v[0]) for s, v in source_means.items()}, bootstrap_lower=None,
        checkpoint_evidence_weights=checkpoint_weights,
        stage_deltas={stage: float(v[0]) for stage, v in stage_means.items()},
        stage_net_deltas={stage: float(v[1]) for stage, v in stage_means.items()},
        stage_source_counts={stage: len(sources) for stage, sources in stage_values.items()},
        nomination_comparisons=nomination_comparisons,
        nomination_pair_count=len(nomination_comparisons),
        nomination_observed_pair_count=sum(c['observed_both'] for c in nomination_comparisons),
        nomination_coverage_rate=(sum(c['observed_both'] for c in nomination_comparisons)/len(nomination_comparisons)
                                  if nomination_comparisons else None),
        nomination_regression_count=sum(c['outcome'] == 'regressed' for c in nomination_comparisons),
        nomination_set_observed_fractions={side: counts['observed']/counts['nominated']
            if counts['nominated'] else None for side, counts in nomination_set_counts.items()},
        interpretation='Source-level engineering screen; repeated checkpoints/seeds are correlated; no significance claim')
    if relative_metrics:
        result.update(comparison_metric=comparison_metric, observed_champion_regressions=champion_regressions)
    if source_means:
        net = [v[1] for v in source_means.values()]
        samples = np.random.default_rng(cfg['bootstrap_seed']).choice(net,
            (cfg['bootstrap_repeats'], len(net))).mean(axis=1)
        result['bootstrap_lower'] = float(np.quantile(samples, .05))
    observed_noms = [c for c in nomination_comparisons if c['observed_both']]
    nomination_sources = {c['source'] for c in observed_noms}
    required_sources = min(2, len(plan['sources']))
    result['nomination_observed_sources'] = len(nomination_sources)
    result['nomination_required_sources'] = required_sources
    stages = nomination_stage_evidence(nomination_comparisons)
    result['nomination_stage_evidence'] = stages
    if plan.get('vote_protocol') != 'matched-four-v1' and complete and len(source_means) >= (cfg['max_sources'] if final else cfg['first_sources']):
        if not changed:
            result.update(status='rejected', reason='no_action_quality_change')
        elif any(v[0] < 0 for v in stage_means.values()):
            result.update(status='rejected', reason='observed_stage_regression')
        elif mean[0] < 0 or mean[2] > 1e-12 or min(v[0] for v in source_means.values()) < -cfg['max_source_regression']:
            result.update(status='rejected', reason='observed_direction_or_source_regression')
        elif observed_noms and all(c['outcome'] == 'regressed' for c in observed_noms):
            result.update(status='rejected', reason='all_observed_nominations_regressed')
        elif any(s['observed_sources'] < required_sources or s['coverage'] < .5 for s in stages.values()):
            result['reason'] = 'insufficient_paired_nomination_evidence'
        elif any(s['balance'] < -1e-12 for s in stages.values()):
            result.update(status='rejected', reason='stage_nomination_regression')
        elif mean[0] > cfg['min_delta'] and mean[1] > cfg['min_delta']:
            result.update(status='accepted', reason='paired_rank_improvement_without_severe_regression')
        else:
            result['reason'] = 'borderline_expand_to_predeclared_sources'
    if plan.get('vote_protocol') == 'matched-four-v1':
        if not final:
            raise ValueError('Matched node voting uses one frozen panel')
        votes = {}
        margin = plan['win_margin'] + 1e-12
        for checkpoint in sorted({key[1] for key in allowed}):
            keys = sorted(key for key in allowed if key[1] == checkpoint)
            delta = (float(np.mean([vote_deltas[key] for key in keys]))
                     if all(key in vote_deltas for key in keys) else None)
            votes[checkpoint] = dict(source=keys[0][0], stage=planned_stages[keys[0]], delta=delta,
                outcome='unknown' if delta is None else 'win' if delta > margin else
                'loss' if delta < -margin else 'tie')
        wins = sum(v['outcome'] == 'win' for v in votes.values())
        accepted = len(votes) == 4 and wins >= 3
        result.update(protocol=plan['vote_protocol'], win_margin=plan['win_margin'], votes=votes, wins=wins,
            nomination_required_sources=0,
            status='accepted' if accepted else 'rejected',
            reason='three_of_four_sampling_wins' if accepted else
                   'insufficient_matching_nodes' if len(votes) < 4 else 'fewer_than_three_sampling_wins')
    if final and result['status'] == 'pending':
        result.update(status='rejected', reason='validation_budget_exhausted:'+result['reason'],
                      evidence_conclusion='inconclusive_not_demonstrated_harm')
    ident = stable_hash(result)
    with store.transaction():
        existing = store.db.execute('SELECT payload FROM gates WHERE id=?', (ident,)).fetchone()
        if existing:
            return json.loads(existing[0])
        current = store.candidate(edit_id)
        if current['status'] not in ('candidate', 'pending') or store.active != edit['parent_hash']:
            raise ValueError('Candidate resolved or active parent changed')
        store.db.execute('INSERT INTO gates VALUES (?,?)', (ident, encoded(result)))
        store.db.execute('UPDATE edits SET status=? WHERE id=?', (result['status'], edit_id))
        if result['status'] == 'accepted':
            store.db.execute('UPDATE meta SET value=? WHERE key="active"', (edit['candidate_hash'],))
    return result
