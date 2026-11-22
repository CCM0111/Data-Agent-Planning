"""Small, evidence-linked conditional experience cards; no synthetic reward labels."""
from collections import defaultdict
import math
from .schemas import stable_hash


def build_skill_cards(cases, *, max_cards=3):
    """Only caller-retrieved, compatible, other-source cases enter these cards.

    Three repeats of one dataset remain one piece of evidence. No causal claim or
    statistical confidence is inferred from a small, adaptively collected sample.
    """
    groups = defaultdict(dict)
    for case in cases:
        if case.get('kind') != 'action_feedback':
            continue
        paired = case.get('feedback', {}).get('paired_comparison', {})
        delta = paired.get('delta_weighted_accuracy')
        if not isinstance(delta, (int, float)) or not math.isfinite(delta):
            continue
        ctx = case.get('context', {})
        action = case.get('action_type')
        if action == 'baseline':
            continue  # no adjustment is not evidence of an effective intervention
        rules = tuple(sorted(ctx.get('rule_types') or []))
        key = (action, rules)
        groups[key][case['dataset_group_id']] = (case, float(delta))
    cards = []
    for (action, rules), sources in groups.items():
        rows = list(sources.values())
        positive = sum(delta > 1e-8 for _, delta in rows)
        negative = sum(delta < -1e-8 for _, delta in rows)
        ids = sorted(case['record_id'] for case, _ in rows)
        exemplar = min((case for case, _ in rows), key=lambda c: c.get('distance', 1))
        ctx = exemplar.get('context', {})
        consistent = len(rows) >= 3 and (positive == len(rows) or negative == len(rows))
        card = {
            'record_id': 'skill:' + stable_hash(ids)[:20],
            'status': 'provisional_pattern' if consistent else 'case_hypothesis',
            'applicability': {'action_type': action, 'rule_types': list(rules),
                'problem_type': ctx.get('problem_type'), 'metric': ctx.get('metric'),
                'nearest_case_region': ctx.get('region_summary', {}),
                'nearest_case_action': ctx.get('action_summary', {})},
            'distinct_sources': len(rows), 'positive_sources': positive,
            'negative_sources': negative, 'tied_sources': len(rows)-positive-negative,
            'mean_delta_weighted_accuracy': sum(d for _, d in rows)/len(rows),
            'evidence_ids': ids,
            'observations': [{'record_id': c['record_id'],
                'source': c['dataset_group_id'],
                **c['feedback']['paired_comparison']} for c, _ in rows],
            'guidance': ('Consider this action only for similar training-side conditions; '
                'compare the supplied positive and negative paired outcomes. Observations '
                'are conditional associations, not a guaranteed improvement or causal effect.'),
            'distance': exemplar.get('distance', 1),
        }
        cards.append(card)
    cards.sort(key=lambda c: (c['distance'], -c['distinct_sources'], c['record_id']))
    chosen = cards[:max_cards]
    negative = next((c for c in cards if c['negative_sources']), None)
    if negative and chosen and negative not in chosen:
        chosen[-1] = negative
    # Compact JSON bounded by characters, without cutting objects or evidence IDs.
    import json
    for card in chosen:
        for key in ('nearest_case_region', 'nearest_case_action'):
            value = card['applicability'][key]
            if len(json.dumps(value, ensure_ascii=False)) > 800:
                card['applicability'][key] = {'see_evidence_id': card['evidence_ids'][0]}
    while len(chosen) > 1 and len(json.dumps(chosen, ensure_ascii=False)) > 6000:
        # Retain a negative card when one is available.
        removable = next((i for i in reversed(range(len(chosen))) if not chosen[i]['negative_sources']), len(chosen)-1)
        chosen.pop(removable)
    return chosen
