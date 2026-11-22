"""Training-side nearest historical cases; retrieval never supplies labels to ridge."""
from __future__ import annotations

import math

VERSION_KEYS = ('feature_schema_version', 'sampler_version', 'reward_version', 'policy_version',
                'reward_protocol_hash')


def compatibility(identity):
    result = {key: identity.get(key) for key in (*VERSION_KEYS, 'problem_type', 'metric_signature')}
    if result['metric_signature'] is None:
        from .context import canonical_metric
        result['metric_signature'] = canonical_metric(identity)[2]
    return result


def compatible(first, second):
    return compatibility(first) == compatibility(second)


def retrieve_memory(context, records, *, identity, excluded_dataset_groups=(), max_cases=5, max_distance=.35):
    if max_cases <= 0:
        return []
    excluded = set(excluded_dataset_groups) | {identity['dataset_group_id']}
    query = context.get('retrieval_features', {})
    if not query:
        return []
    ranked = []
    for record in records:
        previous_identity = record.get('identity', record)
        group = previous_identity.get('dataset_group_id', record.get('dataset_group_id'))
        if group in excluded or not group or not compatible(identity, previous_identity):
            continue
        if record.get('mode', 'develop_online') != 'develop_online':
            continue
        previous = record.get('context', {}).get('retrieval_features', record.get('retrieval_features', {}))
        if set(previous) != set(query):
            continue
        differences = [abs(float(query[k]) - float(previous[k])) for k in sorted(query)]
        if not all(math.isfinite(d) for d in differences):
            continue
        distance = sum(differences) / len(differences)
        if distance > max_distance:
            continue
        is_region = record.get('kind') == 'region_feedback'
        match = (record.get('rule_types') == context.get('rule_types') if is_region else
                 record.get('action_type') == context.get('action_type'))
        ranked.append((not match, distance, str(record.get('record_id', record.get('event_id', ''))), group, record))
    ranked.sort(key=lambda item: item[:3])
    result, seen = [], set()
    for _, distance, record_id, group, record in ranked:
        if group in seen:
            continue
        seen.add(group)
        result.append({'record_id': record_id, 'dataset_group_id': group, 'distance': distance,
                       'kind': record.get('kind', 'action_feedback'),
                       'action_type': record.get('action_type'),
                       'executed_action': record.get('executed_action'),
                       'method_bank_hash': record.get('identity', {}).get('method_bank_hash'),
                       'frozen_config': record.get('frozen_config', {}),
                       'signal_type': record.get('signal_type', 'executed_action_reward'),
                       'feedback': record.get('feedback', {}),
                       'region': record.get('region', {}),
                       'context': record.get('context', {})})
        if len(result) >= min(5, int(max_cases)):
            break
    return result
