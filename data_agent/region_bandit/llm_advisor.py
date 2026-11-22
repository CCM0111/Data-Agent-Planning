"""One batch recommendation request with strict evidence and score validation."""
from __future__ import annotations

import inspect
import json
import math
from ..execution_budget import TaskDeadlineExceeded
from .advisor_repair import with_one_output_repair


@with_one_output_repair
async def score_actions_with_api(contexts, cases, request_fn, *, config=None, comparison_scope=None, skills=None):
    config = config or {}
    config = config.get('region_bandit', config)
    config = config.get('advisor', config)
    contexts = list(contexts.values()) if isinstance(contexts, dict) else list(contexts)
    skills = list(skills or [])
    ids = [c['action_id'] for c in contexts]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('Advisor action IDs must be distinct')
    scopes = {c.get('comparison_scope', comparison_scope or 'initial') for c in contexts}
    if len(scopes) != 1:
        raise ValueError('Advisor action contexts must share one comparison scope')
    scope = scopes.pop()
    if scope not in ('initial', 'mcts_parent_anchor') or (comparison_scope is not None and scope != comparison_scope):
        raise ValueError('Unknown or inconsistent advisor comparison scope')
    reward_instruction = (
        'pairwise ranking accuracy on the three or five methods actually evaluated at full scale. '
        'A three-method reward provides no evidence about the two unobserved methods. Use ')
    if scope == 'mcts_parent_anchor':
        reward_instruction = (
            'pairwise ranking accuracy on the already-full-trained parent and one or two successfully '
            'full-trained children: two or three observed methods. The parent low score is recomputed '
            'on the same round coreset as all children; its unchanged full result is reused. '
            'Unobserved children have no full labels. With a single informative pair, normalized '
            'accuracy is 0, 0.5, or 1; confidence is a separate learning weight, not another reward. '
            'This conditional comparison cannot certify the ranking of all generated children. '
            'The supplied regional discovery diagnostics describe the initial P-to-Q proxies, '
            'not fresh measurements of the current methods. Search summaries may contain already '
            'observed validation feedback; never infer current full outcomes or use test results. '
            'Respect the supplied quota constraints and sample-reuse rules. Use ')
    context_keys = ('action_id', 'action_type', 'problem_type', 'metric',
                    'metric_params', 'task_summary', 'method_summary', 'budget_summary', 'region_summary', 'action_summary')
    if scope == 'mcts_parent_anchor':
        context_keys += ('comparison_scope', 'search_summary')
    scores = {identifier: 0.0 for identifier in ids}
    request = {'instruction': (
        'Rate only the supplied feasible sampling actions, relative to baseline. Use training-side '
        'facts to favor accurate method rankings. The eventual reward is full-score-gap-weighted '
        + reward_instruction +
        'facts and listed historical evidence. These scores are recommendations, not probabilities '
        'or measured gains. Return one entry per action, including baseline with score 0. '
        'Return JSON {"scores":[{"action_id":...,"score":...,"support":...,"risk":...,'
        '"evidence_ids":[...]}]}. Allowed scores: -1,-0.5,0,0.5,1. '
        'Cite only record_id values supplied here; an empty evidence_ids is allowed.'),
        'contexts': [{key: c[key] for key in context_keys
                     if key in c} for c in contexts], 'historical_cases': list(cases)}
    if any(c.get('action_summary', {}).get('sampling_mode') == 'local_edit' for c in contexts):
        request['instruction'] += (' Actions minimally edit each original coreset, preserving original group/class counts. '
            'repeat_effects gives the actual before/after counts and row turnover separately for each seed. '
            'A seed already meeting the target is unchanged; no seed loses region rows. '
            'Score expected mean ranking accuracy across repeats, allowing unchanged repeats. '
            'New rows use marginal coverage accounting for currently retained rows. Consult deletion_rule: '
            'retention_score means sequentially remove an outside-region row minimizing density/hardness plus '
            'current unique coverage contribution, add one new row, and update coverage before the next swap; '
            'uniform means random outside-region deletions. '
            'No more than the declared replacement cap is allowed. These are not whole-coreset redraws.')
    if skills:
        request['experience_skills'] = skills
        request['instruction'] += (' Experience skills are conditional summaries, not established laws. '
            'Cite their IDs when used; inspect their counterexamples and do not invent measured gains.')
    if config.get('experience_skills', False):
        # All actions share the same task and five-method bank. Send those facts
        # once rather than repeating a large bank for every feasible region.
        shared = {}
        for key in ('problem_type', 'metric', 'metric_params', 'task_summary', 'method_summary', 'budget_summary'):
            if all(key in c for c in request['contexts']) and all(
                    c[key] == request['contexts'][0][key] for c in request['contexts']):
                shared[key] = request['contexts'][0][key]
                for c in request['contexts']:
                    del c[key]
        request['shared_context'] = shared
        request['instruction'] += (' shared_context applies identically to every action in contexts. '
            'For EVERY entry, support and risk MUST each be a STRING containing one short sentence '
            'of reasoning, not a numeric confidence, list, boolean or null. Only score is numeric.')
        request['historical_cases'] = [{
            **{key: c.get(key) for key in ('record_id', 'dataset_group_id', 'distance', 'action_type', 'signal_type')},
            'feedback': {key: c.get('feedback', {}).get(key) for key in
                         ('reward', 'status', 'paired_comparison', 'observed_method_count')},
            'region_summary': c.get('context', {}).get('region_summary', {}),
            'action_summary': c.get('context', {}).get('action_summary', {}),
        } for c in cases]
    result = {'scores': scores, 'api_available': 0, 'request': request,
              'response': None, 'error': None, 'calls': 0}
    if len(contexts) == 1 or not config.get('enabled', True) or request_fn is None:
        result['skipped_reason'] = 'baseline_only' if len(contexts) == 1 else 'advisor_disabled'
        return result
    try:
        result['calls'] = 1
        response = request_fn(json.dumps(request, ensure_ascii=False, sort_keys=True, allow_nan=False))
        if inspect.isawaitable(response):
            response = await response
        try:
            json.dumps(response, allow_nan=False)
            result['response'] = response
        except (TypeError, ValueError):
            # Keep the invalid reply inspectable without making the fallback's
            # own JSON artifact fail (e.g. NaN, bytes, unexpected objects).
            result['response'] = repr(response)
        if isinstance(response, str):
            response = json.loads(response)
        if not isinstance(response, dict) or not isinstance(response.get('scores'), list):
            raise ValueError('Advisor response must contain a scores array')
        entries = response['scores']
        if len(entries) != len(ids) or {e.get('action_id') for e in entries if isinstance(e, dict)} != set(ids):
            raise ValueError('Advisor must return every action exactly once')
        evidence = {c['record_id'] for c in [*cases, *skills]}
        baselines = {c['action_id'] for c in contexts if c['action_type'] == 'baseline'}
        validated = {}
        allowed = set(config.get('allowed_scores', [-1, -.5, 0, .5, 1]))
        if not allowed.issubset({-1, -.5, 0, .5, 1}):
            raise ValueError('Invalid allowed advisor scores')
        for entry in entries:
            score = entry.get('score')
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score not in allowed:
                raise ValueError('Invalid advisor score')
            if entry['action_id'] in baselines and score != 0:
                raise ValueError('Baseline score must equal zero')
            references = entry.get('evidence_ids')
            if not isinstance(references, list) or any(not isinstance(v, str) or v not in evidence for v in references):
                raise ValueError('Advisor cited an unknown historical evidence ID')
            if not isinstance(entry.get('support'), str) or not isinstance(entry.get('risk'), str):
                raise ValueError('Advisor support and risk must be strings')
            validated[entry['action_id']] = float(score)
        result.update(scores=validated, api_available=1)
    except TaskDeadlineExceeded:
        raise
    except Exception as error:
        if getattr(error, 'budget_exhausted', False) or type(error).__name__ == 'SearchBudgetExceeded':
            raise
        result['error'] = type(error).__name__ + ': ' + str(error)
    return result
