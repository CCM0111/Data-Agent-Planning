"""One bounded output repair using the original context and actual error."""
from functools import wraps
import json


def with_one_output_repair(advisor_fn):
    @wraps(advisor_fn)
    async def wrapped(contexts, cases, request_fn, *, previous_result=None, **kwargs):
        contexts = contexts if isinstance(contexts, dict) else list(contexts)
        cases = list(cases)
        if kwargs.get('skills') is not None:
            kwargs['skills'] = list(kwargs['skills'])
        first = (previous_result if previous_result is not None else
                 await advisor_fn(contexts, cases, request_fn, **kwargs))
        if (first.get('api_available') or first.get('skipped_reason') or not first.get('error')
                or first.get('calls') != 1 or first.get('repair_attempted') or request_fn is None):
            return first
        errors = [{'attempt': 1, 'error': first['error'], 'response': first.get('response')}]

        def repair_request(prompt):
            original = json.loads(prompt)
            action_contexts = contexts.values() if isinstance(contexts, dict) else contexts
            allowed_ids = [c['action_id'] for c in action_contexts]
            original['output_repair'] = {
                'attempt': 1, 'maximum_repairs': 1,
                'previous_response': first.get('response'), 'validation_error': first['error'],
                'allowed_action_ids': allowed_ids,
                'instruction': 'The previous API call failed or its answer failed validation. '
                    'Return a complete corrected answer using the original required JSON schema. '
                    'Use ONLY the exact action_id values listed in this request; do not combine '
                    'a region ID with a different action prefix. Scores, evidence IDs and all other '
                    'fields must satisfy the original constraints. Do not return a patch or prose. '
                    'Reconsider from the supplied evidence if the prior action is unavailable.'}
            return request_fn(json.dumps(original, ensure_ascii=False, sort_keys=True, allow_nan=False))

        second = dict(await advisor_fn(contexts, cases, repair_request, **kwargs))
        second.update(calls=first['calls']+second.get('calls', 0),
                      repair_attempted=True, repair_succeeded=bool(second.get('api_available')),
                      initial_error=first['error'], initial_response=first.get('response'),
                      attempt_errors=errors)
        if second.get('error'):
            second['attempt_errors'].append({'attempt': 2, 'error': second['error'],
                                             'response': second.get('response')})
        return second
    return wrapped
