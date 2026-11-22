"""Two persistent regeneration chances per method, shared across failure stages."""
import json
from pathlib import Path

from .coreset import write_json


class RegenerationExhausted(RuntimeError):
    pass


def eligible(error):
    # Credentials, transport failures and local IO are not method defects.
    if getattr(error, 'status_code', None) is not None:
        return False
    if isinstance(error, (OSError, TimeoutError, ConnectionError)):
        return False
    if type(error).__name__ in {'APIConnectionError', 'APITimeoutError'}:
        return False
    if isinstance(error, RegenerationExhausted):
        return False
    return isinstance(error, (ValueError, KeyError, TypeError, SyntaxError, RuntimeError))


async def regenerate(method_id, stage, error, prompt, directory, ledger_path,
                     model, costs, pricing, validate, *, resume=False, last_chance=None):
    from .multi_proxy import llm_json
    ledger_path = Path(ledger_path)
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'methods': {}}
    record = ledger['methods'].setdefault(str(method_id), {'used': 0, 'errors': [], 'requests': []})
    if resume:
        from .method_repair import reconcile_interrupted_requests
        reconcile_interrupted_requests(directory)
        for request in reversed(record['requests']):
            artifact = Path(request.get('artifact', ''))
            if request.get('stage') != stage or artifact.parent.resolve() != Path(directory).resolve():
                continue
            if artifact.is_file() and request.get('status') in ('success', 'running', 'interrupted'):
                usage_path = artifact.with_suffix('.usage.json')
                usage = json.loads(usage_path.read_text()) if usage_path.exists() else {}
                journal_path = artifact.with_suffix('.recovery.json')
                journal = json.loads(journal_path.read_text()) if journal_path.exists() else {}
                if (request.get('status') == 'success' or journal.get('status') == 'success'
                        or (usage.get('status') == 'success' and usage.get('finish_reason') == 'stop')):
                    response = json.loads(artifact.read_text())
                    try:
                        (last_chance[1] if last_chance and request.get('policy') == 'allow_slot_redesign' else validate)(response)
                    except (ValueError, KeyError, TypeError, SyntaxError) as invalid:
                        request.update(status='failed', error=str(invalid))
                    else:
                        request.update(status='success', recovered=True)
                        write_json(ledger_path, ledger)
                        return response
        for request in record['requests']:
            if request.get('status') == 'running':
                request.update(status='interrupted', interruption_kind='owner_exited', method_failure=False)
        write_json(ledger_path, ledger)
    if any(r.get('status') == 'running' for r in record['requests']):
        raise RuntimeError('Interrupted method regeneration requires reconciliation')
    record['errors'].append({'stage': stage, 'error': str(error)})
    write_json(ledger_path, ledger)
    while record['used'] < 2:
        record['used'] += 1
        path = Path(directory)/f'method_{method_id}_regeneration_{record["used"]}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        request = {'number': record['used'], 'stage': stage, 'artifact': str(path), 'status': 'running'}
        active_prompt, active_validate = prompt, validate
        if last_chance and record['used'] == 2:
            active_prompt, active_validate = last_chance
            request['policy'] = 'allow_slot_redesign'
        record['requests'].append(request)
        write_json(ledger_path, ledger)  # A failed/interrupted call still consumes its chance.
        try:
            response = await llm_json(model, active_prompt + '\n' + json.dumps({
                'method_id': method_id, 'regeneration_attempt': record['used'],
                'previous_errors': record['errors'],
                'previous_repair_feedback': record.get('repair_feedback', []),
                'instruction': ('Regenerate the complete initial plan to satisfy its schema and exact candidate count. '
                                'No methods have been frozen or evaluated. Do not use validation scores.'
                                if method_id == 'whole_plan' else
                                'Regenerate only this method to address these errors. Do not use validation scores or change other methods.')
            }, ensure_ascii=False), path, costs, pricing or {})
            active_validate(response)
        except Exception as failure:
            request.update(status='failed', error=f'{type(failure).__name__}: {failure}')
            record['errors'].append({'stage': stage, 'error': request['error']})
            if getattr(failure, 'repair_feedback', None):
                record.setdefault('repair_feedback', []).append(failure.repair_feedback)
            write_json(ledger_path, ledger)
            if not eligible(failure):
                raise
        else:
            request['status'] = 'success'
            write_json(ledger_path, ledger)
            return response
    raise RegenerationExhausted(f'Method {method_id} exhausted two regeneration chances; last error: {record["errors"][-1]["error"]}')


async def generate_one(method_id, stage, prompt, output, ledger_path, model, costs, pricing, validate):
    """Initial generation plus up to two method-specific regenerations."""
    from .multi_proxy import llm_json
    from .multi_proxy_trace import identity
    output = Path(output)
    selected = output.with_suffix('.selected.json')
    fingerprint = identity({'prompt': prompt, 'model': getattr(model, 'model_name', None)})
    if selected.exists():
        saved = json.loads(selected.read_text())
        if saved['identity'] != fingerprint:
            raise ValueError('Selected generation identity mismatch')
        validate(saved['response'])
        return saved['response']
    try:
        response = await llm_json(model, prompt, output, costs, pricing)
        validate(response)
    except Exception as error:
        if not eligible(error):
            raise
        response = await regenerate(method_id, stage, f'{type(error).__name__}: {error}',
            prompt, output.parent/'regenerations', ledger_path, model, costs, pricing, validate)
    write_json(selected, {'identity': fingerprint, 'response': response})
    return response
