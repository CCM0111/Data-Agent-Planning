"""Versioned fit/inference caches around the existing isolated model executor."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from zipfile import BadZipFile

import numpy as np

from ..coreset import file_hash
from ..execution_budget import TaskDeadlineExceeded, bounded_timeout, check_deadline
from ..multi_proxy import run_fit
from .schemas import clean, environment_identity, read_json, stable_hash, write_json


EXECUTION_VERSION = 'regional_frozen_execution_v1'
FIT_FILES = ('fit_report.json', 'predictions.npz', 'fitted.joblib', 'unfitted.joblib')
PREDICT_FILES = ('prediction_report.json', 'predictions.npz')


@contextmanager
def _locked(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.cache.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield
        fcntl.flock(lock, fcntl.LOCK_UN)


def _failure_kind(error, directory):
    if isinstance(error, TaskDeadlineExceeded):
        return 'infrastructure_failed', f'{type(error).__name__}: {error}'
    path = Path(directory)/'failure.json'
    try:
        detail = read_json(path) if path.is_file() else {}
    except (OSError, ValueError):
        detail = {}
    message = str(detail.get('error', error))
    infra = ('FileNotFoundError:', 'PermissionError:', 'OSError:', 'IOError:',
             'MemoryError:', 'ModuleNotFoundError:', 'ImportError:')
    if isinstance(error, (OSError, MemoryError)) or message.startswith(infra):
        return 'infrastructure_failed', message
    if 'exceeded' in message.lower() and ('budget' in message.lower() or 'seconds' in message.lower()):
        return 'execution_failed', message
    # An uncaught worker exception is a confirmed fixed-method execution failure.
    # Missing worker output / signal death / launcher failures are not model labels.
    if detail and (re.match(r'^[A-Za-z_]+(?:Error|Exception):', message) or detail.get('error_type')):
        return 'execution_failed', message
    return 'infrastructure_failed', message


def _artifacts(directory, names):
    result = {}
    for name in names:
        path = Path(directory)/name
        if not path.is_file():
            return None
        result[name] = {'path': str(path), 'sha256': file_hash(path)}
    report_name = names[0]
    try:
        if read_json(Path(directory)/report_name).get('status') != 'success':
            return None
        with np.load(Path(directory)/'predictions.npz', allow_pickle=False) as values:
            predictions = values['predictions'];positions = values['row_positions']
            if predictions.ndim != 1 or not np.array_equal(positions, np.arange(len(predictions))):
                return None
    except (OSError, ValueError, KeyError, EOFError, BadZipFile, json.JSONDecodeError):
        return None
    return result


def _verified_artifacts(directory, names, expected=None):
    actual = _artifacts(directory, names)
    if actual is None:
        return None
    if expected and any(actual[name]['sha256'] != expected[name]['sha256'] for name in names):
        return None
    return actual


def _promote(source, out, names):
    for name in names:
        temporary = out/(name+'.promoting')
        # Cache identity is byte-based. Shared mounts may deny timestamp/chmod
        # preservation even when artifact data can be written successfully.
        shutil.copyfile(source/name, temporary)
        temporary.replace(out/name)


def _task_identity(task):
    result = clean(task)
    # Initial and reference pipeline paths do not by themselves identify bytes.
    for key in ('initial_pipeline_path', 'reference_pipeline_path'):
        if task.get(key):
            result[key+'_actual_sha256'] = file_hash(task[key])
    return result


def _inputs(code, train, predict, task, classes, budget, seed, batch_size, identity):
    core = {'execution_version': EXECUTION_VERSION, 'code_sha256': file_hash(code),
            'train_sha256': file_hash(train), 'task': _task_identity(task),
            'classes': clean(classes), 'budget': clean(budget), 'seed': int(seed),
            'batch_size': int(batch_size), 'environment': environment_identity(),
            'identity': clean(identity),
            'runtime_sha256': file_hash(Path(__file__).parents[1]/'multi_proxy_runtime.py'),
            'model_registry_sha256': file_hash(Path(__file__).parents[1]/'model_registry.py'),
            'io_contract_sha256': file_hash(Path(__file__).parents[1]/'multi_proxy_io.py')}
    training_key = stable_hash(core)
    full = {**core, 'predict_sha256': file_hash(predict)}
    return training_key, stable_hash(full), full


def _attempt_records(out):
    folder = out/'attempts';folder.mkdir(exist_ok=True)
    result = []
    for path in sorted(folder.glob('attempt_*.request.json')):
        record = read_json(path)
        record['_request_path'] = str(path)
        record['_directory'] = str(path.with_name(path.name.removesuffix('.request.json')))
        result.append(record)
    return result


def _next_attempt(out, records, request):
    existing = [int(Path(record['_directory']).name.split('_')[-1]) for record in records]
    # A crash can leave an orphan directory; never reuse or overwrite it.
    existing += [int(path.name.split('_')[-1]) for path in (out/'attempts').glob('attempt_*') if path.is_dir()]
    number = max(existing, default=0)+1
    destination = out/'attempts'/f'attempt_{number:04d}'
    write_json(destination.with_name(destination.name+'.request.json'), request)
    return destination


def _result(status, out, *, report=None, cache_key=None, attempts=None, cache_hit=False,
            error=None, fit_calls=0, prediction_calls=0, **extra):
    result = {'status': status, 'report': report, 'cache_key': cache_key,
              'cache_hit': cache_hit, 'attempts': attempts or [], 'error': error,
              'fit_calls_this_call': fit_calls, 'prediction_calls_this_call': prediction_calls,
              'artifacts': {name: str(out/name) for name in FIT_FILES if (out/name).is_file()}, **extra}
    return clean(result)


def _invoke_worker(request_path, out, timeout_seconds):
    # The shortened wait is an execution constraint, not part of the model or
    # inference cache identity. Never launch a new worker after the task expires.
    check_deadline()
    environment = {key: value for key, value in os.environ.items() if 'API_KEY' not in key and not key.endswith('_TOKEN')}
    environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    with (out/'execution.log').open('w') as log:
        with subprocess.Popen([sys.executable, str(Path(__file__).with_name('prediction_worker.py')), str(request_path)],
                              cwd=out, env=environment, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                process.wait(timeout=bounded_timeout(timeout_seconds))
            except BaseException as error:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                if isinstance(error, subprocess.TimeoutExpired):
                    # Cleanup happens before converting global expiry to a
                    # resumable infrastructure outcome. Fixed method timeouts
                    # retain their original exception and classification.
                    check_deadline()
                raise
    check_deadline()
    return process.returncode


def inspect_proxy_importance(code, fitted, original_columns, *, out=None, timeout_seconds=30):
    """Best-effort isolated inspection; never import generated code in the parent.

    An optional output directory retains the request/result/error audit. Without
    one, the temporary worker files are removed after returning the feature list.
    """
    if out is None:
        with TemporaryDirectory(prefix='regional-importance-') as temporary:
            return inspect_proxy_importance(code, fitted, original_columns, out=temporary,
                                            timeout_seconds=timeout_seconds)
    output = Path(out).resolve();output.mkdir(parents=True, exist_ok=True)
    try:
        code, fitted = Path(code).resolve(), Path(fitted).resolve()
        request = {'operation': 'inspect_importance', 'code': str(code), 'fitted': str(fitted),
                   'original_columns': list(original_columns), 'output': str(output),
                   'input_hashes': {'code': file_hash(code), 'fitted': file_hash(fitted)}}
        path = output/'worker_context.json';write_json(path, request)
        (output/'importance.json').unlink(missing_ok=True)
        if _invoke_worker(path, output, timeout_seconds) != 0:
            return []
        result = read_json(output/'importance.json')
        return result.get('importance', []) if result.get('status') == 'success' else []
    except TaskDeadlineExceeded:
        # A task interruption is not evidence that feature importance is absent.
        raise
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        write_json(output/'inspection_failure.json', {'error': f'{type(error).__name__}: {error}'})
        return []


def fit_cached(code, train, predict, task, classes, budget, seed, out, batch_size=1024,
               *, identity=None, infrastructure_retries=1):
    """Fit with immutable inputs; reuse an identical fit for new prediction rows.

    Successful attempts remain recoverable even if promotion or cache-marker
    writing was interrupted. No automatic code/parameter repair is performed.
    """
    out = Path(out).resolve()
    code, train, predict = [Path(path).resolve() for path in (code, train, predict)]
    if type(infrastructure_retries) is not int or infrastructure_retries < 0 or batch_size < 1:
        raise ValueError('Invalid retry or batch-size setting')
    with _locked(out):
        try:
            training_key, key, frozen = _inputs(code, train, predict, task, classes, budget, seed, batch_size, identity)
        except (OSError, MemoryError) as error:
            return _result('infrastructure_failed', out, error=f'{type(error).__name__}: {error}')
        records = _attempt_records(out)
        compatible = [record for record in records if record['training_key'] == training_key]
        previous = read_json(out/'cache.json') if (out/'cache.json').is_file() else None
        if previous and previous.get('cache_key') == key and previous.get('status') == 'ok':
            valid = _verified_artifacts(out, FIT_FILES, previous['artifacts'])
            if valid:
                return _result('ok', out, report=read_json(out/'fit_report.json'), cache_key=key,
                               attempts=previous.get('attempts'), cache_hit=True, fit_cache_hit=True)
        successful = []
        for record in compatible:
            directory = Path(record['_directory'])
            completion = read_json(directory/'completion.json') if (directory/'completion.json').is_file() else None
            valid = _verified_artifacts(directory, FIT_FILES, completion.get('artifacts') if completion else None)
            if valid:
                successful.append((record, directory, valid))
        exact = [item for item in successful if item[0]['cache_key'] == key]
        chosen = exact[-1] if exact else successful[-1] if successful else None
        attempts = [{'directory': record['_directory'], 'cache_key': record['cache_key'],
                     'status': read_json(Path(record['_directory'])/'outcome.json').get('status')
                     if (Path(record['_directory'])/'outcome.json').is_file() else 'interrupted_or_completed'}
                    for record in compatible]
        fit_calls = 0
        if chosen is None:
            # A confirmed method failure under these exact inputs is not retried.
            for record in reversed(compatible):
                outcome = Path(record['_directory'])/'outcome.json'
                if record['cache_key'] == key and outcome.is_file():
                    value = read_json(outcome)
                    if value['status'] == 'execution_failed':
                        return _result('execution_failed', out, cache_key=key, attempts=attempts,
                                       cache_hit=True, error=value['error'])
            request = {'cache_key': key, 'training_key': training_key, 'inputs': frozen}
            for _ in range(infrastructure_retries+1):
                try:
                    check_deadline()
                except TaskDeadlineExceeded as error:
                    return _result('infrastructure_failed', out, cache_key=key, attempts=attempts,
                                   error=f'{type(error).__name__}: {error}', fit_calls=fit_calls,
                                   budget_exhausted=True)
                destination = _next_attempt(out, _attempt_records(out), request)
                try:
                    fit_calls += 1
                    report = run_fit(code, train, predict, clean(task), clean(classes), clean(budget),
                                     int(seed), destination, int(batch_size))
                    if _inputs(code, train, predict, task, classes, budget, seed, batch_size, identity)[1] != key:
                        raise OSError('Frozen fit inputs changed while worker was running')
                    valid = _artifacts(destination, FIT_FILES)
                    if valid is None:
                        raise OSError('Fit worker did not produce a complete successful artifact set')
                    write_json(destination/'completion.json', {'cache_key': key, 'artifacts': valid})
                    write_json(destination/'outcome.json', {'status': 'ok'})
                    attempts.append({'directory': str(destination), 'cache_key': key, 'status': 'ok'})
                    chosen = (request, destination, valid)
                    break
                except TaskDeadlineExceeded as error:
                    detail = f'{type(error).__name__}: {error}'
                    destination.mkdir(parents=True, exist_ok=True)
                    write_json(destination/'outcome.json', {'status': 'infrastructure_failed',
                               'error': detail, 'budget_exhausted': True})
                    attempts.append({'directory': str(destination), 'cache_key': key,
                                     'status': 'infrastructure_failed', 'error': detail})
                    return _result('infrastructure_failed', out, cache_key=key, attempts=attempts,
                                   error=detail, fit_calls=fit_calls, budget_exhausted=True)
                except Exception as error:
                    status, detail = _failure_kind(error, destination)
                    destination.mkdir(parents=True, exist_ok=True)
                    write_json(destination/'outcome.json', {'status': status, 'error': detail})
                    attempts.append({'directory': str(destination), 'cache_key': key, 'status': status, 'error': detail})
                    if status == 'execution_failed':
                        return _result(status, out, cache_key=key, attempts=attempts, error=detail, fit_calls=fit_calls)
            if chosen is None:
                return _result('infrastructure_failed', out, cache_key=key, attempts=attempts,
                               error=attempts[-1]['error'], fit_calls=fit_calls)
        record, directory, _ = chosen
        prediction_calls = 0
        if record['cache_key'] == key:
            _promote(directory, out, FIT_FILES)
            report = read_json(out/'fit_report.json')
        else:
            prediction = predict_cached(code, directory/'fitted.joblib', predict, task, classes,
                                        out/'inference'/key, timeout_seconds=budget.get('timeout_seconds', 300),
                                        batch_size=batch_size, identity=identity)
            prediction_calls = prediction.get('prediction_calls_this_call', 0)
            if prediction['status'] != 'ok':
                return _result(prediction['status'], out, cache_key=key, attempts=attempts,
                               error=prediction['error'], fit_calls=fit_calls, prediction_calls=prediction_calls,
                               fit_cache_hit=True, budget_exhausted=prediction.get('budget_exhausted', False))
            _promote(directory, out, ('fitted.joblib', 'unfitted.joblib'))
            _promote(Path(prediction['artifact']).parent, out, ('predictions.npz',))
            report = {**read_json(directory/'fit_report.json'),
                      **{k: v for k, v in prediction['report'].items() if k != 'fit_calls'},
                      'reused_training_artifact': str(directory/'fitted.joblib')}
            write_json(out/'fit_report.json', report)
        artifacts = _artifacts(out, FIT_FILES)
        write_json(out/'cache.json', {'status': 'ok', 'cache_key': key, 'training_key': training_key,
                                     'inputs': frozen, 'artifacts': artifacts, 'attempts': attempts})
        return _result('ok', out, report=report, cache_key=key, attempts=attempts,
                       cache_hit=not fit_calls and record['cache_key'] == key, fit_cache_hit=not bool(fit_calls),
                       fit_calls=fit_calls, prediction_calls=prediction_calls)


def predict_cached(code, fitted, predict, task, classes, out, *, timeout_seconds=300,
                   batch_size=1024, identity=None):
    """Fresh import/load/predict only; no factory or fitting in this process."""
    out = Path(out).resolve()
    code, fitted, predict = [Path(path).resolve() for path in (code, fitted, predict)]
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0 or batch_size < 1:
        raise ValueError('Invalid prediction timeout or batch size')
    with _locked(out):
        try:
            hashes = {name: file_hash(path) for name, path in [('code', code), ('fitted', fitted), ('predict', predict)]}
            frozen = {'version': EXECUTION_VERSION, 'input_hashes': hashes, 'task': _task_identity(task),
                      'classes': clean(classes), 'batch_size': batch_size, 'timeout_seconds': timeout_seconds,
                      'environment': environment_identity(),
                      'identity': clean(identity), 'worker_sha256': file_hash(Path(__file__).with_name('prediction_worker.py'))}
            key = stable_hash(frozen)
        except (OSError, MemoryError) as error:
            return {'status': 'infrastructure_failed', 'cache_hit': False, 'artifact': None,
                    'error': f'{type(error).__name__}: {error}', 'prediction_calls_this_call': 0}
        metadata = read_json(out/'cache.json') if (out/'cache.json').is_file() else None
        if metadata and metadata.get('cache_key') == key:
            valid = _verified_artifacts(out, PREDICT_FILES, metadata['artifacts'])
            if valid:
                return {'status': 'ok', 'cache_hit': True, 'cache_key': key,
                        'artifact': str(out/'predictions.npz'), 'report': read_json(out/'prediction_report.json'),
                        'error': None, 'prediction_calls_this_call': 0}
        request_path = out/'worker_context.json'
        if (metadata is None or metadata.get('cache_key') != key) and request_path.is_file() and read_json(request_path).get('cache_key') == key:
            valid = _verified_artifacts(out, PREDICT_FILES)
            if valid:
                write_json(out/'cache.json', {'cache_key': key, 'inputs': frozen, 'artifacts': valid})
                return {'status': 'ok', 'cache_hit': True, 'cache_key': key,
                        'artifact': str(out/'predictions.npz'), 'report': read_json(out/'prediction_report.json'),
                        'error': None, 'prediction_calls_this_call': 0, 'recovered_completed_worker': True}
        try:
            check_deadline()
        except TaskDeadlineExceeded as error:
            return {'status': 'infrastructure_failed', 'cache_hit': False, 'cache_key': key,
                    'artifact': None, 'error': f'{type(error).__name__}: {error}',
                    'prediction_calls_this_call': 0, 'budget_exhausted': True}
        for name in (*PREDICT_FILES, 'failure.json'):
            (out/name).unlink(missing_ok=True)
        request = {'code': str(code), 'fitted': str(fitted), 'predict': str(predict), 'task': clean(task),
                   'classes': clean(classes), 'batch_size': int(batch_size), 'output': str(out),
                   'input_hashes': hashes, 'cache_key': key}
        write_json(request_path, request)
        started = time.monotonic()
        try:
            returncode = _invoke_worker(request_path, out, timeout_seconds)
            if returncode:
                raise RuntimeError(f'Prediction worker exited {returncode}')
            if any(file_hash(path) != hashes[name] for name, path in [('code', code), ('fitted', fitted), ('predict', predict)]):
                raise OSError('Frozen prediction inputs changed while worker was running')
            artifacts = _artifacts(out, PREDICT_FILES)
            if artifacts is None:
                raise OSError('Prediction worker did not produce complete successful artifacts')
            write_json(out/'cache.json', {'cache_key': key, 'inputs': frozen, 'artifacts': artifacts})
            return {'status': 'ok', 'cache_hit': False, 'cache_key': key, 'artifact': str(out/'predictions.npz'),
                    'report': read_json(out/'prediction_report.json'), 'error': None,
                    'prediction_calls_this_call': 1, 'process_wall_seconds': time.monotonic()-started}
        except TaskDeadlineExceeded as error:
            return {'status': 'infrastructure_failed', 'cache_hit': False, 'cache_key': key,
                    'artifact': None, 'error': f'{type(error).__name__}: {error}',
                    'prediction_calls_this_call': 1, 'budget_exhausted': True}
        except subprocess.TimeoutExpired:
            error = f'Prediction exceeded fixed budget of {timeout_seconds:g} seconds'
            write_json(out/'failure.json', {'error': error, 'error_type': 'TimeoutError'})
            return {'status': 'execution_failed', 'cache_hit': False, 'cache_key': key,
                    'artifact': None, 'error': error, 'prediction_calls_this_call': 1}
        except Exception as error:
            status, detail = _failure_kind(error, out)
            return {'status': status, 'cache_hit': False, 'cache_key': key,
                    'artifact': None, 'error': detail, 'prediction_calls_this_call': 1}
