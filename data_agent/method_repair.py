"""Bounded error-only repairs, with immutable attempts and complete task reruns."""
import copy
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import shutil

from .execution_budget import (TaskDeadlineExceeded, InfrastructureInterruption,
                               infrastructure_failure, check_deadline, bounded_timeout)

from .coreset import file_hash, write_json
from .multi_proxy_runtime import validate_code


def predictor_constructors(code):
    tree = ast.parse(code)
    aliases = {name.asname or name.name: name.name for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) for name in node.names}
    plain = {'LogisticRegression', 'Ridge', 'LinearRegression', 'ElasticNet',
             'Lasso', 'BayesianRidge', 'GaussianNB', 'BernoulliNB'}
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = aliases.get(node.func.id, node.func.id) if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
            if name in plain or name.endswith(('Classifier', 'Regressor')):
                names.add(name)
    return names


def preflight_methods(methods, task, cfg, output, stages=None, *, update_stage_parameters=False):
    failures = []
    sizes = task['training_context']
    definitions = [('proxy', sizes['proxy_train_rows'], cfg.scoring_budget),
                   ('evaluation', sizes['coreset_rows'], cfg.evaluation_budget)]
    if cfg.full_train_enabled:
        definitions.append(('full_train', sizes['fixed_train_rows'], cfg.full_train_budget))
    if stages is not None:
        if (not isinstance(stages, (list, tuple)) or not stages or len(set(stages)) != len(stages)
                or not set(stages) <= {stage for stage, _, _ in definitions}):
            raise ValueError('Preflight stages must be a nonempty subset of enabled stage names')
        if 'full_train' in stages and 'evaluation' not in stages:
            raise ValueError('Full-train preflight requires evaluation for the configuration reference')
        definitions = [definition for definition in definitions if definition[0] in stages]
    env = {k: v for k, v in os.environ.items() if 'API_KEY' not in k and not k.endswith('_TOKEN')}
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for i, method in enumerate(methods):
        check_deadline()
        folder = Path(output)/'preflight'/f'method_{i}'
        folder.mkdir(parents=True, exist_ok=True)
        context = {'code': method['code_path'], 'task': task, 'idea': method['idea'],
                   'family': method['family'], 'seed': cfg.model_seed,
                   'stages': definitions, 'output': str(folder),
                   'complete_stage_parameters': any(stage == 'full_train' for stage, _, _ in definitions)}
        write_json(folder/'context.json', context)
        timeout = bounded_timeout(60)
        try:
            with (folder/'execution.log').open('w') as log:
                result = subprocess.run([sys.executable, str(Path(__file__).with_name('method_preflight.py')),
                    str(folder/'context.json')], env=env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=timeout)
            if result.returncode:
                failure = json.loads((folder/'failure.json').read_text()) if (folder/'failure.json').exists() else {'error': f'Preflight subprocess exit {result.returncode}'}
                if result.returncode < 0 or not (folder/'failure.json').exists() or infrastructure_failure(failure.get('error', '')):
                    raise InfrastructureInterruption(failure['error'])
                failures.append({'method_id': i, **failure})
            else:
                report = json.loads((folder/'report.json').read_text())
                completed = report['stage_parameters']
                if completed != method['idea']['stage_parameters']:
                    write_json(folder/'stage_parameter_completion.json', {
                        'before': method['idea']['stage_parameters'], 'after': completed,
                        'basis': 'Missing other-stage values from its original unfitted factory',
                        'proxy_unchanged': True, 'fit_calls': 0})
                    if update_stage_parameters:
                        method['idea']['stage_parameters'] = completed
        except subprocess.TimeoutExpired:
            if timeout < 60:
                raise TaskDeadlineExceeded('Task deadline expired during method preflight')
            failures.append({'method_id': i, 'error': 'Preflight exceeded 60 seconds'})
        except (OSError, MemoryError, ImportError) as error:
            write_json(Path(output)/'preflight/summary.json', {'failures': failures, 'fit_calls': 0,
                'status': 'interrupted', 'failure_kind': 'infrastructure_interruption',
                'method_id': i, 'error': f'{type(error).__name__}: {error}'})
            raise
    write_json(Path(output)/'preflight/summary.json', {'failures': failures, 'fit_calls': 0})
    return failures


def failure_methods(run, manifest):
    stage = manifest.get('failure_stage')
    if manifest.get('method_failure') is False or infrastructure_failure(manifest.get('error', '')):
        return None, []
    if stage == 'method_preflight':
        path = run/'preflight/summary.json'
        summary = json.loads(path.read_text()) if path.exists() else {}
        return 'preflight', summary.get('failures', [])
    if stage not in {'proxy_fits', 'formal_evaluation', 'full_train_evaluation'}:
        return None, []
    match = re.search(r'method_(\d+)', manifest.get('active_execution', ''))
    return ('runtime', [{'method_id': int(match.group(1)), 'error': manifest['error']}]) if match else (None, [])


def _transform_failure_context(source, task, code, error):
    """Observed training-domain evidence for a failing log1p pipeline."""
    if 'log1p' not in code or not re.search(r'nan|infinity|infinite', str(error), re.I):
        return {}
    columns = task.get('numeric_columns') or []
    path = source/'proxy_train.csv'
    if not columns or not path.is_file():
        return {}
    import pandas as pd
    counts = {}
    for chunk in pd.read_csv(path, usecols=columns, chunksize=4096):
        for name in columns:
            values = pd.to_numeric(chunk[name], errors='coerce')
            count = int((values <= -1).sum())
            if count:
                counts[name] = counts.get(name, 0) + count
    if not counts:
        return {}
    return {'scope':'P only, before feature engineering', 'log1p_nonfinite_input_counts':
            dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]),
            'instruction':'If raw values from these columns reach log1p, use a domain-safe transform '
            'or omit it in the redesign. Imputation before log1p cannot fix finite values <= -1.'}


async def repair_cache(source, destination, failures, kind, model, pricing, state, state_path):
    from .multi_proxy import method_contract
    from .implementation_feedback import repair_diagnostic, component_interfaces
    from .multi_proxy_trace import identity, normalize_plan, validate_idea, validate_stage_parameters
    resuming = destination.exists()
    destination.mkdir(parents=True, exist_ok=True)
    completed = destination/'manifest.json'
    if completed.exists() and json.loads(completed.read_text()).get('status') == 'success':
        cache = destination/'cache'
        saved = json.loads((cache/'manifest.json').read_text())
        if saved.get('repair_source') != str(source):
            raise ValueError('Completed repair cache belongs to another attempt')
        for method in saved['methods']:
            if file_hash(method['code_path']) != method['code_sha256']:
                raise ValueError('Completed repair code changed')
        return cache
    previous = json.loads((source/'manifest.json').read_text())
    methods = copy.deepcopy(previous['methods'])
    calls = json.loads(completed.read_text()).get('llm_calls', []) if completed.exists() else []
    audit = {'status': 'running', 'llm_model': previous['llm_model'], 'llm_calls': calls,
             'source_run': str(source), 'kind': kind, 'changes': [],
             'policy': 'Local repair; MCTS initialization may redesign the failed slot on its last chance; fixed budgets, no validation scores; complete task rerun'}
    cache = destination/'cache'
    cache.mkdir(exist_ok=True)
    # The project data volume does not support chmod/copystat.
    for path in (source/'methods').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            target = cache/'methods'/path.relative_to(source/'methods')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    task = json.loads((source/'task_context.json').read_text())
    try:
        for failure in failures:
            i = failure['method_id']
            counts = state['repair_counts'].setdefault(str(i), {'preflight': 0, 'runtime': 0})
            path = cache/'methods'/f'method_{i}'/'method.py'
            old = path.read_text()
            domain = _transform_failure_context(source, task, old, failure['error'])
            prompt = (
                'Repair only the concrete implementation error below. Return JSON {"code":"complete Python source"}. '
                'Keep the original modeling family, exact main estimator class, FE intent, fixed parameter values, '
                'scaling rules, and both declared stage parameter dictionaries. Do not tune for performance, '
                'change resource budgets, add predictors, or use evaluation data. Correct unsupported constructor '
                'keywords/API mismatches or implementation defects using the installed sklearn interface. '
                'If an intended parameter is unsupported, remove only that invalid constructor argument. '
                'n_jobs is not supported by HistGradientBoosting. No validation metrics are supplied.\n'
                + method_contract(output_format=False) + '\n' + json.dumps({
                    'strategy': {k: methods[i][k] for k in ('name', 'family', 'idea')},
                    'task': task, 'error': failure['error'], 'original_code': old,
                    'repair_diagnostic': repair_diagnostic(old, failure['error']),
                    'installed_interfaces': component_interfaces(methods[i]['idea'], old),
                    **({'transform_diagnostic': domain} if domain else {}),
                    'sklearn_version': __import__('sklearn').__version__}, ensure_ascii=False))
            from .method_regeneration import regenerate
            from .code_repair_contract import audit_error_only_repair
            repair_audits = []
            def validate_response(response):
                if 'strategy' in response:
                    raise ValueError('Local repair cannot replace the strategy')
                code = response['code']
                validate_code(code)
                try:
                    original_predictors = predictor_constructors(old)
                except SyntaxError:
                    original_predictors = None
                if original_predictors is not None and original_predictors != predictor_constructors(code):
                    raise ValueError('Regeneration may not change predictor constructor classes')
                contract = audit_error_only_repair(old, code, failure['error'], idea=methods[i]['idea'])
                repair_audits.append(contract)
                write_json(destination/f'method_{i}_repair_contract.json', {'attempts': repair_audits})
                if contract['violations']:
                    error = ValueError('; '.join(contract['violations']))
                    error.repair_feedback = contract
                    raise error
            last_chance = None
            if (previous.get('configuration', {}).get('mcts_bootstrap_only') and
                    previous.get('failure_stage') in ('method_preflight', 'proxy_fits')):
                fields = tuple(k for k in ('name', 'family', 'reason', 'fe_config', 'model_config', 'idea') if k in methods[i])
                siblings = [m for j, m in enumerate(methods) if j != i]
                redesign_prompt = (
                    'Final chance for this failed initial slot. If the earlier fix failed, prefer a simpler, '
                    'well-supported replacement suited to this task and distinct from the other slots. '
                    'You may change the estimator, family, features and parameters. For a replacement return '
                    'JSON with strategy (name, family, idea), fe_config, model_config and complete code. '
                    'Inside idea return summary, model, feature_engineering, fixed_parameters, scaling_rules and stage_parameters. '
                    'Stage budgets are supplied by the runner. Follow the original stage parameter policy. '
                    'Use a new name; do not copy another method. Alternatively, for a precise local fix '
                    'return only {"code":"complete source"}; the original error-only audit then applies. '
                    'Prefer standard transformers with supported signatures. Do not use evaluation outcomes.\n'
                    + method_contract(output_format=False) + '\n' + json.dumps({
                        **({'transform_diagnostic': domain} if domain else {}),
                        'task': task, 'failed_strategy': {k: methods[i][k] for k in fields},
                        'original_code': old, 'other_slots': [dict(name=m['name'], family=m['family'],
                            model=m['idea']['model']) for m in siblings],
                        'allowed_families': ['linear','bagging','boosting','neural','neighbors','naive_bayes','tree'],
                        'sklearn_version': __import__('sklearn').__version__}, ensure_ascii=False))
                def validate_last(response):
                    if 'strategy' not in response:
                        return validate_response(response)
                    st = copy.deepcopy(response['strategy'])
                    if not isinstance(st, dict):
                        raise ValueError('Replacement strategy must be an object')
                    for key in ('fe_config', 'model_config'):
                        if key in response:
                            if key in st and identity(st[key]) != identity(response[key]):
                                raise ValueError('Conflicting replacement ' + key)
                            st[key] = response[key]
                    config = previous.get('configuration', {})
                    guarded = config.get('parameter_feedback') and config.get('parameter_feedback_mode') == 'guarded_once'
                    normalized, _ = normalize_plan({'strategies': [st]},
                        budgets=methods[i]['idea']['stage_budgets'], mirror_full=guarded)
                    st = normalized['strategies'][0]
                    required = ('name', 'family', 'idea', 'fe_config', 'model_config')
                    if st['idea'].get('plan_format') != 'brief_v1':
                        required += ('reason',)
                    if (not isinstance(st.get('name'), str) or not st['name'].strip() or
                            st['name'].strip().casefold() in {m['name'].strip().casefold() for m in methods} or
                            st.get('family') not in ('linear','bagging','boosting','neural','neighbors','naive_bayes','tree') or
                            ('reason' in required and (not isinstance(st.get('reason'),str) or not st['reason'].strip())) or
                            not all(isinstance(st.get(k), dict) for k in ('fe_config', 'model_config'))):
                        raise ValueError('Invalid or duplicate replacement method metadata')
                    extra = set(st) - set(required) - {'reason', 'budget_usage', 'structure_notes'}
                    if extra:
                        raise ValueError('Unexpected replacement fields: ' + str(sorted(extra)))
                    stages = st['idea']['stage_parameters']
                    if guarded and stages.get('coreset') != stages.get('full_train'):
                        raise ValueError('Guarded replacement stages must match')
                    if 'stage_parameters' in st['model_config']:
                        declared = st['model_config']['stage_parameters']
                        if guarded and isinstance(declared, dict) and set(declared) == {'full_train'}:
                            declared['coreset'] = copy.deepcopy(declared['full_train'])
                        if identity(declared) != identity(stages):
                            raise ValueError('Model configuration disagrees with declared stage parameters')
                    from .model_registry import normalize_strategy_family
                    normalize_strategy_family(st)
                    response['strategy'] = {key: st[key] for key in required}
                    validate_idea(st['idea'])
                    validate_stage_parameters(st['idea'])
                    if st['idea']['stage_budgets'] != methods[i]['idea']['stage_budgets']:
                        raise ValueError('Replacement may not change stage budgets')
                    validate_code(response['code'])
                    if response['code'].strip() == old.strip():
                        raise ValueError('Replacement repeats the failed implementation')
                    if any(response['code'].strip() == Path(m['code_path']).read_text().strip() for m in siblings):
                        raise ValueError('Replacement duplicates another initial implementation')
                last_chance = (redesign_prompt, validate_last)
            ledger_path = state_path.parent/'method_regeneration.json'
            before_used = json.loads(ledger_path.read_text())['methods'].get(str(i), {}).get('used', 0) if ledger_path.exists() else 0
            try:
                response = await regenerate(i, kind, failure['error'], prompt, destination,
                    ledger_path, model, calls, pricing, validate_response, resume=resuming, last_chance=last_chance)
            finally:
                if ledger_path.exists():
                    used = json.loads(ledger_path.read_text())['methods'].get(str(i), {}).get('used', 0)
                    counts[kind] += used - before_used
                    write_json(state_path, state)
            from .constructor_compatibility import normalize_constructor_controls
            code, compatibility = normalize_constructor_controls(response['code'])
            write_json(path.parent/'constructor_normalization.json', {'changes': compatibility})
            path.write_text(code)
            method = methods[i]
            before = method['code_sha256']
            replaced = 'strategy' in response
            if replaced:
                strategy = response['strategy']
                method.pop('reason', None)
                method.update(strategy, implementation_configuration={k:strategy[k] for k in ('fe_config','model_config')})
                plan_path = cache/'methods/plan.normalized.json'
                if plan_path.exists():
                    plan = json.loads(plan_path.read_text())
                    plan['strategies'][i] = {key: value for key, value in strategy.items()
                        if strategy['idea'].get('plan_format') != 'brief_v1' or key not in ('fe_config','model_config')}
                    write_json(plan_path, plan)
                write_json(path.parent/'idea.json', dict(version=1, idea_id=identity(strategy), strategy=strategy))
            mode = 'slot_redesign' if replaced else 'error_only'
            write_json(path.parent/'implementation.json', dict(code=code,
                **method.get('implementation_configuration',{key:method[key] for key in ('fe_config','model_config')})))
            method.update(code_sha256=file_hash(path), code_path=str(path))
            requests = json.loads(ledger_path.read_text())['methods'][str(i)]['requests']
            active = next(r for r in reversed(requests) if r['status']=='success' and
                          Path(r['artifact']).parent.resolve()==destination.resolve())
            method.setdefault('repairs', []).append({'kind': kind, 'response_artifact': active['artifact'], 'mode': mode, 'source_sha256': before,
                    'repaired_sha256': method['code_sha256'], 'audit': str(destination/'manifest.json')})
            write_json(path.parent/'method.json', method)
            audit['changes'].append({'method_id': i, 'mode': mode, 'error': failure['error'],
                'before_sha256': before, 'after_sha256': method['code_sha256']})
        for i, method in enumerate(methods):
            folder = cache/'methods'/f'method_{i}'
            method['code_path'] = str(folder/'method.py')
            write_json(folder/'method.json', method)
        write_json(cache/'methods/methods.json', methods)
        derived = {**previous, 'status': 'derived_repaired_cache', 'methods': methods,
                   'llm_calls': [], 'repair_source': str(source), 'repair_audit': str(destination/'manifest.json')}
        write_json(cache/'manifest.json', derived)
        audit['status'] = 'success'
        return cache
    except TaskDeadlineExceeded as error:
        audit.update(status='interrupted', error=f'{type(error).__name__}: {error}',
                     interruption_kind='task_deadline', method_failure=False)
        raise
    except BaseException as error:
        audit.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(destination/'manifest.json', audit)


def assert_interrupted_bootstrap_inactive(source):
    """Reject orphan workers before recovering an attempt under its parent lock.

    Legacy attempts have no PID record. Their isolated workers nevertheless carry
    a context path below the attempt in argv; checking it avoids copying files
    while a surviving worker is still writing them. Never stop another process.
    """
    source = Path(source).resolve()
    proc = Path('/proc')
    if not proc.is_dir():
        raise ValueError('Interrupted bootstrap recovery requires process-owner verification')
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            arguments = (entry/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # Other users' processes cannot own this private bootstrap output.
            continue
        for argument in arguments:
            if not argument.startswith(b'/'):
                continue
            try:
                path = Path(os.fsdecode(argument)).resolve()
            except (OSError, ValueError):
                continue
            if path == source or source in path.parents:
                raise ValueError(f'Bootstrap worker is still active (PID {entry.name}); recovery deferred')


def interrupted_bootstrap_manifest(source):
    """Read a legacy hard-killed attempt without overwriting its old manifest.

    Older checkpoints persisted trace.json only. Recover the cache identity from
    that trace and the completed method records, then let ordinary cache checks
    compare it to freshly verified task inputs before any method can be reused.
    """
    source = Path(source)
    previous = json.loads((source/'manifest.json').read_text())
    trace_path = source/'trace.json'
    if previous.get('status') not in ('running', 'interrupted') or not trace_path.exists():
        return previous
    trace = json.loads(trace_path.read_text())
    dataset = trace.get('dataset', {})
    if (Path(dataset.get('task_path', '')).resolve() != Path(previous['task_path']).resolve()
            or trace.get('model') != previous.get('llm_model')):
        raise ValueError('Interrupted trace does not match the frozen task and generation model')
    # Do not replace newer manifest state with an older trace. The legacy case
    # lacks the generation cache identity altogether.
    if previous.get('methods_cache_key'):
        return previous
    if not dataset.get('data_hashes') or not trace.get('internal_split_id'):
        return previous
    from .multi_proxy import MultiProxyConfig, method_cache_key
    configuration = trace.get('generation_configuration') or trace['configuration']
    cfg = MultiProxyConfig(**configuration)
    result = {**previous, 'configuration': trace['configuration'],
        'generation_configuration': configuration,
        'generation_model': trace.get('generation_model') or trace['model'],
        'input_sha256': dataset['data_hashes'], 'split_id': dataset.get('split_id'),
        'internal_split_id': trace['internal_split_id'],
        'methods_cache_key': method_cache_key(cfg, trace['model'], dataset['data_hashes'], trace['internal_split_id']),
        'interrupted_metadata_recovered_from': str(trace_path)}
    methods = source/'methods/methods.json'
    design = source/'coreset_design.json'
    if methods.is_file() and design.is_file():
        result['methods'] = json.loads(methods.read_text())
        result['coreset_design'] = json.loads(design.read_text())
    return result


def reconcile_interrupted_requests(directory):
    """Reconcile copied request journals after the owner and workers are gone.

    A complete saved response is reused. A request without a complete response
    remains an explicitly recorded transport interruption; its attempt count is
    retained and it cannot consume a semantic method-repair allowance.
    """
    from .generation_recovery import parse_object
    for path in Path(directory).rglob('*.recovery.json'):
        state = json.loads(path.read_text())
        changed = False
        for kind, record in state.get('requests', {}).items():
            interrupted = record.get('in_flight') or 'TaskDeadlineExceeded:' in str(record.get('error', ''))
            if not interrupted:
                continue
            artifact = path.parent / record.get('artifact', path.name.replace('.recovery.json', '.json'))
            raw = artifact.with_suffix('.response.txt')
            usage_path = artifact.with_suffix('.usage.json')
            usage = json.loads(usage_path.read_text()) if usage_path.exists() else {}
            available = False
            if raw.exists() and usage.get('finish_reason') == 'stop':
                try:
                    response = parse_object(raw.read_text())
                except (ValueError, TypeError):
                    pass
                else:
                    write_json(artifact, response)
                    available = True
            state.setdefault('interruption_history', []).append({'kind': kind, **record})
            record.update(in_flight=False, terminal=available, success=available)
            if not available:
                # Preserve bounded transport accounting. A fresh explicit resume
                # can issue one further request after an exhausted interrupted
                # slot, without turning a transport interruption into a repair.
                if record.get('attempts', 0) >= 3:
                    record['artifact_offset'] = record.get('artifact_offset', 0) + record['attempts'] - 2
                    record['attempts'] = 2
                record['error'] = 'Interrupted transport; resumed after process-owner verification'
            changed = True
        if changed:
            state['status'] = 'running'
            write_json(path, state)


def reuse_interrupted_proxy(source, destination, code, train, predict, task, classes, budget, seed, batch_size):
    """Reuse only complete, input-identical P fits from an interrupted attempt."""
    import numpy as np
    source, destination = Path(source), Path(destination)
    context_path = source/'context.json'
    if not context_path.exists():
        context_path = source/'source_context.json'
    names = ('fit_report.json', 'predictions.npz', 'unfitted.joblib', 'fitted.joblib')
    if not context_path.exists() or not all((source/name).is_file() for name in names):
        return None
    context = json.loads(context_path.read_text())
    expected = {'task': task, 'classes': classes, 'budget': budget, 'seed': seed, 'batch_size': batch_size}
    if any(context.get(key) != value for key, value in expected.items()):
        return None
    for key, value in (('code', code), ('train', train), ('predict', predict)):
        old = Path(context.get(key, ''))
        if not old.is_file() or file_hash(old) != file_hash(value):
            return None
    report = json.loads((source/'fit_report.json').read_text())
    if report.get('status') != 'success':
        return None
    try:
        with np.load(source/'predictions.npz', allow_pickle=False) as saved:
            predictions, rows = saved['predictions'], saved['row_positions']
            if (predictions.ndim != 1 or len(predictions) != report.get('prediction_rows')
                    or not np.array_equal(rows, np.arange(len(predictions)))):
                return None
    except (OSError, ValueError, KeyError, EOFError):
        return None
    destination.mkdir(parents=True, exist_ok=False)
    for name in names:
        shutil.copyfile(source/name, destination/name)
    shutil.copyfile(context_path, destination/'source_context.json')
    write_json(destination/'reuse.json', {'source': str(source.resolve()), 'new_fit': False,
        'reason': 'Identical successful P fit preserved after infrastructure interruption',
        'sha256': {name: file_hash(source/name) for name in names}})
    return report


def interrupted_feedback_request(request, source):
    """Replay only response journals belonging to the identical feedback prompt."""
    source = Path(source)
    async def wrapped(model, prompt, output, costs, pricing):
        check_deadline()
        previous = source/Path(output).name
        request_path = previous.with_suffix('.request.json')
        if request_path.is_file():
            saved = json.loads(request_path.read_text())
            if saved.get('prompt') == prompt and saved.get('model') == model.model_name:
                for path in source.glob(previous.stem+'.*'):
                    if path.is_file():
                        shutil.copyfile(path, Path(output).parent/path.name)
                reconcile_interrupted_requests(Path(output).parent)
                from .generation_recovery import reopen_transport_requests
                reopen_transport_requests(Path(output).parent)
                usage_path = previous.with_suffix('.usage.json')
                usage = json.loads(usage_path.read_text()) if usage_path.exists() else {}
                if previous.is_file() and usage.get('status') == 'success' and usage.get('finish_reason') == 'stop':
                    response = json.loads(previous.read_text())
                    write_json(Path(output).with_suffix('.reuse.json'), {'source': str(previous),
                        'sha256': file_hash(previous), 'new_api_call': False})
                    return response
        return await request(model, prompt, output, costs, pricing)
    return wrapped


def resolve_bootstrap_attempt(base):
    """Publish only attempts with a durable manifest; reconcile old publication gaps."""
    base = Path(base).resolve()
    path = base/'attempt_state.json'
    if not path.exists():
        return base
    state = json.loads(path.read_text())
    current = Path(state.get('effective_run', base))
    pending = state.get('pending_attempt')
    if pending:
        destination = Path(pending['destination'])
        if (destination/'manifest.json').is_file():
            current = destination
            if str(current) not in state['attempts']:
                state['attempts'].append(str(current))
            state.pop('pending_attempt')
        else:
            current = Path(pending['source'])
    if not (current/'manifest.json').is_file():
        candidates = [Path(value) for value in reversed(state.get('attempts', []))] + [base]
        recovered = next((value for value in candidates if (value/'manifest.json').is_file()), None)
        if recovered is None:
            raise ValueError('Bootstrap has no durable source manifest to recover')
        state.setdefault('publication_recovery', []).append({'missing_attempt': str(current), 'source': str(recovered)})
        if current.name.startswith('attempt_'):
            cache = base/'repairs'/('repair_' + current.name.removeprefix('attempt_'))/'cache'
            if (cache/'manifest.json').is_file():
                saved = json.loads((cache/'manifest.json').read_text())
                if saved.get('status') == 'derived_repaired_cache' and saved.get('repair_source') == str(recovered):
                    for method in saved['methods']:
                        if file_hash(method['code_path']) != method['code_sha256']:
                            raise ValueError('Legacy completed repair code changed')
                    state['pending_attempt'] = {'kind': 'repair', 'source': str(recovered),
                        'destination': str(current), 'cache': str(cache)}
        elif current.name.startswith('generation_recovery_'):
            state['pending_attempt'] = {'kind': 'generation', 'source': str(recovered), 'destination': str(current)}
        current = recovered
    state['effective_run'] = str(current)
    write_json(path, state)
    return current


async def run_with_repairs(task, output, cfg, model, identity, pricing=None, reuse=None,
                           resume_failed=False, resume_interrupted=False):
    from .multi_proxy import run_task
    from .generation_recovery import transport_failure_text
    from .method_regeneration import RegenerationExhausted
    base = Path(output).resolve()
    state_path = base/'attempt_state.json'
    if base.exists() and not resume_failed:
        raise ValueError('Existing experiment requires --resume-failed')
    if resume_failed and base.exists():
        assert_interrupted_bootstrap_inactive(base)
        resolve_bootstrap_attempt(base)
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'repair_counts': {}, 'attempts': [], 'effective_run': str(base)}
    state['policy'] = 'Two total method regeneration chances shared across generation, preflight and runtime; complete task rerun'
    ledger_path = base/'method_regeneration.json'
    if base.exists() and not ledger_path.exists():
        ledger = {'methods': {key: {'used': sum(counts.values()), 'errors': [], 'requests': [],
                   'legacy_repair_counts': counts} for key, counts in state.get('repair_counts', {}).items()}}
        for attempt in state.get('attempts', []):
            manifest_path = Path(attempt)/'manifest.json'
            if manifest_path.exists():
                old = json.loads(manifest_path.read_text())
                stage, errors = failure_methods(Path(attempt), old)
                for error in errors:
                    entry = ledger['methods'].get(str(error['method_id']))
                    if entry is not None:
                        entry['errors'].append({'stage': stage, 'error': error['error'], 'source': attempt})
        write_json(ledger_path, ledger)
    current = Path(state['effective_run'])

    async def execute_attempt(pending):
        nonlocal current
        source, destination = Path(pending['source']), Path(pending['destination'])
        state['pending_attempt'] = pending
        state.pop('pending_repair', None)
        write_json(state_path, state)
        # A hard kill between mkdir and the first manifest can leave an empty directory.
        if destination.exists() and not (destination/'manifest.json').exists():
            if any(destination.iterdir()):
                raise ValueError('Unpublished attempt has unexpected files: ' + str(destination))
            destination.rmdir()
        options = {'regeneration_state_path': ledger_path}
        if pending['kind'] == 'generation':
            options['generation_resume'] = source
        elif pending['kind'] == 'repair':
            options['reuse_methods'] = Path(pending['cache'])
        else:
            previous = interrupted_bootstrap_manifest(source)
            complete = ((source/'methods/methods.json').is_file()
                        and len(previous.get('methods', [])) == cfg.n_strategies)
            options.update(reuse_methods=source if complete else None,
                generation_resume=source if not complete and previous.get('methods_cache_key') else None,
                interrupted_resume=source, interrupted_manifest=previous)
        try:
            await run_task(task, destination, cfg, model, identity, pricing, **options)
        except TaskDeadlineExceeded:
            raise
        except Exception:
            if not (destination/'manifest.json').exists():
                raise
        finally:
            if (destination/'manifest.json').exists():
                current = destination
                if str(current) not in state['attempts']:
                    state['attempts'].append(str(current))
                state['effective_run'] = str(current)
                state.pop('pending_attempt', None)
                write_json(state_path, state)

    def next_attempt(prefix):
        index = len(state['attempts']) + 1
        while (base/'attempts'/f'{prefix}_{index}').exists():
            index += 1
        return base/'attempts'/f'{prefix}_{index}'

    if state.get('pending_attempt'):
        await execute_attempt(state['pending_attempt'])
        resume_interrupted = False
    elif resume_failed and (current/'manifest.json').exists():
        previous = interrupted_bootstrap_manifest(current)
        recoverable = (previous.get('status') in ('running', 'interrupted') or
            infrastructure_failure(previous.get('error', '')) or
            (previous.get('failure_stage') == 'parameter_feedback' and transport_failure_text(previous.get('error'))))
        if resume_interrupted or recoverable:
            if ledger_path.exists():
                ledger = json.loads(ledger_path.read_text())
                for record in ledger.get('methods', {}).values():
                    for request in record.get('requests', []):
                        if request.get('status') == 'running':
                            request.update(status='interrupted', interruption_kind='owner_exited', method_failure=False)
                write_json(ledger_path, ledger)
            destination = next_attempt('interrupted_resume')
            state.setdefault('infrastructure_resumes', []).append({'source': str(current),
                'destination': str(destination), 'previous_status': previous['status'],
                'source_manifest_sha256': file_hash(current/'manifest.json'), 'method_repairs_used': 0})
            await execute_attempt({'kind': 'interruption', 'source': str(current), 'destination': str(destination)})
        elif previous.get('failure_stage') == 'llm_generation' and transport_failure_text(previous.get('error')):
            state['generation_resume_count'] = 0
    if not base.exists():
        try:
            await run_task(task, base, cfg, model, identity, pricing, reuse, regeneration_state_path=ledger_path)
        except TaskDeadlineExceeded:
            raise
        except Exception:
            if not (base/'manifest.json').exists():
                raise
        state['attempts'].append(str(base))
        write_json(state_path, state)
    while True:
        manifest = json.loads((current/'manifest.json').read_text())
        if manifest['status'] == 'success':
            return manifest
        if manifest['status'] != 'failed':
            raise InfrastructureInterruption(manifest.get('error', 'Bootstrap interrupted; resume after resolving the host/transport failure'))
        if manifest.get('failure_stage') == 'llm_generation':
            failure = manifest.get('error', '')
            if failure.startswith('RegenerationExhausted: '):
                raise RegenerationExhausted(failure.removeprefix('RegenerationExhausted: '))
            if state.get('generation_resume_count', 0) >= 1:
                raise RuntimeError('Generation recovery already attempted: '+manifest.get('error', ''))
            state['generation_resume_count'] = 1
            await execute_attempt({'kind': 'generation', 'source': str(current),
                                   'destination': str(next_attempt('generation_recovery'))})
            continue
        kind, failures = failure_methods(current, manifest)
        if not failures or len(manifest.get('methods', [])) != manifest.get('configuration', {}).get('n_strategies', 3):
            raise RuntimeError(manifest.get('error', 'Failure is not eligible for method repair'))
        # Older preflight ran sklearn's inherited generic parameter validator
        # on FeatureUnion even though it exposes no constraint table. Recheck
        # the frozen originals once; rejected repair responses stay rejected.
        if (resume_failed and kind == 'preflight' and not state.get('original_preflight_recheck')
                and all(row.get('error') == "AttributeError: 'FeatureUnion' object has no attribute '_parameter_constraints'"
                        for row in failures)):
            destination = next_attempt('preflight_recheck')
            state['original_preflight_recheck'] = {'source': str(current), 'destination': str(destination)}
            await execute_attempt({'kind': 'repair', 'source': str(current), 'cache': str(current),
                                   'destination': str(destination)})
            continue
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'methods': {}}
        pending = state.get('pending_repair')
        # Older versions did not persist the repair transaction before requesting.
        if pending is None and resume_failed:
            for record in ledger['methods'].values():
                for request in record.get('requests', []):
                    artifact = Path(request.get('artifact', ''))
                    if request.get('status') == 'running' and base/'repairs' in artifact.parents:
                        pending = {'source': str(current), 'destination': str(artifact.parent), 'kind': kind, 'failures': failures}
        if pending is None:
            for failure in failures:
                if ledger['methods'].get(str(failure['method_id']), {}).get('used', 0) >= 2:
                    raise RegenerationExhausted('Method regeneration allowance exhausted: '+manifest.get('error', ''))
            index = len(state['attempts']) + 1
            while (base/'repairs'/f'repair_{index}').exists():
                index += 1
            pending = {'source': str(current), 'destination': str(base/'repairs'/f'repair_{index}'),
                       'kind': kind, 'failures': failures}
        if pending['source'] != str(current):
            raise ValueError('Pending repair belongs to another bootstrap attempt')
        state['pending_repair'] = pending
        write_json(state_path, state)
        cache = await repair_cache(current, Path(pending['destination']), pending['failures'],
                                   pending['kind'], model, pricing, state, state_path)
        await execute_attempt({'kind': 'repair', 'source': str(current), 'cache': str(cache),
                               'destination': str(next_attempt('attempt'))})
