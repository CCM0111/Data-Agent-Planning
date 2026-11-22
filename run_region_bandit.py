"""Run one frozen five-method region-bandit task; explicit invocation uses real API.

No MCTS or test-set access. --dry-run validates settings without an API client,
model fitting, output directory creation, or memory updates.
"""
import argparse
import asyncio
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import sys

import yaml

from data_agent.region_bandit.config import normalize_config, full_train_count
from data_agent.region_bandit.schemas import dataset_group

ROOT = Path(__file__).resolve().parent


def settings_for(path, mode=None):
    settings = yaml.safe_load(Path(path).read_text())
    if not isinstance(settings, dict):
        raise ValueError('Region configuration must be a YAML object')
    unknown = set(settings) - {'initialization_config', 'multi_proxy', 'region_bandit', 'model', 'pricing'}
    if unknown:
        raise ValueError('Unknown region config sections: ' + str(sorted(unknown)))
    initial = Path(settings.get('initialization_config', 'config/coverage_regression30.yaml'))
    if not initial.is_absolute():
        initial = ROOT / initial
    source = yaml.safe_load(initial.read_text())
    values = {**source.get('workflow', {}), **settings.get('multi_proxy', {})}
    if 'lambda' in values:
        inherited_lambda = values.pop('lambda')
        values.setdefault('similarity_lambda', inherited_lambda)
    region = deepcopy(settings.get('region_bandit', {}))
    if mode is not None:
        region['mode'] = mode
    config = {'multi_proxy': values, 'region_bandit': normalize_config(region)}
    if not config['region_bandit']['enabled']:
        config['region_bandit']['full_train_methods'] = 'all'
    model = deepcopy(settings.get('model', source.get('model', {})))
    if not isinstance(model, dict) or not model.get('model_name'):
        raise ValueError('A model configuration with model_name is required')
    return config, model, settings.get('pricing', source.get('pricing'))


def _public_model(settings):
    # Whitelist connection metadata; never include credential values or headers.
    keys = ('model_name', 'base_url', 'temperature', 'max_tokens', 'timeout', 'api_key_env_var')
    public = {key: settings[key] for key in keys if key in settings}
    if 'reasoning_effort' in settings.get('request_options', {}):
        public['reasoning_effort'] = settings['request_options']['reasoning_effort']
    return public


async def run(args):
    config, model_settings, pricing = settings_for(args.config, args.mode)
    task_path = args.task.resolve()
    task = json.loads(task_path.read_text())
    region = config['region_bandit']
    group = dataset_group(task, args.dataset_group_id) if region['enabled'] else None
    if not region['enabled'] and region['mode'] == 'evaluate_frozen':
        raise ValueError('Disabled regional sampling cannot use evaluate_frozen')
    if args.extend_wall_seconds < 0 or (args.extend_wall_seconds and not args.resume):
        raise ValueError('--extend-wall-seconds must be nonnegative and requires --resume')
    evaluation = json.loads(args.evaluation_manifest.read_text()) if args.evaluation_manifest else None
    if region['mode'] == 'evaluate_frozen':
        if not args.snapshot_id or evaluation is None:
            raise ValueError('evaluate_frozen requires --snapshot-id and --evaluation-manifest')
        if group not in evaluation.get('evaluation_dataset_groups', []):
            raise ValueError('Task dataset_group_id is absent from the frozen evaluation manifest')
        if group in evaluation.get('development_dataset_groups', []):
            raise ValueError('Evaluation dataset group also appears in development manifest')
        declared_snapshots = evaluation.get('snapshot_ids')
        if declared_snapshots is not None:
            allowed = [entry.get('snapshot_id') if isinstance(entry, dict) else entry for entry in declared_snapshots]
            if args.snapshot_id not in allowed:
                raise ValueError('Snapshot ID is absent from the frozen evaluation manifest')
        from data_agent.region_bandit.runtime import validate_evaluation_manifest, evaluation_protocol
        validate_evaluation_manifest(evaluation, group)
        if evaluation['evaluation_protocol'] != evaluation_protocol(config, _public_model(model_settings)):
            raise ValueError('Frozen evaluation protocol differs from the declared model, prompts or configuration')
    elif args.snapshot_id or evaluation is not None:
        raise ValueError('--snapshot-id and --evaluation-manifest are reserved for evaluate_frozen')
    identity = _public_model(model_settings)
    if args.dry_run:
        if not region['enabled']:
            print(json.dumps({'task': str(task_path), 'output': str(args.output.resolve()),
                'region_selection_enabled': False, 'mcts_enabled': False,
                'flow': 'Original coverage sampling, five core evaluations, five full evaluations',
                'full_train_methods': 'all', 'full_train_method_count': 5,
                'configuration': config, 'model': identity,
                'theoretical_prediction_model_fits': 15, 'real_api_calls': 0,
                'actual_model_fits': 0, 'test_access': False, 'bandit_memory_access': False},
                ensure_ascii=False, indent=2))
            return 0
        repeats = region['coreset_repeats']
        print(json.dumps({'task': str(task_path), 'dataset_group_id': group,
                          'output': str(args.output.resolve()), 'memory': str(args.memory.resolve()),
                          'mode': region['mode'], 'configuration': config, 'model': identity,
                          'theoretical_prediction_model_fits': 5 + full_train_count(region) + 5 * repeats,
                          'full_train_method_count': full_train_count(region),
                          'baseline_audit_additional_fits_if_nonbaseline': 5 * repeats if region['baseline_audit'] else 0,
                          'gmm_fits': 1, 'region_diagnostic_additional_fits': 0,
                          'boundaries': 'Decisions use P/Q only; V_core and external validation are postdecision; no test access',
                          'real_api_calls': 0, 'actual_model_fits': 0}, ensure_ascii=False, indent=2))
        return 0
    # Construct the external client only after all cheap identity/config checks.
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.region_bandit.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another region-bandit process is using this output directory') from exc
        if (args.output / 'state.json').exists() and not args.resume:
            raise ValueError('Region run already exists; use --resume with unchanged inputs')
        from dotenv import load_dotenv
        from internagent.mas.models.openai_model import OpenAIModel
        from data_agent.region_bandit.runtime import run_region_task
        load_dotenv(args.env_file)
        model = OpenAIModel(**model_settings)
        try:
            result = await run_region_task(task_path, args.output.resolve(), config=config,
                model=model, model_identity=identity, pricing=pricing, memory_path=args.memory.resolve(),
                dataset_group_id=group, snapshot_id=args.snapshot_id, evaluation_manifest=evaluation,
                resume=args.resume, extend_wall_seconds=args.extend_wall_seconds)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get('status') in ('success', 'completed', 'feedback_committed') or result.get('phase') == 'feedback_committed' else 1
        finally:
            if getattr(model, 'client', None) is not None:
                await model.client.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--task', type=Path, required=True, help='Prepared task.json with fixed training/validation splits')
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--config', type=Path, default=ROOT / 'config/region_bandit.yaml')
    result.add_argument('--memory', type=Path, default=ROOT / 'results/region_bandit_policy/memory.sqlite3')
    result.add_argument('--dataset-group-id', help='Required unless task metadata contains the shared source dataset identity')
    result.add_argument('--mode', choices=('develop_online', 'evaluate_frozen'))
    result.add_argument('--snapshot-id', help='Frozen policy snapshot ID from evaluate_region_bandit.py')
    result.add_argument('--evaluation-manifest', type=Path)
    result.add_argument('--env-file', type=Path, default=Path('/home/cmcui/InternAgent/.env'))
    result.add_argument('--resume', action='store_true')
    result.add_argument('--extend-wall-seconds', type=int, default=0,
                        help='Add execution time on resume; leaves frozen methods, samples and reward protocol unchanged')
    result.add_argument('--dry-run', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
