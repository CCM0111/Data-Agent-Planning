"""Run optional regional sampling, P/Q feedback, MCS and MCTS from one config."""
import argparse
import asyncio
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import sys

import yaml

from data_agent.mcts_policy import MCTSConfig
from data_agent.region_bandit.config import normalize_config
from data_agent.region_bandit.schemas import dataset_group
from data_agent.workflow import module_settings
from run_region_bandit import _public_model

ROOT = Path(__file__).resolve().parent


def _merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = (_merge(result[key], value) if isinstance(value, dict)
                       and isinstance(result.get(key), dict) else deepcopy(value))
    return result


def settings_for(path):
    from run_region_bandit import settings_for as region_settings
    from run_mcts_data_agent import settings_for as search_settings
    settings = yaml.safe_load(Path(path).read_text())
    allowed = {'modules', 'region_config', 'search_config', 'multi_proxy', 'region_bandit',
               'search', 'memory', 'model', 'pricing', 'subgroups'}
    if not isinstance(settings, dict) or set(settings) - allowed:
        raise ValueError('Unknown or invalid workflow configuration sections')
    modules = module_settings(settings.get('modules'))
    def resolve(name, default):
        value = Path(settings.get(name, default))
        return value if value.is_absolute() else ROOT/value
    config, model, pricing = region_settings(resolve('region_config', 'config/region_bandit.yaml'))
    original, search, memory, _, _ = search_settings(resolve('search_config', 'config/mcts.yaml'))
    config = _merge(config, {k: settings[k] for k in ('multi_proxy', 'region_bandit') if k in settings})
    # Module switches are authoritative over inherited legacy configuration.
    config['multi_proxy']['parameter_feedback'] = modules['parameter_feedback']
    config['multi_proxy'].setdefault('stage1_selection', {})['enabled'] = modules['mcs']
    region = config['region_bandit']
    region['enabled'] = modules['region_selection']
    # The search switch owns the initial full-evaluation policy. It overrides
    # both inherited settings and an explicit conflicting legacy override.
    region['full_train_methods'] = 'top3' if modules['mcts'] else 'all'
    region['advisor']['enabled'] = modules['region_selection'] and modules['region_advisor']
    region['post_full_feedback']['enabled'] = modules['region_selection'] and modules['post_full_feedback']
    config['region_bandit'] = normalize_config(region)
    if modules['region_selection'] and (region['coreset_repeats'] != 1 or region['baseline_audit']):
        raise ValueError('Combined workflow uses one coreset without baseline audit; '
                         'use run_region_bandit.py for repeated-sampling or baseline-audit experiments')
    if not modules['region_selection'] and region['mode'] == 'evaluate_frozen':
        raise ValueError('Disabled regional sampling cannot use evaluate_frozen')
    if (modules['region_selection'] and modules['mcts'] and region['mode']=='evaluate_frozen'
            and config['region_bandit']['mcts_rounds']['enabled']):
        raise ValueError('Existing frozen policies cover initialization only; set '
                         'region_bandit.mcts_rounds.enabled=false for evaluate_frozen')
    search = MCTSConfig.from_dict({**search.to_dict(),
                                  'infrastructure_retries': region['runtime']['infrastructure_retries'],
                                  **settings.get('search', {}),
                                  'enabled': modules['mcts'], 'mcs_enabled': modules['mcs']})
    memory = {**memory, **settings.get('memory', {}), 'enabled': modules['search_memory']}
    if set(memory) - {'enabled', 'default_count', 'max_count', 'token_budget'}:
        raise ValueError('Unknown search memory settings')
    count, maximum, tokens = (memory.get('default_count', 4), memory.get('max_count', 6),
                              memory.get('token_budget', 3000))
    if (any(type(v) is not int for v in (count, maximum, tokens)) or
            not 1 <= count <= maximum <= 6 or tokens < 128):
        raise ValueError('Invalid search memory count/token budget')
    from data_agent.region_bandit.runtime import bootstrap_config
    bootstrap_config(config['multi_proxy']).validate()
    model = deepcopy(settings.get('model', model))
    if not isinstance(model, dict) or not model.get('model_name'):
        raise ValueError('A generation model_name is required')
    return config, search, memory, modules, model, settings.get('pricing', pricing), settings.get('subgroups', original.get('subgroups'))


async def run(args):
    config, search, memory, modules, model_settings, pricing, subgroups = settings_for(args.config)
    if args.extend_wall_seconds < 0 or (args.extend_wall_seconds and not args.resume):
        raise ValueError('--extend-wall-seconds must be nonnegative and requires --resume')
    if args.extend_wall_seconds and not modules['region_selection']:
        raise ValueError('Wall extensions are only supported by the regional sampling stage')
    task_path = args.task.resolve()
    task = json.loads(task_path.read_text())
    group = dataset_group(task, args.dataset_group_id) if modules['region_selection'] else None
    identity = _public_model(model_settings)
    evaluation = json.loads(args.evaluation_manifest.read_text()) if args.evaluation_manifest else None
    if config['region_bandit']['mode'] == 'evaluate_frozen':
        if not args.snapshot_id or evaluation is None:
            raise ValueError('evaluate_frozen requires --snapshot-id and --evaluation-manifest')
        from data_agent.region_bandit.runtime import validate_evaluation_manifest, evaluation_protocol
        evaluation = validate_evaluation_manifest(evaluation, group)
        ids = [v.get('snapshot_id') if isinstance(v, dict) else v for v in evaluation['snapshot_ids']]
        if args.snapshot_id not in ids or evaluation['evaluation_protocol'] != evaluation_protocol(config, identity):
            raise ValueError('Frozen snapshot or configuration differs from the evaluation manifest')
    elif args.snapshot_id or evaluation is not None:
        raise ValueError('Snapshot/manifest options require enabled frozen regional sampling')
    if args.dry_run:
        rounds=(modules['region_selection'] and modules['mcts'] and
                config['region_bandit']['mcts_rounds']['enabled'])
        print(json.dumps({'modules': modules, 'configuration': config, 'search': search.to_dict(),
            'memory': memory, 'model': identity, 'task': str(task_path), 'dataset_group_id': group,
            'flow': ['generate_five_and_fit_P'] + (['P_Q_parameter_feedback'] if modules['parameter_feedback'] else []) +
                    (['region_discovery_and_action_selection'] if modules['region_selection'] else ['original_coverage_sampling']) +
                    ['five_core_evaluations', 'MCS_diagnostics_and_ranking' if modules['mcs'] else 'ranking_only',
                     'three_full_evaluations' if modules['mcts'] else 'five_full_evaluations'] +
                    ([f'bandit_feedback_from_initial_{search.initial_full_count}'] if modules['region_selection'] else []) +
                    (['MCTS_expansion'] if modules['mcts'] else []) +
                    (['per_expansion_freeze_methods_and_select_coreset', 'parent_anchor_and_three_child_low_fits',
                      'one_or_two_new_full_fits', 'anchored_bandit_feedback'] if rounds else []),
            'full_train_methods': search.full_train_methods,
            'initial_full_method_count': search.initial_full_count,
            'initial_predictive_fits': 10 + search.initial_full_count,
            'mcts_full_budget_includes_initial_three': modules['mcts'],
            'per_expansion_region_bandit': rounds,
            'real_api_calls': 0, 'actual_model_fits': 0, 'test_access': False}, ensure_ascii=False, indent=2))
        return 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.workflow.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another workflow owns this output directory') from error
        from dotenv import load_dotenv
        from internagent.mas.models.openai_model import OpenAIModel
        from data_agent.workflow import run_workflow
        load_dotenv(args.env_file)
        options = model_settings.pop("request_options", {})
        model = OpenAIModel(**model_settings)
        model.request_options = options
        try:
            result = await run_workflow(task_path, output, config=config, search=search, memory=memory,
                modules=modules, model=model, model_identity=identity, pricing=pricing,
                memory_path=args.memory.resolve(), dataset_group_id=group, subgroups=subgroups,
                snapshot_id=args.snapshot_id, evaluation_manifest=evaluation, resume=args.resume,
                extend_wall_seconds=args.extend_wall_seconds)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get('status') in ('completed', 'budget_exhausted') and result.get('initialization_complete') else 1
        finally:
            if getattr(model, 'client', None) is not None:
                await model.client.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--task', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--config', type=Path, default=ROOT/'config/workflow.yaml')
    result.add_argument('--env-file', type=Path, default=Path('/home/cmcui/InternAgent/.env'))
    result.add_argument('--memory', type=Path, default=ROOT/'results/region_bandit_policy/memory.sqlite3')
    result.add_argument('--dataset-group-id')
    result.add_argument('--snapshot-id')
    result.add_argument('--evaluation-manifest', type=Path)
    result.add_argument('--resume', action='store_true')
    result.add_argument('--extend-wall-seconds', type=int, default=0,
                        help='Add regional sampling execution time on resume; frozen configuration and MCTS budget stay unchanged')
    result.add_argument('--dry-run', action='store_true')
    return result


def main(argv=None):
    try:
        return asyncio.run(run(parser().parse_args(argv)))
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
