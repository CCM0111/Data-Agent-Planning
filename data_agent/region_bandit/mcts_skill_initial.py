"""Initialization checkpoints use the same bounded replay as search rounds."""
from copy import deepcopy
from pathlib import Path

import numpy as np

from ..coreset import file_hash
from ..multi_proxy_io import read_typed
from ..stage1_metrics import metric_direction, primary_score
from .mcts_rounds import _binding, _method_bank, _validate_binding
from .mcts_skill_reward import ABSOLUTE_METRICS, checkpoint_reward, relative_fact
from .regional_sampler import sample_from_action
from .runtime import SAMPLER, _freeze_artifacts, _load_predictions, _verify_artifacts
from .schemas import read_json, stable_hash, write_json
from .skill_sampling import combine_scores, single_branch_trigger
from .skill_store import SkillStore, sampling_path


def proxy_validation_scores(root, source, context, task):
    from .execution import predict_cached
    labels = read_typed(source/'core_validation.csv', context['csv_schema'])
    features = source/'core_validation_features.csv'
    if not features.exists():
        features = root/'proxy_validation_features.csv'
        labels.drop(columns=[task['target']]).to_csv(features, index=False)
    methods = read_json(source/'methods/methods.json')
    config = read_json(source/'manifest.json')['configuration']
    scores = {}
    for index, method in enumerate(methods['methods'] if isinstance(methods, dict) else methods):
        cached=source/'guided_predictions'/str(index)
        prediction_root=cached if (source/'validation_guidance.json').exists() else root/'proxy_validation'/str(index)
        result = predict_cached(method['code_path'], source/f'proxies/method_{index}/fitted.joblib',
            features, context, context['classes'], prediction_root,
            batch_size=config['batch_size'], timeout_seconds=config['evaluation_budget']['timeout_seconds'])
        if result['status'] != 'ok':
            raise RuntimeError('Initial proxy validation failed: '+str(result['error']))
        values = _load_predictions(result['artifact'], context['classes'])
        scores[str(index)] = float(primary_score(task, labels[task['target']], values['predictions'],
            values['probabilities'], context['classes']))
    if len(scores) != 5 or not np.isfinite(list(scores.values())).all():
        raise ValueError('Expected five finite random proxy V_core scores')
    return scores


def freeze_initial_reference(root, source, *, relative=False):
    """Freeze the Q metric scale and, for relative rewards, P-trained scores on V_core."""
    root, source = Path(root), Path(source)
    path = root/'initial_reward_reference.json'
    if path.exists():
        saved = read_json(path)
        if relative and 'baseline_scores' not in saved:
            raise ValueError('Relative protocol requires a frozen proxy V_core baseline')
        return saved
    task_path = Path(read_json(source/'manifest.json')['task_path'])
    task = read_json(task_path)
    extra = {}
    if relative:
        extra.update(baseline_kind='random_proxy', evaluation_set='V_core',
            training_indices_sha256=file_hash(source/'P_indices.json'),
            validation_indices_sha256=file_hash(source/'V_core_indices.json'),
            validation_csv_sha256=file_hash(source/'core_validation.csv'))
        extra['baseline_scores'] = proxy_validation_scores(root, source, read_json(source/'task_context.json'), task)
    if task['metric'] in ABSOLUTE_METRICS:
        value = dict(**extra, metric=task['metric'], reference_scale=1.,
            scope='Fixed absolute metric tolerance; no outcome-dependent reference needed')
        write_json(path, value)
        return value
    context = read_json(source/'task_context.json')
    positions = read_json(source/'Q_indices.json')['train_positions']
    frame = read_typed(task_path.parent/task['files']['train'], context['csv_schema'],
                       usecols=[task['target']])
    labels = frame.iloc[positions][task['target']]
    scores, artifacts = [], {}
    for index in range(5):
        prediction = source/f'proxies/method_{index}/predictions.npz'
        values = _load_predictions(prediction, context['classes'])
        scores.append(float(primary_score(task, labels, values['predictions'],
                                          values['probabilities'], context['classes'])))
        artifacts[str(prediction)] = file_hash(prediction)
    if not np.isfinite(scores).all():
        raise ValueError('Initial reward requires five finite, predecision proxy scores')
    value = dict(**extra, metric=task['metric'], reference_scale=float(np.median(np.abs(scores))),
        proxy_scores=scores, prediction_hashes=artifacts,
        scope='Frozen P-trained proxy metric on Q; no coreset or full outcomes')
    write_json(path, value)
    return value


def save_initial_checkpoint(controller, engine, initial, fact, reference):
    """Persist original decision inputs separately from observed full results."""
    from .mcts_skills import _freeze
    folder = controller.root/'mcts_skill_initial'
    path = folder/'checkpoint.json'
    methods = [engine.state['candidates'][i] for i in engine.state['initial_ids']]
    if path.exists():
        _verify_artifacts(folder)
        saved = read_json(path)
        if (saved['frozen']['methods'] != _method_bank(methods)
                or saved['full_scores'] != fact['full_scores'] or saved['fact'] != fact):
            raise ValueError('Initial checkpoint methods or observations changed')
        return str(folder)
    controller._load()
    decision = read_json(controller.root/'decision.json')
    seed = decision['sampling_seeds'][0]
    actions = deepcopy(decision['actions'])
    orders = {}
    for action in actions:
        order, _ = sample_from_action(action, controller.reference,
            controller.inputs['density'], controller.inputs['hardness'], controller.graph,
            controller.inputs['groups'], controller.regions.get(action.get('region_id'), {}).get('mask'),
            seed=seed, config=SAMPLER, decision_seed=controller.cfg['decision_seed'])
        preview = action['context']['action_summary']['cross_region_preview'][str(seed)]
        if stable_hash(sorted(order.tolist())) != preview['sample_hash']:
            raise ValueError('Initial replay sample differs from its original preview')
        orders[action['action_id']] = order.tolist()
    frozen = dict(stage='initialization', checkpoint_id=fact['checkpoint_id'],
        bank_hash=stable_hash(_method_bank(methods)), methods=_method_bank(methods),
        actions=actions, sample_orders=orders, seed=seed, decision_seed=decision['decision_seed'],
        sampler_decision_seed=controller.cfg['decision_seed'],
        parent_binding=_binding(controller.source/'coreset_0.csv',
                                controller.source/'coreset_0_indices.json'),
        history=deepcopy(initial['procedural']['history']),
        reference_scale=reference['reference_scale'], initial_bundle_id=initial['bundle_id'])
    if 'baseline_scores' in fact:
        frozen['baseline_scores'] = fact['baseline_scores']
    checkpoint = dict(frozen=frozen, decision=decision,
        screening=deepcopy(engine.state['initial_skill_screening']), methods=deepcopy(methods),
        full_scores=deepcopy(fact['full_scores']), fact=deepcopy(fact))
    _freeze(path, checkpoint)
    _freeze_artifacts(folder, 'initial_replay_checkpoint',
                      [path, controller.root.parent/'initial_reward_reference.json'])
    return str(folder)


async def replay_initial_checkpoint(controller, engine, checkpoint_folder, bundle_id, output, *, history=None,
                                    sampling_seed=None):
    """Re-select the initial top three; reuse the same three observed full fits."""
    from .mcts_skills import _freeze, prepare_replay_inputs, replay_sample_pool
    controller._load()
    checkpoint_folder, output = Path(checkpoint_folder), Path(output)
    _verify_artifacts(checkpoint_folder)
    checkpoint = read_json(checkpoint_folder/'checkpoint.json')
    frozen = prepare_replay_inputs(controller, checkpoint, history, sampling_seed)
    if frozen.get('stage') != 'initialization':
        raise ValueError('Expected an initialization checkpoint')
    _freeze(output/'checkpoint_identity.json', dict(checkpoint=stable_hash(checkpoint),
        bundle=bundle_id, predecision=stable_hash(frozen)))
    decision = await controller._choose(engine, output, frozen, bundle_id)
    if (output/'fact.json').exists():
        fact = read_json(output/'fact.json')
    else:
        methods = checkpoint['methods']
        pending = engine.state.setdefault('skill_validation', {}).setdefault(
            stable_hash(dict(checkpoint=frozen['checkpoint_id'], bundle=bundle_id,
                             predecision=stable_hash(frozen))), {})

        async def branch(action_id):
            action = next(a for a in frozen['actions'] if a['action_id'] == action_id)
            sample_hash = action['context']['action_summary']['cross_region_preview'][str(frozen['seed'])]['sample_hash']
            previous = next((b for b in checkpoint['screening']['branches']
                             if b['sample_hash'] == sample_hash), None)
            if previous:
                result = deepcopy(previous)
                result['action'] = {k: action.get(k) for k in ('action_id', 'type', 'region_id')}
                result['cross_region_preview'] = deepcopy(action['context']['action_summary']['cross_region_preview'])
                return result
            pool = replay_sample_pool(checkpoint_folder, frozen, checkpoint['frozen']['seed'])
            binding = controller._materialize(engine, pool, frozen, action_id)
            scores, reports = await controller._probe_scores(engine, pending, methods, binding,
                'replay:'+frozen['checkpoint_id'])
            return controller._branch(frozen, action_id, binding, scores, reports)

        branches = [await branch(decision['chosen_action_id'])]
        direction = metric_direction(engine.backend.task)
        trigger = single_branch_trigger()
        combined = combine_scores(branches, direction=direction, ranking_settings={},
                                  aggregation=trigger['aggregation'])
        args = dict(expected_ids=[m['id'] for m in methods], direction=direction,
            metric=engine.backend.task['metric'], reference_scale=frozen['reference_scale'],
            settings=controller.reward_settings)
        original_reports = {m['id']: m['low_report'] for m in methods}
        for value in branches:
            binding = _validate_binding(value['binding'])
            if set(value['reports']) != set(value['scores']):
                raise ValueError('Initial replay scores lack matching method reports')
            for ident, report in value['reports'].items():
                # Native initialization reports precede per-candidate bindings.
                # Only the exact original report and training file may use that
                # format; newly fitted probes must carry the explicit binding.
                same_sample = ((report.get('coreset_version') == binding['version']
                    and report.get('coreset_binding_sha256') == stable_hash(binding))
                    if report.get('coreset_binding_sha256') is not None else
                    (report == original_reports.get(ident)
                     and report.get('fit_train_sha256') == binding['csv_sha256']))
                if report['score'] != value['scores'][ident] or not same_sample:
                    raise ValueError('Initial replay reward refers to a different sample/report')
            value['reward'] = checkpoint_reward(value['scores'], checkpoint['full_scores'], **args)
        reward = checkpoint_reward(combined['scores'], checkpoint['full_scores'], **args)
        fact = dict(task=controller.task_id, source=controller.dataset_source, role=controller.role,
            bundle_hash=bundle_id, bank_hash=frozen['bank_hash'], checkpoint_id=frozen['checkpoint_id'],
            seed=frozen['seed'], sample_hash=combined['sample_hash'], reward=reward,
            decision_id=decision['decision_id'], decision_protocol_hash=stable_hash(frozen),
            sampling_policy_hash=decision['sampling_plan']['protocol_hash'],
            skill_execution=decision['skill_execution'],
            sampling_path=sampling_path(next(a['context'] for a in frozen['actions']
                if a['action_id'] == decision['chosen_action_id']), frozen['methods'], actions=frozen['actions']),
            action=branches[0]['action'],
            branches=branches, branch_count=len(branches), core_scores=combined['scores'],
            full_scores=deepcopy(checkpoint['full_scores']), combination=combined, uncertainty=trigger,
            stage='initialization', search_stage='initialization', execution_kind='checkpoint_replay',
            trajectory_id=str(controller.root.parent), nominated_method_ids=combined['ranking'][:3],
            nominated_method_id=combined['ranking'][0], selected_method_id=None,
            successful_full_method_ids=[], best_full_method_id=None,
            scope='Initial top-three replay; full outcomes only for the original observed three of five')
        relative_fact(fact, frozen.get('baseline_scores'), 'random_proxy')
        _freeze(output/'fact.json', fact)
    with SkillStore(controller.memory_path) as store:
        fact_id = store.record_fact(fact)
    return dict(fact_id=fact_id, fact=fact, checkpoint_id=frozen['checkpoint_id'], new_full_fits=0)
