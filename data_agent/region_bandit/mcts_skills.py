"""Single-action sampling decisions and frozen, low-fit-only MCTS replays."""
from copy import copy, deepcopy
import json
from pathlib import Path

import numpy as np

from ..coreset import export_rows, file_hash
from ..stage1_metrics import metric_direction
from .mcts_skill_reward import FIXED_PARENT_PROTOCOLS, RELATIVE_PROTOCOLS, relative_fact
from .config import normalize_config
from .mcts_rounds import RoundBandit, _binding, _method_bank, _validate_binding, round_feature_schema
from .policy import derive_decision_seed
from .procedural_skills import choose_with_skill_repair
from .regional_sampler import sample_from_action
from .runtime import SAMPLER, _verify_artifacts, _freeze_artifacts
from .schemas import read_json, stable_hash, write_json
from .skill_sampling import mcts_sampling_policy, plan_sampling, single_branch_trigger, combine_scores
from .skill_store import SkillStore, sampling_path
from .mcts_skill_validation import IncompleteValidation

PROTOCOL = 'mcts-procedural-checkpoint-v1'


def _freeze(path, value):
    path = Path(path)
    if path.exists() and read_json(path) != value:
        raise ValueError('Frozen skill checkpoint changed: '+str(path))
    write_json(path, value)
    return value


def with_replay_history(frozen, history=None):
    """Copy decision inputs; a paired replay may pin a shared development history.

    The caller supplies the complete, already frozen history, including only
    local facts available before this checkpoint. Original artifacts are never
    rewritten, and history participates in the ordinary decision identity.
    """
    result = deepcopy(frozen)
    if history is not None:
        result['history'] = deepcopy(history)
    return result


def prepare_replay_inputs(controller, checkpoint, history=None, sampling_seed=None):
    """Freeze a paired validation repeat without changing the model trajectory.

    Only regional within-cell sampling changes. The baseline is the exact
    checkpoint parent sample; quotas, cell schedule, methods, split and observed
    full results stay fixed. Old/new skills must use this same returned input.
    """
    frozen = with_replay_history(checkpoint['frozen'], history)
    if sampling_seed is None:
        return frozen
    if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, (int, np.integer)) or sampling_seed < 0:
        raise ValueError('Validation sampling seed must be a nonnegative integer')
    sampling_seed = int(sampling_seed)
    if sampling_seed == frozen['seed']:
        return frozen
    controller._load()
    current = controller._parent_order(_validate_binding(frozen['parent_binding']))
    if 'node_regions' in frozen:
        controller = copy(controller)
        controller.regions = controller._region_masks(frozen['node_regions'])
        controller.node_sampling = True
        if 'hardness_file' in frozen:
            path = frozen['hardness_file']
            if file_hash(path) != frozen['hardness_sha256']:
                raise ValueError('Frozen node hardness changed')
            controller.inputs = {**controller.inputs, 'hardness': np.load(path, allow_pickle=False)}
    current=np.asarray(frozen.get('round_reference',current),int)
    # Initialization originally used the configuration seed for its cell
    # schedule; later rounds use their checkpoint decision seed.
    schedule_seed = frozen.get('sampler_decision_seed',
        controller.cfg['decision_seed'] if frozen.get('stage') == 'initialization'
        else frozen['decision_seed'])
    frozen['seed'] = sampling_seed
    frozen['sample_orders'] = controller._preview(frozen['actions'], current,
        sampling_seed, schedule_seed, update_search_context=False)
    return frozen


def replay_sample_pool(checkpoint_folder, frozen, original_seed):
    """Separate changed samples while retaining the original resume path."""
    root = Path(checkpoint_folder)/'replay_pool'
    return root if frozen['seed'] == original_seed else root/f'seed_{frozen["seed"]}'


def make_sampling_probe(backend, candidate, binding, probe_id):
    """Clone the actual low-stage constructor, without needing a child full fit."""
    if not probe_id or Path(probe_id).name != probe_id or probe_id in ('.', '..'):
        raise ValueError('Invalid sampling probe ID')
    report = candidate.get('low_report') or {}
    if not report and candidate.get('full_report') and candidate.get('direct_full_certificate'):
        # A direct full comparison may have skipped the low fit entirely. Reuse
        # its verified frozen constructor when another sample actually needs it.
        backend.full_evaluation_identity(candidate)
        report = backend._direct_low_constructor(candidate)
    required = ('unfitted_pipeline_path', 'unfitted_pipeline_sha256', 'pipeline_spec')
    if not all(report.get(k) for k in required):
        raise ValueError('A sampling probe needs a verified original low constructor')
    source = Path(report['unfitted_pipeline_path'])
    if (file_hash(source) != report['unfitted_pipeline_sha256'] or
            file_hash(candidate['code_path']) != candidate['code_sha256'] or
            backend._contract(candidate) != candidate['contract_sha256']):
        raise ValueError('Frozen method changed before sampling replay')
    binding = _validate_binding(binding)
    directory = backend.output/'anchors'/probe_id
    identity = dict(method_id=candidate['id'], code_sha256=candidate['code_sha256'],
        contract_sha256=candidate['contract_sha256'], pipeline_sha256=file_hash(source),
        pipeline_spec=report['pipeline_spec'], binding=binding)
    if (directory/'candidate.json').exists():
        probe = read_json(directory/'candidate.json')
        if probe.get('sampling_probe_identity') != identity:
            raise ValueError('Sampling probe identity changed on resume')
        frozen = probe.get('frozen_stages', {}).get('coreset', {})
        if (probe.get('id') != probe_id or probe.get('anchor_of') != candidate['id'] or
                backend._contract(probe) != candidate['contract_sha256'] or
                file_hash(probe['code_path']) != candidate['code_sha256'] or
                frozen.get('expected_pipeline_spec') != report['pipeline_spec'] or
                frozen.get('initial_pipeline_sha256') != report['unfitted_pipeline_sha256'] or
                file_hash(frozen['initial_pipeline_path']) != report['unfitted_pipeline_sha256']):
            raise ValueError('Frozen sampling probe constructor changed')
        backend._candidate_coreset(probe)
        return probe
    directory.mkdir(parents=True, exist_ok=True)
    code = directory/'method.py'
    code.write_bytes(Path(candidate['code_path']).read_bytes())
    pipeline = directory/'coreset_unfitted.joblib'
    pipeline.write_bytes(source.read_bytes())
    probe = {k: deepcopy(candidate[k]) for k in ('name', 'family', 'idea',
        'parameter_budget_mode', 'parent_baseline', 'modification_direction',
        'allowed_model_classes') if k in candidate}
    probe.update(id=probe_id, parent_id=candidate['id'], anchor_of=candidate['id'],
        branch_id=candidate.get('branch_id', candidate['id']), origin='sampling_probe',
        code_path=str(code), code_sha256=file_hash(code), contract_sha256=candidate['contract_sha256'],
        low_score=None, full_score=None, status='generated', repairs_used=0,
        sampling_probe_identity=identity, frozen_stages={'coreset': {
            'initial_pipeline_path': str(pipeline), 'initial_pipeline_sha256': file_hash(pipeline),
            'expected_pipeline_spec': deepcopy(report['pipeline_spec'])}})
    if backend._contract(probe) != candidate['contract_sha256']:
        raise ValueError('Sampling probe changed the model contract')
    backend.bind_coreset(probe, binding)
    return probe


def replay_nominees(backend, policy, branch, child_ids):
    """Replay the online MCS shortlist and pre-full constructor deduplication."""
    from types import SimpleNamespace
    from ..mcts_search import compare_candidates
    from ..mcts_policy import shortlist_candidates
    from ..mcts_changes import full_comparison_plan
    candidates = [dict(id=i, low_report=branch['reports'][i], low_score=branch['scores'][i])
                  for i in child_ids if i in branch['reports']]
    if not candidates:
        return []
    direction = metric_direction(backend.task)
    mcs = compare_candidates(backend, candidates, SimpleNamespace(**policy), policy['bootstrap_seed'])
    selected = shortlist_candidates(candidates, mcs, direction, limit=policy.get("shortlist_cap", 2), minimum=1)['candidate_ids']
    if policy['full_keys'] is None:
        return selected
    sign = 1 if direction == 'maximize' else -1
    ranked = sorted(candidates, key=lambda c: (-sign*c['low_score'], str(c['id'])))
    keys = {**policy['prior_keys'], **policy['full_keys']}
    plan = full_comparison_plan(ranked, selected, [dict(id=i) for i in policy['prior_keys']],
                                lambda c, phase: keys[c['id']])
    primary = plan['primary_ids']
    trial = next((i for i in plan['eligible_ids'] if i in policy['single_trials']), None)
    if trial:
        primary = [trial] + [i for i in primary if i != trial]
    return primary[:policy.get("shortlist_cap", 2)]


class RoundSkills(RoundBandit):
    """One unified skill library; each decision pins its own immutable bundle."""

    def __init__(self, sampling_root, *, config, memory_path, source, task, role,
                 bundle_id=None, model_identity=None, sampling_policy=None,
                 reward_settings=None, shared_pool_replay=False):
        if shared_pool_replay and role != "validation":
            raise ValueError("Shared pool replay facts must use the validation role")
        self.root = Path(sampling_root).resolve()
        self.cfg = normalize_config(config.get('region_bandit', config))
        self.memory_path = Path(memory_path).resolve()
        self.dataset_source, self.task_id, self.role = source, task, role
        self.bundle_id = bundle_id
        self.sampling = mcts_sampling_policy(sampling_policy)
        self.reward_settings = deepcopy(reward_settings or {})
        self.schema = round_feature_schema()
        self.source = Path(read_json(self.root/'sampling_preparation.json')['source']).resolve()
        self.base_identity = read_json(self.root/'task_identity.json')
        self.identity = dict(protocol=PROTOCOL, config=self.cfg, source=source, task=task,
            role=role, memory_path=str(self.memory_path), model_identity=model_identity,
            sampling_policy=self.sampling, reward_settings=self.reward_settings,
            discovery_sha256=file_hash(self.root/'regions.json'),
            action_plans_sha256=file_hash(self.root/'action_plans.json'),
            implementation_sha256=file_hash(Path(__file__)))
        # bundle_id intentionally does not belong to trajectory identity: a
        # staged campaign may promote a bundle between completed expansions.
        self._loaded = False
        with SkillStore(self.memory_path) as store:
            store.register_sources([source], "development" if shared_pool_replay else role)
            store.bundle(bundle_id)

    def _folder(self, pending):
        return self.root/'mcts_skill_rounds'/f'expansion_{int(pending["iteration"]):04d}'

    def _history(self, store, bank_hash):
        history = store.decision_history(self.dataset_source, bank_hash,
            allow_local=False, limit=None)
        # MCTS changes the method bank on every expansion. Previous local
        # checkpoints remain useful for every role. Role-isolated validation
        # facts stay out of the development optimizer and cross-task retrieval.
        local = [store.fact(row[0]) for row in store.db.execute(
            'SELECT id FROM facts WHERE source=? AND role=? ORDER BY rowid',
            (self.dataset_source, self.role))]
        local = [f for f in local if f['task'] == self.task_id
                 and f.get('trajectory_id') == str(self.root.parent)
                 and f.get('execution_kind') != 'checkpoint_replay']
        for fact in local:
            card = store.compact(store.history_observation(fact))
            card.update(trajectory_id=fact['trajectory_id'],
                execution_kind=fact.get('execution_kind', 'online'),
                scope='task_local' if fact['bank_hash'] == bank_hash else 'previous_local_checkpoint',
                comparison_warning=('Exact method bank; paired support still requires identical samples.'
                    if fact['bank_hash'] == bank_hash else
                    'Different method bank; observational applicability only.'))
            for compact, branch in zip(card.get('branches', []), fact.get('branches', [])):
                compact['sample_hash'] = branch.get('sample_hash')
            history.append(card)
        return history

    def _preview(self, actions, current, seed, decision_seed, *, update_search_context=True):
        orders = {}
        for action in actions:
            if action['type'] == 'baseline':
                order = current.copy()
            else:
                reference = current if getattr(self, 'node_sampling', False) or action.get('variant') == 'gentle_cover' else self.reference
                order, _ = sample_from_action(action, reference, self.inputs['density'],
                    self.inputs['hardness'], self.graph, self.inputs['groups'],
                    self.regions[action['region_id']]['mask'], seed=seed,
                    config=SAMPLER, decision_seed=decision_seed)
            if update_search_context and action['type'] != 'baseline':
                region = action['context']['region_summary']
                # Discovery statistics remain fixed, but undercoverage is
                # measured against the actual parent sample at this node.
                region['coverage_gap'] = max(0., region['q_fraction']-region['c0_fraction'])
                context = action['context']
                for key, value in (('region.coverage_gap', region['coverage_gap']),
                        ('interaction.increase_gap', action['actual_increase_fraction']*region['coverage_gap'])):
                    context['features'][context['feature_names'].index(key)] = value
                    if key in context['retrieval_features']:
                        context['retrieval_features'][key] = value
            orders[action['action_id']] = order.tolist()
            preview = dict(sample_hash=stable_hash(sorted(order.tolist())),
                regions={rid: dict(before=int(r['mask'][current].sum()),
                    after=int(r['mask'][order].sum()),
                    change=int(r['mask'][order].sum()-r['mask'][current].sum()))
                    for rid, r in self.regions.items()},
                sample_replaced_count=len(order)-len(set(order)&set(current)),
                sample_replaced_fraction=1-len(set(order)&set(current))/len(order))
            action['context']['action_summary']['cross_region_preview'] = {str(seed): preview}
            if update_search_context:
                action['context']['search_stage'] = 'mcts_expansion'
                action['context'].setdefault('search_features', {})['is_initialization'] = 0.
                action['context']['proxy_scope'] = ('Frozen initialization-region evidence; '
                    'not measurements of the new child implementations.')
                if getattr(self, 'node_sampling', False):
                    action['context']['proxy_scope'] = 'Current parent and child models trained on the inherited coreset, compared on held-out Q rows.'
        return orders

    async def _choose(self, engine, folder, frozen, bundle_id):
        path = folder/'decision.json'
        if path.exists():
            decision = read_json(path)
            if (decision['decision_protocol_hash'] != stable_hash(frozen) or
                    decision['skill_execution']['skill_bundle_hash'] != bundle_id):
                raise ValueError('Skill decision inputs or pinned bundle changed')
            return decision
        with SkillStore(self.memory_path) as store:
            bundle = store.bundle(bundle_id)
        calls, fatal = 0, []
        async def request(prompt):
            nonlocal calls
            calls += 1
            if calls > 2:
                raise RuntimeError('Skill decision repair allowance exhausted')
            directory = folder/'advisor'/str(calls)
            value = json.loads(prompt)
            if (directory/'parsed_result.json').exists():
                if read_json(directory/'request.json') != value:
                    raise ValueError('Replayed skill request changed')
                return read_json(directory/'parsed_result.json')
            interrupted = (directory/'request.json').exists()
            _freeze(directory/'request.json', value)
            if interrupted:
                raise RuntimeError('Interrupted skill API attempt has no committed response; use baseline fallback')
            engine.state['active_evaluation'] = {'operation': 'region_skill',
                                               'checkpoint_id': frozen['checkpoint_id']}
            try:
                fn = getattr(engine.backend, '_request_once', engine.backend._request)
                result = await fn(prompt, directory/'response.json')
                _freeze(directory/'parsed_result.json', result)
                return result
            except Exception as error:
                if type(error).__name__ in ('SearchBudgetExceeded', 'TaskDeadlineExceeded'):
                    fatal.append(error)
                raise
            finally:
                engine.state.pop('active_evaluation', None)
        advisor = await choose_with_skill_repair([a['context'] for a in frozen['actions']],
            bundle, request, history=frozen['history'], sampling=self.sampling)
        if fatal:
            raise fatal[0]
        execution = {k: advisor[k] for k in ('skill_bundle_hash', 'skills_used', 'checks',
            'reason', 'decision_status', 'error', 'alternative_action_id', 'skill_proposals')}
        decision = dict(actions=frozen['actions'], chosen_action_id=advisor['chosen_action_id'],
            skill_execution=execution, decision_seed=frozen['decision_seed'],
            sampling_seeds=[frozen['seed']], decision_protocol_hash=stable_hash(frozen),
            decision_id=stable_hash(dict(checkpoint=frozen['checkpoint_id'], bundle=bundle_id)),
            checkpoint_id=frozen['checkpoint_id'], sampling_seed=frozen['seed'])
        decision['sampling_plan'] = plan_sampling(decision, self.sampling)
        _freeze(folder/'advisor.json', advisor)
        return _freeze(path, decision)

    def _materialize(self, engine, folder, frozen, action_id):
        action = next(a for a in frozen['actions'] if a['action_id'] == action_id)
        if action['type'] == 'baseline' and 'round_reference' not in frozen:
            return _validate_binding(frozen['parent_binding'])
        directory = folder/'samples'/action_id
        path = directory/'binding.json'
        if path.exists():
            return _validate_binding(read_json(path))
        order = np.asarray(frozen['sample_orders'][action_id], dtype=int)
        expected = action['context']['action_summary']['cross_region_preview'][str(frozen['seed'])]['sample_hash']
        if stable_hash(sorted(order.tolist())) != expected:
            raise ValueError('Frozen sample preview changed')
        positions = np.sort(self.qpos[order])
        csv, indices = directory/'coreset.csv', directory/'coreset_indices.json'
        write_json(indices, dict(train_positions=positions.tolist(), selection_order_q_indices=order.tolist()))
        export_rows(engine.backend.train_path, [(csv, positions)])
        binding = _binding(csv, indices)
        _freeze(path, binding)
        return binding

    def _region_masks(self, specs):
        regions = {}
        for region in specs:
            mask = np.zeros(len(self.qpos), dtype=bool)
            mask[np.asarray(region['indices'], dtype=int)] = True
            regions[region['id']] = {**region, 'mask': mask}
        return regions

    async def _node_evidence(self, engine, pending, methods, binding, folder, current, seed):
        from .execution import predict_cached
        from .runtime import _load_predictions
        from ..multi_proxy_io import read_typed
        from .action_planner import plan_actions
        from .regions import build_regions

        if pending.get('search_stage'):
            pending['reuse_equivalent_parameter_low'] = True
        scores, reports = await self._probe_scores(engine, pending, methods, binding,
            'parent-reference:'+str(pending['iteration']))
        engine.check_sampling_budget()
        task = engine.backend.task_context
        classes = task['classes']
        frame = read_typed(self.source/'Q_features.csv', task['csv_schema'])
        labels = read_typed(engine.backend.train_path, task['csv_schema'], usecols=[task['target']])
        frame[task['target']] = labels.iloc[self.qpos][task['target']].to_numpy()
        heldout = np.setdiff1d(np.arange(len(frame)), current)
        rules = read_json(self.root/'regions.json')['discovery']['candidate_rules']
        predictions = []
        measured = [m for m in methods if m['id'] in reports]
        for method in measured:
            report = reports[method['id']]
            result = predict_cached(method['code_path'], report['fitted_pipeline_path'], self.source/'Q_features.csv',
                task, classes, folder/'reference_predictions'/method['id'])
            if result['status'] != 'ok':
                raise RuntimeError('Parent-sample diagnostic prediction failed: '+str(result['error']))
            predictions.append(_load_predictions(result['artifact'], classes))
        from ..multi_proxy import error_importance
        from ..density_hardness_ablation import group_percentiles
        from .context import admission_evidence
        hardness = self.inputs['hardness'].copy()
        self.summary = dict(self.summary)
        self.summary.pop('proxy_disagreement', None)
        if len(heldout) and predictions:
            y = frame[task['target']].to_numpy()[heldout]
            ranks = np.zeros(len(heldout))
            classification = task['problem_type'] == 'classification'
            for pred in predictions:
                probabilities = pred['probabilities'][heldout] if classification else None
                ranks += error_importance(y, pred['predictions'][heldout], probabilities, classes,
                    task['problem_type'], engine.backend.configuration['error_clip_quantile'])[1]
            hardness[heldout] = group_percentiles(ranks / len(predictions), self.inputs['groups'][heldout])
            if len(predictions) >= 2:
                key = 'probabilities' if classification else 'predictions'
                spread = np.std([p[key][heldout] for p in predictions], axis=0).mean()
                scale = 1. if classification else max(float(np.std(y)), 1e-12)
                self.summary['proxy_disagreement'] = float(np.clip(spread/scale, 0, 1))
        self.inputs = {**self.inputs, 'hardness': hardness}
        hardness_path = folder/'node_hardness.npy'
        np.save(hardness_path, hardness, allow_pickle=False)
        children = [m['id'] for m in methods if m['id'] != pending['parent_id']]
        self.admission = admission_evidence(children, {i: scores[i] for i in children if i in scores},
            1, metric_direction(task), 'parent-coreset V_core scores; primary rollout before optional extras')
        guide={}
        if (self.source/'validation_guidance.json').exists():
            from ..guided_coreset import row_errors,guided_reference,protected_plans
            children=[m['id'] for m in measured if m['id']!=pending['parent_id']]
            errors=[row_errors(engine.backend.load_prediction(reports[c]),engine.backend.y_core,classes) for c in children]
            current,selection=guided_reference(self.source,task,frame,self.inputs,self.graph,errors,seed)
            guide=dict(round_reference=current.tolist(),validation_guidance={**selection,'methods':children})
            heldout=np.setdiff1d(heldout,current)
        regions = []
        diagnostic = {'candidate_rule_count': len(rules), 'status': 'insufficient_models_or_heldout_rows'}
        if len(heldout) >= 2 and len(measured) >= 2:
            discovered = build_regions(frame, task, predictions, current, self.inputs['components'],
                self.inputs['density'], self.inputs['hardness'], config=self.cfg, seed=seed,
                candidate_rules=rules, evidence_indices=heldout, row_ids=self.qpos)
            regions, diagnostic = discovered['regions'], discovered['discovery']
            for region in regions:
                if region.get('primary_pair'):
                    region['primary_pair_method_ids'] = [measured[i]['id'] for i in region['primary_pair']]
        class_ids = None
        if classes:
            lookup = {str(c): i for i, c in enumerate(classes)}
            class_ids = np.asarray([lookup[str(y)] for y in frame[task['target']]])
        if guide:
            self.plans=protected_plans(regions,current,self.inputs['groups'],class_ids,selection['protected'],
                config=self.cfg,sampler_config=SAMPLER)['actions']
        else:
            self.plans = plan_actions(regions, current, self.inputs['groups'], class_ids,
            size=len(current), config=self.cfg, sampler_config=SAMPLER)['actions']
        self.regions = {r['id']: r for r in regions}
        _freeze(folder/'regional_evidence.json', dict(method_ids=[m['id'] for m in measured],
            missing_baseline_methods=[m['id'] for m in methods if m['id'] not in reports],
            baseline_binding=binding, heldout_rows=len(heldout), discovery=diagnostic))
        return dict(**guide,hardness_file=str(hardness_path), hardness_sha256=file_hash(hardness_path),
            baseline_scores=scores, baseline_reports=reports,
            node_regions=[{k: v for k, v in r.items() if k != 'mask'} for r in regions])

    async def prepare(self, engine, pending, candidates):
        engine.check_wall()
        self._load()
        self = copy(self)
        self.node_sampling = self.reward_settings.get('version') in RELATIVE_PROTOCOLS or (self.source/'validation_guidance.json').exists()
        folder = self._folder(pending)
        folder.mkdir(parents=True, exist_ok=True)
        _verify_artifacts(folder)
        parent = engine.state['candidates'][pending['parent_id']]
        binding = self._parent_binding(engine.backend, parent)
        bank = _method_bank([parent, *candidates])
        identities = dict(parent=engine.backend.full_evaluation_identity(parent), methods=bank,
                          binding=binding, controller=self.identity)
        _freeze(folder/'identities.json', identities)
        frozen_path = folder/'predecision.json'
        if frozen_path.exists():
            frozen = read_json(frozen_path)
        else:
            if any(c.get('low_report') or c.get('full_report') for c in candidates):
                raise ValueError('Sampling must be selected before current child fits')
            current = self._parent_order(binding)
            decision_seed = derive_decision_seed(self.cfg['decision_seed'], identities)
            seed = derive_decision_seed(self.cfg['sampling_seeds'][0], dict(round_seed=decision_seed))
            evidence = {}
            if self.node_sampling:
                evidence = await self._node_evidence(engine, pending, [parent, *candidates], binding,
                    folder, current, decision_seed)
            current=np.asarray(evidence.get("round_reference",current),int)
            actions, rejected = self._contexts(engine, pending, _method_bank(candidates), current,
                                               api_available=True)
            orders = self._preview(actions, current, seed, decision_seed)
            if self.sampling.get('enable_gentle_cover'):
                from .gentle_cover import plan_gentle_cover
                variants = []
                for cover in actions:
                    if cover['type'] == 'cover':
                        previews = cover['context']['action_summary']['cross_region_preview']
                        gentle = plan_gentle_cover(cover, {seed: current}, previews, self.inputs['groups'],
                                                   self.regions[cover['region_id']]['mask'], decision_seed)
                        if gentle:
                            variants.append(gentle)
                if variants:
                    extra, skipped = self._contexts(engine, pending, _method_bank(candidates), current,
                                                    api_available=True, plans=variants)
                    orders.update(self._preview(extra, current, seed, decision_seed))
                    actions.extend(extra)
                    rejected.extend(skipped)
            with SkillStore(self.memory_path) as store:
                bundle_id = self.bundle_id or store.active
                history = self._history(store, stable_hash(bank))
            frozen = dict(**evidence, checkpoint_id=stable_hash(dict(source=self.dataset_source, task=self.task_id,
                iteration=pending['iteration'], identities=identities)), bank_hash=stable_hash(bank),
                parent_id=parent['id'], parent_binding=binding, methods=bank, actions=actions,
                rejected_actions=rejected, sample_orders=orders, seed=seed,
                decision_seed=decision_seed, history=history, initial_bundle_id=bundle_id,
                iteration=pending['iteration'], round_number=pending['round_number'])
            if pending.get('search_stage'):
                frozen['reuse_equivalent_parameter_low'] = True
            _freeze(frozen_path, frozen)
        if not (folder/'decision.json').exists():
            engine.check_sampling_budget()
        decision = await self._choose(engine, folder, frozen, frozen['initial_bundle_id'])
        from .mcts_skill_reward import practical_tolerance
        tolerance = practical_tolerance(engine.backend.task['metric'],
            abs(parent['full_report']['score']), settings=self.reward_settings)
        for candidate in candidates:
            candidate['sampling_practical_tolerance'] = tolerance
        if not (folder/'binding.json').exists():
            engine.check_sampling_budget()
        binding = self._materialize(engine, folder, frozen, decision['chosen_action_id'])
        if 'baseline_reports' in frozen:
            if binding == frozen['parent_binding'] and parent['id'] in frozen['baseline_reports']:
                parent_report = frozen['baseline_reports'][parent['id']]
                pending['region_anchor'] = pending['region_probes'][parent_report['candidate_id']]
            for candidate in candidates:
                report = frozen['baseline_reports'].get(candidate['id'])
                if report is None:
                    continue
                candidate.setdefault('frozen_stages', {}).setdefault('coreset', dict(
                    initial_pipeline_path=report['unfitted_pipeline_path'],
                    initial_pipeline_sha256=report['unfitted_pipeline_sha256'],
                    expected_pipeline_spec=report['pipeline_spec']))
                if binding == frozen['parent_binding'] and not candidate.get('low_report'):
                    engine.backend.bind_coreset(candidate, binding)
                    engine.backend._bind_measured_low(candidate, report)
        if self.reward_settings.get('version') in FIXED_PARENT_PROTOCOLS and binding != frozen['parent_binding']:
            pending['skip_parent_action_fit'] = True
            pending['parent_sample_probe_ids'] = {
                method_id: report['candidate_id'] for method_id, report in frozen['baseline_reports'].items()
            }
        _freeze(folder/'binding.json', binding)
        files = [folder/'identities.json', frozen_path, folder/'decision.json', folder/'advisor.json', folder/'binding.json']
        if 'node_regions' in frozen:
            files.append(folder/'regional_evidence.json')
        if 'hardness_file' in frozen:
            files.append(Path(frozen['hardness_file']))
        _freeze_artifacts(folder, 'skill_decision', files)
        pending['skill_checkpoint_folder'] = str(folder)
        if frozen.get('reuse_equivalent_parameter_low'):
            pending['reuse_equivalent_parameter_low'] = True
        return binding

    async def _probe_scores(self, engine, pending, methods, binding, tag, *, allow_partial=False):
        scores, reports = {}, {}
        probes = pending.setdefault('region_probes', {})
        reuse = pending.get('reuse_equivalent_parameter_low')
        parent_reference = tag.startswith('parent-reference:')
        order = methods
        if reuse:
            # A shared parent fit is the natural reference for several children.
            order = sorted(methods, key=lambda c: c['id'] != pending.get('parent_id'))
        measured = []
        for candidate in order:
            probe_id = 'skill_'+stable_hash(dict(tag=tag, method=candidate['id'],
                sample=binding['version']))[:24]
            if parent_reference:
                # The backend's sibling duplicate check groups by the ID prefix.
                # Reference fits at other nodes are independent probe contexts.
                probe_id += '_reference'
            if probe_id not in probes:
                saved = engine.backend.output/'anchors'/probe_id/'candidate.json'
                old_probe = saved.exists() and read_json(saved).get('origin') == 'sampling_probe'
                if candidate.get('full_report') and (candidate.get('low_report') or candidate.get('direct_full_certificate')) and not old_probe:
                    # The backend validates the full cache identity; an exact
                    # parent baseline reuses predictions without another fit.
                    probes[probe_id] = engine.backend.make_coreset_anchor(candidate, binding, probe_id)
                elif not candidate.get('low_report') and candidate.get('sampling_prepared') and parent_reference:
                    directory = saved.parent
                    directory.mkdir(parents=True, exist_ok=True)
                    probe = deepcopy(candidate)
                    code = directory/'method.py'
                    code.write_bytes(Path(candidate['code_path']).read_bytes())
                    probe.update(id=probe_id, anchor_of=candidate['id'], code_path=str(code),
                        origin='parent_reference_probe')
                    engine.backend.bind_coreset(probe, binding)
                    probes[probe_id] = probe
                else:
                    probes[probe_id] = make_sampling_probe(engine.backend, candidate, binding, probe_id)
                engine.checkpoint()
            probe = probes[probe_id]
            if reuse and candidate.get('implementation_mode') == 'inherited_parameters' and not probe.get('low_report'):
                for reference, source in measured:
                    if engine.backend.prepare_equivalent_low(probe, source, method=candidate, reference=reference):
                        engine.checkpoint()
                        break
            report = await engine.evaluate(probe, 'low')
            if report is None:
                if parent_reference or allow_partial:
                    continue
                raise IncompleteValidation('Incomplete alternate/replay fit; no paired policy conclusion')
            scores[candidate['id']], reports[candidate['id']] = report['score'], deepcopy(report)
            measured.append((candidate, probe))
        return ({c['id']: scores[c['id']] for c in methods if c['id'] in scores},
                {c['id']: reports[c['id']] for c in methods if c['id'] in reports})

    def _branch(self, frozen, action_id, binding, scores, reports):
        action = next(a for a in frozen['actions'] if a['action_id'] == action_id)
        return dict(action={k: action.get(k) for k in ('action_id', 'type', 'region_id', 'size',
                'quota_table', 'feasible_region_count', 'current_parent_region_count',
                'actual_increase_fraction')},
            binding=binding, scores=scores, reports=reports,
            cross_region_preview=action['context']['action_summary']['cross_region_preview'],
            sample_hash=action['context']['action_summary']['cross_region_preview'][str(frozen['seed'])]['sample_hash'])

    async def after_low(self, engine, pending, candidates):
        if not candidates:
            return
        folder = self._folder(pending)
        if not (folder/'decision.json').exists():
            return
        if (folder/'screening.json').exists():
            pending['skill_screening'] = read_json(folder/'screening.json')
            return
        frozen, decision = read_json(folder/'predecision.json'), read_json(folder/'decision.json')
        parent = engine.state['candidates'][pending['parent_id']]
        anchor = pending.get('region_anchor') or {}
        if candidates and candidates[0].get('sampling_practical_tolerance') is not None:
            anchor['sampling_practical_tolerance'] = candidates[0]['sampling_practical_tolerance']
        scores = {c['id']: c['low_report']['score'] for c in candidates}
        reports = {c['id']: deepcopy(c['low_report']) for c in candidates}
        if anchor.get('low_report'):
            scores[parent['id']] = anchor['low_report']['score']
            reports[parent['id']] = deepcopy(anchor['low_report'])
        binding = _validate_binding(read_json(folder/'binding.json'))
        branches = [self._branch(frozen, decision['chosen_action_id'], binding, scores, reports)]
        trigger = single_branch_trigger()
        _freeze(folder/'trigger.json', trigger)
        combined = combine_scores(branches, direction=metric_direction(engine.backend.task),
                                  ranking_settings={}, aggregation=trigger['aggregation'])
        screening = dict(branches=branches, combined=combined, branch_count=len(branches), trigger=trigger)
        screening['ambiguity'] = False
        _freeze(folder/'screening.json', screening)
        pending['skill_screening'] = screening
        engine.checkpoint()

    async def after_initial_low(self, engine, candidates):
        """Record the selected initial sample before admitting three full fits."""
        if not candidates:
            return
        from .mcts_skill_reward import practical_tolerance
        tolerance = practical_tolerance(engine.backend.task['metric'],
            getattr(self, 'initial_reference_scale', None), settings=self.reward_settings)
        for candidate in candidates:
            candidate['sampling_practical_tolerance'] = tolerance
        folder = self.root/'mcts_skill_initial'
        _verify_artifacts(self.root)
        _verify_artifacts(folder)
        if (folder/'screening.json').exists():
            engine.state['initial_skill_screening'] = read_json(folder/'screening.json')
            return
        decision = read_json(self.root/'decision.json')
        seed = decision['sampling_seeds'][0]
        actions = {a['action_id']: a for a in decision['actions']}
        primary = actions[decision['chosen_action_id']]
        binding = _binding(engine.backend.coreset_path,
            engine.backend.source_run/f'coreset_{engine.backend.coreset_index}_indices.json')
        scores = {c['id']: c['low_report']['score'] for c in candidates}
        reports = {c['id']: deepcopy(c['low_report']) for c in candidates}
        frozen = dict(actions=decision['actions'], seed=seed)
        branches = [self._branch(frozen, primary['action_id'], binding, scores, reports)]
        trigger = single_branch_trigger()
        _freeze(folder/'trigger.json', trigger)
        combined = combine_scores(branches, direction=metric_direction(engine.backend.task),
            ranking_settings={}, aggregation=trigger.get('aggregation', 'equal_metric_mean_v1'))
        screening = dict(branches=branches, combined=combined, branch_count=len(branches), trigger=trigger,
            decision_id=decision['decision_id'], seed=seed,
            decision_protocol_hash=stable_hash(dict(decision=decision, methods=_method_bank(candidates))),
            method_ids=[c['id'] for c in candidates], scope='Initialization: three observed full methods of five')
        _freeze(folder/'screening.json', screening)
        _freeze_artifacts(folder, 'initial_skill_screening', [folder/'screening.json', folder/'trigger.json'])
        engine.state['initial_skill_screening'] = screening
        engine.checkpoint()

    def _fact(self, engine, frozen, decision, screening, methods, *, role, folder, execution=None, replay_primary=None):
        from .mcts_skill_reward import checkpoint_reward
        full = {c['id']: c['full_report']['score'] for c in methods if c.get('full_report')}
        expected = [m['id'] for m in frozen['methods']]
        args = dict(expected_ids=expected, direction=metric_direction(engine.backend.task),
            metric=engine.backend.task['metric'], anchor_id=frozen['parent_id'], settings=self.reward_settings)
        branches = deepcopy(screening['branches'])
        if len(branches) != 1:
            raise ValueError('Skill MCTS requires single-branch screening')
        for branch in branches:
            binding = _validate_binding(branch['binding'])
            if set(branch['reports']) != set(branch['scores']):
                raise ValueError('Branch scores lack matching method reports')
            for ident, report in branch['reports'].items():
                if (report['score'] != branch['scores'][ident] or
                        report.get('coreset_version') != binding['version'] or
                        report.get('coreset_binding_sha256') != stable_hash(binding)):
                    raise ValueError('Branch reward refers to a different sample/report')
            branch['reward'] = checkpoint_reward(branch['scores'], full, **args)
        combined = screening['combined']
        reward = checkpoint_reward(combined['scores'], full, **args)
        children = set(expected)-{frozen['parent_id']}
        nominees = [i for i in combined['ranking'] if i in children]
        if execution is not None:
            nominees = []
        if execution is not None and execution.get('shortlist', {}).get('candidate_ids'):
            # Single-branch MCS can exclude the raw score leader. Persist the
            # actual pre-full nomination, even when its fit later fails.
            nominees = execution['shortlist']['candidate_ids']
            if not set(nominees) <= children:
                raise ValueError('Primary shortlist includes methods outside the frozen children')
        ranking_leader = next((i for i in combined['ranking'] if i in children), None)
        if execution is not None and 'full_comparison_plan' in execution:
            nominees = execution['full_comparison_plan']['primary_ids']
        if execution is None:
            nominees = replay_primary or []
        primary_reward = branches[0]['reward']['reward']
        chosen = next(a for a in frozen['actions'] if a['action_id'] == decision['chosen_action_id'])
        sign = 1 if metric_direction(engine.backend.task) == 'maximize' else -1
        successful = sorted(children & set(full)) if execution is not None else []
        primary_success = execution.get('primary_success_id') if execution is not None else None
        if primary_success is not None and primary_success not in successful:
            raise ValueError('Recorded successful primary lacks its full result')
        best_full = min(successful, key=lambda i: (-sign*full[i], str(i))) if successful else None
        fact = dict(task=self.task_id, source=self.dataset_source, role=role,
            bundle_hash=decision['skill_execution']['skill_bundle_hash'], bank_hash=frozen['bank_hash'],
            seed=frozen['seed'], sample_hash=combined['sample_hash'], reward=reward,
            decision_id=decision['decision_id'], checkpoint_id=frozen['checkpoint_id'],
            decision_protocol_hash=decision['decision_protocol_hash'],
            skill_execution=decision['skill_execution'], sampling_policy_hash=decision['sampling_plan']['protocol_hash'],
            action=branches[0]['action'], branches=branches, branch_count=1, branch_reward_delta=None,
            combination_gain=None if reward['reward'] is None or primary_reward is None else reward['reward']-primary_reward,
            core_scores=combined['scores'], full_scores=full, combination=combined,
            uncertainty=screening['trigger'], nominated_method_id=nominees[0] if nominees else None,
            nomination_protocol="scheduled-primary-v1", ranking_leader_id=ranking_leader, selected_method_id=primary_success, primary_success_id=primary_success,
            successful_full_method_ids=successful, best_full_method_id=best_full,
            execution_kind='online' if execution is not None else 'checkpoint_replay',
            evidence_scope=reward.get('evidence_scope'), expected_method_ids=expected,
            search_stage='mcts_expansion', round_number=frozen['round_number'],
            trajectory_id=str(self.root.parent),
            sampling_path=sampling_path(chosen['context'], frozen['methods'], actions=frozen['actions'],
                                        regions=frozen.get('node_regions', ())),
            decision_context=deepcopy(chosen['context'].get('region_summary', {})),
            search_features=deepcopy(chosen['context'].get('search_features', {})),
            cross_region_preview=deepcopy(chosen['context']['action_summary']['cross_region_preview']),
            artifact_directory=str(folder), model_search_gain={i: sign*(full[i]-full[frozen['parent_id']])
                for i in full if i != frozen['parent_id'] and frozen['parent_id'] in full},
            model_search_gain_interpretation='Positive means improvement in the primary metric; separate from sampling rank reward.')

        return relative_fact(fact, frozen.get('baseline_scores'), 'parent_coreset')

    def finish(self, engine, pending, candidates):
        folder = self._folder(pending)
        if not (folder/'decision.json').exists():
            return dict(status='no_skill_decision', policy_updated=False)
        _verify_artifacts(folder)
        if (folder/'feedback_commit.json').exists():
            return read_json(folder/'feedback_commit.json')
        frozen, decision = read_json(folder/'predecision.json'), read_json(folder/'decision.json')
        parent = engine.state['candidates'][pending['parent_id']]
        expected = {m['id'] for m in frozen['methods']}
        methods = [c for c in [parent, *candidates] if c['id'] in expected]
        if _method_bank(methods) != frozen['methods']:
            raise ValueError('Observed method bank changed after the skill decision')
        identities = read_json(folder/'identities.json')
        if engine.backend.full_evaluation_identity(parent) != identities['parent']:
            raise ValueError('Frozen parent full evidence changed')
        for c in methods:
            if c['id'] == parent['id'] or not c.get('full_report'):
                continue
            identity = engine.backend.full_evaluation_identity(c)
            if any(identity.get(k) != identities['parent'].get(k) for k in
                   ('train_sha256', 'validation_sha256', 'seed', 'metric', 'metric_params')):
                raise ValueError('Child and parent full protocols differ')
        if not (folder/'screening.json').exists():
            anchor = pending.get('region_anchor') or {}
            scores = {parent['id']: anchor['low_report']['score']} if anchor.get('low_report') else {}
            reports = {parent['id']: anchor['low_report']} if anchor.get('low_report') else {}
            primary = self._branch(frozen, decision['chosen_action_id'],
                read_json(folder/'binding.json'), scores, reports)
            trigger = dict(run_alternative=False, reasons=['no_successful_child_low_fits'])
            combined = dict(scores=scores, sample_hash=primary['sample_hash'],
                aggregation='equal_metric_mean_v1', branch_count=1, ranking=list(scores),
                branch_rankings=[list(scores)], reversed_method_pairs=[])
            screening = dict(branches=[primary], combined=combined, branch_count=1, trigger=trigger)
            _freeze(folder/'screening.json', screening)
            _freeze(folder/'trigger.json', trigger)
        else:
            screening = read_json(folder/'screening.json')
        fact = self._fact(engine, frozen, decision, screening, methods, role=self.role, folder=folder,
                          execution=pending)
        checkpoint = dict(frozen=frozen, decision=decision, screening=screening,
                          methods=deepcopy(methods), full_scores=fact['full_scores'], fact=fact)
        if pending.get('nomination_policy') is not None:
            checkpoint['nomination_policy'] = pending['nomination_policy']
        _freeze(folder/'checkpoint.json', checkpoint)
        _freeze_artifacts(folder, 'skill_checkpoint', [folder/'checkpoint.json', folder/'screening.json',
            folder/'trigger.json'])
        with SkillStore(self.memory_path) as store:
            fact_id = store.record_fact(fact)
        relative = {}
        if 'baseline_kind' in fact:
            relative.update(baseline_kind=fact['baseline_kind'], reward_status=fact['reward']['status'],
                            order_gain=fact['reward']['order_gain'])
        return _freeze(folder/'feedback_commit.json', dict(**relative, status='completed', fact_id=fact_id,
            checkpoint_id=frozen['checkpoint_id'], checkpoint_folder=str(folder),
            reward=fact['reward']['reward'], observed_method_count=len(fact['full_scores']),
            policy_updated=False, fact_recorded=True, bundle_hash=fact['bundle_hash']))

    async def replay_checkpoint(self, engine, checkpoint_folder, bundle_id, output, *, history=None,
                                sampling_seed=None):
        """Replay a pinned checkpoint: API decisions and low fits, zero full fits.

        Both bundles receive the same frozen history. Optional history is a
        proposal-time development snapshot plus checkpoint-preceding local
        facts; no later local outcomes are fetched. Unknown results stay unknown.
        """
        self._load()
        _verify_artifacts(Path(checkpoint_folder))
        checkpoint = read_json(Path(checkpoint_folder)/'checkpoint.json')
        frozen = prepare_replay_inputs(self, checkpoint, history, sampling_seed)
        folder = Path(output).resolve()
        _freeze(folder/'checkpoint_identity.json', dict(checkpoint=stable_hash(checkpoint), bundle=bundle_id,
            replay_inputs=stable_hash(frozen)))
        decision = await self._choose(engine, folder, frozen, bundle_id)
        if (folder/'fact.json').exists():
            fact = read_json(folder/'fact.json')
        else:
            methods = [c for c in checkpoint['methods'] if c.get('low_report') or
                (frozen.get('reuse_equivalent_parameter_low') and c['id'] == frozen['parent_id'] and
                 c.get('full_report') and c.get('frozen_stages', {}).get('coreset'))]
            # Keep probes attached to resumable state for accounting/recovery,
            # without inserting them into the MCTS tree or full rollouts.
            replay_state = engine.state.setdefault('skill_validation', {}).setdefault(
                stable_hash(dict(checkpoint=frozen['checkpoint_id'], bundle=bundle_id,
                                 replay_inputs=stable_hash(frozen))), {})
            if frozen.get('reuse_equivalent_parameter_low'):
                replay_state.update(parent_id=frozen['parent_id'], reuse_equivalent_parameter_low=True)
            async def branch(action_id):
                preview = next(a for a in frozen['actions'] if a['action_id'] == action_id)[
                    'context']['action_summary']['cross_region_preview'][str(frozen['seed'])]
                previous = next((b for b in checkpoint['screening']['branches']
                                 if b['sample_hash'] == preview['sample_hash']), None)
                if previous:
                    result = deepcopy(previous)
                    result['action'] = next({k: a.get(k) for k in ('action_id', 'type', 'region_id', 'size',
                        'quota_table', 'feasible_region_count', 'current_parent_region_count', 'actual_increase_fraction')}
                        for a in frozen['actions'] if a['action_id'] == action_id)
                    result['cross_region_preview'] = {str(frozen['seed']): deepcopy(preview)}
                    return result
                # A shared frozen sample/probe pool lets old and new bundles
                # selecting the same previously unmeasured action reuse a fit.
                pool = replay_sample_pool(checkpoint_folder, frozen, checkpoint['frozen']['seed'])
                binding = self._materialize(engine, pool, frozen, action_id)
                probe_methods = methods
                if self.reward_settings.get('version') in FIXED_PARENT_PROTOCOLS and binding != frozen['parent_binding']:
                    probe_methods = [m for m in methods if m['id'] != frozen['parent_id']]
                if binding == frozen['parent_binding'] and all(m['id'] in frozen.get('baseline_reports', {}) for m in methods):
                    reports = {m['id']: frozen['baseline_reports'][m['id']] for m in methods}
                    scores = {i: r['score'] for i, r in reports.items()}
                else:
                    scores, reports = await self._probe_scores(engine, replay_state, probe_methods, binding,
                                                               frozen['checkpoint_id'])
                return self._branch(frozen, action_id, binding, scores, reports)
            branches = [await branch(decision['chosen_action_id'])]
            trigger = single_branch_trigger()
            _freeze(folder/'trigger.json', trigger)
            combined = combine_scores(branches, direction=metric_direction(engine.backend.task),
                ranking_settings={}, aggregation=trigger['aggregation'])
            screening = dict(branches=branches, combined=combined, branch_count=len(branches), trigger=trigger)
            policy = checkpoint.get('nomination_policy')
            primary = (replay_nominees(engine.backend, policy, branches[0],
                       [m['id'] for m in methods if m['id'] != frozen['parent_id']])
                       if policy is not None else [])
            fact = self._fact(engine, frozen, decision, screening, checkpoint['methods'],
                              role=self.role, folder=folder, replay_primary=primary)
            if fact['full_scores'] != checkpoint['full_scores']:
                raise ValueError('Replay changed frozen observed full evidence')
            _freeze(folder/'fact.json', fact)
        with SkillStore(self.memory_path) as store:
            fact_id = store.record_fact(fact)
        return dict(fact_id=fact_id, fact=fact, checkpoint_id=frozen['checkpoint_id'],
                    new_full_fits=0)
