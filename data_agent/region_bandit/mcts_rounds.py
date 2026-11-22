"""One frozen regional intervention per MCTS expansion, with a parent anchor.

Discovery and original quota constraints stay fixed. Only the current method
bank, search context and selected coreset change. No new proxy fit or test read
is performed here; callers evaluate the parent anchor on the chosen sample.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from ..coreset import export_rows, file_hash
from ..stage1_metrics import metric_direction
from .config import normalize_config, reward_protocol_hash
from .context import default_feature_schema, build_context
from .llm_advisor import score_actions_with_api
from .memory_retrieval import retrieve_memory
from .memory_store import MemoryStore
from .policy import select_action, derive_decision_seed
from .regional_sampler import sample_from_action
from .reward import compute_action_reward, anchored_sample_weight
from .runtime import SAMPLER, _verify_artifacts, _freeze_artifacts
from .schemas import stable_hash, read_json, write_json

PROTOCOL = 'mcts-region-parent-anchor-v1'
ROUND_FEATURES = ('progress', 'parent_depth', 'prior_conflict_fraction',
                  'current_replacement_fraction', 'model_diversity')


def round_feature_schema():
    schema = deepcopy(default_feature_schema())
    schema['version'] = PROTOCOL + '-features'
    for key in ROUND_FEATURES:
        schema['features'].append({'name': 'search.' + key, 'scaling': 'clip(x,0,1)', 'missing': 0})
        for action in ('cover', 'focus'):
            schema['features'].append({'name': 'search.' + key + '.' + action,
                                      'scaling': 'clip(x,0,1)', 'missing': 0})
    schema.pop('schema_id')
    schema['schema_id'] = stable_hash(schema)
    return schema


def _binding(csv, indices):
    values = {'csv_sha256': file_hash(csv), 'indices_sha256': file_hash(indices)}
    return {'schema_version': 'candidate-coreset-v1', 'version': stable_hash(values),
            'csv_path': str(Path(csv).resolve()), 'indices_path': str(Path(indices).resolve()), **values}


def _validate_binding(binding):
    actual = _binding(binding['csv_path'], binding['indices_path'])
    if actual != {key: binding.get(key) for key in actual}:
        raise ValueError('Frozen per-candidate coreset binding changed')
    return actual


def _method_bank(candidates):
    rows = []
    for candidate in candidates:
        path = Path(candidate['code_path'])
        rows.append({key: deepcopy(candidate.get(key)) for key in
                     ('id', 'name', 'family', 'idea', 'contract_sha256')})
        rows[-1]['code_sha256'] = file_hash(path) if path.is_file() else None
        rows[-1]['stage_parameters'] = deepcopy(candidate.get('idea', {}).get('stage_parameters', {}))
    return rows


class RoundBandit:
    def __init__(self, sampling_root, *, config, memory_path, model=None,
                 model_identity=None, pricing=None):
        self.root = Path(sampling_root).resolve()
        self.cfg = normalize_config(config.get('region_bandit', config))
        if self.cfg['mode'] == 'evaluate_frozen':
            raise ValueError('Per-expansion region sampling requires a separately predeclared round policy; '
                             'set region_bandit.mcts_rounds.enabled=false for existing frozen evaluations')
        self.memory_path = Path(memory_path).resolve()
        self.schema = round_feature_schema()
        self.source = Path(read_json(self.root/'sampling_preparation.json')['source']).resolve()
        self.base_identity = read_json(self.root/'task_identity.json')
        self.identity = {'protocol': PROTOCOL, 'schema_id': self.schema['schema_id'],
            'reward_protocol_hash': reward_protocol_hash(self.cfg, comparison_scope='mcts_parent_anchor'),
            'configuration': self.cfg, 'memory_path': str(self.memory_path),
            'implementation_sha256': file_hash(Path(__file__)), 'model_identity': model_identity,
            'algorithm_sha256': {name: file_hash(Path(__file__).parent/name) for name in
                ('reward.py', 'config.py', 'context.py', 'policy.py', 'regional_sampler.py',
                 'action_planner.py', 'memory_store.py', 'memory_retrieval.py', 'llm_advisor.py')},
            'discovery_sha256': file_hash(self.root/'regions.json'),
            'action_plans_sha256': file_hash(self.root/'action_plans.json')}
        self._loaded = False

    def _load(self):
        _verify_artifacts(self.root)
        if self._loaded:
            return
        with np.load(self.source/'coverage_selection_inputs.npz', allow_pickle=False) as data:
            self.inputs = {key: data[key].copy() for key in data.files}
        self.qpos = self.inputs['train_positions'].astype(int)
        self.graph = sparse.load_npz(self.source/'similarity_graph.npz')
        self.reference = np.asarray(read_json(self.root/'reference_coresets.json')['seeds'][str(self.cfg['sampling_seeds'][0])]['q_indices'], int)
        self.regions = {}
        for region in read_json(self.root/'regions.json')['regions']:
            value = deepcopy(region)
            value['mask'] = np.zeros(len(self.qpos), dtype=bool)
            value['mask'][np.asarray(value['indices'], int)] = True
            self.regions[value.get('region_id', value.get('id'))] = value
        self.plans = read_json(self.root/'action_plans.json')['actions']
        self.summary = read_json(self.root/'task_summary.json')
        self._loaded = True

    def _folder(self, pending):
        return self.root/'mcts_rounds'/f'expansion_{int(pending["iteration"]):04d}'

    def _parent_binding(self, backend, parent):
        binding = parent.get('coreset_binding') or parent.get('inherited_region_binding')
        if binding:
            return _validate_binding(binding)
        return _binding(backend.coreset_path, backend.source_run/f'coreset_{backend.coreset_index}_indices.json')

    def _parent_order(self, binding):
        positions = read_json(binding['indices_path'])['train_positions']
        lookup = {int(position): index for index, position in enumerate(self.qpos)}
        if len(positions) != len(self.reference) or len(set(positions)) != len(positions):
            raise ValueError('Parent coreset has a different size or duplicate rows')
        try:
            return np.asarray([lookup[int(position)] for position in positions], int)
        except KeyError as error:
            raise ValueError('Parent coreset contains rows outside frozen Q') from error

    def _contexts(self, engine, pending, bank, current, *, api_available, plans=None):
        search = deepcopy(engine.search_context())
        parent = engine.state['candidates'][pending['parent_id']]
        comparison_bank = _method_bank([parent]) + bank
        conflict = search.get('proxy_reliability', {})
        compared = conflict.get('comparable_pairs', conflict.get('pair_count', 0))
        conflicts = conflict.get('conflicts', conflict.get('conflict_count', 0))
        if not isinstance(compared, (int, float)):
            compared = 0
        if not isinstance(conflicts, (int, float)):
            conflicts = 0
        depth = len(engine.ancestors(engine.state['candidates'][pending['parent_id']]))
        values = {'progress': len(engine.state.get('expansions', []))/max(1, engine.cfg.max_expansions),
            'parent_depth': depth/max(1, engine.cfg.max_depth),
            'prior_conflict_fraction': conflicts/max(1, compared),
            'current_replacement_fraction': 1-len(set(current)&set(self.reference))/len(self.reference),
            'model_diversity': len({method['family'] for method in comparison_bank})/len(comparison_bank)}
        bank_context = {'methods': comparison_bank, 'formal_budgets': {
            'coreset': engine.backend.evaluation_budget, 'full_train': engine.backend.full_train_budget,
            'search_state': search,
            'sampling_protocol': {'baseline': 'reuse exact parent sample',
                'alternatives': 'frozen original-reference quotas; no cumulative quota drift',
                'reward': 'parent anchor plus one or two successful full-trained children'}}}
        if getattr(self, 'node_sampling', False):
            bank_context['formal_budgets']['admission'] = self.admission
            bank_context['formal_budgets']['sampling_protocol']['alternatives'] = 'quota-constrained edits relative to the inherited parent sample'
        result, rejected = [], []
        for original in (self.plans if plans is None else plans):
            action = deepcopy(original)
            region = deepcopy(self.regions.get(action.get('region_id')))
            if region is not None:
                editable=np.setdiff1d(current,action.get('protected_q_indices',[]))
                current_count = int(region['mask'][editable].sum())
                if action['feasible_region_count'] <= current_count:
                    rejected.append({'action_id': action['action_id'],
                                     'reason': 'no_positive_increase_over_current_parent',
                                     'current_region_count': current_count})
                    continue
                region['c0_fraction'] = current_count/len(editable)
                action['current_parent_region_count'] = current_count
                # Keep the original reference count/quota table executable;
                # explicit context records the intervention relative to parent.
                action['actual_increase_fraction'] = (action['feasible_region_count']-current_count)/len(editable)
            action['quota_reference'] = 'original_reference' if region is not None else 'exact_parent_sample'
            if getattr(self, 'node_sampling', False):
                action['quota_reference'] = 'exact_parent_sample'
            if action.get('variant') == 'gentle_cover':
                action['quota_reference'] = 'partial_repair_from_current_parent'
            if 'protected_q_indices' in action:
                action['quota_reference'] = 'current_round_editable_reference'
                bank_context['formal_budgets']['sampling_protocol'].update(baseline='fresh 10% MMD + 5% per-method error + 85% coverage sample', alternatives='adjust only the editable 85%; protected rows stay fixed')
            context = build_context(self.summary, bank_context, region, action, api_available=api_available)
            extra = {}
            for key, value in values.items():
                value = min(1., max(0., float(value)))
                extra['search.'+key] = value
                for kind in ('cover', 'focus'):
                    extra['search.'+key+'.'+kind] = value * (action['type'] == kind)
            context['features'].extend(extra.values())
            context['feature_names'].extend(extra.keys())
            context['schema_id'] = self.schema['schema_id']
            context['feature_schema_version'] = self.schema['version']
            context['comparison_scope'] = 'mcts_parent_anchor'
            context['search_summary'] = search
            context['action_summary'].update(quota_reference=action['quota_reference'],
                current_parent_region_count=action.get('current_parent_region_count'),
                increase_relative_to_current_parent=action.get('actual_increase_fraction', 0.))
            context['search_features'] = values
            context['retrieval_features'].update({'search.'+key: float(value) for key, value in values.items()})
            action['context'] = context
            result.append(action)
        return result, rejected

    def _export(self, snapshot):
        directory = self.memory_path.parent/'mcts_rounds_policy'
        write_json(directory/'feature_schema.json', self.schema)
        write_json(directory/'snapshots'/f'{snapshot["snapshot_id"]}.json', snapshot)

    async def prepare(self, engine, pending, candidates):
        """Freeze bank, policy, API advice and action before any child low fit."""
        engine.check_wall()
        self._load()
        folder = self._folder(pending)
        folder.mkdir(parents=True, exist_ok=True)
        _verify_artifacts(folder)
        if not 1 <= len(candidates) <= 3 or len({c['id'] for c in candidates}) != len(candidates):
            raise ValueError('A round requires one to three distinct preflight-valid children')
        if not (folder/'decision.json').exists() and any(
                c.get('low_report') or c.get('full_report') for c in candidates):
            raise ValueError('Regional selection must be frozen before child evaluation')
        parent = engine.state['candidates'][pending['parent_id']]
        parent_binding = self._parent_binding(engine.backend, parent)
        current = self._parent_order(parent_binding)
        bank = _method_bank(candidates)
        anchor_identity = engine.backend.full_evaluation_identity(parent)
        frozen_input = {'protocol': PROTOCOL, 'parent_id': parent['id'], 'parent_binding': parent_binding,
            'anchor_evaluation_identity': anchor_identity, 'methods': bank,
            'round_number': pending['round_number'], 'iteration': pending['iteration'],
            'controller_identity': self.identity}
        input_path = folder/'frozen_input.json'
        if input_path.exists() and read_json(input_path) != frozen_input:
            raise ValueError('Round methods, parent, sample or full anchor changed after freeze')
        write_json(input_path, frozen_input)
        _freeze_artifacts(folder, 'inputs', [input_path])
        identity = {**self.base_identity, 'feature_schema_version': self.schema['version'],
            'method_bank_hash': stable_hash(bank),
            'reward_protocol_hash': self.identity['reward_protocol_hash']}
        with MemoryStore(self.memory_path) as store:
            if (folder/'decision.json').exists():
                decision = read_json(folder/'decision.json')
                if decision['identity'] != identity or decision['input_sha256'] != file_hash(input_path):
                    raise ValueError('Frozen round decision no longer matches its inputs')
            else:
                snapshot_path = folder/'policy_snapshot.json'
                snapshot = read_json(snapshot_path) if snapshot_path.exists() else store.latest_snapshot(
                    identity, schema=self.schema, config=self.cfg)
                write_json(snapshot_path, snapshot)
                actions, rejected = self._contexts(engine, pending, bank, current, api_available=True)
                cases = store.cases(identity, snapshot=snapshot)
                retrieved = []
                for action in actions:
                    retrieved.extend(retrieve_memory(action['context'], cases, identity=identity,
                        max_cases=self.cfg['advisor']['max_history_datasets'],
                        max_distance=self.cfg['advisor']['max_retrieval_distance']))
                unique, seen = [], set()
                for case in sorted(retrieved, key=lambda item: (item['distance'], item['record_id'])):
                    if case['dataset_group_id'] not in seen:
                        unique.append(case); seen.add(case['dataset_group_id'])
                    if len(unique) >= self.cfg['advisor']['max_history_datasets']:
                        break
                advisor_path = folder/'advisor/result.json'
                started_path = folder/'advisor/started.json'
                if advisor_path.exists():
                    advisor = read_json(advisor_path)
                elif started_path.exists():
                    advisor = {'scores': {a['action_id']: 0. for a in actions}, 'api_available': 0,
                        'calls': 0, 'error': 'Interrupted advisor attempt; not repeated'}
                else:
                    fatal = []
                    request_count = 0
                    async def request(prompt):
                        nonlocal request_count
                        request_count += 1
                        if request_count > 2:
                            raise RuntimeError('Advisor initial call plus one repair budget exhausted')
                        value = json.loads(prompt)
                        engine.state['active_evaluation'] = {'operation': 'region_advisor',
                                                            'iteration': pending['iteration']}
                        attempt_dir = folder/'advisor' if request_count == 1 else folder/'advisor/repair_1'
                        write_json(attempt_dir/'started.json', {'attempt': request_count, 'prompt_hash': stable_hash(value)})
                        try:
                            request_once=getattr(engine.backend,'_request_once',engine.backend._request)
                            return await request_once(json.dumps(value, ensure_ascii=False),
                                                      attempt_dir/'response.json')
                        except Exception as error:
                            if type(error).__name__ in ('SearchBudgetExceeded', 'TaskDeadlineExceeded'):
                                fatal.append(error)
                            raise
                        finally:
                            engine.state.pop('active_evaluation', None)
                    advisor = await score_actions_with_api([a['context'] for a in actions], unique, request,
                                                          config=self.cfg['advisor'])
                    if fatal:
                        raise fatal[0]
                write_json(advisor_path, advisor)
                actions, rejected = self._contexts(engine, pending, bank, current,
                                                    api_available=bool(advisor['api_available']))
                seed = derive_decision_seed(self.cfg['decision_seed'], {
                    'protocol': PROTOCOL, 'parent_sample': parent_binding['version'],
                    'method_bank': stable_hash(bank), 'iteration': pending['iteration']})
                decision = select_action(snapshot, actions, advisor['scores'], seed,
                                         mode=self.cfg['mode'], config=self.cfg['policy'])
                decision.update(identity=identity, comparison_scope='mcts_parent_anchor',
                    anchor_method_id=parent['id'], anchor_evaluation_identity=anchor_identity,
                    input_sha256=file_hash(input_path), sampling_seed=derive_decision_seed(
                        self.cfg['sampling_seeds'][0], {'round_seed': seed, 'purpose': 'within_cells'}),
                    frozen_config={'methods': bank, 'sampler': SAMPLER, 'parent_binding': parent_binding},
                    api_available=advisor['api_available'], rejected_actions=rejected)
                decision['decision_id'] = stable_hash({'directory': str(folder), 'input': frozen_input,
                                                       'snapshot': snapshot['snapshot_id']})
                write_json(folder/'retrieved_cases.json', unique)
                write_json(folder/'decision.json', decision)
                _freeze_artifacts(folder, 'decision', [folder/'decision.json', snapshot_path,
                    advisor_path, folder/'retrieved_cases.json'])
            store.freeze_decision(decision)
            chosen = next(action for action in decision['actions'] if action['action_id'] == decision['chosen_action_id'])
            binding_path = folder/'binding.json'
            if binding_path.exists():
                return _validate_binding(read_json(binding_path))
            if chosen['type'] == 'baseline':
                binding = parent_binding
                sampling = {'mechanism': 'exact_parent_sample_reuse', 'q_indices': current.tolist()}
            else:
                engine.check_wall()
                order, sampling = sample_from_action(chosen, self.reference, self.inputs['density'],
                    self.inputs['hardness'], self.graph, self.inputs['groups'],
                    self.regions[chosen['region_id']]['mask'], seed=decision['sampling_seed'],
                    config=SAMPLER, decision_seed=decision['decision_seed'])
                positions = np.sort(self.qpos[order])
                csv, indices = folder/'coreset.csv', folder/'coreset_indices.json'
                write_json(indices, {'train_positions': positions.tolist(),
                    'selection_order_q_indices': order.tolist(), 'sampling': sampling})
                export_rows(engine.backend.train_path, [(csv, positions)])
                binding = _binding(csv, indices)
            write_json(folder/'sampling.json', sampling)
            write_json(binding_path, binding)
            _freeze_artifacts(folder, 'sample', [binding_path, folder/'sampling.json',
                Path(binding['csv_path']), Path(binding['indices_path'])])
            self._export(read_json(folder/'policy_snapshot.json'))
            return binding

    def finish(self, engine, pending, candidates):
        """Commit one resolved anchored observation, including uninformative ones."""
        folder = self._folder(pending)
        if not (folder/'decision.json').exists():
            return {'status': 'no_frozen_round_decision', 'policy_updated': False}
        _verify_artifacts(folder)
        decision = read_json(folder/'decision.json')
        parent = engine.state['candidates'][decision['anchor_method_id']]
        if engine.backend.full_evaluation_identity(parent) != decision['anchor_evaluation_identity']:
            raise ValueError('Parent full evaluation changed after round selection')
        frozen_bank = decision['frozen_config']['methods']
        by_id = {candidate['id']: candidate for candidate in candidates}
        if any(method['id'] not in by_id for method in frozen_bank):
            raise ValueError('Round feedback omitted a frozen method')
        if _method_bank([by_id[method['id']] for method in frozen_bank]) != frozen_bank:
            raise ValueError('Method implementation changed after regional selection')
        binding = _validate_binding(read_json(folder/'binding.json'))
        with MemoryStore(self.memory_path) as store:
            committed = store.get_committed_feedback(decision['decision_id'], decision=decision)
            feedback_path = folder/'feedback_pending.json'
            feedback = (committed['event']['feedback'] if committed else
                        read_json(feedback_path) if feedback_path.exists() else None)
            if feedback is None:
                anchor = pending.get('region_anchor') or {}
                observed = [by_id[method['id']] for method in frozen_bank
                            if by_id[method['id']].get('full_report')]
                if len(observed) > 2:
                    raise ValueError('Anchored round supports at most two successful full children')
                if not anchor.get('low_report') or not observed:
                    feedback = {'reward': None, 'status': 'reward_uninformative', 'update_eligible': False,
                        'comparison_scope': 'mcts_parent_anchor', 'anchor_method_id': parent['id'],
                        'comparable_pair_count': 0, 'sample_weight': anchored_sample_weight(0, self.cfg),
                        'reason': 'parent_anchor_failed' if not anchor.get('low_report') else 'no_successful_full_children'}
                else:
                    if (anchor.get('anchor_of') != parent['id'] or
                            anchor.get('anchor_full_identity') != decision['anchor_evaluation_identity']):
                        raise ValueError('Low anchor does not identify the frozen full parent')
                    for candidate in [anchor, *observed]:
                        if _validate_binding(candidate['coreset_binding']) != binding:
                            raise ValueError('Reward low predictions came from a different coreset')
                        report = candidate['low_report']
                        if report.get('coreset_version') != binding['version']:
                            raise ValueError('Low report version differs from its frozen coreset')
                        if report.get('coreset_binding_sha256') != stable_hash(binding):
                            raise ValueError('Low report binding hash differs from its frozen coreset')
                    for candidate in observed:
                        child_identity = engine.backend.full_evaluation_identity(candidate)
                        for key in ('train_sha256', 'validation_sha256', 'seed', 'metric', 'metric_params'):
                            if child_identity.get(key) != decision['anchor_evaluation_identity'].get(key):
                                raise ValueError('Child and parent full results use different evaluation protocols')
                    core = {parent['id']: anchor['low_report']['score'], **{
                        candidate['id']: candidate['low_report']['score'] for candidate in observed}}
                    full = {parent['id']: parent['full_report']['score'], **{
                        candidate['id']: candidate['full_report']['score'] for candidate in observed}}
                    feedback = compute_action_reward(core, full, direction=metric_direction(engine.backend.task),
                        config=self.cfg, comparison_scope='mcts_parent_anchor', anchor_method_id=parent['id'])
                feedback['action_id'] = decision['chosen_action_id']
                write_json(feedback_path, feedback)
            elif not feedback_path.exists():
                write_json(feedback_path, feedback)
            _freeze_artifacts(folder, 'feedback', [feedback_path])
            snapshot = committed['snapshot'] if committed else store.commit_feedback(
                decision, feedback, [], schema=self.schema, config=self.cfg)
            self._export(snapshot)
            result = {'status': 'completed', 'decision_id': decision['decision_id'],
                'reward': feedback['reward'], 'reward_status': feedback['status'],
                'observed_method_count': feedback.get('observed_method_count', 0),
                'sample_weight': feedback.get('sample_weight', 0.),
                'update_eligible': bool(feedback.get('update_eligible')),
                'policy_updated': bool(feedback.get('update_eligible')),
                'snapshot_id': snapshot['snapshot_id']}
            write_json(folder/'feedback_commit.json', result)
            return result
