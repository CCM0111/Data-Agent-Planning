"""Real, budgeted MCTS fits on a frozen source run; never reads test data.

The adapter deliberately imports neither historical validation scores nor fitted
source estimators. Source parameter-feedback *unfitted* artifacts are reused only
after their hashes have been checked. Parent calibration anchors may reuse a
verified low evaluation from this search when the complete fit identity matches.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from .coreset import file_hash, write_json
from .method_repair import preflight_methods, predictor_constructors
from .multi_proxy import method_contract, evaluation_metrics, _llm_json_once, load_fixed, run_fit
from .multi_proxy_io import apply_schema, read_raw, read_typed
from .multi_proxy_runtime import (validate_code, task_metric_feedback, prepare_pipeline, pipeline_spec,
                                  ModelOutsideSearch, check_search_model)
from .multi_proxy_trace import identity, normalize_plan, validate_idea, validate_stage_parameters
from .stage1_metrics import metric_direction
from .mcts_changes import audit_changes, estimator_parameter_changes, repair_audit_feedback
from .data_profile import (feature_structure_profile, feature_distribution_difference,
                           feature_metadata_for_task)
from .parameter_feedback import complete_stage_parameters
from .prepared_protocol import validate_prepared_partition
from .mcts_memory import repair_only_context
from .code_repair_contract import audit_error_only_repair
from .execution_budget import infrastructure_failure
from .region_bandit.schemas import stable_hash as binding_identity


def _read(path):
    return json.loads(Path(path).read_text())


async def _callback(function, *args):
    if function is not None:
        result = function(*args)
        if inspect.isawaitable(result):
            await result


def _positions(path, total):
    values = _read(path)['train_positions']
    if (not isinstance(values, list) or not values
            or any(type(i) is not int or not 0 <= i < total for i in values)
            or len(set(values)) != len(values)):
        raise ValueError('Invalid or duplicated frozen row positions: ' + str(path))
    return values


def _same_rows(actual, expected, description):
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    except AssertionError as error:
        raise ValueError(description + ' is not aligned with frozen training positions') from error


class MCTSBackend:
    """Source/data adapter used by the search driver.

    ``request_guard()`` runs before every logical API request. ``before_fit`` is
    called with ``(stage, candidate)`` before each actual fit, including low-stage
    repair retries. A full-stage failure never changes an already scored method.
    """

    def __init__(self, source_run, coreset_index, output, *, model=None, pricing=None,
                 seed=None, max_repairs=2, request_guard=None, before_fit=None,
                 request_fn=None, subgroup_rules=None, subgroup_min_rows=5):
        self.source_run = Path(source_run).resolve()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        if type(coreset_index) is not int or coreset_index < 0:
            raise ValueError('coreset_index must be a nonnegative integer')
        if type(max_repairs) is not int or not 0 <= max_repairs <= 2:
            raise ValueError('max_repairs must be between zero and two')
        self.coreset_index, self.model = coreset_index, model
        self.pricing, self.max_repairs = pricing or {}, max_repairs
        self.request_guard, self.before_fit = request_guard, before_fit
        self.repair_memory = None
        self.search_context_provider = None
        self.request_fn = request_fn
        self.calls = []
        self.manifest = _read(self.source_run/'manifest.json')
        if self.manifest.get('status') != 'success':
            raise ValueError('MCTS requires a completed source (deferred full training is allowed)')
        self.source = Path(self.manifest.get('source') or self.source_run).resolve()
        self.source_manifest = (_read(self.source/'manifest.json')
                                if self.source != self.source_run else self.manifest)
        self.configuration = deepcopy(self.manifest['configuration'])
        self.cfg = SimpleNamespace(**self.configuration)
        self.seed = self.configuration['model_seed'] if seed is None else seed
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('Invalid model seed')
        self.task_path = Path(self.manifest['task_path']).resolve()
        task, split, paths, hashes, train = load_fixed(self.task_path)
        self.split_id = split['split_id']
        from .stage1_selection import _dependency_evidence
        self.dependency_evidence = _dependency_evidence(task, {'split': split,
                                                               'metadata': self.manifest.get('split_metadata', {})})
        self.supports_iid_mcs = not self.dependency_evidence
        self.task_context = _read(self._artifact('task_context.json'))
        # Older source contexts omitted metric options (e.g. F1 pos_label).
        # The frozen task definition is authoritative for every diagnostic.
        if 'metric_params' in task:
            self.task_context['metric_params'] = deepcopy(task['metric_params'])
        self.task = deepcopy(self.task_context)
        self.metric = self.task['metric']
        self.maximize = metric_direction(self.task) == 'maximize'
        self.classes = self.task['classes']
        self.schema = self.task_context['csv_schema']
        train = apply_schema(train, self.schema)
        self.train_path = paths['train']
        self.coreset_path = self.source_run/f'coreset_{coreset_index}.csv'
        indices_path = self.source_run/f'coreset_{coreset_index}_indices.json'
        self.core_indices = _positions(indices_path, len(train))
        validation_indices_path = self._artifact('V_core_indices.json')
        validation_path = self._artifact('core_validation.csv')
        self.v_core_indices_sha256 = file_hash(validation_indices_path)
        self.v_core_sha256 = file_hash(validation_path)
        for source_manifest in (self.manifest, self.source_manifest):
            saved = source_manifest.get('evaluation_set', {})
            for key, actual in (('labels_sha256', self.v_core_sha256),
                                ('indices_sha256', self.v_core_indices_sha256)):
                if saved.get(key) is not None and saved[key] != actual:
                    raise ValueError('Source V_core fingerprint mismatch: ' + key)
        self.validation_indices = _positions(validation_indices_path, len(train))
        if set(self.core_indices) & set(self.validation_indices):
            raise ValueError('Fixed coreset overlaps V_core')
        q_path = self._artifact('Q_indices.json', required=False)
        if q_path is not None and not set(self.core_indices) <= set(_positions(q_path, len(train))):
            raise ValueError('Fixed coreset is not contained in Q')
        p_path = self._artifact('P_indices.json', required=False)
        if p_path is not None and set(self.core_indices) & set(_positions(p_path, len(train))):
            raise ValueError('Fixed coreset overlaps P')
        self._binding_partitions = {name: (path, file_hash(path), set(_positions(path, len(train))))
                                    for name, path in (('P', p_path), ('Q', q_path)) if path is not None}
        self._validated_bindings = set()
        coreset = read_typed(self.coreset_path, self.schema, low_memory=False)
        core = read_typed(validation_path, self.schema, low_memory=False)
        _same_rows(coreset, train.iloc[self.core_indices], 'Coreset CSV')
        _same_rows(core, train.iloc[self.validation_indices], 'V_core CSV')
        validate_prepared_partition(task, self.task_path, split,
            _positions(p_path, len(train)) if p_path is not None else [], self.validation_indices,
            _positions(q_path, len(train)) if q_path is not None else [])
        # Resolve only train and validation. In particular no test path is opened,
        # hashed, copied, or read, including when a test file is missing.
        external_path = (self.task_path.parent/task['files']['validation']).resolve()
        if external_path == self.train_path:
            raise ValueError('External validation cannot equal training data')
        if file_hash(external_path) != split['sha256'].get(external_path.name):
            raise ValueError('External validation fingerprint mismatch')
        original_indices = split.get('indices', {})
        if set(original_indices.get('train', [])) & set(original_indices.get('validation', [])):
            raise ValueError('External validation source rows overlap training')
        external = read_typed(external_path, self.schema, low_memory=False)
        if len(external) != split['rows']['validation'] or not len(external):
            raise ValueError('External validation row count mismatch')
        if set(external.columns) != set(train.columns):
            raise ValueError('External validation schema mismatch')
        self.y_core = core[self.task['target']].to_numpy()
        self.y_external = external[self.task['target']].to_numpy()
        for labels in (self.y_core, self.y_external):
            if pd.isna(labels).any():
                raise ValueError('Evaluation target contains missing values')
            if self.task['problem_type'] == 'regression' and not np.isfinite(labels.astype(float)).all():
                raise ValueError('Evaluation target contains nonfinite values')
            if self.classes and not set(labels) <= set(self.classes):
                raise ValueError('Evaluation target contains unknown classes')
        self._prepare_subgroups(core, external, subgroup_rules or [], subgroup_min_rows)
        self.eda = _read(self._artifact('eda.json'))
        # These cached planner facts use features from P/Q only. V_core and
        # external validation remain evaluation data, never EDA observations.
        if p_path is not None and q_path is not None:
            p_positions = set(_positions(p_path, len(train)))
            q_positions = set(_positions(q_path, len(train)))
            if p_positions & q_positions or (p_positions | q_positions) & set(self.validation_indices):
                raise ValueError('Planning P/Q positions overlap each other or V_core')
            reference_positions = sorted(p_positions | q_positions)
            reference_scope = 'P + Q (frozen positions; excludes V_core)'
        else:
            reference_positions = sorted(set(range(len(train))) - set(self.validation_indices))
            reference_scope = 'training rows excluding frozen V_core; legacy P/Q indices unavailable'
        feature_columns = self.task_context['feature_columns']
        metadata_task = deepcopy(task)
        if metadata_task.get('preparation_audit'):
            metadata_task['preparation_audit'] = str(
                (self.task_path.parent / metadata_task['preparation_audit']).resolve())
        feature_metadata = feature_metadata_for_task(metadata_task, feature_columns)
        if feature_metadata:
            self.task_context['feature_metadata'] = feature_metadata
        reference_features = train.iloc[reference_positions][feature_columns]
        core_features = coreset[feature_columns]
        self.planning_data_profile = {
            'schema_version': 1,
            'feature_structure': feature_structure_profile(
                reference_features, categorical_columns=self.task_context.get('categorical_columns', []),
                scope=reference_scope, metadata=feature_metadata, max_details=8, seed=self.seed),
            'coreset_distribution': feature_distribution_difference(
                reference_features, core_features,
                categorical_columns=self.task_context.get('categorical_columns', []),
                reference_scope=reference_scope, max_details=6, seed=self.seed)}
        self.planning_data_profile['fingerprint'] = identity(self.planning_data_profile)
        profile_path = self.output/'planning_data_profile.json'
        if profile_path.exists() and _read(profile_path) != self.planning_data_profile:
            raise ValueError('Planning data profile changed; use a new search output directory')
        if not profile_path.exists():
            write_json(profile_path, self.planning_data_profile)
        self.planning_eda = {**self.eda, **self.planning_data_profile}
        self.evaluation_budget = deepcopy(self.configuration['evaluation_budget'])
        self.full_train_budget = deepcopy(self.configuration['full_train_budget'])
        self.batch_size = self.configuration.get('batch_size', 2048)
        data = self.output/'data'
        data.mkdir(exist_ok=True)
        self.core_features = data/'core_validation_features.csv'
        self.external_features = data/'external_validation_features.csv'
        core.drop(columns=[self.task['target']]).to_csv(self.core_features, index=False)
        external.drop(columns=[self.task['target']]).to_csv(self.external_features, index=False)
        self.task_context['training_context'].update(coreset_rows=len(coreset), fixed_train_rows=len(train),
                                                     core_validation_rows=len(core))
        self.task = deepcopy(self.task_context)
        self.task_context_path = self.output/'task_context.json'
        write_json(self.task_context_path, self.task_context)
        plan_path = self._artifact('parameter_feedback/frozen_plan.json', required=False)
        self.feedback_plan = _read(plan_path) if plan_path else None
        if self.configuration.get('parameter_feedback') and self.feedback_plan is None:
            raise ValueError('Parameter-feedback source lacks its frozen plan')
        if self.feedback_plan and self.seed != self.configuration['model_seed']:
            raise ValueError('Frozen feedback requires the source model seed; use a separate search RNG seed')
        self.metadata = {'source_run': str(self.source_run), 'source': str(self.source),
                         'task_path': str(self.task_path), 'task_id': self.task.get('name'),
                         'task_sha256': file_hash(self.task_path),
                         'task_context_path': str(self.task_context_path),
                         'task_context_sha256': file_hash(self.task_context_path),
                         'split_id': self.split_id, 'internal_split_id': self.manifest.get('internal_split_id'),
                         'coreset_index': coreset_index, 'coreset_indices_sha256': file_hash(indices_path),
                         'coreset_sha256': file_hash(self.coreset_path), 'train_sha256': hashes['train'],
                         'v_core_sha256': self.v_core_sha256,
                         'v_core_indices_sha256': self.v_core_indices_sha256,
                         'external_validation_sha256': file_hash(external_path), 'model_seed': self.seed,
                         'train_rows': len(train), 'coreset_rows': len(coreset), 'v_core_rows': len(core),
                         'external_validation_rows': len(external), 'metric': self.metric,
                         'maximize': self.maximize, 'test_accessed': False,
                         'historical_scores_reused': False,
                         'subgroup_rules': self.subgroup_rules, 'subgroup_min_rows': self.subgroup_min_rows}
        self.identity = {**self.metadata, 'seed': self.seed,
                         'generation_protocol': 'incremental_evidence_v5',
                         'planning_data_profile_sha256': identity(self.planning_data_profile),
                         'method_contract_sha256': identity(method_contract('planned', formal_only=True)),
                         'arm': 'fixed_coreset_' + self.metadata['coreset_indices_sha256'],
                         'supports_iid_mcs': self.supports_iid_mcs,
                         'execution_configuration': deepcopy(self.configuration),
                         'execution_configuration_sha256': identity(self.configuration),
                         'eda_sha256': file_hash(self._artifact('eda.json')),
                         'feedback_plan_sha256': file_hash(plan_path) if plan_path else None,
                         'method_definitions': identity([{key: m[key] for key in ('name', 'family', 'idea', 'code_sha256')}
                                                         for m in self.manifest['methods']])}
        write_json(self.output/'backend_source.json', self.metadata)

    def _prepare_subgroups(self, core, external, rules, minimum):
        """Fix feature-only masks before any candidate is evaluated."""
        if type(minimum) is not int or minimum < 1:
            raise ValueError('subgroup_min_rows must be a positive integer')
        if not isinstance(rules, list) or len(rules) > 6:
            raise ValueError('At most six configured subgroups plus two missingness groups')
        self.subgroup_min_rows = minimum
        self.subgroup_rules = [{'name': 'missing_any', 'op': 'missing_any'},
                               {'name': 'no_missing', 'op': 'no_missing'}]
        names = {'missing_any', 'no_missing'}
        for rule in rules:
            if not isinstance(rule, dict) or set(rule) != {'name', 'column', 'op', 'value'}:
                raise ValueError('Subgroup rules require exactly name, column, op, value')
            if not isinstance(rule['name'], str) or not rule['name'] or rule['name'] in names:
                raise ValueError('Subgroup names must be nonempty and unique')
            if rule['column'] not in self.task['feature_columns'] or rule['op'] not in {'eq', 'le', 'ge'}:
                raise ValueError('Subgroups must use declared feature columns and eq/le/ge')
            if not isinstance(rule['value'], (str, int, float, bool)) or isinstance(rule['value'], float) and not math.isfinite(rule['value']):
                raise ValueError('Subgroup thresholds must be finite JSON scalars')
            names.add(rule['name'])
            self.subgroup_rules.append(deepcopy(rule))
        self._subgroup_masks = {}
        for stage, frame in [('low', core), ('full', external)]:
            missing = frame[self.task['feature_columns']].isna().any(axis=1).to_numpy()
            masks = {'missing_any': missing, 'no_missing': ~missing}
            for rule in rules:
                column, value = frame[rule['column']], rule['value']
                if rule['op'] == 'eq':
                    mask = column == value
                else:
                    numeric = pd.to_numeric(column, errors='raise')
                    threshold = float(value)
                    mask = numeric <= threshold if rule['op'] == 'le' else numeric >= threshold
                masks[rule['name']] = mask.fillna(False).to_numpy(dtype=bool)
            self._subgroup_masks[stage] = masks

    def _subgroup_feedback(self, stage, labels, values):
        summaries = []
        for rule in self.subgroup_rules:
            mask = self._subgroup_masks[stage][rule['name']]
            count = int(mask.sum())
            summary = {'name': rule['name'], 'rows': count, 'rule': rule,
                       'small_sample': count < self.subgroup_min_rows, 'metric': self.metric,
                       'score': None, 'metrics': {},
                       'scope': 'Predefined descriptive validation subgroup; not an independent comparison'}
            if count >= self.subgroup_min_rows:
                actual, predicted = labels[mask], values['predictions'][mask]
                probabilities = values['probabilities']
                metrics, reason = evaluation_metrics(self.task, actual, predicted,
                                                     probabilities[mask] if probabilities is not None else None,
                                                     self.classes)
                summary.update(score=metrics.get(self.metric), metrics=metrics, metric_reason=reason)
                if self.classes:
                    summary['error_rate'] = float(np.mean(actual != predicted))
                else:
                    error = actual.astype(float)-predicted.astype(float)
                    summary.update(rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))))
            summaries.append(summary)
        return summaries

    def _artifact(self, relative, required=True):
        for directory in (self.source_run, self.source):
            path = directory/relative
            if path.exists():
                return path
        if required:
            raise FileNotFoundError('Source artifact missing: ' + relative)
        return None

    def initial_candidates(self):
        """Copy five initial definitions, with no old scores or learned models."""
        methods = self.manifest['methods']
        if len(methods) != 5:
            raise ValueError('Initialization requires exactly five source methods')
        result = []
        for i, method in enumerate(methods):
            folder = self.output/'candidates'/f'initial_{i}'
            folder.mkdir(parents=True, exist_ok=True)
            code = Path(method['code_path'])
            if not code.exists():
                code = self._artifact(f'methods/method_{i}/method.py')
            if file_hash(code) != method['code_sha256']:
                raise ValueError('Source method code fingerprint mismatch')
            path = folder/'method.py'
            path.write_bytes(code.read_bytes())
            candidate = {key: deepcopy(method[key]) for key in ('name', 'family', 'idea')}
            candidate.update(id=f'initial_{i}', parent_id='root', branch_id=f'initial_{i}',
                             code_path=str(path), code_sha256=file_hash(path), low_score=None,
                             full_score=None, status='generated', repairs_used=0,
                             experience_use=[],
                             source_method_id=i, origin='source_initial', parameter_budget_mode='exact')
            if self.feedback_plan:
                item = next(x for x in self.feedback_plan['methods'] if x['method_id'] == i)
                if item['source_code_sha256'] != candidate['code_sha256']:
                    raise ValueError('Frozen parameter plan refers to another implementation')
                candidate['idea']['stage_parameters'] = deepcopy(item['stage_parameters'])
                candidate['idea']['scaling_rules'] = deepcopy(item['rules'])
                candidate['idea']['adaptation_reason'] = item['reason']
                candidate['parameter_budget_mode'] = 'planned'
                candidate['frozen_stages'] = {}
                for phase, artifact in item['pipeline_artifacts'].items():
                    source_path = Path(artifact['path'])
                    if file_hash(source_path) != artifact['sha256']:
                        raise ValueError('Frozen initial pipeline fingerprint mismatch')
                    destination = folder/f'{phase}_unfitted.joblib'
                    destination.write_bytes(source_path.read_bytes())
                    candidate['frozen_stages'][phase] = {
                        'initial_pipeline_path': str(destination),
                        'initial_pipeline_sha256': artifact['sha256'],
                        'expected_pipeline_spec': deepcopy(artifact['spec'])}
            candidate['contract_sha256'] = self._contract(candidate)
            self._save(candidate)
            result.append(candidate)
        return result

    @staticmethod
    def _contract(candidate):
        fields = ('idea', 'family', 'parameter_budget_mode', 'parent_baseline')
        if 'allowed_model_classes' in candidate:
            fields += ('allowed_model_classes',)
        return identity({key: candidate.get(key) for key in fields})

    def _save(self, candidate):
        write_json(Path(candidate['code_path']).parent/'candidate.json', candidate)

    def _validate_coreset_binding(self, binding):
        """Verify a fixed-size sample from Q without using validation outcomes."""
        required = {'version', 'csv_path', 'indices_path', 'csv_sha256', 'indices_sha256'}
        if not isinstance(binding, dict) or not required <= set(binding) or set(binding) - required - {'schema_version'}:
            raise ValueError('Invalid candidate coreset binding fields')
        if binding.get('schema_version', 'candidate-coreset-v1') != 'candidate-coreset-v1':
            raise ValueError('Unknown candidate coreset binding schema')
        version = binding_identity({key: binding[key] for key in ('csv_sha256', 'indices_sha256')})
        if binding['version'] != version:
            raise ValueError('Coreset version must identify its CSV and indices hashes')
        canonical = {**binding, 'schema_version': 'candidate-coreset-v1',
                     'csv_path': str(Path(binding['csv_path']).resolve()),
                     'indices_path': str(Path(binding['indices_path']).resolve())}
        for key in ('csv', 'indices'):
            if file_hash(canonical[key+'_path']) != canonical[key+'_sha256']:
                raise ValueError('Bound coreset artifact changed: ' + key)
        if set(self._binding_partitions) != {'P', 'Q'}:
            raise ValueError('Dynamic coreset binding requires frozen P and Q indices')
        for name, (path, checksum, _) in self._binding_partitions.items():
            if file_hash(path) != checksum:
                raise ValueError('Frozen sampling partition changed: ' + name)
        if file_hash(self.train_path) != self.metadata['train_sha256']:
            raise ValueError('Frozen training data changed before binding')
        signature = binding_identity(canonical)
        if signature not in self._validated_bindings:
            positions = _positions(canonical['indices_path'], self.metadata['train_rows'])
            selected = set(positions)
            if len(positions) != len(self.core_indices):
                raise ValueError('Bound coreset must preserve the original coreset row count')
            if (not selected <= self._binding_partitions['Q'][2] or
                    selected & (self._binding_partitions['P'][2] | set(self.validation_indices))):
                raise ValueError('Bound coreset must be a subset of Q, disjoint from P and V_core')
            actual = read_typed(canonical['csv_path'], self.schema, low_memory=False)
            # Stream the full CSV; hold only the selected rows, in their exact
            # declared order. Wide/full training frames need not stay resident.
            parts, offset = [], 0
            for chunk in read_typed(self.train_path, self.schema, low_memory=False, chunksize=8192):
                chunk.index = np.arange(offset, offset + len(chunk))
                keep = [position for position in chunk.index if position in selected]
                if keep:
                    parts.append(chunk.loc[keep])
                offset += len(chunk)
            expected = pd.concat(parts).loc[positions]
            _same_rows(actual, expected, 'Bound coreset CSV')
            self._validated_bindings.add(signature)
        return canonical

    def bind_coreset(self, candidate, binding):
        """Freeze a candidate's sampling contract before its first fitted result."""
        canonical = self._validate_coreset_binding(binding)
        current = candidate.get('coreset_binding')
        if current == canonical:
            self._candidate_coreset(candidate)
            self._save(candidate)
            return deepcopy(canonical)
        if (current or candidate.get('sampling_contract_frozen') or candidate.get('fit_attempts') or
                candidate.get('low_report') or candidate.get('full_report')):
            raise ValueError('Cannot change an already frozen/evaluated candidate coreset')
        if self._contract(candidate) != candidate['contract_sha256'] or file_hash(candidate['code_path']) != candidate['code_sha256']:
            raise ValueError('Candidate changed before sampling contract freeze')
        candidate.update(coreset_binding=canonical, sampling_contract_frozen=True,
            sampling_contract_sha256=identity({'code_sha256': candidate['code_sha256'],
                'contract_sha256': candidate['contract_sha256'], 'coreset_binding': canonical}))
        self._save(candidate)
        return deepcopy(canonical)

    def _candidate_coreset(self, candidate):
        binding = candidate.get('coreset_binding')
        if binding is None:
            return self.coreset_path, None
        canonical = self._validate_coreset_binding(binding)
        checksum = identity({'code_sha256': candidate['code_sha256'],
            'contract_sha256': candidate['contract_sha256'], 'coreset_binding': canonical})
        if (binding != canonical or not candidate.get('sampling_contract_frozen') or
                candidate.get('sampling_contract_sha256') != checksum):
            raise ValueError('Candidate sampling contract changed after decision')
        return Path(canonical['csv_path']), binding_identity(canonical)

    async def prepare_sampling_candidate(self, candidate):
        """Finish constructor checks/allowed repairs before selecting a sample."""
        if candidate.get('low_report') or candidate.get('full_report'):
            raise ValueError('Sampling preparation requires an unscored candidate')
        stamp = {'code_sha256': candidate.get('code_sha256'), 'contract_sha256': self._contract(candidate)}
        if candidate.get('sampling_preflight') == stamp:
            if file_hash(candidate['code_path']) != stamp['code_sha256']:
                raise ValueError('Prepared candidate implementation changed')
            return candidate
        if candidate.get('sampling_contract_frozen'):
            raise ValueError('Sampling preparation cannot change a frozen decision')
        while True:
            numbers = [int(p.name.rsplit('_', 1)[1]) for p in Path(candidate['code_path']).parent.glob('preflight_attempt_*')
                       if p.name.rsplit('_', 1)[1].isdigit()]
            try:
                await asyncio.to_thread(self._preflight, candidate, max(numbers, default=0) + 1)
                candidate['sampling_preflight'] = {'code_sha256': file_hash(candidate['code_path']),
                                                   'contract_sha256': self._contract(candidate)}
                if candidate['sampling_preflight'] != {'code_sha256': candidate['code_sha256'],
                                                       'contract_sha256': candidate['contract_sha256']}:
                    raise ValueError('Sampling preflight changed an unfrozen implementation identity')
                self._save(candidate)
                return candidate
            except (ValueError, TypeError, RuntimeError) as error:
                if (isinstance(error, ModelOutsideSearch) or getattr(error, 'budget_exhausted', False)
                        or infrastructure_failure(error) or
                        candidate.get('frozen_stages') or self.model is None or
                        candidate.get('repairs_used', 0) >= self.max_repairs):
                    raise
                await self._repair(candidate, error, 'sampling_preflight')

    def full_evaluation_identity(self, candidate):
        """Verified reusable parent full evidence, independent of its low sample."""
        report = candidate.get('full_report')
        if not report or candidate.get('anchor_of'):
            raise ValueError('An evaluated full parent is required for an anchor')
        if (file_hash(candidate['code_path']) != candidate['code_sha256'] or
                self._contract(candidate) != candidate['contract_sha256'] or
                report.get('code_sha256') != candidate['code_sha256'] or
                report.get('contract_sha256') != candidate['contract_sha256']):
            raise ValueError('Parent full evaluation implementation/contract mismatch')
        context_path = Path(report['prediction_path']).parent/'context.json'
        context = _read(context_path)
        if (Path(context['train']).resolve() != self.train_path.resolve() or context['seed'] != self.seed or
                context['task'].get('fit_stage') != 'full_train' or
                file_hash(self.train_path) != self.metadata['train_sha256'] or
                report.get('fit_rows') != self.metadata['train_rows'] or
                report.get('stage') != 'full' or report.get('metric') != self.metric or
                report.get('direction') != ('maximize' if self.maximize else 'minimize')):
            raise ValueError('Parent full training identity mismatch')
        if (report.get('fit_train_sha256', self.metadata['train_sha256']) != self.metadata['train_sha256'] or
                report.get('model_seed', self.seed) != self.seed):
            raise ValueError('Parent full training hash/seed mismatch')
        self.load_prediction(report)
        for name in ('unfitted_pipeline', 'fitted_pipeline'):
            if file_hash(report[name+'_path']) != report[name+'_sha256']:
                raise ValueError('Parent full pipeline artifact changed')
        if not isinstance(report.get('score'), (int, float)) or isinstance(report['score'], bool) or not math.isfinite(report['score']):
            raise ValueError('Parent full score must be finite')
        return {'candidate_id': candidate['id'], 'code_sha256': candidate['code_sha256'],
                'contract_sha256': candidate['contract_sha256'], 'train_sha256': self.metadata['train_sha256'],
                'validation_sha256': self.metadata['external_validation_sha256'], 'seed': self.seed,
                'metric': self.metric, 'metric_params': deepcopy(self.task.get('metric_params', {})),
                'score': report['score'], 'report_sha256': identity(report),
                'predictions_sha256': file_hash(report['prediction_path']),
                'fitted_pipeline_sha256': report['fitted_pipeline_sha256']}

    def _direct_low_constructor(self, candidate):
        """Return verified constructor evidence only; no low score or fitted model exists."""
        certificate = candidate.get('direct_full_certificate')
        if (candidate.get('implementation_mode') != 'inherited_parameters'
                or not certificate or set(certificate) != {'coreset', 'full_train'}
                or candidate.get('sampling_preflight') != {'code_sha256': candidate['code_sha256'],
                                                          'contract_sha256': self._contract(candidate)}
                or any(not digest or self.parameter_stage_key(candidate, phase) != digest
                       for phase, digest in certificate.items())):
            raise ValueError('Direct full evaluation requires the frozen parameter comparison')
        stage = candidate['frozen_stages']['coreset']
        return {'unfitted_pipeline_path': stage['initial_pipeline_path'],
                'unfitted_pipeline_sha256': stage['initial_pipeline_sha256'],
                'pipeline_spec': stage['expected_pipeline_spec']}

    def make_coreset_anchor(self, parent, binding, anchor_id):
        """Freeze a low-only calibration anchor, reusing an exact parent fit."""
        if not isinstance(anchor_id, str) or not anchor_id or Path(anchor_id).name != anchor_id or anchor_id in {'.', '..'}:
            raise ValueError('Anchor ID must be a safe nonempty directory name')
        full_identity = self.full_evaluation_identity(parent)
        canonical = self._validate_coreset_binding(binding)
        folder = self.output/'anchors'/anchor_id
        if (folder/'candidate.json').exists():
            anchor = _read(folder/'candidate.json')
            if (anchor.get('anchor_of') != parent['id'] or anchor.get('anchor_full_identity') != full_identity or
                    anchor.get('coreset_binding') != canonical):
                raise ValueError('Saved anchor differs from its frozen parent/sample')
            self._candidate_coreset(anchor)
            if anchor.get('parent_low_reuse') or anchor.get('equivalent_low_reuse'):
                expected = self._parent_low_reuse_report(anchor)
                if anchor.get('low_report') is not None:
                    if anchor['low_report'] != expected:
                        raise ValueError('Saved anchor reused report changed')
                    self._commit_parent_low_reuse(anchor)
            return anchor
        folder.mkdir(parents=True, exist_ok=True)
        anchor = {key: deepcopy(parent[key]) for key in
            ('name', 'family', 'idea', 'parameter_budget_mode', 'parent_baseline', 'modification_direction',
             'allowed_model_classes') if key in parent}
        path = folder/'method.py'
        path.write_bytes(Path(parent['code_path']).read_bytes())
        anchor.update(id=anchor_id, parent_id=parent['id'], branch_id=parent.get('branch_id', parent['id']),
                      anchor_of=parent['id'], anchor_full_identity=full_identity, code_path=str(path),
                      code_sha256=file_hash(path), low_score=None, full_score=None, status='generated',
                      repairs_used=0, origin='coreset_anchor', frozen_stages={})
        for phase, stage in (('coreset', 'low'), ('full_train', 'full')):
            report = parent.get(stage+'_report') or {}
            if stage == 'low' and not report and parent.get('direct_full_certificate'):
                report = self._direct_low_constructor(parent)
            if not report.get('unfitted_pipeline_path') or not report.get('pipeline_spec'):
                raise ValueError('Anchor requires parent actual unfitted stage pipelines')
            source = Path(report['unfitted_pipeline_path'])
            if file_hash(source) != report['unfitted_pipeline_sha256']:
                raise ValueError('Parent unfitted pipeline changed before anchor creation')
            destination = folder/(phase+'_unfitted.joblib')
            destination.write_bytes(source.read_bytes())
            anchor['frozen_stages'][phase] = {'initial_pipeline_path': str(destination),
                'initial_pipeline_sha256': file_hash(destination), 'expected_pipeline_spec': deepcopy(report['pipeline_spec'])}
        anchor['contract_sha256'] = self._contract(anchor)
        if anchor['contract_sha256'] != parent['contract_sha256']:
            raise ValueError('Anchor must preserve the exact parent method contract')
        self.bind_coreset(anchor, canonical)
        report = parent.get('low_report') or {}
        # Old reports without a complete cache identity remain valid evidence,
        # but are not eligible for this optimization. A different sample/budget
        # also falls through to the ordinary calibration fit.
        if report.get('evaluation_reuse', {}).get('protocol') == 'inherited-effective-low-v1':
            self._bind_measured_low(anchor, report)
        elif (report.get('low_cache_identity') and
                report['low_cache_identity'] == self._low_cache_identity(anchor, report)):
            source_path = Path(report['prediction_path']).parent/'evaluation.json'
            if _read(source_path) != report:
                raise ValueError('Parent low report changed before anchor reuse')
            anchor['parent_low_reuse'] = {
                'protocol': 'exact-parent-low-v1', 'source_candidate_id': parent['id'],
                'source_report_path': str(source_path), 'source_report_sha256': file_hash(source_path)}
            self._parent_low_reuse_report(anchor)
            self._save(anchor)
        return anchor

    def parameter_stage_key(self, candidate, phase):
        """Identify a measured or programmatically inherited effective constructor."""
        report = candidate.get({'coreset': 'low_report', 'full_train': 'full_report'}[phase])
        if report:
            spec = report.get('pipeline_spec')
            path = report.get('unfitted_pipeline_path')
            digest = report.get('unfitted_pipeline_sha256')
        elif candidate.get('implementation_mode') == 'inherited_parameters':
            frozen = candidate.get('frozen_stages', {}).get(phase, {})
            spec = frozen.get('expected_pipeline_spec')
            path = frozen.get('initial_pipeline_path')
            digest = frozen.get('initial_pipeline_sha256')
        else:
            return None
        if not spec or not path or not digest:
            return None
        if (file_hash(path) != digest or file_hash(candidate['code_path']) != candidate['code_sha256']
                or self._contract(candidate) != candidate['contract_sha256']):
            raise ValueError('Stage constructor changed before equivalence check')
        return identity({'code': candidate['code_sha256'], 'specification': spec, 'model_seed': self.seed})

    def prepare_equivalent_low(self, target, source, *, method, reference):
        """Reuse a measured same-sample constructor for an inherited parameter edit."""
        if (target.get('low_report') or target.get('parent_low_reuse') or target.get('low_error')
                or any(a['stage'] == 'low' for a in target.get('fit_attempts', []))):
            return False
        if target.get('equivalent_low_reuse'):
            return True
        if method.get('implementation_mode') != 'inherited_parameters' or not method.get('parent_id'):
            return False
        same_parent = (reference['id'] == method['parent_id'] or
                       (reference.get('implementation_mode') == 'inherited_parameters' and
                        reference.get('parent_id') == method['parent_id']))
        if not same_parent or not source.get('low_report'):
            return False
        if (target is not method and target.get('anchor_of') != method['id'] or
                source is not reference and source.get('anchor_of') != reference['id']):
            return False
        key = self.parameter_stage_key(method, 'coreset')
        if not key or key != self.parameter_stage_key(reference, 'coreset'):
            return False
        return self._bind_measured_low(target, source['low_report'])

    def _bind_measured_low(self, target, report):
        frozen = target.get('frozen_stages', {}).get('coreset', {})
        if (not target.get('coreset_binding') or not report.get('low_cache_identity')
                or frozen.get('expected_pipeline_spec') != report.get('pipeline_spec')
                or target['code_sha256'] != report.get('code_sha256')):
            return False
        path = Path(report['prediction_path']).parent / 'evaluation.json'
        measured = _read(path)
        digest = file_hash(path)
        if report.get('fit_calls') == 1:
            if measured != report:
                raise ValueError('Source low report changed before equivalence check')
        elif report.get('evaluation_reuse', {}).get('source_report_sha256') != digest:
            raise ValueError('Original fitted report changed before equivalence check')
        identity_fields = self._low_cache_identity(target, measured)
        cache = measured.get('low_cache_identity') or {}
        # Full contracts and low-stage declarations can differ while the frozen
        # effective low pipeline and every fit/evaluation input coincide.
        if any(cache.get(k) != v for k, v in identity_fields.items()
               if k not in {'contract_sha256', 'stage_parameters'}):
            return False
        target['equivalent_low_reuse'] = dict(protocol='inherited-effective-low-v1',
            source_candidate_id=measured['candidate_id'], source_report_path=str(path),
            source_report_sha256=digest)
        self._save(target)
        return True

    def _low_cache_identity(self, candidate, report):
        """Content identities, including row order, rather than artifact paths."""
        sample, _ = self._candidate_coreset(candidate)
        binding = candidate.get('coreset_binding')
        indices = (Path(binding['indices_path']) if binding else
                   self.source_run/f'coreset_{self.coreset_index}_indices.json')
        if not binding and (file_hash(sample) != self.metadata['coreset_sha256'] or
                            file_hash(indices) != self.metadata['coreset_indices_sha256']):
            raise ValueError('Original coreset changed before cache identity check')
        return {
            'protocol': 'exact-parent-low-v1', 'code_sha256': candidate['code_sha256'],
            'contract_sha256': candidate['contract_sha256'],
            'unfitted_pipeline_sha256': report['unfitted_pipeline_sha256'],
            'pipeline_spec_sha256': identity(report['pipeline_spec']),
            'train_sha256': file_hash(sample),
            'ordered_positions_sha256': identity(_positions(indices, self.metadata['train_rows'])),
            'fit_rows': len(self.core_indices), 'model_seed': self.seed,
            'budget': deepcopy(self.evaluation_budget), 'batch_size': self.batch_size,
            'stage_parameters': deepcopy(candidate['idea']['stage_parameters']['coreset']),
            'task_context_sha256': identity(self.task_context),
            'validation_id': self.v_core_indices_sha256, 'evaluation_labels_sha256': self.v_core_sha256,
            'validation_features_sha256': file_hash(self.core_features),
            'metric': self.metric, 'metric_params': deepcopy(self.task.get('metric_params', {})),
            'direction': 'maximize' if self.maximize else 'minimize',
            'subgroup_rules': deepcopy(self.subgroup_rules), 'subgroup_min_rows': self.subgroup_min_rows}

    def _parent_low_reuse_report(self, anchor):
        """Recheck immutable provenance before admitting a zero-fit anchor."""
        equivalent = bool(anchor.get('equivalent_low_reuse'))
        reuse = anchor['equivalent_low_reuse'] if equivalent else anchor['parent_low_reuse']
        if (reuse.get('protocol') != ('inherited-effective-low-v1' if equivalent else 'exact-parent-low-v1') or
                (not equivalent and (reuse.get('source_candidate_id') != anchor.get('anchor_of')
                                     or not anchor.get('anchor_full_identity'))) or
                self._contract(anchor) != anchor['contract_sha256'] or
                file_hash(anchor['code_path']) != anchor['code_sha256']):
            raise ValueError('Invalid parent low reuse provenance')
        source = Path(reuse['source_report_path'])
        if file_hash(source) != reuse['source_report_sha256']:
            raise ValueError('Parent low source report changed after cache freeze')
        report = _read(source)
        cache = report.get('low_cache_identity')
        expected = self._low_cache_identity(anchor, report)
        if equivalent and cache:
            expected['contract_sha256'] = report['contract_sha256']
            expected['stage_parameters'] = report['stage_parameters']
        if (report.get('candidate_id') != reuse['source_candidate_id'] or report.get('stage') != 'low' or
                report.get('evaluation_set') != 'V_core' or report.get('fit_calls') != 1 or
                report.get('code_sha256') != anchor['code_sha256'] or
                (not equivalent and report.get('contract_sha256') != anchor['contract_sha256']) or
                cache != expected):
            raise ValueError('Parent low cache identity mismatch')
        for key in ('fit_rows', 'model_seed', 'metric', 'direction', 'validation_id',
                    'evaluation_labels_sha256', 'stage_parameters'):
            if report.get(key) != cache[key]:
                raise ValueError('Parent low report differs from cache identity: ' + key)
        if (report.get('requested_budget') != cache['budget'] or
                report.get('fit_train_sha256') != cache['train_sha256']):
            raise ValueError('Parent low report training budget/sample mismatch')
        frozen = anchor['frozen_stages']['coreset']
        if (file_hash(frozen['initial_pipeline_path']) != frozen['initial_pipeline_sha256'] or
                (not equivalent and frozen['initial_pipeline_sha256'] != report['unfitted_pipeline_sha256']) or
                frozen['expected_pipeline_spec'] != report['pipeline_spec']):
            raise ValueError('Anchor effective pipeline changed before cache reuse')
        for name in ('unfitted_pipeline', 'fitted_pipeline'):
            if file_hash(report[name+'_path']) != report[name+'_sha256']:
                raise ValueError('Parent low pipeline changed before cache reuse')
        context_path = Path(report['prediction_path']).parent/'context.json'
        if (file_hash(report['prediction_path']) != report.get('prediction_sha256') or
                file_hash(context_path) != report.get('context_sha256')):
            raise ValueError('Parent low prediction/context artifact changed before cache reuse')
        context = _read(context_path)
        if (context.get('budget') != cache['budget'] or context.get('seed') != cache['model_seed'] or
                context.get('batch_size') != cache['batch_size'] or
                context['task'].get('fit_stage') != 'evaluation' or
                context['task'].get('stage_parameters') != cache['stage_parameters'] or
                file_hash(context['train']) != cache['train_sha256'] or
                file_hash(context['predict']) != cache['validation_features_sha256']):
            raise ValueError('Parent low execution context mismatch')
        values = self.load_prediction(report)
        metrics, _ = evaluation_metrics(self.task, self.y_core, values['predictions'],
                                        values['probabilities'], self.classes)
        if (not isinstance(report.get('score'), (int, float)) or isinstance(report['score'], bool) or
                not math.isfinite(report['score']) or metrics[self.metric] != report['score']):
            raise ValueError('Parent low cached score does not match its predictions')
        result = deepcopy(report)
        result.update(candidate_id=anchor['id'], fit_calls=0,
                      coreset_binding_sha256=binding_identity(anchor['coreset_binding']),
                      coreset_version=anchor['coreset_binding']['version'])
        if equivalent:
            result.update(contract_sha256=anchor['contract_sha256'],
                          stage_parameters=deepcopy(anchor['idea']['stage_parameters']['coreset']))
            result['low_cache_identity'] = self._low_cache_identity(anchor, result)
        costs = {key: result[key] for key in ('fit_seconds', 'prediction_seconds',
            'prediction_wall_seconds', 'worker_wall_seconds', 'process_wall_seconds') if key in result}
        result.update({key: 0.0 for key in costs})
        result['evaluation_reuse'] = {**deepcopy(reuse), 'fit_calls_saved': 1,
                                     'source_execution_cost': costs}
        return result

    def _commit_parent_low_reuse(self, candidate):
        report = self._parent_low_reuse_report(candidate)
        destination = Path(candidate['code_path']).parent/'low_reuse'/'evaluation.json'
        if destination.exists() and _read(destination) != report:
            raise ValueError('Committed parent low reuse report changed')
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, report)
        candidate.update(low_report=report, low_score=report['score'], status='low_evaluated')
        self._save(candidate)
        return report

    async def _request_once(self, prompt, path):
        """One charged provider attempt, with no transport or JSON repair call.

        Regional advice has its own one-attempt policy. It must not inherit the
        method generator's bounded recovery loop. Local JSON parsing still uses
        the ordinary adapter, whose SDK transport retries are explicitly zero.
        """
        if self.model is None:
            raise RuntimeError('A real model adapter is required for generation or repair')
        try:
            await _callback(self.request_guard)
            request = self.request_fn if self.request_fn is not None else _llm_json_once
            return await request(self.model, prompt, path, self.calls, self.pricing)
        finally:
            write_json(self.output/'backend_llm_calls.json', self.calls)

    async def _request(self, prompt, path):
        if self.model is None:
            raise RuntimeError('A real model adapter is required for generation or repair')
        try:
            if self.request_fn is not None:
                await _callback(self.request_guard)
                return await self.request_fn(self.model, prompt, path, self.calls, self.pricing)
            from .generation_recovery import recover_json
            async def guarded_call(*args):
                await _callback(self.request_guard)
                return await _llm_json_once(*args)
            return await recover_json(guarded_call, self.model, prompt, path, self.calls, self.pricing)
        finally:
            write_json(self.output/'backend_llm_calls.json', self.calls)

    def _validate_strategy(self, strategy, allowed_evidence_ids=()):
        for key in ('name', 'family', 'idea', 'modification_direction', 'experience_use'):
            if key not in strategy:
                raise ValueError('Generated strategy missing ' + key)
        if not isinstance(strategy['name'], str) or not strategy['name'].strip():
            raise ValueError('Generated strategy requires a nonempty name')
        if strategy['family'] not in {'linear', 'bagging', 'boosting', 'neural', 'neighbors', 'naive_bayes', 'tree'}:
            raise ValueError('Unsupported model family')
        if strategy['modification_direction'] not in {'features', 'model', 'parameters'}:
            raise ValueError('Choose one main modification direction')
        brief = strategy['idea'].get('plan_format') == 'brief_v1'
        if brief and (not isinstance(strategy['idea'].get('change_reason'), str) or not strategy['idea']['change_reason'].strip()):
            raise ValueError('Expanded method requires one short change_reason')
        usage = strategy['experience_use']
        if not isinstance(usage, list):
            raise ValueError('experience_use must be an array, empty when no experience applies')
        allowed = set(allowed_evidence_ids)
        seen = set()
        for item in usage:
            if not isinstance(item, dict) or (set(item) != {'evidence_id', 'adoption_reason', 'applicability_difference'} and not (brief and set(item) == {'evidence_id'})):
                raise ValueError('Use an object containing evidence_id' if brief else
                                 'Experience use requires evidence_id, adoption_reason, applicability_difference')
            identifier = item['evidence_id']
            if not isinstance(identifier, str) or identifier not in allowed:
                raise ValueError('Experience use references an evidence ID not retrieved for this expansion')
            if identifier in seen:
                raise ValueError('An evidence ID may appear only once in experience_use')
            seen.add(identifier)
            for key in set(item) - {'evidence_id'}:
                if not isinstance(item[key], str) or not item[key].strip():
                    raise ValueError('Experience use requires nonempty ' + key)
        validate_idea(strategy['idea'])
        # Existing feedback supports explicitly planned tree/iteration budgets.
        # Permit trees here too, but never random seeds, parallelism, or estimators.
        idea = deepcopy(strategy['idea'])
        stages = idea.get('stage_parameters')
        if not isinstance(stages, dict) or any(not isinstance(p, dict) for p in stages.values()):
            raise ValueError('Each stage requires a parameter object')
        for phase, parameters in stages.items():
            trees = parameters.pop('model__n_estimators', None)
            if trees is not None:
                cap = (self.evaluation_budget if phase == 'coreset' else self.full_train_budget)['max_trees']
                if type(trees) is not int or not 1 <= trees <= cap:
                    raise ValueError('Planned tree count exceeds the stage budget')
        validate_stage_parameters(idea)
        json.dumps(strategy, allow_nan=False)

    def _normalize_child_strategy(self, proposal, parent, evidence_ids, names):
        plan, _ = normalize_plan({'strategies': [proposal]}, budgets={'coreset': self.evaluation_budget, 'full_train': self.full_train_budget})
        strategy = plan['strategies'][0]
        self._resolve_parent_inheritance(strategy, parent)
        self._validate_strategy(strategy, evidence_ids)
        if strategy['name'] in names:
            raise ValueError('Expansion candidate names must be distinct')
        if strategy['modification_direction'] == 'parameters':
            strategy['idea']['stage_parameters'] = {phase: {
                **deepcopy(parent['idea']['stage_parameters'][phase]),
                **strategy['idea']['stage_parameters'][phase]}
                for phase in ('coreset', 'full_train')}
            self._validate_strategy(strategy, evidence_ids)
        return strategy

    async def _prepare_child_strategy(self, proposal, candidate, parent, evidence_ids, names):
        """Correct only this invalid proposal, sharing its existing repair allowance."""
        repair = None
        while True:
            try:
                strategy = self._normalize_child_strategy(proposal, parent, evidence_ids, names)
                if repair is not None:
                    repair['status'] = 'success'
                return strategy
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                if repair is not None:
                    repair.update(status='failed', repair_error=f'{type(error).__name__}: {error}')
                if self.model is None or candidate['repairs_used'] >= self.max_repairs:
                    raise
                candidate['repairs_used'] += 1
                repair = {'attempt': candidate['repairs_used'], 'stage': 'planning',
                          'error': f'{type(error).__name__}: {error}', 'status': 'started'}
                candidate.setdefault('repairs', []).append(repair)
                self._save(candidate)
                prompt = '''Correct only the schema/contract error in this one child strategy.
Preserve its valid experiment intent and settings; do not redesign it or change sibling candidates.
Return {"strategy": <the complete corrected strategy object>}, with name, family, idea,
modification_direction and experience_use. family is linear|bagging|boosting|neural|neighbors|naive_bayes|tree;
modification_direction is features|model|parameters. Names must avoid occupied_names.
For experience_use, return [] or a list of objects with evidence_id from allowed_evidence_ids,
only. Use idea.summary and one short idea.change_reason for the compact plan.
Both stage parameter dictionaries are required.
The parent idea is supplied only to resolve inheritance. No new scores or test data are supplied.
''' + method_contract('planned', output_format=False, formal_only=True) + '\n' + json.dumps({
                    'strategy': proposal, 'error': repair['error'], 'parent_idea': parent['idea'],
                    'allowed_evidence_ids': list(evidence_ids), 'occupied_names': sorted(names),
                    'budgets': {'coreset': self.evaluation_budget, 'full_train': self.full_train_budget}},
                    ensure_ascii=False)
                try:
                    response = await self._request(prompt, Path(candidate['code_path']).parent /
                                                   f'plan_repair_{candidate["repairs_used"]}.json')
                    corrected = response.get('strategy') if isinstance(response, dict) else None
                    if (isinstance(proposal, dict) and isinstance(corrected, dict)
                            and str(error).startswith(('Experience use ', 'experience_use ',
                                                      'An evidence ID may appear only once',
                                                      'Generated strategy missing experience_use'))):
                        # A citation repair cannot rewrite inherited code settings or the proposed trial.
                        proposal = {**proposal, 'experience_use': corrected.get('experience_use')}
                    else:
                        proposal = corrected
                except Exception as failure:
                    repair.update(status='failed', repair_error=f'{type(failure).__name__}: {failure}')
                    raise
                finally:
                    self._save(candidate)

    @staticmethod
    def _resolve_parent_inheritance(strategy, parent):
        """Explicit markers avoid asking the planner to reproduce omitted prose."""
        markers = {'model': '__inherit_parent__',
                   'feature_engineering': ['__inherit_parent__'],
                   'fixed_parameters': {'__inherit_parent__': True}}
        idea = strategy.get('idea', {})
        for key, marker in markers.items():
            if idea.get(key) == marker:
                if strategy.get('modification_direction') != 'parameters':
                    raise ValueError('Parent inheritance markers are only valid for parameter strategies')
                idea[key] = deepcopy(parent['idea'][key])

    def _inherit_parameter_implementation(self, candidate, parent):
        """Reuse actual unfitted stage pipelines when every change is declarative.

        A direction label alone is insufficient: the model, FE and fixed plan
        must be unchanged, both parent artifacts must be verified, and every
        override must be a valid parameter of those same concrete pipelines.
        """
        if (candidate['modification_direction'] != 'parameters'
                or candidate['family'] != parent['family']
                or any(candidate['idea'].get(key) != parent['idea'].get(key)
                       for key in ('model', 'feature_engineering', 'fixed_parameters'))):
            return False
        sources = {}
        for phase, stage in [('coreset', 'low'), ('full_train', 'full')]:
            report = parent.get(stage + '_report') or {}
            frozen = parent.get('frozen_stages', {}).get(phase, {})
            path = report.get('unfitted_pipeline_path') or frozen.get('initial_pipeline_path')
            checksum = report.get('unfitted_pipeline_sha256') or frozen.get('initial_pipeline_sha256')
            if not path or not checksum:
                return False
            if file_hash(path) != checksum:
                raise ValueError('Parent unfitted stage artifact changed after evaluation')
            sources[phase] = path
        from .multi_proxy_feedback import load_method_module
        from .method_preflight import validate_estimators
        path = Path(candidate['code_path'])
        if file_hash(parent['code_path']) != parent['code_sha256']:
            raise ValueError('Parent implementation changed after evaluation')
        path.write_bytes(Path(parent['code_path']).read_bytes())
        load_method_module(path)
        baselines = {phase: clone(joblib.load(source)) for phase, source in sources.items()}
        original_parameters = deepcopy(candidate['idea']['stage_parameters'])
        candidate['idea']['stage_parameters'] = complete_stage_parameters(original_parameters, baselines)
        candidate['stage_parameter_completion'] = {
            phase: {key: value for key, value in values.items() if key not in original_parameters[phase]}
            for phase, values in candidate['idea']['stage_parameters'].items()}
        frozen_stages = {}
        for phase, stage, budget in [('coreset', 'evaluation', self.evaluation_budget),
                                      ('full_train', 'full_train', self.full_train_budget)]:
            baseline = baselines[phase]
            candidate['parent_baseline']['stages'][phase] = pipeline_spec(baseline)
            pipeline = clone(baseline)
            task = {**self.task_context, 'fit_stage': stage,
                    'stage_parameters': candidate['idea']['stage_parameters'][phase],
                    'parameter_budget_mode': 'planned'}
            if 'allowed_model_classes' in candidate:
                task['allowed_model_classes'] = candidate['allowed_model_classes']
            if phase == 'full_train':
                task['reference_pipeline_path'] = frozen_stages['coreset']['initial_pipeline_path']
            _, specification = prepare_pipeline(pipeline, task, budget, self.seed)
            validate_estimators(pipeline)
            artifact = path.parent / f'{phase}_unfitted.joblib'
            joblib.dump(pipeline, artifact)
            frozen_stages[phase] = {'initial_pipeline_path': str(artifact),
                'initial_pipeline_sha256': file_hash(artifact), 'expected_pipeline_spec': specification}
        candidate.update(frozen_stages=frozen_stages, implementation_mode='inherited_parameters',
                         code_sha256=file_hash(path), status='generated')
        candidate['contract_sha256'] = self._contract(candidate)
        write_json(path.parent/'implementation.json', {
            'source': 'programmatic_parent_inheritance', 'api_called': False,
            'parent_id': parent['id'], 'parent_code_sha256': parent['code_sha256'],
            'stage_parameters': candidate['idea']['stage_parameters'],
            'stage_parameter_completion': candidate['stage_parameter_completion']})
        return True

    async def generate(self, parent, evidence, iteration, *, resume_generation=False):
        """One batch planning call, then implementations; no extra experience summary."""
        if type(iteration) is not int or iteration < 0:
            raise ValueError('Iteration must be nonnegative')
        directory = self.output/'generation'/f'expansion_{iteration:04d}'
        directory.mkdir(parents=True, exist_ok=resume_generation)
        parent_payload = {key: deepcopy(parent.get(key)) for key in
                          ('id', 'name', 'family', 'idea', 'low_score', 'full_score')}
        parent_payload['code'] = Path(parent['code_path']).read_text()
        parent_payload['stage_feedback'] = {
            stage: {key: deepcopy(parent[stage+'_report'].get(key)) for key in
                    ('score', 'metric', 'direction', 'evaluation_set', 'fit_rows', 'fit_seconds',
                     'training_feedback', 'warnings', 'subgroup_errors', 'stage_parameters', 'actual_budget',
                     'coreset_version')}
            for stage in ('low', 'full') if parent.get(stage+'_report')}
        # The driver supplies bounded evidence cards; raw historical manifests
        # and historical source scores are never included here.
        search_context = deepcopy(self.search_context_provider() if self.search_context_provider else {})
        legacy_best = search_context.pop('global_best', None)
        if isinstance(legacy_best, dict):
            search_context.setdefault('global_best_id', legacy_best.get('id'))
        from .region_bandit.skill_prompt import method_brief
        parent_method = method_brief(parent)
        parent_method.setdefault('fe', parent['idea'].get('feature_engineering', []))
        for key in ('fixed_parameters', 'scaling_rules'):
            if parent['idea'].get(key):
                parent_method[key] = parent['idea'][key]
        parent_method['change_reason'] = parent['idea'].get('change_reason') or parent['idea'].get('rationale')
        parent_method['scores'] = {k: parent[k] for k in ('low_score', 'full_score') if parent.get(k) is not None}
        payload = {'parent_id': parent['id'], 'parent_method': parent_method,
                   'evidence': evidence, 'task': self.task_context,
                   'eda': self.planning_eda, 'budgets': {'coreset': self.evaluation_budget,
                                              'full_train': self.full_train_budget},
                   'search_context': search_context}
        if isinstance(evidence, dict) and evidence.get('disabled'):
            # Turning off historical retrieval must not hide the current parent.
            payload['parent_feedback'] = parent_payload['stage_feedback']
        allowed_evidence_ids = evidence.get('evidence_ids', []) if isinstance(evidence, dict) else []
        if not isinstance(allowed_evidence_ids, list) or any(not isinstance(i, str) for i in allowed_evidence_ids):
            raise ValueError('Retrieved evidence_ids must be an array of strings')
        stage = evidence.get('search_stage', {}) if isinstance(evidence, dict) else {}
        count = search_context.get('expansion_candidates', 3)
        allowed = None
        if stage.get('phase') == 'parameter_focus':
            allowed = stage.get('allowed_model_classes')
            if not isinstance(allowed, list) or not allowed or not all(isinstance(c, str) and c for c in allowed):
                raise ValueError('Parameter-focused expansion requires measured estimator classes')
            strategy_guidance = '''Use only the exact estimator classes in evidence.search_stage.allowed_model_classes.
For each requested candidate, change one effective estimator parameter from this
parent at full scale, keep its estimator, preprocessing and other parameters fixed, and use the inheritance
markers below. Give concrete parent and child values and check whether the coreset budget can distinguish them.
Prefer changes whose effects can be screened on the coreset when equally promising. If the coreset
settings coincide, describe that limitation. Avoid repeating previously tested settings.'''
        elif stage.get('phase') == 'single_parameter_check':
            strategy_guidance = ('Include one feasible pure parameter child: inherit the parent model, features and '
                'fixed settings using the exact inheritance markers below. Change exactly one effective full-training '
                'estimator parameter to an untested value within budget. This comparison uses the primary rollout. '
                'Choose remaining proposals freely.')
        else:
            strategy_guidance = '''Unlike the five diverse initialization strategies, the children have NO direction or model-family quota.
All may modify parameters, all may modify features, or any mixture may be used.
Do NOT produce one features / one model / one parameters candidate merely to cover categories.'''
        if count == 2 and stage.get('phase') not in ('direct_parameter_tuning', 'parameter_focus'):
            strategy_guidance += """
Use the two slots to test distinct hypotheses about this parent. Build one child around the
best-supported improvement in the measured paths. For the other, identify what that first
child leaves unresolved. For example, keep useful features and try a compatible estimator,
or keep the estimator and test a different feature relationship. Preserve the parent's other
settings where feasible so a result will be interpretable. These examples imply no family quota.
Two nearby parameter values are useful when comparable full results suggest a promising interval
and the values test distinguishable predictions. Otherwise spend the second slot on a different
plausible mechanism. In change_reason briefly name the evidence and hypothesis;
if evidence is sparse, ground it in the parent's diagnostics or the training EDA."""
        instruction = f'''The search has already selected parent_id. Propose exactly {count} feasible children of this parent.
Full rollouts are scarce. Your objective is to produce a child that improves the task's primary metric
over this parent after full training on the fixed external validation protocol. Use the remaining rollout
budget and rollout schedule in search_context.
Each child makes one main change to features, model or parameters; include any necessary compatible preprocessing.
''' + strategy_guidance + '''
Use evidence.full_method_paths, when present, to check what has already been tried across this task.
Follow parent_id to distinguish successive changes from sibling alternatives. Use a relevant full result
to choose, adapt or avoid a change for this parent. When several things changed together, the individual
effect is uncertain. Avoid repeating an evaluated configuration. If history offers no close comparison,
base the proposal on the parent's diagnostics and training EDA.
Use evidence.cards for details: roles identifies the parent and current best; effective_parameters.shared
and by_stage describe their settings. Check parameter sources and evaluation scopes. Missing values are
unknown. Interpret sampled EDA only within its stated scope; respect confirmed units and preprocessing.
Use actual remaining resources; primary and audit reservations are not freely available. Historical ranking
conflicts describe only full-evaluated methods. Omitted settings inherit the parent.
For a parameter-only strategy, explicitly inherit unchanged plan fields using these exact existing-field values:
idea.model="__inherit_parent__", idea.feature_engineering=["__inherit_parent__"],
idea.fixed_parameters={"__inherit_parent__":true}. The runner resolves them to the exact parent fields before validation.
Express all parameter changes in stage_parameters. Do not reconstruct unseen parent wording or invent fixed defaults.
If the model or feature mechanism itself must change, declare that direction and its concrete plan instead.
Omit stock caveats, motivational prose, and repeated protocol descriptions. Give one short change_reason linking the concrete change to measured evidence (candidate ID where applicable).
Consider disagreement between coreset and full outcomes; unmeasured benefits are hypotheses. Never access test data.
Freeze both coreset and full_train parameter dictionaries and their scale rules now.
Keep the same FE mechanism and estimator class between these stages. Full training uses the entire original
training set and scores external validation; coreset training scores fixed V_core. Stage scores are not interchangeable.
Return {"strategies":[{"name":"...","family":"linear|bagging|boosting|neural|neighbors|naive_bayes|tree",
"modification_direction":"features|model|parameters",
"experience_use":[{"evidence_id":"retrieved ID actually used"}],
"idea":{"summary":"one sentence","change_reason":"one short sentence","feature_engineering":["concrete step"],
"model":"estimator class","fixed_parameters":{},"scaling_rules":{},"stage_parameters":{"coreset":{},"full_train":{}}}}]}.
Pipeline parameter paths are required. Explicit planned model__n_estimators and model__max_iter are allowed within
stage resource caps. Preserve parent choices not related to the main change, unless feasibility requires a documented change.
No random_state, n_jobs, warm_start, early_stopping, estimator or pipeline replacement in stage_parameters.
For each strategy, experience_use records only experience actually adopted; it may be empty even when cards exist.
Reference only evidence_ids explicitly supplied in this retrieval. If no evidence_ids are supplied, return [].
Never invent historical evidence or treat an unmeasured full result as a measured improvement. Be concise about
the adopted parameter or mechanism and its applicability; do not copy whole experience cards into your answer.
'''
        if search_context.get('regional_sampling'):
            instruction += '''\nThis search can change the shared coreset between expansions. Compare low scores only when
their coreset_version matches, or use the explicitly supplied parent anchor evaluated on that same sample.
Do not interpret a difference between historical low scores from different samples as a method improvement.
Full scores remain comparable on the fixed full training/external validation protocol. The static EDA
coreset_distribution describes the initialization sample, not the newly selected sample for this expansion.\n'''
        tuning = stage.get('phase') == 'direct_parameter_tuning'
        expected = 1 if tuning else count
        if tuning:
            instruction = '''Return exactly ONE strategy in {"strategies":[...]} for direct full-training
parameter tuning of this selected parent. Use the measured full_method_paths and metric direction to
choose an untried value for one effective full_train estimator parameter within budget.
Keep family, code, preprocessing and estimator unchanged. Set modification_direction="parameters".
Include name, family, experience_use ([] or objects containing only evidence_id), and idea with summary,
change_reason (one sentence grounded in the previous result), feature_engineering=["__inherit_parent__"],
model="__inherit_parent__", fixed_parameters={"__inherit_parent__":true}, scaling_rules and stage_parameters.
Return only changed parameter overrides. The runner inherits all omitted parent settings. There is no coreset evaluation,
candidate comparison or sampling decision. In change_reason cite one relevant full-validation observation. Omit stock caveats.'''
        response = await self._request(instruction + '\nThe following implementation contract constrains the plan; '
            'return only the strategies JSON above.\n' + method_contract('planned', output_format=False, formal_only=True) + '\n'
            + json.dumps(payload, ensure_ascii=False, separators=(',', ':')), directory/'plan.json')
        if (not isinstance(response, dict) or not isinstance(response.get('strategies'), list)
                or len(response['strategies']) != expected):
            raise ValueError(f'Expansion requires {expected} strategies')
        plan = {'strategies': deepcopy(response['strategies']), 'planning_errors': {}}
        names = set()
        candidates = []
        for i, proposal in enumerate(response['strategies']):
            identifier = f'expansion_{iteration:04d}_{i}'
            folder = self.output/'candidates'/identifier
            folder.mkdir(parents=True, exist_ok=resume_generation)
            saved = folder/'candidate.json'
            if resume_generation and saved.exists():
                previous = _read(saved)
                if (previous.get('id') != identifier or previous.get('parent_id') != parent['id']
                        or previous.get('branch_id') != parent['branch_id']):
                    raise ValueError('Saved generation belongs to a different candidate')
                if previous.get('status') == 'generated':
                    if (previous['code_path'] != str(folder/'method.py') or
                            file_hash(previous['code_path']) != previous.get('code_sha256') or
                            self._contract(previous) != previous.get('contract_sha256')):
                        raise ValueError('Generated candidate changed before admission')
                    candidates.append(previous)
                    names.add(previous['name'])
                    plan['strategies'][i] = {k: deepcopy(previous[k]) for k in
                        ('name','family','idea','modification_direction','experience_use')}
                    continue
                if previous.get('status') == 'failed' or previous.get('repairs_used'):
                    raise ValueError(f'Cannot replay failed or interrupted candidate repair: {identifier}')
            candidate = {'name': identifier, 'family': 'unknown', 'idea': {},
                         'modification_direction': 'unknown', 'experience_use': [],
                         'id': identifier, 'parent_id': parent['id'],
                         'branch_id': parent['branch_id'], 'code_path': str(folder/'method.py'),
                         'low_score': None, 'full_score': None, 'status': 'generating',
                         'repairs_used': 0, 'origin': 'generated', 'parameter_budget_mode': 'planned'}
            if allowed is not None:
                candidate['allowed_model_classes'] = list(allowed)
            candidates.append(candidate)
            try:
                # In the one-parameter phase the direction is fixed by the runner.
                # Older planners sometimes put it inside idea or omit the duplicate label.
                if tuning and isinstance(proposal, dict) and 'modification_direction' not in proposal:
                    proposal = deepcopy(proposal)
                    idea = proposal.get('idea')
                    if isinstance(idea, dict):
                        proposal['modification_direction'] = idea.pop('modification_direction', 'parameters')
                strategy = await self._prepare_child_strategy(
                    proposal, candidate, parent, allowed_evidence_ids, names)
            except Exception as error:
                candidate.update(status='failed', failure_stage='planning',
                                 error=f'{type(error).__name__}: {error}', raw_strategy=proposal)
                self._save(candidate)
                plan['planning_errors'][str(i)] = candidate['error']
                write_json(directory/'plan.normalized.json', plan)
                if getattr(error, 'budget_exhausted', False):
                    raise
                continue
            names.add(strategy['name'])
            plan['strategies'][i] = strategy
            write_json(directory/'plan.normalized.json', plan)
            candidate.update({key: deepcopy(strategy[key]) for key in
                              ('name', 'family', 'idea', 'modification_direction', 'experience_use')})
            candidate['parent_baseline'] = {
                'code_path': parent['code_path'], 'code_sha256': file_hash(parent['code_path']),
                'idea': deepcopy(parent['idea']),
                'stages': {phase: deepcopy(parent[key+'_report'].get('pipeline_spec'))
                           for phase,key in [('coreset','low'),('full_train','full')]
                           if parent.get(key+'_report')},
                'actual_stage_parameters': {stage:{key:deepcopy(report.get(key))
                    for key in ('stage_parameters','actual_budget','fit_rows')}
                    for stage,report in parent_payload['stage_feedback'].items()}}
            if not parent.get('low_report') and parent.get('depth_direct') and parent.get('preflight_low_spec'):
                candidate['parent_baseline']['stages']['coreset']=deepcopy(parent['preflight_low_spec'])
            if not parent.get('low_report') and parent.get('direct_full_certificate'):
                candidate['parent_baseline']['stages']['coreset'] = self._direct_low_constructor(parent)['pipeline_spec']
            candidate['contract_sha256'] = self._contract(candidate)
            self._save(candidate)
            from .implementation_feedback import component_interfaces
            implementation_prompt = method_contract('planned', formal_only=True) + '''
Implement strategy incrementally from parent_code and its actual stage parameters; return the complete modified code.
Change only what the strategy declares. Preserve unrelated column selection, preprocessing order, missing-value
handling, encoders and model parameters. Do not rewrite unchanged code for refactoring or style.
Check input types, sparse/dense compatibility, prediction interfaces and both stage parameter dictionaries.
Necessary compatibility changes must be explained by the strategy; never silently replace the model or FE mechanism.
For parameter-only changes, express changed values in stage_parameters using exact pipeline paths for both stages.
Keep custom transformation state in serializable pipeline parameters, not mutable module globals initialized only by build_method.
Implement a fresh unfitted factory, no fitting or IO. Return complete code, fe_config and model_config JSON.
''' + json.dumps({'strategy': strategy, 'installed_interfaces': component_interfaces(strategy, parent_payload['code']), 'parent_code': parent_payload['code'],
                  **({'allowed_model_classes': allowed} if allowed is not None else {}),
                  'parent_effective_stages': candidate['parent_baseline']['actual_stage_parameters'],
                  'parent_stage_parameters': parent['idea'].get('stage_parameters',{}),
                  'task': self.task_context, 'eda': self.eda,
                  'evaluation_budget': self.evaluation_budget, 'full_train_budget': self.full_train_budget}, ensure_ascii=False)
            try:
                if self._inherit_parameter_implementation(candidate, parent):
                    self._save(candidate)
                    continue
                if tuning:
                    raise ValueError('Direct parameter tuning requires an inherited parameter-only implementation')
                implementation = await self._request(implementation_prompt, folder/'implementation.json')
                # Keep invalid source text so bounded repair has the actual defect.
                from .constructor_compatibility import normalize_constructor_code
                implementation['code'], compatibility = normalize_constructor_code(implementation.get('code', ''))
                if compatibility:
                    candidate['compatibility_changes'] = compatibility
                code = implementation.get('code', '')
                Path(candidate['code_path']).write_text(code)
                try:
                    validate_code(code)
                except (ValueError, SyntaxError) as error:
                    await self._repair(candidate, error, 'generation')
                candidate.update(code_sha256=file_hash(candidate['code_path']), status='generated')
            except Exception as error:
                candidate.update(status='failed', failure_stage='generation', error=f'{type(error).__name__}: {error}')
                if getattr(error, 'budget_exhausted', False):
                    self._save(candidate)
                    raise
            self._save(candidate)
        return candidates

    async def _repair(self, candidate, error, stage):
        if (candidate.get('low_report') or candidate.get('full_report') or candidate.get('frozen_stages') or
                candidate.get('sampling_contract_frozen') or candidate.get('anchor_of')):
            raise RuntimeError('An evaluated or source-frozen method cannot be repaired in place') from error
        if self.model is None or candidate.get('repairs_used', 0) >= self.max_repairs:
            raise RuntimeError('Method repair allowance exhausted') from error
        candidate['repairs_used'] = candidate.get('repairs_used', 0) + 1
        repair = {'attempt': candidate['repairs_used'], 'stage': stage,
                  'error': f'{type(error).__name__}: {error}', 'status': 'started'}
        candidate.setdefault('repairs', []).append(repair)
        self._save(candidate)
        path = Path(candidate['code_path'])
        old = path.read_text() if path.exists() else ''
        repair_experience = repair_only_context(
            self.repair_memory(repair['error']) if self.repair_memory else [])
        incremental_feedback = repair_audit_feedback(
            _read(candidate['incremental_audit']) if candidate.get('incremental_audit') else None, old)
        from .implementation_feedback import repair_diagnostic, component_interfaces
        prior_repairs = [{k: r[k] for k in ('attempt', 'error', 'repair_error', 'contract_differences') if k in r}
                         for r in candidate.get('repairs', [])[:-1]]
        prompt = '''Repair only the concrete implementation error. Return {"code":"complete Python source"}.
Preserve the exact strategy, estimator classes, feature engineering intent, fixed parameters and both stage dictionaries.
Do not tune for performance; no evaluation scores are supplied. Do not change budgets or add methods.
When incremental_audit_feedback is current, fix its structured differences: expected includes the declared stage
overrides, not just parent defaults. Code locations are supplied only for unique literal matches; null means
unknown. Other differences may be omitted; rerun the normal audit after repair. Do not alter the frozen plan.
''' + method_contract(candidate['parameter_budget_mode'], output_format=False, formal_only=True) + '\n' + json.dumps({'strategy': {key: candidate[key] for key in ('name', 'family', 'idea')},
                                                'task': self.task_context, 'error': repair['error'],
                                                'original_code': old,
                                                'repair_diagnostic': repair_diagnostic(old, repair['error']),
                                                'installed_interfaces': component_interfaces(candidate['idea'], old),
                                                'previous_repairs': prior_repairs,
                                                'parent_baseline': candidate.get('parent_baseline'),
                                                'parent_code': (Path(candidate['parent_baseline']['code_path']).read_text()
                                                                if candidate.get('parent_baseline') else None),
                                                'incremental_audit_feedback': incremental_feedback,
                                                'related_repair_experience': repair_experience}, ensure_ascii=False)
        try:
            response = await self._request(prompt, path.parent/f'repair_{candidate["repairs_used"]}.json')
            code = response['code']
            validate_code(code)
            try:
                old_predictors = predictor_constructors(old)
            except SyntaxError:
                old_predictors = None
            if old_predictors and predictor_constructors(code) != old_predictors:
                raise ValueError('Error-only repair cannot change estimator classes')
            audit = audit_error_only_repair(old, code, repair['error'], idea=candidate['idea'],
                        allowed_restorations=incremental_feedback.get('differences', []))
            audit_path = path.parent/f'repair_{candidate["repairs_used"]}_contract.json'
            write_json(audit_path, audit)
            repair['contract_audit'] = str(audit_path)
            repair['contract_differences'] = audit.get('parameter_changes', [])
            if audit['violations']:
                raise ValueError('Error-only repair violated fixed configuration: ' + '; '.join(audit['violations']))
            path.write_text(code)
            candidate['code_sha256'] = file_hash(path)
            repair['status'] = 'success'
        except Exception as failure:
            repair.update(status='failed', repair_error=f'{type(failure).__name__}: {failure}')
            raise
        finally:
            self._save(candidate)

    def _preflight(self, candidate, attempt):
        if candidate.get('frozen_stages') and candidate.get('implementation_mode') != 'inherited_parameters':
            # Runtime verifies the frozen pipeline spec/hash, then clones it.
            return
        task = {**self.task_context, 'parameter_budget_mode': candidate['parameter_budget_mode'],
                'complete_stage_parameters': candidate.get('origin') == 'generated' and
                    not candidate.get('sampling_contract_frozen')}
        cfg = SimpleNamespace(scoring_budget=self.configuration['scoring_budget'],
                              evaluation_budget=self.evaluation_budget, full_train_budget=self.full_train_budget,
                              full_train_enabled=True, model_seed=self.seed)
        directory = Path(candidate['code_path']).parent/f'preflight_attempt_{attempt}'
        if candidate.get('implementation_mode') == 'inherited_parameters':
            directory.mkdir(parents=True, exist_ok=True)
            stages = {}
            for phase, frozen in candidate['frozen_stages'].items():
                if file_hash(frozen['initial_pipeline_path']) != frozen['initial_pipeline_sha256']:
                    raise ValueError('Inherited stage pipeline changed after planning')
                stages[phase] = frozen['expected_pipeline_spec']
        else:
            failures = preflight_methods([candidate], task, cfg, directory,
                                         stages=('evaluation', 'full_train'))
            if failures:
                raise ValueError(json.dumps(failures, ensure_ascii=False))
            preflight=_read(directory/'preflight'/'method_0'/'report.json')
            if 'allowed_model_classes' in candidate:
                for result in preflight['stages']:
                    check_search_model(result['model_class'], candidate['allowed_model_classes'])
            if preflight.get('stage_parameters') != candidate['idea']['stage_parameters'] and preflight.get('stage_parameters'):
                if candidate.get('sampling_contract_frozen'):
                    raise ValueError('Stage parameters cannot change after a sampling decision')
                # Resolve omitted counterparts before the first measured fit.
                # No scored node is ever changed by this constructor-only step.
                old = candidate['idea']['stage_parameters']
                candidate['idea']['stage_parameters'] = preflight['stage_parameters']
                candidate['stage_parameter_completion'] = {
                    phase: {key: value for key, value in params.items() if key not in old[phase]}
                    for phase, params in preflight['stage_parameters'].items()}
                candidate['contract_sha256'] = self._contract(candidate)
                self._save(candidate)
            stages={('coreset' if r['stage']=='evaluation' else 'full_train'):r['pipeline_spec']
                    for r in preflight['stages']}
        if candidate.get('depth_direct'):
            # Runtime imports the same source as generated_method, not preflight_generated_method.
            def runtime_names(v):
                if isinstance(v,list):return [runtime_names(i) for i in v]
                if isinstance(v,dict):return {k:(s.replace('preflight_generated_method.','generated_method.',1)
                    if k in ('class','callable') and isinstance(s,str) and s.startswith('preflight_generated_method.')
                    else runtime_names(s)) for k,s in v.items()}
                return v
            candidate['preflight_low_spec']=runtime_names(stages['coreset'])
            candidate['preflight_full_spec']=runtime_names(stages['full_train'])
        baseline=candidate.get('parent_baseline')
        if baseline:
            if file_hash(baseline['code_path']) != baseline['code_sha256']:
                raise ValueError('Parent implementation changed after candidate planning')
            audit=audit_changes(baseline['stages'],stages,candidate['idea']['stage_parameters'],
                                candidate['modification_direction'],Path(baseline['code_path']).read_text(),
                                Path(candidate['code_path']).read_text(),baseline['idea'],candidate['idea'])
            write_json(directory/'incremental_audit.json',audit)
            candidate['incremental_audit']=str(directory/'incremental_audit.json')
            candidate['configuration_fingerprint']=audit['configuration_fingerprint']
            self._save(candidate)
            if audit['violations']:
                raise ValueError('Incremental implementation violation: '+'; '.join(audit['violations']))
            if candidate['modification_direction'] == 'parameters':
                candidate['preflight_full_parameter_changes'] = estimator_parameter_changes(
                    baseline['stages'].get('full_train'), stages['full_train'])
            prefix=candidate['id'].rsplit('_',1)[0]
            for path in Path(candidate['code_path']).parent.parent.glob(prefix+'_*/candidate.json'):
                sibling=_read(path)
                if (sibling['id']!=candidate['id'] and sibling.get('low_report') and
                        sibling.get('configuration_fingerprint')==candidate['configuration_fingerprint']):
                    raise ValueError('Effective stages duplicate already evaluated sibling '+sibling['id']+
                                     '; implement the declared substantive difference')

    def load_prediction(self, report):
        if report.get('evaluation_set') == 'V_core':
            validation_id, labels_hash, count = (self.v_core_indices_sha256,
                                                self.v_core_sha256, len(self.y_core))
        elif report.get('evaluation_set') == 'external_validation':
            validation_id = labels_hash = self.metadata['external_validation_sha256']
            count = len(self.y_external)
        else:
            raise RuntimeError('Prediction validation identity is missing or unknown')
        # Integrity errors must stop search, not become an undefined-MCS fallback.
        if (report.get('validation_id') != validation_id
                or report.get('evaluation_labels_sha256') != labels_hash):
            raise RuntimeError('Prediction validation identity mismatch')
        with np.load(report['prediction_path'], allow_pickle=False) as saved:
            predictions, probabilities = saved['predictions'], saved['probabilities']
            rows = saved['row_positions']
        if (not np.array_equal(rows, np.arange(count)) or predictions.shape != (count,)
                or report.get('prediction_rows') != count
                or (probabilities.size and probabilities.shape != (count, len(self.classes)))):
            raise RuntimeError('Prediction row order or count mismatch')
        if self.classes:
            mapping = {str(c): c for c in self.classes}
            predictions = np.asarray([mapping[str(x)] for x in predictions])
        return {'predictions': predictions, 'probabilities': probabilities if probabilities.size else None,
                'classes': self.classes}

    def recover_evaluation(self, candidate, stage):
        """Recover a committed result after a coordinator exit, without refitting.

        Ordinary fits require this candidate's charged, recorded attempts. An
        anchor can instead recover its verified parent-reuse record. A raw fit
        report is insufficient: score and immutable identities must have been
        committed by this adapter.
        """
        folder = Path(candidate['code_path']).parent
        saved_path = folder/'candidate.json'
        saved = _read(saved_path) if saved_path.exists() else candidate
        if (self._contract(candidate) != candidate['contract_sha256']
                or file_hash(candidate['code_path']) != candidate['code_sha256']):
            raise ValueError('Candidate implementation/plan mutated before recovery')
        if (saved.get('contract_sha256') != candidate['contract_sha256']
                or saved.get('code_sha256') != candidate['code_sha256']):
            raise ValueError('Saved candidate identity changed before recovery')
        if stage == 'full' and candidate.get('anchor_of'):
            raise ValueError('Coreset calibration anchors cannot receive full training')
        _, binding_hash = self._candidate_coreset(candidate)
        if stage == 'low' and saved.get('coreset_binding') != candidate.get('coreset_binding'):
            raise ValueError('Saved candidate sampling binding changed before recovery')
        if stage == 'low' and (saved.get('parent_low_reuse') or saved.get('equivalent_low_reuse')):
            reuse_key = 'parent_low_reuse' if saved.get('parent_low_reuse') else 'equivalent_low_reuse'
            if saved[reuse_key] != candidate.get(reuse_key):
                raise ValueError('Saved parent low reuse provenance changed before recovery')
            if (folder/'low_reuse'/'evaluation.json').exists():
                return self._commit_parent_low_reuse(candidate)
        attempts = saved.get('fit_attempts', candidate.get('fit_attempts', []))
        candidate['fit_attempts'] = deepcopy(attempts)
        for attempt in reversed(attempts):
            if attempt.get('stage') != stage:
                continue
            result_path = Path(attempt['path'])/'evaluation.json'
            if not result_path.exists():
                continue
            report = _read(result_path)
            if (report.get('candidate_id') != candidate['id'] or report.get('stage') != stage
                    or report.get('code_sha256') != candidate['code_sha256']
                    or report.get('contract_sha256') != candidate['contract_sha256']):
                raise ValueError('Recovered evaluation identity mismatch')
            if stage == 'low' and report.get('coreset_binding_sha256') != binding_hash:
                raise ValueError('Recovered low evaluation uses a different coreset binding')
            if stage == 'low' and binding_hash is not None and (
                    report.get('coreset_version') != candidate['coreset_binding']['version'] or
                    report.get('fit_train_sha256') != candidate['coreset_binding']['csv_sha256'] or
                    report.get('fit_rows') != len(self.core_indices)):
                raise ValueError('Recovered low evaluation sample version/rows mismatch')
            for name in ('unfitted_pipeline', 'fitted_pipeline'):
                if file_hash(report[name+'_path']) != report[name+'_sha256']:
                    raise ValueError('Recovered pipeline artifact changed: ' + name)
            self.load_prediction(report)  # validates labels, validation ID and row alignment
            candidate.update({stage+'_report': report, stage+'_score': report['score'],
                              'status': 'low_evaluated' if stage == 'low' else 'full_evaluated'})
            return report
        return None

    async def evaluate(self, candidate, stage='low'):
        if stage not in {'low', 'full'}:
            raise ValueError('stage must be low or full')
        if stage == 'full' and candidate.get('anchor_of'):
            raise ValueError('Coreset calibration anchors cannot receive full training')
        if candidate.get(stage+'_report') is not None:
            raise ValueError('A candidate cannot repeat an already completed evaluation')
        if candidate.get('status') == 'failed':
            raise ValueError('Cannot evaluate a failed candidate')
        if self._contract(candidate) != candidate['contract_sha256']:
            raise ValueError('Candidate plan mutated after freezing')
        if file_hash(candidate['code_path']) != candidate['code_sha256']:
            raise ValueError('Candidate implementation mutated after freezing')
        low_reference = candidate.get('low_report')
        if stage == 'full' and not low_reference and candidate.get('depth_direct'):
            if candidate.get('sampling_preflight')!={'code_sha256':candidate['code_sha256'],'contract_sha256':self._contract(candidate)}:
                raise ValueError('Deep rollout requires audited stage constructors')
            # _preflight already checked the full factory against the low constructor.
        elif stage == 'full' and not low_reference:
            if not candidate.get('direct_full_certificate'):
                raise ValueError('Full rollout requires this candidate\'s completed low fit')
            low_reference = self._direct_low_constructor(candidate)
        candidate_coreset_path, binding_hash = self._candidate_coreset(candidate)
        if stage == 'low' and (candidate.get('parent_low_reuse') or candidate.get('equivalent_low_reuse')):
            return self._commit_parent_low_reuse(candidate)
        phase = 'coreset' if stage == 'low' else 'full_train'
        budget = self.evaluation_budget if stage == 'low' else self.full_train_budget
        task = {**self.task_context, 'expected_family': candidate['family'],
                'fit_stage': 'evaluation' if stage == 'low' else 'full_train',
                'stage_parameters': candidate['idea']['stage_parameters'][phase],
                'parameter_budget_mode': candidate['parameter_budget_mode'], 'collect_training_feedback': True}
        if 'allowed_model_classes' in candidate:
            task['allowed_model_classes'] = candidate['allowed_model_classes']
        task.update(candidate.get('frozen_stages', {}).get(phase, {}))
        if stage=='full' and candidate.get('depth_direct'):
            task['expected_pipeline_spec']=candidate['preflight_full_spec']
        if stage == 'full' and low_reference:
            task['reference_pipeline_path'] = low_reference['unfitted_pipeline_path']
            if candidate.get('low_report'):
                task['expected_model_class'] = low_reference['actual_budget']['model_class']
        # A retry/resume never overwrites the interrupted attempt's artifacts.
        patterns = [stage+'_attempt_*'] + (['preflight_attempt_*'] if stage == 'low' else [])
        previous = [int(path.name.rsplit('_', 1)[1])
                    for pattern in patterns
                    for path in Path(candidate['code_path']).parent.glob(pattern)
                    if path.name.rsplit('_', 1)[1].isdigit()]
        count = max(previous, default=0)
        while True:
            count += 1
            folder = Path(candidate['code_path']).parent/f'{stage}_attempt_{count}'
            can_repair = False
            try:
                if stage == 'low':
                    can_repair = True
                    await asyncio.to_thread(self._preflight, candidate, count)
                    task['stage_parameters'] = candidate['idea']['stage_parameters'][phase]
                can_repair = False
                await _callback(self.before_fit, stage, candidate)
                candidate.setdefault('fit_attempts', []).append({'stage': stage, 'path': str(folder)})
                self._save(candidate)
                can_repair = True
                report = await asyncio.to_thread(
                    run_fit, Path(candidate['code_path']), candidate_coreset_path if stage == 'low' else self.train_path,
                    self.core_features if stage == 'low' else self.external_features, task, self.classes,
                    budget, self.seed, folder, self.batch_size)
                can_repair = False
                report.update(prediction_path=str(folder/'predictions.npz'),
                              evaluation_set='V_core' if stage == 'low' else 'external_validation',
                              validation_id=(self.v_core_indices_sha256 if stage == 'low'
                                             else self.metadata['external_validation_sha256']),
                              evaluation_labels_sha256=(self.v_core_sha256 if stage == 'low'
                                                        else self.metadata['external_validation_sha256']))
                values = self.load_prediction(report)
                labels = self.y_core if stage == 'low' else self.y_external
                metrics, reason = evaluation_metrics(self.task, labels, values['predictions'],
                                                     values['probabilities'], self.classes)
                measured = metrics[self.metric]
                if measured is None or not math.isfinite(measured):
                    raise ValueError('Undefined rollout score: ' + str(reason))
                report.update(score=float(measured), metric=self.metric, metrics=metrics, stage=stage,
                              candidate_id=candidate['id'],
                              fit_train_sha256=(candidate.get('coreset_binding', {}).get('csv_sha256',
                                  self.metadata['coreset_sha256']) if stage == 'low' else self.metadata['train_sha256']),
                              model_seed=self.seed,
                              direction='maximize' if self.maximize else 'minimize',
                              unfitted_pipeline_path=str(folder/'unfitted.joblib'),
                              fitted_pipeline_path=str(folder/'fitted.joblib'),
                              unfitted_pipeline_sha256=file_hash(folder/'unfitted.joblib'),
                              fitted_pipeline_sha256=file_hash(folder/'fitted.joblib'),
                              code_sha256=candidate['code_sha256'], contract_sha256=candidate['contract_sha256'])
                if stage == 'low' and binding_hash is not None:
                    report.update(coreset_binding_sha256=binding_hash,
                                  coreset_version=candidate['coreset_binding']['version'])
                if stage == 'low':
                    report.update(low_cache_identity=self._low_cache_identity(candidate, report),
                                  prediction_sha256=file_hash(report['prediction_path']),
                                  context_sha256=file_hash(folder/'context.json'))
                report['subgroup_errors'] = self._subgroup_feedback(stage, labels, values)
                report['evaluation_feedback'] = task_metric_feedback(
                    self.task, labels, values['predictions'], values['probabilities'], self.classes,
                    {'fit_stage': phase, 'evaluation_split': report['evaluation_set'],
                     'training_rows': report['fit_rows']}, measured_value=measured)
                write_json(folder/'evaluation.json', report)
                candidate.update({stage+'_score': report['score'], stage+'_report': report,
                                  'status': 'low_evaluated' if stage == 'low' else 'full_evaluated'})
                self._save(candidate)
                return report
            except Exception as error:
                candidate.setdefault('execution_errors', []).append({'stage': stage, 'attempt': count,
                                                                      'error': f'{type(error).__name__}: {error}'})
                if getattr(error, 'budget_exhausted', False):
                    self._save(candidate)
                    raise
                if infrastructure_failure(error):
                    # Coordinator handles bounded identical retries. Host errors
                    # must not rewrite code or permanently invalidate a method.
                    candidate.update(status='interrupted', failure_stage=stage,
                                     failure_kind='infrastructure_interruption',
                                     error=f'{type(error).__name__}: {error}')
                    self._save(candidate)
                    raise
                # API/budget/IO failures are not reasons to rewrite a model.
                repairable = (can_repair and stage == 'low' and not candidate.get('frozen_stages')
                              and not isinstance(error, ModelOutsideSearch)
                              and not candidate.get('sampling_contract_frozen') and not candidate.get('anchor_of') and self.model is not None
                              and candidate.get('repairs_used', 0) < self.max_repairs
                              and isinstance(error, (ValueError, TypeError, RuntimeError))
                              and not getattr(error, 'budget_exhausted', False))
                if repairable:
                    try:
                        await self._repair(candidate, error, 'low_fit')
                    except Exception as repair_error:
                        error = repair_error
                    else:
                        continue
                candidate.update(status='failed', failure_stage=stage, error=f'{type(error).__name__}: {error}')
                self._save(candidate)
                raise error
