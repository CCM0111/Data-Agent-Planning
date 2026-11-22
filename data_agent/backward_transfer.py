"""One-hop, experiment-backed experience updates after a normal task completes."""

from copy import deepcopy
from dataclasses import fields
import json
import math
from pathlib import Path
import time

from .diagnostics import file_hash, runtime_environment
from .experience_library import _atomic_write, _digest, record_hash, task_key
from .experience_lineage import generation_lineage
from .executor import StageExecutor
from .search import Node, Search


POLICY_VERSION = 1


def _rows(nodes):
    return {n['id'] if isinstance(n, dict) else n.id: n if isinstance(n, dict) else vars(n) for n in nodes}


def _parent(row):
    parent = row.get('parent')
    return parent if isinstance(parent, int) else getattr(parent, 'id', None)


def _value(row, metric):
    value = row.get('result', {}).get('metrics', {}).get(metric)
    return value if type(value) in (int, float) and math.isfinite(value) else None


def trigger_decision(donor, source, nodes):
    """Choose a source-directed revisit; never infer a failed method from no execution."""
    if donor is None or donor.get('summary', {}).get('status') != 'completed':
        return {'eligible': False, 'reason': 'donor_not_summarized'}
    if donor.get('backward_transfer') is not None:
        return {'eligible': False, 'reason': 'no_recursive_transfer'}
    if donor.get('summary', {}).get('library_action') not in {'insert', 'replace'}:
        return {'eligible': False, 'reason': 'donor_not_admitted'}
    if source is None:
        return {'eligible': False, 'reason': 'no_source'}
    if donor['task_key'] == source['task_key']:
        return {'eligible': False, 'reason': 'same_task'}
    rows = _rows(nodes)
    winner = rows.get(donor['source_refs']['mt_node_id'])
    if winner is None or winner.get('status') != 'success':
        return {'eligible': False, 'reason': 'winner_unavailable'}
    source_version = record_hash(source)

    def guided(row):
        ref = row.get('experience_source') or {}
        return (row.get('generation_mode') == 'experience'
                and row.get('origin', 'generate') == 'generate'
                and ref.get('task_key') == source['task_key']
                and ref.get('record_sha256') == source_version)

    def guided_path(row):
        fe = rows.get(_parent(row), {})
        return guided(row) and fe.get('stage') == 'FE' and guided(fe)

    lineage = generation_lineage(winner['id'], nodes)
    base = {'eligible': False, 'winner_node_id': winner['id'], 'winner_lineage': lineage}
    if guided_path(winner):
        return {**base, 'reason': 'direct_guided_winner'}
    if lineage['kind'] == 'unknown':
        return {**base, 'reason': 'unknown_winner_lineage'}
    metric = donor['protocol']['metric']
    maximize = donor['result']['validation']['direction'] == 'maximize'
    winner_value = _value(winner, metric)
    if winner_value is None or winner_value != donor['result']['validation']['value']:
        return {**base, 'reason': 'winner_metric_mismatch'}
    successful = [row for row in rows.values() if row.get('stage') == 'MT'
                  and row.get('status') == 'success' and guided_path(row) and _value(row, metric) is not None]
    if successful:
        best = (max if maximize else min)(successful, key=lambda row: _value(row, metric))
        old = _value(best, metric)
        delta = winner_value - old if maximize else old - winner_value
        return {**base, 'eligible': delta > 0,
                'reason': 'exploration_improved' if delta > 0 else 'no_strict_improvement',
                'guided_node_id': best['id'], 'guided_value': old, 'winner_value': winner_value,
                'improvement': delta}
    failures = [row['id'] for row in rows.values() if row.get('status') == 'failed'
                and ((row.get('stage') == 'MT' and guided_path(row))
                     or (row.get('stage') == 'FE' and guided(row)))]
    if failures:
        return {**base, 'eligible': True, 'reason': 'guided_execution_failed',
                'failed_guided_node_ids': failures, 'winner_value': winner_value}
    return {**base, 'reason': 'guided_route_not_evaluated'}


class ReplayUnavailable(ValueError):
    pass


class CountingModel:
    def __init__(self, model):
        self.model, self.calls = model, []

    async def generate(self, prompt, **kwargs):
        request = json.loads(prompt)
        row = {'operation': request.get('operation'), 'stage': request.get('current_stage')}
        self.calls.append(row)
        start = time.monotonic()
        try:
            return await self.model.generate(prompt, **kwargs)
        finally:
            row['elapsed_seconds'] = time.monotonic() - start


class BackwardTransfer:
    def __init__(self, library, directory=None, *, enabled=True):
        if type(enabled) is not bool:
            raise ValueError('backward_transfer.enabled must be boolean')
        self.library = library
        self.directory = Path(directory) if directory is not None else library.directory / 'backward_transfers'
        self.enabled = enabled

    async def process(self, donor, source, nodes, model, output, *, timeout=120, max_repairs=1):
        """Optional post-task work; its failure never discards the donor search."""
        output = Path(output)
        try:
            result = await self._process(donor, source, nodes, model, timeout, max_repairs)
        except Exception as error:
            result = {'status': 'failed', 'reason': type(error).__name__, 'error': str(error)[:1000]}
        _atomic_write(output / 'backward_transfer_status.json', result)
        return result

    async def _process(self, donor, source, nodes, model, timeout, max_repairs):
        if not self.enabled or not self.library.enabled:
            return {'status': 'skipped', 'reason': 'disabled'}
        trigger = trigger_decision(donor, source, nodes)
        if not trigger['eligible']:
            return {'status': 'skipped', **trigger}
        donor_version, source_version = record_hash(donor), record_hash(source)
        identity = {'policy_version': POLICY_VERSION, 'donor_version': donor_version,
                    'source_version': source_version, 'target_task_key': source['task_key']}
        event_id = _digest(identity)
        directory = self.directory / event_id
        directory.parent.mkdir(parents=True, exist_ok=True)
        try:
            directory.mkdir()  # Atomic claim: repeated/concurrent calls cannot run this event twice.
        except FileExistsError:
            return {'status': 'skipped', 'reason': 'event_already_claimed', 'event_id': event_id,
                    'event_dir': str(directory)}
        event = {**identity, 'event_id': event_id, 'status': 'running', 'trigger': trigger,
                 'donor_task_key': donor['task_key'], 'recursive_transfer': False,
                 'max_backward_updates': self.library.max_backward_updates}
        _atomic_write(directory / 'event.json', event)
        _atomic_write(directory / 'donor_experience.json', donor)
        _atomic_write(directory / 'retrieved_source.json', source)
        counter, search = CountingModel(model), None
        started = time.monotonic()
        decision = {'status': 'interrupted', 'reason': 'unfinished'}
        try:
            baseline = self.library._existing(source)
            if baseline is None:
                raise ReplayUnavailable('target_missing')
            event['target_start_version'] = record_hash(baseline)
            _atomic_write(directory / 'target_before.json', baseline)
            if self.library.backward_update_count(baseline) >= self.library.max_backward_updates:
                raise ReplayUnavailable('backward_update_limit')
            if baseline['protocol'] != source['protocol']:
                raise ReplayUnavailable('source_protocol_changed')
            task, eda = self._restore(baseline, directory / 'replay_run')
            executor = StageExecutor(counter, task, directory / 'replay_run', timeout=timeout,
                                     max_repairs=max_repairs, experience_library=self.library)
            search = Search(executor, directory / 'replay_run', rollouts=1,
                            branches={'EDA': 1, 'FE': 1, 'MT': 1})
            eda.parent = search.root
            search.root.children.append(eda)
            search.nodes.append(eda)
            for stage in ('FE', 'MT'):
                parent = search.nodes[-1]
                node = Node(len(search.nodes), stage, parent=parent, origin='backward_transfer')
                parent.children.append(node)
                search.nodes.append(node)
                search.save()
                await executor.generate_guided(node, donor)
                search.save()
                await executor.execute(node)
                search.save()
                if node.status != 'success':
                    decision = {'status': 'rejected', 'reason': f'{stage.lower()}_execution_failed'}
                    break
            else:
                search.best = search.nodes[-1]
                search.backpropagate(search.best, search.best.reward)
                search.record_rollout(search.best)
                search.stop_reason = 'backward_single_pipeline_completed'
                search.save()
                context = {'event_id': event_id, 'donor_task_key': donor['task_key'],
                           'donor_version': donor_version, 'source_version': source_version,
                           'target_start_version': event['target_start_version']}
                # No Search.run/finalizer here: this path cannot schedule another transfer.
                candidate = await self.library.curate(task, search.nodes, search.output, counter,
                    search.history, backward_transfer=context, baseline=baseline)
                action = candidate['summary'].get('library_action', candidate['summary'].get('reason'))
                accepted = candidate['summary']['status'] == 'completed' and action == 'replace'
                value = candidate['result']['validation']
                old = baseline['result']['validation']['value']
                delta = value['value'] - old if value['direction'] == 'maximize' else old - value['value']
                decision = {'status': 'accepted' if accepted else 'rejected',
                            'reason': action or 'summary_failed',
                            'baseline_at_start': baseline['result']['validation'],
                            'candidate_validation': value, 'improvement_vs_start': delta,
                            'candidate_summary_status': candidate['summary']['status']}
                if accepted:
                    decision.update(new_version=record_hash(candidate), lifecycle=candidate['lifecycle'])
            latest = self.library._existing(source)
            if latest is not None:
                decision['target_after_version'] = record_hash(latest)
                decision['backward_updates_after'] = self.library.backward_update_count(latest)
        except ReplayUnavailable as error:
            decision = {'status': 'skipped', 'reason': str(error)}
        except Exception as error:
            decision = {'status': 'failed', 'reason': type(error).__name__, 'error': str(error)[:1000]}
        finally:
            if search is not None:
                search.save()
            attempts = [{'stage': n.stage, **a} for n in search.nodes if n.stage in {'FE', 'MT'}
                        for a in n.attempts] if search is not None else []
            event.update(status=decision['status'], elapsed_seconds=time.monotonic() - started,
                         costs={'llm_calls': len(counter.calls), 'calls': counter.calls,
                                'execution_attempts': len(attempts),
                                'mt_fit_attempts': sum(a['stage'] == 'MT' and a.get('fit_started', False) for a in attempts),
                                'execution_seconds': sum(a.get('elapsed_seconds', 0) for a in attempts),
                                'token_usage': None})
            _atomic_write(directory / 'event.json', event)
            _atomic_write(directory / 'decision.json', decision)
        return {**decision, 'event_id': event_id, 'event_dir': str(directory)}

    def _restore(self, baseline, output):
        """Reuse the target's recorded task/split/EDA; never split data or run EDA again."""
        protocol = baseline['protocol']
        if (not protocol.get('comparable_across_runs') or type(protocol.get('seed')) is not int
                or protocol.get('runtime_environment') is None):
            raise ReplayUnavailable('incomplete_replay_protocol')
        if protocol['runtime_environment'] != runtime_environment():
            raise ReplayUnavailable('runtime_environment_changed')
        run_dir = Path(baseline['source_refs']['run_dir'])
        try:
            task = json.loads((run_dir / 'task.json').read_text())
            tree = json.loads((run_dir / 'tree.json').read_text())
            if task_key(task) != baseline['task_key']:
                raise ReplayUnavailable('task_identity_changed')
            for key in ('seed', 'metric', 'target', 'problem_type'):
                if task.get(key) != protocol.get(key):
                    raise ReplayUnavailable(f'{key}_changed')
            for split in ('train', 'validation'):
                path = Path(task[f'{split}_path'])
                path = path if path.is_absolute() else run_dir / path
                if file_hash(path) != protocol[f'{split}_sha256']:
                    raise ReplayUnavailable(f'{split}_data_changed')
                task[f'{split}_path'] = str(path.resolve())
            rows = {n['id']: n for n in tree}
            fe = rows[baseline['source_refs']['fe_node_id']]
            row = rows[fe['parent']]
            if row['stage'] != 'EDA' or row['status'] != 'success':
                raise ReplayUnavailable('successful_eda_missing')
            code_path = Path(row['code_path'])
            code_path = code_path if code_path.is_absolute() else run_dir / code_path
            if not row.get('code_hash') or file_hash(code_path) != row['code_hash']:
                raise ReplayUnavailable('eda_code_changed')
            report_path = Path(row['result']['artifacts']['eda'])
            report_path = report_path if report_path.is_absolute() else run_dir / report_path
            report = json.loads(report_path.read_text())
            if report != row['result']['report']:
                raise ReplayUnavailable('eda_report_changed')
            code = code_path.read_text()
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ReplayUnavailable('replay_artifacts_unavailable') from error
        task.pop('test_path', None)
        output.mkdir(parents=True)
        _atomic_write(output / 'task.json', task)
        _atomic_write(output / 'runtime_environment.json', runtime_environment())
        restored = output / 'node_0001_EDA'
        restored.mkdir()
        (restored / 'code.py').write_text(code)
        _atomic_write(restored / 'eda.json', report)
        _atomic_write(restored / 'reuse.json', {'source_run_dir': str(run_dir), 'source_node_id': row['id'],
                                               'executed_again': False})
        allowed = {f.name for f in fields(Node)} - {'id', 'stage', 'parent', 'children'}
        node = Node(1, 'EDA', **{k: deepcopy(v) for k, v in row.items() if k in allowed})
        node.origin = 'reused_eda'
        node.code_path = str(restored / 'code.py')
        node.attempts = []
        node.visits, node.total_reward = 0, 0.0
        node.result['artifacts'] = {'eda': str(restored / 'eda.json')}
        return task, node
