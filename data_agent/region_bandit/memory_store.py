"""Transactional immutable decisions, feedback and bounded-policy snapshots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import time

from .context import default_feature_schema
from .memory_retrieval import compatible, compatibility
from .policy import initial_snapshot, rebuild_snapshot


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def validate_frozen(snapshot, identity, excluded_groups=(), records=()):
    excluded = set(excluded_groups) | {identity['dataset_group_id']}
    overlap = excluded & set(snapshot.get('training_dataset_groups', []))
    if overlap:
        raise ValueError('Frozen policy was trained on evaluation dataset groups: ' + ', '.join(sorted(overlap)))
    for record in records:
        previous = record.get('identity', record)
        if compatible(identity, previous) and previous.get('dataset_group_id') in excluded:
            raise ValueError('Frozen memory contains evaluation dataset group: ' + previous['dataset_group_id'])
    if snapshot.get('compatibility') and snapshot['compatibility'] != compatibility(identity):
        raise ValueError('Frozen policy versions or metric do not match this task')
    return True


class MemoryStore:
    def __init__(self, path, *, read_only=False):
        self.path, self.read_only = Path(path), bool(read_only)
        if self.read_only:
            self.connection = sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True, timeout=30)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(str(self.path), timeout=30)
            self.connection.execute('PRAGMA journal_mode=WAL')
            self.connection.executescript('''
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY, payload TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS feedback (
                    decision_id TEXT PRIMARY KEY, event_id TEXT UNIQUE NOT NULL,
                    payload TEXT NOT NULL, snapshot_id TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY, compatibility TEXT NOT NULL,
                    payload TEXT NOT NULL, created REAL NOT NULL);
            ''')
            self.connection.commit()

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _writable(self):
        if self.read_only:
            raise RuntimeError('Frozen memory store is read-only')

    def get_snapshot(self, snapshot_id):
        row = self.connection.execute('SELECT payload FROM snapshots WHERE snapshot_id=?', (snapshot_id,)).fetchone()
        if row is None:
            raise KeyError('Unknown policy snapshot: ' + str(snapshot_id))
        return json.loads(row[0])

    def save_snapshot(self, snapshot, identity=None):
        self._writable()
        value = dict(snapshot)
        key = compatibility(identity) if identity else value.get('compatibility', {})
        with self.connection:
            self._insert_snapshot(value, key)
        return value

    def _insert_snapshot(self, snapshot, key):
        payload = _json(snapshot)
        existing = self.connection.execute('SELECT payload FROM snapshots WHERE snapshot_id=?', (snapshot['snapshot_id'],)).fetchone()
        if existing and existing[0] != payload:
            raise ValueError('Attempt to change an immutable snapshot')
        self.connection.execute('INSERT OR IGNORE INTO snapshots VALUES (?,?,?,?)',
                                (snapshot['snapshot_id'], _json(key), payload, time.time()))

    def latest_snapshot(self, identity=None, *, schema=None, config=None):
        if identity:
            # A manually frozen development subset may have been saved after
            # the online snapshot. "Latest" for online development means all
            # compatible committed events, never the most recently saved file.
            snapshot = rebuild_snapshot(self.events(identity), schema=schema or default_feature_schema(),
                                        config=config, identity=compatibility(identity))
            if not self.read_only:
                self.save_snapshot(snapshot, identity)
            return snapshot
        row = self.connection.execute('SELECT payload FROM snapshots ORDER BY created DESC,rowid DESC LIMIT 1').fetchone()
        if row:
            return json.loads(row[0])
        snapshot = initial_snapshot(schema, config)
        if not self.read_only:
            self.save_snapshot(snapshot, identity)
        return snapshot

    def snapshots(self):
        return [json.loads(row[0]) for row in self.connection.execute('SELECT payload FROM snapshots ORDER BY created,rowid')]

    def get_decision(self, decision_id):
        row = self.connection.execute('SELECT payload FROM decisions WHERE decision_id=?', (decision_id,)).fetchone()
        return None if row is None else json.loads(row[0])

    def get_committed_feedback(self, decision_id, *, decision=None):
        """Recover the original atomic commit without recomputing diagnostics.

        A crash can occur after SQLite commits but before filesystem artifacts
        are written. Return both the immutable event and its original snapshot
        in one read, rather than rebuilding against newer global history.
        """
        row = self.connection.execute('''
            SELECT feedback.payload, snapshots.payload, decisions.payload
            FROM feedback
            JOIN snapshots ON snapshots.snapshot_id=feedback.snapshot_id
            JOIN decisions ON decisions.decision_id=feedback.decision_id
            WHERE feedback.decision_id=?
        ''', (decision_id,)).fetchone()
        if row is None:
            return None
        if decision is not None and row[2] != _json(decision):
            raise ValueError('Committed feedback belongs to a different frozen decision')
        return {'event': json.loads(row[0]), 'snapshot': json.loads(row[1])}

    def decisions(self):
        return [json.loads(row[0]) for row in self.connection.execute('SELECT payload FROM decisions ORDER BY created,rowid')]

    def freeze_decision(self, decision):
        self._writable()
        identifier, payload = decision['decision_id'], _json(decision)
        with self.connection:
            previous = self.get_decision(identifier)
            if previous is not None and _json(previous) != payload:
                raise ValueError('Attempt to change an immutable decision')
            self.connection.execute('INSERT OR IGNORE INTO decisions VALUES (?,?,?)', (identifier, payload, time.time()))
        return decision

    def events(self, identity=None, *, snapshot=None):
        rows = [json.loads(row[0]) for row in self.connection.execute('SELECT payload FROM feedback ORDER BY created,rowid')]
        cutoff = set(snapshot['memory_event_ids']) if snapshot is not None else None
        selected = [row for row in rows if (identity is None or compatible(identity, row['identity'])) and
                    (cutoff is None or row['event_id'] in cutoff)]
        if snapshot is not None:
            if {row['event_id'] for row in selected} != cutoff or len(selected) != len(cutoff):
                raise ValueError('Frozen memory is missing referenced events or contains incompatible events')
            ordered = sorted(selected, key=lambda row: row['event_id'])
            if snapshot.get('memory_data_hash') is not None and _hash(ordered) != snapshot['memory_data_hash']:
                raise ValueError('Frozen memory event contents changed')
            training = [row for row in ordered if row.get('update_eligible', True) and row.get('reward') is not None]
            if ([row['event_id'] for row in training] != snapshot['training_event_ids']
                    or _hash(training) != snapshot['training_data_hash']):
                raise ValueError('Frozen policy training events changed')
        return selected

    def cases(self, identity=None, *, snapshot=None):
        cases = []
        for event in self.events(identity, snapshot=snapshot):
            base = {key: event.get(key) for key in ('identity', 'dataset_group_id', 'context', 'action_type',
                    'executed_action', 'frozen_config', 'mode')}
            cases.append({**base, 'record_id': event['event_id'], 'kind': 'action_feedback',
                          'feedback': event['feedback'], 'signal_type': 'executed_action_reward'})
            for index, region in enumerate(event.get('region_feedback', [])):
                region_context = region.get('context')
                # No fabricated predecision vector for a region that was not stored.
                if not region_context:
                    continue
                cases.append({**base, 'context': region_context, 'record_id': event['event_id'] + ':region:' + str(index),
                              'kind': 'region_feedback', 'feedback': region,
                              'region': region.get('region', {}), 'rule_types': region.get('rule_types'),
                              'signal_type': region.get('signal_type', 'specific_coreset_reversal_diagnostic')})
        return cases

    def validate_frozen(self, snapshot, identity, excluded_groups=()):
        return validate_frozen(snapshot, identity, excluded_groups, self.events(identity, snapshot=snapshot))

    def commit_feedback(self, decision, action_feedback, region_feedback, *, mode='develop_online', schema=None, config=None):
        if mode == 'evaluate_frozen':
            return {'status': 'record_only', 'snapshot_id': decision['policy_snapshot_id'], 'updated': False}
        if mode != 'develop_online':
            raise ValueError('Unknown region memory mode')
        if decision.get('mode', mode) != mode:
            raise ValueError('A frozen evaluation decision cannot update development memory')
        self._writable()
        identity = decision['identity']
        comparison_scope = decision.get('comparison_scope', 'initial')
        if comparison_scope not in ('initial', 'mcts_parent_anchor'):
            raise ValueError('Unknown reward comparison scope')
        if comparison_scope == 'mcts_parent_anchor' and not identity.get('reward_protocol_hash'):
            raise ValueError('Anchored feedback requires an explicit reward protocol identity')
        if identity.get('reward_protocol_hash') is not None:
            from .config import reward_protocol_hash
            if identity['reward_protocol_hash'] != reward_protocol_hash(config, comparison_scope=comparison_scope):
                raise ValueError('Feedback reward configuration differs from the frozen decision')
        identifier = decision['decision_id']
        chosen = decision.get('chosen_action_id', decision.get('chosen_action'))
        if isinstance(chosen, dict):
            chosen = chosen['action_id']
        matches = [a for a in decision['actions'] if a['action_id'] == chosen]
        if len(matches) != 1:
            raise ValueError('Selected action must occur exactly once in the frozen decision')
        if action_feedback.get('action_id', chosen) != chosen:
            raise ValueError('Unexecuted actions cannot receive reward labels')
        action = matches[0]
        context = action.get('context', action)
        feedback = dict(action_feedback)
        section = (config or {}).get('region_bandit', config or {})
        objective = section.get('reward', {}).get('objective', 'absolute')
        if feedback.get('reward_objective', 'absolute') != objective:
            raise ValueError('Feedback objective differs from frozen reward configuration')
        seed_rewards = feedback.get('seed_rewards')
        if seed_rewards:
            from .reward import aggregate_seed_rewards
            if objective == 'paired_delta':
                for row in seed_rewards:
                    if row.get('reward') is not None:
                        before, after = row.get('before_weighted_accuracy'), row.get('after_weighted_accuracy')
                        if (before is None or after is None or not 0 <= before <= 1 or not 0 <= after <= 1
                                or abs(row['reward']-(after-before)) > 1e-12):
                            raise ValueError('Delta reward must match its paired accuracies')
            aggregated = aggregate_seed_rewards(seed_rewards, objective=objective)
            if feedback.get('reward') is not None and (aggregated['reward'] is None
                    or abs(float(feedback['reward'])-aggregated['reward']) > 1e-12):
                raise ValueError('Aggregated reward must equal the mean of informative per-seed rewards')
            aggregated['update_eligible'] &= feedback.get('update_eligible', True)
            feedback.update(aggregated)
        eligible = (feedback.get('update_eligible', True) and feedback.get('reward') is not None and
                    feedback.get('status', 'ok') not in ('reward_missing', 'reward_uninformative', 'infrastructure_failure'))
        if feedback.get('reward') is not None and not (-1 if objective == 'paired_delta' else 0) <= float(feedback['reward']) <= 1:
            raise ValueError('Action reward outside objective range')
        if objective == 'paired_delta' and feedback.get('reward') is not None:
            paired = feedback.get('paired_comparison', {})
            before, after = paired.get('before_weighted_accuracy'), paired.get('after_weighted_accuracy')
            if (before is None or after is None or not 0 <= before <= 1 or not 0 <= after <= 1
                    or abs(feedback['reward']-(after-before)) > 1e-12):
                raise ValueError('Delta feedback must match paired comparison')
        if comparison_scope == 'mcts_parent_anchor':
            from .reward import anchored_sample_weight
            anchor = decision.get('anchor_method_id')
            if anchor is None or feedback.get('anchor_method_id', anchor) != anchor:
                raise ValueError('Anchored feedback must reference its frozen parent method')
            if feedback.get('comparison_scope', comparison_scope) != comparison_scope:
                raise ValueError('Feedback comparison scope differs from the frozen decision')
            pair_count = feedback.get('comparable_pair_count', 0)
            expected_weight = anchored_sample_weight(pair_count, config)
            sample_weight = feedback.get('sample_weight', expected_weight)
            if (type(sample_weight) not in (int, float) or sample_weight != expected_weight):
                raise ValueError('Anchored sample weight must match its frozen confidence protocol')
            if eligible:
                observed = feedback.get('method_ids', [])
                if (feedback.get('comparison_scope') != comparison_scope
                        or feedback.get('anchor_method_id') != anchor
                        or len(observed) not in (2, 3) or len(set(observed)) != len(observed)
                        or anchor not in observed
                        or feedback.get('observed_method_count') != len(observed)
                        or not 1 <= pair_count <= len(observed) * (len(observed) - 1) // 2):
                    raise ValueError('Informative anchored feedback requires parent plus one or two children')
            feedback.update(comparison_scope=comparison_scope, anchor_method_id=anchor,
                            sample_weight=float(sample_weight))
        elif 'sample_weight' in feedback or feedback.get('comparison_scope') == 'mcts_parent_anchor':
            raise ValueError('Anchored confidence weights require an explicit MCTS comparison protocol')
        event = {'event_id': 'action:' + identifier, 'decision_id': identifier, 'identity': identity,
                 'dataset_group_id': identity['dataset_group_id'], 'mode': mode,
                 'features': context['features'], 'schema_id': context['schema_id'], 'context': context,
                 'action_type': action.get('type', action.get('action_type')),
                 'executed_action': {k: v for k, v in action.items() if k not in ('context', 'features', 'action_features')},
                 'api_score': action.get('api_score', 0), 'reward': feedback.get('reward'),
                 'update_eligible': bool(eligible), 'feedback': feedback,
                 'frozen_config': decision.get('frozen_config', context.get('method_summary', {})),
                 'region_feedback': list(region_feedback or [])}
        if comparison_scope == 'mcts_parent_anchor':
            event.update(comparison_scope=comparison_scope, anchor_method_id=anchor,
                         sample_weight=float(sample_weight))
        connection = self.connection
        connection.execute('BEGIN IMMEDIATE')
        try:
            stored_decision = self.get_decision(identifier)
            if stored_decision is None or _json(stored_decision) != _json(decision):
                raise ValueError('Feedback requires the exact previously frozen decision')
            previous = connection.execute('SELECT payload,snapshot_id FROM feedback WHERE decision_id=?', (identifier,)).fetchone()
            if previous:
                if previous[0] != _json(event):
                    raise ValueError('Decision feedback was already committed with different contents')
                connection.commit()
                return self.get_snapshot(previous[1])
            legal = self.events(identity) + [event]
            snapshot = rebuild_snapshot(legal, schema=schema or default_feature_schema(), config=config,
                                        identity=compatibility(identity))
            self._insert_snapshot(snapshot, compatibility(identity))
            connection.execute('INSERT INTO feedback VALUES (?,?,?,?,?)',
                               (identifier, event['event_id'], _json(event), snapshot['snapshot_id'], time.time()))
            connection.commit()
            return snapshot
        except BaseException:
            connection.rollback()
            raise
