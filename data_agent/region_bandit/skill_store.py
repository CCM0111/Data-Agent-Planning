"""Transactional facts, immutable skill versions, and verified promotion gates."""
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

import numpy as np

from .procedural_skills import initial_bundle, validate_bundle, apply_edit
from .skill_prompt import method_brief
from .schemas import stable_hash
from .context import pair_at_boundary

GATE_DEFAULTS = dict(min_sources=6, min_delta=.01, bootstrap_repeats=2000,
                     bootstrap_seed=2026, min_bootstrap_lower=0., extra_branch_penalty=.01)


class InsufficientDevelopmentEvidence(ValueError):
    """Not enough distinct development sources have an unconsumed decision."""


def development_decision_key(fact):
    # A later seed, branch or bundle replay is not a new decision checkpoint.
    return stable_hash({k: fact[k] for k in ('source', 'task', 'bank_hash', 'decision_id')})


def gate_settings(settings=None):
    cfg = {**GATE_DEFAULTS, **(settings or {})}
    if (set(cfg) != set(GATE_DEFAULTS) or type(cfg['min_sources']) is not int or cfg['min_sources'] < 2
            or type(cfg['bootstrap_repeats']) is not int or not 100 <= cfg['bootstrap_repeats'] <= 100000
            or type(cfg['bootstrap_seed']) is not int or cfg['bootstrap_seed'] < 0
            or any(type(cfg[k]) not in (int, float) or not np.isfinite(cfg[k]) or cfg[k] < 0
                   for k in ('min_delta', 'min_bootstrap_lower', 'extra_branch_penalty'))):
        raise ValueError('Invalid predeclared validation gate')
    return cfg


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def sampling_path(context, methods=None, *, actions=(), regions=()):
    """Bounded pre-action context to interpret a completed sampling reward."""
    region = context.get('region_summary') or {}
    bank = []
    for method in methods or context.get('method_summary', []):
        idea = method.get('idea') or {}
        brief = method_brief(method)
        entry = {k: brief[k] for k in ('id', 'method_id', 'name', 'family', 'model',
                                      'implementation', 'parameters', 'fe', 'fixed_parameters', 'change_reason') if k in brief}
        params = method.get('stage_parameters') or (idea.get('stage_parameters') if isinstance(idea, dict) else None)
        if params:
            entry['parameter_hash'] = stable_hash(params)
        bank.append(entry)
    summary = context.get('action_summary', {})
    effects = {}
    for seed, preview in (summary.get('cross_region_preview') or {}).items():
        effect = {k: preview[k] for k in ('sample_replaced_count', 'sample_replaced_fraction') if k in preview}
        target = preview.get('regions', {}).get(summary.get('region_id'))
        if target:
            effect['target_counts'] = {k: target[k] for k in ('before', 'after') if k in target}
        if effect:
            effects[str(seed)] = effect
    options = {}
    for region_spec in regions:
        rid = region_spec['id']
        stats = {**region_spec.get('statistics', {}), **region_spec}
        options[rid] = {k: stats[k] for k in ('rule', 'discovery_kind', 'q_fraction',
                        'reference_fraction', 'coverage_gap', 'direction_consistent') if k in stats}
        options[rid]['actions'] = []
    for action in actions:
        aid = action['action_id']
        rid = action.get('region_id')
        if rid is None:
            continue
        observed = action.get('context', {}).get('region_summary', {})
        option = options.setdefault(rid, {k: observed[k] for k in ('rule', 'discovery_kind',
            'q_fraction', 'c0_fraction', 'coverage_gap', 'direction_consistent') if k in observed})
        if 'protected_q_indices' in action and observed:
            option.update({k:observed[k] for k in ('q_fraction','c0_fraction','coverage_gap') if k in observed})
            if 'c0_fraction' in observed:
                option['reference_fraction'] = observed['c0_fraction']
            option['quota_scope'] = 'editable_rows'
        option.setdefault('actions', []).append(aid)
    return dict(**({'candidate_regions': options} if actions or regions else {}), **({'sample_effect_by_seed': effects} if effects else {}), eda={k: v for k, v in (context.get('task_summary') or {}).items()
                    if k in ('problem_type', 'metric', 'n_rows', 'n_features', 'categorical_fraction',
                             'missing_fraction', 'target_imbalance', 'target_tail_fraction',
                             'coreset_fraction', 'proxy_disagreement')},
                methods=bank, region={k: region[k] for k in ('rule', 'discovery_kind',
                    'primary_pair', 'primary_pair_method_ids', 'coverage_gap', 'direction_consistent') if k in region}
                | ({'pair_at_boundary': pair_at_boundary(context)} if pair_at_boundary(context) is not None else {}))


class SkillStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS bundles(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY, role TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS facts(id TEXT PRIMARY KEY, source TEXT NOT NULL,
            role TEXT NOT NULL, bundle TEXT NOT NULL, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS skill_trajectories(
            fact_id TEXT PRIMARY KEY REFERENCES facts(id), payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS historical_context(
            fact_id TEXT PRIMARY KEY REFERENCES facts(id), payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS edits(id TEXT PRIMARY KEY, parent TEXT NOT NULL,
            candidate TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS gates(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS proposal_failures(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        ''')
        bundle = initial_bundle()
        with self.transaction():
            ident = self._bundle(bundle)
            self.db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', ('active', ident))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def _bundle(self, value):
        validate_bundle(value)
        ident = stable_hash(value)
        self.db.execute('INSERT OR IGNORE INTO bundles VALUES (?,?)', (ident, encoded(value)))
        return ident

    @property
    def active(self):
        return self.db.execute('SELECT value FROM meta WHERE key="active"').fetchone()[0]

    def bundle(self, ident=None):
        row = self.db.execute('SELECT payload FROM bundles WHERE id=?', (ident or self.active,)).fetchone()
        if row is None:
            raise ValueError('Unknown skill version')
        return json.loads(row[0])

    def register_sources(self, groups, role):
        if role not in ('development', 'validation', 'test') or not groups:
            raise ValueError('Explicit source roles required')
        with self.transaction():
            for group in groups:
                if not isinstance(group, str) or not group.strip():
                    raise ValueError('Nonempty source ID required')
                old = self.db.execute('SELECT role FROM sources WHERE id=?', (group,)).fetchone()
                if old and old[0] != role:
                    raise ValueError('Source overlap between skill splits: '+group)
                self.db.execute('INSERT OR IGNORE INTO sources VALUES (?,?)', (group, role))

    def initialize_guidance(self, bundle, provenance):
        """Seed an empty library; never import old outcomes as new observations."""
        validate_bundle(bundle)
        identity = stable_hash(dict(bundle=bundle, provenance=provenance))
        with self.transaction():
            previous = self.db.execute('SELECT value FROM meta WHERE key="initial_guidance"').fetchone()
            if previous:
                if previous[0] != identity:
                    raise ValueError('Initial guidance changed on resume')
                return
            if any(self.db.execute('SELECT count(*) FROM '+table).fetchone()[0]
                   for table in ('facts', 'edits', 'gates')):
                raise ValueError('Initial guidance requires an empty experiment library')
            ident = self._bundle(bundle)
            self.db.execute('UPDATE meta SET value=? WHERE key="active"', (ident,))
            self.db.execute('INSERT INTO meta VALUES (?,?)', ('initial_guidance', identity))
            self.db.execute('INSERT INTO meta VALUES (?,?)', ('initial_guidance_provenance', encoded(provenance)))

    def exploration_context(self, source, limit):
        role = self.db.execute('SELECT role FROM sources WHERE id=?', (source,)).fetchone()
        if role is None:
            raise ValueError('Register source before planning exploration')
        used = self.db.execute('SELECT value FROM meta WHERE key="focus_exploration_sources"').fetchone()
        sources = json.loads(used[0]) if used else []
        return dict(role=role[0], source=source, max_sources=limit,
                    eligible=role[0] == 'development' and limit > 0
                    and (source in sources or len(sources) < limit))

    def reserve_focus_exploration(self, source, limit):
        with self.transaction():
            context = self.exploration_context(source, limit)
            if not context['eligible']:
                raise ValueError('Focus exploration budget or role disallows this source')
            old = self.db.execute('SELECT value FROM meta WHERE key="focus_exploration_sources"').fetchone()
            sources = json.loads(old[0]) if old else []
            if source not in sources:
                sources.append(source)
            self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',
                            ('focus_exploration_sources', encoded(sources)))

    def record_fact(self, fact):
        role = self.db.execute('SELECT role FROM sources WHERE id=?', (fact['source'],)).fetchone()
        paired_replay = (role is not None and role[0] == 'development' and fact['role'] == 'validation'
                         and fact.get('execution_kind') == 'checkpoint_replay')
        if paired_replay:
            plans = self.db.execute(
                "SELECT value FROM meta WHERE key LIKE 'mcts_gate:%'")
            paired_replay = any(plan.get('vote_protocol') == 'matched-four-v1' and
                fact['bundle_hash'] in plan['bundle_hashes'] and
                any(all(fact[k] == c[k] for k in ('source', 'task', 'checkpoint_id', 'seed'))
                    for c in plan['cases']) for (raw,) in plans for plan in [json.loads(raw)])
        if role is None or (role[0] != fact['role'] and not paired_replay):
            raise ValueError('Register source role before execution')
        self.bundle(fact['bundle_hash'])
        required = {'task', 'source', 'role', 'bundle_hash', 'bank_hash', 'seed', 'sample_hash',
                    'reward', 'decision_id', 'skill_execution', 'action'}
        if not required <= set(fact):
            raise ValueError('Incomplete skill fact')
        ident = stable_hash(fact)
        with self.transaction():
            self.db.execute('INSERT OR IGNORE INTO facts VALUES (?,?,?,?,?)',
                (ident, fact['source'], fact['role'], fact['bundle_hash'], encoded(fact)))
            identified = {**fact, 'fact_id': ident}
            trajectory = self.compact(identified)
            trajectory['_history_group'] = self._history_group(identified)
            trajectory.update({k: fact[k] for k in ('decision_id', 'decision_protocol_hash', 'experiment_id',
                'run_id', 'trajectory_id', 'checkpoint_id', 'round_id', 'node_id', 'parent_id',
                'lineage_id', 'candidate_pool_hash') if k in fact})
            self.db.execute('INSERT OR IGNORE INTO skill_trajectories VALUES (?,?)', (ident, encoded(trajectory)))
        return ident

    def fact(self, ident):
        row = self.db.execute('SELECT payload FROM facts WHERE id=?', (ident,)).fetchone()
        if row is None:
            raise ValueError('Unknown fact')
        return {'fact_id': ident, **json.loads(row[0])}

    def attach_historical_context(self, ident, path):
        """Backfill a legacy decision from its frozen inputs without changing its fact ID."""
        self.fact(ident)
        with self.transaction():
            self.db.execute('INSERT OR IGNORE INTO historical_context VALUES (?,?)', (ident, encoded(path)))
            if self.db.execute('SELECT payload FROM historical_context WHERE fact_id=?',
                               (ident,)).fetchone()[0] != encoded(path):
                raise ValueError('Historical sampling context differs from the archived decision')

    def history_observation(self, fact):
        if 'sampling_path' in fact:
            return fact
        row = self.db.execute('SELECT payload FROM historical_context WHERE fact_id=?',
                              (fact.get('fact_id'),)).fetchone()
        return {**fact, 'sampling_path': json.loads(row[0])} if row else fact

    def development_facts(self):
        return [self.fact(row[0]) for row in self.db.execute(
            'SELECT id FROM facts WHERE role="development" ORDER BY rowid')]

    @staticmethod
    def compact(fact):
        # Full facts remain immutable in SQLite. API history contains bounded summaries.
        def action(a):
            return {k: a[k] for k in ('action_id', 'type', 'region_id', 'action_ids',
                'actual_region_increase_fraction', 'aggregation') if k in a}
        def reward(r):
            result = {k: r[k] for k in ('reward', 'status', 'update_eligible', 'champion_miss',
                'champion_retained', 'order_accuracy', 'retained_ids', 'evidence_scope',
                'full_reference_complete', 'observed_method_ids', 'expected_method_ids',
                'predicted_champion_id', 'baseline_predicted_champion_id', 'nomination_source',
                'reward_basis', 'parent_child_comparisons', 'composite_gain', 'order_gain', 'paired_method_ids', 'baseline_order_accuracy', 'absolute_score', 'baseline_absolute_score',
                'practical_tolerance', 'comparable_pair_count', 'sample_weight', 'severe_error_rate') if k in r}
            result['reversed_method_pairs'] = [p['method_ids'] for p in r.get('pairs', []) if p.get('reversed')]
            return result
        result = {k: fact[k] for k in ('fact_id', 'task', 'source', 'bank_hash', 'seed', 'bundle_hash',
            'baseline_kind', 'branch_order_gain', 'paired_baseline', 'branch_count', 'branch_reward_delta', 'combination_gain',
            'sampling_policy_hash', 'search_stage', 'round_number', 'search_features',
            'model_search_gain', 'model_search_gain_direction') if k in fact}
        if 'validation_actions' in fact:
            result['validation_actions'] = fact['validation_actions']
        if 'sampling_path' in fact:
            result['sampling_path'] = fact['sampling_path']
        result['evidence_time'] = 'completed_historical_experiment_only'
        result['action'] = action(fact.get('action', {}))
        result['reward'] = reward(fact.get('reward', {}))
        if 'combined_reward' in fact:
            result['combined_reward'] = reward(fact['combined_reward'])
        result['skill_execution'] = {k: fact['skill_execution'][k] for k in
            ('decision_status', 'skills_used', 'skill_bundle_hash') if k in fact.get('skill_execution', {})}
        result['skill_execution']['reason_excerpt'] = fact.get('skill_execution', {}).get('reason', '')[:240]
        selected_id = result['action'].get('action_id')
        citations = [c.get('field') for c in fact.get('skill_execution', {}).get('checks', [])
                     if c.get('action_id') == selected_id and c.get('field')]
        if citations:
            result['skill_execution']['evidence_fields'] = list(dict.fromkeys(citations))[:3]
        result['decision_context'] = {k: fact.get('decision_context', {}).get(k) for k in
            ('coverage_gap', 'q_fraction', 'c0_fraction', 'mean_density', 'mean_hardness',
             'mean_disagreement', 'discover_score', 'check_score', 'direction_consistent')}
        # Keep the relevant seed's strongest cross-region counterexamples, not all tables.
        preview = fact.get('cross_region_preview') or {}
        regions = preview.get(str(fact.get('seed')), {}).get('regions', {})
        losses = [dict(region_id=rid, before=r['before'], after=r['after'],
                       loss_fraction=(r['before']-r['after'])/r['before'])
            for rid, r in regions.items() if r.get('before', 0) > 0 and r.get('after', 0) < r['before']]
        result['cross_region_losses'] = sorted(losses, key=lambda r: (-r['loss_fraction'], r['region_id']))[:3]
        result['uncertainty'] = {k: fact.get('uncertainty', {})[k] for k in
            ('reasons', 'relative_boundary_gap', 'run_alternative') if k in fact.get('uncertainty', {})}
        if 'branches' in fact:
            result['branches'] = [dict(action=action(b['action']), scores=b['scores'],
                                      reward=reward(b['reward'])) for b in fact['branches']]
        if fact.get('uncertainty', {}).get('purpose') == 'exploration_control':
            result['sampling_purpose'] = 'bounded_development_exploration; secondary is not an API choice and does not affect final ranking'
        return result

    @staticmethod
    def _history_group(fact, index=None):
        identity_fields = ('source', 'task', 'bank_hash', 'bundle_hash', 'decision_id',
            'sampling_policy_hash', 'decision_protocol_hash', 'experiment_id', 'run_id',
            'trajectory_id', 'checkpoint_id', 'round_id', 'node_id', 'parent_id',
            'lineage_id', 'candidate_pool_hash', 'decision_context', 'skill_execution')
        identity = {k: fact.get(k) for k in identity_fields}
        identity['reward_protocol_hash'] = fact.get('reward', {}).get('protocol_hash')
        if not fact.get('decision_id'):
            identity['unshared_fact'] = fact.get('fact_id', index)
        return stable_hash(identity)

    @staticmethod
    def _history_groups(facts):
        """Group sampling replicates, never different decisions or search checkpoints.

        A decision ID is mandatory for sharing evidence. Extra identity fields defend
        against callers reusing a decision ID across protocols or search trajectories.
        Duplicate seeds are left as individual cards rather than double-weighted.
        """
        groups = {}
        for index, fact in enumerate(facts):
            key = fact.get('_history_group') or SkillStore._history_group(fact, index)
            groups.setdefault(key, []).append((index, fact))
        result = []
        for observations in groups.values():
            counts = defaultdict(int)
            for _, fact in observations:
                counts[encoded(fact.get('seed'))] += 1
            unique = []
            for index, fact in observations:
                if counts[encoded(fact.get('seed'))] == 1:
                    unique.append((index, fact))
                else:
                    result.append((index, [fact]))
            if unique:
                result.append((unique[-1][0], [fact for _, fact in unique]))
        return [facts for _, facts in sorted(result, key=lambda pair: pair[0])]

    @classmethod
    def _history_card(cls, facts, compact=None):
        """Share repeated context while retaining each seed's observed failures."""
        if compact is None:
            compact = [cls.compact(fact) for fact in facts]
        if len(facts) == 1:
            return {**compact[0], 'history_format': 'single_observation_v1',
                    'source_count': 1, 'seed_count': 1}
        first = compact[0]
        common = ('task', 'source', 'bank_hash', 'bundle_hash', 'sampling_policy_hash',
                  'evidence_time', 'skill_execution', 'decision_context', 'sampling_path')
        card = {k: first[k] for k in common if k in first}
        card.update(history_format='sampling_seed_group_v1', source_count=1,
                    seed_count=len(facts), decision_id=facts[0]['decision_id'],
                    evidence_unit='one decision on one source; seeds are correlated replicates')
        # Preserve checkpoint identity when this path is used by an MCTS caller.
        card.update({k: facts[0][k] for k in ('decision_protocol_hash', 'experiment_id',
            'run_id', 'trajectory_id', 'checkpoint_id', 'round_id', 'node_id',
            'parent_id', 'lineage_id', 'candidate_pool_hash') if k in facts[0]})

        def summary(values):
            observed = [v for v in values if type(v) in (int, float) and np.isfinite(v)]
            return dict(observed=len(observed), missing=len(values)-len(observed),
                        mean=sum(observed)/len(observed) if observed else None,
                        min=min(observed) if observed else None,
                        max=max(observed) if observed else None)

        card['summary'] = {
            'reward': summary([f['reward'].get('reward') for f in compact]),
            'champion_miss': summary([f['reward'].get('champion_miss') for f in compact]),
            'order_accuracy': summary([f['reward'].get('order_accuracy') for f in compact]),
            'baseline_delta': summary([f.get('paired_baseline', {}).get('delta') for f in compact]),
            'branch_reward_delta': summary([f.get('branch_reward_delta') for f in compact]),
            'combination_gain': summary([f.get('combination_gain') for f in compact]),
        }
        # Factor only truly identical reward metadata. A partial reference must
        # remain visibly partial even if another seed has a complete reference.
        reward_context = {}
        rewards = [f['reward'] for f in compact]
        rewards.extend(branch['reward'] for f in compact for branch in f.get('branches', []))
        for key in ('status', 'update_eligible', 'evidence_scope', 'full_reference_complete',
                    'observed_method_ids', 'expected_method_ids'):
            if key in first['reward'] and all(r.get(key) == first['reward'][key]
                                             and key in r for r in rewards):
                reward_context[key] = first['reward'][key]
        card['reward_context'] = reward_context
        shared_uncertainty = (first.get('uncertainty') if all(f.get('uncertainty') ==
                              first.get('uncertainty') for f in compact) else None)
        if shared_uncertainty is not None:
            card['uncertainty'] = shared_uncertainty
        shared_purpose = (first.get('sampling_purpose') if all(f.get('sampling_purpose') ==
                          first.get('sampling_purpose') for f in compact) else None)
        if shared_purpose is not None:
            card['sampling_purpose'] = shared_purpose
        actions, action_indices = [], {}

        def action_index(action):
            key = encoded(action)
            if key not in action_indices:
                action_indices[key] = len(actions)
                actions.append(action)
            return action_indices[key]

        outcomes = []
        for fact in compact:
            outcome = {k: fact[k] for k in ('fact_id', 'seed', 'paired_baseline', 'branch_count',
                'baseline_kind', 'branch_reward_delta', 'branch_order_gain', 'combined_reward', 'combination_gain', 'cross_region_losses', 'uncertainty',
                'sampling_purpose') if k in fact}
            if shared_uncertainty is not None:
                outcome.pop('uncertainty', None)
            if shared_purpose is not None:
                outcome.pop('sampling_purpose', None)
            outcome['action_index'] = action_index(fact['action'])
            outcome['reward'] = {k: v for k, v in fact['reward'].items() if k not in reward_context}
            if 'branches' in fact:
                outcome['branches'] = [dict(action_index=action_index(branch['action']),
                    scores=branch['scores'], reward={k: v for k, v in branch['reward'].items()
                        if k not in reward_context or v != reward_context[k]}) for branch in fact['branches']]
            outcomes.append(outcome)
        card['actions'] = actions
        card['reward_context_applies_to'] = 'all seed and branch rewards unless explicitly overridden'
        card['seed_outcomes'] = outcomes
        # Explicitly point the API to adverse replicates; averaging is descriptive,
        # not permission to discard a miss, a reversal, or a harmful combination.
        card['counterexample_seeds'] = [f['seed'] for f in compact if
            f['reward'].get('champion_miss') == 1
            or f['reward'].get('reversed_method_pairs')
            or any(type(value) in (int, float) and value < 0 for value in
                (f['reward'].get('reward'), f.get('paired_baseline', {}).get('delta'), f.get('branch_reward_delta'),
                 f.get('branch_order_gain'), f['reward'].get('order_gain'), f.get('combination_gain')))
            or any(branch['reward'].get('champion_miss') == 1
                   or branch['reward'].get('reversed_method_pairs') for branch in f.get('branches', []))]
        return card

    def record_proposal_failure(self, payload):
        ident = stable_hash(payload)
        with self.transaction():
            self.db.execute('INSERT OR IGNORE INTO proposal_failures VALUES (?,?)', (ident, encoded(payload)))
        return ident

    def decision_history(self, source, bank_hash=None, *, allow_local=False, limit=8):
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError('History limit must be a nonnegative integer')
        if limit == 0:
            return []
        facts = []
        for ident, cached, legacy, backfill in self.db.execute("""
                SELECT f.id, t.payload, CASE WHEN t.fact_id IS NULL THEN f.payload END, h.payload
                FROM facts f LEFT JOIN skill_trajectories t ON t.fact_id=f.id
                LEFT JOIN historical_context h ON h.fact_id=f.id
                WHERE f.role='development' ORDER BY f.rowid"""):
            if cached:
                row = json.loads(cached)
            else:
                raw = dict(json.loads(legacy), fact_id=ident)
                row = {**raw, '_legacy_fact': True}
            if backfill and 'sampling_path' not in row:
                row['sampling_path'] = json.loads(backfill)
            facts.append(row)
        # Local evidence requires an exact fixed-bank identity and completed facts.
        local = self._history_groups([f for f in facts if allow_local and f['source'] == source
                                     and bank_hash is not None and f['bank_hash'] == bank_hash])
        remote = self._history_groups([f for f in facts if f['source'] != source])
        def recent(groups, budget):
            # Keep the existing raw-observation budget: changing 16 facts into
            # 16 three-seed cards would silently make the prompt much larger.
            selected = []
            for group in reversed(groups):
                if len(group) > budget:
                    if not selected and budget:
                        selected.append(group[-budget:])
                    break
                selected.append(group)
                budget -= len(group)
            return list(reversed(selected))
        if limit is None:
            limit = len(facts)
        local_limit = max(1, limit//2) if allow_local else 0
        local = recent(local, local_limit)
        remote_limit = limit-sum(len(group) for group in local)
        selected = recent(remote, remote_limit) + local
        cards = []
        for group in selected:
            rows = [self.compact(f) if f.get('_legacy_fact') else
                    {k: v for k, v in f.items() if k != '_history_group'} for f in group]
            cards.append({**self._history_card(group, rows),
                          'scope': 'task_local' if group[0]['source'] == source else 'cross_task'})
        return cards

    def optimization_context(self, min_new_sources=5):
        if type(min_new_sources) is not int or min_new_sources < 5:
            raise ValueError('At least five independent development sources with fresh decisions are required')
        consumed, legacy_sources = set(), set()
        edits = []
        for payload, status in self.db.execute('SELECT payload,status FROM edits ORDER BY rowid'):
            item = json.loads(payload)
            if 'development_window' in item:
                consumed.update(item['development_window']['decision_keys'])
            else:
                legacy_sources.update(item['new_sources'])
            edits.append(dict(edit_id=item['edit_id'], edit=item['edit'], status=status,
                              rejection_reason=item.get('rejection_reason')))
        failures = [json.loads(p) for p, in self.db.execute('SELECT payload FROM proposal_failures ORDER BY rowid')]
        for failure in failures:
            if failure.get('round_exhausted'):
                if 'development_window' in failure:
                    consumed.update(failure['development_window']['decision_keys'])
                else:
                    legacy_sources.update(failure['new_sources'])
        facts = [f for f in self.development_facts() if f['reward'].get('update_eligible')
                 and f['skill_execution'].get('decision_status') != 'fallback']
        if legacy_sources:
            # Old edits recorded source names, not their evidence boundary. Freeze a
            # conservative migration watermark once, before accepting future facts.
            # Historical outcomes cannot retroactively become new development work.
            key = 'legacy_development_window_v1'
            with self.transaction():
                saved = self.db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
                legacy = json.loads(saved[0]) if saved else dict(sources=[], decision_keys=[])
                unseen = legacy_sources-set(legacy['sources'])
                legacy['decision_keys'] = sorted(set(legacy['decision_keys']) | {
                    development_decision_key(f) for f in facts if f['source'] in unseen})
                legacy['sources'] = sorted(set(legacy['sources']) | legacy_sources)
                self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key, encoded(legacy)))
            consumed.update(legacy['decision_keys'])
        fresh_facts = [f for f in facts if development_decision_key(f) not in consumed]
        fresh = sorted({f['source'] for f in fresh_facts})
        if len(fresh) < min_new_sources:
            raise InsufficientDevelopmentEvidence(
                f'Need {min_new_sources} independent development sources with fresh decisions; have {len(fresh)}')
        chosen_sources = fresh[:min_new_sources]
        window_facts = [f for f in fresh_facts if f['source'] in chosen_sources]
        # Bound the prompt without allowing one long trajectory to evict another
        # source. The whole selected window is consumed, including omitted seeds.
        per_source = max(1, 30//len(chosen_sources))
        selected = [f for source in chosen_sources
                    for f in [x for x in window_facts if x['source'] == source][-per_source:]]
        window = dict(protocol='fresh-development-decisions-v1',
                      decision_keys=sorted({development_decision_key(f) for f in window_facts}),
                      fact_ids=[f['fact_id'] for f in window_facts], sources=chosen_sources)
        # Include prior negative paired observations, without exposing validation answers.
        negatives = [f for f in facts if any(v is not None and v < 0 for v in
            (f['reward'].get('reward'), f.get('paired_baseline', {}).get('delta'), f.get('branch_reward_delta'),
             f.get('branch_order_gain'), f['reward'].get('order_gain')))][-5:]
        selected = {f['fact_id']: f for f in [*selected, *negatives]}
        gates = [json.loads(row[0]) for row in self.db.execute('SELECT payload FROM gates ORDER BY rowid')]
        return dict(parent_hash=self.active, bundle=self.bundle(), new_sources=chosen_sources,
            development_window=window,
            trigger_interpretation='Distinct sources with newly completed decisions; sources may recur '
                'across proposal rounds. Seeds and branches of a consumed decision never refresh it.',
            facts=[self.compact(f) for f in selected.values()], prior_edits=edits[-10:],
            rejected_proposals=[dict(error=f['error'][:1200], attempt=f['attempt'],
                round_exhausted=f['round_exhausted'],
                proposed_rule=str(f.get('response', {}).get('value', ''))[:1200]
                    if isinstance(f.get('response'), dict) else 'invalid schema') for f in failures[-6:]],
            gate_feedback=[{k: g[k] for k in ('candidate_hash', 'status', 'reason', 'mean_delta',
                'mean_miss_delta', 'bootstrap_lower', 'mean_branch_delta', 'mean_net_delta') if k in g}
                for g in gates[-10:]])

    def propose(self, edit, context):
        if not isinstance(edit, dict) or not isinstance(edit.get('skill_id'), str):
            raise ValueError('Proposal requires a string skill_id')
        edit = dict(edit)
        submitted_skill_id = edit.get('skill_id')
        # Accept only an exact alias for this frozen parent version, never arbitrary stripping.
        aliases = {s['id']+'@v'+str(s['version']): s['id'] for s in context['bundle']['skills']}
        edit['skill_id'] = aliases.get(submitted_skill_id, submitted_skill_id)
        candidate = apply_edit(context['bundle'], edit, [f['fact_id'] for f in context['facts']])
        # Evidence must be development evidence from this store, including at least two sources.
        cited = [self.fact(i) for i in edit['evidence_ids']]
        if any(f['role'] != 'development' for f in cited) or len({f['source'] for f in cited}) < 2:
            raise ValueError('Proposal needs multiple independent development sources')
        payload = dict(parent_hash=context['parent_hash'], edit=edit, new_sources=context['new_sources'],
                       candidate_hash=stable_hash(candidate), evidence_ids=edit['evidence_ids'])
        if 'development_window' in context:
            window = context['development_window']
            window_facts = [self.fact(i) for i in window['fact_ids']]
            if (window.get('protocol') != 'fresh-development-decisions-v1'
                    or sorted({f['source'] for f in window_facts}) != sorted(context['new_sources'])
                    or sorted(window['sources']) != sorted(context['new_sources'])
                    or sorted({development_decision_key(f) for f in window_facts}) != window['decision_keys']
                    or any(f['role'] != 'development' or not f['reward'].get('update_eligible')
                           or f['skill_execution'].get('decision_status') == 'fallback' for f in window_facts)):
                raise ValueError('Development window must identify completed eligible development decisions')
            payload['development_window'] = window
        ident = stable_hash(payload)
        payload['edit_id'] = ident
        with self.transaction():
            if self.active != context['parent_hash']:
                raise ValueError('Active skill changed while proposing')
            if 'development_window' in payload and not self.db.execute(
                    'SELECT id FROM edits WHERE id=?', (ident,)).fetchone():
                used = set()
                for old, in self.db.execute('SELECT payload FROM edits'):
                    used.update(json.loads(old).get('development_window', {}).get('decision_keys', []))
                for old, in self.db.execute('SELECT payload FROM proposal_failures'):
                    failure = json.loads(old)
                    if failure.get('round_exhausted'):
                        used.update(failure.get('development_window', {}).get('decision_keys', []))
                if used.intersection(payload['development_window']['decision_keys']):
                    raise ValueError('Development window already consumed by another proposal round')
            pending = self.db.execute('SELECT id FROM edits WHERE status IN ("candidate","pending")').fetchone()
            if pending and pending[0] != ident:
                raise ValueError('Resolve the existing candidate before proposing another')
            self._bundle(candidate)
            self.db.execute('INSERT OR IGNORE INTO edits VALUES (?,?,?,?,?)',
                (ident, context['parent_hash'], payload['candidate_hash'], 'candidate', encoded(payload)))
        return payload

    def candidate(self, ident):
        row = self.db.execute('SELECT payload,status FROM edits WHERE id=?', (ident,)).fetchone()
        if row is None:
            raise ValueError('Unknown candidate edit')
        return {**json.loads(row[0]), 'status': row[1]}

    def bind_gate(self, edit_id, settings=None):
        """Freeze acceptance criteria before validation; pending retries cannot relax them."""
        self.candidate(edit_id)
        cfg = gate_settings(settings)
        key = 'gate:'+edit_id
        with self.transaction():
            old = self.db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
            if old and json.loads(old[0]) != cfg:
                raise ValueError('Acceptance criteria already frozen for this candidate')
            self.db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', (key, encoded(cfg)))
        return cfg

    def gate(self, edit_id, pairs, *, settings=None):
        """Promote only paired validation FACTS; source means weight every task equally."""
        cfg = self.bind_gate(edit_id, settings)
        candidate = self.candidate(edit_id)
        task_rows, seen = defaultdict(list), set()
        changed = 0
        new_misses = repaired_misses = 0
        protocols = set()
        incomplete = False
        for old_id, new_id in pairs:
            old, new = self.fact(old_id), self.fact(new_id)
            if (old['role'] != 'validation' or new['role'] != 'validation'
                    or old['bundle_hash'] != candidate['parent_hash'] or new['bundle_hash'] != candidate['candidate_hash']):
                raise ValueError('Gate requires old/new skill validation facts')
            for key in ('task', 'source', 'bank_hash', 'seed'):
                if old[key] != new[key]:
                    raise ValueError('Unpaired validation: '+key)
            if old['reward']['protocol_hash'] != new['reward']['protocol_hash']:
                raise ValueError('Reward protocol differs between arms')
            protocols.add(old['reward']['protocol_hash'])
            if len(protocols) > 1:
                raise ValueError('Validation tasks mix different reward protocols')
            if old.get('decision_protocol_hash') != new.get('decision_protocol_hash'):
                raise ValueError('Decision context/model/settings differ between arms')
            if old.get('sampling_policy_hash') != new.get('sampling_policy_hash'):
                raise ValueError('Sampling trigger/aggregation policy differs between arms')
            counts = [f.get('branch_count', 1) for f in (old, new)]
            if any(type(n) is not int or n not in (1, 2) for n in counts):
                raise ValueError('Validation requires one or two branches per policy')
            if old['sample_hash'] == new['sample_hash'] and old['reward'] != new['reward']:
                raise ValueError('Identical sample and bank have inconsistent reward evidence')
            key = (old['source'], old['task'], old['seed'])
            if key in seen:
                raise ValueError('Duplicate validation pair')
            seen.add(key)
            changed += old['sample_hash'] != new['sample_hash']
            if (old['reward']['reward'] is None or new['reward']['reward'] is None
                    or old['skill_execution']['decision_status'] == 'fallback'
                    or new['skill_execution']['decision_status'] == 'fallback'):
                incomplete = True
                continue
            delta = new['reward']['reward']-old['reward']['reward']
            branch_delta = counts[1]-counts[0]
            task_rows[(old['source'], old['task'])].append((delta,
                new['reward']['champion_miss']-old['reward']['champion_miss'], branch_delta,
                delta-cfg['extra_branch_penalty']*branch_delta, *counts))
            new_misses += new['reward']['champion_miss'] > old['reward']['champion_miss']
            repaired_misses += new['reward']['champion_miss'] < old['reward']['champion_miss']
        sources = defaultdict(list)
        for (source, _), rows in task_rows.items():
            sources[source].append(np.mean(rows, axis=0))
        means = np.asarray([np.mean(rows, axis=0) for rows in sources.values()])
        result = dict(candidate_hash=candidate['candidate_hash'], parent_hash=candidate['parent_hash'],
            edit_id=edit_id, settings=cfg, pairs=pairs, source_count=len(sources), changed_samples=changed,
            status='pending', reason='insufficient_evidence', mean_delta=None, mean_miss_delta=None,
            bootstrap_lower=None, mean_branch_delta=None, mean_net_delta=None,
            mean_old_branches=None, mean_new_branches=None,
            bootstrap_target='cost_adjusted_reward_delta',
            newly_missed_task_seeds=new_misses, repaired_task_seeds=repaired_misses,
            interpretation='Engineering selection gate, not a significance or generalization guarantee')
        if len(means):
            result.update(mean_delta=float(means[:, 0].mean()), mean_miss_delta=float(means[:, 1].mean()),
                mean_branch_delta=float(means[:, 2].mean()), mean_net_delta=float(means[:, 3].mean()),
                mean_old_branches=float(means[:, 4].mean()), mean_new_branches=float(means[:, 5].mean()))
            draws = np.random.default_rng(cfg['bootstrap_seed']).choice(means[:, 3],
                size=(cfg['bootstrap_repeats'], len(means)), replace=True).mean(axis=1)
            result['bootstrap_lower'] = float(np.quantile(draws, .05))
        if incomplete:
            result['reason'] = 'incomplete_or_fallback_evaluation'
        elif not changed:
            result['reason'] = 'no_action_quality_change'
        elif len(means) >= cfg['min_sources']:
            if result['mean_miss_delta'] > 1e-12 or result['mean_delta'] < -1e-12:
                result.update(status='rejected', reason='retention_or_mean_reward_regression')
            elif (result['mean_delta'] > cfg['min_delta'] and result['mean_net_delta'] > cfg['min_delta']
                  and result['bootstrap_lower'] > cfg['min_bootstrap_lower']):
                result.update(status='accepted', reason='paired_improvement_and_no_retention_regression')
            elif result['mean_delta'] > cfg['min_delta'] and result['mean_net_delta'] <= cfg['min_delta']:
                result['reason'] = 'effect_not_worth_extra_branch_cost'
            else:
                result['reason'] = 'effect_too_small_or_unstable'
        gate_id = stable_hash(result)
        with self.transaction():
            existing = self.db.execute('SELECT payload FROM gates WHERE id=?', (gate_id,)).fetchone()
            if existing:
                return json.loads(existing[0])
            if candidate['status'] not in ('candidate', 'pending') or self.active != candidate['parent_hash']:
                raise ValueError('Candidate already resolved or parent no longer active')
            self.db.execute('INSERT INTO gates VALUES (?,?)', (gate_id, encoded(result)))
            self.db.execute('UPDATE edits SET status=? WHERE id=?', (result['status'], edit_id))
            if result['status'] == 'accepted':
                self.db.execute('UPDATE meta SET value=? WHERE key="active"', (candidate['candidate_hash'],))
        return result

    def close_inconclusive(self, edit_id, gate, *, expected_cases):
        """Archive a complete but inconclusive gate after its declared one-pass budget.

        Preserve the original pending gate; rejection here means not promoted,
        not evidence of harm. Idempotent across a crash before campaign logging.
        """
        gate_id = stable_hash(gate)
        with self.transaction():
            saved = self.db.execute('SELECT payload FROM gates WHERE id=?', (gate_id,)).fetchone()
            if not saved or json.loads(saved[0]) != gate or gate.get('edit_id') != edit_id:
                raise ValueError('Resolution requires the recorded gate for this candidate')
            if (gate.get('status') != 'pending' or gate.get('reason') not in (
                    'no_action_quality_change', 'effect_not_worth_extra_branch_cost',
                    'effect_too_small_or_unstable') or
                    gate['source_count'] < gate['settings']['min_sources']):
                raise ValueError('Only complete inconclusive validation can be archived')
            expected = set(expected_cases)
            actual = []
            for old_id, _ in gate['pairs']:
                fact = self.fact(old_id)
                actual.append((fact['source'], fact['task'], fact['seed']))
            if not expected or len(actual) != len(expected) or set(actual) != expected:
                raise ValueError('Resolution requires the complete predeclared validation matrix')
            candidate = self.candidate(edit_id)
            previous = candidate.get('pending_resolution')
            if previous:
                if previous['gate_id'] != gate_id or candidate['status'] != 'rejected':
                    raise ValueError('Candidate resolution differs from the recorded gate')
                return previous
            if candidate['status'] != 'pending' or self.active != candidate['parent_hash']:
                raise ValueError('Candidate is not pending on the active parent')
            reason = 'validation_budget_exhausted:'+gate['reason']
            resolution = dict(edit_id=edit_id, gate_id=gate_id, status='rejected',
                reason=reason, gate_status='pending', retained_bundle=self.active,
                policy='one_complete_validation_then_archive',
                interpretation='Not promoted because evidence is insufficient; not proof of harm')
            payload = {k: v for k, v in candidate.items() if k != 'status'}
            payload.update(rejection_reason=reason, pending_resolution=resolution)
            self.db.execute('UPDATE edits SET status="rejected",payload=? WHERE id=?',
                            (encoded(payload), edit_id))
            return resolution

    def reject_pending(self, edit_id, reason):
        """Explicit budget/no-effect rejection retains the entire attempted patch."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Rejection reason required')
        with self.transaction():
            candidate = self.candidate(edit_id)
            if candidate['status'] not in ('candidate', 'pending'):
                raise ValueError('Candidate already resolved')
            payload = {k: v for k, v in candidate.items() if k != 'status'}
            payload['rejection_reason'] = reason
            self.db.execute('UPDATE edits SET status="rejected",payload=? WHERE id=?', (encoded(payload), edit_id))
