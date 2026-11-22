"""Deterministic MCTS policies; no fitting, file access or language-model calls.

Low-cost scores are used only for shortlisting and unobserved-node priors. Only
full-training scores enter backed-up rewards. Statistical MCS membership is
preserved separately from policy backfills and truncation. Bootstrap results
describe V_core sampling uncertainty, not probability of full-training success.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import copy
import math
from numbers import Real
from typing import Any

import numpy as np

from .stage1_metrics import metric_direction, primary_score, requires_stratified_bootstrap


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _sign(direction):
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be maximize or minimize")
    return 1 if direction == "maximize" else -1


@dataclass(frozen=True)
class MCTSConfig:
    enabled: bool = True
    mcs_enabled: bool = True
    initial_candidates: int = 5
    initial_keep: int = 3
    expansion_candidates: int = 3
    shortlist_cap: int = 2
    max_expansions: int = 6
    # Reserve primaries only through this many regular expansions. Remaining
    # max_expansions are fallback opportunities when evidence extras were unused.
    # None preserves the historical reservation policy and checkpoint identity.
    base_expansions: int | None = None
    max_search_attempts: int = 12
    # Deprecated compatibility field: attempts are recorded, never quota-limited.
    max_rollout_attempts: int = 12
    max_rollout_successes: int = 12
    max_low_fit_attempts: int = 60
    max_api_calls: int = 60
    max_wall_seconds: float = 21600.0
    repair_attempts: int = 2
    # Same frozen method retry after a host/worker failure; never an LLM repair.
    infrastructure_retries: int = 1
    max_children_per_node: int = 3
    max_depth: int = 6
    exploration_constant: float = math.sqrt(2)
    minimum_full_improvement: float = 0.0
    backup_win_threshold: float = 0.25
    second_rollout_every: int = 4
    bootstrap_replicates: int = 200
    bootstrap_seed: int = 2026
    mcs_alpha: float = 0.05
    # Default: 3 initial + 6 primary + at most 3 extra successes. One extra
    # is an independent audit; the others use uncertainty/conflict evidence.
    # The old reserved/adaptive schedules remain explicitly selectable.
    rollout_schedule: str = "evidence_reserved"
    exploration_rounds: tuple = (3, 6)
    calibration_rounds: tuple = (2, 4, 6)
    independent_audit_round: int = 2
    staged_search: bool = False
    root_trials: bool = False

    def __post_init__(self):
        if any(type(flag) is not bool for flag in (self.enabled, self.mcs_enabled, self.staged_search, self.root_trials)):
            raise ValueError('enabled, mcs_enabled and staged_search must be boolean')
        if self.rollout_schedule not in {"evidence_reserved", "reserved", "adaptive"}:
            raise ValueError("rollout_schedule must be evidence_reserved, reserved or adaptive")
        if type(self.independent_audit_round) is not int or self.independent_audit_round < 1:
            raise ValueError("independent_audit_round must be a positive integer")
        if self.root_trials and (not self.staged_search or self.rollout_schedule != 'evidence_reserved'
                                or self.independent_audit_round < 4):
            raise ValueError('Root trials require staged evidence_reserved search and audit round >= 4')
        for name in ("exploration_rounds", "calibration_rounds"):
            rounds = getattr(self, name)
            if not isinstance(rounds, (list, tuple)) or any(type(r) is not int or r < 1 for r in rounds):
                raise ValueError(f"{name} must contain positive round numbers")
            if len(set(rounds)) != len(rounds):
                raise ValueError(f"{name} must not contain duplicate rounds")
            object.__setattr__(self, name, tuple(sorted(rounds)))
        if type(self.expansion_candidates) is not int or self.expansion_candidates not in (2, 3):
            raise ValueError("expansion_candidates must be 2 or 3")
        if type(self.shortlist_cap) is not int or self.shortlist_cap not in (1, 2):
            raise ValueError("shortlist_cap must be 1 or 2")
        for name, exact in [("initial_candidates", 5), ("initial_keep", 3)]:
            if type(getattr(self, name)) is not int or getattr(self, name) != exact:
                raise ValueError(f"{name} must equal {exact} for the five/three/one policy")
        for name in ["max_expansions", "max_search_attempts", "max_api_calls", "max_depth"]:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.base_expansions is not None:
            if type(self.base_expansions) is not int or not 1 <= self.base_expansions <= self.max_expansions:
                raise ValueError('base_expansions must be between one and max_expansions')
            if self.rollout_schedule != 'evidence_reserved':
                raise ValueError('base_expansions requires the evidence_reserved schedule')
        if type(self.max_rollout_attempts) is not int or self.max_rollout_attempts < 1:
            raise ValueError("legacy max_rollout_attempts must be a positive integer")
        if type(self.max_rollout_successes) is not int or self.max_rollout_successes < self.initial_keep:
            raise ValueError("max_rollout_successes must allow all three initial rollouts")
        if not self.enabled:
            # Without search, evaluate the whole initial bank, irrespective of
            # the inherited expansion budget. There are no additional rollouts.
            object.__setattr__(self, 'max_rollout_successes', self.initial_candidates)
        if type(self.max_low_fit_attempts) is not int or self.max_low_fit_attempts < self.initial_candidates:
            raise ValueError("max_low_fit_attempts must cover five initial candidates")
        if self.max_search_attempts < self.max_expansions:
            raise ValueError("max_search_attempts must cover max_expansions")
        if type(self.max_children_per_node) is not int or self.max_children_per_node < 2:
            raise ValueError("max_children_per_node must be at least two to permit branching")
        for name in ["repair_attempts", "infrastructure_retries", "second_rollout_every", "bootstrap_seed"]:
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.repair_attempts > 2:
            raise ValueError("repair_attempts must be between zero and two")
        if type(self.bootstrap_replicates) is not int or self.bootstrap_replicates < 2:
            raise ValueError("bootstrap_replicates must be at least two")
        for name in ["exploration_constant", "minimum_full_improvement"]:
            if not _finite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not _finite(self.max_wall_seconds) or self.max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be finite and positive")
        if not _finite(self.backup_win_threshold) or not 0 <= self.backup_win_threshold <= 1:
            raise ValueError("backup_win_threshold must be in [0, 1]")
        if not _finite(self.mcs_alpha) or not 0 < self.mcs_alpha < 1:
            raise ValueError("mcs_alpha must be in (0, 1)")

    @property
    def initial_full_count(self):
        return self.initial_keep if self.enabled else self.initial_candidates

    @property
    def full_train_methods(self):
        return 'top3' if self.enabled else 'all'

    @classmethod
    def from_dict(cls, value=None):
        if isinstance(value, cls):
            return value
        if value is not None and not isinstance(value, dict):
            raise ValueError("MCTS configuration must be an object")
        unknown = set(value or {}) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown MCTS settings: {sorted(unknown)}")
        return cls(**(value or {}))

    def to_dict(self):
        result = asdict(self)
        if result['base_expansions'] is None:
            result.pop('base_expansions')
        if not self.root_trials:
            result.pop('root_trials')
        if not self.staged_search:
            result.pop('staged_search')  # preserve manifests for existing searches
        # Manifests compare their decoded JSON on resume. Tuples serialize as
        # arrays but do not compare equal to decoded lists before serialization.
        for key in ('exploration_rounds', 'calibration_rounds'):
            result[key] = list(result[key])
        return result


def shortlist_candidates(candidates, mcs_result, direction, limit=2, minimum=1):
    """Rank MCS survivors by low score, then cap or backfill as explicit policy.

    Initial admission uses limit=minimum=3. Expansion uses limit=2, minimum=1.
    Missing/degenerate MCS evidence falls back to all finite scored candidates;
    it never fabricates p-values. Invalid or failed candidates are not backfilled.
    """
    sign = _sign(direction)
    if type(limit) is not int or type(minimum) is not int or not 1 <= minimum <= limit:
        raise ValueError("Require 1 <= minimum <= limit")
    ids = [c["id"] for c in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be unique")
    valid = [c for c in candidates if _finite(c.get("low_score"))
             and c.get("status") not in {"failed", "invalid"}]
    order = [c["id"] for c in sorted(valid, key=lambda c: (-sign * c["low_score"], str(c["id"])))]
    result = mcs_result or {}
    survivors = list(result.get("survivor_method_ids", []))
    if not set(survivors) <= set(ids):
        raise ValueError("MCS contains unknown candidate IDs")
    reliable = result.get("selection_status") == "completed" and bool(survivors)
    eligible = [i for i in order if i in survivors] if reliable else list(order)
    fallback = not reliable or not eligible
    if not eligible:
        eligible = list(order)
    selected = eligible[:limit]
    backfilled = []
    for candidate_id in order:
        if len(selected) >= minimum:
            break
        if candidate_id not in selected:
            selected.append(candidate_id)
            backfilled.append(candidate_id)
    selected.sort(key=order.index)
    return {
        "candidate_ids": selected, "mcs_survivor_ids": survivors,
        "backfilled_ids": backfilled, "truncated_ids": [i for i in eligible if i not in selected],
        "fallback_to_score_ranking": fallback,
        "source_selection_status": result.get("selection_status", "unavailable"),
        "complete": len(selected) >= minimum,
        "policy": "rank_survivors_then_cap_or_backfill; MCS membership is unchanged",
    }


def _prediction_arrays(record, n):
    predicted = np.asarray(record["predictions"])
    if predicted.shape != (n,):
        raise ValueError("Predictions must be aligned one-dimensional V_core rows")
    proba = record.get("probabilities")
    if proba is not None:
        proba = np.asarray(proba)
        if proba.ndim != 2 or proba.shape[0] != n:
            raise ValueError("Probabilities must be aligned with V_core rows")
    return predicted, proba, record.get("classes")


def paired_bootstrap_probability(task, y, primary, backup, n_bootstrap=200, seed=0):
    """Fraction of paired V_core resamples where backup strictly beats primary.

    Predictions must be in exactly the same row order as y. Task primary metric
    (including parameters and probability class order) is recomputed per draw.
    AUC uses class-stratified resampling so all original classes remain present.
    This is a search heuristic, not a posterior probability or significance test.
    """
    if type(n_bootstrap) is not int or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least two")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    actual = np.asarray(y)
    if actual.ndim != 1 or len(actual) < 2:
        raise ValueError("At least two aligned V_core labels required")
    pa, pp, pc = _prediction_arrays(primary, len(actual))
    ba, bp, bc = _prediction_arrays(backup, len(actual))
    direction = metric_direction(task)
    sign = _sign(direction)
    original = [primary_score(task, actual, p, prob, classes)
                for p, prob, classes in [(pa, pp, pc), (ba, bp, bc)]]
    strata = ([np.flatnonzero(actual == label) for label in np.unique(actual)]
              if requires_stratified_bootstrap(task) else None)
    rng = np.random.default_rng(seed)
    wins = ties = 0
    for _ in range(n_bootstrap):
        positions = (rng.integers(0, len(actual), size=len(actual)) if strata is None else
                     np.concatenate([rng.choice(group, size=len(group)) for group in strata]))
        first = primary_score(task, actual[positions], pa[positions],
                              None if pp is None else pp[positions], pc)
        second = primary_score(task, actual[positions], ba[positions],
                               None if bp is None else bp[positions], bc)
        delta = sign * (second - first)
        wins += delta > 0
        ties += delta == 0
    return {
        "backup_win_probability": wins / n_bootstrap,
        "tie_probability": ties / n_bootstrap,
        "primary_score": original[0], "backup_score": original[1],
        "replicates": n_bootstrap, "seed": seed, "direction": direction,
        "bootstrap_scheme": "stratified_by_true_class" if strata is not None else "iid_rows",
        "scope": "V_core sampling stability only; not full-rollout success probability",
    }


def should_rollout_second(parent_full_score, first_full_score, backup_win_probability,
                          expansion_index, config=None, *, direction="maximize",
                          budget_available=True, backup_available=True):
    """Decide after the first full fit; expansion_index is one-based.

    minimum_full_improvement is an absolute primary-metric tolerance. Missing full
    score means execution failed or no evidence, never a fabricated bad score.
    """
    cfg = MCTSConfig.from_dict(config)
    sign = _sign(direction)
    if type(expansion_index) is not int or expansion_index < 1:
        raise ValueError("expansion_index must be positive")
    if backup_win_probability is not None and (not _finite(backup_win_probability)
                                              or not 0 <= backup_win_probability <= 1):
        raise ValueError("backup_win_probability must be None or in [0, 1]")
    if not backup_available or not budget_available:
        return {"run_second": False, "reasons": ["no_backup" if not backup_available else "budget_exhausted"]}
    reasons = []
    if not _finite(first_full_score):
        reasons.append("first_rollout_failed_or_missing")
    elif not _finite(parent_full_score):
        reasons.append("parent_full_evidence_missing")
    elif sign * (first_full_score - parent_full_score) <= cfg.minimum_full_improvement:
        reasons.append("insufficient_improvement_over_parent")
    if backup_win_probability is None:
        reasons.append("bootstrap_evidence_unavailable")
    elif backup_win_probability >= cfg.backup_win_threshold:
        reasons.append("backup_competitive_on_paired_bootstrap")
    if (cfg.second_rollout_every and
            expansion_index % cfg.second_rollout_every == 0):
        reasons.append("periodic_second_candidate_exploration")
    return {"run_second": bool(reasons), "reasons": reasons or ["first_improved_and_low_ranking_stable"]}


class RolloutBudget:
    """Reserve successful-fit quota before execution, releasing it on failure.

    All attempts remain auditable. The legacy max_rollout_attempts setting is
    ignored; only completed successful fits consume quota. A valid full score
    counts as success even if it does not improve on the parent or global best.
    """

    def __init__(self, config=None):
        self.config = MCTSConfig.from_dict(config)
        self.attempts = {}
        self.successes = 0

    @property
    def available(self):
        return (self.successes + sum(v == "running" for v in self.attempts.values())
                < self.config.max_rollout_successes)

    def begin(self, attempt_id):
        if attempt_id in self.attempts:
            raise ValueError("Rollout attempt ID already exists")
        if not self.available:
            raise ValueError("Full rollout budget exhausted")
        self.attempts[attempt_id] = "running"

    def finish(self, attempt_id, success):
        if type(success) is not bool:
            raise ValueError("success must be boolean")
        if self.attempts.get(attempt_id) != "running":
            raise ValueError("Unknown or already finished rollout attempt")
        self.attempts[attempt_id] = "success" if success else "failed"
        self.successes += success

    def to_dict(self):
        return {"attempts": dict(self.attempts), "attempt_count": len(self.attempts),
                "successful_fits": self.successes, "available": self.available}


class SearchTree:
    """Retained nodes form a branching tree; all candidates stay in an archive.

    Root is virtual (None). Candidate schema: id, parent_id, branch_id,
    low_score, full_score. Extra fields survive. Initial candidates have no
    parent and their own ID as branch ID. Full evaluations of both alternatives
    contribute once to ancestor reward statistics even if only one is retained.
    ``search_visits`` counts expansion selections, never fitted models.
    """

    def __init__(self, direction, reward_reference=None, reward_scale=None):
        self.sign = _sign(direction)
        self.direction = direction
        if reward_reference is not None and not _finite(reward_reference):
            raise ValueError("reward_reference must be finite")
        if reward_scale is not None and (not _finite(reward_scale) or reward_scale <= 0):
            raise ValueError("reward_scale must be finite and positive")
        self.reward_reference = reward_reference
        self.reward_scale = reward_scale
        self.nodes: dict[Any, dict] = {}
        self.candidates: dict[Any, dict] = {}
        self.global_best_id = None
        self.root_visits = 0
        self.full_evaluations = {}

    def add_candidate(self, candidate, retain=True):
        record = dict(candidate)
        identity = record["id"]
        if identity is None or identity in self.candidates:
            raise ValueError("Candidate must have a new non-null ID")
        parent = record.get("parent_id")
        if parent is not None and parent not in self.nodes:
            raise ValueError("Parent must already be an admitted node")
        branch = self.nodes[parent]["branch_id"] if parent is not None else identity
        if record.get("branch_id", branch) != branch:
            raise ValueError("Candidate branch differs from parent branch")
        if record.get("low_score") is not None and not _finite(record["low_score"]):
            raise ValueError("low_score must be finite or None")
        supplied_full = record.get("full_score")
        if supplied_full is not None:
            raise ValueError("Register full results with register_rollout, not add_candidate")
        record.update(parent_id=parent, branch_id=branch, retained=False,
                      depth=self.nodes[parent]["depth"] + 1 if parent is not None else 1,
                      children=[], search_visits=0, expansion_count=0,
                      full_rollout_count=0, full_reward_sum=0.0, full_score=None)
        self.candidates[identity] = record
        if retain:
            self.retain_candidate(identity)
        return record

    def retain_candidate(self, candidate_id):
        record = self.candidates[candidate_id]
        if record["retained"]:
            return record
        record["retained"] = True
        self.nodes[candidate_id] = record
        if record["parent_id"] is not None:
            self.nodes[record["parent_id"]]["children"].append(candidate_id)
        return record

    def register_rollout(self, candidate_id, score, attempt_id):
        if not _finite(score):
            raise ValueError("Only finite successful full scores can produce rewards")
        if attempt_id in self.full_evaluations:
            previous = self.full_evaluations[attempt_id]
            if previous["candidate_id"] == candidate_id and previous["score"] == score:
                return False
            raise ValueError("Attempt ID reused for a different evaluation")
        record = self.candidates[candidate_id]
        if record["full_score"] is not None:
            raise ValueError("Same candidate/seed is already fully evaluated; create a distinct run")
        if self.reward_reference is None:
            self.reward_reference = float(score)
        if self.reward_scale is None:
            self.reward_scale = (abs(self.reward_reference) if self.direction == "minimize"
                                 and abs(self.reward_reference) > 1e-12 else 1.0)
        reward = self.sign * (score - self.reward_reference) / self.reward_scale
        record["full_score"] = float(score)
        self.full_evaluations[attempt_id] = {"candidate_id": candidate_id,
                                           "score": float(score), "reward": reward}
        current = record
        while current is not None:
            current["full_rollout_count"] += 1
            current["full_reward_sum"] += reward
            current = self.nodes[current["parent_id"]] if current["parent_id"] is not None else None
        if (self.global_best_id is None or
                self.sign * (score - self.candidates[self.global_best_id]["full_score"]) > 0):
            self.global_best_id = candidate_id
        return True

    def record_expansion(self, parent_id):
        """Call once when an expansion is attempted, including failed batches."""
        parent = self.nodes[parent_id]
        parent["expansion_count"] += 1
        current = parent
        while current is not None:
            current["search_visits"] += 1
            current = self.nodes[current["parent_id"]] if current["parent_id"] is not None else None
        self.root_visits += 1

    def search_stage(self, config, records, full_successes, *, frozen_model_classes=None):
        """Freeze the eligible parents for a new expansion using prior full evidence."""
        def rank(node):
            return (-self.sign * node['full_score'], node['id'] != self.global_best_id, str(node['id']))

        expandable = [n for n in self.nodes.values() if _finite(n.get('full_score'))
                      and n['depth'] < config.max_depth and len(n['children']) < config.max_children_per_node]
        stage = {'full_successes_at_start': full_successes}
        if full_successes < 6:
            roots = sorted((n for n in self.nodes.values()
                            if n['parent_id'] is None and _finite(n.get('full_score'))), key=rank)
            eligible = roots[:2]
            if len(roots) > 2:
                second, third = roots[1:3]
                thresholds = (records[n['id']].get('sampling_practical_tolerance') for n in (second, third))
                tolerance = max((v for v in thresholds if _finite(v) and v >= 0),
                                default=config.minimum_full_improvement)
                gap = self.sign * (second['full_score'] - third['full_score'])
                if gap <= tolerance or math.isclose(gap, tolerance, abs_tol=1e-12):
                    eligible.append(third)
                stage.update(third_place_gap=gap, practical_tolerance=tolerance)
            roots_allowed = [n['id'] for n in eligible]
            stage.update(phase='competitive_branches', eligible_root_ids=roots_allowed,
                         eligible_parent_ids=[n['id'] for n in expandable if n['branch_id'] in roots_allowed])
            return stage

        # The actual estimator may differ from its ancestor and several methods
        # can share a broad family such as bagging. Rank concrete fitted classes.
        def estimator(node):
            report = records[node['id']].get('full_report') or {}
            return (report.get('actual_budget') or {}).get('model_class')

        representatives = {}
        for node in sorted(expandable, key=rank):
            representatives.setdefault(estimator(node), node['id'])
        observed = sorted((n for n in self.candidates.values() if _finite(n.get('full_score'))), key=rank)
        ranked_classes = list(dict.fromkeys(estimator(n) for n in observed))
        if frozen_model_classes is None and None in ranked_classes:
            # Older/custom evaluation reports can lack the concrete estimator.
            # Keep their full evidence and avoid treating "bagging" as a model.
            stage.update(phase='model_identity_unavailable',
                         eligible_parent_ids=[n['id'] for n in expandable])
            return stage
        # Archive-only results still rank a model. Only retained nodes may be
        # parents. A saturated branch leaves fewer parents within the frozen
        # shortlist; it cannot promote a previously excluded model.
        allowed = ranked_classes[:2] if frozen_model_classes is None else list(frozen_model_classes)
        stage.update(phase='parameter_focus', allowed_model_classes=allowed,
                     eligible_parent_ids=[representatives[kind] for kind in allowed if kind in representatives])
        return stage

    def select_parent(self, config=None, *, selection_mode=None, eligible_parent_ids=None):
        """UCT descent compares re-expansion with exploring existing children.

        A parent can acquire multiple children, but a promising existing child
        may be extended before filling its parent's child cap. Re-expansion uses
        the node's backed-up full value, with expansion_count as its exploration
        denominator. Child exploration uses search_visits plus one when its own
        full score exists: the saved fit is evidence, not another search visit.
        No extra reward or fit count is created. Failed generation does not use
        child slots (global max_search_attempts bounds retry cost). Low scores
        provide only within-sibling rank priors when no full evidence exists.
        """
        cfg = MCTSConfig.from_dict(config)

        eligible = None if eligible_parent_ids is None else set(eligible_parent_ids)

        def expandable(node):
            return ((eligible is None or node["id"] in eligible) and node["depth"] < cfg.max_depth and
                    len(node["children"]) < cfg.max_children_per_node)

        def viable(node):
            return expandable(node) or any(viable(self.nodes[c]) for c in node["children"])

        if selection_mode not in {None, "exploit", "explore"}:
            raise ValueError("Unknown parent selection mode")
        if selection_mode == "exploit":
            choices = [n for n in self.nodes.values() if expandable(n) and _finite(n.get("full_score"))]
            if choices:
                # Prefer the current winner on ties, falling back to the best
                # expandable full-evaluated node when depth/child caps bind.
                return min(choices, key=lambda n: (-self.sign * n["full_score"],
                           n["id"] != self.global_best_id, str(n["id"])))['id']

        def choose(ids, visits):
            choices = [self.nodes[i] for i in ids if viable(self.nodes[i])]
            if not choices:
                return None, -math.inf
            ordered = sorted(choices, key=lambda n: (
                not _finite(n.get("low_score")),
                -self.sign * n["low_score"] if _finite(n.get("low_score")) else 0,
                str(n["id"])))
            prior = {n["id"]: 1 - rank / max(1, len(ordered) - 1) for rank, n in enumerate(ordered)}
            if len({n.get('coreset_binding',{}).get('version') for n in choices})>1:
                # Dynamic-core siblings from different expansion batches have
                # no common low-score ranking; use a neutral unmeasured prior.
                prior={n['id']:.5 for n in choices}

            def uct(n):
                value = (n["full_reward_sum"] / n["full_rollout_count"]
                         if n["full_rollout_count"] else prior[n["id"]])
                observations = n["search_visits"] + int(n["full_score"] is not None)
                if not observations:
                    return math.inf
                return value + cfg.exploration_constant * math.sqrt(
                    math.log(1 + visits) / observations)

            chosen = min(choices, key=lambda n: (-uct(n), str(n["id"])))
            return chosen, uct(chosen)

        roots = [n for n in self.nodes.values() if n["parent_id"] is None and viable(n)]
        untried_roots = [n for n in roots if n["expansion_count"] == 0 and expandable(n)]
        # Legacy schedules deliberately visited every initial root first.
        # The default exploration round uses actual UCT evidence immediately:
        # already full-evaluated roots are not unobserved arms.
        if untried_roots and not (selection_mode == "explore" and cfg.rollout_schedule == "evidence_reserved"):
            return min(untried_roots, key=lambda n: str(n["id"]))["id"]
        current, _ = choose([n["id"] for n in roots], self.root_visits)
        while current is not None:
            child, child_score = choose(current["children"], current["search_visits"])
            if expandable(current):
                if current["expansion_count"] == 0 or child is None:
                    return current["id"]
                expansion_value = (current["full_reward_sum"] / current["full_rollout_count"]
                                   if current["full_rollout_count"] else 0.0)
                expansion_score = expansion_value + cfg.exploration_constant * math.sqrt(
                    math.log(1 + current["search_visits"]) / current["expansion_count"])
                if expansion_score >= child_score:
                    return current["id"]
            current = child
        return None

    def to_dict(self):
        return copy.deepcopy({"direction": self.direction, "reward_reference": self.reward_reference,
                "reward_scale": self.reward_scale, "global_best_id": self.global_best_id,
                "root_visits": self.root_visits, "nodes": list(self.nodes.values()),
                "candidates": list(self.candidates.values()),
                "full_evaluations": dict(self.full_evaluations)})

    @classmethod
    def from_dict(cls, value):
        """Restore the same policy state, preserving deduplication and archives.

        Reward statistics are reconstructed from full evaluation evidence, not
        blindly copied. A partially corrupted checkpoint fails instead of
        silently doubling rollout counts or accepting changed full scores.
        """
        tree = cls(value["direction"], value.get("reward_reference"), value.get("reward_scale"))
        source_nodes = {r["id"]: r for r in value["nodes"]}
        if len(source_nodes) != len(value["nodes"]):
            raise ValueError("Duplicate nodes in checkpoint")
        records = copy.deepcopy(value.get("candidates", value["nodes"]))
        originals = {r["id"]: r for r in records}
        if len(originals) != len(records) or not set(source_nodes) <= set(originals):
            raise ValueError("Invalid candidate archive in checkpoint")
        pending = list(records)
        while pending:
            before = len(pending)
            for record in list(pending):
                if record.get("parent_id") is not None and record["parent_id"] not in tree.nodes:
                    continue
                retain = record["id"] in source_nodes
                if bool(record.get("retained", retain)) != retain:
                    raise ValueError("Checkpoint admitted-node membership mismatch")
                clean = {**record, "full_score": None}
                tree.add_candidate(clean, retain=retain)
                pending.remove(record)
            if len(pending) == before:
                raise ValueError("Checkpoint has unknown parents or a cycle")
        for attempt_id, evidence in value.get("full_evaluations", {}).items():
            tree.register_rollout(evidence["candidate_id"], evidence["score"], attempt_id)
            if not math.isclose(tree.full_evaluations[attempt_id]["reward"], evidence["reward"], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Checkpoint reward disagrees with frozen reference")
        if tree.global_best_id != value.get("global_best_id"):
            raise ValueError("Checkpoint global best disagrees with full results")
        for identity, original in originals.items():
            node = tree.candidates[identity]
            for field in ["full_score", "full_rollout_count", "full_reward_sum"]:
                if original.get(field, node[field]) != node[field]:
                    raise ValueError(f"Checkpoint {field} disagrees with full results")
            for field in ["search_visits", "expansion_count"]:
                count = original.get(field, 0)
                if type(count) is not int or count < 0:
                    raise ValueError(f"Checkpoint {field} must be a nonnegative integer")
                node[field] = count
            if original.get("children", node["children"]) != node["children"]:
                raise ValueError("Checkpoint child links disagree with admitted nodes")
        visits = value.get("root_visits", 0)
        if type(visits) is not int or visits < 0:
            raise ValueError("Checkpoint root_visits must be a nonnegative integer")
        tree.root_visits = visits
        return tree
