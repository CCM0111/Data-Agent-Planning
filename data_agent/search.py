"""FE selection and MT-reference UCB with bounded feedback expansion."""

from dataclasses import dataclass, field
import json
import math
import traceback
from pathlib import Path

from .compatibility import CompatibilityTracker
from .experience import save_records
from .fe_reuse import feedback_used, reuse_routes
from .traces import describe_error, path_errors, pipeline_context


STAGES = ("ROOT", "EDA", "FE", "MT")


@dataclass
class Node:
    id: int
    stage: str
    parent: "Node | None" = field(default=None, repr=False)
    children: list = field(default_factory=list, repr=False)
    idea: dict = field(default_factory=dict)
    method: str = ""
    code_path: str = ""
    result: dict = field(default_factory=dict)
    feedback: str = ""
    status: str = "pending"
    visits: int = 0
    total_reward: float = 0.0
    reward: float | None = None
    prior: float = 0.0
    prior_scores: dict = field(default_factory=dict)
    prior_reason: str = ""
    origin: str = "generate"
    generation_mode: str = "unknown"
    experience_source: dict | None = None
    based_on: int | None = None
    source_rollout: int | None = None
    feedback_node_ids: list = field(default_factory=list)
    context_node_ids: list = field(default_factory=list)
    reference_mt_id: int | None = None
    reference_rollout_id: int | None = None
    change_reason: str = ""
    code_hash: str = ""
    attempts: list = field(default_factory=list)
    comparison: dict = field(default_factory=dict)
    # These are direct feedback-expansion outcomes, not executions of this model.
    expansion_visits: int = 0
    expansion_total_reward: float = 0.0
    reference_selection: dict = field(default_factory=dict)
    compatibility_issue_ids: list = field(default_factory=list)
    repair_issue_id: str | None = None

    def ancestors(self):
        path, node = [], self.parent
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))


class Search:
    def __init__(self, executor, output, rollouts=8, branches=None, exploration=1.414, prior_weight=0.2,
                 mt_improvements_per_parent=0, fe_improvements=0, fe_review_interval=3,
                 cross_fe_retries=1):
        self.executor, self.output = executor, Path(output)
        self.rollouts, self.exploration = rollouts, exploration
        self.prior_weight = prior_weight
        self.mt_improvements_per_parent = mt_improvements_per_parent
        self.fe_improvements = fe_improvements
        self.fe_review_interval = fe_review_interval
        self.cross_fe_retries = cross_fe_retries
        self.fe_reuse_checks = []
        self.last_fe_review_at = 0
        self.fe_reviews = []
        self.compatibility = CompatibilityTracker()
        self.stop_reason = None
        self.history = []
        self.summarized_node_ids = set()
        self.active_node = None
        self.active_operation = None
        self.active_rollout = None
        if not isinstance(mt_improvements_per_parent, int) or mt_improvements_per_parent < 0:
            raise ValueError("mt_improvements_per_parent must be a nonnegative integer")
        if type(fe_improvements) is not int or fe_improvements < 0:
            raise ValueError("fe_improvements must be a nonnegative integer")
        if type(fe_review_interval) is not int or fe_review_interval < 1:
            raise ValueError("fe_review_interval must be a positive integer")
        if type(cross_fe_retries) is not int or cross_fe_retries < 0:
            raise ValueError("cross_fe_retries must be a nonnegative integer")
        self.branches = branches or {"EDA": 1, "FE": 3, "MT": 3}
        if self.branches["EDA"] != 1:
            raise ValueError("EDA is shared by all branches; set branches.EDA to 1")
        if not 0 <= prior_weight <= 1:
            raise ValueError("prior_weight must be between zero and one")
        if rollouts < 1 or any(self.branches[stage] < 1 for stage in STAGES[1:]):
            raise ValueError("Search and branch budgets must be positive")
        self.root = Node(0, "ROOT", status="success")
        self.nodes = [self.root]
        self.best = None

    def exhausted(self, node):
        # Completed MTs are exhausted for execution, but can still be references
        # via can_improve(FE). Never run/backpropagate their old result again.
        if node.status == "failed" or (node.stage == "MT" and node.status == "success"):
            return True
        if node.stage == "EDA" and self.fe_review_due(node):
            return False
        if self.can_improve(node):
            return False
        return bool(node.children) and all(self.exhausted(child) for child in node.children)

    def can_improve(self, parent):
        return (parent.stage == "FE" and parent.status == "success"
                and bool(parent.children)
                and all(child.status in {"success", "failed"} for child in parent.children)
                and feedback_used(parent) < self.mt_improvements_per_parent)

    def reuse_plan(self):
        return reuse_routes(self.nodes, self.compatibility.pending(), self.cross_fe_retries,
                            self.mt_improvements_per_parent)

    def resolve_existing_results(self, routes):
        for route in routes:
            if route["decision"] == "recovered":
                self.compatibility.reuse_success(route["reference_mt_id"],
                                                 self.nodes[route["recovery_mt_id"]], self.root.visits)

    async def maybe_reuse_fe(self):
        """Use one existing feedback slot and rollout, with no LLM generation."""
        if self.root.visits >= self.rollouts:
            return None
        routes = self.reuse_plan()
        if routes:
            self.fe_reuse_checks.append({"after_rollout": self.root.visits, "routes": routes})
        self.resolve_existing_results(routes)
        route = next((item for item in routes if item["decision"] == "retry"), None)
        if route is None:
            return None
        parent, reference = self.nodes[route["target_fe_id"]], self.nodes[route["reference_mt_id"]]
        child = Node(len(self.nodes), "MT", parent=parent, origin="cross_fe_replay",
                     based_on=reference.id, reference_mt_id=reference.id,
                     reference_rollout_id=self.rollout_for(reference.id)["rollout_id"],
                     source_rollout=self.root.visits, feedback_node_ids=[reference.id],
                     repair_issue_id=route["issue_id"], change_reason=route["reason"])
        parent.children.append(child)
        self.nodes.append(child)
        self.active_node, self.active_operation = child, "cross_fe_replay"
        try:
            await self.executor.replay_mt(child, reference)
        finally:
            self.save()
        return child

    def completed_models(self):
        return [n for n in self.nodes if n.stage == "MT" and n.visits > 0
                and n.status in {"success", "failed"} and n.parent.status == "success"]

    def fe_review_due(self, eda):
        """Check only at completed-rollout multiples, with budget for a repair."""
        if (eda.status != "success" or not eda.children
                or self.root.visits >= self.rollouts
                or self.root.visits <= self.last_fe_review_at
                or sum(n.origin == "fe_improve" for n in eda.children) >= self.fe_improvements):
            return False
        if any(n.origin == "fe_improve" and n.visits == 0 for n in eda.children):
            return False
        return self.root.visits % self.fe_review_interval == 0

    def rollout_for(self, node_id):
        return next(record for record in reversed(self.history)
                    if record["path"][-1]["node_id"] == node_id)

    async def maybe_fork_fe(self):
        if not self.root.children:
            return
        eda = self.root.children[0]
        if not self.fe_review_due(eda):
            return
        self.active_node, self.active_operation = eda, "review_fe"
        completed = self.completed_models()
        pending = self.compatibility.pending()
        routes = self.reuse_plan()
        allowed = {(route["issue_id"], route["reference_mt_id"]) for route in routes
                   if route["decision"] == "fork_allowed"}
        issues = [{**issue, "active_mt_ids": [node_id for node_id in issue["active_mt_ids"]
                   if (issue["issue_id"], node_id) in allowed]} for issue in pending]
        issues = [issue for issue in issues if issue["active_mt_ids"]]
        review = {"after_rollout": self.root.visits, "action": "pending",
                  "compatibility_issues": issues,
                  "feedback_node_ids": sorted({node_id for issue in issues
                                               for node_id in issue["active_mt_ids"]})}
        self.fe_reviews.append(review)
        try:
            self.last_fe_review_at = self.root.visits
            if not issues:
                review.update(action="skip", reason="existing_fe_reuse_not_exhausted" if pending
                              else "no_new_unresolved_compatibility", reuse_routes=routes)
                return
            proposal = await self.executor.propose_fe(completed, self.history, issues)
            if proposal is None or proposal.get("action") == "keep":
                review.update(action="keep", reason=(proposal or {}).get("reason", ""))
                self.compatibility.mark_reviewed(issues, self.root.visits)
                return
            if proposal.get("action") != "fork":
                raise ValueError("FE feedback must choose keep or fork")
            reference_id = proposal.get("reference_mt_id")
            issue = next((item for item in issues if item["issue_id"] == proposal.get("issue_id")), None)
            if issue is None:
                raise ValueError("FE issue_id must identify a pending compatibility issue")
            references = {n.id: n for n in completed if n.status == "failed"
                          and n.parent.id == issue["fe_node_id"] and n.id in issue["active_mt_ids"]}
            if type(reference_id) is not int or reference_id not in references:
                raise ValueError("FE reference_mt_id must identify an unresolved MT in the selected issue")
            reference = references[reference_id]
            child = Node(len(self.nodes), "FE", parent=eda, origin="fe_improve",
                         based_on=reference.parent.id, reference_mt_id=reference.id,
                         reference_rollout_id=self.rollout_for(reference.id)["rollout_id"],
                         source_rollout=self.root.visits, change_reason=proposal.get("reason", ""),
                         repair_issue_id=issue["issue_id"],
                         feedback_node_ids=review["feedback_node_ids"])
            eda.children.append(child)
            self.nodes.append(child)
            review.update(action="fork", node_id=child.id, reference_mt_id=reference.id,
                          issue_id=issue["issue_id"], reason=child.change_reason)
            # Mark only the chosen reference; unseen references remain eligible
            # for reuse or a later review when their FE observations are ready.
            self.compatibility.mark_reviewed([{**issue, "active_mt_ids": [reference.id]}], self.root.visits)
            self.active_node, self.active_operation = child, "fork_fe"
            await self.executor.fork_fe(child, reference, proposal, self.history[-1])
        except Exception as error:
            review.update(action="error", error=str(error))
            raise
        finally:
            self.save()

    async def replay(self, parent):
        """The reference replay is the new FE's first MT, not an extra model."""
        reference = self.nodes[parent.reference_mt_id]
        child = Node(len(self.nodes), "MT", parent=parent, origin="replay", based_on=reference.id,
                     reference_mt_id=reference.id, reference_rollout_id=parent.reference_rollout_id,
                     source_rollout=parent.source_rollout, feedback_node_ids=[reference.id],
                     repair_issue_id=parent.repair_issue_id)
        parent.children.append(child)
        self.nodes.append(child)
        self.active_node, self.active_operation = child, "replay_mt"
        try:
            await self.executor.replay_mt(child, reference)
        finally:
            self.save()

    async def improve(self, parent):
        completed = [n for n in parent.children if n.status in {"success", "failed"}]
        reference, selection = self.select_reference(parent)
        ids = {n.id for n in completed}
        feedback = next(record for record in reversed(self.history)
                        if record["path"][-1]["node_id"] in ids)
        child = Node(len(self.nodes), "MT", parent=parent, origin="improve",
                     based_on=reference.id, source_rollout=feedback["rollout_id"],
                     reference_mt_id=reference.id,
                     reference_rollout_id=self.rollout_for(reference.id)["rollout_id"],
                     feedback_node_ids=[n.id for n in completed], reference_selection=selection)
        parent.children.append(child)
        self.nodes.append(child)
        self.active_node, self.active_operation = child, "improve_mt"
        try:
            await self.executor.improve(child, reference, feedback, completed)
        finally:
            self.save()

    def reference_statistics(self, node):
        """Warm-start a modification route with its own measured reward.

        Only direct, newly executed feedback candidates update this route. The
        model's own score is a one-observation anchor, not a claimed estimate of
        a causal improvement. No LLM prior or infinite 'unvisited' bonus is used.
        """
        count = 1 + node.expansion_visits
        mean = (node.reward + node.expansion_total_reward) / count
        bonus = self.exploration * math.sqrt(math.log(node.parent.visits + 1) / count)
        return {"node_id": node.id, "status": node.status, "own_reward": node.reward,
                "expansion_visits": node.expansion_visits,
                "expansion_total_reward": node.expansion_total_reward,
                "effective_count": count, "mean_reward": mean,
                "exploration_bonus": bonus, "ucb": mean + bonus}

    def select_reference(self, parent):
        if not self.can_improve(parent):
            raise ValueError("MT reference selection requires completed candidates and FE expansion quota")
        candidates = [n for n in parent.children if n.visits > 0
                      and n.status in {"success", "failed"}]
        scores = [self.reference_statistics(n) for n in candidates]
        chosen = max(range(len(candidates)), key=lambda i: scores[i]["ucb"])
        reference = candidates[chosen]
        return reference, {"policy": "mt_reference_ucb", "fe_node_id": parent.id,
                           "fe_visits": parent.visits, "reference_mt_id": reference.id,
                           "candidates": scores}

    def update_reference(self, node):
        if node.stage == "MT" and node.origin == "improve":
            reference = self.nodes[node.reference_mt_id]
            if reference.parent is not node.parent:
                raise ValueError("MT feedback must update a reference in the same FE")
            reference.expansion_visits += 1
            reference.expansion_total_reward += node.reward

    def record_rollout(self, node):
        metric = self.executor.task["metric"]
        path = node.ancestors()[1:] + [node]
        errors = path_errors(path, self.root.visits)
        evaluation = None
        if node.stage == "MT" and node.status == "success":
            evaluation = {"split": "validation", "metric": metric,
                          "direction": "minimize" if metric in {"mse", "mae"} else "maximize",
                          "value": node.result["metrics"][metric], "reward": node.reward}
        record = {
            "trace_schema_version": 2, "status": "completed",
            "rollout_id": self.root.visits,
            "path": [{"node_id": n.id, "stage": n.stage, "idea": dict(n.idea), "method": n.method,
                      "status": n.status, "code_path": n.code_path}
                     for n in path],
            "errors": errors,
            "pipeline_context": pipeline_context(self.executor.task, path, self.output),
            "evaluation": evaluation, "reward": node.reward,
            "failure": {"node_id": node.id, "stage": node.stage, "error": node.feedback}
            if node.status == "failed" else None,
            "origin": node.origin, "based_on": node.based_on,
            "reference_mt_id": node.reference_mt_id, "reference_rollout_id": node.reference_rollout_id,
            "comparison": node.comparison,
            "compatibility_issue_ids": node.compatibility_issue_ids,
            "repair_issue_id": node.repair_issue_id,
            "attempts": node.attempts, "code_hash": node.code_hash,
            "reference_selection": node.reference_selection,
            "reference_update": self.reference_statistics(self.nodes[node.reference_mt_id])
            if node.stage == "MT" and node.origin == "improve" else None,
        }
        directory = self.output / "rollouts"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"rollout_{self.root.visits:04d}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2))
        self.history.append(record)
        save_records(self.output, self.executor.task, self.nodes, self.root.visits,
                     seen=self.summarized_node_ids)

    def record_interruption(self, error):
        """Preserve an incomplete run without inventing a rollout or method reward."""
        node = self.active_node
        path = [n for n in (node.ancestors() + [node] if node else []) if n.stage != "ROOT"]
        interruption = describe_error(error, "search_interruption", self.active_operation)
        if node is not None and not node.code_path:
            coding_error = self.output / f"node_{node.id:04d}_{node.stage}" / "coding_error.json"
            if coding_error.exists():
                interruption = {**json.loads(coding_error.read_text()), "error_path": str(coding_error)}
        record = {"trace_schema_version": 2, "status": "interrupted",
                  "completed_rollouts": self.root.visits, "attempted_rollout": self.active_rollout,
                  "evaluation": None, "reward": None,
                  "path": [{"node_id": n.id, "stage": n.stage, "idea": n.idea, "status": n.status,
                            "code_path": n.code_path} for n in path],
                  "pending_node_ids": [n.id for n in self.nodes if n.status == "pending"],
                  "errors": path_errors(path),
                  "pipeline_context": pipeline_context(self.executor.task, path, self.output),
                  "interruption": {**interruption, "search_operation": self.active_operation,
                                   "traceback": traceback.format_exc()}}
        (self.output / "interrupted_trace.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        save_records(self.output, self.executor.task, self.nodes, seen=self.summarized_node_ids)

    def ucb(self, node):
        alpha = self.prior_weight / (1 + node.visits)
        return ((1 - alpha) * node.total_reward / node.visits + alpha * node.prior
                + self.exploration * math.sqrt(math.log(node.parent.visits + 1) / node.visits))

    def select(self, parent):
        candidates = [child for child in parent.children if not self.exhausted(child)]
        unvisited = [child for child in candidates if child.visits == 0]
        if unvisited:
            # A new FE's fixed-model comparison runs before its other initial MTs.
            if parent.stage == "FE":
                replay = next((n for n in unvisited if n.origin == "replay"), None)
                if replay is not None:
                    return replay
            # max preserves generation order for equal priors. Low priors never prune.
            return max(unvisited, key=lambda child: child.prior)
        return max(candidates, key=self.ucb)

    async def expand(self, parent, count=None):
        stage = STAGES[STAGES.index(parent.stage) + 1]
        count = self.branches[stage] if count is None else count
        plan = self.executor.generation_plan(parent, count) if hasattr(self.executor, "generation_plan") else None
        if plan is not None:
            count = sum(group["count"] for group in plan)
        children = [Node(len(self.nodes) + i, stage, parent=parent) for i in range(count)]
        parent.children.extend(children)
        self.nodes.extend(children)
        self.active_node, self.active_operation = parent, f"generate_or_score_{stage}"
        try:
            if plan is None:
                await self.executor.generate(children)
            else:
                await self.executor.generate(children, plan=plan)
        finally:
            self.save()

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def save(self):
        self.output.mkdir(parents=True, exist_ok=True)
        records = [{
            "id": n.id, "stage": n.stage, "parent": n.parent.id if n.parent else None,
            "children": [c.id for c in n.children], "status": n.status, "idea": n.idea, "method": n.method,
            "code_path": n.code_path, "result": n.result, "feedback": n.feedback,
            "visits": n.visits, "total_reward": n.total_reward, "reward": n.reward,
            "prior": n.prior, "prior_scores": n.prior_scores, "prior_reason": n.prior_reason,
            "origin": n.origin, "generation_mode": n.generation_mode,
            "experience_source": n.experience_source,
            "based_on": n.based_on, "source_rollout": n.source_rollout,
            "feedback_node_ids": n.feedback_node_ids, "context_node_ids": n.context_node_ids,
            "reference_mt_id": n.reference_mt_id, "reference_rollout_id": n.reference_rollout_id,
            "change_reason": n.change_reason, "code_hash": n.code_hash,
            "attempts": n.attempts, "comparison": n.comparison,
            "expansion_visits": n.expansion_visits,
            "expansion_total_reward": n.expansion_total_reward,
            "reference_selection": n.reference_selection,
            "compatibility_issue_ids": n.compatibility_issue_ids,
            "repair_issue_id": n.repair_issue_id,
        } for n in self.nodes]
        (self.output / "tree.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
        summary = {"completed_rollouts": self.root.visits, "nodes": len(self.nodes) - 1,
                   "mt_candidates": sum(n.stage == "MT" for n in self.nodes),
                   "pending_models": sum(n.stage == "MT" and n.status == "pending" for n in self.nodes),
                   "successful_models": sum(n.stage == "MT" and n.status == "success" for n in self.nodes),
                   "best_node": self.best.id if self.best else None,
                   "best_path": [n.id for n in self.best.ancestors()[1:]] + [self.best.id] if self.best else [],
                   "best_validation": self.best.result["metrics"] if self.best else None,
                   "stop_reason": self.stop_reason,
                   "fe_variants": sum(n.origin == "fe_improve" for n in self.nodes),
                   "cross_fe_retries": sum(n.origin == "cross_fe_replay" for n in self.nodes),
                   "cross_fe_retry_limit": self.cross_fe_retries,
                   "unresolved_compatibility_issues": sum(bool(issue["active_mt_ids"])
                                                         for issue in self.compatibility.issues.values()),
                   "completed_mt_expansions": sum(n.expansion_visits for n in self.nodes),
                   "stage_attempts": sum(len(n.attempts) for n in self.nodes),
                   "mt_execution_attempts": sum(len(n.attempts) for n in self.nodes if n.stage == "MT"),
                   "model_fit_attempts": sum(bool(a.get("fit_started"))
                                             for n in self.nodes if n.stage == "MT" for a in n.attempts),
                   "execution_seconds": sum(a.get("elapsed_seconds", 0)
                                            for n in self.nodes for a in n.attempts)}
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2))
        (self.output / "fe_reviews.json").write_text(json.dumps(self.fe_reviews, ensure_ascii=False, indent=2))
        (self.output / "compatibility_issues.json").write_text(
            json.dumps(self.compatibility.snapshot(), ensure_ascii=False, indent=2))
        (self.output / "fe_reuse.json").write_text(json.dumps({
            "retry_limit": self.cross_fe_retries, "checks": self.fe_reuse_checks,
            "retry_node_ids": [n.id for n in self.nodes if n.origin == "cross_fe_replay"]},
            ensure_ascii=False, indent=2))

    async def run(self):
        try:
            return await self._run()
        except Exception as error:
            self.stop_reason = "interrupted"
            self.output.mkdir(parents=True, exist_ok=True)
            self.record_interruption(error)
            self.save()
            raise

    async def _run(self):
        for _ in range(self.rollouts):
            self.active_rollout = self.root.visits + 1
            self.active_node, self.active_operation = None, "select"
            retry = await self.maybe_reuse_fe()
            if retry is None:
                await self.maybe_fork_fe()
            if self.exhausted(self.root):
                self.stop_reason = "tree_exhausted"
                break
            node = retry or self.root
            if retry is not None:
                self.active_node, self.active_operation = node, "execute_cross_fe_replay"
                print(f"Replay MT {node.reference_mt_id} as node {node.id} on existing FE {node.parent.id}",
                      flush=True)
                try:
                    await self.executor.execute(node)
                finally:
                    self.save()
            while node.stage != "MT" and node.status != "failed":
                if not node.children:
                    if node.stage == "FE" and node.origin == "fe_improve":
                        await self.replay(node)
                        if self.branches["MT"] > 1:
                            await self.expand(node, count=self.branches["MT"] - 1)
                    else:
                        await self.expand(node)
                elif self.can_improve(node):
                    await self.improve(node)
                node = self.select(node)
                if node.status == "pending":
                    self.active_node, self.active_operation = node, "implement_and_execute"
                    print(f"Execute node {node.id}: {node.stage}, prior={node.prior:.3f}", flush=True)
                    try:
                        await self.executor.execute(node)
                    finally:
                        self.save()
            reward = node.reward
            if node.stage == "MT" and node.based_on is not None and hasattr(self.executor, "compare"):
                node.comparison = self.executor.compare(node, self.nodes[node.based_on])
            self.backpropagate(node, reward)
            self.update_reference(node)
            self.compatibility.observe(node, self.root.visits)
            # Reconciling already observed evidence costs no rollout. Do it even
            # after the final execution, when no further retry may be scheduled.
            recovered = [route for route in self.reuse_plan() if route["decision"] == "recovered"]
            if recovered:
                self.fe_reuse_checks.append({"after_rollout": self.root.visits,
                                            "phase": "reuse_observed_result", "routes": recovered})
                self.resolve_existing_results(recovered)
            self.record_rollout(node)
            if node.stage == "MT" and node.status == "success":
                if self.best is None or reward > self.best.reward:
                    self.best = node
            self.save()
            print(f"Rollout {self.root.visits}/{self.rollouts}: reward={reward:.6f}", flush=True)
        if self.stop_reason is None:
            self.stop_reason = "rollout_budget"
        self.save()
        if hasattr(self.executor, "finalize_experience"):
            await self.executor.finalize_experience(self.nodes, self.history)
        return self.best
