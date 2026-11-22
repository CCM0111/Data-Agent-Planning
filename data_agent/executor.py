"""Generate and score stage ideas; implement and execute only selected candidates."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np

from .data import reward_for
from .diagnostics import file_hash
from .ideas import IDEA_FIELDS, validate_idea
from .experience_retrieval import eda_metadata
from .experience_library import record_hash
from .idea_context import compact_task
from .traces import describe_error
from .prompts import (SYSTEM, CODING_SYSTEM, REPAIR_SYSTEM, SCORING_SYSTEM, SCORE_WEIGHTS,
                      FE_REVIEW_SYSTEM, build_prompt, build_coding_prompt, build_scoring_prompt,
                      build_improvement_prompt, build_fe_review_prompt)


def run_process(context_path, timeout):
    workspace = Path(context_path).parent
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    # An explicit script path keeps execution independent of the caller's cwd.
    with (workspace / "execution.log").open("w") as log:
        try:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("runtime.py")), str(context_path)],
                cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            message = f"Stage exceeded {timeout} seconds"
            (workspace / "process_error.json").write_text(json.dumps({
                "kind": "timeout", "exception_type": "TimeoutExpired", "message": message,
                "timeout_seconds": timeout}, indent=2))
            return False, message
    feedback = (workspace / "execution.log").read_text()
    if result.returncode != 0 and not feedback.strip():
        feedback = f"Process exited with return code {result.returncode}"
    if result.returncode != 0:
        (workspace / "process_error.json").write_text(json.dumps({
            "kind": "process_exit", "exception_type": None, "returncode": result.returncode,
            "message": f"Process exited with return code {result.returncode}"}, indent=2))
    return result.returncode == 0, feedback


class StageExecutor:
    def __init__(self, model, task, output, timeout=120, max_repairs=1, experience_library=None):
        self.model, self.task = model, task
        self.output = Path(output).resolve()
        self.timeout, self.max_repairs = timeout, max_repairs
        self.completed_mts = []
        self.experience_library = experience_library
        self.experience_record = None
        self.last_fe_review_node_ids = []
        self.experience_source = None
        self.experience_source_checked = False
        self.experience_source_ref = None
        if isinstance(task.get("data_profile"), dict):
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / "data_profile.json").write_text(
                json.dumps(task["data_profile"], ensure_ascii=False, indent=2, allow_nan=False))

    def resolve_experience_source(self, prefix):
        """Freeze one source (including a no-match decision) for this entire run."""
        if self.experience_source_checked:
            return self.experience_source
        metadata = eda_metadata(prefix)
        library = self.experience_library
        source = library.select_source(self.task, metadata) if library is not None else None
        self.experience_source = source
        self.experience_source_checked = True
        if source is not None:
            self.experience_source_ref = {
                "task_key": source["task_key"],
                "task_id": source["source_refs"]["task_id"],
                "record_sha256": record_hash(source),
            }
        self.output.mkdir(parents=True, exist_ok=True)
        snapshot = {"status": "selected" if source is not None else "no_source",
                    "query": {"task": compact_task(self.task), "eda_metadata": metadata},
                    "source_ref": self.experience_source_ref, "record": source,
                    "retrieval": library.last_retrieval_report if library is not None else None}
        (self.output / "experience_source.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False))
        return source

    def generation_plan(self, parent, count):
        """The standard three-slot expansion uses two free + one transfer slot.

        MT expands the transfer slot to two only on a hit. A repaired FE's fixed
        replay occupies one free slot. Nonstandard branch counts retain their
        configured all-free behavior, useful for small smoke runs.
        """
        if parent.stage not in {"EDA", "FE"}:
            return [{"mode": "free", "count": count}]
        reserved = sum(child.origin == "replay" for child in parent.children)
        if count + reserved != 3:
            return [{"mode": "free", "count": count}]
        tail_count = 1
        if parent.stage == "FE":
            prefix = self.idea_prefix(parent)
            if self.resolve_experience_source(prefix) is not None:
                tail_count = 2
        return [{"mode": "free", "count": 2 - reserved},
                {"mode": "transfer_or_free", "count": tail_count}]

    async def finalize_experience(self, nodes, history):
        if self.experience_library is not None:
            try:
                record = await self.experience_library.curate(self.task, nodes, self.output, self.model, history)
                self.experience_record = record
                if record is not None:
                    print(f"Experience summary: {record['summary']['status']}", flush=True)
                return record
            except Exception as error:
                # This optional post-search step must not discard a valid search result.
                (self.output / "experience_status.json").write_text(json.dumps({
                    "status": "failed", "error_type": type(error).__name__}, indent=2))
                print(f"Experience export failed ({type(error).__name__}); search results retained", flush=True)

    def summary(self, node):
        return {"node_id": node.id, "fe_node_id": node.parent.id, "idea": node.idea, "method": node.method,
                "status": node.status, "code_hash": getattr(node, "code_hash", ""),
                "validation_metrics": node.result.get("metrics") if node.status == "success" else None,
                "diagnostics": node.result.get("diagnostics"),
                "model_config": node.result.get("model_config"), "reward": node.reward,
                "feedback_expansions": {"completed": node.expansion_visits,
                                        "total_reward": node.expansion_total_reward},
                "comparison": getattr(node, "comparison", {}),
                "error_details": (node.attempts[-1].get("error_details")
                                  if node.status == "failed" and node.attempts else None),
                "error": node.feedback[-30000:] if node.status == "failed" else None}

    def prefix(self, node):
        """Full executed context for implementation, repair, and runtime artifacts."""
        return [{"stage": ancestor.stage, "idea": ancestor.idea, "method": ancestor.method,
                 "code": Path(ancestor.code_path).read_text(),
                 "result": ancestor.result, "execution_feedback": ancestor.feedback}
                for ancestor in node.ancestors() if ancestor.stage != "ROOT"]

    def idea_prefix(self, node):
        """Recorded facts without reading source; builders bound their prompt view."""
        return [{"node_id": ancestor.id, "stage": ancestor.stage, "status": ancestor.status,
                 "idea": ancestor.idea, "method": ancestor.method, "result": ancestor.result,
                 "attempts": ancestor.attempts, "feedback": ancestor.feedback}
                for ancestor in node.ancestors() if ancestor.stage != "ROOT"]

    async def generate(self, nodes, plan=None, source_record=None, score=True):
        stage = nodes[0].stage
        prefix = self.idea_prefix(nodes[0])
        plan = plan or [{"mode": "free", "count": len(nodes)}]
        if sum(group["count"] for group in plan) != len(nodes):
            raise ValueError("Generation plan must account for every candidate")
        directory = self.output / f"expansion_{nodes[0].parent.id:04d}_{stage}"
        directory.mkdir(parents=True)
        existing = [{"node_id": n.id, **n.idea}
                    for n in nodes[0].parent.children if n not in nodes and n.idea]
        proposals, batches = [], []
        for index, group in enumerate(plan):
            count = group["count"]
            if not count:
                continue
            source = None
            mode = group["mode"]
            if mode == "guided":
                if source_record is None or self.experience_library is None or stage not in {"FE", "MT"}:
                    raise ValueError("Guided generation requires a source experience and FE/MT stage")
                source = source_record
                mode = "experience"
            elif mode == "transfer_or_free":
                source = self.resolve_experience_source(prefix)
                mode = "experience" if source is not None else "free_fallback"
            batch_nodes = nodes[len(proposals):len(proposals) + count]
            prior_ideas = [{"node_id": node.id, **proposal}
                           for node, proposal in zip(nodes, proposals)]
            prompt = build_prompt(stage, self.task, prefix, count=count,
                                  existing_candidates=[*existing, *prior_ideas])
            if source is not None:
                prompt = self.experience_library.inject_source(prompt, source, stage)
            batch_dir = directory if len(plan) == 1 else directory / f"batch_{index + 1}_{mode}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            (batch_dir / "generation_prompt.json").write_text(prompt)
            for node in batch_nodes:
                node.generation_mode = "experience" if source is not None else "free"
                node.experience_source = ({"task_key": source["task_key"],
                    "task_id": source["source_refs"]["task_id"], "record_sha256": record_hash(source)}
                    if source is not None else None)
            batches.append({"mode": mode, "node_ids": [node.id for node in batch_nodes],
                            "prompt": str((batch_dir / "generation_prompt.json").relative_to(directory))})
            (directory / "generation_plan.json").write_text(json.dumps(batches, indent=2))
            response = await self.model.generate(prompt, system_prompt=SYSTEM,
                                                 response_format={"type": "json_object"})
            (batch_dir / "generation_response.txt").write_text(response)
            batch = json.loads(response)["candidates"]
            if not isinstance(batch, list) or len(batch) != count:
                raise ValueError(f"Expected {count} candidates in {mode} batch")
            proposals.extend(validate_idea(proposal) for proposal in batch)
        await self.save_and_score(nodes, proposals, prefix, directory, score=score)

    async def generate_guided(self, node, source_record):
        """One explicit experience-guided candidate, independent of normal quotas."""
        await self.generate([node], plan=[{"mode": "guided", "count": 1}],
                            source_record=source_record, score=False)

    async def improve(self, node, reference, feedback, completed):
        node.context_node_ids = sorted({reference.id, *[n.id for n in completed],
                                        *[n.id for n in self.completed_mts]})
        prefix = self.idea_prefix(node)
        directory = self.output / f"improvement_{node.parent.id:04d}_{node.id:04d}"
        directory.mkdir(parents=True)
        reference_candidate = {**self.summary(reference), "code": Path(reference.code_path).read_text()}
        comparisons = [self.summary(n) for n in completed if n.status in {"success", "failed"}]
        snapshot = {"experiment_feedback": feedback, "comparison_candidates": comparisons,
                    "reference_selection": node.reference_selection}
        (directory / "feedback.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
        prompt = build_improvement_prompt(self.task, prefix, reference_candidate, feedback, comparisons,
                                         history=[self.summary(n) for n in self.completed_mts],
                                         selection=node.reference_selection)
        (directory / "prompt.json").write_text(prompt)
        response = await self.model.generate(prompt, system_prompt=SYSTEM,
                                             response_format={"type": "json_object"})
        (directory / "response.txt").write_text(response)
        context = {"reference_candidate": reference_candidate, **snapshot,
                   "executed_mt_history": [self.summary(n) for n in self.completed_mts]}
        await self.save_and_score([node], [json.loads(response)], prefix, directory,
                                  implementation_context=context)

    async def propose_fe(self, completed_mts, history, issues):
        eligible = [n for n in completed_mts if n.stage == "MT" and n.status == "failed"
                    and n.parent.stage == "FE" and n.parent.status == "success"
                    and any(n.parent.id == issue["fe_node_id"] and n.id in issue["active_mt_ids"]
                            for issue in issues)]
        if not eligible:
            return None
        self.last_fe_review_node_ids = [n.id for n in eligible]
        cases = [{**self.summary(n), "prefix": self.idea_prefix(n)} for n in eligible]
        rollout = history[-1]["rollout_id"] if history else 0
        directory = self.output / f"fe_review_{rollout:04d}"
        directory.mkdir(parents=True)
        prompt = build_fe_review_prompt(self.task, cases, issues)
        (directory / "prompt.json").write_text(prompt)
        response = await self.model.generate(prompt, system_prompt=FE_REVIEW_SYSTEM,
                                             response_format={"type": "json_object"})
        (directory / "response.txt").write_text(response)
        proposal = json.loads(response)
        if proposal.get("action") not in {"keep", "fork"} or not isinstance(proposal.get("reason"), str):
            raise ValueError("FE review must return keep/fork with a reason")
        if proposal["action"] == "keep":
            return proposal
        reference_id = proposal.get("reference_mt_id")
        issue = next((item for item in issues if item["issue_id"] == proposal.get("issue_id")), None)
        if issue is None:
            raise ValueError("FE review issue_id must identify a pending compatibility issue")
        if (type(reference_id) is not int or reference_id not in
                {n.id for n in eligible if n.parent.id == issue["fe_node_id"]
                 and n.id in issue["active_mt_ids"]}):
            raise ValueError("FE review reference must identify an unresolved MT in the selected issue")
        validate_idea(proposal)
        return proposal

    async def fork_fe(self, node, reference_mt, proposal, feedback):
        if node.stage != "FE" or node.parent is not reference_mt.parent.parent:
            raise ValueError("New FE must be a sibling of the reference FE")
        node.context_node_ids = sorted({reference_mt.id, reference_mt.parent.id, *self.last_fe_review_node_ids})
        directory = self.output / f"fe_fork_{node.id:04d}"
        directory.mkdir(parents=True)
        (directory / "feedback.json").write_text(json.dumps({"reference_mt_id": reference_mt.id,
            "proposal": proposal, "feedback": feedback}, ensure_ascii=False, indent=2))
        context = {"reference_candidate": {**self.summary(reference_mt),
                    "code": Path(reference_mt.code_path).read_text(), "prefix": self.prefix(reference_mt)},
                   "experiment_feedback": feedback, "reason": proposal.get("reason", ""),
                   "issue_id": proposal.get("issue_id")}
        await self.save_and_score([node], [proposal], self.idea_prefix(node), directory,
                                  implementation_context=context)

    async def replay_mt(self, node, reference_mt):
        if node.stage != "MT" or reference_mt.stage != "MT":
            raise ValueError("Replay requires an MT node and MT reference")
        source = Path(reference_mt.code_path)
        recorded_hash = getattr(reference_mt, "code_hash", "")
        if recorded_hash and file_hash(source) != recorded_hash:
            raise ValueError("Reference model code changed after its recorded execution")
        directory = self.output / f"node_{node.id:04d}_MT"
        directory.mkdir(parents=True)
        if node.origin != "cross_fe_replay":
            node.origin = "replay"
        node.idea = validate_idea({
            "name": f"replay_mt_{reference_mt.id}",
            "evidence": f"Reference MT {reference_mt.id} supplies the recorded model code for this FE comparison.",
            "method": f"Reuse the exact model code/configuration from MT {reference_mt.id} on the target FE.",
            "reason": "Keeping the recorded model fixed lets this run assess the changed FE's compatibility.",
        })
        node.method = node.idea["method"]
        (directory / "idea.json").write_text(json.dumps(node.idea, ensure_ascii=False, indent=2))
        node.code_path = str(directory / "proposal.py")
        Path(node.code_path).write_bytes(source.read_bytes())
        node.code_hash = file_hash(node.code_path)
        (directory / "replay_reference.json").write_text(json.dumps({
            "reference_mt_id": reference_mt.id, "source_code_path": str(source),
            "source_code_hash": node.code_hash, "model_config": reference_mt.result.get("model_config"),
            "repair_allowed": False}, indent=2))
        node.prior = 0.0
        node.prior_reason = "Fixed reference model replay; no candidate ranking required"

    async def save_and_score(self, nodes, proposals, prefix, directory, implementation_context=None, *, score=True):
        stage = nodes[0].stage
        candidates = [{"candidate_id": node.id, **validate_idea(proposal)}
                      for node, proposal in zip(nodes, proposals)]
        for node, candidate in zip(nodes, candidates):
            node_dir = self.output / f"node_{node.id:04d}_{stage}"
            node_dir.mkdir()
            node.idea = {key: candidate[key] for key in IDEA_FIELDS}
            node.method = candidate["method"]
            (node_dir / "idea.json").write_text(json.dumps(node.idea, ensure_ascii=False, indent=2))
            if implementation_context is not None:
                (node_dir / "idea_context.json").write_text(
                    json.dumps(implementation_context, ensure_ascii=False, indent=2))

        if not score or (stage == "MT" and len(nodes) == 1):
            nodes[0].prior = 0.0
            nodes[0].prior_reason = "Single selected candidate; no candidate ranking required"
            return

        prompt = build_scoring_prompt(stage, self.task, prefix, candidates)
        (directory / "scoring_prompt.json").write_text(prompt)
        response = await self.model.generate(prompt, system_prompt=SCORING_SYSTEM,
                                             temperature=0, response_format={"type": "json_object"})
        (directory / "scoring_response.txt").write_text(response)
        scores = json.loads(response)["scores"]
        if len(scores) != len(nodes) or {s["candidate_id"] for s in scores} != {n.id for n in nodes}:
            raise ValueError("Scoring must include each candidate ID exactly once")
        by_id = {s["candidate_id"]: s for s in scores}
        for node in nodes:
            item = by_id[node.id]
            criteria = {key: float(item[key]) for key in SCORE_WEIGHTS}
            if not all(0 <= value <= 10 for value in criteria.values()):
                raise ValueError("LLM criteria must be finite scores between zero and ten")
            node.prior_scores = criteria
            node.prior = sum(SCORE_WEIGHTS[key] * value for key, value in criteria.items()) / 10
            node.prior_reason = item["reason"]

    async def implement(self, node):
        """Called only for a selected pending node, before its first execution."""
        if node.code_path:
            return
        idea = validate_idea(node.idea)
        directory = self.output / f"node_{node.id:04d}_{node.stage}"
        context_path = directory / "idea_context.json"
        context = json.loads(context_path.read_text()) if context_path.exists() else None
        prompt = build_coding_prompt(node.stage, self.task, self.prefix(node), idea, context=context)
        (directory / "coding_prompt.json").write_text(prompt)
        phase = "coding_generation"
        try:
            response = await self.model.generate(prompt, system_prompt=CODING_SYSTEM,
                                                 response_format={"type": "json_object"})
            (directory / "coding_response.txt").write_text(response)
            phase = "parse_coding_response"
            proposal = json.loads(response)
            if (not isinstance(proposal, dict) or not isinstance(proposal.get("code"), str)
                    or not proposal["code"].strip()):
                raise ValueError("Coding must return a nonempty code string for the selected idea")
        except Exception as error:
            (directory / "coding_error.json").write_text(json.dumps(
                describe_error(error, "api_error" if phase == "coding_generation" else "response_error", phase),
                ensure_ascii=False, indent=2))
            raise
        # Do not mark a node implemented before a valid response has been saved.
        path = directory / "proposal.py"
        path.write_text(proposal["code"])
        node.code_path, node.code_hash = str(path), file_hash(path)

    async def execute(self, node):
        await self.implement(node)
        prefix = self.prefix(node)
        node_dir = self.output / f"node_{node.id:04d}_{node.stage}"
        feedback = None
        code = Path(node.code_path).read_text()
        node.attempts = []
        replay = node.origin in {"replay", "cross_fe_replay"}
        replay_reference = json.loads((node_dir / "replay_reference.json").read_text()) if replay else None
        if replay and file_hash(node.code_path) != replay_reference["source_code_hash"]:
            raise ValueError("Replay source code changed after it was recorded")
        for attempt in range((0 if replay else self.max_repairs) + 1):
            workspace = node_dir / f"attempt_{attempt}"
            workspace.mkdir()
            node.result = {}
            started = time.monotonic()
            attempt_record = {"attempt_id": attempt, "status": "preparing", "fit_started": False,
                              "fit_completed": False, "elapsed_seconds": 0.0}
            node.attempts.append(attempt_record)
            if attempt:
                repair_feedback = dict(feedback)
                if len(repair_feedback["error"]) > 30000:
                    repair_feedback["error"] = (
                        "[Error log truncated; showing the last 30000 characters.]\n"
                        + repair_feedback["error"][-30000:]
                    )
                prompt = build_prompt(node.stage, self.task, prefix, feedback=repair_feedback, idea=node.idea)
                (workspace / "prompt.json").write_text(prompt)
                # API failures stop search rather than becoming method rewards.
                try:
                    response = await self.model.generate(prompt, system_prompt=REPAIR_SYSTEM,
                                                         response_format={"type": "json_object"})
                except Exception as error:
                    attempt_record.update(status="generation_error", elapsed_seconds=time.monotonic() - started,
                        error=str(error), error_details=describe_error(error, "api_error", "repair_generation"))
                    (workspace / "attempt.json").write_text(json.dumps(attempt_record, indent=2))
                    raise
                (workspace / "response.txt").write_text(response)
            phase = "parse_response" if attempt else "compile"
            try:
                if attempt:
                    proposal = json.loads(response)
                    attempt_record["repair_description"] = proposal["method"]
                    code = proposal["code"]
                (workspace / "code.py").write_text(code)
                node.code_path = str(workspace / "code.py")
                node.code_hash = file_hash(node.code_path)
                attempt_record.update(code_path=node.code_path, code_hash=node.code_hash)
                phase = "compile"
                compile(code, str(workspace / "code.py"), "exec")
                phase = "prepare_context"
                context = {"stage": node.stage, "task": self.task, "node_id": node.id, "attempt_id": attempt}
                if replay:
                    context["expected_model_config"] = replay_reference["model_config"]
                for ancestor in prefix:
                    artifacts = ancestor["result"]["artifacts"]
                    if ancestor["stage"] == "EDA":
                        context["eda_path"] = artifacts["eda"]
                    elif ancestor["stage"] == "FE":
                        context["features_path"] = artifacts["features"]
                context_path = workspace / "context.json"
                context_path.write_text(json.dumps(context, indent=2))
                phase = "runtime"
                success, log = run_process(context_path, self.timeout)
                if not success:
                    raise ValueError(log)
                phase = "read_result"
                result = json.loads((workspace / "result.json").read_text())
                if node.stage == "MT":
                    node.reward = reward_for(self.task["metric"], result["metrics"][self.task["metric"]])
                node.code_path, node.result, node.feedback = str(workspace / "code.py"), result, log
                node.status = "success"
                attempt_record.update(result.get("execution", {}))
                attempt_record.update(status="success", elapsed_seconds=time.monotonic() - started)
                (workspace / "attempt.json").write_text(json.dumps(attempt_record, indent=2))
                self.finish_execution(node)
                return
            except (ValueError, TypeError, KeyError, OSError, SyntaxError) as error:
                feedback = {"method": node.method, "code": code, "error": str(error)}
                (workspace / "failure.json").write_text(json.dumps(feedback, indent=2))
                node.feedback = str(error)
                for name, key in [("model_config.json", "model_config"),
                                  ("transformer_config.json", "transformer_config"),
                                  ("execution_state.json", "execution")]:
                    path = workspace / name
                    if path.exists():
                        node.result[key] = json.loads(path.read_text())
                attempt_record.update(node.result.get("execution", {}))
                detail_path = workspace / "error.json"
                process_path = workspace / "process_error.json"
                process = json.loads(process_path.read_text()) if process_path.exists() else {}
                if detail_path.exists() and process.get("kind") != "timeout":
                    detail = json.loads(detail_path.read_text())
                    detail.pop("traceback", None)  # Full traceback stays in error.json and execution.log.
                    detail["error_path"] = str(detail_path)
                elif process:
                    detail = {**process, "phase": node.result.get("execution", {}).get("phase", phase),
                              "error_path": str(process_path)}
                else:
                    detail = describe_error(error, "response_error" if phase == "parse_response"
                                            else "execution_error", phase)
                attempt_record.update(error_details=detail,
                    log_path=str(workspace / "execution.log") if (workspace / "execution.log").exists() else None,
                    failure_path=str(workspace / "failure.json"))
                compatibility_path = workspace / "compatibility.json"
                if node.stage == "MT" and compatibility_path.exists():
                    attempt_record["compatibility"] = json.loads(compatibility_path.read_text())
                    attempt_record["model_config"] = node.result.get("model_config")
                attempt_record.update(status="failed", elapsed_seconds=time.monotonic() - started,
                                      error=str(error))
                (workspace / "attempt.json").write_text(json.dumps(attempt_record, indent=2))
        node.status, node.reward = "failed", 0.0
        self.finish_execution(node)

    def finish_execution(self, node):
        node.result.setdefault("execution", {}).update(
            attempt_count=len(node.attempts),
            fit_attempt_count=sum(bool(a.get("fit_started")) for a in node.attempts),
            elapsed_seconds=sum(a["elapsed_seconds"] for a in node.attempts))
        if node.stage == "MT":
            self.completed_mts.append(node)

    def compare(self, node, reference_mt):
        comparison = {"reference_mt_id": reference_mt.id, "candidate_mt_id": node.id,
                      "reference_status": reference_mt.status, "candidate_status": node.status}
        if reference_mt.status != "success":
            return {**comparison, "kind": "execution_recovery",
                    "recovered": node.status == "success", "performance_comparable": False,
                    "reason": "Reference has no successful predictions; no performance attribution is possible"}
        if node.status != "success":
            return {**comparison, "kind": "execution_failure", "performance_comparable": False,
                    "reason": "Candidate has no successful predictions"}
        try:
            before = joblib.load(reference_mt.result["artifacts"]["predictions"])
            after = joblib.load(node.result["artifacts"]["predictions"])
        except (KeyError, OSError):
            return {**comparison, "kind": "unavailable", "performance_comparable": False,
                    "reason": "Prediction artifacts are unavailable"}
        if (before["split_identity"] != after["split_identity"]
                or not np.array_equal(before["row_id"], after["row_id"])
                or not np.array_equal(before["target"], after["target"])):
            return {**comparison, "kind": "incomparable", "performance_comparable": False,
                    "reason": "Validation split, row order, or targets differ"}
        fixed_model = (bool(node.code_hash) and node.code_hash == reference_mt.code_hash
                       and node.result.get("model_config") is not None
                       and node.result["model_config"] == reference_mt.result.get("model_config"))
        same_fe = node.parent is reference_mt.parent
        metric = self.task["metric"]
        old_score, new_score = reference_mt.result["metrics"][metric], node.result["metrics"][metric]
        comparison.update(kind="prediction_comparison", performance_comparable=True,
                          changed_scope="MT" if same_fe else "FE" if fixed_model else "FE+MT",
                          fixed_model_configuration=bool(fixed_model), metric=metric,
                          reference_value=old_score, candidate_value=new_score,
                          raw_delta=new_score - old_score,
                          improvement=old_score - new_score if metric in {"mse", "mae"} else new_score - old_score,
                          limitations="One observed comparison on fixed validation rows; not a causal or generalization claim")
        targets = np.asarray(before["target"])
        old, new = np.asarray(before["prediction"]), np.asarray(after["prediction"])
        if self.task["problem_type"] == "classification":
            old_correct, new_correct = old == targets, new == targets
            comparison.update(repaired=int(np.sum(~old_correct & new_correct)),
                              damaged=int(np.sum(old_correct & ~new_correct)),
                              unchanged_correct=int(np.sum(old_correct & new_correct)),
                              unchanged_wrong=int(np.sum(~old_correct & ~new_correct)),
                              count_interpretation="Correctness counts describe accuracy, not additive Macro-F1 gains")
        else:
            old_loss = (old.astype(float) - targets.astype(float)) ** 2
            new_loss = (new.astype(float) - targets.astype(float)) ** 2
            if metric == "mae":
                old_loss, new_loss = np.sqrt(old_loss), np.sqrt(new_loss)
            difference = old_loss - new_loss
            comparison.update(improved_rows=int(np.sum(difference > 0)),
                              worsened_rows=int(np.sum(difference < 0)),
                              unchanged_rows=int(np.sum(difference == 0)),
                              mean_row_loss_reduction=float(difference.mean()),
                              row_loss="absolute_error" if metric == "mae" else "squared_error")
        return comparison

    def evaluate_best(self, best, test_path):
        fe = best.parent
        workspace = self.output / "best_test"
        workspace.mkdir()
        context = {"stage": "TEST", "task": self.task, "test_path": str(test_path),
                   "transformer_path": fe.result["artifacts"]["transformer"],
                   "model_path": best.result["artifacts"]["model"]}
        context_path = workspace / "context.json"
        context_path.write_text(json.dumps(context, indent=2))
        success, log = run_process(context_path, self.timeout)
        if not success:
            raise RuntimeError(f"Best-path test evaluation failed: {log}")
        return json.loads((workspace / "result.json").read_text())["metrics"]
