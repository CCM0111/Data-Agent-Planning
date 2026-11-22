"""Observed FE/MT input mismatches and a per-run, deduplicated repair queue."""

from copy import deepcopy
import re

import numpy as np
import pandas as pd
from scipy import sparse


def feature_profile(features):
    """Describe the executed representation without densifying sparse matrices."""
    is_sparse = sparse.issparse(features)
    values = features.tocsr(copy=False).data if is_sparse else np.asarray(features)
    numeric = values.dtype.kind in "biufc"
    return {"shape": list(features.shape), "dtype": str(values.dtype), "sparse": is_sparse,
            "format": features.format if is_sparse else "dense",
            "has_nan": bool(pd.isna(values).any()),
            "has_inf": bool(np.isinf(values).any()) if numeric else False,
            "has_negative": bool((values < 0).any()) if values.dtype.kind in "biuf" else False,
            "estimated_dense_bytes": int(np.prod(features.shape, dtype=object)) * values.dtype.itemsize}


def detect_input_mismatch(error, profile, phase):
    """Require both an explicit X-input error and a matching FE property.

    Unknown errors stay ordinary execution feedback. This deliberately does not
    diagnose target errors, timeouts, memory exhaustion, or model parameter bugs.
    """
    if phase not in {"fit", "predict"} or not isinstance(error, (ValueError, TypeError)):
        return None
    message = str(error)
    lower = message.lower()
    kind = None
    if (profile.get("sparse") and "dense data is required" in lower
            and ("sparse data was passed" in lower or "a sparse matrix was passed" in lower)):
        kind = "sparse_requires_dense"
    elif profile.get("has_nan") and re.search(r"\bInput X contains NaN\b", message):
        kind = "missing_values"
    elif profile.get("has_inf") and re.search(r"\bInput X contains infinity\b", message):
        kind = "infinite_values"
    elif (profile.get("has_negative") and "Negative values in data passed to" in message
          and "(input X)" in message):
        kind = "nonnegative_required"
    if kind is None:
        return None
    return {"kind": kind, "phase": phase, "error_type": type(error).__name__,
            "error": message[:6000], "feature_profile": profile,
            "evidence_basis": "model input error plus executed FE representation"}


class CompatibilityTracker:
    def __init__(self):
        self.issues = {}

    @staticmethod
    def refresh(issue):
        active = sorted({o["mt_node_id"] for o in issue["occurrences"] if o["status"] == "unresolved"})
        issue["active_mt_ids"] = active
        issue["status"] = "unresolved" if active else "inactive"

    def observe(self, node, rollout_id):
        """Register actual attempts after a rollout, including locally repaired ones."""
        ids = []
        if node.stage == "MT" and node.parent.status == "success":
            for index, attempt in enumerate(node.attempts):
                evidence = attempt.get("compatibility")
                if not evidence:
                    continue
                issue_id = f"fe_{node.parent.id}:{evidence['kind']}"
                issue = self.issues.setdefault(issue_id, {
                    "issue_id": issue_id, "fe_node_id": node.parent.id,
                    "fe_code_hash": node.parent.code_hash, "kind": evidence["kind"],
                    "version": 0, "reviewed_version": 0, "occurrences": [], "repair_outcomes": []})
                # Only the final failed code version can be replayed as the reference.
                state = ("resolved_by_mt_repair" if node.status == "success" else
                         "unresolved" if index == len(node.attempts) - 1 else "superseded_attempt")
                issue["occurrences"].append({**deepcopy(evidence), "mt_node_id": node.id,
                    "rollout_id": rollout_id, "attempt_id": attempt.get("attempt_id", index),
                    "code_path": attempt.get("code_path", node.code_path),
                    "code_hash": attempt.get("code_hash", node.code_hash),
                    "model_config": deepcopy(attempt.get("model_config")), "status": state})
                if state == "unresolved":
                    issue["version"] += 1
                self.refresh(issue)
                if issue_id not in ids:
                    ids.append(issue_id)
        node.compatibility_issue_ids = ids

        # An unchanged model running under a new FE is an execution recovery, not
        # evidence that every model affected by this grouped issue is now fixed.
        if node.repair_issue_id and (node.origin in {"replay", "cross_fe_replay"} or node.status == "failed"):
            issue = self.issues[node.repair_issue_id]
            issue["repair_outcomes"].append({"node_id": node.id, "stage": node.stage,
                "rollout_id": rollout_id, "status": node.status,
                "reference_mt_id": node.reference_mt_id,
                "new_fe_id": node.parent.id if node.stage == "MT" else node.id,
                "origin": node.origin})
            if node.origin in {"replay", "cross_fe_replay"} and node.status == "success":
                for occurrence in issue["occurrences"]:
                    if (occurrence["mt_node_id"] == node.reference_mt_id
                            and occurrence["status"] == "unresolved"):
                        occurrence.update(status="resolved_in_existing_fe" if node.origin == "cross_fe_replay"
                                          else "resolved_in_new_fe", recovery_mt_id=node.id)
                self.refresh(issue)
                # The fixed model can resolve multiple recorded constraints for
                # this source; never mark other models in the group recovered.
                if node.origin == "cross_fe_replay":
                    for related in self.issues.values():
                        for occurrence in related["occurrences"]:
                            if (occurrence["mt_node_id"] == node.reference_mt_id
                                    and occurrence["status"] == "unresolved"):
                                occurrence.update(status="resolved_in_existing_fe", recovery_mt_id=node.id)
                        self.refresh(related)

    def reuse_success(self, reference_id, recovery, rollout_id):
        """Link verified existing evidence; never execute or reward the model again.

        The caller verifies matching code hash and actual model configuration.
        Original failed nodes and the successful node's provenance remain intact.
        """
        if recovery.status != "success" or recovery.stage != "MT":
            raise ValueError("Existing recovery evidence must be a successful MT")
        for issue in self.issues.values():
            occurrences = [item for item in issue["occurrences"]
                           if item["mt_node_id"] == reference_id and item["status"] == "unresolved"]
            if not occurrences:
                continue
            for item in occurrences:
                item.update(status="resolved_in_existing_fe", recovery_mt_id=recovery.id,
                            recovery_basis="matching_code_hash_and_model_configuration")
            issue["repair_outcomes"].append({"node_id": recovery.id, "stage": "MT",
                "rollout_id": rollout_id, "status": "success", "reference_mt_id": reference_id,
                "new_fe_id": recovery.parent.id, "origin": "existing_result_reuse",
                "additional_execution": False})
            self.refresh(issue)

    def pending(self):
        result = []
        for issue in self.issues.values():
            active = [node_id for node_id in issue["active_mt_ids"]
                      if node_id not in issue.get("reviewed_mt_ids", [])]
            if active and issue["version"] > issue["reviewed_version"]:
                snapshot = deepcopy(issue)
                snapshot["active_mt_ids"] = active
                result.append(snapshot)
        return result

    def mark_reviewed(self, issues, rollout_id):
        for snapshot in issues:
            issue = self.issues[snapshot["issue_id"]]
            reviewed = set(issue.get("reviewed_mt_ids", [])) | set(snapshot["active_mt_ids"])
            issue.update(reviewed_mt_ids=sorted(reviewed), last_review_rollout=rollout_id)
            if set(issue["active_mt_ids"]) <= reviewed:
                issue["reviewed_version"] = snapshot["version"]

    def snapshot(self):
        return {"version": 1, "issues": list(self.issues.values())}
