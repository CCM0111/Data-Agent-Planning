"""Plan bounded fixed-model retries using already recorded FE observations only."""


INPUT_FLAGS = {"sparse_requires_dense": "sparse", "missing_values": "has_nan",
               "infinite_values": "has_inf", "nonnegative_required": "has_negative"}
FEEDBACK_ORIGINS = {"improve", "cross_fe_replay"}


def feedback_used(fe):
    return sum(child.origin in FEEDBACK_ORIGINS for child in fe.children)


def assess_fe(fe, kinds):
    """A possible match satisfies known constraints, not all estimator contracts."""
    result = {"fe_node_id": fe.id, "status": "unknown", "reason": "fe_not_executed"}
    if fe.status == "failed":
        return {**result, "status": "excluded", "reason": "fe_execution_failed"}
    if fe.status != "success":
        return result
    profiles = fe.result.get("feature_profiles")
    profiles = profiles if isinstance(profiles, dict) else {}
    unknown = not kinds or any(kind not in INPUT_FLAGS for kind in kinds)
    violations = []
    for split in ("train", "validation"):
        profile = profiles.get(split)
        profile = profile if isinstance(profile, dict) else {}
        for kind in kinds:
            flag = INPUT_FLAGS.get(kind)
            value = profile.get(flag)
            if value is True:
                violations.append({"split": split, "kind": kind, "field": flag})
            elif value is not False:
                unknown = True
    if violations:
        return {**result, "status": "excluded", "reason": "known_input_mismatch",
                "violations": violations}
    if unknown:
        return {**result, "reason": "missing_representation_facts"}
    train = profiles.get("train", {})
    shape = train.get("shape", [])
    width = shape[1] if isinstance(shape, (list, tuple)) and len(shape) == 2 else None
    size = train.get("estimated_dense_bytes")
    return {**result, "status": "possible", "reason": "known_constraints_satisfied",
            "feature_count": width if type(width) is int and width >= 0 else None,
            "estimated_dense_bytes": size if type(size) is int and size >= 0 else None}


def reuse_routes(nodes, issues, retry_limit, feedback_limit):
    """Produce auditable decisions without reading artifacts or invoking a model."""
    by_id = {node.id: node for node in nodes}
    retries = [node for node in nodes if node.origin == "cross_fe_replay"]
    routes = []
    for issue in issues:
        for reference_id in issue["active_mt_ids"]:
            reference = by_id[reference_id]
            # A failed migration remains evidence for its original route, not a
            # new route that could bypass the global retry budget or fork gate.
            if reference.origin == "cross_fe_replay":
                continue
            kinds = sorted({item["kind"] for item in issues if reference_id in item["active_mt_ids"]})
            alternatives = []
            for fe in reference.parent.parent.children:
                if fe is reference.parent:
                    continue
                observation = assess_fe(fe, kinds)
                attempted = next((node for node in retries if node.reference_mt_id == reference_id
                                  and node.parent is fe), None)
                if attempted is not None:
                    observed_kind = (attempted.attempts[-1].get("compatibility", {}).get("kind")
                                     if attempted.attempts else None)
                    observation.update(retry_node_id=attempted.id)
                    if attempted.status == "success":
                        observation.update(status="recovered", reason="fixed_model_succeeded")
                    elif attempted.status == "failed" and observed_kind in kinds:
                        observation.update(status="excluded", reason="retry_same_input_mismatch")
                    else:
                        observation.update(status="unknown", reason="retry_did_not_confirm_same_mismatch")
                # Never rerun an already observed code/configuration + FE pair,
                # even when it was recorded under a different reference node ID.
                matching = [child for child in fe.children if reference.code_hash
                            and child.code_hash == reference.code_hash
                            and reference.result.get("model_config")
                            and child.result.get("model_config") == reference.result["model_config"]]
                failed = next((child for child in reversed(matching) if child.status == "failed"), None)
                if failed is not None:
                    failure_kind = (failed.attempts[-1].get("compatibility", {}).get("kind")
                                    if failed.attempts else None)
                    observation.update(status="excluded" if failure_kind in kinds else "unknown",
                                       reason="existing_exact_model_failed_same_mismatch" if failure_kind in kinds
                                       else "existing_exact_model_failed_other_error", observed_mt_id=failed.id)
                # A successful observation takes precedence, without copying its
                # reward, creating a new node, or running an additional fit.
                verified = next((child for child in matching if child.status == "success"), None)
                if fe.status == "success" and verified is not None:
                    observation.update(status="recovered", reason="existing_exact_model_succeeded",
                                       recovery_mt_id=verified.id)
                observation["feedback_slots_remaining"] = max(0, feedback_limit - feedback_used(fe))
                observation["initial_models_completed"] = bool(fe.children) and all(
                    child.status in {"success", "failed"} for child in fe.children)
                alternatives.append(observation)
            route = {"issue_id": issue["issue_id"], "reference_mt_id": reference_id,
                     "source_fe_id": reference.parent.id, "known_mismatches": kinds,
                     "alternatives": alternatives, "decision": "defer"}
            if any(item["status"] == "recovered" for item in alternatives):
                recovered = next(item for item in alternatives if item["status"] == "recovered")
                route.update(decision="recovered", reason="existing_fe_recovered_execution",
                             recovery_mt_id=recovered.get("recovery_mt_id", recovered.get("retry_node_id")))
            elif all(item["status"] == "excluded" for item in alternatives):
                route.update(decision="fork_allowed", reason="no_existing_fe_can_resolve_known_mismatch")
            else:
                ready = [item for item in alternatives if item["status"] == "possible"
                         and item["feedback_slots_remaining"] and item["initial_models_completed"]]
                if ready and len(retries) < retry_limit and reference.result.get("model_config"):
                    def cost(item):
                        return (item["estimated_dense_bytes"] is None, item["estimated_dense_bytes"] or 0,
                                item["feature_count"] is None, item["feature_count"] or 0, item["fe_node_id"])
                    chosen = min(ready, key=cost)
                    route.update(decision="retry", target_fe_id=chosen["fe_node_id"],
                                 reason="smallest_ready_representation_satisfying_known_constraints")
                else:
                    route["reason"] = ("retry_budget_exhausted" if ready and len(retries) >= retry_limit
                                       else "await_observations_or_available_feedback_slot")
            routes.append(route)
    return routes
