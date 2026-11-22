"""Short, separate idea fields for generation and deterministic legacy exports."""

import re


IDEA_FIELDS = ("name", "evidence", "method", "reason")
IDEA_LIMITS = {"name": 48, "evidence": 160, "method": 240, "reason": 120}

_LABELS = {
    "evidence": "evidence", "observed evidence": "evidence", "证据": "evidence", "依据": "evidence",
    "chosen model": "model", "chosen transformation": "action", "chosen method": "action",
    "method": "action", "implementation": "action", "proposed change": "action",
    "方法": "action", "实施步骤": "action", "具体方法": "action", "改动": "action",
    "reason": "reason", "rationale": "reason", "selection rationale": "reason",
    "选择原因": "reason", "选择理由": "reason", "原因": "reason", "理由": "reason",
    "expected effect": "expectation", "expected improvement": "expectation",
    "预期效果": "expectation", "预期收益": "expectation",
    "main tradeoff": "tradeoff", "tradeoff": "tradeoff", "trade-off": "tradeoff",
    "主要权衡": "tradeoff", "权衡": "tradeoff",
    "limitations": "limitations", "limitation": "limitations", "局限": "limitations",
}
_SECTIONS = re.compile(
    r"(?:^|[\n.;。；])\s*(?P<label>" + "|".join(re.escape(key) for key in _LABELS) + r")\s*[:：]\s*",
    re.IGNORECASE,
)


def validate_idea(proposal):
    """Reject malformed/verbose plans before selection; never clip coding input."""
    if not isinstance(proposal, dict) or not all(
            isinstance(proposal.get(key), str) and proposal[key].strip() for key in IDEA_FIELDS):
        raise ValueError("Idea must include nonempty name, evidence, method, and reason strings")
    if "code" in proposal:
        raise ValueError("Idea generation must not include code; coding follows selection")
    idea = {key: proposal[key].strip() for key in IDEA_FIELDS}
    for key, limit in IDEA_LIMITS.items():
        if len(idea[key]) > limit:
            raise ValueError(f"Idea {key} must be at most {limit} characters; received {len(idea[key])}")
    if any(_LABELS[match["label"].lower()] in {"evidence", "reason", "expectation", "tradeoff", "limitations"}
           for match in _SECTIONS.finditer(idea["method"])):
        raise ValueError("Idea method must contain actions only; put observed evidence in evidence "
                         "and the choice rationale in reason; omit expected effects and tradeoffs")
    return idea


def _text(value):
    return value.strip() if isinstance(value, str) and value.strip() else None


def _clip(value, limit):
    """Keep old retrieval text bounded without ending in half an English word."""
    if value is None or len(value) <= limit:
        return value
    prefix = value[:limit - 1].rstrip()
    boundary = prefix.rfind(" ")
    if boundary >= limit * .7:
        prefix = prefix[:boundary]
    return prefix + "…"


def compact_idea(node):
    """Return (idea, omitted_fields, normalization_notes) from recorded text only.

    Legacy fields can be separated only when the old text has explicit labels.
    Descriptions/expected gains are never converted to observed evidence or an
    inferred selection reason. Missing names receive a technical identifier,
    not an inferred model name. Original
    tree/idea artifacts remain the authoritative unmodified source.
    """
    source = node.get("idea") if isinstance(node.get("idea"), dict) else {}
    idea = {key: _text(source.get(key)) for key in IDEA_FIELDS}
    idea["method"] = idea["method"] or _text(node.get("method"))
    notes, omitted = {}, []
    if not all(key in source for key in IDEA_FIELDS):
        notes["source"] = "legacy"
    if not idea["name"]:
        idea["name"] = f"{str(node.get('stage', 'stage')).lower()}_node_{node['id']}"
        notes["name_generated"] = True

    method = idea["method"] or ""
    matches = list(_SECTIONS.finditer(method))
    if matches:
        actions, evidence, reasons, dropped = [], [], [], []
        # Preserve unlabeled leading text; its role cannot safely be guessed.
        lead = method[:matches[0].start()].strip()
        if lead:
            actions.append(lead)
            notes["unclassified_method_prefix"] = True
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(method)
            value = method[match.end():end].strip().rstrip(".;。；")
            kind = _LABELS[match["label"].lower()]
            if not value:
                continue
            if kind == "evidence":
                evidence.append(value)
            elif kind == "reason":
                reasons.append(value)
            elif kind in {"action", "model"}:
                actions.append("Use " + value if kind == "model" else value)
            elif kind not in dropped:
                dropped.append(kind)
        idea["method"] = "; ".join(actions) or None
        # An already separate evidence field is authoritative; don't replace it.
        if not idea["evidence"]:
            idea["evidence"] = "; ".join(evidence) or None
        if not idea["reason"]:
            idea["reason"] = "; ".join(reasons) or None
        notes["split_labeled_method"] = True
        if dropped:
            notes["omitted_sections"] = dropped
            omitted.append("idea.method.rationale_sections")
    if not idea["evidence"]:
        notes["evidence_missing"] = True
    if not idea["method"]:
        notes["method_missing"] = True
    if not idea["reason"]:
        notes["reason_missing"] = True
    if "description" in source:
        notes["description_omitted"] = True
        omitted.append("idea.description")
    for key, limit in IDEA_LIMITS.items():
        short = _clip(idea[key], limit)
        if short != idea[key]:
            omitted.append(f"idea.{key}")
        idea[key] = short
    return idea, omitted, notes
