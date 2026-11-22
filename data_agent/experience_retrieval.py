"""Training EDA metadata and MiniLM-based task selection."""

import math

from .idea_context import _eda
from .task_similarity import SimilaritySelector, structure_from_report


def eda_metadata(prefix):
    """Keep bounded observed statistics, not EDA ideas or transferable lessons."""
    eda = next((row for row in reversed(prefix)
                if row.get("stage") == "EDA" and row.get("status") == "success"), None)
    report = eda.get("result", {}).get("report") if eda else None
    summary = _eda(report)
    result = {"schema_version": 1, "scope": "train",
              "status": "unknown" if summary.get("status") == "unknown" else "observed",
              "source_node_id": eda.get("node_id", eda.get("id")) if eda else None,
              "summary": summary, "structure": structure_from_report(report)}
    counts = report.get("target_class_counts", report.get("class_counts")) if isinstance(report, dict) else None
    # Calculate from all recorded classes, before the bounded summary truncates
    # per-class entries. Invalid or absent counts remain unknown.
    if (isinstance(counts, dict) and counts
            and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in counts.values())):
        positive = [v for v in counts.values() if v > 0]
        total = sum(positive)
        if positive and math.isfinite(total):
            ratio = max(positive) / min(positive) if len(positive) > 1 else None
            result["class_balance"] = {
                "observed_classes": len(positive), "labeled_rows": total,
                "minority_fraction": min(positive) / total,
                "majority_fraction": max(positive) / total,
                "max_min_ratio": ratio if ratio is None or math.isfinite(ratio) else None,
            }
    return result


def select_similar_task(query, candidates):
    """Select one source with the default MiniLM + training EDA policy."""
    return SimilaritySelector()(query, candidates)
