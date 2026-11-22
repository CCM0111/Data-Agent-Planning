"""Paired-bootstrap MCS range elimination for already computed method scores.

The inputs are method-level comparison quantities for which smaller is better.
This module neither samples data nor computes a task metric; callers can use the
same implementation for sample-mean losses and metric-level bootstrap values.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def mcs_range(
    theta: np.ndarray,
    bootstrap_theta: np.ndarray,
    method_ids: Sequence[str | int],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute the complete range-elimination path and its confidence set.

    ``theta`` has shape (methods,), and ``bootstrap_theta`` has shape
    (replicates, methods). The bootstrap values must already use the same paired
    resamples across methods and fixed training conditions. Array columns retain
    the input order; exact elimination ties use the lexical method-ID order.

    Standard errors use ``ddof=1``. An exactly identical pair has zero t/tau.
    Other pairs with SE <= 64 * machine-epsilon * max(1, |d|, max |delta|)
    invalidate elimination and conservatively retain every candidate. This
    tolerance detects numerically constant differences without replacing the
    denominator. Undefined t/tau entries are NaN and MCS p-values are None.
    """
    theta = np.asarray(theta, dtype=np.float64)
    bootstrap_theta = np.asarray(bootstrap_theta, dtype=np.float64)
    method_ids = list(method_ids)
    if theta.ndim != 1 or not len(theta):
        raise ValueError("theta must be a nonempty one-dimensional array")
    if bootstrap_theta.ndim != 2 or bootstrap_theta.shape[1] != len(theta):
        raise ValueError("bootstrap_theta must have shape (replicates, methods)")
    if bootstrap_theta.shape[0] < 2:
        raise ValueError("at least two bootstrap replicates are required for ddof=1")
    if len(method_ids) != len(theta) or len(set(method_ids)) != len(method_ids):
        raise ValueError("method_ids must be unique and match theta")
    if len({str(method_id) for method_id in method_ids}) != len(method_ids):
        raise ValueError("method_ids must also have unique string representations")
    if not np.isfinite(theta).all() or not np.isfinite(bootstrap_theta).all():
        raise ValueError("comparison quantities must be finite")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")

    n_methods = len(method_ids)
    n_replicates = len(bootstrap_theta)
    with np.errstate(over="ignore", invalid="ignore"):
        difference = theta[:, None] - theta[None, :]
        bootstrap_difference = bootstrap_theta[:, :, None] - bootstrap_theta[:, None, :]
        standard_error = bootstrap_difference.std(axis=0, ddof=1)
        bootstrap_standard_error = bootstrap_theta.std(axis=0, ddof=1)
    t_statistic = np.full((n_methods, n_methods), np.nan)
    centered_bootstrap = np.full_like(bootstrap_difference, np.nan)
    np.fill_diagonal(t_statistic, 0.0)
    for i in range(n_methods):
        centered_bootstrap[:, i, i] = 0.0

    degenerate_pairs = []
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            d = difference[i, j]
            delta = bootstrap_difference[:, i, j]
            se = standard_error[i, j]
            if d == 0.0 and np.all(delta == 0.0):
                t_statistic[i, j] = t_statistic[j, i] = 0.0
                centered_bootstrap[:, i, j] = centered_bootstrap[:, j, i] = 0.0
                continue
            scale = max(1.0, abs(d), float(np.max(np.abs(delta))))
            tolerance = 64 * np.finfo(np.float64).eps * scale
            if not np.isfinite(d) or not np.isfinite(se) or se <= tolerance:
                degenerate_pairs.append({
                    "method_i": method_ids[i],
                    "method_j": method_ids[j],
                    "mean_loss_difference": float(d),
                    "standard_error": float(se),
                    "zero_se_tolerance": float(tolerance),
                })
                continue
            t_statistic[i, j] = d / se
            t_statistic[j, i] = -t_statistic[i, j]
            centered_bootstrap[:, i, j] = (delta - d) / se
            centered_bootstrap[:, j, i] = -centered_bootstrap[:, i, j]

    result = {
        "method_ids": method_ids,
        "mean_loss_difference": difference,
        "standard_error": standard_error,
        "t_statistic": t_statistic,
        "centered_bootstrap_statistics": centered_bootstrap,
        "bootstrap_standard_error": bootstrap_standard_error,
        "mcs_pvalues": {method_id: None for method_id in method_ids},
        "survivor_method_ids": list(method_ids),
        "inactive_method_ids": [],
        "elimination_trace": [],
        "degenerate_pairs": degenerate_pairs,
        "selection_status": "statistical_degeneracy" if degenerate_pairs else "completed",
        "selection_reason": (
            "A nonidentical method pair has a zero, numerically near-zero, or nonfinite "
            "standard error; all candidates are retained."
            if degenerate_pairs else "MCS range elimination completed using centered paired-bootstrap statistics."
        ),
    }
    if degenerate_pairs:
        return result

    current = sorted(range(n_methods), key=lambda i: str(method_ids[i]))
    previous_pvalue = 0.0
    while len(current) > 1:
        pairs = [(i, j) for i in current for j in current if i != j]
        max_statistic = max(abs(t_statistic[i, j]) for i, j in pairs)
        bootstrap_max = np.max(
            np.stack([np.abs(centered_bootstrap[:, i, j]) for i, j in pairs]), axis=0
        )
        raw_pvalue = float((1 + np.count_nonzero(bootstrap_max >= max_statistic)) / (n_replicates + 1))
        adjusted_pvalue = max(previous_pvalue, raw_pvalue)
        elimination_statistics = {
            i: max(t_statistic[i, j] for j in current if i != j) for i in current
        }
        # Python's max retains the first item on an exact tie, and current is
        # ordered by method ID rather than incidental input column order.
        removed = max(current, key=elimination_statistics.__getitem__)
        result["elimination_trace"].append({
            "round": len(result["elimination_trace"]) + 1,
            "alpha": float(alpha),
            "significantly_worse": adjusted_pvalue < alpha,
            "decision": "significantly_worse" if adjusted_pvalue < alpha else "retained_at_alpha",
            "reason": ("Largest worst-pair standardized loss difference among current methods; "
                       "adjusted p-value < alpha." if adjusted_pvalue < alpha else
                       "Full MCS ordering continues for p-values; adjusted p-value >= alpha, "
                       "so this ordering removal is NOT a significant elimination."),
            "current_method_ids": [method_ids[i] for i in current],
            "removed_method_id": method_ids[removed],
            "max_statistic": float(max_statistic),
            "raw_pvalue": raw_pvalue,
            "adjusted_pvalue": adjusted_pvalue,
            "elimination_statistics": {
                method_ids[i]: float(elimination_statistics[i]) for i in current
            },
        })
        result["mcs_pvalues"][method_ids[removed]] = adjusted_pvalue
        current.remove(removed)
        previous_pvalue = adjusted_pvalue

    result["mcs_pvalues"][method_ids[current[0]]] = 1.0
    result["survivor_method_ids"] = [
        method_id for method_id in method_ids if result["mcs_pvalues"][method_id] >= alpha
    ]
    result["inactive_method_ids"] = [
        method_id for method_id in method_ids if result["mcs_pvalues"][method_id] < alpha
    ]
    return result
