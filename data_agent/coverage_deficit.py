"""Greedy coverage-deficit times relative-hardness sampling.

No hard component quotas or extra similarity penalty. The graph is used only
in the existing relative-hardness utility. Ties choose the lowest pool index.
"""
import numpy as np
from .component_coreset import component_groups, typicality, utilities

VARIANTS = ('coverage_deficit',)


def deficit_greedy(utility, cells, size, eta=.25):
    utility = np.asarray(utility, dtype=float)
    cells = np.asarray(cells, dtype=int)
    n = len(utility)
    if n == 0 or cells.shape != (n,) or np.any(cells < 0):
        raise ValueError('Invalid cells')
    if not 0 <= size <= n or eta < 0 or not np.isfinite(eta):
        raise ValueError('Invalid size or eta')
    if not np.isfinite(utility).all() or np.any((utility < 0) | (utility > 1)):
        raise ValueError('Expected finite utilities in [0,1]')
    counts = np.bincount(cells)
    mass = counts / n
    # Within a cell the deficit is shared; its best remaining utility wins.
    # eta=0 must use pool-index tie breaking, independent of utility.
    order = np.lexsort((np.arange(n), -utility if eta else np.zeros(n), cells))
    ends = np.cumsum(counts)
    starts = ends - counts
    ptr = starts.copy()
    selected_counts = np.zeros(len(counts), dtype=int)
    chosen = []
    positive_gaps = []
    for step in range(size):
        active = np.flatnonzero(ptr < ends)
        candidates = order[ptr[active]]
        gap = mass[active] - selected_counts[active] / (step + 1)
        positive = gap > 0
        if not positive.any():
            raise ValueError('No remaining positive-deficit cell')
        active, candidates, gap = active[positive], candidates[positive], gap[positive]
        scores = gap * (1 + eta * utility[candidates])
        best = np.lexsort((candidates, -scores))[0]
        cell = active[best]
        chosen.append(int(candidates[best]))
        positive_gaps.append(float(gap[best]))
        ptr[cell] += 1
        selected_counts[cell] += 1
    return np.asarray(chosen, dtype=int), {
        'eta': eta, 'minimum_selected_deficit': min(positive_gaps) if size else None,
        'candidate_radial_counts': counts.tolist(),
        'final_radial_counts': selected_counts.tolist(),
        'radial_mass_total_variation': float(np.abs(selected_counts / size - mass).sum()/2) if size else None,
    }


def select(density, hardness, graph, groups, quota, radius, variant,
           seed=2030, alpha=.5, rho=.4):
    if variant not in VARIANTS:
        raise ValueError(variant)
    groups = np.asarray(groups, dtype=int)
    size = int(np.sum(quota))
    t = typicality(radius, groups)
    cells = groups * 4 + np.minimum(((1-t)*4).astype(int), 3)
    u = utilities(hardness, graph).mean(0)
    ix, info = deficit_greedy(u, cells, size, eta=.25)
    info.update(variant=variant, final_size=size,
                selection_rule='argmax positive (N_cell/N - selected_cell/(step+1)) * (1+.25*mean_U); lowest pool index breaks ties',
                final_group_counts=np.bincount(groups[ix], minlength=len(quota)).tolist(),
                reference_group_allocation=np.asarray(quota).tolist(),
                selected_utility_mean=float(u[ix].mean()),
                selected_typicality_mean=float(t[ix].mean()),
                hard_quotas=False, extra_similarity_penalty=False)
    return ix, info
