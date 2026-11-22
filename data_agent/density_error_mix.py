"""Exact radial quotas split between density and direct proxy error."""
import heapq
import numpy as np
from .component_coreset import component_groups
from .radial_quota_coreset import radial_quotas

RATIOS = {'density80_error20': .8, 'density70_error30': .7,
          'density60_error40': .6, 'density50_error50': .5}
VARIANTS = (*RATIOS, 'radial_quota_random', 'component_random')


def mixed_greedy(values, graph, cells, allocation, ratio, tie_seed=20261026):
    """Two lazy greedy queues; both roles penalize the same selected set."""
    values = np.asarray(values, dtype=float)
    cells = np.asarray(cells, dtype=int)
    allocation = np.asarray(allocation, dtype=int)
    n = len(cells); size = int(allocation.sum())
    if values.shape != (2, n) or not np.isfinite(values).all():
        raise ValueError('Expected two finite value vectors')
    if not 0 <= ratio <= 1 or np.any(allocation < 0) or np.any(allocation > np.bincount(cells, minlength=len(allocation))):
        raise ValueError('Invalid role ratio or allocation')
    graph = graph.tocsr()
    if graph.shape != (n, n) or not np.isfinite(graph.data).all() or np.any(graph.data < 0):
        raise ValueError('Expected finite nonnegative similarity graph')
    if (graph - graph.T).nnz and np.max(np.abs((graph - graph.T).data)) > 1e-12:
        raise ValueError('Expected symmetric graph')
    density_quota = np.floor(ratio * allocation + .5).astype(int)
    remaining = np.array([density_quota, allocation - density_quota])
    planned = remaining.copy(); density_total = int(density_quota.sum())
    order = np.random.default_rng(tie_seed).permutation(n)
    tie = np.empty(n, dtype=int); tie[order] = np.arange(n)
    heaps = [[(-float(values[r, i]), int(tie[i]), i) for i in range(n)
              if remaining[r, cells[i]] > 0] for r in range(2)]
    for heap in heaps: heapq.heapify(heap)
    used = np.zeros(n, dtype=bool); load = np.zeros(n)
    selected = []; roles = []; density_count = 0
    scale = 2 / min(20, size - 1) if size > 1 else 0.
    for step in range(size):
        target = ((step + 1) * density_total * 2 + size) // (2 * size)
        role = 0 if density_count < target else 1
        heap = heaps[role]
        while heap:
            entry = heapq.heappop(heap); i = entry[2]
            if used[i] or remaining[role, cells[i]] == 0: continue
            current = (-float(values[role, i] - scale * load[i]), int(tie[i]), i)
            # Stale scores are upper bounds because similarity loads only increase.
            if heap and current > heap[0]:
                heapq.heappush(heap, current); continue
            break
        else: raise RuntimeError('Role quota exhausted prematurely')
        selected.append(i); roles.append(role); used[i] = True
        remaining[role, cells[i]] -= 1; density_count += role == 0
        a, b = graph.indptr[i:i+2]
        np.add.at(load, graph.indices[a:b], graph.data[a:b])
    if remaining.any(): raise RuntimeError('Unfilled role quotas')
    return np.array(selected, dtype=int), np.array(roles, dtype=int), planned


def select(density, hardness, graph, groups, quota, radius, variant,
           seed=2030, alpha=.5, rho=.4):
    if variant not in RATIOS: raise ValueError(variant)
    density = np.asarray(density, dtype=float); hardness = np.asarray(hardness, dtype=float)
    groups = np.asarray(groups, dtype=int); quota = np.asarray(quota, dtype=int)
    if hardness.shape != (5, len(groups)) or not np.isfinite(hardness).all() or np.any((hardness < 0) | (hardness > 1)):
        raise ValueError('Expected five normalized direct errors')
    cells, counts, allocation = radial_quotas(groups, quota, radius, seed)
    ix, roles, planned = mixed_greedy(np.array([density, hardness.mean(0)]), graph, cells, allocation, RATIOS[variant])
    actual = np.bincount(cells[ix], minlength=len(allocation))
    if not np.array_equal(actual, allocation) or len(np.unique(ix)) != quota.sum():
        raise ValueError('Radial quota invariant failed')
    return ix, dict(variant=variant, final_size=len(ix), group_allocation=quota.tolist(),
        final_group_counts=np.bincount(groups[ix], minlength=len(quota)).tolist(),
        candidate_radial_counts=counts.tolist(), radial_allocation=allocation.tolist(),
        final_radial_counts=actual.tolist(), radial_bins=4, density_fraction=RATIOS[variant],
        density_role_allocation=planned[0].tolist(), hardness_role_allocation=planned[1].tolist(),
        final_role_counts=np.bincount(roles, minlength=2).tolist(),
        selection_order_pool_indices=ix.tolist(), selection_roles=roles.tolist(),
        lambda_density=1., lambda_hardness=1., relative_difference=False, neighbor_smoothing=False,
        density_mean=float(density[ix].mean()), hardness_mean=float(hardness[:, ix].mean()),
        selection_rule='Exact radial quotas; per-cell density count floor(ratio*quota+0.5); remaining direct-error count; globally interleaved roles by cumulative rounded role totals; shared S',
        scores='density I - R or mean of five normalized direct errors - R; R=2*sum_selected K/min(20,B-1)',
        tie_seed=20261026)
