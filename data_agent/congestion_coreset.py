"""Matched C scoring with linear and nonlinear accumulated redundancy."""
import heapq
import numpy as np
from .component_coreset import component_groups, utilities, typicality

VARIANTS = ('linear_original', 'linear_strong', 'quadratic_excess')
PARAMETERS = {'linear_original': (1., 0., .2), 'linear_strong': (2., 0., .2),
              'quadratic_excess': (1., 4., .2)}


def congestion_greedy(values, graph, groups, quota, variant):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=int)
    quota = np.asarray(quota, dtype=int)
    n = len(values); size = int(quota.sum()); k = min(20, size-1)
    if variant not in PARAMETERS or size < 2 or not np.isfinite(values).all():
        raise ValueError('Invalid variant, size, or values')
    if np.any(quota < 0) or np.any(quota > np.bincount(groups, minlength=len(quota))):
        raise ValueError('Infeasible quotas')
    lam, gamma, tau = PARAMETERS[variant]
    remaining = quota.copy(); load = np.zeros(n); selected = np.zeros(n, dtype=bool)
    versions = np.zeros(n, dtype=int)
    heap = [(-float(v/size), i, 0) for i, v in enumerate(values)]
    heapq.heapify(heap); chosen = []; loads = []; scores = []
    while len(chosen) < size:
        if not heap: raise ValueError('No feasible point')
        neg, i, version = heapq.heappop(heap)
        if selected[i] or version != versions[i] or remaining[groups[i]] == 0: continue
        selected[i] = True; remaining[groups[i]] -= 1
        chosen.append(i); loads.append(2*load[i]/k); scores.append(-neg*size)
        for j, w in zip(graph.indices[graph.indptr[i]:graph.indptr[i+1]], graph.data[graph.indptr[i]:graph.indptr[i+1]]):
            if not selected[j] and remaining[groups[j]]:
                load[j] += w; versions[j] += 1
                r = 2*load[j]/k
                # Preserve original floating-point arithmetic for gamma=0, lambda=1.
                gain = values[j]/size - 2*lam/(size*k)*load[j] - gamma/size*max(0., r-tau)**2
                heapq.heappush(heap, (-float(gain), int(j), int(versions[j])))
    ix = np.asarray(chosen)
    if not np.array_equal(np.bincount(groups[ix], minlength=len(quota)), quota):
        raise ValueError('Quota invariant failed')
    return ix, {'lambda_similarity': lam, 'gamma': gamma, 'tau': tau,
                'k_eff': k, 'negative_score_steps': int(np.sum(np.asarray(scores)<0)),
                'selected_load_at_insertion_mean': float(np.mean(loads)),
                'above_threshold_steps': int(np.sum(np.asarray(loads)>tau)),
                'selection_score_sum': float(sum(scores)),
                'score_sum_is_set_objective': False,
                'selection_rule': 'C - lambda*R - gamma*max(0,R-tau)^2; R=2*sum_selected_similarity/k_eff; ties lowest pool index'}


def select(density, hardness, graph, groups, quota, radius, variant,
           seed=2030, alpha=.5, rho=.4):
    density = np.asarray(density)
    u = utilities(hardness, graph)
    mu = (u*density).sum(1)/density.sum() if density.sum() else np.zeros(len(u))
    values = (density[None,:]*(1+.25*(u-mu[:,None]))).mean(0)
    ix, info = congestion_greedy(values, graph, groups, quota, variant)
    cells = np.asarray(groups)*4 + np.minimum(((1-typicality(radius, groups))*4).astype(int),3)
    info.update(variant=variant, eta=.25, final_size=len(ix), importance_mean=float(values[ix].mean()),
                group_allocation=np.asarray(quota).tolist(),
                final_group_counts=np.bincount(np.asarray(groups)[ix], minlength=len(quota)).tolist(),
                candidate_radial_counts=np.bincount(cells,minlength=4*len(quota)).tolist(),
                final_radial_counts=np.bincount(cells[ix],minlength=4*len(quota)).tolist(),
                radial_quotas=False, normalization_denominator=len(ix)*min(20,len(ix)-1))
    return ix, info
