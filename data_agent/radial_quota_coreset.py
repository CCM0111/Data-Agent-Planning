"""Exact class/component/radial quotas with relative utility and diversity."""
import numpy as np
from .component_coreset import allocate, component_groups, typicality, utilities
from .coreset import greedy_select, stratified_sample

VARIANTS = ('radial_quota_random', 'radial_quota_utility')


def radial_quotas(groups, quota, radius, seed):
    groups = np.asarray(groups, dtype=int)
    quota = np.asarray(quota, dtype=int)
    t = typicality(radius, groups)
    cells = groups*4 + np.minimum(((1-t)*4).astype(int), 3)
    counts = np.bincount(cells, minlength=4*len(quota))
    allocation = np.zeros_like(counts)
    for g, q in enumerate(quota):
        sl = slice(g*4, g*4+4)
        allocation[sl] = allocate(counts[sl], int(q), seed+101+g)
    return cells, counts, allocation


def select(density, hardness, graph, groups, quota, radius, variant,
           seed=2030, alpha=.5, rho=.4):
    if variant not in VARIANTS:
        raise ValueError(variant)
    groups = np.asarray(groups, dtype=int)
    quota = np.asarray(quota, dtype=int)
    size = int(quota.sum())
    cells, counts, allocation = radial_quotas(groups, quota, radius, seed)
    if variant == 'radial_quota_random':
        ix = stratified_sample(cells, allocation, 20261026)
        info = {'selection_rule': 'Uniform within exact class/component/radial quotas',
                'random_seed': 20261026}
    else:
        values = utilities(hardness, graph).mean(0)
        # Randomize only tie order, preserving original indices in the output.
        permutation = np.random.default_rng(20261026).permutation(len(groups))
        if size < 2:
            candidates = permutation[allocation[cells[permutation]] > 0]
            ix = candidates[np.argmax(values[candidates]) : np.argmax(values[candidates])+1]
            info = {}
        else:
            local, info = greedy_select(values[permutation], graph[permutation][:, permutation],
                                        cells[permutation], allocation, 1., normalization_degree=20)
            ix = permutation[local]
        info.update(selection_rule='Mean relative utility minus incremental graph similarity penalty; exact radial quotas',
                    lambda_similarity=1., utility_aggregation='arithmetic mean of 5 neighbor-smoothed relative utilities',
                    utility_mean=float(values[ix].mean()), utility_candidate_mean=float(values.mean()),
                    tie_seed=20261026)
    actual = np.bincount(cells[ix], minlength=len(allocation))
    if len(np.unique(ix)) != size or not np.array_equal(actual, allocation):
        raise ValueError('Exact radial quota invariant failed')
    if not np.array_equal(np.bincount(groups[ix], minlength=len(quota)), quota):
        raise ValueError('Parent component quota invariant failed')
    info.update(variant=variant, final_size=size, group_allocation=quota.tolist(),
                final_group_counts=np.bincount(groups[ix], minlength=len(quota)).tolist(),
                candidate_radial_counts=counts.tolist(), radial_allocation=allocation.tolist(),
                final_radial_counts=actual.tolist(), radial_bins=4,
                radius_ties='average ranks; equal-radius observations stay in the same bin',
                density_used_in_score=False)
    return ix, info
