"""Direct normalized proxy error, without relative or density transformations."""
import numpy as np
from .component_coreset import component_groups, typicality
from .unified_coreset import constrained_greedy, select as unified_select

VARIANTS = ('linear_original', 'component_random', 'direct_error')


def select(density, hardness, graph, groups, quota, radius, variant,
           seed=2030, alpha=.5, rho=.4):
    if variant in ('linear_original', 'component_random'):
        return unified_select(density, hardness, graph, groups, quota, radius,
                              'unified_c' if variant=='linear_original' else variant,
                              seed=seed, alpha=alpha, rho=rho)
    if variant != 'direct_error': raise ValueError(variant)
    hardness = np.asarray(hardness, dtype=float)
    groups = np.asarray(groups, dtype=int); quota=np.asarray(quota, dtype=int)
    if hardness.shape != (5, len(groups)) or not np.isfinite(hardness).all() or np.any((hardness<0)|(hardness>1)):
        raise ValueError('Expected 5 normalized proxy error vectors in [0,1]')
    values = hardness.mean(axis=0)
    cells = groups*4 + np.minimum(((1-typicality(radius, groups))*4).astype(int),3)
    ix, info = constrained_greedy(values, graph, groups, quota, cells, np.zeros(4*len(quota),dtype=int))
    info.update(variant=variant, final_size=len(ix), lambda_similarity=1.,
                selection_rule='mean_m normalized_error_m - 2*sum_selected_similarity/min(20,size-1); exact parent quotas; lowest pool index ties',
                error_normalization='Existing per-proxy full-Q error clipping at configured quantile followed by average percentile ranks; no further normalization',
                relative_difference=False, neighbor_smoothing=False, density_in_score=False,
                importance_mean=float(values[ix].mean()), candidate_importance_mean=float(values.mean()),
                group_allocation=quota.tolist(),
                final_group_counts=np.bincount(groups[ix],minlength=len(quota)).tolist(),
                candidate_radial_counts=np.bincount(cells,minlength=4*len(quota)).tolist(),
                final_radial_counts=np.bincount(cells[ix],minlength=4*len(quota)).tolist(),
                radial_quotas=False)
    return ix, info
