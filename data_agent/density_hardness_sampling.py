"""Bounded density/error tilting of within-component random sampling."""
import numpy as np
from scipy.stats import rankdata
from .component_coreset import component_groups

SEEDS = (20261026, 20261027, 20261028)
ETAS = (0., .1, .2, .3)
SETTINGS = {f'eta{round(eta*100):02d}_seed{seed}': (eta, seed)
            for seed in SEEDS for eta in ETAS}
VARIANTS = tuple(SETTINGS)


def group_percentiles(values, groups):
    values = np.asarray(values, dtype=float)
    out = np.empty(len(values))
    for g in np.unique(groups):
        mask = groups == g; n = int(mask.sum())
        out[mask] = (rankdata(values[mask], method='average') - 1) / (n - 1) if n > 1 else .5
    return out


def weighted_sample(weights, groups, quota, seed):
    """Exponential races implement successive weighted sampling without replacement.

    Shared exponential draws across eta provide paired sampling randomness.
    """
    weights = np.asarray(weights, dtype=float); groups = np.asarray(groups, dtype=int)
    quota = np.asarray(quota, dtype=int)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Weights must be positive and finite')
    if weights.shape != groups.shape or np.any(groups < 0) or np.any(groups >= len(quota)):
        raise ValueError('Invalid groups')
    counts = np.bincount(groups, minlength=len(quota))
    if np.any(quota < 0) or np.any(quota > counts): raise ValueError('Invalid quota')
    races = np.random.default_rng(seed).exponential(size=len(groups)) / weights
    selected = []
    for g, q in enumerate(quota):
        members = np.flatnonzero(groups == g)
        selected.extend(members[np.lexsort((members, races[members]))[:q]])
    return np.array(selected, dtype=int)


def select(density, hardness, graph, groups, quota, radius, variant,
           seed=2030, alpha=.5, rho=.4):
    eta, sampling_seed = SETTINGS[variant]
    groups = np.asarray(groups, dtype=int); quota = np.asarray(quota, dtype=int)
    density = np.asarray(density, dtype=float); hardness = np.asarray(hardness, dtype=float)
    if density.shape != groups.shape or hardness.shape != (5, len(groups)):
        raise ValueError('Expected density and five proxy errors aligned to Q')
    if not np.isfinite(density).all() or not np.isfinite(hardness).all() or np.any((hardness < 0) | (hardness > 1)):
        raise ValueError('Invalid density/error values')
    d = group_percentiles(density, groups)
    h = group_percentiles(hardness.mean(0), groups)
    weights = 1 + eta * (d + h - 1)
    ix = weighted_sample(weights, groups, quota, sampling_seed)
    actual = np.bincount(groups[ix], minlength=len(quota))
    if not np.array_equal(actual, quota) or len(np.unique(ix)) != quota.sum():
        raise ValueError('Component quota invariant failed')
    return ix, dict(variant=variant, eta=eta, sampling_seed=sampling_seed,
        final_size=len(ix), group_allocation=quota.tolist(), final_group_counts=actual.tolist(),
        formula='w=1+eta*(within_group_rank(D)+within_group_rank(mean_m H_m)-1)',
        sampling='Successive weighted sampling without replacement via exponential races; shared draws across eta',
        density_definition='Within-class/component percentile rank of existing GMM density I',
        hardness_definition='Within-class/component percentile rank of mean of five existing normalized direct errors',
        singleton_rank=.5, rank_ties='average', relative_difference=False, neighbor_smoothing=False,
        radial_quotas=False, similarity_penalty=False,
        weight_min=float(weights.min()), weight_max=float(weights.max()),
        density_percentile_selected_mean=float(d[ix].mean()),
        hardness_percentile_selected_mean=float(h[ix].mean()))
