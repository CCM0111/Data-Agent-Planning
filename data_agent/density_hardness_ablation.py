"""Density/hardness ablations with paired component sampling."""
import numpy as np
from scipy.stats import rankdata
from .component_coreset import component_groups

SEEDS = (20261026, 20261027, 20261028)
ETA = .2
MODES = ('uniform', 'additive', 'density', 'hardness', 'gated')
FORMULAS = {'uniform':'1', 'additive':'1+eta*(d+h-1)',
            'density':'1+eta*(2*d-1)', 'hardness':'1+eta*(2*h-1)',
            'gated':'1+eta*d*(2*h-1)'}
SETTINGS = {f'{mode}_seed{seed}': (mode, seed) for seed in SEEDS for mode in MODES}
VARIANTS = tuple(SETTINGS)


def sampling_weights(d, h, mode, eta=ETA):
    if not 0 <= eta < 1: raise ValueError('eta must be in [0,1)')
    d=np.asarray(d); h=np.asarray(h)
    if d.shape != h.shape or not np.isfinite(d).all() or not np.isfinite(h).all() or np.any((d<0)|(d>1)|(h<0)|(h>1)):
        raise ValueError('Expected aligned percentiles in [0,1]')
    signals={'uniform':np.zeros_like(d), 'additive':d+h-1,
             'density':2*d-1, 'hardness':2*h-1, 'gated':d*(2*h-1)}
    return 1+eta*signals[mode]


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
    mode, sampling_seed = SETTINGS[variant]
    eta = 0. if mode == "uniform" else ETA
    groups = np.asarray(groups, dtype=int); quota = np.asarray(quota, dtype=int)
    density = np.asarray(density, dtype=float); hardness = np.asarray(hardness, dtype=float)
    if density.shape != groups.shape or hardness.shape != (5, len(groups)):
        raise ValueError('Expected density and five proxy errors aligned to Q')
    if not np.isfinite(density).all() or not np.isfinite(hardness).all() or np.any((hardness < 0) | (hardness > 1)):
        raise ValueError('Invalid density/error values')
    d = group_percentiles(density, groups)
    h = group_percentiles(hardness.mean(0), groups)
    weights = sampling_weights(d, h, mode, eta)
    ix = weighted_sample(weights, groups, quota, sampling_seed)
    actual = np.bincount(groups[ix], minlength=len(quota))
    if not np.array_equal(actual, quota) or len(np.unique(ix)) != quota.sum():
        raise ValueError('Component quota invariant failed')
    return ix, dict(variant=variant, mode=mode, eta=eta, sampling_seed=sampling_seed,
        final_size=len(ix), group_allocation=quota.tolist(), final_group_counts=actual.tolist(),
        formula=FORMULAS[mode],
        sampling='Successive weighted sampling without replacement via exponential races; shared draws across eta',
        density_definition='Within-class/component percentile rank of existing GMM density I',
        hardness_definition='Within-class/component percentile rank of mean of five existing normalized direct errors',
        singleton_rank=.5, rank_ties='average', relative_difference=False, neighbor_smoothing=False,
        radial_quotas=False, similarity_penalty=False,
        weight_min=float(weights.min()), weight_max=float(weights.max()),
        density_percentile_selected_mean=float(d[ix].mean()),
        hardness_percentile_selected_mean=float(h[ix].mean()))
