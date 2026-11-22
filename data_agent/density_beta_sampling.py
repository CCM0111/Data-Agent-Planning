"""Density-led sampling with a nonnegative direct-hardness supplement."""
import numpy as np
from .density_hardness_ablation import group_percentiles, sampling_weights, component_groups, SEEDS

ETA = .2
BETAS = (.1,.2)
MODES = tuple(f'density_b{round(beta*100):02d}_l{round(lam*100):02d}' for beta in BETAS for lam in (.1,.2))
SETTINGS = {f'{mode}_seed{seed}':(mode,seed) for seed in SEEDS for mode in MODES}
VARIANTS = tuple(SETTINGS)
FORMULAS = {'weight':'1+0.2*(2*d-1)+beta*h-lambda*R', 'R':'sum selected within-group K / sum all within-group K; zero for isolated points', 'density_beta':'1+0.2*(2*d-1)+beta*h'}


from .similarity_sampling import dynamic_sample


def select(density,hardness,graph,groups,quota,radius,variant,seed=2030,alpha=.5,rho=.4):
    mode,sampling_seed=SETTINGS[variant]
    kind='density_beta';beta=int(mode.split('_b')[1].split('_')[0])/100;lam=int(mode.split('_l')[1])/100
    groups=np.asarray(groups,dtype=int);quota=np.asarray(quota,dtype=int)
    density=np.asarray(density);hardness=np.asarray(hardness)
    if density.shape!=groups.shape or hardness.shape!=(5,len(groups)) or not np.isfinite(density).all() or not np.isfinite(hardness).all() or np.any((hardness<0)|(hardness>1)):raise ValueError('Invalid density/error scores')
    d=group_percentiles(density,groups);h=group_percentiles(hardness.mean(0),groups)
    base=1+ETA*(2*d-1)+beta*h
    ix,r=dynamic_sample(base,graph,groups,quota,lam,sampling_seed)
    actual=np.bincount(groups[ix],minlength=len(quota))
    if len(np.unique(ix))!=quota.sum() or not np.array_equal(actual,quota):raise ValueError('Quota mismatch')
    return ix,dict(variant=variant,mode=mode,eta=ETA,beta=beta,lambda_similarity=lam,sampling_seed=sampling_seed,
        final_size=len(ix),group_allocation=quota.tolist(),final_group_counts=actual.tolist(),
        formula=FORMULAS[kind]+' - lambda*R',penalty_definition=FORMULAS['R'],
        selection_rule='Sequential weighted sampling without replacement, same uniform draws across arms within each group',
        radial_quotas=False,relative_difference=False,neighbor_smoothing=False,
        selection_order_pool_indices=ix.tolist(),selected_R_at_draw=r.tolist(),
        mean_R_at_draw=float(r.mean()),max_R_at_draw=float(r.max()),
        mean_penalty_at_draw=float(lam*r.mean()),minimum_possible_weight=float(base.min()-lam))
