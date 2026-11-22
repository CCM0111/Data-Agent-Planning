"""Fixed coverage ablations: weaker hardness versus mean-preserving density flattening."""
import numpy as np
from .component_coreset import allocate, component_groups, utilities, typicality
from .unified_coreset import constrained_greedy
from .coreset import stratified_sample

VARIANTS=('coverage_weak_hardness','coverage_weak_density')


def score_values(density, hardness, graph, eta, power):
    density=np.asarray(density,dtype=float)
    u=utilities(hardness,graph)
    flat=density**power
    scale=density.sum()/flat.sum() if flat.sum() else 1.
    flat=flat*scale
    mu=(u*flat).sum(1)/flat.sum() if flat.sum() else np.zeros(len(u))
    values=(flat[None,:]*(1+eta*(u-mu[:,None]))).mean(0)
    return values, {'eta':eta,'density_power':power,'density_scale':float(scale),
                    'density_original_mean':float(density.mean()),'density_transformed_mean':float(flat.mean()),
                    'centering':'per-proxy U mean weighted by transformed density over full Q',
                    'candidate_score_mean':float(values.mean())}


def select(density,hardness,graph,groups,quota,radius,variant,seed=2030,alpha=.5,rho=.4):
    if variant not in VARIANTS or not 0<=rho<=1:raise ValueError('Invalid variant/rho')
    groups=np.asarray(groups,dtype=int);quota=np.asarray(quota,dtype=int);size=int(quota.sum())
    t=typicality(radius,groups);cells=groups*4+np.minimum(((1-t)*4).astype(int),3)
    counts=np.bincount(cells,minlength=4*len(quota));floors=np.zeros_like(counts)
    if variant in VARIANTS:
        group_floor=allocate(quota,int(np.floor(rho*size+.5)),seed+100)
        for g,q in enumerate(group_floor):
            sl=slice(g*4,g*4+4);floors[sl]=allocate(counts[sl],int(q),seed+101+g)
    eta=.1 if variant=='coverage_weak_hardness' else .25
    power=.5 if variant=='coverage_weak_density' else 1.
    values,score_info=score_values(density,hardness,graph,eta,power)
    ix,info=constrained_greedy(values,graph,groups,quota,cells,floors)
    info.update(score_info)
    info.update(selection_rule='Single-stage mean C feasible greedy; fixed radial floors; no reranking',
                importance_mean=float(values[ix].mean()),rho=rho)
    actual=np.bincount(groups[ix],minlength=len(quota))
    if len(np.unique(ix))!=size or not np.array_equal(actual,quota):raise ValueError('Group quota invariant failed')
    info.update(variant=variant,group_allocation=quota.tolist(),final_group_counts=actual.tolist(),
                final_size=size,minimum_radial_counts=floors.tolist(),candidate_radial_counts=counts.tolist(),
                final_radial_counts=np.bincount(cells[ix],minlength=len(floors)).tolist(),
                normalization_denominator=size*min(20,size-1))
    return ix,info
