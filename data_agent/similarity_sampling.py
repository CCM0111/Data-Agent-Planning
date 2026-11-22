"""Bounded accumulated similarity penalty for probabilistic coreset selection."""
import numpy as np
from .density_hardness_ablation import group_percentiles, sampling_weights, component_groups, SEEDS, ETA

MODES = ('uniform',)+tuple(f'{mode}_l{round(lam*100):02d}' for mode in ('density','hardness','gated') for lam in (0.,.1,.2))
SETTINGS = {f'{mode}_seed{seed}':(mode,seed) for seed in SEEDS for mode in MODES}
VARIANTS = tuple(SETTINGS)
FORMULAS = {'weight':'base_weight-lambda*R', 'R':'sum selected within-group K / sum all within-group K; zero for isolated points',
            'density':'1+.2*(2*d-1)', 'hardness':'1+.2*(2*h-1)', 'gated':'1+.2*d*(2*h-1)', 'uniform':'1'}


def dynamic_sample(base, graph, groups, quota, lam, seed):
    base=np.asarray(base,dtype=float);groups=np.asarray(groups,dtype=int);quota=np.asarray(quota,dtype=int)
    if not 0<=lam<=.2 or base.shape!=groups.shape or not np.isfinite(base).all() or np.any(base<=lam):raise ValueError('Invalid weights or lambda')
    if np.any(quota<0) or np.any(quota>np.bincount(groups,minlength=len(quota))):raise ValueError('Invalid quotas')
    graph=graph.tocsr(copy=True);graph.setdiag(0);graph.eliminate_zeros()
    if graph.shape!=(len(base),len(base)) or not np.isfinite(graph.data).all() or np.any(graph.data<0):raise ValueError('Invalid graph')
    diff=graph-graph.T
    if diff.nnz and np.max(np.abs(diff.data))>1e-12:raise ValueError('Graph must be symmetric')
    rng=np.random.default_rng(seed);chosen=[];penalties=[]
    for g,q in enumerate(quota):
        ids=np.flatnonzero(groups==g)
        if not q:continue
        local=graph[ids][:,ids].tocsr();degree=np.asarray(local.sum(axis=1)).ravel()
        load=np.zeros(len(ids));used=np.zeros(len(ids),bool)
        for u in rng.random(int(q)):
            r=np.divide(load,degree,out=np.zeros_like(load),where=degree>0)
            r=np.clip(r,0,1);weights=base[ids]-lam*r;weights[used]=0
            cumulative=np.cumsum(weights);target=u*cumulative[-1]
            i=int(np.searchsorted(cumulative,target,side='right'))
            if i>=len(ids) or used[i]:raise RuntimeError('Invalid weighted draw')
            chosen.append(ids[i]);penalties.append(r[i]);used[i]=True
            a,b=local.indptr[i:i+2];np.add.at(load,local.indices[a:b],local.data[a:b])
    return np.array(chosen,dtype=int), np.array(penalties)


def select(density,hardness,graph,groups,quota,radius,variant,seed=2030,alpha=.5,rho=.4):
    mode,sampling_seed=SETTINGS[variant]
    kind,lam=('uniform',0.) if mode=='uniform' else (mode.split('_l')[0],int(mode.split('_l')[1])/100)
    groups=np.asarray(groups,dtype=int);quota=np.asarray(quota,dtype=int)
    density=np.asarray(density);hardness=np.asarray(hardness)
    if density.shape!=groups.shape or hardness.shape!=(5,len(groups)) or not np.isfinite(density).all() or not np.isfinite(hardness).all() or np.any((hardness<0)|(hardness>1)):raise ValueError('Invalid density/error scores')
    d=group_percentiles(density,groups);h=group_percentiles(hardness.mean(0),groups)
    base=sampling_weights(d,h,kind,ETA)
    ix,r=dynamic_sample(base,graph,groups,quota,lam,sampling_seed)
    actual=np.bincount(groups[ix],minlength=len(quota))
    if len(np.unique(ix))!=quota.sum() or not np.array_equal(actual,quota):raise ValueError('Quota mismatch')
    return ix,dict(variant=variant,mode=mode,eta=ETA,lambda_similarity=lam,sampling_seed=sampling_seed,
        final_size=len(ix),group_allocation=quota.tolist(),final_group_counts=actual.tolist(),
        formula=FORMULAS[kind]+' - lambda*R',penalty_definition=FORMULAS['R'],
        selection_rule='Sequential weighted sampling without replacement, same uniform draws across arms within each group',
        radial_quotas=False,relative_difference=False,neighbor_smoothing=False,
        selection_order_pool_indices=ix.tolist(),selected_R_at_draw=r.tolist(),
        mean_R_at_draw=float(r.mean()),max_R_at_draw=float(r.max()),
        mean_penalty_at_draw=float(lam*r.mean()),minimum_possible_weight=float(base.min()-lam))
