"""Probabilistic selection with selected-value-weighted edge deductions."""
import numpy as np
from .density_hardness_ablation import group_percentiles, component_groups, SEEDS

DENSITY_COEFFICIENTS=(.2,.3)
BETAS=(.1,.2)
WEIGHT_FLOOR=1e-6
MODES=tuple(f'density_a{round(a*100):02d}_b{round(b*100):02d}_l{round(l*100):02d}'
            for a in DENSITY_COEFFICIENTS for b in BETAS for l in (.1,.2))
SETTINGS={f'{mode}_seed{seed}':(mode,seed) for seed in SEEDS for mode in MODES}
VARIANTS=tuple(SETTINGS)
FORMULAS={'initial':'1+a*(2*d-1)+beta*h',
          'update':'w_i <- max(epsilon,w_i-lambda*K_ij*w_j_at_selection), for unselected within-group neighbors i',
          'sampling':'P(i)=w_i/sum_unselected(w); no greedy argmax, no graph-degree normalization'}


def edge_sample(base,graph,groups,quota,lam,seed,floor=WEIGHT_FLOOR,mechanism='probability'):
    if mechanism not in ('probability','greedy'):raise ValueError('Unknown selection mechanism')
    base=np.asarray(base,dtype=float);groups=np.asarray(groups,dtype=int);quota=np.asarray(quota,dtype=int)
    if base.shape!=groups.shape or not np.isfinite(base).all() or not np.isfinite(lam) or lam<0 or not np.isfinite(floor) or floor<=0 or np.any(base<floor):
        raise ValueError('Invalid base weights, lambda or floor')
    if np.any(groups<0) or np.any(groups>=len(quota)) or np.any(quota<0) or np.any(quota>np.bincount(groups,minlength=len(quota))):raise ValueError('Invalid groups or quota')
    graph=graph.tocsr(copy=True);graph.sum_duplicates();graph.setdiag(0);graph.eliminate_zeros()
    if graph.shape!=(len(base),len(base)) or not np.isfinite(graph.data).all() or np.any(graph.data<0):raise ValueError('Invalid graph')
    diff=graph-graph.T
    if diff.nnz and np.max(np.abs(diff.data))>1e-12:raise ValueError('Expected symmetric graph')
    if mechanism=='probability' and seed is None:raise ValueError('Probability sampling requires a seed')
    rng=np.random.default_rng(seed) if mechanism=='probability' else None
    selected=[];at_draw=[];ever_floor=np.zeros(len(base),bool)
    edge_updates=0;floor_updates=0;raw_deduction=0.;effective_deduction=0.
    for g,q in enumerate(quota):
        ids=np.flatnonzero(groups==g)
        if not q:continue
        local=graph[ids][:,ids].tocsr();weights=base[ids].copy();used=np.zeros(len(ids),bool)
        for u in (rng.random(int(q)) if rng is not None else [None]*int(q)):
            if mechanism=='probability':
                cumulative=np.cumsum(weights);i=int(np.searchsorted(cumulative,u*cumulative[-1],side='right'))
            else:
                # Deterministic tie-break: smallest original pool index.
                tied=np.flatnonzero((~used)&(weights==weights[~used].max()))
                i=int(tied[0])
            if i>=len(ids) or used[i]:raise RuntimeError('Invalid draw')
            selected_value=float(weights[i])
            selected.append(ids[i]);at_draw.append(selected_value);used[i]=True;weights[i]=0.
            start,end=local.indptr[i:i+2];neighbors=local.indices[start:end];edges=local.data[start:end]
            mask=~used[neighbors];neighbors=neighbors[mask];edges=edges[mask]
            if not len(neighbors):continue
            before=weights[neighbors].copy();deduction=lam*edges*selected_value
            after=np.maximum(floor,before-deduction)
            hit=(before-deduction)<=floor
            ever_floor[ids[neighbors[hit]]]=True
            weights[neighbors]=after
            edge_updates+=len(neighbors);floor_updates+=int(hit.sum())
            raw_deduction+=float(deduction.sum());effective_deduction+=float((before-after).sum())
    ix=np.asarray(selected,dtype=int);draw=np.asarray(at_draw)
    return ix,dict(mechanism=mechanism,weight_floor=floor,selected_weights_at_draw=draw.tolist(),
        mean_selected_weight_at_draw=float(draw.mean()) if len(draw) else None,
        mean_selected_deduction=float((base[ix]-draw).mean()) if len(draw) else None,
        candidates_ever_at_floor=int(ever_floor.sum()),candidate_count=len(base),
        candidates_ever_at_floor_fraction=float(ever_floor.mean()) if len(base) else 0.,
        selected_at_floor=int(np.sum(draw<=floor)),edge_updates=edge_updates,floor_updates=floor_updates,
        total_raw_deduction=raw_deduction,total_effective_deduction=effective_deduction)


def select(density,hardness,graph,groups,quota,radius,variant,seed=2030,alpha=.5,rho=.4):
    mode,sampling_seed=SETTINGS[variant]
    a=int(mode.split('_a')[1].split('_')[0])/100;b=int(mode.split('_b')[1].split('_')[0])/100;lam=int(mode.split('_l')[1])/100
    groups=np.asarray(groups,dtype=int);quota=np.asarray(quota,dtype=int)
    density=np.asarray(density);hardness=np.asarray(hardness)
    if density.shape!=groups.shape or hardness.shape!=(5,len(groups)) or not np.isfinite(density).all() or not np.isfinite(hardness).all() or np.any((hardness<0)|(hardness>1)):raise ValueError('Invalid density/error scores')
    d=group_percentiles(density,groups);h=group_percentiles(hardness.mean(0),groups)
    base=1+a*(2*d-1)+b*h
    ix,info=edge_sample(base,graph,groups,quota,lam,sampling_seed)
    actual=np.bincount(groups[ix],minlength=len(quota))
    if len(np.unique(ix))!=quota.sum() or not np.array_equal(actual,quota):raise ValueError('Quota mismatch')
    info.update(variant=variant,mode=mode,eta=a,density_coefficient=a,beta=b,lambda_similarity=lam,sampling_seed=sampling_seed,
        final_size=len(ix),group_allocation=quota.tolist(),final_group_counts=actual.tolist(),
        formula=FORMULAS,selection_order_pool_indices=ix.tolist(),
        radial_quotas=False,relative_difference=False,neighbor_smoothing=False,
        degree_normalization=False,graph_definition='Existing frozen similarity K, restricted to class/component; no new distance kernel',
        minimum_possible_weight=WEIGHT_FLOOR)
    return ix,info
