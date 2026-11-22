"""Minimum class/density coverage followed by unconstrained-density C selection."""
import numpy as np
from .coreset import greedy_select
from .multi_proxy import split_quotas, relative_density_importance
from .shared_coreset import shared_select


def coverage_select(density, hardness, graph, groups, allocation, rho=.4, seed=2030):
    if not 0 <= rho <= 1:
        raise ValueError('rho must be in [0,1]')
    density=np.asarray(density);groups=np.asarray(groups,dtype=int)
    allocation=np.asarray(allocation,dtype=int);size=int(allocation.sum())
    if rho==0:
        return shared_select(density,hardness,graph,groups,allocation,'independent_merge',seed=seed)
    values=np.array([relative_density_importance(density,hardness,graph,i,local=True)[0]
                     for i in range(len(hardness))])
    if size<2 or size>len(density):raise ValueError('Invalid final size')
    degree=min(20,size-1);coverage=int(np.floor(rho*size+.5))
    if coverage<1:raise ValueError('Positive rho rounds to empty coverage')
    # D is already a within-class percentile. Keep density ties in the same bin.
    bins=np.minimum((density*4).astype(int),3)
    joint=groups*4+bins;counts=np.bincount(joint,minlength=len(allocation)*4)
    class_floor=split_quotas(allocation,coverage,seed,lower=np.zeros(len(allocation),dtype=int),upper=allocation)
    floor=np.zeros(len(counts),dtype=int)
    for g,total in enumerate(class_floor):
        ids=np.arange(4*g,4*g+4);active=ids[counts[ids]>0]
        if total:
            floor[active]=split_quotas(counts[active],int(total),seed+g+1,
                lower=np.zeros(len(active),dtype=int),upper=counts[active])
    rng=np.random.default_rng(seed);m=len(values)
    quotas=np.full(m,coverage//m,dtype=int);quotas[rng.permutation(m)[:coverage%m]]+=1
    left=floor.copy();parts=[]
    for a,q in enumerate(quotas):
        aq=np.zeros(len(counts),dtype=int);active=np.flatnonzero(left)
        if q:
            aq[active]=split_quotas(left[active],int(q),seed+a+1,
                lower=np.zeros(len(active),dtype=int),upper=left[active])
        left-=aq
        if q==0:part=np.array([],dtype=int)
        elif q==1:
            candidates=np.flatnonzero(aq[joint]>0)
            part=np.array([candidates[np.argmax(values[a,candidates])]])
        else:
            # Multiplying every gain by M/q leaves the ordering unchanged;
            # adjust lambda to retain the FINAL k_eff even for tiny quotas.
            part,_=greedy_select(values[a],graph,joint,aq,min(20,int(q)-1)/degree,normalization_degree=20)
        parts.append(part)
    union=np.unique(np.concatenate(parts))
    if coverage==1:body=union
    else:
        body,_=greedy_select(values.mean(0),graph,joint,floor,
            min(20,coverage-1)/degree,normalization_degree=20,initial=union)
    ix,info=greedy_select(values.mean(0),graph,groups,allocation,1.,normalization_degree=20,initial=body)
    final_counts=np.bincount(joint[ix],minlength=len(counts))
    if len(np.unique(ix))!=size or np.any(final_counts<floor):raise ValueError('Coverage invariant failed')
    if not np.array_equal(np.bincount(groups[ix],minlength=len(allocation)),allocation):raise ValueError('Class allocation invariant failed')
    info.update(variant='coverage40',rho=rho,final_size=size,coverage_rows=coverage,
        density_bin_edges=[0,.25,.5,.75,1],candidate_class_bin_counts=counts.tolist(),
        minimum_class_bin_counts=floor.tolist(),final_class_bin_counts=final_counts.tolist(),
        coverage_method_quotas=quotas.tolist(),coverage_union_size=len(union),
        coverage_dedup_fill=coverage-len(union),coverage_q_indices=body.tolist(),
        independent_q_indices=[p.tolist() for p in parts],free_competition_rows=size-coverage,
        normalization_denominator=size*degree,
        selection_rule='Independent per-method C coverage proposals, dedup and fill lower quotas using mean C; global mean C completion with final class quotas and similarity to all selected samples')
    return ix,info,values
