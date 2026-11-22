"""Single-stage mean-C selection under component quotas and radial lower bounds."""
import heapq
import numpy as np
from .component_coreset import allocate, component_groups, utilities, typicality
from .coreset import stratified_sample, objective

VARIANTS=('unified_c','unified_c_coverage','component_random')


def constrained_greedy(values, graph, groups, quota, cells, floors):
    values=np.asarray(values);groups=np.asarray(groups);quota=np.asarray(quota)
    cells=np.asarray(cells);floors=np.asarray(floors)
    size=int(quota.sum());remaining=quota.copy();deficit=floors.copy()
    if np.any(floors<0) or np.any(floors>np.bincount(cells,minlength=len(floors))):
        raise ValueError('Infeasible cell floors')
    needed=floors.reshape(-1,4).sum(1)
    if np.any(needed>remaining) or np.any(quota>np.bincount(groups,minlength=len(quota))):
        raise ValueError('Infeasible group quotas')
    selected=np.zeros(len(values),dtype=bool);versions=np.zeros(len(values),dtype=int)
    similarity=np.zeros(len(values));degree=min(20,size-1)
    factor=2/(size*degree) if degree else 0.
    heap=[(-float(v/size),i,0) for i,v in enumerate(values)];heapq.heapify(heap)
    chosen=[];gains=[];blocked=0
    while len(chosen)<size:
        if not heap:raise ValueError('No feasible candidate')
        neg,i,version=heapq.heappop(heap);g=groups[i];c=cells[i]
        if selected[i] or version!=versions[i] or remaining[g]==0:continue
        # When slack reaches zero, cells with no deficit stay infeasible forever.
        if deficit[c]==0 and remaining[g]==needed[g]:
            blocked+=1;continue
        selected[i]=True;chosen.append(i);gains.append(-neg);remaining[g]-=1
        if deficit[c]>0:deficit[c]-=1;needed[g]-=1
        start,end=graph.indptr[i:i+2]
        for j,w in zip(graph.indices[start:end],graph.data[start:end]):
            if not selected[j] and remaining[groups[j]]:
                similarity[j]+=w;versions[j]+=1
                heapq.heappush(heap,(-float(values[j]/size-factor*similarity[j]),int(j),int(versions[j])))
    ix=np.asarray(chosen,dtype=int)
    actual=np.bincount(cells[ix],minlength=len(floors))
    if np.any(deficit) or np.any(remaining) or np.any(actual<floors):raise ValueError('Coverage invariant failed')
    value=objective(values,graph,ix,size,1.,normalization_degree=20) if size>1 else float(values[ix].sum())
    if not np.isclose(sum(gains),value,rtol=1e-9,atol=1e-10):raise ValueError('Objective/gain mismatch')
    return ix,{'objective':value,'gain_sum':float(sum(gains)),'coverage_blocked_pops':blocked,
               'negative_gain_steps':int(np.sum(np.asarray(gains)<0))}


def select(density,hardness,graph,groups,quota,radius,variant,seed=2030,alpha=.5,rho=.4):
    if variant not in VARIANTS or not 0<=rho<=1:raise ValueError('Invalid variant/rho')
    groups=np.asarray(groups,dtype=int);quota=np.asarray(quota,dtype=int);size=int(quota.sum())
    t=typicality(radius,groups);cells=groups*4+np.minimum(((1-t)*4).astype(int),3)
    counts=np.bincount(cells,minlength=4*len(quota));floors=np.zeros_like(counts)
    if variant=='unified_c_coverage':
        group_floor=allocate(quota,int(np.floor(rho*size+.5)),seed+100)
        for g,q in enumerate(group_floor):
            sl=slice(g*4,g*4+4);floors[sl]=allocate(counts[sl],int(q),seed+101+g)
    if variant=='component_random':
        ix=stratified_sample(groups,quota,20261026)
        info={'seed':20261026,'selection_rule':'Uniform without replacement within identical component quota groups'}
    else:
        u=utilities(hardness,graph);density=np.asarray(density)
        mu=(u*density).sum(1)/density.sum() if density.sum() else np.zeros(len(u))
        values=(density[None,:]*(1+.25*(u-mu[:,None]))).mean(0)
        ix,info=constrained_greedy(values,graph,groups,quota,cells,floors)
        info.update(selection_rule='Single-stage mean C feasible greedy; no initial body or score reranking',eta=.25,
                    importance_mean=float(values[ix].mean()),rho=rho if variant=='unified_c_coverage' else 0.)
    actual=np.bincount(groups[ix],minlength=len(quota))
    if len(np.unique(ix))!=size or not np.array_equal(actual,quota):raise ValueError('Group quota invariant failed')
    info.update(variant=variant,group_allocation=quota.tolist(),final_group_counts=actual.tolist(),
                final_size=size,minimum_radial_counts=floors.tolist(),candidate_radial_counts=counts.tolist(),
                final_radial_counts=np.bincount(cells[ix],minlength=len(floors)).tolist(),
                normalization_denominator=size*min(20,size-1))
    return ix,info
