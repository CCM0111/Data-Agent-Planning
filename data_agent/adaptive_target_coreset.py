"""Target-quantile quotas with frozen component scores and cross-bin edge penalties."""
import numpy as np
from .coreset import strata
from .component_coreset import allocate


def target_component_quotas(target,components,size,k=5,seed=2039,min_per_cell=50,max_bins=10):
    components=np.asarray(components,int)
    if size<1 or size>len(target) or components.shape!=(len(target),) or np.any((components<0)|(components>=k)):
        raise ValueError('Invalid size/components')
    requested=max(1,min(max_bins,size//(k*min_per_cell)))
    bins,info=strata(target,'regression',requested)
    counts=np.bincount(bins);bquota=allocate(counts,size,seed)
    groups=bins*k+components;cellcounts=np.bincount(groups,minlength=len(counts)*k);quota=np.zeros_like(cellcounts)
    for b,q in enumerate(bquota):
        sl=slice(b*k,(b+1)*k);quota[sl]=allocate(cellcounts[sl],int(q),seed+b+1)
    assert quota.sum()==size and np.all(quota<=cellcounts)
    info.update(rule='max(1,min(10,floor(B/(K*50))))',minimum_average_cell_budget=min_per_cell,
                bin_counts=counts.tolist(),bin_quotas=bquota.tolist(),cell_counts=cellcounts.tolist(),
                cell_quotas=quota.tolist(),empty_cells=int(np.sum(cellcounts==0)),
                nonempty_zero_quota_cells=int(np.sum((cellcounts>0)&(quota==0))))
    return groups,quota,cellcounts,bins,info


def constrained_edge_sample(base,graph,groups,quota,penalty_groups,lam,seed,floor=1e-6):
    """Quota cells may split penalty groups; retain edges across target-bin boundaries."""
    base=np.asarray(base,float);groups=np.asarray(groups,int);quota=np.asarray(quota,int);pg=np.asarray(penalty_groups,int)
    if base.shape!=groups.shape or pg.shape!=groups.shape or not np.isfinite(base).all() or np.any(base<floor) or not np.isfinite(lam) or lam<0 or floor<=0:
        raise ValueError('Invalid weights/groups')
    if np.any(groups<0) or np.any(groups>=len(quota)) or np.any(quota<0) or np.any(quota>np.bincount(groups,minlength=len(quota))):raise ValueError('Invalid quota')
    graph=graph.tocsr(copy=True);graph.sum_duplicates();graph.setdiag(0);graph.eliminate_zeros()
    if graph.shape!=(len(base),len(base)) or not np.isfinite(graph.data).all() or np.any(graph.data<0):raise ValueError('Invalid graph')
    delta=graph-graph.T
    if delta.nnz and abs(delta.data).max()>1e-12:raise ValueError('Asymmetric graph')
    rng=np.random.default_rng(seed);weights=base.copy();used=np.zeros(len(base),bool);selected=[];draws=[];updates=0;cross=0
    order=np.arange(len(quota)) if np.array_equal(groups,pg) else np.random.default_rng(seed+104729).permutation(len(quota))
    for g in order:
        q=quota[g]
        ids=np.flatnonzero(groups==g)
        for u in rng.random(int(q)):
            cum=np.cumsum(weights[ids]);local=int(np.searchsorted(cum,u*cum[-1],side='right'));i=int(ids[local])
            if used[i]:raise RuntimeError('Duplicate selection')
            v=float(weights[i]);selected.append(i);draws.append(v);used[i]=True;weights[i]=0
            start,end=graph.indptr[i:i+2];neighbors=graph.indices[start:end];edges=graph.data[start:end]
            keep=(~used[neighbors])&(pg[neighbors]==pg[i]);neighbors=neighbors[keep];edges=edges[keep]
            weights[neighbors]=np.maximum(floor,weights[neighbors]-lam*edges*v)
            updates+=len(neighbors);cross+=int(np.sum(groups[neighbors]!=g))
    ix=np.asarray(selected,int)
    if len(np.unique(ix))!=quota.sum() or not np.array_equal(np.bincount(groups[ix],minlength=len(quota)),quota):raise RuntimeError('Quota mismatch')
    return ix,dict(mechanism='probability',sampling_seed=seed,lambda_edge=lam,weight_floor=floor,
        selection_order_pool_indices=selected,selected_weights_at_draw=draws,edge_updates=updates,cross_target_bin_edge_updates=cross,
        scoring_groups='original GMM components (class x component for classification), unchanged percentile ranks',
        penalty_groups='original components; cross-target-bin edges retained',
        cell_schedule='seeded permutation of cells to avoid always prioritizing low targets; original order when one bin',cell_order=order.tolist(),
        formula='base=1+0.3*(2*D-1)+0.1*H; w_i=max(epsilon,w_i-lambda*K_ij*w_j_at_selection)')
