"""Separate density/hardness quotas with a shared accumulated edge penalty."""
import numpy as np


def split_edge_sample(d, h, graph, groups, quota, seed=None, mechanism='probability',
                      alpha=.3, lam=.3, floor=1e-6):
    d=np.asarray(d,float);h=np.asarray(h,float);groups=np.asarray(groups,int);quota=np.asarray(quota,int)
    if mechanism not in ('probability','greedy') or (mechanism=='probability' and seed is None):
        raise ValueError('Invalid mechanism/seed')
    if d.shape!=h.shape or d.shape!=groups.shape or d.ndim!=1 or not np.isfinite([alpha,lam,floor]).all() or not 0<=alpha<1 or lam<0 or floor<=0:
        raise ValueError('Invalid inputs')
    if not np.isfinite(d).all() or not np.isfinite(h).all() or np.any((d<0)|(d>1)|(h<0)|(h>1)):
        raise ValueError('Expected percentile scores')
    if np.any(groups<0) or np.any(groups>=len(quota)) or np.any(quota<0) or np.any(quota>np.bincount(groups,minlength=len(quota))):
        raise ValueError('Invalid quota')
    graph=graph.tocsr(copy=True);graph.sum_duplicates();graph.setdiag(0);graph.eliminate_zeros()
    if graph.shape!=(len(d),len(d)) or not np.isfinite(graph.data).all() or np.any(graph.data<0):raise ValueError('Invalid graph')
    delta=graph-graph.T
    if delta.nnz and abs(delta.data).max()>1e-12:raise ValueError('Asymmetric graph')
    # Largest remainder, deterministic group-id tie break; nearest integer total.
    hquota=quota//5;remaining=(int(quota.sum())+2)//5-int(hquota.sum())
    order=np.lexsort((np.arange(len(quota)),-(quota%5)))
    hquota[order[:remaining]]+=1
    rng=np.random.default_rng(seed) if mechanism=='probability' else None
    selected=[];modes=[];draws=[];penalties=[];updates=0
    for g,q in enumerate(quota):
        if not q:continue
        ids=np.flatnonzero(groups==g);local=graph[ids][:,ids].tocsr()
        base=np.stack([1+alpha*(2*d[ids]-1),1+alpha*(2*h[ids]-1)])
        penalty=np.zeros(len(ids));used=np.zeros(len(ids),bool)
        for step in range(int(q)):
            # Evenly interleave exactly hquota[g] H draws; D comes first.
            mode=int(((step+1)*hquota[g])//q > (step*hquota[g])//q)
            w=np.maximum(floor,base[mode]-penalty);w[used]=0
            if rng is None:
                i=int(np.flatnonzero((~used)&(w==w[~used].max()))[0])
            else:
                cumulative=np.cumsum(w);i=int(np.searchsorted(cumulative,rng.random()*cumulative[-1],side='right'))
            if i>=len(ids) or used[i]:raise RuntimeError('Invalid draw')
            value=float(w[i]);selected.append(int(ids[i]));modes.append('H' if mode else 'D');draws.append(value);penalties.append(float(penalty[i]));used[i]=True
            start,end=local.indptr[i:i+2];neighbors=local.indices[start:end];edges=local.data[start:end];keep=~used[neighbors]
            penalty[neighbors[keep]]+=lam*edges[keep]*value;updates+=int(keep.sum())
    ix=np.asarray(selected,int)
    if len(np.unique(ix))!=quota.sum() or not np.array_equal(np.bincount(groups[ix],minlength=len(quota)),quota):raise RuntimeError('Selection quota mismatch')
    return ix,dict(mechanism=mechanism,alpha=alpha,lambda_edge=lam,sampling_seed=seed,
        hardness_fraction=.2,density_group_allocation=(quota-hquota).tolist(),hardness_group_allocation=hquota.tolist(),
        selection_order_pool_indices=selected,selection_modes=modes,selected_weights_at_draw=draws,
        selected_accumulated_penalties=penalties,edge_updates=updates,weight_floor=floor,
        shared_penalty=True,schedule='floor((t+1)*Hquota/B)>floor(t*Hquota/B)',
        formulas={'density':'1+alpha*(2*d-1)','hardness':'1+alpha*(2*h-1)',
                  'weight':'max(epsilon,b_i^mode-P_i)', 'update':'P_i += lambda*K_ij*w_j^mode_at_selection; same-group unselected neighbors'},
        graph_definition='Frozen symmetric graph restricted to class/component; no degree normalization')
