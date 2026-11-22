"""Frozen-proxy shared coresets: mean scores, round-robin, or independent merge."""
import heapq
import numpy as np
from .coreset import greedy_select
from .multi_proxy import relative_density_importance,split_quotas


def shared_select(density, hardness, graph, groups, allocation, variant, *, seed=2026):
    values=np.array([relative_density_importance(density,hardness,graph,i,local=True)[0]
                     for i in range(len(hardness))])
    p=int(np.sum(allocation));n=len(density);m=len(values);degree=min(20,p-1)
    if p<2 or p>n or m<2:raise ValueError('Infeasible shared selection')
    groups=np.asarray(groups,dtype=int);allocation=np.asarray(allocation,dtype=int)
    if len(groups)!=n or np.any(allocation<0) or np.any(allocation>np.bincount(groups,minlength=len(allocation))):
        raise ValueError('Infeasible group allocation')
    if variant=='mean_u':
        ix,info=greedy_select(values.mean(0),graph,groups,allocation,1.,normalization_degree=20)
        return ix,{**info,'variant':variant,'formula':'mean of centered local-relative C importance; equivalent to mean U then weighted centering'},values
    rng=np.random.default_rng(seed)
    method_quotas=np.full(m,p//m,dtype=int);method_quotas[rng.permutation(m)[:p%m]]+=1
    if variant=='independent_merge':
        class_left=allocation.copy();parts=[];class_parts=[]
        for a,q in enumerate(method_quotas):
            # Allocate class slots before independent selection. Column sums equal
            # final class quotas, so deduplication can only create deficits.
            aq=np.zeros(len(allocation),dtype=int)
            active=np.flatnonzero(class_left)
            if q:
                aq[active]=split_quotas(class_left[active],int(q),seed+a+1,
                    lower=np.zeros(len(active),dtype=int),upper=class_left[active])
            class_left-=aq;class_parts.append(aq.tolist())
            if q==0:part=np.array([],dtype=int)
            elif q==1:
                candidates=np.flatnonzero(aq[groups]>0)
                part=np.array([candidates[np.argmax(values[a,candidates])]])
            else:
                # q is only a positive common scaling; keep the final set's k_eff.
                sub_degree=min(20,int(q)-1)
                part,_=greedy_select(values[a],graph,groups,aq,sub_degree/degree,normalization_degree=20)
            parts.append(part)
        union=np.unique(np.concatenate(parts))
        ix,info=greedy_select(values.mean(0),graph,groups,allocation,1.,normalization_degree=20,initial=union)
        assert np.all(class_left==0) and len(np.unique(ix))==p
        info.update(variant=variant,method_quotas=method_quotas.tolist(),
            independent_class_quotas=class_parts,independent_q_indices=[x.tolist() for x in parts],
            union_size=len(union),duplicate_slots=p-len(union),supplement_size=p-len(union),
            supplement_rule='mean C importance; final remaining class quotas; similarity to complete union and supplements',
            quota_scope='equal per-method proposals before deduplication; no exclusive final ownership quota',
            independent_normalization_degree=degree)
        return ix,info,values
    if variant!='method_quota':raise ValueError(variant)
    remaining=method_quotas.copy();class_remaining=allocation.copy()
    selected=np.zeros(n,dtype=bool);versions=np.zeros(n,dtype=int);sim=np.zeros(n)
    heaps=[[(-float(values[a,i]/p),i,0) for i in range(n)] for a in range(m)]
    for h in heaps:heapq.heapify(h)
    chosen=[];owners=[];gains=[];factor=2/(p*degree)
    while len(chosen)<p:
        for a in rng.permutation(np.flatnonzero(remaining)):
            heap=heaps[a]
            while heap:
                negative,i,version=heapq.heappop(heap)
                if not selected[i] and class_remaining[groups[i]] and version==versions[i]:break
            else:raise ValueError('Cannot fill method quota')
            chosen.append(i);owners.append(int(a));gains.append(-negative)
            selected[i]=True;remaining[a]-=1;class_remaining[groups[i]]-=1
            start,end=graph.indptr[i:i+2]
            for j,weight in zip(graph.indices[start:end],graph.data[start:end]):
                if not selected[j] and class_remaining[groups[j]]:
                    sim[j]+=weight;versions[j]+=1
                    for b in np.flatnonzero(remaining):
                        gain=values[b,j]/p-factor*sim[j]
                        heapq.heappush(heaps[b],(-float(gain),int(j),int(versions[j])))
    ix=np.array(chosen,dtype=int);owners=np.array(owners,dtype=int)
    penalty=float(graph[ix][:,ix].sum()/(p*degree));importance=float(values[owners,ix].mean())
    if not np.isclose(sum(gains),importance-penalty,rtol=1e-9,atol=1e-10):
        raise ValueError('Assigned-importance marginal invariant failed')
    assert np.all(remaining==0) and np.all(class_remaining==0) and len(np.unique(ix))==p
    return ix,{'variant':variant,'method_quotas':method_quotas.tolist(),
        'owner_method_ids':owners.tolist(),'marginal_gains':gains,'round_order':'seeded fresh permutation each round',
        'importance_term':importance,'similarity_penalty':penalty,'objective':importance-penalty,
        'objective_definition':'assigned owner importance / final size minus global similarity penalty',
        'normalization_degree':degree,'normalization_denominator':p*degree,
        'duplicate_policy':'selected rows excluded from every method; each admitted row charged only to selecting method'},values
