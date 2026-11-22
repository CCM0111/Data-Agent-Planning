"""GMM component-mass quotas and within-component radial coverage experiments."""
import numpy as np
from scipy.stats import rankdata
from .coreset import greedy_select
from .multi_proxy import split_quotas

VARIANTS=('component_quota','component_additive','component_radial')


def allocate(counts,total,seed):
    counts=np.asarray(counts,dtype=int);out=np.zeros(len(counts),dtype=int);active=np.flatnonzero(counts)
    if total:
        out[active]=split_quotas(counts[active],int(total),seed,lower=np.zeros(len(active),dtype=int),upper=counts[active])
    return out


def component_groups(classes,class_quotas,components,n_components,seed):
    groups=np.asarray(classes)*n_components+np.asarray(components)
    counts=np.bincount(groups,minlength=len(class_quotas)*n_components);quota=np.zeros_like(counts)
    for c,q in enumerate(class_quotas):
        sl=slice(c*n_components,(c+1)*n_components)
        quota[sl]=allocate(counts[sl],int(q),seed+c)
    return groups,quota,counts


def utilities(hardness,graph):
    h=np.asarray(hardness);result=[];degree=np.asarray(graph.sum(axis=1)).ravel()
    for i in range(len(h)):
        relative=np.maximum(h[i]-np.median(np.delete(h,i,axis=0),axis=0),0.)
        neighbors=np.divide(graph@relative,degree,out=relative.copy(),where=degree>0)
        result.append(.5*(relative+neighbors))
    return np.asarray(result)


def typicality(radius,groups):
    t=np.empty(len(radius))
    for group in np.unique(groups):
        mask=groups==group;n=int(mask.sum())
        t[mask]=1-(rankdata(np.asarray(radius)[mask],method='average')-1)/(n-1) if n>1 else .5
    return t


def greedy_final_scale(values,graph,groups,quota,final_size,initial=None):
    count=int(np.sum(quota));initial=np.asarray([] if initial is None else initial,dtype=int)
    if count==len(initial):return initial.copy()
    if count==1:
        candidates=np.flatnonzero(np.asarray(quota)[groups]>0)
        return np.array([candidates[np.argmax(values[candidates])]])
    # Every gain is multiplied by final_size/count; k_eff remains final-size k_eff.
    return greedy_select(values,graph,groups,quota,min(20,count-1)/min(20,final_size-1),
                         normalization_degree=20,initial=initial)[0]


def independent_values(values,graph,groups,quota,final_size,seed,initial=None):
    body=np.asarray([] if initial is None else initial,dtype=int);body_counts=np.bincount(groups[body],minlength=len(quota))
    left=np.asarray(quota)-body_counts
    if np.any(left<0):raise ValueError('Initial body exceeds group quotas')
    remaining=int(left.sum());m=len(values);rng=np.random.default_rng(seed)
    shares=np.full(m,remaining//m,dtype=int);shares[rng.permutation(m)[:remaining%m]]+=1
    parts=[];allocations=[]
    for i,q in enumerate(shares):
        aq=allocate(left,int(q),seed+i+1);left-=aq;allocations.append(aq.tolist())
        chosen=greedy_final_scale(values[i],graph,groups,body_counts+aq,final_size,body)
        parts.append(chosen[~np.isin(chosen,body)])
    union=np.unique(np.concatenate([body,*parts]))
    selected=greedy_final_scale(values.mean(0),graph,groups,quota,final_size,union)
    return selected,{'initial_rows':len(body),'method_proposal_quotas':shares.tolist(),
                     'method_group_quotas':allocations,'union_rows':len(union),
                     'dedup_fill_rows':int(np.sum(quota))-len(union)}


def select(density,hardness,graph,groups,quota,radius,variant,seed=2030,alpha=.5,rho=.4):
    if variant not in VARIANTS:raise ValueError(variant)
    if not 0<=alpha<=1 or not 0<=rho<=1:raise ValueError('Invalid alpha/rho')
    density=np.asarray(density);groups=np.asarray(groups,dtype=int);quota=np.asarray(quota,dtype=int)
    size=int(quota.sum());u=utilities(hardness,graph);t=typicality(radius,groups)
    if variant=='component_quota':
        mu=np.sum(u*density,axis=1)/density.sum() if density.sum() else np.zeros(len(u))
        values=density[None,:]*(1+.25*(u-mu[:,None]))
    else:values=(1-alpha)*u+alpha*t[None,:]
    initial=None;coverage_info=None;floor=None;radial_groups=None
    if variant=='component_radial':
        # t=1 is the center, t=0 is peripheral. Rank ties stay together.
        radial=np.minimum(((1-t)*4).astype(int),3);radial_groups=groups*4+radial
        counts=np.bincount(radial_groups,minlength=len(quota)*4)
        floor=np.zeros_like(counts);group_floor=allocate(quota,int(np.floor(rho*size+.5)),seed+100)
        for g,q in enumerate(group_floor):
            sl=slice(4*g,4*g+4);floor[sl]=allocate(counts[sl],int(q),seed+101+g)
        initial,coverage_info=independent_values(values,graph,radial_groups,floor,size,seed)
        coverage_info.update(minimum_radial_counts=floor.tolist(),coverage_rows=len(initial))
    selected,info=independent_values(values,graph,groups,quota,size,seed,initial)
    actual=np.bincount(groups[selected],minlength=len(quota))
    if len(np.unique(selected))!=size or not np.array_equal(actual,quota):raise ValueError('Component quota invariant failed')
    if floor is not None:
        final_radial=np.bincount(radial_groups[selected],minlength=len(floor))
        if np.any(final_radial<floor):raise ValueError('Radial coverage invariant failed')
        coverage_info['final_radial_counts']=final_radial.tolist()
    info.update(variant=variant,alpha=alpha if variant!='component_quota' else None,
                rho=rho if variant=='component_radial' else None,group_allocation=quota.tolist(),
                final_group_counts=actual.tolist(),coverage=coverage_info,
                final_size=size,normalization_denominator=size*min(20,size-1),
                selected_typicality_mean=float(t[selected].mean()),
                selection_rule='Independent per-method proposals and mean-score dedup filling, with a shared radial-coverage body only in component_radial; all penalties include initial body.')
    return selected,info
