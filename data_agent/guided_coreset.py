"""Fixed-size, validation-guided sampling in Q-local coordinates."""
import numpy as np
import joblib
from scipy.spatial.distance import cdist,pdist
from scipy.spatial import cKDTree
from .marginal_coverage import coverage_sample
from .multi_proxy_io import read_typed


def alloc(weights,cap,n):
    weights=np.asarray(weights,float);cap=np.asarray(cap,int)
    if n==0:return np.zeros(len(cap),int)
    if n>cap.sum() or weights.sum()<=0:raise ValueError('Infeasible allocation')
    a=np.minimum(np.floor(n*weights/weights.sum()).astype(int),cap)
    ideal=n*weights/weights.sum()
    while a.sum()<n:
        a[np.argmax(np.where(a<cap,ideal-a,-np.inf))]+=1
    return a
def herd(x,y,n):
    # Exact RBF empirical MMD. RBF diagonal equals one.
    if not n:return np.empty(0,int)
    z=y[np.linspace(0,len(y)-1,min(len(y),256)).astype(int)]
    distances=pdist(z,'sqeuclidean'); pos=distances[distances>0]
    bandwidth=np.median(pos) if len(pos) else 1.
    x=np.ascontiguousarray(x,dtype=np.float32); y=np.ascontiguousarray(y,dtype=np.float32)
    norm=(x*x).sum(1); mean=np.zeros(len(x))
    for start in range(0,len(x),512):
        d=cdist(x[start:start+512],y,'sqeuclidean')
        mean[start:start+len(d)]=np.exp(-d/(2*bandwidth)).mean(1)
    total=np.zeros(len(x)); used=np.zeros(len(x),bool); order=[]
    for j in range(n):
        score=(j+1)*mean-total;score[used]=-np.inf;i=score.argmax();used[i]=True;order.append(i)
        d=np.maximum(norm+norm[i]-2*(x@x[i]),0)
        total+=np.exp(-d/(2*bandwidth))
    return np.array(order)
def high_by_method(x, y, errors, allowed, amount):
    """Round-robin slots; skip claimed anchors, cycle only after exhausting validation."""
    n = len(errors)
    if not n or errors.shape != (n,len(y)) or not len(y) or not 0 <= amount <= len(allowed):
        raise ValueError('invalid per-method budget or anchors')
    orders = np.argsort(-errors,axis=1,kind='stable'); cursors = [0]*n
    claimed, used, picks, anchors = set(), set(), [], []
    tree = cKDTree(x[allowed]) if amount else None
    for slot in range(amount):
        j = slot % n
        if len(claimed) == len(y): claimed.clear(); cursors = [0]*n
        while orders[j,cursors[j]] in claimed: cursors[j] += 1
        anchor = int(orders[j,cursors[j]]); claimed.add(anchor); cursors[j] += 1
        k = 1
        while True:
            near = np.atleast_1d(tree.query(y[anchor],k=k)[1])
            pick = next((int(i) for i in near if i not in used),None)
            if pick is not None: break
            k = min(2*k,len(allowed))
        used.add(pick); picks.append(allowed[pick]); anchors.append(anchor)
    return np.asarray(picks,int), np.asarray(anchors,int)

def row_errors(pred, labels, classes):
    if not classes:return np.abs(pred['predictions']-np.asarray(labels,float))
    lookup={str(k):i for i,k in enumerate(classes)}
    return 1-pred['probabilities'][np.arange(len(labels)),[lookup[str(k)] for k in labels]]

def guided_reference(source,context,qframe,inputs,graph,errors,seed):
    features=context['feature_columns']; v=read_typed(source/'core_validation.csv',context['csv_schema'])
    rep=joblib.load(source/'similarity_transform.joblib')
    def embed(df):
        return np.vstack([rep.transform(df.iloc[i:i+2048][features]) for i in range(0,len(df),2048)])
    x,y=embed(qframe),embed(v)
    vg=joblib.load(source/'density_gmm/fitted.joblib').predict(y); qg=inputs['components']
    gr=inputs['groups']; quota=inputs['quota']; n=int(sum(quota)); k=round(.10*n); h=round(.15*n)-k
    errors=np.asarray(errors)
    if errors.ndim!=2 or not len(errors) or errors.shape[1]!=len(y) or not np.isfinite(errors).all():raise ValueError('Incomplete V_core method errors')
    components=np.unique(qg);counts=np.array([(vg==g).sum() for g in components])
    caps=np.array([(qg==g).sum() if c else 0 for g,c in zip(components,counts)])
    if caps.sum()<k:raise ValueError('Insufficient Q support for V_core component quotas')
    m=[]
    if k:
        for g,count in zip(components,alloc(counts,caps,k)):
            if count:
                rows=np.flatnonzero(qg==g);m.extend(rows[herd(x[rows],y[vg==g],count)])
    pool=np.setdiff1d(np.arange(len(x)),m)
    hard,anchors=high_by_method(x,y,errors,pool,h)
    fixed=np.asarray([*m,*hard],int);pool=np.setdiff1d(pool,hard)
    have=np.bincount(gr[fixed],minlength=len(quota))
    mutable=alloc(np.maximum(quota-have,0),np.bincount(gr[pool],minlength=len(quota)),n-len(fixed))
    if context['classes']:
        # A protected majority must not crowd the last example of a class out.
        labels=np.asarray(qframe[context['target']])
        for label in np.unique(labels):
            target_groups=np.unique(gr[labels==label])
            if (have+mutable)[target_groups].sum():continue
            donors=[g for g in np.flatnonzero(mutable) if (have+mutable)[np.unique(gr[labels==labels[np.flatnonzero(gr==g)[0]]])].sum()>1]
            if not donors:raise ValueError('Protected budget leaves insufficient slots for class coverage')
            mutable[max(donors,key=lambda g:mutable[g])]-=1
            mutable[max(target_groups,key=lambda g:np.sum(gr[pool]==g))]+=1
    drawn,_=coverage_sample(inputs['density'][pool],inputs['hardness'][pool],graph.tocsr()[pool][:,pool],gr[pool],mutable,seed,gamma=2,beta=.6)
    ref=np.r_[fixed,pool[drawn]]
    if len(np.unique(ref))!=n:raise ValueError('Guided coreset size or overlap')
    return ref,dict(protected=fixed.tolist(),mmd=list(map(int,m)),hard=hard.tolist(),anchors=anchors.tolist())

def initial_errors(source,context,validation,methods,cfg):
    from .region_bandit.execution import predict_cached
    from .region_bandit.runtime import _load_predictions
    file=source/'core_validation_features.csv'
    validation[context['feature_columns']].to_csv(file,index=False)
    errors=[]
    for i,m in enumerate(methods):
        result=predict_cached(m['code_path'],source/f'proxies/method_{i}/fitted.joblib',file,context,context['classes'],source/'guided_predictions'/str(i),batch_size=cfg.batch_size,timeout_seconds=cfg.evaluation_budget['timeout_seconds'])
        if result['status']!='ok':raise RuntimeError(result['error'])
        errors.append(row_errors(_load_predictions(result['artifact'],context['classes']),validation[context['target']],context['classes']))
    return errors

def coordinates(reference,groups,fixed):
    fixed=np.asarray(fixed,int);pool=np.setdiff1d(np.arange(len(groups)),fixed)
    mutable=np.asarray(reference)[~np.isin(reference,fixed)]
    if len(set(fixed))!=len(fixed) or np.any((fixed<0)|(fixed>=len(groups))) or len(mutable)+len(fixed)!=len(reference):raise ValueError('Missing protected rows in reference')
    return pool,np.searchsorted(pool,mutable)

def protected_plans(regions,reference,groups,classes,fixed,**kwargs):
    from .region_bandit.action_planner import plan_actions
    pool,ref=coordinates(reference,groups,fixed)
    reduced=[{**r,'mask':r['mask'][pool]} for r in regions]
    result=plan_actions(reduced,ref,groups[pool],None if classes is None else classes[pool],size=len(ref),**kwargs)
    for a in result['actions']:a['protected_q_indices']=list(map(int,fixed))
    return result
