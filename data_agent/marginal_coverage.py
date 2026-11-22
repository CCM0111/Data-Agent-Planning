"""Sparse, exact weighted facility-location gains for frozen coreset sampling."""
import numpy as np
from scipy import sparse


def prepare_coverage_graph(graph, size):
    """Apply the frozen graph contract shared by both sampling schedules."""
    graph = sparse.csr_matrix(graph, dtype=float, copy=True)
    graph.sum_duplicates()
    if graph.shape != (size, size) or not np.isfinite(graph.data).all() or np.any(graph.data < 0) or np.any(graph.data > 1+1e-12):
        raise ValueError('Require frozen similarity in [0,1]')
    graph.data = np.minimum(graph.data, 1.)
    graph.setdiag(1.);graph.eliminate_zeros();graph.sort_indices()
    return graph


class GroupCoverageState:
    """One original class/component group, shared across regional quota cells.

    Scheduling cells must never change this group's graph, weight normalization,
    or initial gain scale. This is the same sparse update used by coverage_sample.
    """
    def __init__(self, ids, d, h, graph, *, gamma=1., rho=1., beta=.1,
                 density_coefficient=.3):
        self.ids = np.asarray(ids, int)
        self.K = graph[self.ids][:, self.ids].tocsr()
        self.cols = self.K.tocsc()
        self.a = 1+rho*(1-d[self.ids]);self.a /= self.a.mean()
        self.base = 1+density_coefficient*(2*d[self.ids]-1)+beta*h[self.ids]
        self.gain = np.asarray(self.K.T @ self.a).ravel()
        self.scale = max(float(np.median(self.gain)), 1e-12)
        self.gamma = gamma
        self.coverage = np.zeros(len(self.ids))
        self.used = np.zeros(len(self.ids), bool)

    def weights(self):
        self.gain = np.maximum(self.gain, 0.)
        bonus = self.gain/(self.gain+self.scale)
        weights = self.base+self.gamma*bonus
        weights[self.used] = 0.
        return weights, bonus

    def select(self, i):
        if self.used[i]:
            raise RuntimeError('Repeated selection')
        self.used[i] = True
        start, end = self.cols.indptr[i:i+2]
        for j, similarity in zip(self.cols.indices[start:end], self.cols.data[start:end]):
            old = self.coverage[j]
            if similarity <= old:
                continue
            lo, hi = self.K.indptr[j:j+2]
            neighbors = self.K.indices[lo:hi];values = self.K.data[lo:hi]
            reduction = self.a[j]*(np.maximum(values-old, 0.)-np.maximum(values-similarity, 0.))
            self.gain[neighbors] -= reduction
            self.coverage[j] = similarity

    def summary(self, group):
        return dict(group=int(group), mean_coverage=float(self.coverage.mean()),
                    weighted_mean_coverage=float(self.a@self.coverage/self.a.sum()))


def coverage_sample(d, h, graph, groups, quota, seed=None, mechanism='probability',
                    gamma=1., rho=1., beta=.1, density_coefficient=.3):
    d, h = np.asarray(d, float), np.asarray(h, float)
    groups, quota = np.asarray(groups, int), np.asarray(quota, int)
    if mechanism not in ('probability', 'greedy') or (mechanism == 'probability' and seed is None):
        raise ValueError('Invalid mechanism/seed')
    if not 0 <= density_coefficient < 1 or gamma < 0 or rho < 0 or beta < 0 or not np.isfinite([gamma, rho, beta, density_coefficient]).all():
        raise ValueError('Invalid coefficients')
    if d.shape != groups.shape or h.shape != d.shape or not np.isfinite([d, h]).all() or np.any((d < 0)|(d > 1)|(h < 0)|(h > 1)):
        raise ValueError('Invalid percentiles')
    if np.any(groups < 0) or np.any(groups >= len(quota)) or np.any(quota < 0) or np.any(quota > np.bincount(groups, minlength=len(quota))):
        raise ValueError('Invalid quotas')
    graph = prepare_coverage_graph(graph, len(d))
    rng = np.random.default_rng(seed)
    chosen=[];scales=[];gains_at_draw=[];bonuses=[];coverages=[]
    for g, q in enumerate(quota):
        ids=np.flatnonzero(groups==g)
        if not q:continue
        state=GroupCoverageState(ids,d,h,graph,gamma=gamma,rho=rho,beta=beta,
                                 density_coefficient=density_coefficient)
        n=len(ids)
        scales.append(dict(group=int(g),scale=state.scale))
        for _ in range(int(q)):
            weights,bonus=state.weights()
            if mechanism=='greedy':i=int(np.argmax(weights))
            else:
                cumulative=np.cumsum(weights)
                i=min(int(np.searchsorted(cumulative,rng.random()*cumulative[-1],side='right')),n-1)
            chosen.append(int(ids[i]));gains_at_draw.append(float(state.gain[i]));bonuses.append(float(bonus[i]))
            state.select(i)
        coverages.append(state.summary(g))
    ix=np.asarray(chosen,int)
    if len(np.unique(ix))!=quota.sum() or not np.array_equal(np.bincount(groups[ix],minlength=len(quota)),quota):
        raise RuntimeError('Selection invariant failed')
    info=dict(mechanism=mechanism,sampling_seed=seed,a=density_coefficient,beta=beta,gamma=gamma,rho=rho,
              formula='base=1+a*(2*d-1)+beta*h; a_j=(1+rho*(1-d_j))/group_mean; G_i=sum_j a_j*max(K_ji-c_j,0); w_i=base_i+gamma*G_i/(G_i+s_g)',
              scale_definition='Median initial group gain; fixed throughout selection',
              graph_scope='existing [0,1] similarity restricted to class/component; diagonal=1; absent edges=0',
              edge_penalty=False,selection_order_q_indices=ix.tolist(),group_scales=scales,
              gain_at_draw=gains_at_draw,bonus_at_draw=bonuses,final_coverage=coverages,
              group_allocation=quota.tolist(),tie_rule='smallest Q index for greedy exact ties',
              density_low_quartile_fraction=float(np.mean(d[ix]<.25)),
              density_low_1pct_fraction=float(np.mean(d[ix]<=.01)))
    return ix,info
