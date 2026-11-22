"""Configurable proxy/coreset workflow. No MCTS execution, retrieval, or test use."""
import asyncio
import copy
from dataclasses import asdict, dataclass, field
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import warnings
from threadpoolctl import threadpool_limits

from .coreset import (CoresetConfig, Representation, digest, excluded_fields, export_rows,
                      file_hash, greedy_select, objective, quotas, strata, stratified_sample, timed, write_json)
from .data_profile import training_profile, feature_structure_profile, feature_metadata_for_task
from .runtime import score
from .multi_proxy_runtime import validate_code
from .model_registry import model_contract
from .multi_proxy_io import read_raw, infer_schema, apply_schema
from .deepseek_usage import estimate, is_direct_deepseek, summarize, utc_now
from .multi_proxy_trace import describe_eda, validate_idea, identity, save_trace, validate_stage_parameters, normalize_plan
from .stage1_metrics import primary_score, metric_direction, UndefinedMetricError
from .stage1_selection import normalize_selection_config, analyze_run, compare_full_stage
from .execution_budget import (TaskDeadlineExceeded, check_deadline, bounded_timeout,
                               await_with_budget, call_with_budget)

from .gmm_planning import GMM_PLAN_INSTRUCTION, validate_gmm_design, apply_gmm_design, validate_reused_gmm

VERSION = '2.8.2'


def retrieve_experience(task_context, eda_summary):
    return []


@dataclass
class MultiProxyConfig:
    validation_guided_mix: bool = False
    proxy_train_ratio: float = .05
    proxy_train_max_rows: int = 10000
    proxy_train_min_rows: int = 500
    coreset_ratio: float = .10
    coreset_max_rows: int | None = 5000
    proxy_train_rows: int | None = None
    coreset_rows: int | None = None
    core_validation_ratio: float = .05
    core_validation_min_rows: int = 300
    core_validation_max_rows: int | None = None  # Accepted for compatibility with archived configs.
    core_validation_rows: int | None = None
    defer_full_train: bool = False
    mcts_bootstrap_only: bool = False
    proxy_selection_scheme: str = "original"
    n_strategies: int = 3
    n_coresets: int = 4
    gmm_components: int = 5
    gmm_max_iter: int = 100
    gmm_seed: int = 2026
    regression_bins: int = 5
    error_clip_quantile: float = .95
    similarity_lambda: float = 1.
    split_seed: int = 2026
    representation_seed: int = 2026
    graph_seed: int = 2026
    model_seed: int = 101
    batch_size: int = 2048
    max_dimensions: int = 20
    max_categories: int = 32
    graph_block_size: int = 2000
    neighbors: int = 20
    id_columns: list = field(default_factory=list)
    leakage_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    scoring_budget: dict = field(default_factory=lambda: {'max_trees': 100, 'max_epochs': 5, 'timeout_seconds': 120})
    full_train_enabled: bool = True
    method_preflight: bool = False
    parameter_feedback: bool = False
    parameter_feedback_mode: str = 'guarded_once'
    full_train_budget: dict = field(default_factory=lambda: {'max_trees': 500, 'max_epochs': 200, 'timeout_seconds': 1800})
    evaluation_budget: dict = field(default_factory=lambda: {'max_trees': 100, 'max_epochs': 50, 'timeout_seconds': 300})
    stage1_selection: dict = field(default_factory=normalize_selection_config)

    def validate(self):
        self.stage1_selection = normalize_selection_config(self.stage1_selection)
        if not isinstance(self.defer_full_train,bool) or (self.defer_full_train and not self.full_train_enabled):
            raise ValueError('defer_full_train requires full_train_enabled')
        if self.proxy_selection_scheme not in {'original', 'gmm_anchor80', 'density_hardness', 'density_centered_eta025', 'density_relative', 'density_relative_local', 'edge_mechanism_comparison', 'marginal_coverage_random_comparison', 'marginal_coverage_dual_random_comparison', 'marginal_coverage_only'}:
            raise ValueError('Unknown proxy_selection_scheme')
        if self.proxy_selection_scheme == 'edge_mechanism_comparison' and (self.n_strategies != 5 or self.n_coresets != 4 or self.stage1_selection['enabled']):
            raise ValueError('Edge comparison requires five methods and separate per-arm MCS analysis')
        if self.proxy_selection_scheme in {'marginal_coverage_random_comparison', 'marginal_coverage_dual_random_comparison', 'marginal_coverage_only'} and (self.n_strategies != 5 or self.n_coresets != (1 if self.proxy_selection_scheme == 'marginal_coverage_only' else 9 if self.proxy_selection_scheme == 'marginal_coverage_dual_random_comparison' else 6) or self.gmm_components != 5 or self.stage1_selection['enabled']):
            raise ValueError('Coverage comparison requires five methods, matching paired arms, K=5 and separate per-arm MCS')
        if type(self.n_strategies) is not int or not 2 <= self.n_strategies <= 7 or (self.proxy_selection_scheme not in {'edge_mechanism_comparison', 'marginal_coverage_dual_random_comparison', 'marginal_coverage_only'} and self.n_coresets != self.n_strategies + 1):
            raise ValueError('Require 2..7 strategies and one coreset per strategy plus GMM')
        if len(self.stage1_selection['coreset_weights']) != self.n_coresets:
            raise ValueError('Coreset weights must match n_coresets')
        if self.n_strategies != 3 and self.stage1_selection['mcts_admission'] is not False:
            raise ValueError('Non-three-method runs require mcts_admission=false')
        if self.validation_guided_mix and self.proxy_selection_scheme!='marginal_coverage_only':
            raise ValueError('Validation guidance requires the single marginal coverage scheme')
        for key in ('proxy_train_ratio', 'coreset_ratio', 'core_validation_ratio'):
            if not 0 < getattr(self, key) < 1:
                raise ValueError(f'{key} must be in (0,1)')
        for key in ('proxy_train_min_rows', 'core_validation_min_rows', 'proxy_train_max_rows', 'regression_bins', 'gmm_components', 'gmm_max_iter', 'batch_size', 'max_dimensions', 'max_categories', 'graph_block_size', 'neighbors'):
            if not isinstance(getattr(self,key),int) or getattr(self,key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('proxy_train_rows','coreset_rows','core_validation_rows','core_validation_max_rows','coreset_max_rows'):
            if getattr(self,key) is not None and (not isinstance(getattr(self,key),int) or getattr(self,key)<1):
                raise ValueError(f'{key} must be a positive integer')
        if self.proxy_train_min_rows > self.proxy_train_max_rows:
            raise ValueError('proxy_train minimum exceeds maximum')
        if self.max_dimensions>20 or self.graph_block_size>2000 or self.neighbors>20 or self.max_categories<2:
            raise ValueError('Maximum dimensions/block size/neighbors are 20/2000/20; categories >=2')
        if not math.isfinite(self.similarity_lambda) or self.similarity_lambda < 0:
            raise ValueError('lambda must be finite and nonnegative')
        if not 0 < self.error_clip_quantile <= 1:
            raise ValueError('error_clip_quantile must be in (0,1]')
        for key in ('split_seed','representation_seed','graph_seed','model_seed','gmm_seed'):
            value=getattr(self,key)
            if not isinstance(value,int) or not 0 <= value < 2**32:
                raise ValueError(f'Invalid random seed: {key}')
        if not isinstance(self.full_train_enabled,bool):
            raise ValueError('full_train_enabled must be boolean')
        if not isinstance(self.method_preflight,bool):
            raise ValueError('method_preflight must be boolean')
        if not isinstance(self.parameter_feedback,bool):
            raise ValueError('parameter_feedback must be boolean')
        if self.parameter_feedback_mode not in {'guarded_once','post_selection'}:
            raise ValueError('Unknown parameter_feedback_mode')
        if self.parameter_feedback and self.defer_full_train and not self.mcts_bootstrap_only:
            raise ValueError('parameter_feedback requires inline full training; deferred full replay is not supported')
        if not isinstance(self.mcts_bootstrap_only, bool) or (self.mcts_bootstrap_only and not (self.defer_full_train and self.proxy_selection_scheme == 'marginal_coverage_only' and not self.stage1_selection['enabled'])):
            raise ValueError('MCTS bootstrap requires deferred full fits, coverage-only and no stage-one analysis')
        for budget in (self.scoring_budget,self.evaluation_budget,self.full_train_budget):
            for key in ('max_trees','max_epochs','timeout_seconds'):
                if not isinstance(budget.get(key),int) or budget[key]<1:
                    raise ValueError(f'Budget requires positive integer {key}')


def split_quotas(counts, total, seed, lower=None, upper=None):
    """Bounded proportional allocation with seeded randomized ties."""
    counts=np.asarray(counts)
    order=np.random.default_rng(seed).permutation(len(counts))
    allocated=quotas(counts[order],total,
                     lower=None if lower is None else np.asarray(lower)[order],
                     upper=None if upper is None else np.asarray(upper)[order])
    result=np.empty_like(allocated);result[order]=allocated
    return result


def partition(y, problem_type, cfg):
    """Independent P/V sizes, with simultaneous capacity reservations for P/V/Q."""
    cfg.validate()
    n=len(y)
    ratio_p=max(2,math.ceil(n*cfg.coreset_ratio))
    requested_p=cfg.coreset_rows or (min(cfg.coreset_max_rows,ratio_p) if cfg.coreset_max_rows is not None else ratio_p)
    requested_a=cfg.proxy_train_rows or min(cfg.proxy_train_max_rows,max(cfg.proxy_train_min_rows,math.ceil(n*cfg.proxy_train_ratio)))
    requested_v=cfg.core_validation_rows or max(cfg.core_validation_min_rows,math.ceil(n*cfg.core_validation_ratio))
    # Reserve the entire validation target before constructing Q or selecting any coreset.
    validation_floor=(n+19)//20
    if problem_type=='classification':
        if n==0:raise ValueError('Cannot partition an empty training target')
        groups,meta=strata(y,problem_type)
        counts=np.bincount(groups)
        classes=len(counts)
        if np.any(counts<3):
            scarce={label:int(count) for label,count in zip(meta['labels'],counts) if count<3}
            raise ValueError(f'P, V_core and Q each require at least 1 row per class, requiring at least 3 rows per class; insufficient classes: {scarce}')
    elif problem_type=='regression':
        classes=1
    else:
        raise ValueError(f'Unsupported problem_type: {problem_type}')
    a=max(2,classes,requested_a);v=max(classes,requested_v,validation_floor);p=max(2,classes,requested_p)
    if cfg.proxy_train_rows is None and a>cfg.proxy_train_max_rows:
        raise ValueError(f'P class coverage exceeds configured maximum {cfg.proxy_train_max_rows}; increase its maximum or request explicit rows')
    if a+v+p>n:
        raise ValueError(f'No feasible P/V_core/coreset sizes: N={n}, P={a}, V_core={v}, coreset={p}; require P+V_core+coreset<=N. Reduce explicit rows or minimum/ratio settings; sizes are not silently shrunk')
    if problem_type=='regression':
        groups,meta=strata(y,problem_type,cfg.regression_bins,min(a,v,p),3)
    counts=np.bincount(groups)
    # P reserves at least one row per stratum for both V and Q. Next reserve
    # the entire coreset capacity while retaining V coverage before sampling V.
    pq=split_quotas(counts,a,cfg.split_seed,upper=counts-2)
    remaining_counts=counts-pq
    reserved=split_quotas(remaining_counts,p,cfg.split_seed+1,upper=remaining_counts-1)
    vq=split_quotas(remaining_counts,v,cfg.split_seed+2,upper=remaining_counts-reserved)
    ppos=stratified_sample(groups,pq,cfg.split_seed)
    remaining=np.setdiff1d(np.arange(n),ppos)
    vpos=remaining[stratified_sample(groups[remaining],vq,(cfg.split_seed+1)%(2**32))]
    qpos=np.setdiff1d(remaining,vpos)
    if len(qpos)<p or len(np.unique(np.concatenate([ppos,vpos,qpos])))!=n:
        raise ValueError('Internal partition invariant failed')
    qgroups,qmeta=strata(y.iloc[qpos],problem_type,cfg.regression_bins,max_strata=p)
    qcounts=np.bincount(qgroups)
    final_quota=split_quotas(qcounts,p,cfg.split_seed+3)
    info={'requested_proxy_rows':requested_a,'actual_proxy_rows':len(ppos),'requested_coreset_rows':requested_p,
          'requested_validation_rows':requested_v,'actual_coreset_rows':p,'validation_rows':v,'q_rows':len(qpos),
          'v_over_p':v/p,'v_over_train':v/n,'adjustments':[],'strata':meta,
          'validation_requires_class_coverage':problem_type=='classification',
          'allocation_ties':'seeded permutation; proportional quotas with capacity bounds',
          'stratum_counts':{'train':counts.tolist(),'proxy':pq.tolist(),'validation':vq.tolist(),
                            'q':(remaining_counts-vq).tolist(),'reserved_coreset':reserved.tolist()},
          'coreset_quotas':{'strata':qmeta,'counts':qcounts.tolist(),'allocation':final_quota.tolist()}}
    for requested,actual,name in [(requested_a,a,'P'),(requested_v,v,'V_core'),(requested_p,p,'coreset')]:
        if requested!=actual:
            reason='5% validation floor' if name=='V_core' and actual==validation_floor else 'coverage or representation minimum'
            info['adjustments'].append(f'{name} increased for {reason}')
    return np.sort(ppos),np.sort(vpos),qpos,info


def proxy_select(importance, graph, groups, p, penalty, neighbors, classification,
                 allocation=None, quota_seed=2026):
    """Optimize importance/diversity within class quotas; regression is global."""
    if classification:
        groups=np.asarray(groups)
        _,groups=np.unique(groups,return_inverse=True)
        counts=np.bincount(groups)
        if p<2 or p>len(importance) or p<len(counts) or neighbors<1:
            raise ValueError('Infeasible proxy coreset size or class coverage')
        if allocation is None:
            allocation=split_quotas(counts,p,quota_seed)
        allocation=np.asarray(allocation)
        if (allocation.shape!=counts.shape or not np.issubdtype(allocation.dtype,np.integer)
                or allocation.sum()!=p or np.any(allocation<1) or np.any(allocation>counts)):
            raise ValueError('Infeasible proportional class quotas')
        return greedy_select(importance,graph,groups,allocation,penalty,
                             normalization_degree=neighbors)
    n=len(importance)
    groups=np.asarray(groups)
    _,groups=np.unique(groups,return_inverse=True)
    covered=np.zeros(len(np.unique(groups)),dtype=bool)
    missing=len(covered) if classification else 0
    if p<2 or p>n or missing>p or neighbors<1:
        raise ValueError('Infeasible proxy coreset size or class coverage')
    degree=min(neighbors,p-1)
    selected=np.zeros(n,dtype=bool)
    versions=np.zeros(n,dtype=np.int64)
    similarity_sum=np.zeros(n)
    heap=[(-float(value)/p,i,0) for i,value in enumerate(importance)]
    heapq.heapify(heap)
    chosen=[];gains=[]
    factor=2*penalty/(p*degree)
    while len(chosen)<p:
        if not heap:raise ValueError('Cannot fill proxy coreset with class coverage')
        negative_gain,i,version=heapq.heappop(heap)
        if selected[i] or version!=versions[i]:continue
        if classification and p-len(chosen)==missing and covered[groups[i]]:continue
        selected[i]=True;chosen.append(i);gains.append(-negative_gain)
        if classification and not covered[groups[i]]:
            covered[groups[i]]=True;missing-=1
        start,end=graph.indptr[i:i+2]
        for j,weight in zip(graph.indices[start:end],graph.data[start:end]):
            if not selected[j]:
                similarity_sum[j]+=weight;versions[j]+=1
                gain=importance[j]/p-factor*similarity_sum[j]
                heapq.heappush(heap,(-float(gain),int(j),int(versions[j])))
    chosen=np.asarray(chosen,dtype=np.int64)
    value=objective(importance,graph,chosen,p,penalty,normalization_degree=neighbors)
    if missing or not np.isclose(sum(gains),value,rtol=1e-9,atol=1e-10):
        raise ValueError('Proxy coverage or incremental objective invariant failed')
    return chosen,{'objective':value,'importance_term':float(np.mean(importance[chosen])),
        'similarity_penalty':float(penalty*graph[chosen][:,chosen].sum()/(p*degree)),
        'gain_sum':float(sum(gains)),'negative_gain_steps':int(np.sum(np.asarray(gains)<0)),
        'normalization':'sparse_degree','normalization_degree':degree,'normalization_denominator':p*degree}


def centered_density_importance(density, hardness, eta=.25):
    """Full-Q density-weighted centering; retain mean D, without reranking."""
    density=np.asarray(density,dtype=float);hardness=np.asarray(hardness,dtype=float)
    if (density.shape!=hardness.shape or density.ndim!=1 or not 0<=eta<=1
            or not np.isfinite(density).all() or not np.isfinite(hardness).all()
            or np.any((density<0)|(density>1)) or np.any((hardness<0)|(hardness>1))):
        raise ValueError('Invalid centered density importance inputs')
    total=float(density.sum())
    mu=float(np.dot(density,hardness)/total) if total else 0.
    values=density*(1+eta*(hardness-mu))
    if not np.isclose(values.sum(),total,rtol=1e-12,atol=1e-10):
        raise ValueError('Centered importance failed mean preservation')
    return values,mu


def relative_density_importance(density, hardness_matrix, graph, method, *, local=False):
    """B/C: leave-one-method-out percentile contrast, optionally graph smoothed."""
    h=np.asarray(hardness_matrix,dtype=float)
    if (h.ndim!=2 or h.shape[0]<2 or not 0<=method<h.shape[0]
            or h.shape[1]!=len(density) or not np.isfinite(h).all()
            or np.any((h<0)|(h>1))):
        raise ValueError('Invalid relative hardness matrix')
    utility=np.maximum(h[method]-np.median(np.delete(h,method,axis=0),axis=0),0.)
    if local:
        degree=np.asarray(graph.sum(axis=1)).ravel()
        neighbors=np.divide(graph@utility,degree,out=utility.copy(),where=degree>0)
        utility=.5*(utility+neighbors)
    values,mu=centered_density_importance(density,utility,.25)
    return values,mu


def select_proxy_variant(importance, density_importance, graph, groups, allocation,
                         cfg, classification, gmm_order, *, hardness_matrix=None, method_index=None):
    """Use the original full-Q importance arrays without reranking."""
    p = int(sum(allocation))
    scheme = cfg.proxy_selection_scheme
    if scheme == 'original':
        return proxy_select(importance, graph, groups, p, cfg.similarity_lambda,
                            cfg.neighbors, classification, allocation=allocation)
    initial = np.array([], dtype=np.int64)
    values = np.asarray(importance)
    extra = {'scheme': scheme}
    if scheme == 'gmm_anchor80':
        # Largest-remainder bounded allocation preserves GMM strata approximately;
        # retain the original greedy order within each stratum and globally.
        body_size = int(np.floor(0.8*p + 0.5))
        body_quota = split_quotas(np.asarray(allocation), body_size, cfg.split_seed+4,
                                 lower=np.zeros(len(allocation), dtype=int),
                                 upper=np.asarray(allocation))
        used = np.zeros(len(allocation), dtype=int)
        body = []
        for index in gmm_order:
            group = groups[index]
            if used[group] < body_quota[group]:
                body.append(int(index)); used[group] += 1
        initial = np.asarray(body, dtype=np.int64)
        if len(initial) != body_size:
            raise ValueError('Cannot fill GMM shared body')
        extra.update(body_q_indices=body, body_quotas=body_quota.tolist(),
                     body_size=body_size, final_size=p, formula='original H; GMM shared body')
    elif scheme in {'density_relative','density_relative_local'}:
        values,mu=relative_density_importance(density_importance,hardness_matrix,graph,
            method_index,local=scheme=='density_relative_local')
        extra.update(formula='D*(1+.25*(U-weighted_mean_D(U))); no reranking',
            utility='positive H minus median of other proxies',eta=.25,mu=mu,
            neighborhood_weight=.5 if scheme=='density_relative_local' else 0.,
            isolated_nodes='own relative hardness',mu_scope='full Q; density-weighted mean U')
    elif scheme == 'density_centered_eta025':
        values,mu=centered_density_importance(density_importance,values,.25)
        extra.update(formula='D * (1 + eta * (H - mu)); no reranking',eta=.25,
                     mu=mu,mu_scope='full Q; density-weighted mean H')
    else:
        values = np.asarray(density_importance)*(1+values)/2
        extra['formula'] = 'D * (1 + H) / 2; no reranking'
    select_groups = groups if classification else np.zeros(len(values), dtype=int)
    select_quotas = allocation if classification else np.asarray([p])
    chosen, info = greedy_select(values, graph, select_groups, select_quotas,
                                cfg.similarity_lambda, normalization_degree=cfg.neighbors,
                                initial=initial)
    return chosen, {**info, **extra}


def gmm_density_importance(z_p, z_q, groups, cfg):
    """Fit only P features; rank global GMM log density within Q strata."""
    components=min(cfg.gmm_components,len(z_p))
    gmm=GaussianMixture(n_components=components,covariance_type='diag',
                        max_iter=cfg.gmm_max_iter,n_init=1,reg_covar=1e-6,
                        random_state=cfg.gmm_seed)
    with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
        warnings.simplefilter('always')
        gmm.fit(z_p)
    if not gmm.converged_:
        raise ValueError(f'GMM did not converge in {cfg.gmm_max_iter} iterations; no fallback')
    density=np.empty(len(z_q))
    for begin in range(0,len(z_q),cfg.batch_size):
        density[begin:begin+cfg.batch_size]=gmm.score_samples(z_q[begin:begin+cfg.batch_size])
    if not np.isfinite(density).all():
        raise ValueError('GMM returned nonfinite log density')
    importance=np.empty(len(z_q))
    for group in np.unique(groups):
        mask=groups==group
        count=int(mask.sum())
        importance[mask]=(rankdata(density[mask],method='average')-1)/(count-1) if count>1 else .5
    report={'status':'success','fit_set':'P','fit_rows':len(z_p),'score_set':'Q',
            'prediction_rows':len(z_q),'requested_components':cfg.gmm_components,
            'actual_components':components,'covariance_type':'diag','max_iter':cfg.gmm_max_iter,
            'n_init':1,'reg_covar':1e-6,'seed':cfg.gmm_seed,'converged':True,
            'iterations':int(gmm.n_iter_),'warnings':[str(w.message) for w in caught],
            'normalization':'within-stratum average percentile rank of global GMM log density',
            'scope':'density representativeness heuristic; not guaranteed distribution coverage or learning benefit'}
    return gmm,density,importance,report


def error_importance(y, predictions, probabilities, classes, problem_type, clip_quantile=.95):
    if problem_type=='regression':
        error=np.abs(np.asarray(y,dtype=float)-np.asarray(predictions,dtype=float))
    else:
        indices=pd.Index(classes).get_indexer(y)
        if (indices<0).any():raise ValueError('Q contains a label absent from proxy P')
        onehot=np.zeros_like(probabilities)
        onehot[np.arange(len(y)),indices]=1
        error=np.linalg.norm(probabilities-onehot,axis=1)
    if not np.isfinite(error).all():raise ValueError('Nonfinite proxy error')
    cutoff=float(np.quantile(error,clip_quantile))
    clipped=np.minimum(error,cutoff)
    importance=(rankdata(clipped,method='average')-1)/(len(error)-1) if len(error)>1 else np.full(1,.5)
    return error,importance,cutoff


def mutual_graph(z,cfg):
    """One graph on ALL Q, projection-sorted blocks and mutual kNN."""
    rng=np.random.default_rng(cfg.graph_seed)
    direction=rng.normal(size=z.shape[1])
    direction_norm=np.linalg.norm(direction)
    if not np.isfinite(direction_norm) or direction_norm==0:
        raise ValueError('Similarity representation requires at least one finite projection dimension')
    direction/=direction_norm
    projection=z@direction
    # Shuffle ties reproducibly, then keep nearby projection values in each block.
    tie_order=rng.permutation(len(z))
    permutation=tie_order[np.argsort(projection[tie_order],kind='stable')]
    # Per-block arrays avoid millions of Python edge objects on large Q.
    edge_rows=[];edge_cols=[];edge_distances=[]
    for start in range(0,len(z),cfg.graph_block_size):
        pos=permutation[start:start+cfg.graph_block_size]
        k=min(cfg.neighbors,len(pos)-1)
        if k<1:continue
        block=z[pos]
        nn=NearestNeighbors(n_neighbors=k+1,algorithm='brute',n_jobs=1).fit(block)
        rows=[];cols=[];distances=[]
        for begin in range(0,len(pos),256):
            d,ix=nn.kneighbors(block[begin:begin+256])
            for local,(ds,ns) in enumerate(zip(d,ix),start=begin):
                keep=ns!=local
                ns,ds=ns[keep][:k],ds[keep][:k]
                rows.extend([local]*len(ns));cols.extend(ns.tolist());distances.extend(ds.tolist())
        row,col=np.asarray(rows),np.asarray(cols)
        adjacency=sparse.csr_matrix((np.ones(len(row),dtype=bool),(row,col)),shape=(len(pos),len(pos)))
        mutual=np.asarray(adjacency[col,row]).ravel().astype(bool)
        edge_rows.append(pos[row[mutual]]);edge_cols.append(pos[col[mutual]])
        edge_distances.append(np.asarray(distances)[mutual])
    rows=np.concatenate(edge_rows) if edge_rows else np.array([],dtype=int)
    cols=np.concatenate(edge_cols) if edge_cols else np.array([],dtype=int)
    distances=np.concatenate(edge_distances) if edge_distances else np.array([])
    positive=distances[distances>0]
    bandwidth=float(np.median(positive)) if len(positive) else 1.
    graph=sparse.csr_matrix((np.exp(-.5*(distances/bandwidth)**2),(rows,cols)),shape=(len(z),len(z)))
    graph=graph.minimum(graph.T).tocsr();graph.setdiag(0);graph.eliminate_zeros();graph.sort_indices()
    degrees=np.diff(graph.indptr)
    if (graph!=graph.T).nnz or degrees.max(initial=0)>cfg.neighbors:
        raise ValueError('Mutual graph symmetry/degree invariant failed')
    return graph,{'q_rows':len(z),'bandwidth':bandwidth,'bandwidth_rule':'median positive mutual-neighbor distance; 1 if no positive distance',
                  'block_seed':cfg.graph_seed,'block_size':cfg.graph_block_size,'blocks':math.ceil(len(z)/cfg.graph_block_size),
                  'partition_method':'single_random_projection_sorted','projection_seed':cfg.graph_seed,
                  'projection_direction':direction.tolist(),'projection_ties':'seeded permutation, stable sort',
                  'neighbors':cfg.neighbors,'actual_max_degree':int(degrees.max(initial=0)),'nnz':graph.nnz,
                  'approximation':'projection-sorted block-local mutual kNN; cross-block edges zero; every Q row remains eligible'}


def eda_summary(frame,task,excluded,cfg):
    summary=training_profile(frame,task['target'],task['problem_type'])
    summary.update(scope='P plus Q, excluding V_core',task_type=task['problem_type'],target=task['target'])
    columns={}
    for c in frame.columns:
        if c==task['target']:continue
        values=frame[c]
        constant=values.nunique(dropna=True)<=1
        columns[c]={'dtype':str(values.dtype),'missing':int(values.isna().sum()),'constant':constant}
        if constant:excluded[c]='constant outside V_core'
    summary['columns']=columns;summary['excluded_fields']=excluded
    if task['problem_type']=='classification':
        summary['target_distribution']={str(k):int(v) for k,v in frame[task['target']].value_counts().items()}
    features=[c for c in frame if c != task['target'] and c not in excluded]
    categorical=[c for c in features if c in cfg.categorical_columns or c in task.get('categorical_columns',[]) or not pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c])]
    summary['feature_columns']=features;summary['categorical_columns']=categorical
    summary['numeric_columns']=[c for c in features if c not in categorical]
    if not features:raise ValueError('No effective method features')
    summary['feature_metadata']=feature_metadata_for_task(task,features)
    summary['feature_structure']=feature_structure_profile(
        frame,feature_columns=features,categorical_columns=categorical,
        scope='P + Q only; V_core, external validation and test excluded',
        metadata=summary['feature_metadata'])
    summary['description']=describe_eda(summary)
    return summary


def method_task_context(task,eda,size_info,pframe,quota_info,cfg):
    """Aggregate-only generation context; actual fit_rows is supplied by the worker."""
    context={k:task[k] for k in ('name','target','problem_type','metric','metric_params','description') if k in task}
    context.update({k:eda[k] for k in ('feature_columns','numeric_columns','categorical_columns')})
    context['feature_metadata']=eda.get('feature_metadata',{})
    classes=pd.Index(pframe[task['target']].unique()).sort_values().tolist() if task['problem_type']=='classification' else []
    context.update(classes=classes,fe_config={'max_categories':cfg.max_categories},
                   feature_count=len(eda['feature_columns']))
    context['training_context']={
        'fixed_train_rows':size_info['actual_proxy_rows']+size_info['validation_rows']+size_info['q_rows'],
        'proxy_train_rows':size_info['actual_proxy_rows'],'coreset_rows':size_info['actual_coreset_rows'],
        'candidate_rows':size_info['q_rows'],'core_validation_rows':size_info['validation_rows'],
        'minimum_fit_rows':min(size_info['actual_proxy_rows'],size_info['actual_coreset_rows']),
        'n_classes':len(classes),'proxy_class_counts':[
            {'label':label,'count':int((pframe[task['target']]==label).sum())} for label in classes],
        'coreset_quotas':{**quota_info,'applies_to':'all_coresets' if classes else 'gmm_density_only'},
        'proxy_selection_policy':'classification: maximize importance/diversity within proportional Q class quotas; regression: global without bin quotas',
    }
    if cfg.proxy_selection_scheme == 'edge_mechanism_comparison':
        context['training_context']['proxy_selection_policy'] = (
            'Four shared coresets of the stated coreset_rows: three probability draws and one deterministic greedy selection. '
            'All use density plus mean direct error from all five proxies, with selected-value-weighted edge deductions. '
            'Classification: fixed class x GMM component quotas; regression: fixed GMM component quotas. '
            'One task-level GMM K and identical model stage parameters across all four coresets.')
        context['training_context']['coreset_quotas']['applies_to'] = (
            'class totals shared across all four arms, then subdivided by GMM component mass' if classes
            else 'target-bin allocation is partition metadata only; selection uses GMM component mass')
    if cfg.proxy_selection_scheme in {'marginal_coverage_random_comparison', 'marginal_coverage_dual_random_comparison', 'marginal_coverage_only'}:
        context['training_context']['proxy_selection_policy'] = (
            'Six shared coresets: marginal coverage probability (a=.3, beta=.6, lambda=2, rho=1) '
            'and stratified random, each with three paired sampling seeds. Coverage uses fixed K=5 '
            'and class x component quotas (classification) or component quotas (regression). '
            'Random uses class quotas (classification) or target quantile quotas (regression). '
            'All six share the same model stage parameters and held-out V_core.')
        context['training_context']['coreset_quotas']['applies_to'] = (
            'same class totals across six arms' if classes else 'target quantile quotas apply only to random arms')
        if cfg.proxy_selection_scheme == 'marginal_coverage_dual_random_comparison':
            context['training_context']['proxy_selection_policy'] = (
                'Nine shared coresets: coverage probability (a=.3,beta=.6,lambda=2) and '
                '(a=.3,beta=.3,lambda=.6), rho=1 and K=5; stratified random; three paired '
                'sampling seeds each. All nine share model parameters, P/Q/V_core and fit seeds.')
            context['training_context']['coreset_quotas']['applies_to'] = (
                'same class totals across nine arms' if classes else 'target quantile quotas apply only to random arms')
    if cfg.proxy_selection_scheme == 'marginal_coverage_only':
        context['training_context']['proxy_selection_policy'] = ('One fixed marginal-coverage probability coreset: a=.3,beta=.6,lambda=2,rho=1,K=5; sampling seed 20261026. No random baseline. Five proxies fit P/predict Q; freeze parameter feedback before subsequent formal evaluation.')
        context['training_context']['coreset_quotas']['applies_to'] = 'class x component quotas' if classes else 'component quotas'
    if cfg.validation_guided_mix:
        context['training_context'].update(proxy_selection_policy='Fixed size: 10% validation-distribution MMD, 5% per-method validation errors, 85% marginal coverage probability. Skill actions affect only the editable portion.')
        context['training_context']['coreset_quotas']['applies_to']='Target group counts for filling editable slots after protected rows'
    return context


async def llm_json(model,prompt,out,costs,pricing):
    from .generation_recovery import recover_json
    check_deadline()
    return await await_with_budget(recover_json(_llm_json_once, model, prompt, out, costs, pricing))


async def _llm_json_once(model,prompt,out,costs,pricing):
    """Reuse the existing adapter's client; retain usage that its text API discards."""
    check_deadline()
    pricing = pricing or {}
    started=time.perf_counter()
    write_json(out.with_suffix('.request.json'),{'prompt':prompt,'model':model.model_name,
        'request_options':getattr(model,'request_options',{})})
    entry={'request_id':uuid.uuid4().hex,'call':out.stem,'model':model.model_name,'status':'running','started_at':utc_now()}
    costs.append(entry)
    write_json(out.with_suffix('.usage.json'),entry)
    try:
        request=model.client.with_options(max_retries=0).chat.completions.create(
            model=model.model_name,messages=[{'role':'system','content':'Return one valid JSON object. Follow the exact method interface. Do not use external services.'},{'role':'user','content':prompt}],
            temperature=model.temperature,max_tokens=model.max_tokens,response_format={'type':'json_object'},
            **getattr(model,'request_options',{}))
        response=await await_with_budget(request,default_timeout=model.timeout)
        usage=response.usage.model_dump() if response.usage else {}
        entry.update(status='success',usage=usage,finish_reason=response.choices[0].finish_reason)
        entry['response_model']=getattr(response,'model',None)
        entry['response_id']=getattr(response,'id',None)
        if pricing and all(k in pricing for k in ('input_per_million','output_per_million')):
            entry['estimated_cost']=(usage.get('prompt_tokens',0)*pricing['input_per_million']+usage.get('completion_tokens',0)*pricing['output_per_million'])/1e6
            entry['currency']=pricing.get('currency','unspecified')
            entry['cost_basis']='user configured token rates; not provider invoice'
        else:
            entry['estimated_cost']=None
            entry['cost_basis']='token usage recorded; monetary rates not configured'
        content=response.choices[0].message.content
        out.with_suffix('.response.txt').write_text(content or '')
        if response.choices[0].finish_reason!='stop':raise ValueError('LLM response incomplete: '+str(response.choices[0].finish_reason))
        from .generation_recovery import parse_object
        normalizations=[]
        result=parse_object(content, normalizations)
        entry['local_normalizations']=normalizations
        write_json(out,result)
        return result
    except BaseException as e:
        entry.update(status='failed',error=f'{type(e).__name__}: {e}')
        chain=[]; cause=e; seen=set()
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause)); chain.append({'type':type(cause).__name__,'message':str(cause)})
            cause=cause.__cause__ or cause.__context__
        entry['error_chain']=chain
        raise
    finally:
        entry['wall_seconds']=time.perf_counter()-started
        entry['finished_at']=utc_now()
        if is_direct_deepseek(getattr(model.client,'base_url','')):
            entry['deepseek_cost']=estimate(model.model_name,entry.get('usage'),pricing.get('deepseek_period','unknown'))
        write_json(out.with_suffix('.usage.json'),entry)


_METHOD_CONTRACT_TEMPLATE='''Generate a single complete FE+MT implementation as JSON {"code": "Python source", "fe_config": {...}, "model_config": {...}}.
Define build_method(task, budget, seed) returning a NEW unfitted sklearn.pipeline.Pipeline whose last step is named "model".
The runner invokes fit(X,y), predict(X), and for classification predict_proba(X) and classes_. X is a pandas DataFrame.
Use task['numeric_columns'], task['categorical_columns'], task['feature_columns'] to select columns. Never include target, IDs or leakage fields.
Numeric input infinities are converted to NaN by the runner; handle missing values. Use unknown-safe category encoding and task['fe_config']['max_categories'] for cardinality limits; avoid large dense matrices. Ensure estimator accepts its FE output.
Fit all FE only as part of pipeline.fit on supplied training rows. No preprocessing fit, model fit, prediction, IO, APIs, search, cross-validation, probe models or dataset access in factory/source. Do not read global data or retrieve experience.
Only import sklearn, numpy, pandas, scipy or math. No eval/exec/open/getattr/setattr. Prefer standard serializable sklearn transformers. No custom estimator classes or fitted objects. No FunctionTransformer lambdas.
__RESOURCE_RULE__ Set random_state=seed, n_jobs=1 and warm_start=False ONLY when that exact estimator supports those constructor parameters. HistGradientBoosting and AdaBoost do NOT accept n_jobs; AdaBoost does NOT accept warm_start. Use the supplied sklearn version's API.
Respect wall time as well as tree/epoch caps. In particular, absolute_error tree criteria and unrestricted exact-split forests can be very costly on large tables. Choose feasible loss, depth, leaf size and feature expansion at initial design; do not change loss/model after scoring. The feedback stage may predeclare fewer trees/epochs within caps using P timing and actual stage sizes. Timing extrapolation is only a risk estimate, not a completion guarantee.
''' + model_contract() + '''
task['training_context'] contains actual fixed-train/P/coreset/Q sizes, proxy class counts, and proportional Q coreset quotas; all classification coresets use the same class quotas and optimize importance/diversity within them; regression proxies use no bin quotas and regression GMM retains bin quotas; EDA train_rows is P+Q, NOT the number of rows used to fit a model.
At each factory call the runner provides task['fit_rows'], task['fit_class_counts'] (list of {label,count}), and task['fit_stage'] ('proxy', 'evaluation', or 'full_train'). Use actual fit sizes for proxy safeguards; evaluation/full_train changes must be concrete declared stage_parameters, with valid FE dimensions and structural parameters such as min_samples_leaf, min_samples_split and n_neighbors. For small P, explicitly set tree leaf sizes small enough to split (e.g. 2 or 3 when supported by fit_rows); do not rely on HistGradientBoosting's default 20. Neighbors must not exceed fit_rows. The runner will NOT repair these structural choices.
Same code and modeling mechanism must work on proxy P and every coreset. Use actual fit size for structural parameter rules and budget for tree/iteration limits; do not hardcode scoring limits into formal evaluation.
For evaluation/full_train the coordinator applies idea.stage_parameters.coreset/full_train via Pipeline.set_params before fit. Provide concrete JSON-compatible pipeline parameter paths (e.g. model__max_depth, model__min_samples_leaf, model__C). __STAGE_RULE__ The full_train factory must not introduce changes beyond explicit stage_parameters and runner budget controls; the runner compares against the unfitted coreset pipeline with those overrides applied. Class counts may differ across all coresets. Same method must return the same complete pre-fit Pipeline configuration for all coreset evaluation fits; do not adapt parameters to each coreset class count. Data-dependent fitted statistics (including class_weight=balanced) may differ and must be reported as such. No coreset-specific adaptation or use of validation results. Keep FE mechanism and exact model class the same for full_train; permit only predeclared parameter changes. __EXACT_RULE__ No claim that equal resource caps ensure equal convergence.
Faithfully implement the selected strategy.idea. Its fixed_parameters must remain fixed across fit sizes. Use fit_rows/fit_class_counts/fit_stage only for the explicit scaling_rules and stage_budgets declared in that idea. Do not invent new parameter thresholds or silently change the FE or model mechanism. Treat rationale and anticipated benefits as hypotheses, not measured evidence. Return complete executable code, not pseudocode or markdown fences.'''


def method_contract(parameter_budget_mode='common', *, output_format=True, formal_only=False):
    """One unambiguous contract for the actual executor mode."""
    if parameter_budget_mode == 'exact':
        parameter_budget_mode = 'common'
    if parameter_budget_mode not in {'common', 'planned'}:
        raise ValueError('Unknown parameter budget mode')
    if parameter_budget_mode == 'planned':
        resource = ("This request uses planned mode. budget['max_trees'] and budget['max_epochs'] are resource caps, "
                    "not mandatory counts. Use feasible declared counts within those caps. Linear models fit normally.")
        stage = ("stage_parameters override factory defaults before fitting. Explicit model__n_estimators and "
                 "model__max_iter are allowed within stage caps. Counts below the cap are preserved, never automatically "
                 "increased to the cap. random_state, n_jobs, warm_start and early_stopping remain runner-managed; "
                 "do not declare them in stage_parameters. All parameter claims must match actual execution.")
        exact = ''
    else:
        resource = "Use budget['max_trees'] for ensemble n_estimators/HistGradientBoosting max_iter (default100), budget['max_epochs'] for MLP max_iter (default5, early_stopping=False). Linear models fit normally."
        stage = 'Do not include n_estimators, random_state, n_jobs, warm_start, early_stopping in stage_parameters: the runner owns these controls. Linear solver model__max_iter may be declared in stage_parameters; tree/MLP max_iter is overridden by the common budget.'
        exact = 'Tree counts and MLP epochs are set exactly to the common stage budget for evaluation/full_train.'
    contract = (_METHOD_CONTRACT_TEMPLATE.replace('__RESOURCE_RULE__', resource)
            .replace('__STAGE_RULE__', stage).replace('__EXACT_RULE__', exact))
    if not output_format:
        # Planning and error repair have their own JSON schemas. Reuse only
        # execution constraints, never a competing implementation schema.
        contract = contract.split('\n', 1)[1]
        contract = contract.removesuffix(' Return complete executable code, not pseudocode or markdown fences.')
    if formal_only:
        lines = contract.splitlines()
        for index, line in enumerate(lines):
            if line.startswith('At each factory call'):
                lines[index] = ("This candidate runs only on coreset ('evaluation') and full_train. "
                    "Use task['fit_rows'] and task['fit_stage'] for the declared formal-stage rules. "
                    "Both stages must be valid at their actual sizes; n_neighbors must not exceed fit_rows. "
                    "No proxy P fit or proxy resource cap applies to this candidate.")
            elif line.startswith('Same code and modeling mechanism must work on proxy P'):
                lines[index] = ('Keep the same code and modeling mechanism across coreset/full_train. '
                    'Declare every difference in concrete stage_parameters; respect each stage resource cap.')
        contract = '\n'.join(lines)
    return contract


# Existing non-MCTS callers retain their common-budget contract.
METHOD_CONTRACT = method_contract()


async def generate_methods(model,task_context,eda,experiences,cfg,out,costs,pricing,checkpoint=None,regeneration_state_path=None):
    from .method_regeneration import generate_one, regenerate
    from .model_registry import normalize_strategy_family
    regeneration_state_path = regeneration_state_path or out/'regeneration_state.json'
    budgets = dict(proxy=cfg.scoring_budget, coreset=cfg.evaluation_budget, full_train=cfg.full_train_budget)
    guarded = cfg.parameter_feedback and cfg.parameter_feedback_mode == 'guarded_once'
    plan_prompt = f'Propose exactly {cfg.n_strategies} distinct, strong FE+model methods for the full-training metric. ' + """Use training EDA, actual training sizes and resource caps. Ensure each pipeline works on P and coreset.
Ensure transform domains cover the supplied numeric ranges, including negative values for log transforms.
Prefer meaningful differences in FE or estimator. Families may repeat. Use unique names and supported families
(linear, bagging, boosting, neural, neighbors, naive_bayes, tree). Use only supplied training evidence.
Return JSON strategies with name, family, idea. Inside idea return exactly:
summary (one short Chinese sentence linking a specific EDA observation to the FE and estimator choice), model (exact estimator class), feature_engineering (ordered concrete steps),
fixed_parameters (settings not already supplied in stage_parameters), scaling_rules ({} or exact conditions),
stage_parameters (concrete pipeline parameter maps for coreset and full_train, subject to the mode below).
The runner supplies stage budgets. Use the ordered FE steps and parameters as the implementation specification. Omit preambles, repeated settings, generic caveats and claims of expected superiority.
Implementation contract:
""" + method_contract(output_format=False) + '\n' + json.dumps(dict(task=task_context, eda=eda,
        historical_experiences=experiences, budgets=budgets), ensure_ascii=False, separators=(',', ':'))
    fixed_gmm = cfg.proxy_selection_scheme in {'marginal_coverage_random_comparison',
                    'marginal_coverage_dual_random_comparison', 'marginal_coverage_only'}
    fixed_design = {'gmm_components': 5, 'rationale': 'Fixed coverage protocol.'}
    if not fixed_gmm:
        plan_prompt = GMM_PLAN_INSTRUCTION + '\n' + plan_prompt
    plan_prompt += f'\nAll methods use the same V_core for all {cfg.n_strategies * cfg.n_coresets} fits.'
    plan_prompt += ('\neda.feature_structure provides bounded, sampled training-feature statistics. '
                    'Use its sampling scope and confirmed_feature_metadata when interpreting features. '
                    'Explicit current units and already-applied preprocessing override ambiguous descriptions; '
                    'never repeat a recorded conversion or infer physical units/invariances from column names. '
                    'Unreported columns are not certified normal, and descriptive distribution shifts do not measure model generalization.')
    if cfg.parameter_feedback and cfg.parameter_feedback_mode == 'guarded_once':
        plan_prompt += ('\nThis request uses guarded_once feedback. Generate ONLY the full-data parameter design. '
                        'Return only full_train inside stage_parameters; the runner copies it into coreset; '
                        'set scaling_rules to {}. Put feature normalization (e.g. StandardScaler) in feature_engineering. '
                        'The same frozen full pipeline will first fit P; only strong P/Q underfitting evidence '
                        'may authorize one mild single-parameter change and one P refit before coreset selection. '
                        'That change applies to P and coreset only. Full parameters remain immutable. '
                        'Resource caps for both P and coreset match full-stage caps; wall-time limits still apply. '
                        'Preserve this policy in the implementation and use the supplied P EDA and training sizes.')
    elif cfg.parameter_feedback:
        plan_prompt += ('\nFeedback workflow: the P proxy uses the supplied low-cost scoring budget. '
                        'task.proxy_eda describes the actual P training rows. The generated coreset/full_train '
                        'parameters are provisional: one later request will use real P/Q feedback and actual '
                        'coreset EDA to freeze model parameter rules. Preserve the FE mechanism and exact model class. '
                        'No V_core or external validation observations will be provided for that revision.')
    if fixed_gmm:
        plan_prompt += '\nGMM K=5 is fixed by the runner. Return only strategies.'
    def validate_plan(plan):
        if not isinstance(plan,dict) or not isinstance(plan.get('strategies'),list) or len(plan['strategies'])!=cfg.n_strategies:
            raise ValueError('Plan must contain the expected number of identifiable strategies')
        validate_gmm_design(fixed_design if fixed_gmm else plan.get('coreset_design'),task_context)
    # Separate bounded semantic regeneration from punctuation-only JSON recovery.
    # The whole plan has not been frozen or evaluated at this point.
    plan=await generate_one('whole_plan','plan',plan_prompt,out/'plan.json',
                            regeneration_state_path,model,costs,pricing,validate_plan)
    coreset_design=validate_gmm_design(fixed_design if fixed_gmm else plan.get('coreset_design'),task_context)
    write_json(out/'coreset_design.json',coreset_design)
    methods=plan['strategies']
    normalizations=[]
    allowed_families={'linear','bagging','boosting','neural','neighbors','naive_bayes','tree'}
    for i,original in enumerate(list(methods)):
        frozen_identity={}
        if isinstance(original,dict):
            name=original.get('name')
            if isinstance(name,str) and name.strip() and not any(
                    j!=i and isinstance(other,dict) and other.get('name')==name
                    for j,other in enumerate(methods)):
                frozen_identity['name']=name
            if isinstance(original.get('family'),str) and original['family'] in allowed_families:
                frozen_identity['family']=original['family']
        def validate_strategy(response):
            candidate=response['strategy']
            normalized,_=normalize_plan({'strategies':[candidate]}, budgets=budgets, mirror_full=guarded)
            method=normalized['strategies'][0]
            for key in (('name','family') if method['idea'].get('plan_format') == 'brief_v1' else ('name','family','reason','fe_config','model_config')):
                if key not in method:raise ValueError('Incomplete Plan strategy: '+key)
            if not isinstance(method['name'],str) or not method['name'].strip():
                raise ValueError('Plan strategy requires a nonempty name')
            if not isinstance(method['family'],str) or method['family'] not in allowed_families:
                raise ValueError('Unsupported Plan family: '+str(method['family']))
            for key,value in frozen_identity.items():
                if method[key]!=value:
                    raise ValueError('Single-method regeneration must preserve '+key)
            for j,other in enumerate(methods):
                if j!=i and isinstance(other,dict) and method['name']==other.get('name'):
                    raise ValueError('Regenerated method duplicates another name')
            validate_idea(method.get('idea'))
            validate_stage_parameters(method['idea'])
            stages = method['idea']['stage_parameters']
            if guarded and stages['coreset'] != stages['full_train']:
                raise ValueError('guarded_once requires identical frozen initial stage parameters')
        selected=out/f'plan.method_{i}.selected.json'
        source_identity=identity(original)
        if selected.exists():
            saved=json.loads(selected.read_text())
            if saved['source_identity']!=source_identity:raise ValueError('Regenerated strategy identity mismatch')
            response=saved['response']
            validate_strategy(response)
        else:
            response={'strategy':original}
            try:
                validate_strategy(response)
            except (ValueError,KeyError,TypeError) as error:
                prompt=('Regenerate only the specified strategy as JSON {"strategy":{...}}. '
                    'Preserve the supplied frozen_identity fields, intended method and fixed experimental budgets. '
                    'A missing, invalid or conflicting name/family is not frozen: correct it to a unique nonempty '
                    'name and a supported family consistent with the intended estimator. '
                    'Correct invalid fields; put all idea fields inside idea. Do not change any other strategy. '
                    'Return name, family, idea with summary, model, ordered feature_engineering, fixed_parameters, '
                    'scaling_rules and stage_parameters; no separate rationale or limitations. '
                    + ('Return only full_train parameters.' if guarded else 'Return coreset and full_train parameters.') +
                    'The following contract describes future implementation, not this response format.\n'+METHOD_CONTRACT+'\n'+
                    json.dumps({'failed_strategy':original,'method_id':i,'frozen_identity':frozen_identity,
                        'task':task_context,'eda':eda,
                        'scoring_budget':cfg.scoring_budget,'evaluation_budget':cfg.evaluation_budget,
                        'full_train_budget':cfg.full_train_budget},ensure_ascii=False))
                response=await regenerate(i,'plan',f'{type(error).__name__}: {error}',prompt,
                    out/'regenerations',regeneration_state_path,model,costs,pricing,validate_strategy)
                write_json(selected,{'source_identity':source_identity,'response':response})
        normalized,moves=normalize_plan({'strategies':[response['strategy']]}, budgets=budgets, mirror_full=guarded)
        methods[i]=normalized['strategies'][0]
        family_change=normalize_strategy_family(methods[i])
        if family_change:
            moves.append(family_change)
        normalizations.extend({'method_id':i,**move} for move in moves)
    write_json(out/'plan.normalized.json',{'coreset_design':coreset_design,'strategies':methods})
    write_json(out/'plan.normalization.json',{'changes':normalizations,'policy':'Move known misplaced idea fields; canonicalize family only from an explicitly named estimator; bounded audited regeneration for invalid plans/strategies'})
    # Persist all ideas before starting any implementation request.
    for i,method in enumerate(methods):
        folder=out/f'method_{i}';folder.mkdir(exist_ok=True)
        record={'version':1,'idea_id':identity(method),'strategy':method}
        write_json(folder/'idea.json',record)
    if checkpoint:checkpoint()
    for i,method in enumerate(methods):
        folder=out/f'method_{i}'
        from .implementation_feedback import component_interfaces
        prompt=METHOD_CONTRACT+'\n'+json.dumps({'strategy':method,'installed_interfaces':component_interfaces(method),'task':task_context,'eda':eda,'scoring_budget':cfg.scoring_budget,'evaluation_budget':cfg.evaluation_budget,'full_train_budget':cfg.full_train_budget,'full_train_enabled':cfg.full_train_enabled},ensure_ascii=False)
        def validate_implementation(response):
            if not isinstance(response,dict) or not isinstance(response.get('code'),str):
                raise ValueError('Implementation requires complete code text')
            for key in ('fe_config','model_config'):
                if not isinstance(response.get(key),dict):
                    raise ValueError('Implementation requires object: '+key)
            from .constructor_compatibility import normalize_constructor_code
            response['code'], compatibility = normalize_constructor_code(response['code'])
            if compatibility:
                response['compatibility_changes'] = compatibility
            validate_code(response['code'])
        response=await generate_one(i,'implementation',prompt,folder/'implementation.json',
            regeneration_state_path,model,costs,pricing,validate_implementation)
        from .constructor_compatibility import normalize_constructor_controls
        code,compatibility=normalize_constructor_controls(response['code'])
        write_json(folder/'constructor_normalization.json',{'changes':compatibility})
        path=folder/'method.py';path.write_text(code)
        if method['idea'].get('plan_format') == 'brief_v1':
            method.update(fe_config=response['fe_config'], model_config=response['model_config'])
        method.update(code_path=str(path),code_sha256=file_hash(path),implementation_configuration={k:response[k] for k in ('fe_config','model_config')})
        write_json(folder/'method.json',method)
        if checkpoint:checkpoint()
    return methods


def run_fit(code,train,predict,task,classes,budget,seed,out,batch_size):
    check_deadline()
    out.mkdir(parents=True,exist_ok=False)
    context={'code':str(code),'train':str(train),'predict':str(predict),'task':task,'classes':classes,
             'budget':budget,'seed':seed,'output':str(out),'batch_size':batch_size}
    write_json(out/'context.json',context)
    env=os.environ.copy()
    for key in list(env):
        if 'API_KEY' in key or key.endswith('_TOKEN'):env.pop(key,None)
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    started=time.perf_counter()
    timeout=bounded_timeout(budget['timeout_seconds'])
    try:
        with (out/'execution.log').open('w') as log:
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('multi_proxy_runtime.py')),str(out/'context.json')],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if result.returncode:
            detail=(out/'failure.json').read_text() if (out/'failure.json').exists() else f'exit={result.returncode}; see execution.log'
            from .execution_budget import InfrastructureInterruption, infrastructure_failure
            failure=json.loads(detail) if (out/'failure.json').exists() else {}
            if result.returncode < 0 or not failure or infrastructure_failure(failure.get('error', '')):
                write_json(out/'failure.json', {**failure, 'error': failure.get('error', detail),
                    'returncode': result.returncode, 'failure_kind': 'infrastructure_interruption'})
                raise InfrastructureInterruption(failure.get('error', detail))
            raise RuntimeError(detail)
    except subprocess.TimeoutExpired as e:
        if timeout < budget['timeout_seconds']:
            write_json(out/'failure.json',{'error':'TaskDeadlineExceeded: task wall budget expired',
                                         'failure_kind':'infrastructure_interruption'})
            raise TaskDeadlineExceeded('Task deadline expired during isolated fit/predict') from e
        detail={'error':f'Fit/predict process exceeded {budget["timeout_seconds"]} seconds',
                'stage':task.get('fit_stage'), 'method_code':str(code), 'budget':budget,
                'execution_directory':str(out)}
        write_json(out/'failure.json',detail)
        raise RuntimeError(f'Method exceeded fit/predict wall budget: stage={task.get("fit_stage")}, '
                           f'limit={budget["timeout_seconds"]}s, details={out / "failure.json"}') from e
    report=json.loads((out/'fit_report.json').read_text());report['process_wall_seconds']=time.perf_counter()-started
    write_json(out/'fit_report.json',report)
    return report


def method_cache_key(cfg, model_identity, hashes, internal_id):
    configuration=asdict(cfg)
    configuration.pop('batch_size',None)
    # K is an output of this workflow's Plan, not an input to method generation.
    # Reuse callers may supply the already-resolved manifest configuration.
    configuration.pop('gmm_components',None)
    configuration.pop('proxy_selection_scheme',None)
    configuration.pop('defer_full_train',None)
    # Post-fit analysis settings do not change generated methods or their training.
    configuration.pop('stage1_selection',None)
    configuration.pop('method_preflight',None)
    # Disabled feedback must keep all previously generated-method identities valid.
    if not configuration.get('parameter_feedback'):
        configuration.pop('parameter_feedback',None)
        configuration.pop('parameter_feedback_mode',None)
    for key in ('scoring_budget','evaluation_budget','full_train_budget'):
        configuration[key].pop('timeout_seconds',None)
    model={key:value for key,value in model_identity.items() if key not in {'timeout','timeout_seconds','max_tokens'}}
    return digest({'version':VERSION,'input':hashes,'internal_id':internal_id,
                   'config':configuration,'model':model,'contract':METHOD_CONTRACT})


def evaluation_metrics(task, y, predictions, probabilities, classes):
    """Keep auxiliary scores and share the primary definition with offline analysis."""
    metrics=score(task['problem_type'],y,predictions)
    reason=None
    try:
        metrics[task['metric']]=primary_score(task,y,predictions,probabilities,classes)
    except UndefinedMetricError as error:
        metrics[task['metric']]=None
        reason=str(error)
    metrics={key:float(value) if value is not None and math.isfinite(value) else None
             for key,value in metrics.items()}
    return metrics,reason


def finite_mean(values):
    return float(np.mean(values)) if values and all(v is not None and math.isfinite(v) for v in values) else None


def rank_methods(records, key, maximize):
    """Undefined scores sort last without a rank; exact ties share a rank."""
    records.sort(key=lambda r:(r[key] is None, (-r[key] if maximize else r[key]) if r[key] is not None else 0, r['method_id']))
    previous=None;rank=None
    for position,item in enumerate(records,1):
        if item[key] is None:
            item['rank']=None
            continue
        if previous is None or item[key]!=previous:rank=position
        item['rank']=rank;previous=item[key]
    return records


def load_fixed(task_path):
    task=json.loads(task_path.read_text())
    if task.get('split',{}).get('strategy')!='frozen':raise ValueError('Pass an existing frozen splits/seed_*/task.json')
    split=json.loads((task_path.parent/'split_manifest.json').read_text())
    signed=hashlib.sha256(json.dumps({k:v for k,v in split.items() if k!='split_id'},sort_keys=True).encode()).hexdigest()
    if split['split_id']!=signed:raise ValueError('Fixed split manifest identity mismatch')
    paths={name:(task_path.parent/task['files'][name]).resolve() for name in ('train',)}
    hashes={}
    for name,path in paths.items():
        hashes[name]=file_hash(path)
        if split.get('sha256',{}).get(path.name)!=hashes[name]:raise ValueError('Frozen file changed: '+name)
    if split.get('sha256',{}).get(task_path.name)!=file_hash(task_path):raise ValueError('Frozen task configuration changed')
    train=read_raw(paths['train'],low_memory=False)
    if task['problem_type']=='regression':
        train[task['target']]=pd.to_numeric(train[task['target']],errors='raise')
    if len(train)!=split['rows']['train']:raise ValueError('Training row count mismatch')
    return task,split,paths,hashes,train


async def run_task(task_path,output,cfg,model,model_identity,pricing=None,reuse_methods=None,*,generation_resume=None,regeneration_state_path=None,reuse_shared=None,interrupted_resume=None,interrupted_manifest=None):
    from .guarded_proxy_feedback import enabled as guarded_enabled, effective_config
    cfg.validate()
    cfg = effective_config(cfg)
    cfg.validate();task_path=Path(task_path).resolve();out=Path(output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();timings={};costs=[]
    manifest={'method':'multi_proxy_data_agent','version':VERSION,'status':'running','task_path':str(task_path),
              'configuration':asdict(cfg),'llm_model':model_identity,'llm_calls':costs,'timings_seconds':timings,
              'mcts':False,'historical_experiences':[],'full_train_fits_completed':0,'gmm_fits_completed':0,'fits_completed':0}
    write_json(out/'manifest.json',manifest)
    def checkpoint():
        write_json(out/'manifest.json',manifest)
        try:
            save_trace(out,manifest)
        except Exception as error:
            manifest.setdefault('trace_errors',[]).append(f'{type(error).__name__}: {error}')
    checkpoint()
    shared = Path(reuse_shared).resolve() if reuse_shared else None
    interrupted = Path(interrupted_resume).resolve() if interrupted_resume else None
    if interrupted:
        manifest['interrupted_resume_source'] = str(interrupted)
    if shared and not reuse_methods:
        raise ValueError('Shared fits require the exact reused methods')
    manifest['reused_predictive_fits']=0
    stage='input'
    try:
        check_deadline()
        if cfg.parameter_feedback and shared:
            raise ValueError('Parameter feedback requires fresh P diagnostics; reuse_shared is not supported')
        with timed(timings,stage):
            task,split,paths,hashes,train=load_fixed(task_path)
            from .prepared_protocol import preparation_config, partition_prepared
            cfg=preparation_config(task,cfg)
            manifest['configuration']=asdict(cfg)
            if cfg.full_train_enabled:
                validation_path=(task_path.parent/task['files']['validation']).resolve()
                if validation_path==paths['train']:
                    raise ValueError('External validation cannot be the training file')
                validation_hash=file_hash(validation_path)
                if split.get('sha256',{}).get(validation_path.name)!=validation_hash:
                    raise ValueError('Frozen external validation file changed or lacks a fingerprint')
                external=read_raw(validation_path,low_memory=False)
                if len(external)!=split['rows']['validation'] or len(external)<1:
                    raise ValueError('External validation row count mismatch or empty')
                if set(external.columns)!=set(train.columns) or external[task['target']].isna().any():
                    raise ValueError('External validation schema/target mismatch')
                indices=split.get('indices',{})
                if set(indices.get('train',[])) & set(indices.get('validation',[])):
                    raise ValueError('Frozen train and validation source indices overlap')
                hashes['validation']=validation_hash
                paths['validation']=validation_path
            manifest.update(task=task,split_id=split['split_id'],input_sha256=hashes,
                            split_metadata={k:v for k,v in split.items() if k not in {'indices','sha256'}})
        stage='partition'
        with timed(timings,stage):
            ppos,vpos,qpos,size_info=partition_prepared(train[task['target']],task,task_path,split,cfg)
            internal_id=digest({'train':hashes['train'],'split_id':split['split_id'],'P':ppos.tolist(),'V_core':vpos.tolist(),'Q':qpos.tolist()})
            manifest.update(internal_split_id=internal_id,sizes=size_info)
            original=split.get('indices',{}).get('train')
            if original is not None and (len(original)!=len(train) or len(set(original))!=len(train)):raise ValueError('Invalid source row mapping')
            def index_record(pos):
                return {'train_positions':pos.tolist(),'source_positions':[original[int(i)] for i in pos] if original is not None else None,'source_reference':split.get('index_reference',{}).get('train'),'internal_split_id':internal_id}
            for name,idx in [('P',ppos),('V_core',vpos),('Q',qpos)]:write_json(out/f'{name}_indices.json',index_record(idx))
            export_rows(paths['train'],[(out/'proxy_train.csv',ppos),(out/'core_validation.csv',vpos)])
            schema=infer_schema(train.iloc[np.sort(np.concatenate([ppos,qpos]))],
                                cfg.categorical_columns + task.get('categorical_columns',[]),task['target'],task['problem_type'])
            train=apply_schema(train,schema)
            if cfg.full_train_enabled:
                external=apply_schema(external,schema)
                if task['problem_type']=='classification' and not set(external[task['target']])<=set(train[task['target']]):
                    raise ValueError('External validation contains unseen target classes')
                if task['problem_type']=='regression' and not np.isfinite(external[task['target']]).all():
                    raise ValueError('External validation target must be finite')
                write_json(out/'full_train_indices.json',index_record(np.arange(len(train))))
                external.drop(columns=[task['target']]).to_csv(out/'external_validation_features.csv',index=False)
                manifest['full_train_evaluation_set']={'name':'external_validation','rows':len(external),
                    'path':str(paths['validation']),'sha256':hashes['validation'],
                    'source_positions':split.get('indices',{}).get('validation'),
                    'training_set':'entire fixed train, including V_core'}
            write_json(out/'csv_schema.json',{'columns':schema,'missing_values':[''],'inference_scope':'P + Q'})
            vframe=train.iloc[vpos].copy()
            manifest['evaluation_set']={'name':'V_core','rows':len(vframe),'indices':'V_core_indices.json',
                'labels_sha256':file_hash(out/'core_validation.csv'),
                'indices_sha256':file_hash(out/'V_core_indices.json')}
            # Q is saved as features only for generated method prediction; labels remain coordinator-only.
            pframe=train.iloc[ppos].copy();qframe=train.iloc[qpos].copy()
            qframe.drop(columns=[task['target']]).to_csv(out/'Q_features.csv',index=False)
            groups,stratum_info=strata(qframe[task['target']],task['problem_type'],cfg.regression_bins,
                                      max_strata=size_info['actual_coreset_rows'])
            allocation=split_quotas(np.bincount(groups),size_info['actual_coreset_rows'],cfg.split_seed+3)
            manifest['quotas']={'strata':stratum_info,'counts':np.bincount(groups).tolist(),'allocation':allocation.tolist(),'applies_to':'all_coresets' if task['problem_type']=='classification' else 'gmm_density_only'}
        stage='eda'
        with timed(timings,stage):
            rep_cfg=CoresetConfig(score_seed=cfg.representation_seed,max_dimensions=cfg.max_dimensions,max_categories=cfg.max_categories,
                                  id_columns=cfg.id_columns,leakage_columns=cfg.leakage_columns,categorical_columns=cfg.categorical_columns)
            exclusions=excluded_fields(task,split,rep_cfg)
            eda=eda_summary(train.iloc[np.sort(np.concatenate([ppos,qpos]))],task,exclusions,cfg)
            write_json(out/'eda.json',eda)
            task_context=method_task_context(task,eda,size_info,pframe,manifest['quotas'],cfg)
            task_context['csv_schema']=schema
            if cfg.parameter_feedback:
                from .multi_proxy_feedback import stage_profile
                task_context['proxy_eda']=stage_profile(pframe,task_context,'P only')
            classes=task_context['classes']
            if classes:
                manifest['evaluation_set'].update(classes_present=sorted(vframe[task['target']].unique().tolist()),
                                                  total_training_classes=len(classes))
            write_json(out/'task_context.json',task_context)
            manifest['training_context']=task_context['training_context']
            checkpoint()
            experiences=retrieve_experience(task_context,eda)
            write_json(out/'experience.json',{'status':'empty_interface','historical_experiences':experiences})
            del train
        stage='shared_representation_fit'
        with timed(timings,stage),threadpool_limits(limits=1):
            if shared:
                import joblib
                rep=joblib.load(shared/'similarity_transform.joblib')
                rep_info=json.loads((shared/'representation.json').read_text())
            else:
                rep=Representation(rep_cfg,exclusions,task_context['categorical_columns'])
                rep_info=call_with_budget(rep.fit,pframe)
            write_json(out/'representation.json',rep_info)
        stage='llm_generation'
        with timed(timings,stage):
            methods_key=method_cache_key(cfg,model_identity,hashes,internal_id)
            manifest['methods_cache_key']=methods_key
            checkpoint()  # Persist generation identity before the first paid request.
            method_dir=out/'methods'
            if reuse_methods:
                previous=(interrupted_manifest if interrupted_manifest is not None and interrupted and Path(reuse_methods).resolve()==interrupted
                          else json.loads((Path(reuse_methods)/'manifest.json').read_text()))
                if previous.get('methods_cache_key')!=methods_key:raise ValueError('Generated-method cache identity mismatch')
                manifest.update(methods_cache_hit=True,reuse_source=str(Path(reuse_methods).resolve()),
                                generation_configuration=previous.get('generation_configuration',previous['configuration']),
                                generation_model=previous.get('generation_model',previous['llm_model']))
                source_dir=Path(reuse_methods)/'methods'
                method_dir.mkdir()
                for source in source_dir.rglob('*'):
                    if source.is_file() and '__pycache__' not in source.parts:
                        destination=method_dir/source.relative_to(source_dir)
                        destination.parent.mkdir(parents=True,exist_ok=True)
                        destination.write_bytes(source.read_bytes())
                methods=json.loads((method_dir/'methods.json').read_text())
                for i,m in enumerate(methods):
                    m['code_path']=str(method_dir/f'method_{i}/method.py')
                    if file_hash(m['code_path'])!=previous['methods'][i]['code_sha256'] or m['code_sha256']!=previous['methods'][i]['code_sha256']:raise ValueError('Generated method cache code changed')
                    idea_record=json.loads((method_dir/f'method_{i}/idea.json').read_text())
                    validate_idea(idea_record['strategy']['idea'])
                    validate_stage_parameters(idea_record['strategy']['idea'])
                    if idea_record['idea_id']!=identity(idea_record['strategy']) or any(
                        idea_record['strategy'][k]!=m[k] or m[k]!=previous['methods'][i][k]
                        for k in (('name','family','idea') if m['idea'].get('plan_format') == 'brief_v1'
                                  else ('name','family','reason','fe_config','model_config','idea'))
                    ):raise ValueError('Generated method cache idea changed')
                    if m['idea'].get('plan_format') == 'brief_v1' and any(
                            m[k] != m['implementation_configuration'][k] or m[k] != previous['methods'][i][k]
                            for k in ('fe_config', 'model_config')):
                        raise ValueError('Generated method cache implementation configuration changed')
                    write_json(method_dir/f'method_{i}/method.json',m)
                manifest['methods_cache_hit']=True
            else:
                manifest.update(generation_configuration=asdict(cfg),generation_model=model_identity)
                method_dir.mkdir()
                if generation_resume:
                    source_run=Path(generation_resume)
                    previous=(interrupted_manifest if interrupted_manifest is not None and interrupted and source_run.resolve()==interrupted
                              else json.loads((source_run/'manifest.json').read_text()))
                    if previous.get('methods_cache_key')!=methods_key:
                        raise ValueError('Generation recovery cache identity mismatch')
                    manifest['generation_recovery_source']=str(source_run)
                    for source in (source_run/'methods').rglob('*'):
                        if source.is_file() and '__pycache__' not in source.parts:
                            destination=method_dir/source.relative_to(source_run/'methods')
                            destination.parent.mkdir(parents=True,exist_ok=True)
                            destination.write_bytes(source.read_bytes())
                    from .generation_recovery import reopen_transport_requests
                    if interrupted:
                        from .method_repair import reconcile_interrupted_requests
                        reconcile_interrupted_requests(method_dir)
                    reopen_transport_requests(method_dir)
                methods=await generate_methods(model,task_context,eda,experiences,cfg,method_dir,costs,pricing or {},checkpoint,regeneration_state_path or out/'method_regeneration.json')
                manifest['methods_cache_hit']=False
            # Keep the generation cache key based on its original inputs; the effective
            # GMM configuration is resolved only after loading the task-level plan.
            design_path=method_dir/'coreset_design.json'
            if not design_path.exists():
                raise ValueError('Generated-method cache lacks coreset_design; regenerate with EDA-based GMM planning')
            cfg,coreset_design=apply_gmm_design(cfg,json.loads(design_path.read_text()),task_context)
            if cfg.proxy_selection_scheme in {'marginal_coverage_random_comparison', 'marginal_coverage_dual_random_comparison', 'marginal_coverage_only'}:
                manifest['generated_coreset_design']=coreset_design
                cfg,coreset_design=apply_gmm_design(cfg,{
                    'gmm_components':5,
                    'rationale':'Fixed experiment protocol K=5, matching the previous coverage parameter grid; generated proposal retained separately.'},task_context)
            if reuse_methods and previous.get('coreset_design')!=coreset_design:
                raise ValueError('Reused task-level GMM design changed')
            manifest['coreset_design']=coreset_design
            manifest['configuration']=asdict(cfg)
            write_json(out/'coreset_design.json',coreset_design)
            write_json(method_dir/'methods.json',methods)
            manifest['methods']=methods
            manifest['method_set_id']=digest([m['code_sha256'] for m in methods])
            checkpoint()
        if cfg.method_preflight:
            stage='method_preflight'
            from .method_repair import preflight_methods
            with timed(timings,stage):
                failures=preflight_methods(methods,task_context,cfg,out,update_stage_parameters=True)
            # Completion is part of the effective declaration, not an implicit
            # runtime override. Persist matching records before success/failure
            # so derived repair caches and later reuse see the same identity.
            for i,m in enumerate(methods):
                record_path=method_dir/f'method_{i}/idea.json'
                record=json.loads(record_path.read_text())
                if record['strategy']['idea']!=m['idea']:
                    record['strategy']['idea']=copy.deepcopy(m['idea'])
                    record['idea_id']=identity(record['strategy'])
                    record['version']=record.get('version',1)+1
                    record['stage_parameter_completion']=f'preflight/method_{i}/stage_parameter_completion.json'
                    write_json(record_path,record)
                write_json(method_dir/f'method_{i}/method.json',m)
            write_json(method_dir/'methods.json',methods)
            if failures:
                manifest['active_execution']=f"preflight/method_{failures[0]['method_id']}"
                raise RuntimeError('Method preflight failed: '+json.dumps(failures,ensure_ascii=False))
        if shared:
            from .paired_execution import validate_shared_source
            validate_shared_source(shared, out, manifest)
            manifest['shared_fit_source']=str(shared)
        feedback_plan=None;guarded_items=[]
        manifest['proxy_refits_completed']=0
        stage='proxy_fits'
        with timed(timings,stage):
            importance=[];proxy_reports=[]
            for i,m in enumerate(methods):
                folder=out/'proxies'/f'method_{i}'
                manifest['active_execution']=str(folder.relative_to(out))
                if guarded_enabled(cfg):
                    from .guarded_proxy_feedback import fit_guarded_proxy
                    report,item,new_fits,reused = await fit_guarded_proxy(
                        out,m,i,task_context,pframe,qframe,cfg,model,costs,pricing or {},_llm_json_once,
                        interrupted=interrupted)
                    guarded_items.append(item)
                    manifest['fits_completed']+=new_fits
                    manifest['reused_predictive_fits']+=reused
                    manifest['proxy_refits_completed']+=int(item['proxy_refit'])
                elif shared:
                    from .paired_execution import copy_fit
                    report=copy_fit(shared/'proxies'/f'method_{i}',folder)
                    manifest['reused_predictive_fits']+=1
                else:
                    proxy_task={**task_context,'expected_family':m['family'],'fit_stage':'proxy',
                        **({'collect_training_feedback':True} if cfg.parameter_feedback else {})}
                    report=None
                    if interrupted:
                        from .method_repair import reuse_interrupted_proxy
                        report=reuse_interrupted_proxy(interrupted/'proxies'/f'method_{i}',folder,
                            m['code_path'],out/'proxy_train.csv',out/'Q_features.csv',proxy_task,classes,
                            cfg.scoring_budget,cfg.model_seed,cfg.batch_size)
                    if report is None:
                        report=run_fit(m['code_path'],out/'proxy_train.csv',out/'Q_features.csv',proxy_task,
                            classes,cfg.scoring_budget,cfg.model_seed,folder,cfg.batch_size)
                        manifest['fits_completed']+=1
                    else:
                        manifest['reused_predictive_fits']+=1
                with np.load(folder/'predictions.npz',allow_pickle=False) as result:
                    if len(result['predictions'])!=len(qpos):raise ValueError('Proxy prediction row count mismatch')
                    error,values,cutoff=error_importance(qframe[task['target']],result['predictions'],result['probabilities'],classes,task['problem_type'],cfg.error_clip_quantile)
                np.savez_compressed(folder/'scores.npz',raw_error=error,I=values,train_positions=qpos,source_positions=np.asarray(original)[qpos] if original is not None else np.array([],dtype=int))
                report.update(error_clip_quantile=cfg.error_clip_quantile,error_clip_value=cutoff,normalization='global Q average percentile rank after clipping',method_id=i)
                write_json(folder/'score_report.json',report);importance.append(values);proxy_reports.append(report)
                manifest.pop('active_execution',None)
                checkpoint()
            manifest['proxy_reports']=proxy_reports
        if guarded_enabled(cfg):
            from .guarded_proxy_feedback import freeze_guarded_plan
            from .multi_proxy_feedback import frozen_fit_context
            feedback_plan=freeze_guarded_plan(out,guarded_items)
            manifest['parameter_feedback']={'status':'frozen','mode':'guarded_once','schema_version':1,
                'plan_path':'parameter_feedback/frozen_plan.json',
                'plan_sha256':file_hash(out/'parameter_feedback/frozen_plan.json'),
                'shared_across_all_coresets':True,'proxy_refit':feedback_plan['proxy_refit'],
                'max_adjustments_per_method':1,'full_parameters_immutable':True,'validation_used':False}
            checkpoint()
        stage='shared_representation_and_graph'
        with timed(timings,stage),threadpool_limits(limits=1):
            if shared:
                import joblib
                import shutil
                for filename in ('similarity_graph.npz','similarity_transform.joblib','representation.json','graph.json'):
                    shutil.copyfile(shared/filename,out/filename)
                graph=sparse.load_npz(out/'similarity_graph.npz')
                graph_info=json.loads((out/'graph.json').read_text())
                rep_info=json.loads((out/'representation.json').read_text())
                manifest.update(representation=rep_info,graph=graph_info)
                from .paired_execution import copy_directory
                copy_directory(shared/'density_gmm',out/'density_gmm')
                with np.load(out/'density_gmm/scores.npz',allow_pickle=False) as saved:
                    if not np.array_equal(saved['train_positions'],qpos):
                        raise ValueError('Shared GMM Q positions mismatch')
                    importance.append(saved['I'])
                manifest['gmm']=json.loads((out/'density_gmm/score_report.json').read_text())
                validate_reused_gmm(coreset_design,manifest['gmm'],joblib.load(out/'density_gmm/fitted.joblib').n_components)
                manifest['gmm_fits_reused']=1
            else:
                z=np.empty((len(qframe),rep_info['dimensions']))
                for begin in range(0,len(qframe),cfg.batch_size):z[begin:begin+cfg.batch_size]=call_with_budget(rep.transform,qframe.iloc[begin:begin+cfg.batch_size])
                graph,graph_info=call_with_budget(mutual_graph,z,cfg)
                sparse.save_npz(out/'similarity_graph.npz',graph)
                import joblib
                joblib.dump(rep,out/'similarity_transform.joblib')
                write_json(out/'representation.json',rep_info);write_json(out/'graph.json',graph_info)
                manifest.update(representation=rep_info,graph=graph_info)
                stage='gmm_density_scoring'
                with timed(timings,stage),threadpool_limits(limits=1):
                    folder=out/'density_gmm'
                    folder.mkdir()
                    try:
                        gmm,density,values,gmm_report=call_with_budget(gmm_density_importance,rep.transform(pframe),z,groups,cfg)
                        validate_reused_gmm(coreset_design,gmm_report,gmm.n_components)
                        joblib.dump(gmm,folder/'fitted.joblib')
                        np.savez_compressed(folder/'scores.npz',log_density=density,I=values,train_positions=qpos)
                        gmm_report.update(fit_indices='P_indices.json',representation='similarity_transform.joblib')
                        write_json(folder/'score_report.json',gmm_report)
                        manifest['gmm_fits_completed']=1
                        manifest['gmm']=gmm_report
                        importance.append(values)
                    except TaskDeadlineExceeded:
                        raise
                    except Exception as error:
                        write_json(folder/'failure.json',{'error':f'{type(error).__name__}: {error}'})
                        raise
                    checkpoint()
        stage='coreset_selection'
        with timed(timings,stage):
            if cfg.proxy_selection_scheme in {'edge_mechanism_comparison', 'marginal_coverage_random_comparison', 'marginal_coverage_dual_random_comparison', 'marginal_coverage_only'}:
                if shared:
                    raise ValueError('Comparison prepares one fresh shared source per task')
                if cfg.proxy_selection_scheme in {'marginal_coverage_random_comparison', 'marginal_coverage_dual_random_comparison', 'marginal_coverage_only'}:
                    from .new30_coverage_comparison import build_selections
                else:
                    from .edge_mechanism_comparison import build_selections
                guide_args={}
                if cfg.validation_guided_mix:
                    from .guided_coreset import initial_errors
                    guide_args['validation_errors']=initial_errors(out,task_context,vframe,methods,cfg)
                    guide_args['context']=task_context
                with threadpool_limits(limits=1):
                    chosen_sets=call_with_budget(build_selections,out,qframe,qpos,importance,graph,rep,task,cfg,size_info['actual_coreset_rows'],**guide_args)
                selections=[qpos[ix] for ix in chosen_sets]
                for i,positions in enumerate(selections):
                    write_json(out/f'coreset_{i}_indices.json',index_record(positions))
                manifest['selection']={'scheme':cfg.proxy_selection_scheme,'design':
                    'coverage_comparison_design.json' if cfg.proxy_selection_scheme in {'marginal_coverage_random_comparison', 'marginal_coverage_dual_random_comparison', 'marginal_coverage_only'} else 'edge_comparison_design.json'}
            else:
                manifest['selection']={'objective':'mean_importance - lambda * xTKx / (p * k_eff)',
                                       'k_eff':min(cfg.neighbors,size_info['actual_coreset_rows']-1),
                                       'lambda':cfg.similarity_lambda,'solver':'sparse incremental greedy'}
                manifest['selection']['proxy_scheme']=cfg.proxy_selection_scheme
                gmm_order,gmm_opt=greedy_select(importance[-1],graph,groups,allocation,
                    cfg.similarity_lambda,normalization_degree=cfg.neighbors)
                write_json(out/'gmm_selection_order.json',{'q_indices':gmm_order.tolist(),
                           'train_indices':qpos[gmm_order].tolist()})
                selections=[]
                for i,values in enumerate(importance):
                    if i<cfg.n_strategies:
                        chosen,opt=select_proxy_variant(values,importance[-1],graph,groups,allocation,
                            cfg,task['problem_type']=='classification',gmm_order,
                            hardness_matrix=np.asarray(importance[:cfg.n_strategies]),method_index=i)
                        effective_values = (importance[-1]*(1+values)/2
                            if cfg.proxy_selection_scheme=='density_hardness' else values)
                        if cfg.proxy_selection_scheme=='density_centered_eta025':
                            effective_values,_=centered_density_importance(importance[-1],values,.25)
                        if cfg.proxy_selection_scheme in {'density_relative','density_relative_local'}:
                            effective_values,_=relative_density_importance(importance[-1],
                                np.asarray(importance[:cfg.n_strategies]),graph,i,
                                local=cfg.proxy_selection_scheme=='density_relative_local')
                        np.save(out/f'coreset_{i}_effective_importance.npy',effective_values)
                    else:
                        chosen,opt=gmm_order.copy(),dict(gmm_opt)
                    write_json(out/f'coreset_{i}_selection_order.json',
                        {'q_indices':chosen.tolist(),'train_indices':qpos[chosen].tolist()})
                    chosen=np.sort(chosen);positions=qpos[chosen]
                    actual_counts=np.bincount(groups[chosen],minlength=len(allocation))
                    if len(positions)!=size_info['actual_coreset_rows'] or len(np.unique(positions))!=len(positions):
                        raise ValueError('Coreset size/uniqueness invariant failed')
                    if (i==cfg.n_strategies or task['problem_type']=='classification') and not np.array_equal(actual_counts,allocation):raise ValueError('Coreset proportional quota invariant failed')
                    if i<cfg.n_strategies and task['problem_type']=='classification' and np.any(actual_counts<1):raise ValueError('Proxy coreset class coverage invariant failed')
                    write_json(out/f'coreset_{i}_indices.json',index_record(positions))
                    selections.append(positions);write_json(out/f'coreset_{i}_selection.json',{'proxy_method':i if i<cfg.n_strategies else None,'source':'proxy_error' if i<cfg.n_strategies else 'gmm_density','objective':opt,'q_candidate_count':len(qpos),'stratum_counts':actual_counts.tolist(),
                        'selection_policy':'proportional_stratified_quotas' if task['problem_type']=='classification' or i==cfg.n_strategies else 'global_unconstrained',
                        'required_quotas':allocation.tolist() if task['problem_type']=='classification' or i==cfg.n_strategies else None})
            export_rows(paths['train'],[(out/f'coreset_{i}.csv',pos) for i,pos in enumerate(selections)])
            manifest['overlap']=[{'pair':[i,j],'fraction':len(np.intersect1d(a,b))/len(a)} for i,a in enumerate(selections) for j,b in enumerate(selections) if i<j]
        if cfg.parameter_feedback and not guarded_enabled(cfg):
            stage='parameter_feedback'
            from .multi_proxy_feedback import refine_parameters, frozen_fit_context
            with timed(timings,stage):
                feedback_request=llm_json
                if interrupted:
                    from .method_repair import interrupted_feedback_request
                    feedback_request=interrupted_feedback_request(llm_json,interrupted/'parameter_feedback')
                feedback_plan=await refine_parameters(out,methods,task_context,eda,pframe,qframe,
                    selections,proxy_reports,cfg,model,costs,pricing or {},feedback_request)
            manifest['parameter_feedback']={'status':'frozen','schema_version':feedback_plan['schema_version'],
                'plan_path':'parameter_feedback/frozen_plan.json',
                'plan_sha256':file_hash(out/'parameter_feedback/frozen_plan.json'),
                'shared_across_all_coresets':True,'proxy_refit':False,'validation_used':False}
            write_json(out/'manifest.json',manifest)
            checkpoint()
        if cfg.mcts_bootstrap_only:
            expected_proxies=cfg.n_strategies+manifest['proxy_refits_completed']
            if manifest['fits_completed'] + manifest['reused_predictive_fits'] != expected_proxies:
                raise ValueError('MCTS bootstrap must fit P proxies plus authorized one-shot refits only')
            manifest.update(status='success',full_train_pending=True,formal_evaluation_pending=True,
                            expected_predictive_fits=expected_proxies-manifest['reused_predictive_fits'],bootstrap_scope='P/Q feedback and one coverage coreset; no random baseline or validation fits')
            return manifest
        stage='formal_evaluation'
        with timed(timings,stage):
            validation=vframe
            validation.drop(columns=[task['target']]).to_csv(out/'core_validation_features.csv',index=False)
            class_map={str(c):c for c in classes}
            evaluation=[]; pipeline_specs={}; model_classes={}
            for i,m in enumerate(methods):
                for j in range(cfg.n_coresets):
                    folder=out/'evaluations'/f'method_{i}_coreset_{j}'
                    manifest['active_execution']=str(folder.relative_to(out))
                    if shared and j==cfg.n_strategies:
                        if file_hash(shared/f'coreset_{j}_indices.json')!=file_hash(out/f'coreset_{j}_indices.json'):
                            raise ValueError('Shared GMM coreset differs')
                        report=copy_fit(shared/'evaluations'/f'method_{i}_coreset_{j}',folder)
                        if pipeline_specs.get(i)!=report['pipeline_spec']:
                            raise ValueError('Shared GMM pipeline specification differs')
                        manifest['reused_predictive_fits']+=1
                    else:
                        report=run_fit(m['code_path'],out/f'coreset_{j}.csv',out/'core_validation_features.csv',{**task_context,'expected_family':m['family'],'fit_stage':'evaluation','stage_parameters':m['idea']['stage_parameters']['coreset'],'expected_pipeline_spec':pipeline_specs.get(i),
                            **(frozen_fit_context(feedback_plan,i,'coreset') if feedback_plan else {})},classes,cfg.evaluation_budget,cfg.model_seed,folder,cfg.batch_size)
                        manifest['fits_completed']+=1
                    pipeline_specs.setdefault(i,report['pipeline_spec'])
                    model_classes.setdefault(i,report['actual_budget']['model_class'])
                    with np.load(folder/'predictions.npz',allow_pickle=False) as result:
                        predictions=result['predictions']
                        probabilities=result['probabilities']
                        if task['problem_type']=='classification':predictions=np.asarray([class_map[v] for v in predictions])
                        if len(predictions)!=len(validation):raise ValueError('Validation predictions not aligned')
                    metrics,metric_reason=evaluation_metrics(task,validation[task['target']],predictions,probabilities,classes)
                    report.update(method_id=i,coreset_id=j,method_name=m['name'],metric=task['metric'],metrics=metrics,
                                  primary_metric_status='undefined' if metric_reason else 'defined',primary_metric_reason=metric_reason,
                                  evaluation_set='V_core',evaluation_indices='V_core_indices.json',
                                  evaluation_labels_sha256=manifest['evaluation_set']['labels_sha256'],
                                  evaluation_indices_sha256=manifest['evaluation_set']['indices_sha256'])
                    write_json(folder/'evaluation.json',report);evaluation.append(report)
                    manifest.pop('active_execution',None)
                    checkpoint()
            metric=task['metric'];maximize=metric_direction(task)=='maximize'
            ranking=[{'method_id':i,'method_name':m['name'],'metric':metric,'mean':finite_mean([r['metrics'][metric] for r in evaluation if r['method_id']==i]),
                      'scores':[r['metrics'][metric] for r in evaluation if r['method_id']==i]} for i,m in enumerate(methods)]
            for item in ([] if cfg.proxy_selection_scheme in {'edge_mechanism_comparison', 'marginal_coverage_random_comparison', 'marginal_coverage_dual_random_comparison', 'marginal_coverage_only'} else ranking):
                item['proxy_coresets_mean']=finite_mean(item['scores'][:cfg.n_strategies])
                item['gmm_coreset_score']=item['scores'][cfg.n_strategies]
            rank_methods(ranking,'mean',maximize)
            write_json(out/'ranking.json',{'coreset_sources':[f'proxy_{i}' for i in range(cfg.n_strategies)]+['gmm_density'],'aggregation':'equal mean of all coreset scores','evaluation_set':manifest['evaluation_set'],'scope':'V_core initial performance at the specified training size and budget over all equally weighted coresets; not fully trained potential or statistical significance',
                                         'coreset_rows':size_info['actual_coreset_rows'],'evaluation_budget':cfg.evaluation_budget,
                                         'direction':'maximize' if maximize else 'minimize','methods':ranking})
            if cfg.proxy_selection_scheme in {'edge_mechanism_comparison', 'marginal_coverage_random_comparison', 'marginal_coverage_dual_random_comparison', 'marginal_coverage_only'}:
                if cfg.proxy_selection_scheme in {'marginal_coverage_random_comparison', 'marginal_coverage_dual_random_comparison', 'marginal_coverage_only'}:
                    from .new30_coverage_comparison import comparison_arms
                    ARMS = comparison_arms(cfg.proxy_selection_scheme)
                else:
                    from .edge_mechanism_comparison import ARMS
                write_json(out/'ranking.json',{'coreset_sources':[f'{mode}_seed{seed}' for mode,seed in ARMS],
                    'aggregation':'descriptive mean across arms; NOT the mechanism comparison endpoint',
                    'evaluation_set':manifest['evaluation_set'],'direction':'maximize' if maximize else 'minimize','methods':ranking})
            write_json(out/'evaluations.json',evaluation)
        write_json(out/'fairness.json',{'scope':'coreset configuration consistency','status':'passed','same_pipeline_across_all_coresets':True,
            'policy':('P/Q feedback revision; one frozen parameter mapping shared by all coresets; planned resource values within equal caps; single thread; fixed seed; no validation-guided tuning'
                      if feedback_plan else 'predeclared stage parameters; equal stage caps; exact configured tree/MLP iteration limits; single thread; fixed seed; no validation-guided tuning'),
            'coreset_pipeline_specs':pipeline_specs,'evaluation_budget':cfg.evaluation_budget,
            'full_train_budget':cfg.full_train_budget,'limitations':'equal caps do not imply equal convergence, FLOPs or optimal hyperparameters'})
        if cfg.stage1_selection['enabled']:
            stage='stage1_selection'
            with timed(timings,stage):
                selection=analyze_run(out,cfg.stage1_selection,manifest=manifest)
                manifest['stage1_selection']={'summary_path':'stage1_selection/summary.json',
                    'selection_mode':selection['selection_mode'],'selection_status':selection['selection_status'],
                    'survivor_method_ids':selection['survivor_method_ids'],
                    'inactive_method_ids':selection['inactive_method_ids']}
                if selection.get('mcts_admission'):
                    manifest['stage1_selection'].update(
                        mcts_admission_path='stage1_selection/mcts_admission.json',
                        mcts_active_method_ids=selection['mcts_admission']['active_method_ids'],
                        mcts_inactive_method_ids=selection['mcts_admission']['inactive_method_ids'],
                        mcts_backfilled_method_ids=selection['mcts_admission']['backfilled_method_ids'])
            # Persist the low-fidelity decision before any full-train evaluation.
            write_json(out/'manifest.json',manifest)
            checkpoint()
        if cfg.full_train_enabled and not cfg.defer_full_train:
            stage='full_train_evaluation'
            with timed(timings,stage):
                full_reports=[]
                for i,m in enumerate(methods):
                    folder=out/'full_train'/f'method_{i}'
                    manifest['active_execution']=str(folder.relative_to(out))
                    report=run_fit(m['code_path'],paths['train'],out/'external_validation_features.csv',
                        {**task_context,'expected_family':m['family'],'expected_model_class':model_classes[i],
                         'reference_pipeline_path':str(out/'evaluations'/f'method_{i}_coreset_0'/'unfitted.joblib'),
                         'fit_stage':'full_train','stage_parameters':m['idea']['stage_parameters']['full_train'],
                         **(frozen_fit_context(feedback_plan,i,'full_train') if feedback_plan else {})},
                        classes,cfg.full_train_budget,cfg.model_seed,folder,cfg.batch_size)
                    manifest['fits_completed']+=1
                    manifest['full_train_fits_completed']+=1
                    with np.load(folder/'predictions.npz',allow_pickle=False) as result:
                        predictions=result['predictions']
                        probabilities=result['probabilities']
                        if task['problem_type']=='classification':
                            predictions=np.asarray([class_map[v] for v in predictions])
                        if len(predictions)!=len(external):raise ValueError('External validation predictions not aligned')
                    metrics,metric_reason=evaluation_metrics(task,external[task['target']],predictions,probabilities,classes)
                    report.update(method_id=i,method_name=m['name'],metric=metric,metrics=metrics,
                                  primary_metric_status='undefined' if metric_reason else 'defined',primary_metric_reason=metric_reason,
                                  evaluation_set='external_validation',training_indices='full_train_indices.json')
                    write_json(folder/'evaluation.json',report);full_reports.append(report)
                    write_json(out/'full_train_evaluations.json',full_reports)
                    manifest.pop('active_execution',None);checkpoint()
                full_ranking=[{'method_id':r['method_id'],'method_name':r['method_name'],
                    'metric':metric,'score':r['metrics'][metric]} for r in full_reports]
                rank_methods(full_ranking,'score',maximize)
                write_json(out/'full_train_ranking.json',{'evaluation_set':manifest['full_train_evaluation_set'],
                    'direction':'maximize' if maximize else 'minimize','methods':full_ranking,
                    'scope':'full-train performance with predeclared adapted parameters; different validation set from coreset stage; no test access'})
            if cfg.stage1_selection['enabled']:
                stage='stage1_full_train_comparison'
                with timed(timings,stage):
                    compare_full_stage(out,selection_summary=selection)
                    manifest['stage1_selection']['full_train_comparison_path']='stage1_selection/full_train_comparison.json'
                checkpoint()
        manifest['status']='success'
        manifest['full_train_pending']=bool(cfg.full_train_enabled and cfg.defer_full_train)
        expected_fits=cfg.n_strategies * (1 + cfg.n_coresets + int(cfg.full_train_enabled and not cfg.defer_full_train))+manifest['proxy_refits_completed']-manifest['reused_predictive_fits']
        manifest['expected_predictive_fits']=expected_fits
        if manifest['fits_completed']!=expected_fits:
            raise ValueError(f"Unexpected completed predictive fit count: {manifest['fits_completed']} != {expected_fits}")
    except TaskDeadlineExceeded as error:
        manifest.update(status='interrupted',failure_stage=stage,error=f'{type(error).__name__}: {error}',
                        interruption_kind='task_deadline',method_failure=False)
        raise
    except BaseException as error:
        from .execution_budget import infrastructure_failure
        from .generation_recovery import transient
        infrastructure = infrastructure_failure(error) or transient(error)
        manifest.update(status='interrupted' if infrastructure else 'failed',failure_stage=stage,
                        error=f'{type(error).__name__}: {error}')
        if infrastructure:
            manifest.update(interruption_kind='infrastructure', method_failure=False)
        raise
    finally:
        timings['total_wall']=time.perf_counter()-started
        manifest['llm_usage']={'calls':len(costs),'prompt_tokens':sum(c.get('usage',{}).get('prompt_tokens',0) for c in costs),'completion_tokens':sum(c.get('usage',{}).get('completion_tokens',0) for c in costs),
                               'estimated_cost':sum(c['estimated_cost'] for c in costs) if costs and all(c.get('estimated_cost') is not None for c in costs) else None}
        write_json(out/'manifest.json',manifest)
        if is_direct_deepseek(model_identity.get('base_url','')):
            write_json(out/'deepseek_usage.json',{'summary':summarize(costs),'calls':costs})
        checkpoint()
        write_json(out/'manifest.json',manifest)
    return manifest
