"""Persistent, bounded three-proposal MCTS with fixed or versioned coresets.

All decisions use V_core or external validation, never the final test. Full
training includes V_core; its score is always measured on external validation.
"""
from __future__ import annotations
import asyncio
from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import time
import uuid

import numpy as np
from .mcs_range import mcs_range
from .mcts_changes import coreset_blind_parameter_changes, full_comparison_plan, same_low_on_scored_branches
from .mcts_policy import MCTSConfig, SearchTree, shortlist_candidates, should_rollout_second, paired_bootstrap_probability
from .mcts_memory import ExperienceMemory
from .multi_proxy_io import atomic_json
from .stage1_metrics import primary_score, metric_direction, requires_stratified_bootstrap
from .stage1_selection import bootstrap_indices
from .execution_budget import InfrastructureInterruption, TaskDeadlineExceeded, infrastructure_failure


class SearchBudgetExceeded(RuntimeError):
    budget_exhausted = True


def json_safe(v):
    if isinstance(v, np.ndarray):return json_safe(v.tolist())
    if isinstance(v, np.generic):return json_safe(v.item())
    if isinstance(v, dict):return {str(k):json_safe(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)):return [json_safe(x) for x in v]
    if isinstance(v, Path):return str(v)
    if isinstance(v,float) and not math.isfinite(v):return None
    return v


def compare_candidates(backend,candidates,cfg,seed):
    """Metric-aware paired MCS; undefined/dependent bootstrap falls back explicitly."""
    versions={c.get('coreset_binding',{}).get('version') for c in candidates}
    if len(versions)>1:
        raise ValueError('MCS requires candidates evaluated on the same coreset version')
    ids=[c['id'] for c in candidates]
    base={'method_ids':ids,'survivor_method_ids':ids,'selection_status':'unavailable',
          'scope':'Adaptive-search heuristic; not an independent final significance guarantee.'}
    if not cfg.mcs_enabled:
        return {**base,'selection_status':'disabled','selection_reason':'MCS disabled; rank by coreset primary metric.'}
    if len(ids)==1:return {**base,'selection_status':'single_candidate'}
    if not getattr(backend,'supports_iid_mcs',True):
        return {**base,'selection_status':'unsupported_dependency','selection_reason':'Declared dependent validation rows; rank fallback only.'}
    task=backend.task; y=np.asarray(backend.y_core); maximize=metric_direction(task)=='maximize'
    theta=np.array([c['low_score'] for c in candidates])*(-1 if maximize else 1)
    scheme='stratified_by_true_class' if requires_stratified_bootstrap(task) else 'iid_rows'
    boot=[]
    try:
        pred=[backend.load_prediction(c['low_report']) for c in candidates]
        for ix in bootstrap_indices(y,{'bootstrap_seed':seed,'bootstrap_replicates':cfg.bootstrap_replicates},scheme):
            row=[]
            for p in pred:
                probabilities=p.get('probabilities'); probabilities=None if probabilities is None else np.asarray(probabilities)[ix]
                score=primary_score(task,y[ix],np.asarray(p['predictions'])[ix],probabilities,p.get('classes'))
                row.append(-score if maximize else score)
            boot.append(row)
        result=mcs_range(theta,np.asarray(boot),ids,cfg.mcs_alpha)
        # Huge replicate matrices remain unnecessary for orchestration; retain all decisions.
        result={k:v for k,v in result.items() if k!='centered_bootstrap_statistics'}
        return json_safe({**base,**result,'bootstrap_seed':seed,'bootstrap_replicates':cfg.bootstrap_replicates,'bootstrap_scheme':scheme})
    except (ValueError,FloatingPointError) as exc:
        return {**base,'selection_status':'undefined_bootstrap','selection_reason':str(exc)}


class MCTSSearch:
    def __init__(self,backend,output,config=None,*,resume=False,memory_options=None,
                 region_controller=None,initial_feedback=None):
        self.backend=backend; self.out=Path(output).resolve()
        self.region_controller=region_controller;self.initial_feedback=initial_feedback
        if (self.out/'final_test'/'selection_manifest.json').exists():
            raise ValueError('Final-test winner is already frozen; this search cannot be resumed or changed')
        self.out.mkdir(parents=True,exist_ok=True)
        self.cfg=MCTSConfig.from_dict(config);self.state_path=self.out/'search_state.json'
        if region_controller is not None and not self.cfg.enabled:
            raise ValueError('Per-round regional sampling requires enabled MCTS')
        self.started=time.monotonic();self.elapsed_base=0.;self._clock_running=True
        self.memory_options=dict(memory_options or {})
        self._checkpoint_digest=None;self._summary_digest=None;self._best_digest=None
        identity=json_safe(getattr(backend,'identity',getattr(backend,'metadata',{'task':backend.task,'source':str(getattr(backend,'source_run',''))})))
        if region_controller is not None:
            identity={**identity,'round_bandit':json_safe(region_controller.identity)}
        # Explicit default-on switches are equivalent to pre-module runs. Off
        # switches stay in the identity and cannot be changed during resume.
        frozen_config=self.cfg.to_dict();frozen_memory=deepcopy(self.memory_options)
        for key in ('enabled','mcs_enabled'):
            if frozen_config[key] is True:frozen_config.pop(key)
        if frozen_config['infrastructure_retries'] == 1:
            frozen_config.pop('infrastructure_retries')  # compatible with pre-retry checkpoints
        if frozen_memory.get('enabled') is True:frozen_memory.pop('enabled')
        fingerprint=hashlib.sha256(json.dumps({'backend':identity,'config':frozen_config,'memory':frozen_memory},sort_keys=True).encode()).hexdigest()
        if self.state_path.exists():
            if not resume:raise ValueError('Search already exists; use --resume with unchanged inputs/configuration')
            self.state=json.loads(self.state_path.read_text())
            if self.state['fingerprint']!=fingerprint:raise ValueError('Source/configuration changed after search was frozen')
            self.tree=SearchTree.from_dict(self.state['tree']);self.elapsed_base=self.state.get('elapsed_seconds',0.)
            # Older checkpoints counted a failed-only batch as a completed
            # expansion. Preserve its audit without spending the success quota.
            failed_batches=[batch for batch in self.state['expansions']
                            if not batch.get('winner_id') and not any(
                                self.state['candidates'].get(i,{}).get('full_report')
                                for i in batch.get('candidate_ids',[]))]
            if failed_batches:
                self.state['expansions']=[batch for batch in self.state['expansions'] if batch not in failed_batches]
                self.state.setdefault('aborted_batches',[]).extend(failed_batches)
                self.state['events'].append({'kind':'failed_expansions_reclassified','count':len(failed_batches)})
                if self.state['status']=='completed' and self.state.get('stop_reason')=='expansion_budget':
                    self.state['status']='running';self.state.pop('stop_reason',None)
            if self.state.get('inflight'):
                # Fits can recover committed artifacts or retry their unchanged
                # frozen contract. LLM requests remain non-replayed operations.
                pending=self.state.pop('inflight');self.state.setdefault('interrupted_operations',[]).append(pending)
                if pending['kind']=='fit' or pending.get('candidate_id'):
                    c=self.state['candidates'].get(pending['candidate_id'])
                    if c is None:
                        active_batch=self.state.get('pending') or {}
                        c=next((probe for probe in self._regional_probes()
                                if probe.get('id')==pending['candidate_id']),None)
                        if c is None:c=active_batch.get('region_anchor')
                        if not c or c['id']!=pending['candidate_id']:
                            raise ValueError('Interrupted fit has no frozen candidate or parent anchor')
                    if not c.get(pending['stage']+'_report'):
                        c[pending['stage']+'_interruption']='Interrupted operation: charged attempt; recover artifacts or retry unchanged fit'
                        c['status']='interrupted'
                elif self.state.get('pending') and pending.get('operation')=='region_advisor':
                    self.state['pending']['region_advisor_interrupted']=True
                elif self.state.get('pending'):
                    self.state['pending']['generation_error']='Interrupted generation; partial artifacts retained without silently resending'
                self.state.pop('active_evaluation',None)
            # Older versions made infrastructure failures permanent. Reopen only
            # those typed failures (or the old explicit interruption marker).
            for candidate in self.state['candidates'].values():
                for stage in ('low','full'):
                    error=candidate.get(stage+'_error')
                    if error and (infrastructure_failure(error) or error.startswith('Interrupted operation:')):
                        candidate[stage+'_interruption']=candidate.pop(stage+'_error')
                        candidate['status']='interrupted'
                        candidate['failure_kind']='infrastructure_interruption'
                # A graceful exception checkpoint can include a backend result
                # saved just before the coordinator registered its tree reward.
                # Reconcile that commit gap before scheduling any more work.
                for stage in ('low','full'):
                    report=candidate.get(stage+'_report')
                    missing=(self.tree.candidates[candidate['id']].get('low_score') is None if stage=='low'
                             else candidate['id']+':full' not in self.tree.full_evaluations)
                    if not report or not missing:continue
                    recover=getattr(backend,'recover_evaluation',None)
                    if recover is not None:
                        checked=recover(candidate,stage)
                        if checked is None:raise ValueError('Committed evaluation artifacts missing during recovery')
                        report=checked
                    score=report.get('score')
                    if isinstance(score,bool) or not isinstance(score,Real) or not math.isfinite(score):
                        raise ValueError('Recovered score must be finite')
                    candidate[stage+'_score']=float(score)
                    if stage=='low':self.tree.candidates[candidate['id']]['low_score']=float(score)
                    else:
                        if self.tree.register_rollout(candidate['id'],float(score),attempt_id=candidate['id']+':full'):
                            self.state['full_fit_successes']+=1
                    self.state['events'].append({'kind':'evaluation_commit_reconciled',
                                                'candidate_id':candidate['id'],'stage':stage})
            # Successful-rollout quota follows unique validated tree evidence,
            # including an interruption between tree registration and counting.
            if self.state['full_fit_successes'] != len(self.tree.full_evaluations):
                self.state['events'].append({'kind':'rollout_count_reconciled',
                    'previous':self.state['full_fit_successes'],'current':len(self.tree.full_evaluations)})
                self.state['full_fit_successes']=len(self.tree.full_evaluations)
        else:
            self.tree=SearchTree(metric_direction(backend.task))
            self.state={'version':1,'fingerprint':fingerprint,'run_id':uuid.uuid4().hex,'status':'created','identity':identity,
                        'configuration':self.cfg.to_dict(),'candidates':{},'initial_ids':[],'initial_selected':[],
                        'initialization_complete':False,'expansions':[],'search_attempts':0,'api_calls':0,
                        'low_fit_attempts':0,'full_fit_attempts':0,'full_fit_successes':0,'pending':None,'events':[],
                        'test_access':False}
        if self.state['initial_selected']:
            initial=[self.state['candidates'][i] for i in self.state['initial_ids']]
            expected=self.initial_ranking(initial)['candidate_ids']
            if self.state['initial_selected']!=expected:
                raise ValueError('Frozen initial full-method selection differs from current top3/all policy; use a new output directory')
        self.state['rollout_budget_basis']='successful_full_evaluations'
        anchor=(self.state.get('pending') or {}).get('region_anchor')
        if anchor and anchor.get('low_error') and infrastructure_failure(anchor['low_error']):
            anchor['low_interruption']=anchor.pop('low_error');anchor['status']='interrupted'
        if anchor and anchor.get('low_report') and resume:
            recover=getattr(backend,'recover_evaluation',None)
            if recover is not None and recover(anchor,'low') is None:
                raise ValueError('Committed parent-anchor artifacts missing during recovery')
        for probe in self._regional_probes():
            if probe.get('low_error') and infrastructure_failure(probe['low_error']):
                probe['low_interruption']=probe.pop('low_error');probe['status']='interrupted'
            if probe.get('low_report') and resume:
                recover=getattr(backend,'recover_evaluation',None)
                if recover is not None and recover(probe,'low') is None:
                    raise ValueError('Committed regional-probe artifacts missing during recovery')
        scope={'task_id':str(backend.task.get('name',backend.task.get('target','task'))),
               'split_id':str(identity.get('split_id',fingerprint)), 'arm':str(identity.get('arm',identity.get('coreset_index',0))),
               'seed':int(identity.get('seed',identity.get('model_seed',self.cfg.bootstrap_seed))), 'run_id':self.state['run_id']}
        self.memory=ExperienceMemory(self.out/'experience',scope=scope,**self.memory_options)
        backend.request_guard=self.before_request;backend.before_fit=self.before_fit
        backend.repair_memory=self.memory.retrieve_repairs
        backend.search_context_provider=self.search_context
        self.checkpoint()

    def _regional_probes(self):
        """Auxiliary low fits never become tree candidates or full rollouts."""
        probes=list((self.state.get('pending') or {}).get('region_probes',{}).values())
        probes.extend(self.state.get('initial_skill_pending',{}).get('region_probes',{}).values())
        for replay in self.state.get('skill_validation',{}).values():
            probes.extend(replay.get('region_probes',{}).values())
        return probes

    def initial_ranking(self,initial):
        """Strict initial admission follows the frozen screening metric ranking."""
        screening=self.state.get('initial_skill_screening') or {}
        rows=initial
        if screening.get('branch_count',0)>1:
            scores=screening.get('combined',{}).get('scores',{})
            if (set(scores)!={c['id'] for c in initial} or
                    any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v)
                        for v in scores.values())):
                raise ValueError('Initial dual screening requires finite scores for all five methods')
            rows=[{**c,'low_score':scores[c['id']]} for c in initial]
        return shortlist_candidates(rows,{},self.tree.direction,
            limit=self.cfg.initial_full_count,minimum=self.cfg.initial_full_count)

    def checkpoint(self):
        self.state['tree']=self.tree.to_dict()
        # Do not rewrite large snapshots for no-op metadata checkpoints. Time
        # alone is not a state transition; dispatch/result commits still change
        # the digest and are durably written before subsequent work starts.
        safe=json_safe(self.state)
        digest=hashlib.sha256(json.dumps({k:v for k,v in safe.items() if k!='elapsed_seconds'},sort_keys=True).encode()).hexdigest()
        if digest!=self._checkpoint_digest:
            self.state['elapsed_seconds']=self.active_elapsed()
            safe['elapsed_seconds']=self.state['elapsed_seconds']
            atomic_json(self.state_path,safe);self._checkpoint_digest=digest
        best=self.tree.global_best_id
        summary={'status':self.state['status'],'stop_reason':self.state.get('stop_reason'),
                 'mcts_enabled':self.cfg.enabled,'mcs_enabled':self.cfg.mcs_enabled,
                 'full_train_methods':self.cfg.full_train_methods,
                 'initial_full_method_count':self.cfg.initial_full_count,
                 'initialization_complete':self.state['initialization_complete'],'initial_paths':self.state['initial_selected'],
                 'completed_expansions':len(self.state['expansions']),'search_attempts':self.state['search_attempts'],
                 'candidates':len(self.state['candidates']),'retained_nodes':len(self.tree.nodes),
                 'api_calls':self.state['api_calls'],'low_fit_attempts':self.state['low_fit_attempts'],
                 'full_fit_attempts':self.state['full_fit_attempts'],'full_fit_successes':self.state['full_fit_successes'],
                 'rollout_budget_basis':'successful_full_evaluations',
                 'remaining_rollouts':max(0,self.cfg.max_rollout_successes-self.state['full_fit_successes']),
                 'best_id':best,'best_full_score':self.tree.candidates[best]['full_score'] if best else None,
                 'metric':self.backend.task['metric'],'direction':metric_direction(self.backend.task),'test_access':False,
                 'scope':'Search validation results only; not final test performance.'}
        if self.cfg.base_expansions is not None:
            summary['expansion_plan']={'regular':self.cfg.base_expansions,
                'fallback_limit':self.cfg.max_expansions-self.cfg.base_expansions,
                'successful_full_limit':self.cfg.max_rollout_successes}
        if self.state.get('stage_pause'):
            summary['stage_pause']=deepcopy(self.state['stage_pause'])
        if self.region_controller is not None:
            batches=self.state['expansions']+self.state.get('aborted_batches',[])
            summary['round_bandit']={'enabled':True,
                'decisions':sum(bool(b.get('region_binding')) for b in batches),
                'feedback_events':sum(b.get('region_feedback') is not None for b in batches),
                'informative_rewards':sum(bool((b.get('region_feedback') or {}).get('update_eligible')) for b in batches)}
        summary=json_safe(summary)
        summary_digest=hashlib.sha256(json.dumps(summary,sort_keys=True).encode()).hexdigest()
        if summary_digest!=self._summary_digest:
            atomic_json(self.out/'search_summary.json',summary);self._summary_digest=summary_digest
        if best:
            best_record=json_safe(self.state['candidates'][best])
            best_digest=hashlib.sha256(json.dumps(best_record,sort_keys=True).encode()).hexdigest()
            if best_digest!=self._best_digest:
                atomic_json(self.out/'best_candidate.json',best_record);self._best_digest=best_digest

    def event(self,kind,**details):
        self.state['events'].append({'kind':kind,**json_safe(details)});self.checkpoint()

    def active_elapsed(self):
        """Charged execution time, excluding idle intervals between stage runs."""
        return self.elapsed_base+(time.monotonic()-self.started if self._clock_running else 0.)

    def _start_clock(self):
        if not self._clock_running:
            self.started=time.monotonic();self._clock_running=True

    def _stop_clock(self):
        if self._clock_running:
            self.elapsed_base=self.active_elapsed();self._clock_running=False

    def check_wall(self):
        if self.active_elapsed() >= self.cfg.max_wall_seconds:
            raise SearchBudgetExceeded('wall_time_budget')

    def rollout_allocation(self,round_number=None):
        """Describe remaining success quota using the default scheduler's math.

        Reservations describe capacity, never permission to run an extra fit.
        The current round still requires its primary until any full result has
        committed; this also handles resuming between result and batch commits.
        """
        pending=self.state.get('pending') or {}
        completed=len(self.state['expansions'])
        round_number=round_number if round_number is not None else pending.get('round_number',completed+1)
        remaining=max(0,self.cfg.max_rollout_successes-self.state['full_fit_successes'])
        decision=(pending.get('extra_evaluation') if self.cfg.rollout_schedule=='evidence_reserved' else
                  pending.get('calibration') if self.cfg.rollout_schedule=='reserved' else pending.get('second_decision'))
        decision_status=('not_yet_screened' if not pending.get('shortlist') else
                         'screened_not_decided' if decision is None else
                         'frozen' if decision.get('candidate_id') or decision.get('run_second') else 'screened_not_scheduled')
        result={'remaining_successes':remaining,'initialization_reserved':0,
                'remaining_primary_required':0,'remaining_primary_reserved':0,
                'independent_audit_required':0,'independent_audit_reserved':0,
                'available_extra_successes':0,'unallocated_successes':remaining,
                'shortfall':0,'extra_decision_status':decision_status,
                'scope':'Quota only; until low/MCS and a per-round decision are frozen, no extra evaluation is promised.'}
        if not self.cfg.enabled:
            initial_successes=sum(bool(self.state['candidates'].get(i,{}).get('full_report'))
                                  for i in self.state['initial_selected'])
            required=0 if self.state['initialization_complete'] else max(0,self.cfg.initial_full_count-initial_successes)
            result.update(initialization_reserved=min(remaining,required),
                          unallocated_successes=max(0,remaining-required),
                          shortfall=max(0,required-remaining),
                          scope='MCTS disabled: all five initial methods receive full evaluations; no expansions.')
            return result
        if self.cfg.rollout_schedule!='evidence_reserved':
            result['scope']='Legacy schedule retains its existing per-round decisions; the remaining shared quota is not a promise of default-policy reservations or extra evaluations.'
            return result
        initial_successes=sum(bool(self.state['candidates'].get(i,{}).get('full_report'))
                              for i in self.state['initial_selected'])
        initial_required=0 if self.state['initialization_complete'] else max(0,self.cfg.initial_keep-initial_successes)
        initial_reserved=min(remaining,initial_required)
        current_has_full=any(self.state['candidates'].get(i,{}).get('full_report')
                             for i in pending.get('candidate_ids',[]))
        base_expansions=(self.cfg.max_expansions if self.cfg.base_expansions is None
                         else self.cfg.base_expansions)
        primary_required=max(0,base_expansions-round_number+1-int(current_has_full))
        # Fallback rounds reserve their current primary, not all later fallback
        # rounds. This prevents nine reserved primaries from starving the audit.
        if self.cfg.base_expansions is not None and round_number<=self.cfg.max_expansions:
            primary_required=max(primary_required,int(not current_has_full))
        primary_reserved=min(max(0,remaining-initial_reserved),primary_required)
        past=self.state['expansions']+self.state.get('aborted_batches',[])
        audit_done=any(e.get('extra_evaluation',{}).get('kind')=='independent_audit' and
                       e['extra_evaluation'].get('status')=='success' for e in past)
        current_extra=pending.get('extra_evaluation',{})
        if current_extra.get('kind')=='independent_audit':
            audit_done=audit_done or bool(self.state['candidates'].get(current_extra.get('candidate_id'),{}).get('full_report'))
        rounds_left=max(0,self.cfg.max_expansions-completed)
        audit_required=int(not audit_done and bool(rounds_left))
        # This is exactly the pre-full availability used by evidence_rollouts;
        # clamping a negative shortage cannot change any >0 dispatch condition.
        after_primary=max(0,remaining-initial_required-primary_required)
        audit_reserved=min(after_primary,audit_required)
        available_extra=max(0,after_primary-audit_required) if rounds_left and not self.cfg.root_trials else 0
        result.update(initialization_reserved=initial_reserved,
                      remaining_primary_required=primary_required,
                      remaining_primary_reserved=primary_reserved,
                      independent_audit_required=audit_required,
                      independent_audit_reserved=audit_reserved,
                      available_extra_successes=available_extra,
                      unallocated_successes=max(0,remaining-initial_reserved-primary_reserved-audit_reserved-available_extra),
                      shortfall=(initial_required-initial_reserved)+(primary_required-primary_reserved)+(audit_required-audit_reserved))
        return result

    def search_context(self):
        pending=self.state.get('pending') or {}
        recent=self.state['expansions'][-3:]
        streak=0
        for batch in reversed(self.state['expansions']):
            if batch.get('improved_global') is not False:break
            streak+=1
        conflict=self.historical_ranking_conflicts(set(pending.get('candidate_ids',[])),
            reference_ids={pending.get('parent_id'),self.tree.global_best_id}-{None},max_examples=2)
        comparisons={tuple(row['candidate_ids']):row for row in conflict.pop('example_comparisons',[])}
        examples=[]
        for first,second in conflict.pop('example_pairs'):
            a,b=(self.state['candidates'][i] for i in (first,second))
            low_values=comparisons.get((first,second),{}).get('low_values',[a['low_score'],b['low_score']])
            examples.append({'candidate_ids':[first,second],
                'low_preferred_id':first if self.tree.sign*(low_values[0]-low_values[1])>0 else second,
                'full_preferred_id':first if self.tree.sign*(a['full_score']-b['full_score'])>0 else second})
        conflict.update(examples=examples,
            scope='Previously full-evaluated siblings/parent-child subset only; ties excluded. Selected observational evidence, not an overall proposal error rate. These are candidate IDs; detailed cards may be omitted.')
        elapsed=self.active_elapsed()
        completed=len(self.state['expansions'])
        result={'expansion_candidates':1 if pending.get('depth_direct') else self.cfg.expansion_candidates, 'global_best_id':self.tree.global_best_id,
                'remaining':{'api_calls':max(0,self.cfg.max_api_calls-self.state['api_calls']),
                             'successful_full_rollouts':max(0,self.cfg.max_rollout_successes-self.state['full_fit_successes']),
                             'low_fit_attempts':max(0,self.cfg.max_low_fit_attempts-self.state['low_fit_attempts']),
                             'expansions':max(0,self.cfg.max_expansions-completed),
                             'search_attempts':max(0,self.cfg.max_search_attempts-self.state['search_attempts']),
                             'wall_seconds':round(max(0.,self.cfg.max_wall_seconds-elapsed),3)},
                'progress':{'completed_rounds':completed,
                            'current_round':pending.get('round_number',completed+1 if completed<self.cfg.max_expansions else None),
                            'recent_global_improvements':[
                                {'round':batch.get('round_number',completed-len(recent)+index+1),
                                 'improved_global':batch.get('improved_global'),'winner_id':batch.get('winner_id')}
                                for index,batch in enumerate(recent)],
                            'consecutive_no_improvement_rounds':streak},
                'rollout_allocation':self.rollout_allocation(),
                'proxy_reliability':conflict,
                'rollout_schedule':self.cfg.rollout_schedule,
                'parent_selection':pending.get('parent_selection'),
                'allowed_changes':'One main direction per candidate: FE, model, or parameters; both stage plans frozen before evaluation.'}
        if self.region_controller is not None:
            result['regional_sampling']={
                'mode':'one shared versioned coreset per expansion; parent re-evaluated as anchor',
                'parent_coreset_version':self.state['candidates'].get(pending.get('parent_id'),{}).get('coreset_binding',{}).get('version'),
                'comparison_rule':'Compare low scores only on the same coreset version. Full validation remains fixed.',
                'planning_distribution_scope':'The static EDA coreset distribution describes the initialization coreset.'}
        return result

    def before_request(self):
        self.check_wall()
        if self.state['api_calls']>=self.cfg.max_api_calls:raise SearchBudgetExceeded('api_call_budget')
        self.state['api_calls']+=1;self.state['inflight']={'kind':'generation','call':self.state['api_calls'],
            **self.state.get('active_evaluation',{})};self.checkpoint()

    def full_available(self):
        # Attempts are an audit/cost counter, never the rollout quota. API, wall,
        # low-fit and search-attempt budgets independently bound failed work.
        return self.state['full_fit_successes']<self.cfg.max_rollout_successes

    def check_sampling_budget(self):
        self.check_wall()
        if not self.full_available():
            raise SearchBudgetExceeded('full_rollout_budget')
        if self.state['low_fit_attempts'] >= self.cfg.max_low_fit_attempts:
            raise SearchBudgetExceeded('low_fit_budget')
        if self.state['api_calls'] >= self.cfg.max_api_calls:
            raise SearchBudgetExceeded('api_call_budget')

    def before_fit(self,stage,candidate):
        self.check_wall();full=stage in ('full','full_train');key='full_fit_attempts' if full else 'low_fit_attempts'
        if full and not self.full_available():raise SearchBudgetExceeded('full_rollout_budget')
        if not full and self.state[key]>=self.cfg.max_low_fit_attempts:raise SearchBudgetExceeded('low_fit_budget')
        self.state[key]+=1
        self.state['inflight']={'kind':'fit','stage':'full' if full else 'low','candidate_id':candidate['id'],'attempt':self.state[key]}
        self.checkpoint()

    def parent(self,c):return self.state['candidates'].get(c.get('parent_id'))

    def remember(self,c,*,checkpoint=True):
        parent=self.parent(c)
        if parent and c.get('parent_low_reference'):
            parent={**parent,'low_report':c['parent_low_reference'],
                    'low_score':c['parent_low_reference']['score'],
                    'coreset_binding':c.get('coreset_binding')}
        self.memory.upsert(c,parent=parent)
        if checkpoint:self.checkpoint()

    def add(self,c,retain=False,*,checkpoint=True):
        c=deepcopy(c);ident=c['id']
        if ident in self.state['candidates']:raise ValueError('Candidate IDs cannot be reused')
        self.state['candidates'][ident]=c;self.tree.add_candidate(c,retain=retain)
        if checkpoint:self.checkpoint()
        return c

    async def evaluate(self,c,stage):
        auxiliary=bool(c.get('anchor_of'))
        if auxiliary and stage!='low':raise ValueError('Parent anchors cannot spend full rollout quota')
        if c.get(stage+'_report') is not None:return c[stage+'_report']
        if c.get(stage+'_error') is not None:return None
        for retry in range(self.cfg.infrastructure_retries+1):
            try:
                # The before_request/before_fit guard charges each actual fit;
                # restoring already committed evidence spends no new quota.
                self.state['active_evaluation']={'candidate_id':c['id'],'stage':stage}
                recover=getattr(self.backend,'recover_evaluation',None)
                # A previous iteration may already have admitted this report
                # before a memory/checkpoint write failed. Never refit it.
                result=c.get(stage+'_report') or (recover(c,stage) if recover else None)
                if result is None:
                    self.check_wall()
                    result=await self.backend.evaluate(c,stage)
                else:
                    self.state['events'].append({'kind':'evaluation_recovered','candidate_id':c['id'],'stage':stage})
                score=result['score']
                if isinstance(score,bool) or not isinstance(score,Real) or not math.isfinite(score):
                    raise ValueError('Evaluation score must be a finite numeric scalar, not a boolean')
                c[stage+'_report']=result;c[stage+'_score']=float(result['score'])
                c['status']='low_evaluated' if stage=='low' else 'full_evaluated'
                c.pop(stage+'_interruption',None)
                if c.get('failure_kind')=='infrastructure_interruption':
                    for key in ('failure_kind','failure_stage','error'):c.pop(key,None)
                if stage=='low' and not auxiliary:
                    self.tree.candidates[c['id']]['low_score']=c['low_score']
                    if c.get('coreset_binding'):
                        self.tree.candidates[c['id']]['coreset_binding']=deepcopy(c['coreset_binding'])
                if stage=='full':
                    if self.tree.register_rollout(c['id'],c['full_score'],attempt_id=c['id']+':full'):
                        self.state['full_fit_successes']+=1
                self.state.pop('inflight',None);self.state.pop('active_evaluation',None)
                if auxiliary:self.checkpoint()
                else:self.remember(c)
                return result
            except SearchBudgetExceeded:raise
            except Exception as exc:
                self.state.pop('inflight',None);self.state.pop('active_evaluation',None)
                if infrastructure_failure(exc):
                    c[stage+'_interruption']=f'{type(exc).__name__}: {exc}'
                    c['status']='interrupted'
                    c['failure_kind']='infrastructure_interruption'
                    self.event('fit_infrastructure_interrupted',candidate_id=c['id'],stage=stage,
                               retry_index=retry,error=c[stage+'_interruption'])
                    if isinstance(exc,TaskDeadlineExceeded):raise
                    if retry < self.cfg.infrastructure_retries:
                        continue
                    # Preserve pending initialization/batch for explicit resume.
                    raise InfrastructureInterruption(c[stage+'_interruption']) from exc
                c[stage+'_error']=f'{type(exc).__name__}: {exc}'
                if not auxiliary:self.remember(c,checkpoint=False)
                self.event('anchor_failed' if auxiliary else 'candidate_failed',candidate_id=c['id'],stage=stage,error=c[stage+'_error']);return None

    async def initialize(self):
        if self.state['initialization_complete']:return True
        if not self.state['initial_ids'] or any(i not in self.state['candidates'] for i in self.state['initial_ids']):
            initial=self.backend.initial_candidates()
            if len(initial)!=5:raise ValueError('Initialization requires exactly five source methods')
            self.state['initial_ids']=[c['id'] for c in initial];self.checkpoint()
            for c in initial:
                if c['id'] in self.state['candidates']:continue
                c['parent_id']=None;c['branch_id']=c['id'];self.add(c)
        initial=[self.state['candidates'][i] for i in self.state['initial_ids']]
        for c in initial:await self.evaluate(c,'low')
        valid=[c for c in initial if c.get('low_report')]
        if len(valid)!=5:
            self.state.update(status='initialization_failed',stop_reason='All five initial methods must pass low evaluation after bounded repair');self.checkpoint();return False
        if self.region_controller is not None and hasattr(self.region_controller,'after_initial_low'):
            await self.region_controller.after_initial_low(self,valid)
        if not self.state['initial_selected']:
            override=(self.region_controller.initial_screening_override(self,valid)
                if self.region_controller is not None and hasattr(self.region_controller,'initial_screening_override') else None)
            mcs=(deepcopy(override['mcs']) if override is not None
                 else compare_candidates(self.backend,valid,self.cfg,self.cfg.bootstrap_seed))
            # Initial top3 means strictly the V_core ranking. MCS remains an
            # initial diagnostic and still screens subsequent search children.
            count=self.cfg.initial_full_count
            selected=self.initial_ranking(valid)
            if override is not None:
                proposed=deepcopy(override['shortlist'])
                if proposed.get('candidate_ids')!=selected['candidate_ids']:
                    raise ValueError('Initial screening override must preserve strict frozen-metric top3')
                if not set(mcs.get('survivor_method_ids',[]))<={c['id'] for c in valid}:
                    raise ValueError('Initial screening override returned unknown MCS survivors')
                selected.update(proposed)
            else:
                # Preserve historical diagnostic metadata without changing its
                # role: initial admission is strict ranking, never an MCS filter.
                selected=shortlist_candidates(valid,{**mcs,'selection_status':'ranking_only'},
                    self.tree.direction,limit=count,minimum=count)
            selected.update(policy=self.cfg.full_train_methods,mcs_role='initial_diagnostic_only')
            self.state['initial_selected']=selected['candidate_ids'];self.state['initial_mcs']=mcs;self.state['initial_admission']=selected
            self.state['initial_full_evaluation_plan']={
                'full_train_methods':self.cfg.full_train_methods,'method_count':count,
                'candidate_ids':list(selected['candidate_ids']),
                'selection_source':('frozen branch-metric V_core ranking only' if override is not None else
                    'V_core ranking only' if self.cfg.enabled else 'predeclared complete initial bank')}
            self.checkpoint()
        # Admission is idempotent: a checkpoint may have persisted the shortlist
        # before all selected initial candidates were retained.
        for c in initial:
            c['mcs_survivor']=c['id'] in self.state['initial_mcs']['survivor_method_ids']
            c['admitted']=c['id'] in self.state['initial_selected']
            c['decision']=('initial_three_paths' if self.cfg.enabled else 'initial_all_five') if c['admitted'] else 'initial_not_admitted'
            if c['admitted']:self.tree.retain_candidate(c['id'])
            self.remember(c)
        for ident in self.state['initial_selected']:await self.evaluate(self.state['candidates'][ident],'full')
        if any(not self.state['candidates'][i].get('full_report') for i in self.state['initial_selected']):
            self.state.update(status='initialization_failed',stop_reason=f'All {self.cfg.initial_full_count} scheduled initial methods require successful full evaluations');self.checkpoint();return False
        self.state['initialization_complete']=True;self.event('initialization_completed',retained=self.state['initial_selected']);return True

    def archive_methods(self, candidates, screening, batch, feedback=None):
        """Commit sibling ranks once the batch closes, including partial full observations."""
        combined = (screening or {}).get('combined', {}).get('scores', {})
        low = {c['id']: combined.get(c['id'], c.get('low_score')) for c in candidates}
        full = {c['id']: c.get('full_score') for c in candidates}
        def position(scores, ident):
            observed = [x for x in scores.values() if isinstance(x, Real) and math.isfinite(x)]
            value = scores.get(ident)
            rank = (1 + sum(self.tree.sign*(x-value) > 0 for x in observed)
                    if value in observed else None)
            return dict(rank=rank, observed=len(observed), candidates=len(candidates), score=value)
        records = []
        for c in candidates:
            ident = c['id']
            c['screening'] = dict(batch=batch, coreset=position(low, ident), full=position(full, ident),
                                  admitted=c.get('admitted'), mcs_survivor=c.get('mcs_survivor'))
            self.remember(c, checkpoint=False)
            card = self.memory.cards[ident]
            records.append(dict(reference=card['method_path'], plan=c.get('idea'), code_path=c.get('code_path'),
                                code_sha256=c.get('code_sha256'), decision=c.get('decision'),
                                status={k:r.get('status') for k,r in card['results'].items()}))
        eda = getattr(self.backend, 'eda', {}) or {}
        directory = self.out/'trajectories'
        directory.mkdir(exist_ok=True)
        atomic_json(directory/(batch+'.json'), json_safe(dict(
            batch=batch, task=self.backend.task.get('name'), metric=self.backend.task['metric'],
            direction=self.tree.direction, eda={k:eda[k] for k in ('scope','train_rows','target','task_type') if k in eda},
            numeric_columns=len(eda.get('numeric_columns', [])) if 'numeric_columns' in eda else None,
            categorical_columns=len(eda.get('categorical_columns', [])) if 'categorical_columns' in eda else None,
            skill_fact_id=(feedback or {}).get('fact_id'), methods=records)))

    def ancestors(self,c):
        ids=[]
        while c and c.get('parent_id'):
            ids.append(c['parent_id']);c=self.state['candidates'].get(c['parent_id'])
        return ids

    def finish_batch(self,pending,candidates,*,stop_reason=None):
        """Admit available full evidence and archive the batch without new work."""
        if self.region_controller is not None and pending.get('region_binding') and stop_reason is None:
            pending['region_feedback']=self.region_controller.finish(self,pending,candidates)
        # A full result can commit before its scheduling metadata. In particular,
        # a resumed run may hit a budget guard before re-entering the scheduler.
        # Reconcile from that committed evidence; never infer an unrun outcome.
        for key in ('extra_evaluation','calibration'):
            extra=pending.get(key)
            if not extra or not extra.get('candidate_id'):continue
            candidate=self.state['candidates'].get(extra['candidate_id'],{})
            if candidate.get('full_report'):
                extra['status']='success'
            elif candidate.get('full_error'):
                extra['status']='failed'
        selected_ids=pending.get('rollout_order',pending.get('shortlist',{}).get('candidate_ids',pending['candidate_ids']))
        successful=[self.state['candidates'][i] for i in selected_ids
                    if i in self.state['candidates'] and self.state['candidates'][i].get('full_report')]
        parent=self.state['candidates'][pending['parent_id']]
        if successful:
            # Stable ties follow rollout order, matching global-best registration.
            winner=min(successful,key=lambda c:(-1 if self.tree.direction=='maximize' else 1)*c['full_score'])
            self.tree.retain_candidate(winner['id'])
            pending.update(winner_id=winner['id'],improved_global=self.tree.global_best_id!=pending['best_before'],
                improved_parent=((winner['full_score']-parent['full_score'])*(1 if self.tree.direction=='maximize' else -1)>0))
        else:pending.update(winner_id=None,improved_global=False,improved_parent=False)
        for c in candidates:
            c['admitted']=c['id']==pending['winner_id']
            c['decision']='admitted_batch_winner' if c['admitted'] else ('evaluated_not_selected' if c.get('full_report') else 'archived_without_full_rollout')
            if stop_reason:c['search_stop_reason']=stop_reason
        self.archive_methods(candidates, pending.get('skill_screening'), f"expansion_{pending['iteration']:04d}",
                             pending.get('region_feedback') if isinstance(pending.get('region_feedback'), dict) else None)
        if stop_reason:pending['search_stop_reason']=stop_reason
        # A batch with no valid full evaluation did not complete an expansion.
        if (pending.get('search_stage') or {}).get('phase') == 'single_parameter_check':
            check = self.state['parameter_check']
            candidate = self.state['candidates'].get(check.get('candidate_id'), {})
            if candidate.get('full_report'):
                check['signed_full_gain'] = self.tree.sign * (candidate['full_score'] - parent['full_score'])
            check['status'] = ('observed' if candidate.get('full_report') else
                               'no_valid_candidate' if not candidate else 'not_completed')
        collection='aborted_batches' if not successful else 'expansions'
        self.state.setdefault(collection,[]).append(deepcopy(pending));self.state['pending']=None
        self.event('expansion_aborted' if collection=='aborted_batches' else 'expansion_completed',
                   iteration=pending['iteration'],winner_id=pending['winner_id'],stop_reason=stop_reason)

    def finish_budget_stop(self,reason):
        # Budget guards fire before dispatch: no operation is still running at
        # this point. Preserve already charged attempts and completed evidence.
        self.state.pop('inflight',None);self.state.pop('active_evaluation',None)
        pending=self.state.get('pending')
        if pending:
            for candidate in pending.get('generated',[]):
                if candidate['id'] not in self.state['candidates']:self.add(candidate)
            candidates=[self.state['candidates'][i] for i in pending['candidate_ids'] if i in self.state['candidates']]
            self.finish_batch(pending,candidates,stop_reason=reason)

    def historical_ranking_conflicts(self,current_ids,*,reference_ids=None,max_examples=6):
        """Use completed earlier full evidence only; never current-round outcomes."""
        observed=[c for i,c in self.state['candidates'].items() if i not in current_ids
                  and c.get('low_report') and c.get('full_report')
                  and isinstance(c.get('low_score'),Real) and isinstance(c.get('full_score'),Real)
                  and math.isfinite(c['low_score']) and math.isfinite(c['full_score'])]
        comparisons=0; conflicts=[];details={}
        for pos,a in enumerate(observed):
            for b in observed[pos+1:]:
                # Restrict to siblings or parent/child comparisons, whose
                # relationship is directly relevant to proposal screening.
                related=(a.get('parent_id')==b.get('parent_id') or
                         a.get('parent_id')==b['id'] or b.get('parent_id')==a['id'])
                if not related:continue
                versions=[c.get('coreset_binding',{}).get('version') for c in (a,b)]
                values=[a['low_score'],b['low_score']]
                if versions[0]!=versions[1]:
                    # A child carries its parent's re-fit on that child's core.
                    # Never compare two original scores from different cores.
                    anchor_index=(0 if b.get('parent_id')==a['id'] else
                                  1 if a.get('parent_id')==b['id'] else None)
                    if anchor_index is None:continue
                    child=(b if anchor_index==0 else a)
                    reference=child.get('parent_low_reference') or {}
                    if reference.get('coreset_version')!=versions[1-anchor_index] or not isinstance(reference.get('score'),Real):
                        continue
                    values[anchor_index]=reference['score']
                low=values[0]-values[1];full=a['full_score']-b['full_score']
                # New skill-driven observations carry the reward's frozen
                # practical-tie threshold. For parent/child comparisons use
                # the child's decision threshold, never its parent's earlier
                # initialization scale. Legacy observations remain unchanged.
                child=(a if a.get('parent_id')==b['id'] else
                       b if b.get('parent_id')==a['id'] else None)
                thresholds=[c.get('sampling_practical_tolerance') for c in ([child] if child else (a,b))]
                tolerance=max([v for v in thresholds if isinstance(v,Real) and
                               not isinstance(v,bool) and math.isfinite(v) and v>=0] or [1e-12])
                if abs(low)<=tolerance or abs(full)<=tolerance:continue
                comparisons+=1
                if low*full<0:
                    pair=[a['id'],b['id']];conflicts.append(pair)
                    details[tuple(pair)]={'candidate_ids':pair,'low_values':values,
                                          'practical_tolerance':tolerance}
        examples=conflicts[-max_examples:]
        if reference_ids:
            examples=sorted(reversed(conflicts),key=lambda pair:not bool(set(pair)&set(reference_ids)))[:max_examples]
        return {'comparable_pairs':comparisons,'conflict_count':len(conflicts),
                'example_pairs':examples,'example_comparisons':[details[tuple(pair)] for pair in examples],
                'scope':'Previously full-evaluated siblings/parent-child only; low comparisons require the same coreset version (parent anchors allowed); selected observational evidence.'}

    def plan_direct_full(self, pending, candidates):
        """Freeze a no-screening batch before choosing a region or observing full scores."""
        if 'direct_full' in pending:
            return True
        key = getattr(self.backend, 'parameter_stage_key', None)
        if (not self.cfg.staged_search or self.cfg.rollout_schedule != 'evidence_reserved'
                or not key or not candidates or pending.get('region_binding')
                or any(c.get('implementation_mode') != 'inherited_parameters' for c in candidates)):
            return False
        parent = self.state['candidates'][pending['parent_id']]
        ids = [c['id'] for c in candidates]
        observed = [c for c in self.state['candidates'].values() if c.get('full_report')
                    and (c['id'] == parent['id'] or c.get('parent_id') == parent['id']) and c['id'] not in ids]
        low_keys = {c['id']: key(c, 'coreset') for c in candidates}
        full_keys = {c['id']: key(c, 'full_train') for c in [*candidates, *observed]}
        if any(low_keys[i] is None or full_keys[i] is None for i in ids):
            return False
        plan = full_comparison_plan(candidates, ids, observed, lambda c, phase: full_keys[c['id']])
        eligible = plan['eligible_ids']
        if not eligible or (len(eligible) == 1 and plan['duplicate_full_of']):
            reason = 'no_competing_new_full_configurations'
        elif (len({low_keys[i] for i in eligible}) == 1 and
              (len(eligible) > 1 or low_keys[eligible[0]] == key(parent, 'coreset'))):
            reason = 'identical_effective_coreset_constructor'
        else:
            return False
        self.prioritize_parameter_trial(pending, plan)
        eligible = list(dict.fromkeys(plan['primary_ids'] + eligible))
        pool = eligible[1:]
        allocation = self.rollout_allocation(pending['round_number'])
        extra = dict(kind=None, candidate_id=None, pool_ids=pool, selection_probability=None,
                     status='not_scheduled', selection_frozen_before_full=True)
        audit_round = self.cfg.independent_audit_round
        if not self.cfg.root_trials:
            audit_round = min(audit_round, self.cfg.max_expansions)
        audit_due = pending['round_number'] >= audit_round
        if pool and audit_due and allocation['independent_audit_reserved']:
            seed = (self.cfg.bootstrap_seed + pending['iteration']) % (2**32)
            draw = pool[int(np.random.default_rng(seed).integers(len(pool)))]
            extra.update(kind='independent_audit', candidate_id=draw, seed=seed,
                         selection_probability=1/len(pool), status='pending',
                         sampling='uniform_distinct_non_primary_before_full_outcomes')
        elif pool and not self.cfg.root_trials and allocation['available_extra_successes']:
            extra.update(kind='competitive', candidate_id=pool[0], status='pending',
                         sampling='predeclared_proposal_order', reasons=['identical_coreset_constructor'])
        pending.update(direct_full={'reason': reason, 'low_keys': low_keys,
                       'full_keys': {i: full_keys[i] for i in ids}, 'parent_id': parent['id']},
                       full_comparison_plan=plan, shortlist={'candidate_ids': eligible,
                       'policy': 'full_distinct_proposal_order_without_coreset_ranking'},
                       extra_evaluation=extra, rollout_order=[])
        inherited_binding = parent.get('coreset_binding') or parent.get('inherited_region_binding')
        for c in candidates:
            if c['id'] in eligible:
                c['direct_full_certificate'] = {'coreset': low_keys[c['id']], 'full_train': full_keys[c['id']]}
                if inherited_binding:
                    c['inherited_region_binding'] = deepcopy(inherited_binding)
        self.checkpoint()
        return True

    async def prepare_region_round(self,pending,candidates):
        """Freeze implementations and check for a direct comparison before sampling."""
        if self.region_controller is None:
            # A staged search can also use a fixed coreset. Preflight only when
            # the complete proposal bank is eligible for the constructor check.
            if (not self.cfg.staged_search or not hasattr(self.backend, 'parameter_stage_key')
                    or not all(c.get('implementation_mode') == 'inherited_parameters' for c in candidates)):
                return
        for candidate in candidates:
            if candidate.get('sampling_prepared') or candidate.get('low_error'):continue
            if candidate.get('status') in {'failed','error'}:
                # A failed planning/implementation artifact may have no code
                # file. It is an invalid proposal, not an infrastructure outage.
                candidate['low_error']=candidate.get('error') or 'Candidate generation failed before sampling'
                self.remember(candidate,checkpoint=False);self.checkpoint()
                continue
            self.check_wall()
            self.state['active_evaluation']={'candidate_id':candidate['id'],'stage':'low'}
            try:
                await self.backend.prepare_sampling_candidate(candidate)
                candidate['sampling_prepared']=True
            except SearchBudgetExceeded:raise
            except Exception as exc:
                if infrastructure_failure(exc):raise
                candidate['low_error']=f'{type(exc).__name__}: {exc}'
                self.remember(candidate,checkpoint=False)
            finally:
                self.state.pop('active_evaluation',None)
                self.state.pop('inflight',None)
                self.checkpoint()
        if self.region_controller is None:
            self.plan_direct_full(pending, [c for c in candidates if not c.get('low_error')])
            return
        if 'region_candidate_ids' not in pending:
            pending['region_candidate_ids']=[c['id'] for c in candidates if not c.get('low_error')]
            self.checkpoint()
        # Membership is frozen before sampling; later fit failures cannot
        # rewrite the bank that was supplied to the bandit on resume.
        viable=[c for c in candidates if c['id'] in pending['region_candidate_ids']]
        if not viable:return
        if not pending.get('region_binding'):
            self.check_sampling_budget()
        if self.plan_direct_full(pending, viable):return
        binding=await self.region_controller.prepare(self,pending,viable)
        pending['region_binding']=binding
        for candidate in viable:
            self.backend.bind_coreset(candidate,binding)
            candidate['sampling_contract_frozen']=True
        if pending.get('skip_parent_action_fit'):
            self.checkpoint()
            return
        if not pending.get('region_anchor'):
            parent=self.state['candidates'][pending['parent_id']]
            pending['region_anchor']=self.backend.make_coreset_anchor(
                parent,binding,f"anchor_{pending['iteration']:04d}_{parent['id']}")
        self.checkpoint()
        anchor=pending['region_anchor']
        await self.evaluate(anchor,'low')
        if anchor.get('low_report'):
            for candidate in viable:
                candidate['parent_low_reference']=deepcopy(anchor['low_report'])
        self.checkpoint()

    async def score_expansion_coreset(self, pending, candidates):
        prepare = getattr(self.backend, 'prepare_equivalent_low', None)
        anchor = pending.get('region_anchor')
        references = [(anchor, self.state['candidates'][pending['parent_id']])] if anchor else []
        for child in candidates:
            if self.cfg.staged_search and prepare and child.get('sampling_prepared') and references:
                for source, method in references:
                    if prepare(child, source, method=child, reference=method):
                        self.checkpoint()
                        break
            report = await self.evaluate(child, 'low')
            if report:
                references.append((child, child))

    async def evidence_rollouts(self,pending,valid):
        """Reserve primary fits, one independent audit and evidence-led extras.

        Every selection is frozen before this batch's first full fit. Independent
        audits and conditional competitive choices are explicitly separate.
        """
        selected=pending['shortlist']['candidate_ids']
        round_number=pending['round_number']
        if 'extra_evaluation' not in pending:
            ranking_scores=(pending.get('skill_screening') or {}).get('combined',{}).get('scores',{})
            ranked=sorted(valid,key=lambda c:(-self.tree.sign*ranking_scores.get(c['id'],c['low_score']),str(c['id'])))
            stage_key = getattr(self.backend, 'parameter_stage_key', None)
            # Freeze only pre-rollout identities. A later replay must never learn a
            # constructor equivalence from a subsequently trained child.
            if self.region_controller is not None:
                pending['nomination_policy'] = dict(
                    mcs_enabled=self.cfg.mcs_enabled, mcs_alpha=self.cfg.mcs_alpha,
                    bootstrap_replicates=self.cfg.bootstrap_replicates,
                    bootstrap_seed=(self.cfg.bootstrap_seed + pending['iteration']) % (2**32),
                    shortlist_cap=self.cfg.shortlist_cap, full_keys=None, prior_keys={}, single_trials=[])
            comparison = None
            siblings = []
            if pending.get('search_stage') and callable(stage_key):
                parent_id = pending['parent_id']
                observed = [c for c in self.state['candidates'].values()
                            if c.get('full_report') and c['id'] not in pending['candidate_ids']
                            and (c.get('parent_id') == parent_id or c['id'] == parent_id)]
                if 'nomination_policy' in pending:
                    policy = pending['nomination_policy']
                    policy['full_keys'] = {c['id']: stage_key(c, 'full_train') for c in ranked}
                    policy['prior_keys'] = {c['id']: stage_key(c, 'full_train') for c in observed}
                    if pending['search_stage'].get('phase') == 'single_parameter_check':
                        policy['single_trials'] = [c['id'] for c in ranked if self.single_parameter_trial(c)]
                comparison = full_comparison_plan(ranked, selected, observed, stage_key)
                self.prioritize_parameter_trial(pending, comparison)
                comparison['primary_ids'] = comparison['primary_ids'][:self.cfg.shortlist_cap]
                selected = comparison['primary_ids']
                if self.cfg.root_trials and selected:
                    pending['screening_nominees'] = list(pending['shortlist']['candidate_ids'])
                    pending['shortlist']['candidate_ids'] = list(selected)
                if selected:
                    first = self.state['candidates'][selected[0]]
                    low_key = stage_key(first, 'coreset')
                    full_key = stage_key(first, 'full_train')
                    if low_key is not None and full_key is not None:
                        for c in ranked:
                            if c['id'] not in comparison['eligible_ids'] or c['id'] == first['id']:
                                continue
                            other_full = stage_key(c, 'full_train')
                            if (other_full is not None and other_full != full_key and
                                    stage_key(c, 'coreset') == low_key and same_low_on_scored_branches(first, c, pending)):
                                siblings.append(c['id'])
                comparison['primary_low_equivalent_ids'] = siblings
                pending['full_comparison_plan'] = comparison
            pool=[c for c in ranked if c['id'] not in selected[:1] and not c.get('full_error')
                  and (comparison is None or c['id'] in comparison['eligible_ids'])]
            past=self.state['expansions']+self.state.get('aborted_batches',[])
            allocation=self.rollout_allocation(round_number)
            audit_done=not allocation['independent_audit_required']
            audit_round = self.cfg.independent_audit_round
            if not self.cfg.root_trials:
                audit_round = min(audit_round, self.cfg.max_expansions)
            # Preserve one success for each remaining primary expansion. With
            # defaults this leaves three extra successes over the whole run.
            reserve_primary=allocation['remaining_primary_required']
            available=allocation['independent_audit_reserved']+allocation['available_extra_successes']
            conflict=self.historical_ranking_conflicts(set(pending['candidate_ids']))
            survivors=pending['mcs'].get('survivor_method_ids',[])
            ambiguous=(pending['mcs'].get('selection_status')=='completed' and
                       len([c for c in ranked if c['id'] in survivors])>1)
            branch_disagreement=bool((pending.get('skill_screening') or {}).get('ambiguity'))
            extra={'kind':None,'candidate_id':None,'pool_ids':[c['id'] for c in pool],
                   'selection_probability':None,'selection_frozen_before_full':True,
                   'reserved_primary_successes':reserve_primary,'available_extra_successes':max(0,available),
                   'historical_conflicts':conflict,'mcs_ambiguous':ambiguous,
                   'dual_branch_top_disagreement':branch_disagreement,
                   'low_ranks':{c['id']:i+1 for i,c in enumerate(ranked)},
                   'pool_categories':{c['id']:(
                       'mcs_rejected' if pending['mcs'].get('selection_status')=='completed' and c['id'] not in survivors
                       else 'shortlist_truncated' if c['id'] in pending['shortlist'].get('truncated_ids',[])
                       else 'shortlisted_unselected' if c['id'] in selected else 'rank_fallback') for c in pool},
                   'status':'not_scheduled','reasons':[]}
            blind = {}
            if (pending.get('search_stage') or {}).get('phase') == 'parameter_focus':
                parent = self.state['candidates'][pending['parent_id']]
                for candidate in ranked:
                    changes = coreset_blind_parameter_changes(candidate, parent, pending)
                    if changes:
                        blind[candidate['id']] = changes
                extra['coreset_indistinguishable_parameters'] = blind
            if pool and available>0 and not audit_done and round_number>=audit_round:
                seed=(self.cfg.bootstrap_seed+pending['iteration'])%(2**32)
                draw=pool[int(np.random.default_rng(seed).integers(len(pool)))]
                extra.update(kind='independent_audit',candidate_id=draw['id'],seed=seed,
                             selection_probability=1/len(pool),status='pending',
                             reasons=['reserved_independent_audit'],
                             sampling='uniform_valid_non_primary_before_full_outcomes',
                             scope='Conditional on the recorded round, prior history and valid non-primary pool; not all-proposal precision. Execution failures may censor outcomes.')
            else:
                # Original positions 4/6 define when competitive slots become
                # eligible. An unused slot can move to the next round. Keep an
                # audit success reserved until it has actually been obtained.
                competitive_rounds=[r for r in self.cfg.calibration_rounds if r!=self.cfg.independent_audit_round]
                start=min(competitive_rounds,default=self.cfg.max_expansions)
                released=sum(r<=round_number for r in competitive_rounds)
                used=sum(e.get('extra_evaluation',{}).get('kind') in {'competitive','fallback_exploration'} and
                         e['extra_evaluation'].get('status')=='success' for e in past)
                eligible=round_number>=min(start,self.cfg.max_expansions) and used<released
                spendable=allocation['available_extra_successes']
                extra.update(competitive_slots_released=released,competitive_successes=used)
                reasons=(['mcs_ambiguity'] if ambiguous else [])
                if branch_disagreement:reasons.append('dual_branch_top_disagreement')
                if conflict['conflict_count']:reasons.append('prior_low_full_ranking_conflicts')
                blind_choice = next((candidate for candidate in pool if candidate['id'] in blind), None)
                sibling_choice = next((candidate for candidate in pool if candidate['id'] in siblings), None)
                if sibling_choice:
                    reasons.append('primary_and_alternative_have_identical_coreset_configuration')
                    blind_choice = sibling_choice
                    # This pair supplies its own evidence. Use an unreserved slot
                    # now, even if the usual calibration round has not arrived.
                    eligible = True
                if blind_choice and blind_choice['id'] in blind:
                    reasons.append('full_parameter_change_invisible_on_coreset')
                if not self.cfg.root_trials and pool and spendable>0 and eligible and (reasons or round_number==self.cfg.max_expansions):
                    if blind_choice:
                        draw=blind_choice
                    elif ambiguous or branch_disagreement:
                        alternatives=[c for c in pool if c['id'] in survivors]
                        draw=(alternatives or pool)[0]
                    elif conflict['conflict_count']:
                        draw=pool[-1]  # Stress-test a lower ranked alternative.
                    else:draw=pool[0]
                    extra.update(kind='competitive' if reasons else 'fallback_exploration',
                                 candidate_id=draw['id'],status='pending',
                                 reasons=reasons or ['last_round_reserved_budget_fallback'],
                                 sampling='deterministic_evidence_conditioned_choice',
                                 scope='Conditional competitive/exploratory evaluation; not an unbiased audit.')
                else:
                    extra['reasons']=['no_valid_alternative' if not pool else
                                      'primary_or_audit_budget_reserved' if spendable<=0 else
                                      'waiting_for_competitive_round' if not eligible else 'no_ambiguity_or_prior_conflict']
            pending['extra_evaluation']=extra;pending['rollout_order']=[]
            self.checkpoint()
        selected=pending.get('full_comparison_plan', {}).get('primary_ids', selected)
        extra=pending['extra_evaluation'];extra_id=extra['candidate_id']
        # Keep the pre-drawn candidate independent of primary success/failure.
        for ident in selected:
            if ident==extra_id:continue
            c=self.state['candidates'][ident]
            if ident not in pending['rollout_order']:
                pending['rollout_order'].append(ident);c['rollout_role']='primary'
            if not c.get('full_report') and not c.get('full_error'):
                if not self.full_available():raise SearchBudgetExceeded('full_rollout_budget')
                await self.evaluate(c,'full')
            if c.get('full_report'):
                pending['primary_success_id']=ident;break
        if extra_id and extra['status']=='pending':
            c=self.state['candidates'][extra_id]
            if extra_id not in pending['rollout_order']:
                pending['rollout_order'].append(extra_id);c['rollout_role']=extra['kind']
            if not self.full_available() and not c.get('full_report'):
                raise SearchBudgetExceeded('full_rollout_budget')
            await self.evaluate(c,'full')
            extra['status']='success' if c.get('full_report') else 'failed'
        # finish_batch persists final metadata and admission together. The
        # evaluation result itself has already crossed its own commit point.

    async def reserved_rollouts(self,pending,valid):
        """One primary success; pre-drawn audit candidate on scheduled rounds.

        Audit draws use only low evidence, before any full outcome in this batch.
        Pool membership/probability is saved so an audit is not mistaken for an
        unbiased estimate over all proposals or all rounds.
        """
        round_number=pending.get('round_number',len(self.state['expansions'])+1)
        selected=pending['shortlist']['candidate_ids']
        if 'calibration' not in pending:
            ranking_scores=(pending.get('skill_screening') or {}).get('combined',{}).get('scores',{})
            ranked=sorted(valid,key=lambda c:(-self.tree.sign*ranking_scores.get(c['id'],c['low_score']),str(c['id'])))
            pool=[c for c in ranked if c['id']!=selected[0] and not c.get('full_error')]
            seed=(self.cfg.bootstrap_seed+pending['iteration'])%(2**32)
            due=round_number in self.cfg.calibration_rounds
            draw=pool[int(np.random.default_rng(seed).integers(len(pool)))] if due and pool else None
            pending['calibration']={'scheduled':due,'candidate_id':draw['id'] if draw else None,
                'pool_ids':[c['id'] for c in pool], 'seed':seed,
                'selection_probability':1/len(pool) if draw else None,
                'sampling':'uniform_valid_non_primary_before_full_outcomes',
                'scope':'Conditional on this round and its valid non-primary candidates; not all-proposal precision.',
                'pool_categories':{c['id']:(
                    'mcs_rejected' if pending['mcs'].get('selection_status')=='completed' and
                                     c['id'] not in pending['mcs']['survivor_method_ids'] else
                    'shortlist_truncated' if c['id'] in pending['shortlist'].get('truncated_ids',[]) else
                    'shortlisted_unselected' if c['id'] in selected else 'rank_fallback') for c in pool},
                'low_ranks':{c['id']:i+1 for i,c in enumerate(ranked)},
                'status':'pending' if draw else ('no_valid_alternative' if due else 'not_scheduled')}
            pending['rollout_order']=[]
            self.checkpoint()
        audit=pending['calibration'];audit_id=audit['candidate_id']
        # Reserve the audit draw. A failed primary can try the remaining
        # shortlisted candidate; failure never spends a success slot.
        for ident in selected:
            if ident==audit_id:continue
            c=self.state['candidates'][ident]
            if ident not in pending['rollout_order']:
                pending['rollout_order'].append(ident);c['rollout_role']='primary'
                self.checkpoint()
            if not c.get('full_report') and not c.get('full_error'):
                if not self.full_available():raise SearchBudgetExceeded('full_rollout_budget')
                await self.evaluate(c,'full')
            if c.get('full_report'):
                pending['primary_success_id']=ident
                break
        if audit_id and audit['status']=='pending':
            c=self.state['candidates'][audit_id]
            future_primary=max(0,self.cfg.max_expansions-round_number)
            future_audits=sum(round_number<r<=self.cfg.max_expansions for r in self.cfg.calibration_rounds)
            remaining=self.cfg.max_rollout_successes-self.state['full_fit_successes']
            audit['reserved_future_successes']=future_primary+future_audits
            # If the primary failed, this draw can still provide the round's
            # only successful candidate; it keeps its original audit identity.
            can_run=(not pending.get('primary_success_id') and self.full_available()) or remaining>future_primary+future_audits
            if c.get('full_report') or can_run:
                if audit_id not in pending['rollout_order']:
                    pending['rollout_order'].append(audit_id);c['rollout_role']='calibration'
                    self.checkpoint()
                await self.evaluate(c,'full')
                audit['status']='success' if c.get('full_report') else 'failed'
            else:audit['status']='reserved_for_later_rounds'
            self.checkpoint()

    @staticmethod
    def single_parameter_trial(candidate):
        changes = candidate.get('preflight_full_parameter_changes')
        return candidate.get('modification_direction') == 'parameters' and isinstance(changes, list) and len(changes) == 1

    def root_trial_stage(self):
        records = self.state['candidates']
        sign = self.tree.sign
        roots = sorted((records[i] for i in self.state['initial_selected']),
                       key=lambda c: (-sign*c['full_score'], str(c['id'])))
        measured = [c for c in records.values() if c.get('full_report')]
        trials = {r['id']: [c for c in measured if c.get('parent_id') == r['id']] for r in roots}
        missing = [r for r in roots if not trials[r['id']]]
        if missing:
            # Try untouched roots before retrying a failed branch.
            target = min(missing, key=lambda r: self.tree.nodes[r['id']]['expansion_count'])
            return dict(phase='initial_root_trial', eligible_parent_ids=[target['id']])

        if 'root_pruning' not in self.state:
            weakest = roots[-1]
            def tolerance(candidate):
                return candidate.get('sampling_practical_tolerance') or self.cfg.minimum_full_improvement
            def branch_best(root):
                return max([root, *trials[root['id']]], key=lambda c: sign*c['full_score'])
            best = branch_best(weakest)
            runner_up = min((branch_best(r) for r in roots[:-1]), key=lambda c: sign*c['full_score'])
            gap = sign*(runner_up['full_score'] - best['full_score'])
            improvement = sign*(best['full_score'] - weakest['full_score'])
            threshold = max(tolerance(weakest), tolerance(runner_up), tolerance(best))
            initial_gap = sign*(roots[-2]['full_score'] - weakest['full_score'])
            clear_last = initial_gap > max(tolerance(weakest), tolerance(roots[-2]))
            close = clear_last and gap > threshold and improvement <= tolerance(weakest)
            self.state['root_pruning'] = dict(closed_root_id=weakest['id'] if close else None,
                initial_gap=initial_gap, branch_gap=gap, threshold=threshold, improvement=improvement)
        closed = self.state['root_pruning']['closed_root_id']
        eligible = [n for n in self.tree.nodes.values() if n['branch_id'] != closed
                    and n['depth'] < self.cfg.max_depth and len(n['children']) < self.cfg.max_children_per_node]
        if eligible and self.state['full_fit_successes'] >= 9:
            best = max(eligible, key=lambda n: (sign*n['full_score'], n['id'] == self.tree.global_best_id))
            return dict(phase='direct_parameter_tuning', eligible_parent_ids=[best['id']])
        return dict(phase='open_search', eligible_parent_ids=[n['id'] for n in eligible])

    def prioritize_parameter_trial(self, pending, plan):
        if (pending.get('search_stage') or {}).get('phase') != 'single_parameter_check':
            return
        trial = next((i for i in plan['eligible_ids'] if self.single_parameter_trial(self.state['candidates'][i])), None)
        self.state['parameter_check']['candidate_id'] = trial
        if trial is not None:
            plan['primary_ids'] = [trial] + [i for i in plan['primary_ids'] if i != trial]

    async def expand(self):
        pending=self.state['pending']
        tuning = ((pending.get('search_stage') or {}).get('phase') == 'direct_parameter_tuning'
                  if pending else self.cfg.root_trials and self.state['full_fit_successes'] >= 9)
        if pending is None:
            round_number=len(self.state['expansions'])+1
            selection_mode=(('explore' if round_number in self.cfg.exploration_rounds else 'exploit')
                            if self.cfg.rollout_schedule in {'reserved','evidence_reserved'} else None)
            stage = None
            if self.cfg.root_trials:
                stage = self.root_trial_stage()
            elif self.cfg.staged_search:
                stage = self.tree.search_stage(self.cfg, self.state['candidates'], self.state['full_fit_successes'],
                    frozen_model_classes=self.state.get('parameter_focus_classes'))
            parent_id=self.tree.select_parent(self.cfg,selection_mode=selection_mode,
                eligible_parent_ids=stage['eligible_parent_ids'] if stage is not None else None)
            if parent_id is None:return False
            if isinstance(parent_id,dict):parent_id=parent_id['id']
            parent=self.state['candidates'][parent_id]
            if self.region_controller is not None and not tuning and len(self.ancestors(parent))<3:self.check_sampling_budget()
            self.state['search_attempts']+=1
            self.tree.record_expansion(parent_id)
            evidence=self.memory.generation_evidence(parent_id,ancestor_ids=self.ancestors(parent),query=parent,
                global_best_id=self.tree.global_best_id,
                stalled=len(self.state['expansions'])>=3 and all(not x.get('improved_global') for x in self.state['expansions'][-3:]))
            pending={'iteration':self.state['search_attempts'],'parent_id':parent_id,'evidence':evidence,'candidate_ids':[],
                     'best_before':self.tree.global_best_id,'started':True,
                     'round_number':round_number,'parent_selection':selection_mode or 'adaptive_uct'}
            pending['depth_direct']=len(self.ancestors(parent))+1>3
            if stage is not None:
                pending['search_stage']=stage
                evidence['search_stage']=stage
                if stage['phase']=='parameter_focus':
                    self.state.setdefault('parameter_focus_classes',list(stage['allowed_model_classes']))
            self.state['pending']=pending;self.checkpoint()
        parent=self.state['candidates'][pending['parent_id']]
        if not tuning and not pending.get('depth_direct') and self.region_controller is not None and not pending.get('region_binding') and 'direct_full' not in pending:
            self.check_sampling_budget()
        if not pending['candidate_ids'] and not pending.get('generation_error'):
            try:
                generated=await self.backend.generate(parent,pending['evidence'],pending['iteration'])
                expected = 1 if tuning or pending.get("depth_direct") else self.cfg.expansion_candidates
                if len(generated) != expected:
                    raise ValueError(f'Expansion requires {expected} candidates')
                if len({c['id'] for c in generated}) != expected:raise ValueError('Generated candidate IDs are not unique')
                for c in generated:c['parent_id']=parent['id'];c['branch_id']=parent['branch_id']
                pending['generated']=generated;pending['candidate_ids']=[c['id'] for c in generated]
                self.state.pop('inflight',None);self.checkpoint()
            except SearchBudgetExceeded:raise
            except Exception as exc:
                pending['generation_error']=f'{type(exc).__name__}: {exc}';self.state.pop('inflight',None);self.checkpoint()
        if pending.get('generation_error'):
            self.event('generation_failed',**pending);self.state['pending']=None;self.checkpoint();return True
        for c in pending.get('generated',[]):
            if c['id'] not in self.state['candidates']:self.add(c,checkpoint=False)
        candidates=[self.state['candidates'][i] for i in pending['candidate_ids']]
        if pending.get('depth_direct') and not tuning:
            child,=candidates
            pending.update(direct_full={'reason':'depth_above_three'},rollout_order=[])
            if child.get('status')!='failed' and not child.get('full_error'):
                if not child.get('full_report'):
                    try:
                        child['depth_direct']=True
                        await self.backend.prepare_sampling_candidate(child)
                        self.checkpoint()
                    except SearchBudgetExceeded:raise
                    except Exception as exc:
                        if infrastructure_failure(exc):raise
                        child['full_error']=str(exc)
                    if not child.get('full_error'):await self.evaluate(child,'full')
                if child.get('full_report'):pending['rollout_order']=[child['id']]
            self.finish_batch(pending,candidates)
            return True
        if tuning:
            child, = candidates
            if 'direct_full' not in pending:
                reason = 'invalid_parameter_proposal'
                run = False
                if child.get('implementation_mode') == 'inherited_parameters' and child.get('status') != 'failed':
                    try:
                        await self.backend.prepare_sampling_candidate(child)  # Constructor audit only.
                        key = self.backend.parameter_stage_key
                        certificate = {phase: key(child, phase) for phase in ('coreset', 'full_train')}
                        if self.single_parameter_trial(child) and all(certificate.values()):
                            duplicate = any(key(c, 'full_train') == certificate['full_train']
                                for c in self.state['candidates'].values() if c.get('full_report'))
                            if duplicate:
                                reason = 'already_measured_full_configuration'
                            else:
                                child['direct_full_certificate'] = certificate
                                reason, run = 'direct_parameter_tuning', True
                    except (ValueError, TypeError) as error:
                        child['preflight_error'] = str(error)
                pending.update(direct_full={'reason': reason}, rollout_order=[child['id']] if run else [])
                self.checkpoint()
            if pending['rollout_order']:
                child['rollout_role'] = 'primary'
                if not child.get('full_report') and not child.get('full_error'):
                    await self.evaluate(child, 'full')
                if child.get('full_report'):
                    pending['primary_success_id'] = child['id']
            self.finish_batch(pending, candidates)
            return True
        await self.prepare_region_round(pending,candidates)
        if 'direct_full' in pending:
            await self.evidence_rollouts(pending,candidates)
            self.finish_batch(pending,candidates)
            return True
        await self.score_expansion_coreset(pending,candidates)
        valid=[c for c in candidates if c.get('low_report')]
        if not valid:
            self.event('all_candidates_failed',iteration=pending['iteration'],candidate_ids=pending['candidate_ids'])
            if self.region_controller is not None:self.finish_batch(pending,candidates)
            else:
                self.archive_methods(candidates, None, f"expansion_{pending['iteration']:04d}")
                self.state['pending']=None;self.checkpoint()
            return True
        if self.region_controller is not None and hasattr(self.region_controller,'after_low'):
            await self.region_controller.after_low(self,pending,valid)
        if 'shortlist' not in pending:
            override=(self.region_controller.screening_override(self,pending,valid)
                if self.region_controller is not None and hasattr(self.region_controller,'screening_override') else None)
            if override is not None:
                pending['mcs']=deepcopy(override['mcs'])
                pending['shortlist']=deepcopy(override['shortlist'])
                selected_ids=pending['shortlist'].get('candidate_ids',[])
                if (not 1<=len(selected_ids)<=self.cfg.shortlist_cap or len(set(selected_ids))!=len(selected_ids)
                        or not set(selected_ids)<={c['id'] for c in valid}):
                    raise ValueError('Regional screening override returned an invalid shortlist')
                if not set(pending['mcs'].get('survivor_method_ids',[]))<={c['id'] for c in valid}:
                    raise ValueError('Regional screening override returned unknown MCS survivors')
                pending['screening_override']=True
            else:
                pending['mcs']=compare_candidates(self.backend,valid,self.cfg,(self.cfg.bootstrap_seed+pending['iteration'])%(2**32))
                pending['shortlist']=shortlist_candidates(valid,pending['mcs'],self.tree.direction,limit=self.cfg.shortlist_cap,minimum=1)
            for c in valid:
                c['mcs_survivor']=c['id'] in pending['mcs']['survivor_method_ids'];self.remember(c,checkpoint=False)
            self.checkpoint()
        selected=[self.state['candidates'][i] for i in pending['shortlist']['candidate_ids']]
        if self.cfg.rollout_schedule=='evidence_reserved':
            await self.evidence_rollouts(pending,valid)
            self.finish_batch(pending,candidates)
            return True
        if self.cfg.rollout_schedule=='reserved':
            await self.reserved_rollouts(pending,valid)
            self.finish_batch(pending,candidates)
            return True
        first=selected[0]
        if not first.get('full_report') and not first.get('full_error'):
            if not self.full_available():raise SearchBudgetExceeded('full_rollout_budget')
            await self.evaluate(first,'full')
        if len(selected)>1 and 'second_decision' not in pending:
            prob=None
            if not pending.get('screening_override') and getattr(self.backend,'supports_iid_mcs',True):
                try:
                    answer=paired_bootstrap_probability(self.backend.task,self.backend.y_core,
                       self.backend.load_prediction(first['low_report']),self.backend.load_prediction(selected[1]['low_report']),
                       n_bootstrap=self.cfg.bootstrap_replicates,seed=(self.cfg.bootstrap_seed+pending['iteration'])%(2**32))
                    prob=answer['backup_win_probability'] if isinstance(answer,dict) else answer
                except (ValueError,FloatingPointError):pass
            pending['second_decision']=should_rollout_second(parent.get('full_score'),first.get('full_score'),prob,
                pending['iteration'],self.cfg,direction=self.tree.direction,budget_available=self.full_available())
            self.checkpoint()
        stop_after=None
        if len(selected)>1 and pending.get('second_decision',{}).get('run_second'):
            if self.full_available() or selected[1].get('full_report'):
                try:await self.evaluate(selected[1],'full')
                except SearchBudgetExceeded as exc:stop_after=str(exc)
            else:pending['second_rollout_skipped']='budget_exhausted'
        self.finish_batch(pending,candidates,stop_reason=stop_after)
        if stop_after:raise SearchBudgetExceeded(stop_after)
        return True

    def _pause_at_stage(self,full_success_target):
        if (full_success_target is None or self.state['full_fit_successes']<full_success_target
                or self.state.get('pending') is not None or not self.full_available()
                or len(self.state['expansions'])>=self.cfg.max_expansions):
            return False
        pause={'target_full_successes':full_success_target,
               'actual_full_successes':self.state['full_fit_successes'],
               'completed_expansions':len(self.state['expansions']),
               'boundary':'completed_initialization_or_expansion',
               'feedback_committed':True}
        self.state.update(status='paused_stage',stop_reason='stage_full_success_target',stage_pause=pause)
        self.state.setdefault('stage_history',[]).append(deepcopy(pause))
        self.event('stage_paused',**pause)
        return True

    async def run(self,full_success_target=None):
        """Run to completion or a safe feedback-committed stage boundary.

        The target is orchestration state, not part of the frozen experiment
        configuration. An expansion may contain two full fits and overshoot a
        target by one; it is never split between its fits and feedback commit.
        """
        if (full_success_target is not None and
                (type(full_success_target) is not int or
                 not self.cfg.initial_full_count<=full_success_target<=self.cfg.max_rollout_successes)):
            raise ValueError('full_success_target must be an integer between initial and maximum full successes')
        self._start_clock()
        try:
            return await self._run_stage(full_success_target)
        finally:
            self._stop_clock()
            self.checkpoint()

    async def _run_stage(self,full_success_target):
        # An older runner may have stopped at the attempts limit. An unsealed
        # search can now use its remaining successful-rollout allowance.
        if (self.state['status']=='budget_exhausted' and self.state.get('stop_reason')=='full_rollout_budget'
                and self.full_available() and len(self.state['expansions'])<self.cfg.max_expansions):
            self.state['status']='running';self.state.pop('stop_reason',None)
            self.event('rollout_budget_resumed',basis='successful_full_evaluations')
        if self.state['status']=='completed' or (self.state['status']=='budget_exhausted' and not self.state.get('pending')):
            return json.loads((self.out/'search_summary.json').read_text())
        if self.state['status']=='paused_stage':
            if full_success_target is not None and self.state['full_fit_successes']>=full_success_target:
                return json.loads((self.out/'search_summary.json').read_text())
            self.state.pop('stop_reason',None);self.state.pop('stage_pause',None)
        self.state['status']='running';self.checkpoint()
        try:
            if not await self.initialize():
                self.archive_methods([self.state['candidates'][i] for i in self.state['initial_ids']],
                                     self.state.get('initial_skill_screening'), 'initialization_partial')
                return json.loads((self.out/'search_summary.json').read_text())
            if self.initial_feedback is not None and 'initial_bandit_feedback' not in self.state:
                self.state['initial_bandit_feedback']=self.initial_feedback(self)
                self.checkpoint()
            if not (self.out/'trajectories'/'initialization.json').exists():
                self.archive_methods([self.state['candidates'][i] for i in self.state['initial_ids']],
                                     self.state.get('initial_skill_screening'), 'initialization',
                                     self.state.get('initial_bandit_feedback'))
            if not self.cfg.enabled:
                self.state.update(status='completed',stop_reason='mcts_disabled_initialization_only')
                self.checkpoint()
                return json.loads((self.out/'search_summary.json').read_text())
            while len(self.state['expansions'])<self.cfg.max_expansions:
                if self._pause_at_stage(full_success_target):
                    return json.loads((self.out/'search_summary.json').read_text())
                self.check_wall()
                if self.state['search_attempts']>=self.cfg.max_search_attempts and not self.state['pending']:
                    raise SearchBudgetExceeded('search_attempt_budget')
                if not self.full_available():
                    # Resume can land between the final fit commit and batch admission.
                    self.finish_budget_stop('full_rollout_budget')
                    if len(self.state['expansions']) >= self.cfg.max_expansions:
                        break
                    raise SearchBudgetExceeded('full_rollout_budget')
                if not await self.expand():
                    self.state['stop_reason']='no_expandable_nodes';break
            self.state['status']='completed';self.state.setdefault('stop_reason','expansion_budget')
        except SearchBudgetExceeded as exc:
            self.finish_budget_stop(str(exc))
            self.state.update(status='budget_exhausted',stop_reason=str(exc))
        except BaseException:
            self.state['status']='interrupted';self.checkpoint();raise
        self.checkpoint();return json.loads((self.out/'search_summary.json').read_text())
