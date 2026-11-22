"""Agent-only action and strength selection; no score fusion or random override."""
import inspect
import json
import math
from ..execution_budget import TaskDeadlineExceeded
from .advisor_repair import with_one_output_repair

AGENT_POLICY_VERSION = 'region-agent-choice-v1'


@with_one_output_repair
async def choose_action_with_api(contexts, cases, request_fn, *, config=None, skills=None):
    contexts=list(contexts);cases=list(cases);skills=list(skills or [])
    config = config or {}
    resample = config.get('sampling_mode', 'local_edit') == 'resample'
    delta_reward = config.get('reward_objective', 'absolute') == 'paired_delta'
    ids=[c['action_id'] for c in contexts]
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError('Distinct feasible action IDs are required')
    baseline=[c['action_id'] for c in contexts if c['action_type']=='baseline']
    if len(baseline)!=1:
        raise ValueError('Exactly one unchanged baseline is required')
    shared_keys=('problem_type','metric','metric_params','task_summary','method_summary','budget_summary')
    shared={key:contexts[0][key] for key in shared_keys if key in contexts[0]}
    if any(any(c.get(k)!=v for k,v in shared.items()) for c in contexts):
        raise ValueError('Agent action contexts must share the same task and bank')
    regions={}
    actions=[]
    for c in contexts:
        action=dict(c['action_summary'])
        # Exact quota tables stay on disk; the prompt receives counts and donor risk.
        action.pop('quota_table',None)
        actions.append(action)
        region=c.get('region_summary',{})
        if region.get('region_id'):
            regions[region['region_id']]=region
    request=dict(shared_context=shared,regions=regions,actions=actions,
        historical_cases=[{**{k:c.get(k) for k in ('record_id','dataset_group_id','distance','action_type')},
            'feedback':{k:c.get('feedback',{}).get(k) for k in ('reward','reward_objective','status','paired_comparison','observed_method_count')},
            'region_summary':c.get('context',{}).get('region_summary',{}),
            'action_summary':c.get('context',{}).get('action_summary',{})} for c in cases],
        experience_skills=skills,
        instruction=(
        'You alone choose the sampling action and its strength from the supplied feasible actions. '
        'There is no epsilon exploration, no linear bandit score fusion, and no random override of your choice. '
        'Use current training-side evidence and relevant historical cases/skills. Never use current full results or test data. '
        'The recorded reward remains absolute full-score-gap-weighted pairwise ranking accuracy of the five methods. '
        'Assess whether changing the original coreset is justified; a high absolute past reward alone is not proof of improvement. '
        'A tied historical paired outcome means no observed benefit, not success. Negative evidence must be considered; '
        'case hypotheses are not established rules. With limited evidence you may keep baseline or choose a cautious experiment; '
        'explain your evidence and uncertainty. Do not invent benefits or choose merely because a region is large or retention is high. '
        'All original group/class counts stay fixed. Maximum total row replacement is declared per action; '
        'max_donor_remove_fraction limits deletions from EACH group\'s ORIGINAL outside-region coreset rows, with integer floor. '
        'repeat_effects and donor_protection show feasible maximum edits, protected small groups and the most affected donor groups. '
        'Consider these local deletion fractions as well as global retention. Unmet region targets cannot override these limits. '
        'Exchange quotas are allocated proportionally to original outside-region counts within capacities. '
        'Within each group, sequentially remove the outside row with lowest density/hardness plus unique coverage retention score, '
        'then add an unselected inside row by density/hardness/marginal-coverage probability, updating coverage each time. '
        'Choose adjustment_fraction in (0,1] of this action\'s feasible maximum increment; '
        'the program rounds to the nearest integer, at least one row for a nonzero feasible regional action. '
        'For baseline, adjustment_fraction MUST be 0. You may choose any feasible fraction; '
        'the program validates and materializes it without replacing your decision. '
        'Return only JSON {"chosen_action_id": string, "adjustment_fraction": number, '
        '"rationale": string, "risk": string, "evidence_ids": [string]}. '
        'Cite only supplied record_id values; use [] when there is no relevant historical evidence. '
        'Keep rationale and risk concise, linking the chosen region/strength to actual facts.'))
    if delta_reward:
        request['instruction'] = request['instruction'].replace(
            'The recorded reward remains absolute full-score-gap-weighted pairwise ranking accuracy of the five methods. ',
            'Reward is the SIGNED difference: gap-weighted pairwise ranking accuracy of the adjusted coreset '
            'MINUS accuracy of the ORIGINAL coreset for the same task, seed, five methods and full reference. '
            'Reward lies in [-1,1]: positive is improvement, negative is harm, zero is no ranking benefit. '
            'The unchanged baseline has zero reward when informative. Absolute accuracy is recorded separately. ')
    if resample:
        start = request['instruction'].index('All original group/class counts stay fixed.')
        end = request['instruction'].index('Return only JSON')
        request['instruction'] = request['instruction'][:start] + (
            'Regional actions rebuild the WHOLE coreset from an empty selection using frozen integer quotas. '
            'Original row identities are not protected: there is NO 15% row replacement cap or 25% donor removal cap. '
            'Both inside and outside the region are sampled without replacement by density, hardness and '
            'dynamic marginal coverage probabilities. Coverage is shared inside/outside each original group. '
            'The supplied group_bounds, class_bounds and group_quota_move_budget remain mandatory. '
            'The group quota movement budget limits changes in group counts, NOT replacement of individual rows. '
            'Choose one feasible cover/focus action exactly as planned; adjustment_fraction MUST be 1 for '
            'regional actions and 0 for baseline. Baseline reuses the original coreset exactly. '
            'Consider both potential regional benefit and the uncertainty from redrawing the entire sample. '
        ) + request['instruction'][end:]
    result=dict(scores={i:0. for i in ids},api_available=0,request=request,response=None,error=None,calls=0,
                decision_mode='agent')
    if len(ids)==1:
        result.update(chosen_action_id=baseline[0],adjustment_fraction=0.,rationale='Only unchanged baseline is feasible',
                      risk='No feasible protected regional edit',evidence_ids=[],skipped_reason='baseline_only')
        return result
    try:
        if request_fn is None:
            raise ValueError('Agent decision requires a real request function')
        result['calls']=1
        response=request_fn(json.dumps(request,ensure_ascii=False,sort_keys=True,allow_nan=False))
        if inspect.isawaitable(response):response=await response
        result['response']=response
        if isinstance(response,str):response=json.loads(response)
        if not isinstance(response,dict) or response.get('chosen_action_id') not in ids:
            raise ValueError('Agent must choose a supplied feasible action ID')
        fraction=response.get('adjustment_fraction')
        if type(fraction) not in (float,int) or not math.isfinite(fraction):
            raise ValueError('Agent adjustment_fraction must be a finite number')
        if (response['chosen_action_id'] in baseline and fraction!=0) or (response['chosen_action_id'] not in baseline and not 0<fraction<=1):
            raise ValueError('Invalid adjustment fraction for chosen action')
        if resample and response['chosen_action_id'] not in baseline and fraction != 1:
            raise ValueError('Whole-coreset resampling requires the frozen full quota plan')
        for key in ('rationale','risk'):
            if not isinstance(response.get(key),str) or not response[key].strip():
                raise ValueError('Agent rationale and risk must be nonempty strings')
        evidence={c['record_id'] for c in [*cases,*skills]}
        citations=response.get('evidence_ids')
        if not isinstance(citations,list) or any(not isinstance(x,str) or x not in evidence for x in citations):
            raise ValueError('Agent cited unavailable evidence')
        result.update(api_available=1,**{k:response[k] for k in ('chosen_action_id','adjustment_fraction','rationale','risk','evidence_ids')})
    except TaskDeadlineExceeded:
        raise
    except Exception as error:
        if getattr(error,'budget_exhausted',False) or type(error).__name__ == 'SearchBudgetExceeded':raise
        result['error']=type(error).__name__+': '+str(error)
        # Preserve malformed values for inspection without nonfinite JSON writes.
        try:json.dumps(result['response'],allow_nan=False)
        except (ValueError,TypeError):result['response']=repr(result['response'])
    return result


def materialize_agent_decision(snapshot, actions, advisor, seed, *, mode='develop_online'):
    if not advisor.get('api_available') and advisor.get('skipped_reason')!='baseline_only':
        raise ValueError('No validated Agent decision; fallback is forbidden')
    chosen=advisor['chosen_action_id']
    if sum(a['action_id']==chosen for a in actions)!=1:
        raise ValueError('Agent action missing or duplicated')
    return dict(policy_version=AGENT_POLICY_VERSION,policy_snapshot_id=snapshot['snapshot_id'],
        schema_id=snapshot['schema_id'],actions=[{**a,'api_score':0.,
            'selection_probability':None,'score_used_for_selection':False} for a in actions],
        chosen_action=chosen,chosen_action_id=chosen,decision_seed=int(seed),epsilon=0.,mode=mode,
        selection_probability=None,decision_source='agent',
        agent_decision={k:advisor[k] for k in ('chosen_action_id','adjustment_fraction','rationale','risk','evidence_ids')},
        numerical_snapshot_role='historical accounting only; no scores enter Agent or action choice')
