"""Isolated all-actions, full-information depth-two validation. No test split access."""
import argparse, asyncio, csv, fcntl, itertools, json, sqlite3, sys, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data_agent.mcts_search import MCTSSearch
from data_agent.mcts_policy import MCTSConfig
from data_agent.mcts_skill_workflow import skill_workflow_config,freeze_initial_reference
from data_agent.workflow import WorkflowBackend,source_view
from data_agent.region_bandit.runtime import run_region_task,SAMPLER
from data_agent.region_bandit.skill_sampling import mcts_sampling_policy
from data_agent.region_bandit.mcts_skills import RoundSkills
from data_agent.region_bandit.mcts_rounds import _binding
from data_agent.region_bandit.skill_store import SkillStore, sampling_path
from data_agent.region_bandit.regional_sampler import sample_from_action
from data_agent.region_bandit.schemas import read_json,write_json,stable_hash
from data_agent.region_bandit.mcts_skill_reward import reward_settings,INITIAL_TOP3_VERSION,checkpoint_reward,relative_reward
from data_agent.paired_sampler_experiment import freeze_json
from run_data_agent_workflow import settings_for
from run_region_bandit import _public_model
from scripts.run_region_skill_memory import create_model

ARMS=['baseline','cover','focus','gentle_cover']
def target_actions(actions, chosen, *, compare_baseline_region=False):
    primary = next(a for a in actions if a['action_id']==chosen)
    region = primary.get('region_id') if primary['type']!='baseline' else None
    if region is None and compare_baseline_region:
        regions = {a['region_id'] for a in actions if a.get('region_id')}
        def priority(rid):
            options = [a for a in actions if a.get('region_id') == rid]
            gap = options[0]['context']['region_summary'].get('coverage_gap') or 0
            return (-len(options), -gap, rid)
        region = min(regions, key=priority, default=None)
    return region, [a for a in actions if a['type']=='baseline' or
                    (region is not None and a.get('region_id')==region)]

def ranking(scores,sign):
    return sorted(scores,key=lambda i:(-sign*scores[i],i))
def agreement(scores,full,sign,slots):
    order=ranking(scores,sign);truth=ranking(full,sign)
    pairs=[(scores[a]-scores[b])*(full[a]-full[b]) for a,b in itertools.combinations(full,2) if full[a]!=full[b]]
    return dict(order=order,champion_retained=truth[0] in order[:slots],top1_correct=order[0]==truth[0],
                cutoff_tie=slots<len(order) and scores[order[slots-1]]==scores[order[slots]],
                pair_accuracy=sum(1 if v>0 else .5 if v==0 else 0 for v in pairs)/len(pairs) if pairs else None)
class ValidationConfig(MCTSConfig):
    @property
    def initial_full_count(self):
        return self.initial_candidates

    @property
    def full_train_methods(self):
        return 'all'


class ValidationSearch(MCTSSearch):
    """Validate the oracle admission contract when reopening this experiment."""
    def search_context(self):
        context = super().search_context()
        remaining = context['remaining']['successful_full_rollouts']
        context.update(rollout_schedule='validation_full_oracle',
            rollout_allocation=dict(remaining=remaining, scope='Evaluate every candidate at full scale. '
                'Expand the full-score top three initial methods and one full-score winner per later batch.'))
        return context

    def initial_ranking(self, initial):
        scores = {c['id']: c['full_score'] for c in initial}
        if len(scores) != 5 or any(value is None for value in scores.values()):
            raise ValueError('Oracle admission requires five observed full scores')
        return {'candidate_ids': ranking(scores, self.tree.sign)[:3]}


def measured_rewards(rankings, full, baseline, parent, parent_full, metric, sign, scale, settings):
    observed = dict(full)
    if parent is not None:
        observed[parent] = parent_full
        scale = abs(parent_full)
    result = {}
    for aid, scores in rankings.items():
        if set(scores) != set(full) or not observed.keys() <= baseline.keys():
            result[aid] = dict(reward=None, status='missing_paired_scores')
            continue
        core = dict(scores)
        if parent is not None:
            core[parent] = baseline[parent]
        measured = checkpoint_reward(core, observed, expected_ids=list(observed),
            direction='maximize' if sign == 1 else 'minimize', metric=metric,
            anchor_id=parent, reference_scale=scale, settings=settings)
        detail = relative_reward(core, observed, baseline, measured)
        result[aid] = detail
    return result


async def recover_child_implementation(candidate, backend):
    """Resume through the same parser and request journal as original generation."""
    if candidate.get('failure_stage') != 'generation':
        return
    from data_agent.multi_proxy_runtime import validate_code
    from data_agent.constructor_compatibility import normalize_constructor_code
    from data_agent.coreset import file_hash
    path = Path(candidate['code_path'])
    if path.exists() or candidate.get('fit_attempts') or candidate.get('sampling_contract_frozen'):
        raise ValueError('Failed candidate already has code or training history')
    request = read_json(path.parent/'implementation.request.json')
    response = await backend._request(request['prompt'], path.parent/'implementation.json')
    code, changes = normalize_constructor_code(response['code'])
    validate_code(code)
    if backend._contract(candidate) != candidate['contract_sha256']:
        raise ValueError('Frozen strategy changed during recovery')
    path.write_text(code)
    candidate.update(code_sha256=file_hash(path), status='generated', compatibility_changes=changes)
    for key in ('error', 'failure_stage'):
        candidate.pop(key, None)
    backend._save(candidate)


async def trajectory(row,root,storefile,settings,model,*,accumulate=False):
    config,search,memory,_,ms,pricing,subgroups=settings; cfg=skill_workflow_config(config)
    policy=dict(mcts_sampling_policy(),enable_gentle_cover=True);public=_public_model(ms)
    with SkillStore(storefile) as store:
        bundle=store.active
        procedural=dict(bundle=store.bundle(bundle),sampling_policy=policy,initial_admission_slots=3,
             history=store.decision_history(row['collection_dataset'],allow_local=False,limit=None))
    prep=await run_region_task(row['task_path'],root/'sampling',config=cfg,model=model,model_identity=public,
        pricing=pricing,memory_path=root/'ledger.sqlite',dataset_group_id=row['collection_dataset'],
        sampling_only=True,procedural=procedural,resume=(root/'sampling/state.json').exists())
    if prep.get('status')!='sampled':raise RuntimeError('bootstrap: '+str(prep.get('status')))
    source=source_view(prep['source'],prep['arm'],root/'selected_source')
    ctl=RoundSkills(root/'sampling',config=cfg,memory_path=storefile,source=row['collection_dataset'],task=row['task'],
        role='development',bundle_id=bundle,model_identity=public,sampling_policy=policy,
        reward_settings=reward_settings({'version':INITIAL_TOP3_VERSION}))
    initial_reference=freeze_initial_reference(root,prep['source'],relative=True)
    ctl.initial_reference_scale=initial_reference['reference_scale'];ctl._load()
    out=root/'search';backend=WorkflowBackend(source,0,out/'execution',model=model,pricing=pricing,max_repairs=search.repair_attempts,subgroup_rules=subgroups)
    sc=ValidationConfig.from_dict(dict(search.to_dict(),mcs_enabled=False,max_depth=2,expansion_candidates=2,max_rollout_successes=17,
        max_expansions=6,base_expansions=6,root_trials=False))
    backend.identity.update(generation_model=public,validation_protocol='full_oracle_all_actions_v2')
    eng=ValidationSearch(backend,out,sc,memory_options=memory,region_controller=ctl,resume=(out/'search_state.json').exists())
    state=eng.state; sign=eng.tree.sign
    state['status']='running'
    async def batch(methods,depth,iteration,parent):
        dest=root/'nodes'/str(iteration);dest.mkdir(parents=True,exist_ok=True)
        done=dest/'result.json'
        if done.exists():return read_json(done)['survivors']
        ids=[c['id'] for c in methods];pending=state['pending'];seed=cfg['region_bandit']['sampling_seeds'][0]
        if depth and not pending.get('region_binding'):
            for c in methods:
                if not c.get('sampling_prepared'):
                    if c.get('failure_stage')=='generation':
                        await recover_child_implementation(c,backend)
                        eng.checkpoint()
                    await backend.prepare_sampling_candidate(c);c['sampling_prepared']=True;eng.checkpoint()
            binding=await ctl.prepare(eng,pending,methods)
            for c in methods:backend.bind_coreset(c,binding)
            pending['region_binding']=binding;eng.checkpoint()
        if depth:
            folder=ctl._folder(pending);frozen=read_json(folder/'predecision.json');decision=read_json(folder/'decision.json')
        else:
            decision=read_json(ctl.root/'decision.json');orders={}
            for a in target_actions(decision['actions'],decision['chosen_action_id'],compare_baseline_region=accumulate)[1]:
                order,_=sample_from_action(a,ctl.reference,ctl.inputs['density'],ctl.inputs['hardness'],ctl.graph,ctl.inputs['groups'],
                    ctl.regions.get(a.get('region_id'),{}).get('mask'),seed=seed,config=SAMPLER,decision_seed=ctl.cfg['decision_seed'])
                orders[a['action_id']]=order.tolist()
            frozen=dict(actions=decision['actions'],sample_orders=orders,seed=seed,round_reference=ctl.reference.tolist())
        chosen=decision['chosen_action_id']
        target_region,acts=target_actions(frozen['actions'],chosen,compare_baseline_region=accumulate)
        cache={}; rankings={}; predictions={}; effects={}
        for c in methods:
            await eng.evaluate(c,'low')
        fitted = [c for c in methods if c.get('low_report')]
        chosen_scores={c['id']:c['low_score'] for c in fitted}
        for a in sorted(acts,key=lambda a:a['action_id']!=chosen):
            aid=a['action_id'];order=frozen['sample_orders'][aid];h=stable_hash(sorted(order))
            fixed=set(frozen['actions'][0].get('protected_q_indices', ()))
            if set(a.get('protected_q_indices', ())) != fixed:
                raise ValueError('Action belongs to a different protected sample')
            if h!=a['context']['action_summary']['cross_region_preview'][str(frozen['seed'])]['sample_hash']:raise ValueError('Preview mismatch')
            if len(order)!=len(ctl.reference) or len(set(order))!=len(order) or not set(a.get('protected_q_indices', ()))<=set(order):raise ValueError('Invalid protected action')
            reference=set(frozen['sample_orders']['baseline'])
            selected=set(order)
            region=next((a for a in acts if a.get('region_id')==target_region and a['type']!='baseline'),None)
            preview=a['context']['action_summary'].get('cross_region_preview',{}).get(str(frozen['seed']),{})
            counts=preview.get('regions',{}).get(target_region,{}) if region else {}
            effects[aid]=dict(chosen=aid==chosen,kind=a.get('variant') or a['type'],target_region=target_region,
                target_before=counts.get('before'),target_after=counts.get('after'),target_delta=counts.get('change'),
                added=len(selected-reference),removed=len(reference-selected),sample_size=len(selected),
                sample_hash=h,protected_count=len(fixed))
            if aid==chosen:
                actual = pending['region_binding'] if depth else _binding(backend.coreset_path,
                    backend.source_run/f'coreset_{backend.coreset_index}_indices.json')
                if set(ctl._parent_order(actual)) != selected:
                    raise ValueError('Chosen training sample differs from the action preview')
                cache[h]=(chosen_scores,{c['id']:c['low_report']['prediction_path'] for c in fitted})
            if h not in cache:
                b=ctl._materialize(eng,dest/'alternatives',frozen,aid)
                scores,reports=await ctl._probe_scores(eng,pending,fitted,b,'validation:'+str(iteration),allow_partial=True)
                cache[h]=(scores,{i:v['prediction_path'] for i,v in reports.items()})
            rankings[aid],predictions[aid]=cache[h]
        kinds={a['action_id']:a.get('variant') or a['type'] for a in acts}
        freeze_json(dest/'screening.json',dict(candidate_ids=ids,parent=parent,depth=depth,chosen=chosen,scores=rankings,predictions=predictions,
            rankings={a:ranking(s,sign) if set(s)==set(ids) else None for a,s in rankings.items()},
            missing_methods={a:sorted(set(ids)-set(s)) for a,s in rankings.items()},
            action_effects=effects,comparison_region=target_region,action_types=kinds,
            unavailable_types=sorted(set(ARMS)-set(kinds.values())),tie_break='candidate_id'))
        for c in methods:
            await eng.evaluate(c,'full')
        missing_full = [c for c in methods if not c.get('full_report')]
        if missing_full:
            write_json(dest/'incomplete_full.json',dict(parent=parent, depth=depth, candidate_ids=ids,
                scores={c['id']:c.get('full_score') for c in methods},
                missing={c['id']:c.get('full_error') or c.get('low_error') for c in missing_full}))
            raise ValueError('Oracle admission requires the full bank; missing: '+
                             ', '.join(c['id'] for c in missing_full))
        full={c['id']:c['full_score'] for c in methods}; slots=1 if depth else 3
        winners=ranking(full,sign)[:slots]
        metrics={a:agreement(s,full,sign,slots) if set(s)==set(full) else None for a,s in rankings.items()}
        best=max((v['champion_retained'],v['pair_accuracy'] or 0) for v in metrics.values() if v is not None)
        optimal=[a for a,v in metrics.items() if v is not None and (v['champion_retained'],v['pair_accuracy'] or 0)==best]
        comparable = len(cache)>1 and all(v is not None and v['pair_accuracy'] is not None for v in metrics.values())
        reference_scores = (frozen.get('baseline_scores', {}) if depth else
            {c['id']:initial_reference['baseline_scores'][str(c['source_method_id'])] for c in methods})
        rewards = measured_rewards(rankings,full,reference_scores,parent,
            state['candidates'][parent]['full_score'] if depth else None,
            backend.task['metric'],sign,ctl.initial_reference_scale,ctl.reward_settings)
        for c in methods:
            c['admitted']=c['id'] in winners
            if c['admitted'] and c['id'] not in eng.tree.nodes:eng.tree.retain_candidate(c['id'])
            eng.remember(c)
        eng.archive_methods(methods, {'combined': {'scores': chosen_scores}}, f'validation_{iteration:04d}')
        chosen_context = next(a['context'] for a in acts if a['action_id'] == chosen)
        write_json(dest/'skill_trajectory.json', dict(
            path=sampling_path(chosen_context, methods, actions=frozen['actions'],
                regions=frozen.get('node_regions', ()) if depth else list(ctl.regions.values())),
            chosen=chosen,decision_id=decision.get('decision_id'),bundle_id=bundle,
            skills_used=decision['skill_execution'].get('skills_used',[]),reason=decision['skill_execution'].get('reason'),
            checks=decision['skill_execution'].get('checks', []),
            comparison_region=target_region,comparison_scope=('preselected_control_region' if chosen=='baseline' and target_region else 'api_region_and_baseline'),
            by_action={kind: next(({**effects[aid], 'action_id':aid, 'coreset_scores':rankings[aid],
                'agreement':metrics[aid], 'reward':{k:rewards[aid].get(k) for k in ('reward','status','reward_basis','comparable_pair_count')}} for aid in effects if kinds[aid]==kind),None) for kind in ARMS},
            validation=metrics[chosen], observed_full_ids=ids, reward=rewards[chosen]['reward'],
            reward_status=rewards[chosen]['status'], policy_updated=False,
            decision_status=decision['skill_execution'].get('decision_status'),
            reward_reference=('parent_coreset_scores' if depth else 'five_proxy_scores_on_V_core')))
        if accumulate:
            primary = next(a for a in acts if a['action_id'] == chosen)
            path = read_json(dest/'skill_trajectory.json')['path']
            fact = dict(source=row['collection_dataset'], task=row['task'], role='development',
                trajectory_id=str(root), round_number=iteration, parent_id=parent,
                search_stage='mcts_expansion' if depth else 'initialization',
                bundle_hash=bundle, bank_hash=(frozen['bank_hash'] if depth else decision['identity']['method_bank_hash']),
                seed=seed, sample_hash=effects[chosen]['sample_hash'], decision_id=decision['decision_id'],
                skill_execution=decision['skill_execution'], reward=rewards[chosen], sampling_path=path,
                action={k:primary[k] for k in ('action_id','type','region_id') if k in primary},
                decision_context=chosen_context.get('region_summary',{}),
                cross_region_preview=chosen_context['action_summary']['cross_region_preview'],
                validation_actions={kinds[aid]:dict(reward=rewards[aid]['reward'],
                    pair_accuracy=(metrics[aid] or {}).get('pair_accuracy'), added=effects[aid]['added'])
                    for aid in effects})
            freeze_json(dest/'skill_fact.json',fact)
        if depth:state['expansions'].append(dict(iteration=iteration,parent_id=parent,candidate_ids=ids,region_binding=pending['region_binding'],winner_id=winners[0],improved_global=eng.tree.global_best_id!=pending['best_before']))
        else:state.update(initial_selected=winners,initialization_complete=True)
        staged=dest/'prepared_result.json'
        write_json(staged,dict(depth=depth,parent=parent,survivors=winners,full=full,full_rank=ranking(full,sign),
            full_predictions={c['id']:c['full_report']['prediction_path'] for c in methods},chosen=chosen,
            metrics=metrics,comparison_region=target_region,action_effects=effects,best_actions=optimal,decision_status=decision['skill_execution'].get('decision_status'),skill_matches_best=(chosen in optimal if comparable else None),distinct_samples=len(cache)))
        state['pending']=None;eng.checkpoint()
        staged.replace(done)
        return winners
    queue=[(None,0)];iteration=0
    while queue:
        parent,depth=queue.pop(0); done=root/'nodes'/str(iteration)/'result.json'
        staged=done.with_name('prepared_result.json')
        if staged.exists() and state['pending'] is None and not done.exists():
            committed = (state['initial_selected'] if depth==0 else
                [entry['winner_id'] for entry in state['expansions'] if entry['iteration']==iteration and entry['parent_id']==parent])
            if committed != read_json(staged)['survivors']:
                raise ValueError('Prepared validation result has no matching committed winners')
            staged.replace(done)
        if done.exists():
            record=read_json(done)
            if record['parent']!=parent or record['depth']!=depth:
                raise ValueError('Validation result belongs to a different node')
            w=record['survivors']
        else:
            p=state['pending'] or dict(parent_id=parent,iteration=iteration,round_number=iteration,candidate_ids=[],best_before=eng.tree.global_best_id)
            if (p['iteration'],p['parent_id'])!=(iteration,parent):
                raise ValueError('Pending validation batch belongs to a different node')
            state['pending']=p
            if depth:
                state['search_attempts']=iteration
            eng.checkpoint()
            if not p['candidate_ids']:
                if depth:
                    par=state['candidates'][parent];ev=eng.memory.generation_evidence(parent,ancestor_ids=eng.ancestors(par),query=par,global_best_id=eng.tree.global_best_id,stalled=False)
                    generation_input=root/'nodes'/str(iteration)/'generation_input.json'
                    if generation_input.exists():
                        frozen_input=read_json(generation_input)
                        if frozen_input['parent_id']!=parent or frozen_input['parent_code_sha256']!=par['code_sha256']:
                            raise ValueError('Frozen expansion parent changed')
                    else:
                        frozen_input=dict(parent_id=parent,parent_code_sha256=par['code_sha256'],
                            evidence=ev,context=eng.search_context())
                        write_json(generation_input,frozen_input)
                    backend.search_context_provider=lambda: frozen_input['context']
                    try:
                        bank=await backend.generate(par,frozen_input['evidence'],iteration,resume_generation=True)
                    finally:
                        backend.search_context_provider=eng.search_context
                    state.pop('inflight',None)
                    p.pop('generation_error',None)
                    for c in bank:c.update(parent_id=parent,branch_id=par['branch_id'])
                else:
                    bank=backend.initial_candidates()
                    for c in bank:c.update(parent_id=None,branch_id=c['id'])
                    state['initial_ids']=[c['id'] for c in bank]
                if len(bank)!=(2 if depth else 5):raise ValueError('Unexpected candidate count')
                for c in bank:eng.add(c,checkpoint=False)
                p['candidate_ids']=[c['id'] for c in bank];eng.checkpoint()
            w=await batch([state['candidates'][i] for i in p['candidate_ids']],depth,iteration,parent)
        if accumulate:
            with SkillStore(storefile) as store:
                store.record_fact(read_json(done.parent/'skill_fact.json'))
        if depth<2:queue.extend((i,depth+1) for i in w)
        iteration+=1
    assert iteration==7 and state['full_fit_successes']==17
    state.update(status='completed',stop_reason='validation_depth_complete');eng.checkpoint()
    write_json(root/'completed.json',dict(nodes=7,full=17,test_access=False))

async def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--env-file',type=Path,required=True);mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--skill-db',type=Path);mode.add_argument('--cold-sequential',action='store_true')
    ap.add_argument('--task',action='append',help='Run only the named suitable30 task (repeatable)')
    ap.add_argument('--skip-failed', action='store_true', help='Skip tasks with an existing error.txt on resume')
    args=ap.parse_args()
    settings=settings_for(ROOT/'config/workflow_skill_mcts.yaml');rows=read_json(ROOT/'datasets/suitable30/tasks.json')
    for row in rows:
        row['task_path'] = str((ROOT / row['task_path']).resolve())
    if args.task:
        unknown=set(args.task)-{r['task'] for r in rows}
        if unknown:ap.error('Unknown task: '+', '.join(sorted(unknown)))
        rows=[r for r in rows if r['task'] in args.task]
    base=args.output.resolve();base.mkdir(parents=True,exist_ok=True)
    with (base/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        snapshot=base/'skills.sqlite'
        if args.cold_sequential:
            manifest=base/'sequential_manifest.json'
            if snapshot.exists() and not manifest.exists():
                raise ValueError('Cold run requires a fresh output directory')
            freeze_json(manifest,dict(protocol='cold-sequential-v1',tasks=[r['task'] for r in rows]))
            with SkillStore(snapshot) as store:
                store.register_sources([r['collection_dataset'] for r in rows],'development')
        elif not snapshot.exists():
            with sqlite3.connect(args.skill_db.resolve().as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(snapshot.with_suffix('.tmp')) as dst:src.backup(dst)
            snapshot.with_suffix('.tmp').rename(snapshot)
        model=await create_model(settings[4],args.env_file)
        for r in rows:
            root=base/r['task'];root.mkdir(exist_ok=True)
            if (root/'completed.json').exists():continue
            if args.skip_failed and (root/'error.txt').exists():
                print('SKIP FAILED', r['task'], flush=True)
                continue
            print('START',r['task'],flush=True)
            try:await trajectory(r,root,snapshot,settings,model,accumulate=args.cold_sequential)
            except Exception:
                (root/'error.txt').write_text(traceback.format_exc());print('FAILED',r['task'],flush=True)
            else:
                (root/'error.txt').unlink(missing_ok=True);print('DONE',r['task'],flush=True)
if __name__=='__main__':asyncio.run(main())
