"""Run three-proposal MCTS on one fixed coreset with real DeepSeek generation.

Use --source-run to attach to a prior five-method run, or --task to bootstrap
P/Q feedback and coresets first. No test set is opened by this command.
"""
import argparse,asyncio,fcntl,json,sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import yaml
from dotenv import load_dotenv
from data_agent.mcts_policy import MCTSConfig
from data_agent.mcts_backend import MCTSBackend
from data_agent.mcts_search import MCTSSearch
ROOT=Path(__file__).resolve().parent

def validate_bootstrap_resume(task, cfg, identity, manifest):
 """Compare the frozen effective bootstrap, including preparation overrides."""
 from data_agent.prepared_protocol import preparation_config
 saved=manifest.get('configuration')
 saved_model=manifest.get('llm_model')
 if not isinstance(saved,dict) or not isinstance(saved_model,dict):
  raise ValueError('Bootstrap lacks frozen initialization configuration/model identity; use a new output directory')
 task_definition=json.loads(task.read_text())
 from data_agent.guarded_proxy_feedback import effective_config
 expected=asdict(preparation_config(task_definition,effective_config(deepcopy(cfg))))
 changed=sorted(key for key in set(saved)|set(expected) if saved.get(key)!=expected.get(key))
 if changed:
  raise ValueError('Bootstrap initialization configuration changed: '+', '.join(changed)+'; use a new output directory')
 model_keys=('model_name','base_url','temperature','max_tokens','timeout')
 changed=[key for key in model_keys if saved_model.get(key)!=identity.get(key)]
 if changed:
  raise ValueError('Bootstrap generation model changed: '+', '.join(changed)+'; use a new output directory')

async def bootstrap_with_repairs(task, source, cfg, model, identity, pricing, resume=False):
 from data_agent.method_repair import (run_with_repairs, assert_interrupted_bootstrap_inactive,
                                      interrupted_bootstrap_manifest, resolve_bootstrap_attempt)
 from data_agent.execution_budget import check_deadline
 source=Path(source).resolve();source.parent.mkdir(parents=True,exist_ok=True)
 # A separate lock is needed even when this helper is called outside either CLI.
 # Keep it beside the source so a new bootstrap directory stays absent until run_task.
 with (source.parent/('.'+source.name+'.bootstrap.lock')).open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError as error:raise ValueError('Another bootstrap owner is still active') from error
  state_path=source/'attempt_state.json'
  state=json.loads(state_path.read_text()) if state_path.exists() else {}
  if resume and state:
   assert_interrupted_bootstrap_inactive(source)
   effective=resolve_bootstrap_attempt(source)
  else:effective=Path(state.get('effective_run',source))
  manifest=effective/'manifest.json';interrupted=False
  if manifest.exists():
   old=json.loads(manifest.read_text())
   if not resume:raise ValueError('Existing bootstrap requires --resume')
   if Path(old['task_path']).resolve()!=task.resolve():raise ValueError('Bootstrap task changed')
   if old.get('status')=='success':
    validate_bootstrap_resume(task,cfg,identity,old)
    return effective
   if old.get('status') in ('running','interrupted'):
    assert_interrupted_bootstrap_inactive(effective)
    old=interrupted_bootstrap_manifest(effective)
    validate_bootstrap_resume(task,cfg,identity,old)
    interrupted=True
   elif old.get('status')!='failed':raise ValueError('Unsupported bootstrap state: '+str(old.get('status')))
  check_deadline()
  options={'resume_failed':resume}
  if interrupted:options['resume_interrupted']=True
  await run_with_repairs(task.resolve(),source,cfg,model,identity,pricing,**options)
  state=json.loads(state_path.read_text())
  return Path(state['effective_run'])

def settings_for(path):
 settings=yaml.safe_load(path.read_text())
 unknown=set(settings)-{'search','memory','model','initialization_config','subgroups'}
 if unknown:raise ValueError('Unknown MCTS config sections: '+str(sorted(unknown)))
 cfg=MCTSConfig.from_dict(settings.get('search'))
 memory=settings.get('memory',{})
 if set(memory)-{'enabled','default_count','max_count','token_budget'}:raise ValueError('Unknown memory config')
 initial=Path(settings.get('initialization_config','config/coverage_regression30.yaml'))
 if not initial.is_absolute():initial=ROOT/initial
 source_settings=yaml.safe_load(initial.read_text())
 model_settings=settings.get('model',source_settings['model'])
 return settings,cfg,memory,source_settings,model_settings

async def run(args):
 args.output=args.output.resolve()
 args.output.mkdir(parents=True,exist_ok=True)
 with (args.output/'.search.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError as exc:raise ValueError('Another search or final evaluation is using this output directory') from exc
  if (args.output/'final_test/selection_manifest.json').exists():
   raise ValueError('Final-test winner is already frozen; use its saved results and do not resume this search')
  if (args.output/'search_state.json').exists() and not args.resume:
   raise ValueError('Search already exists; use --resume with unchanged inputs/configuration')
  return await run_locked(args)

async def run_locked(args):
 settings,cfg,memory,source_settings,model_settings=settings_for(args.config)
 args.output=args.output.resolve()
 model=None
 try:
  if args.dry_run and args.task:
   print(json.dumps({'task':str(args.task.resolve()),'output':str(args.output),'coreset_index':args.coreset_index,
      'initialization':'Fresh P/Q feedback, five source methods, one fixed coverage coreset; no random baseline or bootstrap validation fits. Bootstrap cost separate from MCTS budget.',
      'full_train_methods':cfg.full_train_methods,'initial_full_method_count':cfg.initial_full_count,
      'search':cfg.to_dict(),'memory':memory,'test_access':False,'real_api_calls':0},ensure_ascii=False,indent=2));return 0
  if not args.dry_run:
   load_dotenv(args.env_file)
   from internagent.mas.models.openai_model import OpenAIModel
   model=OpenAIModel(**model_settings)
  source=args.source_run
  if source is None:
   from data_agent.multi_proxy import MultiProxyConfig
   values=dict(source_settings['workflow']);values['similarity_lambda']=values.pop('lambda',1.)
   values.setdefault('parameter_feedback',True)
   values.update(n_strategies=5,full_train_enabled=True,defer_full_train=True,mcts_bootstrap_only=True,
                 proxy_selection_scheme='marginal_coverage_only',n_coresets=1)
   values['stage1_selection']={**values.get('stage1_selection',{}),'enabled':False,'mcts_admission':False,'coreset_weights':[1.0]}
   bootcfg=MultiProxyConfig(**values);bootcfg.validate()
   source=args.output/'bootstrap'
   identity={k:v for k,v in model_settings.items() if k not in {'api_key','default_headers'}}
   source=await bootstrap_with_repairs(args.task,source,bootcfg,model,identity,
                                      source_settings.get('pricing'),resume=args.resume)
  backend=MCTSBackend(source,args.coreset_index,args.output/'execution',model=model,
                      pricing=source_settings.get('pricing'),max_repairs=cfg.repair_attempts,
                      subgroup_rules=settings.get('subgroups'))
  backend.identity['generation_model']={key:model_settings.get(key) for key in
     ('model_name','base_url','temperature','max_tokens','timeout')}
  if args.dry_run:
   initial=backend.initial_candidates()
   print(json.dumps({'source':backend.metadata,'search':cfg.to_dict(),'memory':memory,'initial_candidates':len(initial),
                     'full_train_methods':cfg.full_train_methods,'initial_full_method_count':cfg.initial_full_count,
                     'test_access':False,'real_api_calls':0},ensure_ascii=False,indent=2));return 0
  engine=MCTSSearch(backend,args.output,cfg,resume=args.resume,memory_options=memory)
  result=await engine.run();print(json.dumps(result,ensure_ascii=False,indent=2))
  return 0 if result['status'] in {'completed','budget_exhausted'} and result['initialization_complete'] else 1
 finally:
  if model is not None and model.client is not None:await model.client.close()

def main():
 p=argparse.ArgumentParser(description=__doc__);source=p.add_mutually_exclusive_group(required=True)
 source.add_argument('--source-run',type=Path,help='Five-method source/arm directory, no historical scores reused')
 source.add_argument('--task',type=Path,help='Frozen task.json; run initial P/Q feedback before MCTS')
 p.add_argument('--coreset-index',type=int,default=0,help='One fixed source coreset: original paired source indices0..2 coverage,3..5 random; arm directories use0')
 p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'config/mcts.yaml')
 p.add_argument('--env-file',type=Path,default=Path('/home/cmcui/InternAgent/.env'));p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
 args=p.parse_args()
 try:return asyncio.run(run(args))
 except KeyboardInterrupt:return 130
 except Exception as exc:print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 1
if __name__=='__main__':sys.exit(main())
