"""Run the sequential suitable30 main workflow with shared skill experience."""
import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import traceback

from run_data_agent_workflow import settings_for
from run_region_bandit import _public_model
from scripts.run_region_skill_memory import create_model
from data_agent.mcts_skill_workflow import run_skill_task

ROOT = Path(__file__).resolve().parent


async def run(args):
    config, search, memory, _, model_settings, pricing, subgroups = settings_for(ROOT/'config/main.yaml')
    rows = json.loads((ROOT/'datasets/suitable30/tasks.json').read_text())
    if args.task:
        if args.task not in {row['task'] for row in rows}:
            raise ValueError('Unknown task: ' + args.task)
        rows = [row for row in rows if row['task'] == args.task]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model = await create_model(model_settings, args.env_file)
        try:
            for row in rows:
                row['task_path'] = str(ROOT/row['task_path'])
                folder = output/row['task']
                if (folder/'completed.json').exists():
                    continue
                folder.mkdir(exist_ok=True)
                print('START', row['task'], flush=True)
                try:
                    result, _, _ = await run_skill_task(row, folder, config=config,
                        search=search, memory=memory, model=model,
                        model_identity=_public_model(model_settings), pricing=pricing,
                        subgroups=subgroups, store_path=output/'skills.sqlite',
                        role='development', sampling={'enable_gentle_cover': True})
                    if not result.get('initialization_complete') or result.get('status') not in ('completed', 'budget_exhausted'):
                        raise RuntimeError('Incomplete search: ' + str(result.get('status')))
                    (folder/'completed.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
                    (folder/'error.txt').unlink(missing_ok=True)
                    print('DONE', row['task'], flush=True)
                except Exception:
                    (folder/'error.txt').write_text(traceback.format_exc())
                    print('FAILED', row['task'], flush=True)
        finally:
            await model.client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/main')
    parser.add_argument('--env-file', type=Path, default=ROOT/'.env')
    parser.add_argument('--task', help='Run one task from the suitable30 manifest')
    asyncio.run(run(parser.parse_args()))
