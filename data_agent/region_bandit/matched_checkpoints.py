"""Select historical decisions using eligibility and frozen pre-action evidence."""
from pathlib import Path

from .mcts_skill_reward import INITIAL_TOP3_VERSION
from .procedural_skills import activated
from .schemas import read_json


def match_score(skill_id, bundles, frozen):
    actions = frozen['actions']
    contexts = [a['context'] for a in actions]
    applicable = {i for bundle in bundles for skill in activated(bundle, contexts)
                  if skill['id'] == skill_id for i in skill['applicable_action_ids']}
    scores = []
    for action in actions:
        if action['action_id'] not in applicable or action['type'] not in ('cover', 'focus'):
            continue
        ctx = action['context']
        region = ctx.get('region_summary') or {}
        preview = ctx['action_summary']['cross_region_preview'][str(frozen['seed'])]
        if not preview.get('sample_replaced_count'):
            continue
        if skill_id == 'coverage_repair' and action['type'] == 'cover':
            scores.append(region.get('coverage_gap') or 0)
        elif skill_id == 'difference_preservation' and action['type'] == 'focus':
            scores.append(region.get('single_pair_score') or min(
                abs(region.get('discover_score') or 0), abs(region.get('check_score') or 0)))
        elif skill_id == 'intervention_check':
            scores.append(preview['sample_replaced_fraction'])
    return max(scores, default=None)


def matched_cases(roots, edit, store, seeds):
    bundles = [store.bundle(edit[key]) for key in ('parent_hash', 'candidate_hash')]
    pool = {}
    for root in map(Path, roots):
        configs = [root/'skill_workflow.json', *sorted(root.glob('tasks/*/skill_workflow.json')),
                   *sorted(root.glob('*/skill_workflow.json'))]
        for config_path in configs:
            if not config_path.is_file():
                continue
            trajectory = config_path.parent
            config = read_json(config_path)
            if config.get('role') == 'test' or config['reward']['version'] != INITIAL_TOP3_VERSION:
                continue
            for path in sorted((trajectory/'sampling').glob('mcts_skill*/**/checkpoint.json')):
                checkpoint = read_json(path)
                frozen, fact = checkpoint['frozen'], checkpoint['fact']
                ident = frozen['checkpoint_id']
                if ident in pool or not frozen.get('baseline_scores'):
                    continue
                # Only check that a comparison exists, never its sign or success.
                if not fact['reward'].get('update_eligible'):
                    continue
                score = match_score(edit['edit']['skill_id'], bundles, frozen)
                if score is not None:
                    pool[ident] = dict(task=fact['task'], source=fact['source'], stage=fact['search_stage'],
                        checkpoint_id=ident, checkpoint_folder=str(path.parent.resolve()),
                        trajectory=str(trajectory.resolve()), seed=frozen['seed'], match_score=score)
    selected = []
    while pool and len(selected) < 4:
        sources = {c['source'] for c in selected}
        preferred_stage = 'mcts_expansion' if selected else 'initialization'

        def priority(case):
            return (case['stage'] == preferred_stage, case['source'] not in sources,
                    case['match_score'], case['checkpoint_id'])

        case = max(pool.values(), key=priority)
        selected.append(pool.pop(case['checkpoint_id']))
    return [dict(case, seed=seed) for case in selected for seed in (seeds or [case['seed']])]
