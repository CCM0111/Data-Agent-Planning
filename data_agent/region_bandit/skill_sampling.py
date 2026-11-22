"""Frozen, outcome-blind dual-coreset plans and score-level aggregation."""
import math

from .reward import rank_method_ids
from .schemas import stable_hash


DEFAULT_POLICY = dict(version='adaptive-coreset-v1', mode='adaptive',
    coverage_loss_fraction=.10, boundary_gap_fraction=.05,
    trigger_skill_conflict=True, trigger_coverage_loss=True, trigger_boundary_gap=True)


def sampling_policy(settings=None):
    defaults = dict(DEFAULT_POLICY)
    if (settings or {}).get('version') in ('adaptive-coreset-v2', 'adaptive-coreset-v3'):
        defaults.update(version='adaptive-coreset-v2', trigger_unvalidated_intervention=True,
                        trigger_skill_abstention=True)
    if (settings or {}).get('version') == 'adaptive-coreset-v3':
        defaults.update(version='adaptive-coreset-v3', trigger_skill_abstention=False,
                        focus_exploration_max_sources=5, focus_exploration_max_donor_loss_fraction=.10,
                        focus_exploration_max_increase_fraction=.10)
        if 'enable_gentle_cover' in settings:
            if type(settings['enable_gentle_cover']) is not bool:
                raise ValueError('Invalid gentle cover setting')
            defaults['enable_gentle_cover'] = False
        if 'pair_original_control' in settings:
            if type(settings['pair_original_control']) is not bool:
                raise ValueError('Invalid original control setting')
            defaults['pair_original_control'] = False
        if 'trigger_nominated_focus' in settings:
            defaults['trigger_nominated_focus'] = False
    cfg = {**defaults, **(settings or {})}
    if (set(cfg) != set(defaults) or cfg['version'] not in ('adaptive-coreset-v1', 'adaptive-coreset-v2', 'adaptive-coreset-v3')
            or cfg['mode'] not in ('single', 'adaptive', 'dual')):
        raise ValueError('Invalid skill sampling policy')
    for key in ('coverage_loss_fraction', 'boundary_gap_fraction'):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or not 0 <= cfg[key] <= 1:
            raise ValueError('Invalid sampling threshold: '+key)
    for key in (k for k in cfg if k.startswith('trigger_')):
        if type(cfg[key]) is not bool:
            raise ValueError('Invalid sampling trigger: '+key)
    if cfg['version'] == 'adaptive-coreset-v3':
        n = cfg['focus_exploration_max_sources']
        if type(n) is not int or n < 0:
            raise ValueError('Invalid focus exploration budget')
        for key in ('focus_exploration_max_donor_loss_fraction', 'focus_exploration_max_increase_fraction'):
            if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or not 0 <= cfg[key] <= 1:
                raise ValueError('Invalid focus exploration bound')
    return cfg


def mcts_sampling_policy(settings=None):
    if isinstance(settings, str):
        if settings != 'single':
            raise ValueError('Skill MCTS requires single-branch sampling')
        settings = None
    policy = sampling_policy(settings or dict(version='adaptive-coreset-v3', mode='single',
        trigger_skill_conflict=False, trigger_skill_abstention=False, trigger_nominated_focus=True,
        pair_original_control=False, enable_gentle_cover=False, focus_exploration_max_sources=0))
    if policy['mode'] != 'single':
        raise ValueError('Skill MCTS requires single-branch sampling; use a fresh single-branch run')
    return policy


def single_branch_trigger():
    return dict(run_alternative=False, reasons=['single_policy'], aggregation='equal_metric_mean_v1')


def plan_sampling(decision, settings, *, local_support=(), exploration=None):
    """Choose a backup before outcomes; API may nominate it, never set the budget."""
    cfg = sampling_policy(settings)
    actions = {a['action_id']: a for a in decision['actions']}
    primary = actions[decision['chosen_action_id']]
    execution = decision['skill_execution']
    plan = dict(protocol=cfg, protocol_hash=stable_hash(cfg), primary_action_id=primary['action_id'],
                alternative_action_id=None, alternative_source=None, seeds={})
    if cfg['mode'] == 'single':
        for seed in decision['sampling_seeds']:
            plan['seeds'][str(seed)] = dict(alternative_available=False, prefit_reasons=[],
                coverage_losses=[], proposal_disagreement=False, partial_skill_abstention=False)
        return plan
    enabled = execution['decision_status'] != 'fallback'
    def previews(action):
        return action['context']['action_summary']['cross_region_preview']
    def differs(action):
        return any(previews(primary)[str(s)]['sample_hash'] != previews(action)[str(s)]['sample_hash']
                   for s in decision['sampling_seeds'])
    baseline = next(a for a in actions.values() if a['type'] == 'baseline')
    pair_original = cfg.get('pair_original_control', False) and primary['type'] != 'baseline'
    pool = [baseline] if pair_original else actions.values()
    alternatives = [a for a in pool if a['action_id'] != primary['action_id'] and differs(a)]
    preferred = execution.get('alternative_action_id')
    alternate = next((a for a in alternatives if a['action_id'] == preferred), None) if enabled else None
    nominated_focus = (cfg.get('trigger_nominated_focus', False) and alternate is not None
                       and alternate['type'] == 'focus')
    # A large within-coreset score gap is not evidence that an intervention preserves full ranking.
    # Independent sources never stand in for matched local evidence for this guard.
    unsupported = (cfg.get('trigger_unvalidated_intervention', False)
                   and primary['type'] != 'baseline' and differs(baseline)
                   and primary['action_id'] not in local_support)
    if enabled and pair_original and alternatives:
        alternate = baseline
        plan['alternative_source'] = 'paired_original_control'
    elif enabled and unsupported and not nominated_focus:
        alternate = baseline
        plan['alternative_source'] = 'unvalidated_intervention_control'
    if alternate:
        plan['alternative_source'] = plan['alternative_source'] or 'skill_api'
    elif enabled and alternatives:
        # A meaningful fixed contrast: cover/focus on the same region, then baseline.
        alternatives.sort(key=lambda a: (
            0 if primary.get('region_id') and a.get('region_id') == primary['region_id']
                 and {a['type'], primary['type']} == {'cover', 'focus'} else
            1 if a['type'] == 'baseline' else 2, a['action_id']))
        alternate = alternatives[0]
        plan['alternative_source'] = 'deterministic_contrast'
    if alternate:
        plan['alternative_action_id'] = alternate['action_id']
    # A bounded DEVELOPMENT-only experiment, not an API recommendation or an
    # outcome-dependent switch. The baseline primary is never overridden.
    if (enabled and cfg['mode'] == 'adaptive' and cfg['version'] == 'adaptive-coreset-v3'
            and not nominated_focus and exploration and exploration.get('eligible') and exploration.get('role') == 'development'
            and cfg['focus_exploration_max_sources'] > 0 and primary['type'] == 'baseline'):
        candidates = []
        for a in alternatives:
            if a['type'] != 'focus' or a['context']['region_summary'].get('direction_consistent') is not True:
                continue
            max_loss, max_increment, valid = 0., 0., True
            for seed in decision['sampling_seeds']:
                preview = previews(a)[str(seed)]
                target = preview['regions'].get(a['region_id'])
                if (not target or target['after'] <= target['before'] or not a.get('size')
                        or preview['sample_hash'] == previews(primary)[str(seed)]['sample_hash']):
                    valid = False
                    break
                max_increment = max(max_increment, (target['after']-target['before'])/a['size'])
                for rid, r in preview['regions'].items():
                    if rid != a['region_id'] and r['before'] > 0:
                        max_loss = max(max_loss, max(0., (r['before']-r['after'])/r['before']))
            if (valid and max_loss <= cfg['focus_exploration_max_donor_loss_fraction']
                    and max_increment <= cfg['focus_exploration_max_increase_fraction']):
                candidates.append((max_loss, max_increment, a['action_id'], a))
        if candidates:
            _, _, _, alternate = min(candidates, key=lambda x: x[:3])
            plan.update(alternative_action_id=alternate['action_id'],
                        alternative_source='bounded_development_focus_exploration',
                        purpose='exploration_control', aggregation='primary_only_exploration_v1')
    for seed in decision['sampling_seeds']:
        key = str(seed)
        available = alternate is not None and previews(primary)[key]['sample_hash'] != previews(alternate)[key]['sample_hash']
        reasons, losses = [], []
        if available and plan.get('purpose') == 'exploration_control':
            reasons.append('bounded_focus_exploration')
        proposals = execution.get('skill_proposals', [])
        proposal_hashes = {previews(actions[p['action_id']])[key]['sample_hash'] for p in proposals
                           if p['action_id'] != 'abstain'}
        if available and cfg['mode'] == 'dual':
            reasons.append('predeclared_dual_control')
        if available and cfg['mode'] == 'adaptive':
            if nominated_focus:
                reasons.append('skill_nominated_focus')
            elif unsupported:
                reasons.append('unvalidated_intervention')
            if cfg['trigger_skill_conflict'] and len(proposal_hashes) > 1:
                reasons.append('different_skill_proposals')
            if (cfg.get('trigger_skill_abstention', False) and proposal_hashes
                    and any(p['action_id'] == 'abstain' for p in proposals)):
                reasons.append('partial_skill_abstention')
            for rid, counts in previews(primary)[key]['regions'].items():
                if rid != primary.get('region_id') and counts['before'] > 0:
                    loss = max(0., (counts['before']-counts['after'])/counts['before'])
                    if loss > 0 and loss >= cfg['coverage_loss_fraction']:
                        losses.append(dict(region_id=rid, fraction=loss))
            if cfg['trigger_coverage_loss'] and losses:
                reasons.append('cross_region_coverage_loss')
        plan['seeds'][key] = dict(alternative_available=available, prefit_reasons=reasons,
                                  coverage_losses=losses,
                                  proposal_disagreement=len(proposal_hashes) > 1,
                                  partial_skill_abstention=bool(proposal_hashes and any(
                                      p['action_id'] == 'abstain' for p in proposals)))
    return plan


def decide_second_branch(plan, seed, primary_scores, *, keep_k, direction, ranking_settings):
    """The only post-fit trigger reads primary V_core scores, never full results."""
    cfg, entry = plan['protocol'], plan['seeds'][str(seed)]
    reasons = list(entry['prefit_reasons'])
    ranking_settings = {k: ranking_settings[k] for k in ('atol', 'rtol', 'practical_tolerance')
                        if k in ranking_settings}
    ids = rank_method_ids(primary_scores, direction=direction, config=ranking_settings)
    if type(keep_k) is not int or not 1 <= keep_k <= len(ids):
        raise ValueError('Invalid screening keep count')
    gap = spread = relative = None
    if 0 < keep_k < len(ids):
        gap = abs(primary_scores[ids[keep_k-1]]-primary_scores[ids[keep_k]])
        spread = max(primary_scores.values())-min(primary_scores.values())
        relative = gap/spread if spread > 0 else 0.
        if (entry['alternative_available'] and cfg['mode'] == 'adaptive'
                and cfg['trigger_boundary_gap'] and relative <= cfg['boundary_gap_fraction']):
            reasons.append('small_screening_boundary_gap')
    return dict(**entry, reasons=reasons, run_alternative=bool(entry['alternative_available'] and reasons),
                aggregation=plan.get('aggregation', 'equal_metric_mean_v1'),
                purpose=plan.get('purpose', 'screening'),
                boundary_gap=gap, score_range=spread, relative_boundary_gap=relative,
                interpretation='Engineering trigger, not calibrated statistical uncertainty')


def combine_scores(branches, *, direction, ranking_settings, aggregation='equal_metric_mean_v1'):
    """Equal-weight metric means, not prediction ensembling or reward averaging."""
    if not 1 <= len(branches) <= 2:
        raise ValueError('One or two completed branches required')
    if aggregation not in ('equal_metric_mean_v1', 'primary_only_exploration_v1'):
        raise ValueError('Unknown frozen score aggregation')
    ranking_settings = {k: ranking_settings[k] for k in ('atol', 'rtol', 'practical_tolerance')
                        if k in ranking_settings}
    ids = set(branches[0]['scores'])
    if not ids or any(set(b['scores']) != ids for b in branches):
        raise ValueError('Branches must evaluate the same fixed method bank')
    if any(type(v) not in (int, float) or not math.isfinite(v)
           for b in branches for v in b['scores'].values()):
        raise ValueError('Incomplete or nonfinite branch scores')
    if len({b['sample_hash'] for b in branches}) != len(branches):
        raise ValueError('Identical samples must be deduplicated before aggregation')
    scores = (dict(branches[0]['scores']) if aggregation == 'primary_only_exploration_v1' else
              {i: sum(b['scores'][i]/len(branches) for b in branches) for i in ids})
    ranks = [rank_method_ids(b['scores'], direction=direction, config=ranking_settings) for b in branches]
    reversed_pairs = []
    if len(branches) == 2:
        ordered = sorted(ids, key=str)
        for j, a in enumerate(ordered):
            for b in ordered[j+1:]:
                if (branches[0]['scores'][a]-branches[0]['scores'][b]) * (
                        branches[1]['scores'][a]-branches[1]['scores'][b]) < 0:
                    reversed_pairs.append([a, b])
    # Branch order does not change equal-weight predictions or validation identity.
    samples = [b['sample_hash'] for b in branches]
    identity = branches[0]['sample_hash'] if len(branches) == 1 else stable_hash(dict(
        aggregation=aggregation, samples=samples if aggregation == 'primary_only_exploration_v1' else sorted(samples)))
    return dict(scores=scores, sample_hash=identity, aggregation=aggregation,
                branch_count=len(branches), branch_rankings=ranks,
                ranking=rank_method_ids(scores, direction=direction, config=ranking_settings),
                reversed_method_pairs=reversed_pairs)
