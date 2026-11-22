"""Bounded procedural skills, checked decisions and limited schema repair."""
from copy import deepcopy
from pathlib import Path
import inspect
import json
import math
import re

from .skill_prompt import compact_request
from .schemas import stable_hash
from .context import pair_at_boundary

VERSION = 'procedural-region-v1'
FIELDS = {'region_summary.coverage_gap', 'region_summary.direction_consistent',
          'region_summary.discover_score', 'region_summary.check_score',
          'action_summary.actual_region_increase_fraction',
          'search_features.is_initialization', 'search_features.progress',
          'search_features.parent_depth', 'search_features.prior_conflict_fraction',
          'search_features.current_replacement_fraction', 'search_features.model_diversity'}


def validate_decision_rule(text):
    """Keep post-experiment quantities in edit rationale, never executable rules.

    This guards explicit field dependencies; the executor additionally has no access
    to current-task full outcomes. It is not a general natural-language proof checker.
    """
    # This legacy caution explicitly rejects an outcome-based inference; it does
    # not ask the executor to observe a current-task reward before acting.
    if text == 'A high absolute reward does not establish benefit over unchanged sampling.':
        return text
    forbidden = (r'\b(?:branch_reward_delta|combination_gain|branch_count|paired_baseline|'
                 r'full_scores?|champion\w*|rewards?|retained_ids|keep_loss|order_loss)\b|'
                 r'\b(?:full[-_ ](?:scores?|train(?:ing)?|reference)|combined score|'
                 r'branch[-_ ](?:scores?|performance)|validation (?:scores?|metrics?))\b|'
                 r'全量.*(?:成绩|得分|结果|冠军)|奖励|组合收益|分支收益|冠军漏选')
    if not isinstance(text, str) or re.search(forbidden, text, flags=re.IGNORECASE):
        raise ValueError('Executable rule depends on post-experiment outcomes; use pre-decision '
                         'coverage, proxy differences, quota conflicts or historical applicability. '
                         'Keep outcome analysis in reason/evidence_ids, not value.')
    return text


def initial_bundle():
    """Seed the library with warm-start procedures and initial screening guidance."""
    config = Path(__file__).resolve().parents[2] / 'config' / 'region_skill_warmstart_20261118'
    bundle = json.loads((config / 'initial_skills.json').read_text(encoding='utf-8'))
    skills = {skill['id']: skill for skill in bundle['skills']}
    skills['coverage_repair']['steps'].append(
        'For undercoverage alone, prefer cover to restore the pool share. Compare feasible '
        'target additions and displaced donor rows against baseline before intervening.')
    skills['difference_preservation']['steps'].extend([
        'Compare the SAME method IDs in admission.observed_ranking and pair_loss_evidence. '
        'Respect the admission metric direction. Positive regional loss_A_minus_B favors B; '
        'negative favors A. Call it opposing evidence only if both discovery and check favor '
        'the overall trailer, with adequate support. A stable inside/outside contrast alone '
        'does not mean the local winner changed. Regional loss is a diagnostic, not the task metric.',
        'Start with the pair competing for the last admission slot (e.g. third versus fourth). '
        'Check its observed margin, the method IDs, regional support and the pair contrast on '
        'both discovery and check. The same local winner can still be informative when the '
        'region separates this close pair more clearly than outside on both splits. A reversal '
        'is optional. Regional losses and the admission metric have different units.'
    ])
    skills['intervention_check']['steps'].append(
        'For a close boundary pair with replicated separation and adequate counts, prefer a '
        'supplied bounded focus when its preview adds useful rows beyond restoring the pool '
        'share and the donors remain well represented. This can apply even with a small '
        'coverage gap. For a representation gap alone, prefer cover if its preview is affordable. '
        'Keep baseline when the supplied '
        'previews show little useful change or displacement outweighs it. Missing pair evidence '
        'cannot support a focus claim.')
    return validate_bundle(bundle)


def validate_bundle(bundle):
    if set(bundle) != {'schema', 'skills'} or bundle['schema'] != VERSION or len(bundle['skills']) != 3:
        raise ValueError('Expected the three-skill procedural schema')
    if [s['id'] for s in bundle['skills']] != ['coverage_repair', 'difference_preservation', 'intervention_check']:
        raise ValueError('Skill identity/order is fixed')
    for s in bundle['skills']:
        if set(s) != {'id', 'version', 'name', 'activation', 'steps', 'exceptions', 'allowed_actions',
                      'termination', 'evidence_state', 'evidence_ids'}:
            raise ValueError('Unknown or missing skill fields')
        if type(s['version']) is not int or s['version'] < 1:
            raise ValueError('Invalid skill version')
        if s['allowed_actions'] != ['baseline', 'cover', 'focus', 'abstain']:
            raise ValueError('Skill cannot change executable action space')
        if s['termination'] != 'Return one supplied legal action ID or abstain.':
            raise ValueError('Skill termination contract is fixed')
        for key in ('name', 'evidence_state'):
            if not isinstance(s[key], str) or not s[key].strip():
                raise ValueError('Invalid skill text')
        if s['evidence_state'] not in ('engineering_prior', 'candidate_hypothesis'):
            raise ValueError('Acceptance is registry state, never a self-declared skill property')
        if not isinstance(s['evidence_ids'], list) or any(not isinstance(x, str) for x in s['evidence_ids']):
            raise ValueError('Invalid evidence IDs')
        for key in ('steps', 'exceptions'):
            if not isinstance(s[key], list) or len(s[key]) > 12 or (key == 'steps' and not s[key]):
                raise ValueError('Invalid procedure length')
            if any(not isinstance(x, str) or not x.strip() or len(x) > 1000 for x in s[key]):
                raise ValueError('Invalid procedure statement')
        if not isinstance(s['activation'], list) or len(s['activation']) > 8:
            raise ValueError('Invalid activation clauses')
        for clause in s['activation']:
            if (not isinstance(clause, dict) or set(clause) != {'field', 'op', 'value'}
                    or not isinstance(clause['field'], str) or clause['field'] not in FIELDS
                    or clause['op'] not in ('gt', 'ge', 'eq') or type(clause['value']) not in (bool, int, float)
                    or not math.isfinite(clause['value']) or not 0 <= clause['value'] <= 1):
                raise ValueError('Unsupported activation predicate')
    if len(json.dumps(bundle)) > 18000:
        raise ValueError('Skill bundle exceeds text budget')
    return bundle


def _lookup(context, field):
    for key in field.split('.'):
        if not isinstance(context, dict) or key not in context:
            return None
        context = context[key]
    return context


def _citation_path(context, field):
    """Resolve known shortened citations only against the exact supplied action."""
    if _lookup(context, field) is not None:
        return field
    parts = field.split('.')
    if parts[:2] in (['budget_summary', 'proxy_reliability'], ['budget_summary', 'rollout_allocation']):
        candidate = '.'.join([parts[0], 'search_state', *parts[1:]])
    elif field in ('action_summary.sample_replaced_count', 'action_summary.sample_replaced_fraction'):
        previews = _lookup(context, 'action_summary.cross_region_preview')
        if not isinstance(previews, dict) or len(previews) != 1:
            return None
        candidate = 'action_summary.cross_region_preview.' + str(next(iter(previews))) + '.' + parts[-1]
    else:
        return None
    return candidate if _lookup(context, candidate) is not None else None


def activated(bundle, contexts):
    def applies(skill, context):
        for c in skill['activation']:
            value = _lookup(context, c['field'])
            if type(value) not in (int, float, bool) or not math.isfinite(value):
                return False
            if not {'gt': value > c['value'], 'ge': value >= c['value'], 'eq': value == c['value']}[c['op']]:
                return False
        return True
    return [{**s, 'applicable_action_ids': [c['action_id'] for c in contexts if applies(s, c)]}
            for s in validate_bundle(bundle)['skills'] if any(applies(s, c) for c in contexts)]


def apply_edit(bundle, edit, evidence_ids):
    """One ADD/REPLACE/DELETE on one list entry; immutable surrounding contract."""
    validate_bundle(bundle)
    if not isinstance(edit, dict) or set(edit) != {'skill_id', 'section', 'operation', 'index', 'value', 'reason', 'evidence_ids'}:
        raise ValueError('Expected one exact structured edit')
    if not isinstance(edit['skill_id'], str):
        raise ValueError('Proposal requires a string skill_id')
    if edit['section'] not in ('activation', 'steps', 'exceptions') or edit['operation'] not in ('ADD', 'REPLACE', 'DELETE'):
        raise ValueError('Unsupported skill edit')
    if not isinstance(edit['reason'], str) or not edit['reason'].strip() or len(edit['reason']) > 1000:
        raise ValueError('Edit needs a bounded reason')
    citations = edit['evidence_ids']
    if (not isinstance(citations, list) or any(not isinstance(i, str) for i in citations)
            or len(set(citations)) != len(citations) or not citations
            or not set(citations) <= set(evidence_ids)):
        raise ValueError('Edit must cite supplied development facts')
    if len(json.dumps(edit['value'])) > 1200:
        raise ValueError('Single edit exceeds text budget')
    if edit['section'] in ('steps', 'exceptions') and edit['operation'] != 'DELETE':
        validate_decision_rule(edit['value'])
    result = deepcopy(bundle)
    skill = next((s for s in result['skills'] if s['id'] == edit['skill_id']), None)
    if skill is None:
        raise ValueError('Unknown skill')
    rows, index = skill[edit['section']], edit['index']
    if type(index) is not int or not 0 <= index <= len(rows) or (edit['operation'] != 'ADD' and index == len(rows)):
        raise ValueError('Invalid edit position')
    if edit['operation'] == 'ADD':
        rows.insert(index, edit['value'])
    elif edit['operation'] == 'REPLACE':
        if rows[index] == edit['value']:
            raise ValueError('No-op edit')
        rows[index] = edit['value']
    else:
        if edit['value'] is not None:
            raise ValueError('DELETE value must be null')
        rows.pop(index)
    skill.update(version=skill['version']+1, evidence_state='candidate_hypothesis', evidence_ids=citations)
    return validate_bundle(result)


def _factor_action_contexts(contexts):
    """Deduplicate transport fields without rounding or dropping any evidence.

    Regional summaries are shared by exact serialized content, never by a region
    label alone: cover/focus may have different evidence despite naming a region
    alike. Action summaries (including every seed's preview) remain per action.
    """
    actions = [dict(context) for context in contexts]
    encode = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False,
                                       separators=(',', ':'))
    shared = {}
    for key in ('task_summary', 'method_summary', 'budget_summary', 'comparison_scope'):
        if actions and all(key in action for action in actions) and all(
                encode(action[key]) == encode(actions[0][key]) for action in actions):
            shared[key] = actions[0][key]
            for action in actions:
                del action[key]
    # Some budgets differ only in the action-specific quota. Transmit the
    # common fields once and retain exact per-action overrides.
    if actions and all(isinstance(a.get('budget_summary'), dict) for a in actions):
        first = actions[0]['budget_summary']
        common = {k: v for k, v in first.items()
                  if all(k in a['budget_summary'] and encode(a['budget_summary'][k]) == encode(v)
                         for a in actions)}
        if common:
            shared['budget_summary'] = common
            for action in actions:
                action['budget_summary'] = {k: v for k, v in action['budget_summary'].items() if k not in common}
    groups = {}
    for index, action in enumerate(actions):
        if 'region_summary' in action:
            groups.setdefault(encode(action['region_summary']), []).append(index)
    regions = {}
    for indices in groups.values():
        if len(indices) < 2:
            continue
        reference = 'r' + str(len(regions) + 1)
        summary = actions[indices[0]]['region_summary']
        # Tiny shared summaries can cost more to reference than to repeat.
        original_size = len(encode([{'region_summary': summary} for _ in indices]))
        factored_size = len(encode({'region_contexts': {reference: summary},
            'actions': [{'region_context_id': reference} for _ in indices]}))
        if factored_size >= original_size:
            continue
        regions[reference] = summary
        for index in indices:
            actions[index].pop('region_summary')
            actions[index]['region_context_id'] = reference
    return actions, shared, regions


def _expand_action_contexts(request):
    """Recover the original evidence contexts from old or factored requests."""
    contexts = []
    for action in request['actions']:
        context = {**request.get('shared_context', {}), **action}
        if 'budget_summary' in action and 'budget_summary' in request.get('shared_context', {}):
            context['budget_summary'] = {**request['shared_context']['budget_summary'], **action['budget_summary']}
        reference = context.pop('region_context_id', None)
        if reference is not None:
            context['region_summary'] = request['region_contexts'][reference]
        if 'action_summary' in context:
            summary = dict(context['action_summary'])
            if context['action_type'] != 'baseline':
                summary = {**request.get('regional_action_defaults', {}), **summary}
            if 'cross_region_preview' in summary:
                previews = deepcopy(summary['cross_region_preview'])
                for preview in previews.values():
                    if 'region_table' in preview:
                        table = request['preview_tables'][preview.pop('region_table')]
                        preview['regions'] = {k: dict(before=b, after=a, change=a-b) for k,b,a in
                                              zip(table['region_ids'], table['before'], preview.pop('after'))}
                summary['cross_region_preview'] = previews
            context['action_summary'] = summary
        contexts.append(context)
    return contexts


def _advisor_history(card):
    # A single committed MCTS checkpoint has one action and one reward.
    # Grouped seed evidence keeps its separate outcomes and counterexamples.
    if not card.get('sampling_path') or card.get('branch_count') != 1 or not card.get('search_stage'):
        return card
    result = {k: card[k] for k in ('task', 'source', 'scope', 'search_stage', 'round_number',
              'sampling_path', 'action', 'skill_execution', 'baseline_kind', 'comparison_warning')
              if k in card}
    reward = card.get('reward', {})
    result['reward'] = {k: reward[k] for k in ('reward', 'status', 'reward_basis',
                        'paired_method_ids', 'observed_method_ids', 'baseline_order_accuracy',
                        'order_accuracy', 'update_eligible') if k in reward}
    if 'parent_child_comparisons' in reward:
        result['reward']['comparisons'] = [
            {k: pair[k] for k in ('child_id', 'before', 'after', 'gain') if k in pair}
            for pair in reward['parent_child_comparisons']]
    return result


def _history_stage(context):
    if context.get('search_features', {}).get('is_initialization'):
        return 'initialization'
    return context.get('search_stage')


def _variant(action):
    return 'gentle_cover' if (action.get('variant') == 'gentle_cover' or
        str(action.get('action_id', '')).startswith('gentle_cover:')) else action.get('type')


def _region_match(card, contexts):
    path = card['sampling_path']
    region = path.get('region') or {}
    action = card['action']
    variant = _variant(action)
    matches = []
    for ctx in contexts:
        candidate = {**ctx.get('action_summary', {}), 'type': ctx['action_type'], 'action_id': ctx.get('action_id')}
        if variant != _variant(candidate):
            continue
        if variant == 'baseline':
            return 0.
        current = ctx.get('region_summary') or {}
        if not region.get('discovery_kind') or region['discovery_kind'] != current.get('discovery_kind'):
            continue
        score = 0.
        if variant in ('cover', 'gentle_cover'):
            for before, now in [(region.get('coverage_gap'), current.get('coverage_gap')),
                                (action.get('actual_region_increase_fraction'), candidate.get('actual_region_increase_fraction'))]:
                if before is not None and now is not None:
                    score += max(0., 1 - abs(before-now)/max(abs(before), abs(now), .0001))
        elif variant == 'focus':
            for field in ('direction_consistent', 'pair_at_boundary'):
                old = region.get(field)
                if field == 'direction_consistent' and old is None:
                    old = (card.get('decision_context') or {}).get(field)
                new = current.get(field) if field != 'pair_at_boundary' else pair_at_boundary(ctx)
                if old is not None and new is not None:
                    score += int(old == new)
        matches.append(score)
    return max(matches, default=None)


def _rank_history(history, contexts):
    """Bounded paired-evidence retrieval for the MCTS advisor."""
    current = contexts[0]
    stage = _history_stage(current)
    basis = {'initialization': 'observed_initial_champion_order_gain',
             'mcts_expansion': 'fixed_parent_direction_reward'}.get(stage)
    eda = current.get('task_summary', {})
    families = {m.get('family') for m in current.get('method_summary', [])} - {None}
    by_source = {}
    seen_facts = set()
    for card in history:
        reward = card.get('reward', {})
        value = reward.get('reward')
        path = card.get('sampling_path') or {}
        other = path.get('eda', {})
        observed = set(reward.get('observed_method_ids') or [])
        if (not basis or card.get('search_stage') != stage or reward.get('reward_basis') != basis
                or not eda.get('problem_type') or not other.get('problem_type')
                or not card.get('fact_id') or not card.get('source') or not card.get('action')
                or card.get('skill_execution', {}).get('decision_status') != 'selected'
                or reward.get('update_eligible') is not True or type(value) not in (int, float)
                or not math.isfinite(value) or value == 0 or len(observed) < 2
                or not (reward.get('comparable_pair_count') or 0) > 0):
            continue
        children = [p['child_id'] for p in reward.get('parent_child_comparisons', []) if p.get('child_id') in observed]
        if stage == 'mcts_expansion' and not children:
            continue
        if card['fact_id'] in seen_facts:
            continue
        seen_facts.add(card['fact_id'])
        score = _region_match(card, contexts)
        same_type = eda['problem_type'] == other['problem_type']
        if score is None:
            # Only transfer a known action across region kinds within the same task type.
            variant = _variant(card['action'])
            if not same_type or not any(_variant({**c.get('action_summary', {}),
                    'action_id': c.get('action_id'), 'type': c['action_type']}) == variant for c in contexts):
                continue
            match = 'same_task_type_and_action_only'
            score = 0.
        else:
            match = 'same_task_type_and_region' if same_type else 'cross_task_type_same_action_region'
        bank = {m.get('family') for m in path.get('methods', [])} - {None}
        if not same_type and not families & bank:
            continue
        # Large offsets give each tier priority over the bounded similarity score.
        score += {'same_task_type_and_region': 1000, 'same_task_type_and_action_only': 100,
                  'cross_task_type_same_action_region': 0}[match]
        card = dict(card, retrieval_match=match)
        score += 2 * len(families & bank) / max(1, len(families | bank))
        score += 2 * int(bool(other.get('metric')) and other['metric'] == eda.get('metric'))
        for field in ('n_rows', 'n_features'):
            a, b = eda.get(field), other.get(field)
            if a and b:
                score += min(a, b)/max(a, b)
        region = path.get('region') or {}
        mechanism = [(m.get('family'), m.get('implementation') or m.get('name'), m.get('parameter_hash'))
                     for m in path.get('methods', [])]
        key = stable_hash([card.get('task'), _variant(card['action']), region.get('rule'),
                           region.get('discovery_kind'), sorted(map(str, mechanism)), value > 0])
        bucket = by_source.setdefault(card['source'], {})
        old = bucket.get(key)
        repeat = old[1]['similar_history_count'] + 1 if old else 1
        if old and old[0] > score:
            score, card = old
        bucket[key] = (score, dict(card, similar_history_count=repeat))
    # For each source choose zero, one, or two distinct paths. The small
    # bounded panel table avoids a greedy pick consuming a slot needed by the other sign.
    panels = {(0, 0): (0., [])}
    for bucket in by_source.values():
        ranked = sorted(bucket.values(), key=lambda x: (-x[0], x[1]['fact_id']))
        pos = [x for x in ranked if x[1]['reward']['reward'] > 0][:2]
        neg = [x for x in ranked if x[1]['reward']['reward'] < 0][:2]
        updated = dict(panels)
        for (p, n), (total, cards) in panels.items():
            for i in range(len(pos)+1):
                for j in range(len(neg)+1):
                    key = (p+i, n+j)
                    if not 1 <= i+j <= 2 or sum(key) > 10:
                        continue
                    picks = pos[:i]+neg[:j]
                    weight = total + sum(x[0] for x in picks)
                    if key not in updated or weight > updated[key][0]:
                        updated[key] = (weight, cards + picks)
        panels = updated
    best = max(panels, key=lambda k: (sum(k), min(k[0], 7) + min(k[1], 3), panels[k][0]))
    return [c for _, c in sorted(panels[best][1], key=lambda x: -x[0])]


def _pack_paths(history):
    if len(history) == 1 and 'similar_history_count' not in history[0]:
        return [_advisor_history(history[0])], {}
    profiles, methods, cards = {}, {}, []
    for card in history:
        if not card.get('sampling_path') or card.get('history_format') == 'sampling_seed_group_v1':
            cards.append(_advisor_history(card))
            continue
        path = card['sampling_path']
        item = {k: card[k] for k in ('task', 'scope', 'search_stage', 'round_number', 'baseline_kind', 'fact_id', 'similar_history_count', 'retrieval_match') if k in card}
        if card.get('source') != card.get('task'):
            item['source'] = card.get('source')
        profile = path.get('eda', {})
        profile_id = stable_hash(profile)[:16]
        profiles[profile_id] = profile
        item['eda_ref'] = profile_id
        item['methods'] = []
        for method in path.get('methods', []):
            description = {k: v for k, v in method.items() if k not in ('id', 'method_id')}
            ref = stable_hash(description)[:16]
            methods[ref] = description
            item['methods'].append([method.get('id', method.get('method_id')), ref])
        if path.get('sample_effect_by_seed'):
            item['sample_effect_by_seed'] = path['sample_effect_by_seed']
        if 'candidate_regions' in path:
            item['candidate_region_count'] = len(path['candidate_regions'])
        item['region'] = path.get('region')
        item['action'] = card.get('action')
        if card.get('validation_actions'):
            item['observed_action_comparison'] = card['validation_actions']
        item['decision'] = {k: v for k, v in card.get('skill_execution', {}).items()
                            if k in ('decision_status', 'skills_used', 'reason_excerpt', 'evidence_fields')}
        reward = card.get('reward', {})
        item['reward'] = {k: reward[k] for k in ('reward', 'status', 'reward_basis', 'observed_method_ids',
            'baseline_order_accuracy', 'order_accuracy', 'update_eligible') if k in reward}
        item['reward']['comparable_pair_count'] = reward.get('comparable_pair_count')
        item['reward']['parent_child_pairs'] = len(reward.get('parent_child_comparisons') or [])
        cards.append(item)
    return cards, {'eda': profiles, 'methods': methods}


async def choose_with_skills(contexts, bundle, request_fn, *, history=(), sampling=None):
    """One API request, no score fusion, no repair calls; failures are explicit."""
    contexts = list(contexts)
    history = list(history)
    ids = [c['action_id'] for c in contexts]
    baselines = [c['action_id'] for c in contexts if c['action_type'] == 'baseline']
    if len(set(ids)) != len(ids) or len(baselines) != 1:
        raise ValueError('Distinct actions and exactly one baseline required')
    skills = activated(bundle, contexts)
    # Activation identifies the regional evidence a skill may inspect. An active
    # skill can still recommend no intervention, even though the baseline has no
    # region_summary and therefore cannot satisfy a regional activation clause.
    # Expose and validate the same output contract; do not broaden activation.
    for skill in skills:
        skill['allowed_proposal_action_ids'] = list(dict.fromkeys(
            [*skill['applicable_action_ids'], baselines[0], 'abstain']))
    active_ids = {s['id']+'@v'+str(s['version']) for s in skills}
    # Exclude numerical bandit features and posterior labels from action context.
    keys = ('action_id', 'action_type', 'task_summary', 'method_summary', 'budget_summary',
            'region_summary', 'action_summary', 'comparison_scope', 'search_features',
            'search_stage', 'proxy_scope')
    action_contexts = [{k: c[k] for k in keys if k in c} for c in contexts]
    actions, shared, regions = _factor_action_contexts(action_contexts)
    if _history_stage(contexts[0]) in ('initialization', 'mcts_expansion'):
        history = _rank_history(history, contexts)
    history_cards, path_details = _pack_paths(history)
    request = dict(protocol=VERSION, skill_bundle_hash=stable_hash(bundle), skills=skills,
        actions=actions, historical_facts=history_cards,
        instruction=('Execute the supplied procedures in order: proposals then intervention_check. '
          'Activation is already checked by the program. Use only supplied evidence. Initial rules are '
          'engineering priors, not proven causal laws. History is conditional evidence; high absolute '
          'reward does not establish improvement. Never invent unobserved full outcomes. '
          'If a procedure requires unavailable current-task outcomes or other absent evidence, '
          'abstain from that recommendation and explain the missing evidence; do not estimate it. '
          'Choose one supplied action, unchanged quotas/strength, or abstain. No score fusion or random override. '
          'Return JSON {action_id: string or "abstain", skills_used: ["id@vN"], '
          'checks: [{action_id: string, field: dotted path within that supplied action context'
          '}], reason: one short sentence}. Include at least one grounded check for an intervention. '
          'If evidence is insufficient, abstain. Checks are auditable citations, not proof of causation.'))
    if any(c.get('editable_sample_scope') for c in contexts):
        request['instruction'] += (' This decision can only edit the residual 85% of this round. '
            'The MMD and per-method error rows remain protected. Baseline keeps the newly constructed '
            'reference for this round; it does not revert to the previous round.')
    if any(c['action_summary'].get('variant') == 'gentle_cover' for c in contexts if 'action_summary' in c):
        request['instruction'] += (
            ' gentle_cover is an optional partial cover action. Consider it only when ordinary '
            'cover needs a large correction or its preview replaces many rows. Compare the same '
            'region against baseline and full cover using the actual per-seed counts and row '
            'replacements. It closes half the feasible shortfall (rounded up), with at most 10% '
            'of coreset rows exchanged within their original groups. It may leave undercoverage. '
            'Compare all available actions on expected ranking fidelity. Keep baseline when coverage and ordering evidence are adequate; use cover for a material supported shortfall, focus for credible local evidence distinguishing methods, and gentle_cover when a partial repair addresses the gap while avoiding a concrete loss seen in full cover. Small edit size alone is insufficient. Use measured historical alternatives when relevant. Do not target action frequencies.')
    request['instruction'] += (' Shared budget_summary fields apply to every action; merge the per-action fields. '
        'Give one short reason referencing the decisive observation; checks contain citations only. Omit stock caveats, self-justification and restatements of policy. If evidence is missing, name the missing observation once.')
    if path_details.get('eda'):
        request['historical_path_details'] = path_details
        request['instruction'] += (' Historical paths are deliberately stratified by signed reward (target 7 positive and 3 negative; spare slots may use the other sign). Counts do not estimate a success rate. Treat cross-task-type matches and action-only matches as secondary analogies. A positive fixed-parent reward can reflect preserving a correct ordering; it does not establish improvement over the parent sample. Historical paths reference shared EDA and [method ID, method detail ID]. '
            'Use only the most relevant observed fact to justify the choice. Omit the search narrative '
            'and repeated evidence. State the decision reason once.')
    request['shared_context'] = shared
    if any(_lookup(c, 'budget_summary.search_state') is not None for c in contexts):
        request['instruction'] += (
            ' For search diagnostics cite budget_summary.search_state.rollout_allocation or '
            'budget_summary.search_state.proxy_reliability and its supplied fields. For row '
            'replacements cite action_summary.cross_region_preview.<actual seed>.sample_replaced_count.')
    if history:
        request['instruction'] += (
            ' Compare historical sampling paths with the current dataset, method implementations and region. '
            'Use the recorded skills, action and signed reward from comparable stages to weigh both '
            'successful and adverse precedents. Treat fallback choices and missing rewards separately. '
            'Ground the final choice in this task and briefly identify any precedent that influenced it.')
    if any(h.get('history_format') == 'sampling_seed_group_v1' for h in history):
        request['instruction'] += (
            ' Each sampling_seed_group_v1 history card is one decision on one source, not independent tasks. '
            'Summaries are descriptive: inspect seed_outcomes and counterexample_seeds before generalizing. '
            'action_index refers to that card\'s actions array; reward_context applies to seed/branch rewards '
            'unless explicitly overridden. Partial full references do not establish champion retention.')
    if regions:
        request['region_contexts'] = regions
        request['instruction'] += (
            ' For an action with region_context_id, its region_summary is exactly '
            'region_contexts[region_context_id]. This lossless sharing does not change action identity '
            'or action_summary. Cite region_summary.* as usual; region_context_id and region_contexts '
            'are transport metadata, not evidence fields.')
    request['instruction'] += (
        ' shared_context applies to every action; checks may reference its fields as well. '
        'Every checks[].action_id MUST be an actual ID from actions, NEVER "abstain". '
        'Every checks[].field MUST resolve to a non-null field inside that action merged with '
        'shared_context: use task_summary.*, method_summary.*, budget_summary.*, region_summary.* '
        'action_summary.* or search_features.*, WITHOUT a shared_context prefix. historical_facts is NOT an action '
        'field; discuss history only in reason. An abstaining skill needs no evidence check. '
        'Sampling-seed keys such as 20261026 are integer random seeds, not dates. '
        'Return only the requested top-level keys; reason <=2000 characters, at most 24 checks, '
        'omit commentary in checks.')
    dual = sampling is not None and sampling['mode'] != 'single'
    if dual:
        request['sampling_policy'] = sampling
        if sampling.get('trigger_unvalidated_intervention'):
            request['instruction'] += (
                ' Under the supplied sampling policy, a regional intervention that changes the sampled rows '
                'is uncertain even with a large predicted ranking gap. Recent historical summaries '
                'alone cannot waive the control. The program uses the '
                'unchanged baseline as its second branch in that case. '
                'At most two distinct samples are evaluated; their metrics are averaged, not selected '
                'using full outcomes. Do not describe the baseline control as proof of safety.')
        if sampling.get('trigger_nominated_focus'):
            request['instruction'] += (
                ' A final alternative_action_id naming a feasible focus with a distinct sample '
                'runs as the second screening branch and takes precedence over the baseline control. ')
        request['instruction'] += (
            ' Also return alternative_action_id (a different legal action ID, or null) and '
            'skill_proposals: [{skill_id: "id@vN", action_id: legal ID or "abstain"}]. '
            'Give one proposal for each active coverage_repair and difference_preservation skill. '
            'Do NOT include intervention_check in skill_proposals: it reviews those proposals '
            'and expresses its final decision through action_id and alternative_action_id. '
            'Each proposal action_id must be in that skill\'s allowed_proposal_action_ids. '
            'applicable_action_ids identifies regional applicability, not the complete output space. '
            'An active skill may recommend the supplied baseline ID to retain original sampling, '
            'or abstain to express no recommendation. These are distinct conclusions. '
            'Report their actual recommendations even when they disagree; intervention_check chooses '
            'the primary action and a useful experimental contrast. Cite a check for every proposed '
            'regional action, including the alternative. Never invent confidence numbers. The program '
            'uses frozen triggers to decide whether to fit the alternative. Normal screening branches '
            'average per-method validation metrics; a labeled exploration_control retains the primary '
            'ranking and records the alternative separately. You cannot change thresholds, weights or the two-branch cap.')
        request['instruction'] += (' Partial skill abstention is an independent trigger.' if
            sampling.get('trigger_skill_abstention') else
            ' Partial skill abstention alone does NOT trigger a second branch; report abstention honestly.')
        request['instruction'] += (' Different executable skill proposals are an independent trigger.' if
            sampling.get('trigger_skill_conflict') else
            ' Different skill proposals alone do NOT trigger a second branch. Disagreement is recorded '
            'as diagnostic evidence, not a requirement to average scores. The two proposing skills '
            'may recommend the same action or baseline; different roles do not require different actions. '
            'Report genuine proposals without manufacturing agreement or disagreement.')
        request['instruction'] += (
            ' Return ONE JSON object containing ALL SIX fields: action_id, skills_used, checks, reason, '
            'alternative_action_id, skill_proposals. Never append a second JSON object.')
        if sampling['version'] == 'adaptive-coreset-v3':
            request['action_semantics'] = dict(
                baseline=('Keep the newly composed reference for this round, including its protected rows.'
                    if any(c.get('editable_sample_scope') for c in contexts) else
                    'Reuse the exact parent coreset; it may differ from the original initialization sample.'
                    if any(c.get('comparison_scope') == 'mcts_parent_anchor' for c in contexts)
                    else 'Reuse this seed original coreset.'),
                cover='Request region share p_R of Q, subject to feasibility; same regional sampler as focus.',
                focus='Request min((1+relative_increase)*p_R, p_R+max_percentage_points, 1); '
                      'read the actual numerical parameters, feasible counts and per-seed previews.',
                interpretation='Action labels are not quality or safety judgments. A small quota change can '
                    'replace many rows because regional actions resample. No historical focus trials means '
                    'unknown effect, not observed failure. Coverage sufficiency alone does not disqualify focus '
                    'for preserving verified proxy differences. Compare every feasible relevant action on evidence.',
                exploration='A development-only source budget may pair a baseline primary with a feasible focus '
                    'as a labeled experiment. That exploration branch does not enter the final ranking. '
                    'Do not pretend it was your recommendation. Validation and heldout have no forced exploration.')
    gaps = {}
    for c in action_contexts:
        region = c.get('region_summary') or {}
        rid = region.get('region_id')
        gap = region.get('coverage_gap')
        if rid and isinstance(gap, (float, int)) and math.isfinite(gap):
            gaps[rid] = gap
    if gaps:
        ordered = sorted(gaps, key=lambda rid: (-gaps[rid], str(rid)))
        request['coverage_comparison'] = dict(
            ordered_region_ids=ordered, gap_percentage_points=[round(100*gaps[r],4) for r in ordered],
            leader_margin_pp=round(100*(gaps[ordered[0]]-gaps[ordered[1]]),4) if len(ordered)>1 else None)
        request['instruction'] += (' For a coverage decision give Q and reference shares, gap in percentage points, '
            'and actual previewed replacements. If the selected region leads, compare its gap to the runner-up '
            'using coverage_comparison (distinct feasible regions; ties possible). Use up to two short sentences. '
            'Report unavailable comparisons as unknown. Cite the underlying region_summary or action_summary fields.')
    compact_request(request)
    result = dict(scores={i: 0. for i in ids}, chosen_action_id=baselines[0], adjustment_fraction=0.,
        api_available=0, calls=0, skills_used=[], checks=[], reason='', decision_status='fallback',
        skill_bundle_hash=stable_hash(bundle), request=request, response=None, error=None,
        alternative_action_id=None, skill_proposals=[])
    if len(ids) == 1:
        result.update(decision_status='baseline_only', skipped_reason='baseline_only')
        return result
    try:
        result['calls'] = 1
        response = request_fn(json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
        if inspect.isawaitable(response):
            response = await response
        result['response'] = response
        response = json.loads(response) if isinstance(response, str) else response
        # Repairs operate on the parsed object even when the adapter returns JSON text.
        if isinstance(result['response'], str):
            result['raw_response'] = result['response']
        result['response'] = response
        fields = {'action_id', 'skills_used', 'checks', 'reason'}
        if dual:
            fields |= {'alternative_action_id', 'skill_proposals'}
        if not isinstance(response, dict) or set(response) != fields:
            raise ValueError('Invalid skill decision fields')
        action = response['action_id']
        if not isinstance(action, str) or action not in [*ids, 'abstain']:
            raise ValueError('Action is not executable')
        used = response['skills_used']
        if not isinstance(used, list) or any(not isinstance(s, str) or s not in active_ids for s in used) or len(set(used)) != len(used):
            raise ValueError('Unknown/inactive skill version')
        if not isinstance(response['reason'], str) or not response['reason'].strip() or len(response['reason']) > 2000:
            raise ValueError('Bounded decision reason required')
        checks = response['checks']
        if not isinstance(checks, list) or len(checks) > 24:
            raise ValueError('Invalid evidence checks')
        # Validate against the original evidence, not transport metadata. This
        # keeps all existing dotted paths valid and excludes reference IDs.
        by_id = {c['action_id']: c for c in _expand_action_contexts(request)}
        normalized = []
        missing = []
        for index, check in enumerate(checks):
            if (not isinstance(check, dict) or set(check) not in ({'action_id', 'field'}, {'action_id', 'field', 'judgement'})
                    or not isinstance(check['action_id'], str)
                    or check['action_id'] not in by_id or not isinstance(check['field'], str)
                    or ('judgement' in check and (not isinstance(check['judgement'], str)
                        or not check['judgement'].strip() or len(check['judgement']) > 1000))):
                raise ValueError('Check cites absent evidence')
            field = _citation_path(by_id[check['action_id']], check['field'])
            if field is None:
                missing.append(dict(index=index, action_id=check['action_id'], field=check['field']))
            else:
                normalized.append({**check, 'field': field})
        if missing:
            result['invalid_citations'] = missing
            raise ValueError('Check cites absent evidence')
        checks = normalized
        if action not in ('abstain', baselines[0]) and (not used or not checks or
                not any(s.startswith('intervention_check@') for s in used) or
                not any(c['action_id'] == action for c in checks)):
            raise ValueError('Intervention requires skill execution and grounded checks')
        if dual:
            alternate, proposals = response['alternative_action_id'], response['skill_proposals']
            primary = baselines[0] if action == 'abstain' else action
            if alternate is not None and (not isinstance(alternate, str) or
                    alternate not in ids or alternate == primary):
                raise ValueError('Alternative must be a different legal action')
            expected = {s['id']+'@v'+str(s['version']): s for s in skills if s['id'] != 'intervention_check'}
            if not isinstance(proposals, list) or len(proposals) != len(expected):
                raise ValueError('One proposal per active proposing skill required')
            seen = set()
            for p in proposals:
                if (not isinstance(p, dict) or set(p) != {'skill_id', 'action_id'}
                        or not isinstance(p['skill_id'], str) or p['skill_id'] not in expected
                        or p['skill_id'] in seen or p['skill_id'] not in used
                        or not isinstance(p['action_id'], str)
                        or p['action_id'] not in expected[p['skill_id']]['allowed_proposal_action_ids']):
                    raise ValueError('Invalid or inactive skill proposal')
                seen.add(p['skill_id'])
            proposed = {p['action_id'] for p in proposals} | {primary}
            if alternate is not None:
                proposed.add(alternate)
            interventions = proposed - {'abstain', baselines[0]}
            if interventions and (not any(s.startswith('intervention_check@') for s in used)
                    or not interventions <= {c['action_id'] for c in checks}):
                raise ValueError('Every proposed intervention needs a grounded check and review')
            result.update(alternative_action_id=alternate, skill_proposals=proposals)
        result.update(chosen_action_id=baselines[0] if action == 'abstain' else action,
            adjustment_fraction=0. if action in ('abstain', baselines[0]) else 1.,
            skills_used=used, checks=checks, reason=response['reason'], api_available=1,
            decision_status='abstain' if action == 'abstain' else 'selected')
    except Exception as error:
        if getattr(error, 'budget_exhausted', False) or type(error).__name__ in ('TaskDeadlineExceeded', 'SearchBudgetExceeded'):
            raise
        result['error'] = type(error).__name__+': '+str(error)
        try:
            json.dumps(result['response'], allow_nan=False)
        except (ValueError, TypeError):
            result['response'] = repr(result['response'])
    return result


def _decision_commitment(response, request):
    """Freeze choices, not serialization or an extraneous reviewer's proposal.

    Missing, ambiguous, inactive or illegal proposals cannot be repaired without
    making a new decision. Only the known reviewer's redundant row can be dropped;
    every active proposing skill must already have exactly one legal recommendation.
    """
    if not isinstance(response, dict):
        raise ValueError('No structured decision to freeze')
    actions = request['actions']
    ids = [a['action_id'] for a in actions]
    baseline = next(a['action_id'] for a in actions if a['action_type'] == 'baseline')
    action = response.get('action_id')
    if not isinstance(action, str) or action not in [*ids, 'abstain']:
        raise ValueError('No legal primary decision to freeze')
    fixed = dict(action_id=action)
    if 'sampling_policy' not in request:
        return fixed
    if 'alternative_action_id' not in response:
        raise ValueError('Alternative decision missing')
    alternate = response['alternative_action_id']
    primary = baseline if action == 'abstain' else action
    if alternate is not None and (not isinstance(alternate, str) or alternate not in ids or alternate == primary):
        raise ValueError('No legal alternative decision to freeze')
    skills = {s['id']+'@v'+str(s['version']): s for s in request['skills']}
    expected = {k: s for k, s in skills.items() if s['id'] != 'intervention_check'}
    proposals = response.get('skill_proposals')
    if not isinstance(proposals, list):
        raise ValueError('Proposals missing')
    choices = {}
    for p in proposals:
        if not isinstance(p, dict) or not isinstance(p.get('skill_id'), str) or p['skill_id'] not in skills:
            raise ValueError('Unrecognized proposal identity')
        ident = p['skill_id']
        if skills[ident]['id'] == 'intervention_check':
            continue
        choice = p.get('action_id')
        if (ident in choices or not isinstance(choice, str) or
                choice not in expected[ident]['allowed_proposal_action_ids']):
            raise ValueError('Ambiguous or illegal proposing skill decision')
        choices[ident] = choice
    if set(choices) != set(expected):
        raise ValueError('Proposing skill decision missing')
    fixed.update(alternative_action_id=alternate,
        skill_proposals=[dict(skill_id=k, action_id=choices[k]) for k in sorted(choices)])
    return fixed


async def choose_with_skill_repair(contexts, bundle, request_fn, *, history=(), sampling=None):
    """One decision, at most one schema repair; never choose from feedback."""
    contexts = list(contexts)
    first = await choose_with_skills(contexts, bundle, request_fn, history=history, sampling=sampling)
    response = first.get('response')
    # Network/API failures do not justify an unbounded second decision attempt.
    if (first['decision_status'] != 'fallback' or not isinstance(response, dict)
            or not (first.get('error') or '').startswith('ValueError:')
            or response.get('action_id') not in [c['action_id'] for c in contexts]+['abstain']):
        return first
    try:
        fixed = _decision_commitment(response, first['request'])
    except ValueError as error:
        # Filling in missing choices or replacing illegal ones is a new decision,
        # not a schema repair. Preserve the failure for strict callers.
        return dict(first, repair_skipped_reason=str(error))
    async def repair(prompt):
        request = json.loads(prompt)
        request['schema_repair'] = dict(previous_response=response, validation_error=first['error'],
            invalid_citations=first.get('invalid_citations', []),
            immutable_decision_fields=fixed,
            instruction='Repair only invalid schema, citations or skill references using the same supplied '
                'pre-decision evidence. Preserve immutable_decision_fields exactly. Do not reconsider '
                'the action selection. skill_proposals is a mapping by skill_id: list order is '
                'irrelevant, but every frozen skill/action pair must remain unchanged. Remove '
                'any intervention_check proposal row and extra fields; it is a reviewer, not '
                'a proposing skill. Remove invalid extra checks; do not invent evidence. '
                'Return the complete corrected object with only the requested response fields.')
        value = request_fn(json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
        return await value if inspect.isawaitable(value) else value
    second = await choose_with_skills(contexts, bundle, repair, history=history, sampling=sampling)
    repaired_response = second.get('response')
    changed = False
    if isinstance(repaired_response, dict):
        try:
            changed = _decision_commitment(repaired_response, first['request']) != fixed
        except ValueError:
            changed = True
    # A failed transport/JSON parse supplies no decision to compare. Keep its
    # actual error rather than misreporting it as an attempted action change.
    if changed:
        result = dict(first, error='ValueError: Schema repair changed the frozen action proposals')
        result['rejected_repair'] = second
    else:
        result = dict(second)
    result.update(calls=first['calls']+second['calls'], repair_attempted=True,
                  initial_attempt=first)
    return result


def skill_decision(snapshot, actions, advisor, seed):
    chosen = advisor['chosen_action_id']
    if sum(a['action_id'] == chosen for a in actions) != 1:
        raise ValueError('Invalid materialized skill action')
    return dict(policy_version=VERSION, policy_snapshot_id=snapshot['snapshot_id'],
        schema_id=snapshot['schema_id'], actions=[{**a, 'api_score': 0., 'selection_probability': None,
            'score_used_for_selection': False} for a in actions], chosen_action=chosen,
        chosen_action_id=chosen, decision_seed=int(seed), epsilon=0., mode='develop_online',
        selection_probability=None, decision_source='procedural_skills',
        skill_execution={k: advisor[k] for k in ('skill_bundle_hash', 'skills_used', 'checks',
                        'reason', 'decision_status', 'error', 'alternative_action_id', 'skill_proposals')})
