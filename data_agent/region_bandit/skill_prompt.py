"""Compact the decision prompt only. Training plans and stored decisions stay intact."""
import re


def method_brief(method):
    idea = method.get('idea')
    if not isinstance(idea, dict):
        idea = {}
    row = {k: method[k] for k in ('id', 'method_id', 'name', 'family', 'implementation', 'model', 'parameters', 'fe', 'fixed_parameters', 'change_reason') if k in method}
    description = idea.get('description') or idea.get('model')
    if description:
        row['implementation'] = str(description)[:240]
    model = idea.get('model')
    if isinstance(model, str):
        match = re.search(r'[A-Za-z][A-Za-z0-9_]*(?:Regressor|Classifier|Regression|SVC|SVR)\b', model)
        row['model'] = match.group() if match else model.split('(')[0][:100]
    if idea:
        if idea.get('change_reason'):
            row['change_reason'] = idea['change_reason']
        row['fe'] = idea.get('feature_engineering', [])
        if idea.get('fixed_parameters'):
            row['fixed_parameters'] = idea['fixed_parameters']
    stages = method.get('stage_parameters') or idea.get('stage_parameters') or {}
    if stages and all(isinstance(params, dict) for params in stages.values()):
        first = next(iter(stages.values()))
        common = {k: v for k, v in first.items() if all(k in p and p[k] == v for p in stages.values())}
        row['parameters'] = {'stages': list(stages), 'shared': common}
        row['parameters'].update({s: {k: v for k, v in p.items() if k not in common}
                                  for s, p in stages.items() if p.keys() - common.keys()})
    return row


def compact_request(request):
    """Called once after evidence selection and before sending or validating."""
    for context in [request.get('shared_context', {}), *request['actions']]:
        methods = context.get('method_summary')
        if isinstance(methods, list):
            context['method_summary'] = [method_brief(m) for m in methods]
    for skill in request['skills']:
        if skill.get('evidence_ids'):
            skill['evidence_count'] = len(skill.pop('evidence_ids'))

    details = request.get('historical_path_details', {})
    for key, prefix in [('methods', 'm'), ('eda', 'e')]:
        if key not in details:
            continue
        refs = {old: f'{prefix}{i}' for i, old in enumerate(details[key], 1)}
        details[key] = {refs[k]: {field: val for field, val in v.items() if field != 'parameter_hash'}
                        for k, v in details[key].items()}
        for card in request['historical_facts']:
            if 'eda_ref' not in card:
                continue
            if key == 'eda':
                card['eda_ref'] = refs[card['eda_ref']]
            else:
                card['methods'] = [[mid, refs[r]] for mid, r in card['methods']]

    actions = request['actions']
    regional = [a for a in actions if a['action_type'] != 'baseline' and 'action_summary' in a]
    if len(regional) < 2:
        return
    common = {k: v for k, v in regional[0]['action_summary'].items()
              if all(k in a['action_summary'] and a['action_summary'][k] == v for a in regional)}
    if common:
        request['regional_action_defaults'] = common
        for a in regional:
            a['action_summary'] = {k: v for k, v in a['action_summary'].items() if k not in common}
        request['instruction'] += ' Regional action_summary inherits regional_action_defaults.'
    # Preserve every donor region and seed, including adverse side effects.
    # A table shares the baseline counts only when they match exactly.
    tables = []
    for a in actions:
        summary = dict(a.get('action_summary', {}))
        if 'cross_region_preview' not in summary:
            continue
        previews = {}
        for seed, original in summary['cross_region_preview'].items():
            preview = dict(original)
            regions = preview.get('regions', {})
            if regions and all(set(r) == {'before','after','change'} and r['after']-r['before']==r['change']
                               for r in regions.values()):
                ids = sorted(regions)
                table = {'region_ids': ids, 'before': [regions[k]['before'] for k in ids]}
                if table not in tables:
                    tables.append(table)
                preview.pop('regions')
                preview['region_table'] = tables.index(table)
                preview['after'] = [regions[k]['after'] for k in ids]
            previews[seed] = preview
        a['action_summary'] = {**summary, 'cross_region_preview': previews}
    if tables:
        request['preview_tables'] = tables
        request['instruction'] += (' Per seed, region_table indexes preview_tables (zero based). Match after to '
            'region_ids in order. Cite regions.<region ID>.before/after/change; change=after-before.')
