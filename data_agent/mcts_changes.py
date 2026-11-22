"""Audit unfitted parent/child configurations after stage overrides; no fitting."""
import ast
from copy import deepcopy
import hashlib
import json


def normalized_spec(value):
    if isinstance(value, dict):
        result = {k: normalized_spec(v) for k, v in value.items()}
        if 'callable' in result:
            name = result['callable']
            if name.startswith(('generated_method.', 'preflight_generated_method.')):
                result['callable'] = 'method.' + name.split('.', 1)[1]
        return result
    if isinstance(value, list):
        return [normalized_spec(v) for v in value]
    return value


def estimator_parameter_changes(before, after):
    """Changed effective model parameters, or None for incomparable pipelines."""
    models = []
    transforms = []
    for spec in (before, after):
        steps = (spec or {}).get('parameters', {}).get('steps', [])
        if not steps or steps[-1][0] != 'model':
            return None
        models.append(steps[-1][1])
        transforms.append(normalized_spec(steps[:-1]))
    if (transforms[0] != transforms[1] or not models[0].get('class')
            or models[0]['class'] != models[1].get('class')):
        return None
    left, right = (model['parameters'] for model in models)
    if left.keys() != right.keys():
        return None
    return sorted(key for key in left if left[key] != right[key]
                  and key not in {'random_state', 'n_jobs', 'verbose'})


def coreset_blind_parameter_changes(candidate, parent, pending):
    """Check every scored branch against its own measured parent before full fitting."""
    if candidate.get('modification_direction') != 'parameters':
        return []
    changes = candidate.get('preflight_full_parameter_changes')
    if not changes:
        return []
    branches = (pending.get('skill_screening') or {}).get('branches', [])
    if pending.get('skip_parent_action_fit'):
        probe_ids = pending.get('parent_sample_probe_ids', {})
        probes = pending.get('region_probes', {})
        reference, measured = (probes.get(probe_ids.get(method['id']), {}).get('low_report')
                               for method in (parent, candidate))
        action_report = candidate.get('low_report') or {}
        if (not measured or not measured.get('validation_id')
                or action_report.get('validation_id') != measured['validation_id']
                or normalized_spec(action_report.get('pipeline_spec')) != normalized_spec(measured.get('pipeline_spec'))):
            return []
        reports = [(reference, measured)]
    elif branches:
        reports = [(branch.get('reports', {}).get(parent['id']), branch.get('reports', {}).get(candidate['id']))
                   for branch in branches]
    else:
        anchor = pending.get('region_anchor') or parent
        reports = [(anchor.get('low_report'), candidate.get('low_report'))]
    for reference, measured in reports:
        if not reference or not measured:
            return []
        if any(not reference.get(key) or reference[key] != measured.get(key)
               for key in ('fit_train_sha256', 'validation_id')):
            return []
        if estimator_parameter_changes(reference.get('pipeline_spec'), measured.get('pipeline_spec')) != []:
            return []
    return changes


def full_comparison_plan(ranked, nominated, observed, stage_key):
    """Represent each proven full configuration once; unknown configurations stay eligible.

    All candidates share the parent, full training rows, seed and validation split.
    Preserve the screening nomination when multiple proposals represent one configuration.
    """
    previous = {}
    for candidate in observed:
        key = stage_key(candidate, 'full_train')
        if key is not None:
            previous.setdefault(key, candidate['id'])

    by_id = {c['id']: c for c in ranked}
    ordered = list(dict.fromkeys([*nominated, *by_id]))
    eligible, duplicates = [], {}
    for ident in ordered:
        key = stage_key(by_id[ident], 'full_train')
        if key is not None and key in previous:
            duplicates[ident] = previous[key]
            continue
        eligible.append(ident)
        if key is not None:
            previous[key] = ident

    primary = [i for i in nominated if i in eligible]
    if not primary:
        primary = eligible[:1]
    return dict(primary_ids=primary, eligible_ids=eligible, duplicate_full_of=duplicates)


def same_low_on_scored_branches(first, second, pending):
    """Compare actual post-binding configurations and fit/evaluation populations per branch."""
    branches = (pending.get('skill_screening') or {}).get('branches', [])
    if branches:
        pairs = [(b.get('reports', {}).get(first['id']), b.get('reports', {}).get(second['id']))
                 for b in branches]
    else:
        pairs = [(first.get('low_report'), second.get('low_report'))]
    for left, right in pairs:
        if not left or not right or not left.get('pipeline_spec'):
            return False
        if any(not left.get(key) or left[key] != right.get(key)
               for key in ('fit_train_sha256', 'validation_id')):
            return False
        if normalized_spec(left['pipeline_spec']) != normalized_spec(right.get('pipeline_spec')):
            return False
    return True


def apply_parameter(specification, path, value):
    """Apply a sklearn-style path to its serialized unfitted configuration."""
    name, *rest = path.split('__')
    params = specification.get('parameters', {})
    if not rest:
        if name not in params:
            raise ValueError('Declared incremental parameter absent from parent: ' + path)
        params[name] = deepcopy(value)
        return
    child = params.get(name)
    if child is None:
        for collection in ('steps', 'transformers'):
            for entry in params.get(collection, []):
                if entry[0] == name:
                    child = entry[1]
                    break
    if not isinstance(child, dict) or 'parameters' not in child:
        raise ValueError('Declared incremental path does not identify a parent estimator: ' + path)
    apply_parameter(child, '__'.join(rest), value)


def differences(before, after, path=''):
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(set(before) | set(after)):
            result.extend(differences(before.get(key), after.get(key), f'{path}.{key}'.lstrip('.')))
        return result
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [change for i, (a, b) in enumerate(zip(before, after))
                for change in differences(a, b, f'{path}[{i}]')]
    return [] if before == after else [{'path': path, 'before': before, 'after': after}]


def helper_source(code):
    tree = ast.parse(code)
    result = {}
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name != 'build_method':
                result[node.name] = ast.dump(node, include_attributes=False)
            else:
                for nested in ast.walk(node):
                    if nested is not node and isinstance(nested, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        result['build_method.' + nested.name] = ast.dump(nested, include_attributes=False)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            key = ','.join(ast.dump(t, include_attributes=False) for t in targets)
            result['module_state:' + key] = ast.dump(node, include_attributes=False)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bindings = ','.join(n.asname or n.name for n in node.names)
            result['module_import:' + bindings] = ast.dump(node, include_attributes=False)
        elif not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                  and isinstance(node.value.value, str)):
            result['module_statement:' + str(index)] = ast.dump(node, include_attributes=False)
    return result


def _configuration_entries(specification, prefix=''):
    """Name pipeline components in repair paths instead of opaque list indices."""
    if not isinstance(specification, dict) or 'parameters' not in specification:
        return {prefix: specification}
    result = {prefix + '__class' if prefix else '__class': specification.get('class')}
    for name, value in specification['parameters'].items():
        path = prefix + '__' + name if prefix else name
        if name in {'steps', 'transformers'} and isinstance(value, list):
            result[path + '__order'] = [entry[0] for entry in value]
            for entry in value:
                component = prefix + '__' + entry[0] if prefix else entry[0]
                result.update(_configuration_entries(entry[1], component))
                if len(entry) > 2:
                    result[component + '__columns'] = entry[2]
        elif isinstance(value, dict) and 'parameters' in value:
            result.update(_configuration_entries(value, path))
        else:
            result[path] = value
    return result


def _declarative_pipeline(value):
    """Only certify standard estimators without opaque callbacks."""
    if isinstance(value, list):
        return all(_declarative_pipeline(v) for v in value)
    if isinstance(value, dict):
        if 'callable' in value:
            # Common OneHotEncoder dtype; a generated function with the same
            # qualified name across versions can have different behavior.
            return value == {'callable': 'numpy.float64'} or value == {'callable': 'numpy.float32'}
        if 'class' in value and not str(value['class']).startswith('sklearn.'):
            return False
        return all(_declarative_pipeline(v) for v in value.values())
    return True


def _bounded_value(value, limit=400):
    serialized = json.dumps(value, ensure_ascii=False, default=str)
    return value if len(serialized) <= limit else {
        'preview': serialized[:limit], 'truncated': True, 'serialized_characters': len(serialized)}


def _literal_location(code, parameter_path, actual):
    """Only identify a unique matching literal; dynamic values have no location."""
    name = parameter_path.split('__')[-1]
    matches = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.keyword) and node.arg == name:
            try:
                literal = ast.literal_eval(node.value)
                if type(literal) is type(actual) and literal == actual:
                    matches.append({'line': node.lineno, 'column': node.col_offset + 1,
                                    'basis': 'unique matching constructor/set_params keyword literal'})
            except (ValueError, TypeError, SyntaxError):
                pass
    return matches[0] if len(matches) == 1 else None


def _repair_differences(phase, expected, actual, code, prefix=''):
    before = _configuration_entries(expected, prefix)
    after = _configuration_entries(actual, prefix)
    changes = []
    for path in sorted(set(before) | set(after)):
        if path in before and path in after and before[path] == after[path]:
            continue
        changes.append({'stage': phase, 'parameter_path': path,
                        'expected': _bounded_value(before.get(path)),
                        'actual': _bounded_value(after.get(path)),
                        'expected_present': path in before, 'actual_present': path in after,
                        'expected_source': 'parent effective configuration plus declared stage overrides',
                        'code_location': _literal_location(code, path, after.get(path))})
    return changes


def repair_audit_feedback(audit, code, max_differences=12):
    """A bounded, fresh implementation-only view; never forward old audit errors."""
    if not audit or audit.get('child_code_sha256') != hashlib.sha256(code.encode()).hexdigest():
        return {'status': 'unavailable_or_stale', 'differences': []}
    changes = audit.get('repair_differences', [])
    return {'status': 'current_source_audit', 'child_code_sha256': audit['child_code_sha256'],
            'scope': audit['scope'], 'differences': changes[:max_differences],
            'omitted_differences': max(0, len(changes) - max_differences)}


def audit_changes(parent_stages, child_stages, stage_parameters, direction, parent_code, child_code,
                  parent_idea=None, child_idea=None):
    audit = {'stages': {}, 'violations': [], 'repair_differences': [],
             'child_code_sha256': hashlib.sha256(child_code.encode()).hexdigest(),
             'scope': 'Unfitted effective configurations; source differences do not prove semantic equivalence.'}
    for phase in ('coreset', 'full_train'):
        before = normalized_spec(parent_stages.get(phase))
        after = normalized_spec(child_stages.get(phase))
        if before is None or after is None:
            audit['stages'][phase] = {'status': 'parent_or_child_configuration_unavailable'}
            continue
        expected = deepcopy(before)
        audit['stages'][phase] = {'status': 'compared', 'actual_changes': differences(before, after)}
        if direction == 'parameters':
            try:
                for path, value in stage_parameters[phase].items():
                    apply_parameter(expected, path, value)
                unexplained = differences(expected, after)
                audit['stages'][phase]['undeclared_changes'] = unexplained
                if unexplained:
                    audit['repair_differences'].extend(_repair_differences(phase, expected, after, child_code))
                    audit['violations'].append(f'{phase}: undeclared configuration changes: ' +
                                               ', '.join(d['path'] for d in unexplained[:8]))
            except ValueError as error:
                audit['violations'].append(str(error))
                audit['repair_differences'].append({
                    'stage': phase, 'parameter_path': path, 'expected': _bounded_value(value),
                    'actual': None, 'actual_present': False,
                    'expected_source': 'declared stage override; path absent from parent configuration',
                    'code_location': None})
        elif parent_idea is not None and child_idea is not None:
            # Existing structured idea fields identify which component was
            # explicitly replanned. Text changes are declarations, not proof.
            for path, value in stage_parameters[phase].items():
                try:
                    apply_parameter(expected, path, value)
                except ValueError:
                    pass  # Replacing the planned model/FE may remove old paths.
            old_steps = dict(expected.get('parameters', {}).get('steps', []))
            new_steps = dict(after.get('parameters', {}).get('steps', []))
            if direction == 'features' and parent_idea.get('model') == child_idea.get('model'):
                if old_steps.get('model') != new_steps.get('model'):
                    audit['violations'].append(f'{phase}: feature strategy changed undeclared model configuration')
                    audit['repair_differences'].extend(_repair_differences(
                        phase, old_steps.get('model'), new_steps.get('model'), child_code, 'model'))
            if direction == 'model' and parent_idea.get('feature_engineering') == child_idea.get('feature_engineering'):
                old_fe = [(n, v) for n, v in expected.get('parameters', {}).get('steps', []) if n != 'model']
                new_fe = [(n, v) for n, v in after.get('parameters', {}).get('steps', []) if n != 'model']
                if old_fe != new_fe:
                    audit['violations'].append(f'{phase}: model strategy changed undeclared preprocessing')
                    audit['repair_differences'].extend(_repair_differences(
                        phase, {'parameters': {'steps': old_fe}},
                        {'parameters': {'steps': new_fe}}, child_code))
    helpers_before, helpers_after = helper_source(parent_code), helper_source(child_code)
    audit['changed_helpers'] = sorted(k for k in set(helpers_before) | set(helpers_after)
                                      if helpers_before.get(k) != helpers_after.get(k))
    if (direction == 'features' and all(
            audit['stages'][phase].get('actual_changes') == [] and
            (child_stages[phase] or {}).get('parameters', {}).get('steps') and
            _declarative_pipeline(child_stages[phase]) for phase in ('coreset', 'full_train'))):
        audit['violations'].append(
            'Feature edit has no effect on this task: the effective parent and child pipelines '
            'have identical columns, transforms and estimator settings at both stages, with no '
            'custom callback. Check whether the proposed columns already reach the parent model. '
            'Implement the declared difference if feasible; do not change unrelated settings.')
        for path, columns in _configuration_entries(child_stages['full_train']).items():
            if path.endswith('__columns') and len(audit['repair_differences']) < 12:
                audit['repair_differences'].append(dict(stage='full_train', parameter_path=path,
                    actual=_bounded_value(columns), expected_source='identical parent and child column selection'))
    if direction == 'parameters' and audit['changed_helpers']:
        audit['violations'].append('Parameter-only strategy changed custom helper implementation: ' +
                                   ', '.join(audit['changed_helpers']))
        for name in audit['changed_helpers']:
            audit['repair_differences'].append({
                'stage': 'shared_code', 'parameter_path': None, 'code_symbol': name,
                'expected': 'unchanged parent helper/module implementation',
                'actual': 'implementation changed', 'expected_source': 'parameter-only strategy contract',
                'code_location': None})
    fingerprint = {'stages': normalized_spec(child_stages), 'helpers': helpers_after}
    audit['configuration_fingerprint'] = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    return audit
