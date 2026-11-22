"""Conservative source checks for error-only repairs; never execute generated code.

This catches unrelated changes to resolvable sklearn constructor parameters.
It is not a proof of program equivalence; dynamic values and custom FE bodies
still require the existing execution and effective-configuration checks.
"""
import ast
import importlib
import inspect
import math
import re
from collections import defaultdict

from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import validate_parameter_constraints


_CONTAINERS = {'Pipeline', 'ColumnTransformer', 'FeatureUnion'}
_STRUCTURAL = {'steps', 'transformers', 'transformer_list'}


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return repr(value)


def _resolve(node, bindings, seen=()):
    if isinstance(node, ast.Name) and node.id in bindings and node.id not in seen:
        return _resolve(bindings[node.id], bindings, (*seen, node.id))
    return node


def _expression(node, bindings, aliases):
    node = _resolve(node, bindings)
    try:
        value = ast.literal_eval(node)
        return {'token': repr(value), 'value': _json_value(value), 'literal': True, 'raw': value}
    except (ValueError, TypeError, SyntaxError):
        text = ast.unparse(node)
        first = text.split('.', 1)[0]
        if first in aliases and (text == first or text.startswith(first + '.')):
            text = aliases[first] + text[len(first):]
        return {'token': text, 'value': {'expression': text}, 'literal': False}


def _constructors(code):
    tree = ast.parse(code)
    aliases, assigned = {}, defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for name in node.names:
                aliases[name.asname or name.name] = node.module + '.' + name.name
        elif isinstance(node, ast.Import):
            for name in node.names:
                aliases[name.asname or name.name.split('.')[0]] = name.name if name.asname else name.name.split('.')[0]
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned[target.id].append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            assigned[node.target.id].append(node.value)
    bindings = {name: values[0] for name, values in assigned.items() if len(values) == 1}
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    calls = defaultdict(list)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        symbol = ast.unparse(node.func)
        head = symbol.split('.', 1)[0]
        symbol = aliases.get(head, head) + symbol[len(head):]
        if not symbol.startswith('sklearn.') or '.' not in symbol:
            continue
        module_name, name = symbol.rsplit('.', 1)
        try:
            constructor = getattr(importlib.import_module(module_name), name)
            if not inspect.isclass(constructor) or not issubclass(constructor, BaseEstimator):
                continue
            signature = inspect.signature(constructor)
        except (ImportError, AttributeError, TypeError, ValueError):
            continue
        canonical = constructor.__module__ + '.' + constructor.__qualname__
        parameters = {}
        for key, parameter in signature.parameters.items():
            if parameter.default is not inspect.Parameter.empty and key not in _STRUCTURAL:
                value = parameter.default
                parameters[key] = {'token': repr(value), 'value': _json_value(value),
                                   'literal': True, 'raw': value, 'line': None}
        positional = [p.name for p in signature.parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        provided = [(key, value, value.lineno) for key, value in zip(positional, node.args)]
        for keyword in node.keywords:
            if keyword.arg:
                provided.append((keyword.arg, keyword.value, keyword.lineno))
            else:
                expanded = _resolve(keyword.value, bindings)
                if isinstance(expanded, ast.Dict):
                    provided.extend((key.value, value, keyword.lineno)
                                    for key, value in zip(expanded.keys, expanded.values)
                                    if isinstance(key, ast.Constant) and isinstance(key.value, str))
        for key, value, line in provided:
            if key not in _STRUCTURAL:
                parameters[key] = {**_expression(value, bindings, aliases), 'line': line}
        location = []
        ancestor = node
        while ancestor in parents:
            ancestor = parents[ancestor]
            if isinstance(ancestor, ast.Tuple) and ancestor.elts and isinstance(ancestor.elts[0], ast.Constant):
                location.append(str(ancestor.elts[0].value))
            if isinstance(ancestor, ast.Assign):
                location.extend(ast.unparse(target) for target in ancestor.targets)
                break
        calls[canonical].append({'location': tuple(location), 'class': constructor, 'name': name, 'line': node.lineno,
                                 'parameters': parameters, 'signature': signature})
    return calls


def _error_parameters(error):
    text = str(error)
    names = set()
    # Match actual exception clauses, not keywords merely printed in traceback
    # source lines. A constructor line may mention many valid unrelated values.
    for pattern in (
        r"unexpected keyword argument ['\"]([A-Za-z_]\w*)['\"]",
        r"(?:The|the) ['\"]([A-Za-z_]\w*)['\"] parameter",
        r"Invalid parameter ['\"]([A-Za-z_]\w*)['\"]",
        r"\b([A-Za-z_]\w*) (?:must be|must satisfy|has to be|should be)",
    ):
        names.update(re.findall(pattern, text))
    lower = text.lower()
    if 'sparse' in lower and ('dense' in lower or 'center' in lower):
        names.update({'sparse', 'sparse_output', 'with_mean'})
    if 'unknown categor' in lower:
        names.update({'handle_unknown', 'unknown_value', 'encoded_missing_value'})
    if 'strategy' in lower and ('non-numeric' in lower or 'cannot use' in lower):
        names.add('strategy')
    # Known sklearn keyword rename; removing an unsupported old keyword can
    # require introducing the matching current interface.
    for old, new in [('sparse', 'sparse_output'), ('base_estimator', 'estimator')]:
        if old in names:
            names.add(new)
    return names


def _invalid_old_parameter(call, name, value):
    if name not in call['signature'].parameters:
        return True
    constraints = getattr(call['class'], '_parameter_constraints', {})
    if name in constraints and value.get('literal'):
        try:
            validate_parameter_constraints({name: constraints[name]}, {name: value['raw']}, call['name'])
        except (TypeError, ValueError):
            return True
    return False


def audit_error_only_repair(original_code, repaired_code, error, *, idea=None, allowed_restorations=()):
    """Reject demonstrably unrelated constructor drift, preserving repair freedom.

    ``allowed_restorations`` may contain fresh effective-audit entries with
    parameter_path, expected and a unique code_location.line in original_code.
    These authorize restoring that located wrong literal to its frozen value.
    """
    allowed = _error_parameters(error)
    audit = {'status': 'checked', 'violations': [], 'parameter_changes': [],
             'allowed_error_parameters': sorted(allowed),
             'limitations': 'Static constructor checks only; dynamic expressions and custom function semantics are not proved equivalent.'}
    try:
        before = _constructors(original_code)
    except SyntaxError:
        audit['status'] = 'original_syntax_unavailable'
        return audit  # Syntax repair still has the normal idea/runtime contract.
    after = _constructors(repaired_code)
    error_text = str(error).lower()
    compatible_components = set()
    if any(term in error_text for term in ('nan', 'missing value')):
        compatible_components.update({'SimpleImputer', 'MissingIndicator'})
    if any(term in error_text for term in ('non-numeric', 'could not convert', 'unknown categor')):
        compatible_components.update({'OneHotEncoder', 'OrdinalEncoder', 'SimpleImputer'})
    if any(term in error_text for term in ('sparse', 'dense', 'shape', 'dimension')):
        compatible_components.add('FunctionTransformer')
    for symbol in sorted(set(before) | set(after)):
        old_calls, new_calls = before.get(symbol, []), after.get(symbol, [])
        name = (old_calls or new_calls)[0]['name']
        if len(old_calls) != len(new_calls) and name not in _CONTAINERS:
            compatibility = name in compatible_components
            audit['parameter_changes'].append({'constructor': symbol, 'component_count_before': len(old_calls),
                                               'component_count_after': len(new_calls), 'allowed': compatibility})
            if not compatibility:
                audit['violations'].append(f'Error-only repair changed {name} component count without an input-compatibility error')
        # Match named pipeline positions first. Inserting an imputer before
        # another branch must not compare the new step against that branch.
        remaining = list(new_calls)
        pairs, unmatched = [], []
        for old_call in old_calls:
            matches = [c for c in remaining if c['location'] == old_call['location']]
            if old_call['location'] and len(matches) == 1:
                pairs.append((old_call, matches[0]))
                remaining.remove(matches[0])
            else:
                unmatched.append(old_call)
        pairs.extend(zip(unmatched, remaining))
        for index, (old_call, new_call) in enumerate(pairs):
            old, new = old_call['parameters'], new_call['parameters']
            for parameter in sorted(set(old) | set(new)):
                a, b = old.get(parameter), new.get(parameter)
                if a is not None and b is not None and a['token'] == b['token']:
                    continue
                restoration = any(
                    item.get('parameter_path', '').split('__')[-1] == parameter
                    and (item.get('code_location') or {}).get('line') is not None
                    and (item.get('code_location') or {}).get('line') == (a or {}).get('line')
                    and b is not None and b.get('literal') and type(b['value']) is type(item.get('expected'))
                    and b['value'] == item.get('expected')
                    for item in allowed_restorations)
                comparable = a is not None and b is not None and a.get('literal') and b.get('literal')
                permitted = (not comparable or parameter in allowed or restoration
                             or a is not None and _invalid_old_parameter(old_call, parameter, a))
                change = {'constructor': symbol, 'occurrence': index, 'parameter': parameter,
                          'before': a['value'] if a else {'absent': True},
                          'after': b['value'] if b else {'absent': True},
                          'line': (a or {}).get('line'), 'allowed': bool(permitted),
                          'comparison': 'concrete_values' if comparable else 'unresolved_expression_or_interface'}
                audit['parameter_changes'].append(change)
                if not permitted:
                    audit['violations'].append(f'Error-only repair changed unrelated {name}.{parameter}: '
                                               f'{change["before"]!r} -> {change["after"]!r}')
    return audit
