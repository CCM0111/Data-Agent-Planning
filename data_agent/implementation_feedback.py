"""Bounded installed-interface facts for implementation and error repair."""
import importlib
import inspect
import json
import re

import sklearn

from .code_repair_contract import _constructors, _error_parameters


def component_interfaces(strategy, code=''):
    text = json.dumps(strategy, ensure_ascii=False) + '\n' + code
    names = set(re.findall(r'\b[A-Z][A-Za-z0-9]+\b', text))
    modules = ('preprocessing', 'impute', 'compose', 'pipeline', 'ensemble',
               'linear_model', 'tree', 'neighbors', 'svm', 'neural_network',
               'feature_selection', 'decomposition', 'naive_bayes')
    components = {}
    for module in modules:
        namespace = importlib.import_module('sklearn.' + module)
        for name in sorted(names):
            cls = getattr(namespace, name, None)
            if not inspect.isclass(cls) or name in components:
                continue
            try:
                signature = str(inspect.signature(cls))
            except (TypeError, ValueError):
                continue
            components[name] = {'constructor': signature}
            if name == 'OrdinalEncoder':
                components[name]['constraint'] = 'unknown_value must be an integer or np.nan when handle_unknown=use_encoded_value.'
            if name == 'KBinsDiscretizer':
                components[name]['constraint'] = 'Choose output format with encode. Check the installed constructor before using sparse.'
    return {'sklearn_version': sklearn.__version__, 'components': components}


def repair_diagnostic(code, error):
    allowed = _error_parameters(error)
    result = {'error': str(error), 'error_parameters': sorted(allowed),
              'components': [], 'instruction': 'Preserve unrelated configuration. Unresolved expressions require inspection.'}
    try:
        constructors = _constructors(code)
    except SyntaxError as failure:
        result['syntax'] = {'line': failure.lineno, 'offset': failure.offset, 'message': failure.msg}
        return result
    for calls in constructors.values():
        for call in calls:
            parameters = call['parameters']
            related = allowed.intersection(parameters)
            if call['name'] not in str(error) and not related:
                continue
            explicit = {key: {'value': value['value'], 'line': value.get('line')}
                        for key, value in parameters.items() if value.get('line')}
            result['components'].append({'class': call['name'], 'current_explicit_parameters': explicit,
                'supported_parameters': list(call['signature'].parameters),
                'preserve_parameters': sorted(set(explicit) - allowed)})
    return result
