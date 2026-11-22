"""Normalize explicit, unambiguous sklearn constructor compatibility cases."""
import ast
import math
from .code_repair_contract import _constructors


def normalize_constructor_code(code):
    try:
        calls = _constructors(code)
        tree = ast.parse(code)
    except SyntaxError:
        return code, []
    changes = []
    by_line = {node.lineno: node for node in ast.walk(tree) if isinstance(node, ast.Call)}
    for entries in calls.values():
        for call in entries:
            for name, value in call['parameters'].items():
                line = value.get('line')
                if not line:
                    continue
                # Locate a unique explicit keyword, avoiding dynamic dictionaries.
                matches = [(node, kw) for node in by_line.values() for kw in node.keywords
                           if kw.arg == name and kw.value.lineno == line]
                if len(matches) != 1:
                    continue
                node, keyword = matches[0]
                raw = value.get('raw')
                if (call['name'] == 'OrdinalEncoder' and name == 'unknown_value'
                        and type(raw) is float and math.isfinite(raw) and raw.is_integer()):
                    keyword.value = ast.copy_location(ast.Constant(int(raw)), keyword.value)
                    changes.append(dict(component=call['name'], parameter=name, before=raw, after=int(raw)))
                elif (call['name'] == 'KBinsDiscretizer' and name == 'sparse'
                      and name not in call['signature'].parameters and type(raw) is bool):
                    encode = call['parameters'].get('encode', {}).get('raw')
                    explicit = any(k.arg == 'encode' for k in node.keywords)
                    if explicit and ((raw and encode == 'onehot') or (not raw and encode in ('ordinal', 'onehot-dense'))):
                        node.keywords.remove(keyword)
                        changes.append(dict(component=call['name'], parameter=name, before=raw,
                                            after='removed', equivalent_encode=encode))
    return (ast.unparse(tree) + '\n', changes) if changes else (code, [])
