"""CSV contract for multi-proxy runs: preserve categorical tokens across subsets."""
import json
from pathlib import Path

import pandas as pd


def read_raw(path, **kwargs):
    # Only empty fields are missing. Literal category names such as NA remain text.
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[''], **kwargs)


def infer_schema(frame, categorical_columns, target, problem_type):
    categorical = set(categorical_columns)
    schema = {}
    for column in frame:
        values = frame[column].dropna()
        numeric = pd.to_numeric(values, errors='coerce').notna().all()
        leading_zero = values.astype(str).str.match(r'^[+-]?0\d').any()
        schema[column] = 'number' if numeric and not leading_zero and column not in categorical else 'text'
    schema[target] = 'text' if problem_type == 'classification' else 'number'
    return schema


def apply_schema(frame, schema):
    for column in frame:
        if schema.get(column) == 'number':
            frame[column] = pd.to_numeric(frame[column], errors='raise').astype(float)
    return frame


def read_typed(path, schema, **kwargs):
    raw = read_raw(path, **kwargs)
    if kwargs.get('chunksize'):
        return (apply_schema(chunk, schema) for chunk in raw)
    return apply_schema(raw, schema)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)
