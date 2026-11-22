"""JSON identities and durable local checkpoints; no model execution."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
from .reward import REWARD_VERSION

VERSIONS = {'sampler_version': 'region-coverage-v1', 'reward_version': REWARD_VERSION,
            'policy_version': 'region-linear-api-v1'}
STATES = ('prepared', 'methods_frozen', 'decision_frozen', 'sampled', 'core_evaluated',
          'full_evaluated', 'feedback_ready', 'feedback_committed')


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('Nonfinite value in region-bandit artifact')
    return value


def stable_hash(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    from ..multi_proxy_io import atomic_json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, clean(value))


def environment_identity():
    return {'python': platform.python_version(), **{name: version(name) for name in
            ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib')}}


def dataset_group(task, explicit=None):
    value = explicit or task.get('dataset_group_id') or task.get('metadata', {}).get('dataset_group_id')
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Supply --dataset-group-id (shared by all source variants/targets/splits), '
                         'or freeze dataset_group_id in task metadata; a task name is not a source identity')
    return value.strip()


def checkpoint(root, state, phase, **updates):
    from ..execution_budget import account_wall_time
    if phase not in STATES:
        raise ValueError('Unknown region-bandit state')
    old = state.get('phase')
    if old in STATES and STATES.index(phase) < STATES.index(old):
        return
    state.update(updates, phase=phase, updated_at=datetime.now(timezone.utc).isoformat())
    account_wall_time(state)
    write_json(Path(root)/'state.json', state)
    with (Path(root)/'trace.jsonl').open('a') as stream:
        stream.write(json.dumps({'phase': phase, 'at': state['updated_at'], **clean(updates)},
                                ensure_ascii=False, allow_nan=False) + '\n')
