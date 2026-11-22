"""Search admission and mandatory initial exploration; no model/LLM execution.

MCS membership remains statistical evidence. Admission may add exploration
candidates without altering p-values, initial values, visits or rewards.
"""
import math

DEFAULT_ADMISSION = {'min_active_methods': 2, 'min_expansions_per_method': 1}


def normalize_admission_config(value=None):
    if value is not None and not isinstance(value, dict):
        raise ValueError('mcts_admission must be an object')
    if set(value or {}) - set(DEFAULT_ADMISSION):
        raise ValueError('Unknown mcts_admission setting')
    config = {**DEFAULT_ADMISSION, **(value or {})}
    minimum = config['min_active_methods']
    expansions = config['min_expansions_per_method']
    if type(minimum) is not int or not 2 <= minimum <= 3:
        raise ValueError('min_active_methods must be 2 or 3')
    if type(expansions) is not int or expansions < 1:
        raise ValueError('min_expansions_per_method must be a positive integer')
    return config


def build_admission(selection, initial_values, config=None):
    """Produce an independent search policy using stage-one evidence only."""
    cfg = normalize_admission_config(config)
    ids = sorted(selection['method_ids'], key=str)
    if len(ids) != 3 or len(set(ids)) != 3 or set(initial_values) != set(ids):
        raise ValueError('Admission requires exactly three unique scored method IDs')
    survivors = set(selection['survivor_method_ids'])
    if not survivors <= set(ids):
        raise ValueError('MCS survivors contain unknown methods')
    finite = lambda value: value is not None and math.isfinite(value)
    valid_values = all(finite(initial_values[m]) for m in ids)
    order = sorted(ids, key=lambda m: (not finite(initial_values[m]),
        -initial_values[m] if finite(initial_values[m]) else 0, str(m)))
    reliable = selection['selection_status'] == 'completed' and bool(survivors) and valid_values
    reasons = {}
    if reliable:
        active = set(survivors)
        reasons.update({m: 'mcs_survivor' for m in active})
        for method in order:
            if len(active) >= cfg['min_active_methods']:
                break
            if method not in active:
                active.add(method)
                reasons[method] = 'minimum_active_backfill'
    else:
        active = set(ids)
        reasons.update({m: 'screening_unavailable_keep_all' for m in ids})
    active_order = [m for m in order if m in active]
    methods = [{'method_id': m, 'mcs_survivor': m in survivors,
                'mcts_active': m in active,
                'admission_reason': reasons.get(m, 'mcs_excluded'),
                'initial_value': float(initial_values[m]) if finite(initial_values[m]) else None,
                'minimum_expansions': cfg['min_expansions_per_method'] if m in active else 0}
               for m in ids]
    return {
        'policy_version': '1.0', 'policy': 'mcs_with_minimum_active_backfill',
        **cfg, 'source_selection_status': selection['selection_status'],
        'mcs_survivor_method_ids': [m for m in ids if m in survivors],
        'mcs_inactive_method_ids': [m for m in ids if m not in survivors],
        'active_method_ids': active_order,
        'inactive_method_ids': [m for m in ids if m not in active],
        'backfilled_method_ids': [m for m in active_order if reasons[m] == 'minimum_active_backfill'],
        'methods': methods,
        'initial_expansion_queue': [m for _ in range(cfg['min_expansions_per_method']) for m in active_order],
        'minimum_expansion_budget': len(active) * cfg['min_expansions_per_method'],
        'search_execution_status': 'not_started',
        'dispatch_ready': valid_values,
        'dispatch_reason': 'Finite initial values available; search executor not connected' if valid_values else 'Undefined initial values; preserve all candidates but resolve scoring before search',
        'budget_policy': 'Reserve all outstanding minimum expansions before score-based selection; insufficient budget raises an error',
        'scope': 'Admission and scheduling only; no MCTS visits, accumulated rewards, expansions, training or API calls performed',
    }


def select_expansion_method(admission, completed_expansions, remaining_budget, search_scores=None):
    """Select a method for a future search executor, enforcing initial exploration.

    Counts represent actual completed expansions supplied by the caller. This
    function does not increment them or execute work. Once all minimums are met,
    optional search_scores (e.g. UCT computed by the caller) select only active
    methods; without them None hands control back to the normal search policy.
    """
    if not admission['dispatch_ready']:
        raise ValueError('Cannot dispatch search with undefined initial values')
    if type(remaining_budget) is not int or remaining_budget < 0:
        raise ValueError('remaining_budget must be a nonnegative integer')
    active = admission['active_method_ids']
    known = {item['method_id'] for item in admission['methods']}
    if not set(completed_expansions) <= known:
        raise ValueError('Unknown method in expansion counts')
    if any(type(v) is not int or v < 0 for v in completed_expansions.values()):
        raise ValueError('Completed expansion counts must be nonnegative integers')
    minimum = admission['min_expansions_per_method']
    deficits = {m: max(0, minimum-completed_expansions.get(m, 0)) for m in active}
    if remaining_budget < sum(deficits.values()):
        raise ValueError('Insufficient search budget to honor all mandatory initial expansions')
    pending = [m for m in active if deficits[m]]
    if pending:
        # Round-robin minimums, with initial-value order breaking ties.
        return min(pending, key=lambda m: completed_expansions.get(m, 0))
    if not remaining_budget or search_scores is None:
        return None
    if any(m not in search_scores or not math.isfinite(search_scores[m]) for m in active):
        raise ValueError('Finite search scores are required for every active method')
    return min(active, key=lambda m: (-search_scores[m], str(m)))
