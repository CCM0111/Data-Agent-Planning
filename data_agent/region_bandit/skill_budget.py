"""Execution allowances scale with seeds and branches, without changing frozen fits."""
import math


DEFAULT_BUDGET = dict(selection_wall_seconds_per_seed=43200,
    minimum_evaluation_wall_seconds=21600, evaluation_overhead_seconds=3600,
    evaluation_timeout_multiplier=2., infrastructure_retries=2, history_limit=16)


def budget_settings(settings=None):
    cfg = {**DEFAULT_BUDGET, **(settings or {})}
    if set(cfg) != set(DEFAULT_BUDGET):
        raise ValueError('Unknown skill execution budget')
    for key in ('selection_wall_seconds_per_seed', 'minimum_evaluation_wall_seconds',
                'evaluation_overhead_seconds', 'infrastructure_retries', 'history_limit'):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == 'infrastructure_retries' else 1):
            raise ValueError('Invalid skill execution budget: '+key)
    if (type(cfg['evaluation_timeout_multiplier']) not in (int, float)
            or not math.isfinite(cfg['evaluation_timeout_multiplier'])
            or cfg['evaluation_timeout_multiplier'] < 2):
        raise ValueError('Allow at least one fit timeout and one prediction timeout per attempt')
    return cfg


def evaluation_allowance(fit_budget, *, seed_count, branch_count, settings=None):
    cfg = budget_settings(settings)
    if type(seed_count) is not int or seed_count < 1 or type(branch_count) is not int or branch_count not in (1, 2):
        raise ValueError('Positive seeds and one or two branches required')
    seconds = fit_budget['timeout_seconds']
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Frozen fit timeout must be positive')
    fits = 5*seed_count*branch_count
    attempts = fits*(1+cfg['infrastructure_retries'])
    wall = math.ceil(max(cfg['minimum_evaluation_wall_seconds'],
        attempts*seconds*cfg['evaluation_timeout_multiplier']+cfg['evaluation_overhead_seconds']))
    return dict(max_unique_core_fits=fits, max_fit_attempts_with_retries=attempts,
                evaluation_wall_seconds=wall, infrastructure_retries=cfg['infrastructure_retries'],
                frozen_fit_budget=dict(fit_budget))


def batch_budget_plan(fit_budgets, *, seed_count, policy_count, mode, model,
                      selection_wall_seconds, settings=None):
    cfg = budget_settings(settings)
    if type(policy_count) is not int or policy_count not in (1, 2) or mode not in ('single', 'adaptive', 'dual'):
        raise ValueError('Invalid skill batch budget shape')
    branch_count = 1 if mode == 'single' else 2
    tasks = {task: evaluation_allowance(budget, seed_count=seed_count,
        branch_count=branch_count, settings=cfg) for task, budget in fit_budgets.items()}
    calls = len(tasks)*policy_count
    return dict(tasks=tasks, seed_count=seed_count, policy_count=policy_count,
        max_branches_per_policy_seed=branch_count, max_decision_api_calls=calls,
        max_decision_api_attempts=2*calls, max_schema_repairs_per_decision=1,
        api_timeout_seconds=model.get('timeout'), api_max_output_tokens_per_call=model.get('max_tokens'),
        max_decision_output_tokens=(2*calls*model['max_tokens'] if model.get('max_tokens') is not None else None),
        max_unique_core_fits=sum(t['max_unique_core_fits'] for t in tasks.values())*policy_count,
        max_fit_attempts_with_retries=sum(t['max_fit_attempts_with_retries'] for t in tasks.values())*policy_count,
        selection_wall_seconds_per_policy=selection_wall_seconds,
        serial_stage_allowance_seconds=policy_count*sum(selection_wall_seconds+t['evaluation_wall_seconds']
            for t in tasks.values()), new_full_fits=0,
        interpretation='Cold-cache envelope per invocation; not a runtime forecast or batch deadline. '
            'Full references and per-fit training budgets remain frozen; successful caches reduce actual work.')
