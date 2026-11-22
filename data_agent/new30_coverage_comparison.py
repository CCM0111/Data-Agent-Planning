"""Paired coresets for fixed marginal-coverage settings versus random."""
import numpy as np

from .coreset import strata, stratified_sample
from .density_hardness_ablation import group_percentiles, SEEDS
from .edge_penalty_sampling import component_groups
from .marginal_coverage import coverage_sample
from .multi_proxy_io import atomic_json

SCHEME = 'marginal_coverage_random_comparison'
DUAL_SCHEME = 'marginal_coverage_dual_random_comparison'
COVERAGE_MODE = 'coverage_a30_b60_l200'
SECOND_MODE = 'coverage_a30_b30_l060'
PARAMETERS = {COVERAGE_MODE: (.3, .6, 2.), SECOND_MODE: (.3, .3, .6)}
ARMS = tuple((mode, seed) for mode in (COVERAGE_MODE, 'random') for seed in SEEDS)


def comparison_arms(scheme=SCHEME):
    if scheme == 'marginal_coverage_only':
        return ((COVERAGE_MODE, SEEDS[0]),)
    if scheme not in (SCHEME, DUAL_SCHEME):
        raise ValueError('Unknown coverage comparison scheme')
    modes = (COVERAGE_MODE, SECOND_MODE, 'random') if scheme == DUAL_SCHEME else (COVERAGE_MODE, 'random')
    return tuple((mode, seed) for mode in modes for seed in SEEDS)


def build_selections(out, qframe, qpos, importance, graph, rep, task, cfg, size, *, validation_errors=None, context=None):
    import joblib
    from .multi_proxy import split_quotas
    gmm = joblib.load(out/'density_gmm/fitted.joblib')
    if gmm.n_components != 5 or cfg.gmm_components != 5:
        raise ValueError('The new30 comparison requires fixed K=5')
    components = np.concatenate([
        gmm.predict_proba(rep.transform(qframe.iloc[i:i+cfg.batch_size])).argmax(1)
        for i in range(0, len(qframe), cfg.batch_size)])
    class_groups = np.zeros(len(qpos), dtype=int)
    class_quota = np.array([size])
    if task['problem_type'] == 'classification':
        class_groups, _ = strata(qframe[task['target']], 'classification')
        class_quota = split_quotas(np.bincount(class_groups), size, cfg.split_seed+3)
    groups, quota, counts = component_groups(
        class_groups, class_quota, components, 5, cfg.split_seed+10)
    d = group_percentiles(importance[-1], groups)
    h = group_percentiles(np.asarray(importance[:5]).mean(0), groups)
    np.savez_compressed(out/'coverage_selection_inputs.npz', train_positions=qpos,
                        components=components, groups=groups, quota=quota,
                        density=d, hardness=h)
    random_groups, random_quota = class_groups, class_quota
    if task['problem_type'] == 'regression':
        random_groups, _ = strata(qframe[task['target']], 'regression',
                                  cfg.regression_bins, max_strata=size)
        random_quota = split_quotas(np.bincount(random_groups), size, cfg.split_seed+3)
    results = []
    arms = comparison_arms(cfg.proxy_selection_scheme)
    for j, (mode, seed) in enumerate(arms):
        if validation_errors is not None:
            from .guided_coreset import guided_reference
            chosen,guide=guided_reference(out,context,qframe,dict(density=d,hardness=h,components=components,groups=groups,quota=quota),graph,validation_errors,seed)
            atomic_json(out/'validation_guidance.json',guide)
            selected_groups=groups
            selected_quota=np.bincount(groups[chosen],minlength=len(quota))
            info=dict(mechanism='protected_mmd10_error5_coverage85',sampling_seed=seed,protected_rows=len(guide['protected']))
        elif mode in PARAMETERS:
            a, beta, reward = PARAMETERS[mode]
            chosen, info = coverage_sample(d, h, graph, groups, quota, seed,
                'probability', gamma=reward, rho=1., beta=beta, density_coefficient=a)
            selected_groups, selected_quota = groups, quota
            info.update(candidate_group_counts=counts.tolist(), n_components=5,
                group_definition='class_id * 5 + component_id' if task['problem_type']=='classification' else 'component_id')
        else:
            chosen = stratified_sample(random_groups, random_quota, seed)
            selected_groups, selected_quota = random_groups, random_quota
            info = {'mechanism':'stratified_random', 'sampling_seed':seed,
                'sampling':'uniform without replacement within classes / target quantile strata',
                'group_definition':'class' if task['problem_type']=='classification' else 'target_quantile',
                'group_allocation':random_quota.tolist()}
        actual = np.bincount(selected_groups[chosen], minlength=len(selected_quota))
        if len(np.unique(chosen)) != size or not np.array_equal(actual, selected_quota):
            raise ValueError('Selection size, uniqueness or quota mismatch')
        if validation_errors is None and task['problem_type']=='classification' and not np.array_equal(
                np.bincount(class_groups[chosen], minlength=len(class_quota)), class_quota):
            raise ValueError('Class totals differ between comparison arms')
        info.update(final_size=size, class_allocation=class_quota.tolist(),
                    final_group_counts=actual.tolist())
        atomic_json(out/f'coreset_{j}_selection_order.json',
                    {'q_indices':chosen.tolist(), 'train_indices':qpos[chosen].tolist()})
        atomic_json(out/f'coreset_{j}_selection.json', info)
        results.append(np.sort(chosen))
    atomic_json(out/'coverage_comparison_design.json', {
        'arms':[{'coreset_id':j,'mechanism':mode,'seed':seed} for j,(mode,seed) in enumerate(arms)],
        'parameters': {m:dict(zip(('a','beta','lambda'), PARAMETERS[m])) for m in dict(arms) if m != 'random'},
        **({'a':.3, 'beta':.6, 'lambda':2.} if cfg.proxy_selection_scheme in (SCHEME, 'marginal_coverage_only') else {}),
        'rho':1., 'gmm_components':5,
        'fixed_model_seed':cfg.model_seed,
        'shared':'fresh actual DeepSeek plan/code, five proxies, P/Q/V_core, graph, full reference, budgets, model seeds',
        'random_policy':None if cfg.proxy_selection_scheme == 'marginal_coverage_only' else 'class stratified for classification; target quantile stratified for regression; no GMM quotas',
        'independent_unit':'task; three selection seeds are paired repeated measurements'})
    return results
