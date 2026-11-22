"""Four paired coresets from one frozen task-level DeepSeek plan and geometry."""
import numpy as np
from .edge_penalty_sampling import edge_sample, WEIGHT_FLOOR
from .density_hardness_ablation import group_percentiles, SEEDS
from .component_coreset import component_groups
from .coreset import strata
from .multi_proxy import split_quotas

ARMS = tuple(('probability', seed) for seed in SEEDS) + (('greedy', None),)


def build_selections(out, qframe, qpos, importance, graph, rep, task, cfg, size):
    import joblib
    from .multi_proxy_io import atomic_json
    gmm = joblib.load(out/'density_gmm/fitted.joblib')
    if gmm.n_components != cfg.gmm_components:
        raise ValueError('GMM differs from DeepSeek plan')
    components = np.concatenate([gmm.predict_proba(rep.transform(qframe.iloc[i:i+cfg.batch_size])).argmax(1)
                                 for i in range(0, len(qframe), cfg.batch_size)])
    classes = np.zeros(len(qpos), dtype=int)
    class_quota = np.array([size])
    if task['problem_type'] == 'classification':
        classes, _ = strata(qframe[task['target']], 'classification')
        class_quota = split_quotas(np.bincount(classes), size, cfg.split_seed+3)
    groups, quota, counts = component_groups(classes, class_quota, components, gmm.n_components, cfg.split_seed+10)
    d = group_percentiles(importance[-1], groups)
    h = group_percentiles(np.asarray(importance[:5]).mean(0), groups)
    base = 1 + .3*(2*d-1) + .1*h
    np.savez_compressed(out/'edge_selection_inputs.npz', train_positions=qpos, components=components,
                        groups=groups, quota=quota, density=d, hardness=h, initial_weights=base)
    results = []
    for j, (mode, seed) in enumerate(ARMS):
        chosen, info = edge_sample(base, graph, groups, quota, .2, seed, mechanism=mode)
        actual = np.bincount(groups[chosen], minlength=len(quota))
        if len(np.unique(chosen)) != size or not np.array_equal(actual, quota):
            raise ValueError('Edge selection quota/uniqueness mismatch')
        info.update(a=.3, beta=.1, lambda_edge=.2, sampling_seed=seed,
                    gmm_components=gmm.n_components, group_allocation=quota.tolist(),
                    candidate_group_counts=counts.tolist(), final_group_counts=actual.tolist(),
                    class_allocation=class_quota.tolist(), final_size=size,
                    formula='w0=1+0.3*(2*d-1)+0.1*h; w_i=max(1e-6,w_i-0.2*K_ij*w_j_at_selection)',
                    seed_scope='probability draws only; greedy uses smallest pool index for exact ties',
                    graph_degree_normalization=False)
        atomic_json(out/f'coreset_{j}_selection_order.json', {'q_indices':chosen.tolist(), 'train_indices':qpos[chosen].tolist()})
        atomic_json(out/f'coreset_{j}_selection.json', info)
        results.append(np.sort(chosen))
    atomic_json(out/'edge_comparison_design.json', {
        'arms':[{'coreset_id':j,'mechanism':mode,'seed':seed} for j,(mode,seed) in enumerate(ARMS)],
        'a':.3, 'beta':.1, 'lambda_edge':.2, 'weight_floor':WEIGHT_FLOOR,
        'gmm_components':gmm.n_components, 'fixed_model_seed':cfg.model_seed,
        'shared':'EDA, plan, code, stage parameters, P/Q/V_core, GMM, graph, class/component quotas',
        'independent_unit':'task; three selection seeds are repeated measurements',
        'greedy_runs':1})
    return results
