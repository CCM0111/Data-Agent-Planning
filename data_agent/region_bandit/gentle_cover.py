"""A partial coverage repair with a hard budget on actual row replacements."""
import numpy as np

from ..multi_proxy import split_quotas
from .local_edit import LOCAL_SAMPLER_VERSION, cell_counts, reference_hash


def plan_gentle_cover(cover, references, previews, groups, mask, decision_seed):
    """Build a cover variant only when the preview of full cover is disruptive."""
    if 'protected_q_indices' in cover:
        from ..guided_coreset import coordinates
        cover=dict(cover);fixed=cover.pop('protected_q_indices')
        pool,_=coordinates(next(iter(references.values())),groups,fixed)
        editable_previews = {}
        for seed, ref in references.items():
            preview = previews[str(seed)]
            replaced = preview.get('sample_replaced_count',
                                   round(preview['sample_replaced_fraction'] * len(ref)))
            editable_previews[str(seed)] = {**preview, 'sample_replaced_fraction': replaced / cover['size']}
        result=plan_gentle_cover(cover,{s:coordinates(r,groups,fixed)[1] for s,r in references.items()},
                                 editable_previews,groups[pool],mask[pool],decision_seed)
        if result:result['protected_q_indices']=fixed
        return result
    size = cover['size']
    cap = size // 10
    values = np.asarray(cover['group_values'])
    plans = {}
    substantial = False
    for seed, reference in references.items():
        preview = previews[str(seed)]
        before = cell_counts(reference, groups, mask, values)
        count = int(before[:, 1].sum())
        gap = max(0, cover['feasible_region_count'] - count)
        if gap and (preview['sample_replaced_fraction'] >= .50 or
                    gap >= .10 * size or gap >= .50 * max(count, 1)):
            substantial = True
        available = mask.copy()
        available[reference] = False
        capacity = np.minimum(before[:, 0], [np.count_nonzero(available & (groups == g)) for g in values])
        swaps = min((gap + 1) // 2, cap, int(capacity.sum()))
        table = before.copy()
        if swaps:
            active = capacity > 0
            allocation = split_quotas(capacity[active], swaps, decision_seed,
                                      lower=np.zeros(active.sum(), int), upper=capacity[active])
            table[active, 0] -= allocation
            table[active, 1] += allocation
        plans[str(seed)] = dict(reference_hash=reference_hash(np.sort(reference)), quota_table=table.tolist(),
            reference_region_count=count, requested_region_count=count + (gap + 1) // 2,
            feasible_region_count=count + swaps, actual_region_increase=swaps)
    if not substantial or not any(p['actual_region_increase'] for p in plans.values()):
        return None
    first = next(iter(plans.values()))
    return dict(action_id='gentle_cover:' + cover['region_id'], type='cover', variant='gentle_cover',
        region_id=cover['region_id'], region_mask_hash=cover['region_mask_hash'], rule=cover.get('rule'),
        region_fraction=cover['region_fraction'], size=size, group_values=values.tolist(),
        quota_columns=['outside', 'inside'], group_moved_count=0, class_moved_count=0,
        solver_status='bounded_within_group_exchanges', sampling_mode='local_edit',
        sampler_version=LOCAL_SAMPLER_VERSION, sampler_config=cover['sampler_config'],
        max_replaced_rows=cap, full_cover_target=cover['feasible_region_count'],
        actual_region_increase_fraction=first['actual_region_increase'] / size,
        per_seed_plans=plans, **first)
