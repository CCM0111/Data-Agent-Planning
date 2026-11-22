"""Strict settings for the independent region-bandit workflow."""
from copy import deepcopy
import hashlib
import json
import math

DEFAULTS = {
    'enabled': True, 'mode': 'develop_online', 'coreset_repeats': 1, 'baseline_audit': False,
    'decision_seed': 2026, 'sampling_seeds': None, 'full_train_methods': 'top3',
    'sampling': {'mode': 'resample', 'max_replace_fraction': .1, 'resample_control': False,
                 'deletion_rule': 'uniform', 'independent_bandit': False,
                 'max_donor_remove_fraction': 1., 'allocation': 'capacity'},
    'regions': {
        'diagnostic_max_rows': 100000, 'discover_fraction': .5, 'max_features': 8,
        'numeric_quantiles': [.2, .4, .6, .8], 'max_basic_rules': 90,
        'max_conjunction_rules': 30, 'max_rule_conditions': 2, 'min_rows_per_half': 30,
        'min_expected_coreset_rows': 5, 'max_region_fraction': .9,
        'max_behavior_regions': 4, 'max_single_pair_regions': 0, 'max_coverage_regions': 2, 'jaccard_dedup_threshold': .8,
    },
    'actions': {'types': ['baseline', 'cover', 'focus'], 'focus_relative_increase': .25,
                'focus_max_percentage_points': .05, 'require_positive_actual_increase': True},
    'quotas': {'class_relative_slack': .1, 'subgroup_relative_slack': .3,
               'total_move_fraction': .1, 'solver_time_limit_per_action_seconds': 2.},
    'advisor': {'enabled': True, 'max_attempts': 1, 'allowed_scores': [-1, -.5, 0, .5, 1],
                'baseline_score': 0, 'experience_skills': False, 'decision_mode': 'score_fusion', 'max_history_datasets': 5, 'max_retrieval_distance': .35},
    'policy': {'ridge_theta': 1., 'ridge_api': 1., 'api_coefficient_prior': .1,
               'api_coefficient_bounds': [0., .2], 'cold_start_distinct_tasks': 12,
               'cold_start_epsilon': .5, 'epsilon': .2},
    'reward': {'atol': 1e-12, 'rtol': 1e-8, 'practical_tolerance': 0., 'objective': 'absolute'},
    'mcts_rounds': {'enabled': True, 'single_pair_weight': .5},
    'post_full_feedback': {'enabled': True, 'predict_core_on_external_validation': True,
                           'min_region_rows': 30, 'bootstrap_repeats': 300},
    'memory': {'dataset_total_weight': 1., 'exclude_current_dataset_from_retrieval': True,
                'require_compatible_versions': True},
    'runtime': {'max_wall_seconds': 21600, 'infrastructure_retries': 1},
}


def reward_protocol_hash(config=None, *, comparison_scope='initial'):
    """Identify every configurable part of the reward's meaning.

    Accept a full application config or its region_bandit section. Unstamped
    historical events remain an unknown protocol; they must not be silently
    interpreted as having used these defaults. Selected-top-three accuracy
    and all-five accuracy are distinct learning targets, even though they use
    the same pairwise formula. Audits/repeats use the effective all-five scope.
    """
    config = config or {}
    section = config.get('region_bandit', config)
    supplied = section.get('reward', {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS['reward']):
        raise ValueError('Unknown or invalid reward settings')
    values = {**DEFAULTS['reward'], **supplied}
    canonical = {}
    for key, value in values.items():
        if key == 'objective':
            if value not in ('absolute', 'paired_delta'):
                raise ValueError('Unknown reward objective')
            if value != 'absolute':
                canonical[key] = value
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid nonnegative reward setting: ' + key)
        canonical[key] = float(value)
    from .reward import REWARD_VERSION
    canonical['reward_version'] = REWARD_VERSION
    selection = normalize_config({key: section[key] for key in
        ('mode', 'full_train_methods', 'baseline_audit', 'coreset_repeats')
        if key in section})
    if comparison_scope == 'initial':
        # Keep the existing task-level learning target and its hash unchanged.
        canonical['method_scope'] = ('all_five' if full_train_count(selection) == 5
                                     else 'selected_top3')
    elif comparison_scope == 'mcts_parent_anchor':
        rounds = normalize_config({'mcts_rounds': section.get('mcts_rounds', {})})['mcts_rounds']
        canonical.update(method_scope='mcts_parent_anchor',
                         comparison_protocol='parent_plus_one_or_two_children_v1',
                         single_pair_weight=float(rounds['single_pair_weight']))
    else:
        raise ValueError('Unknown reward comparison scope')
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def normalize_config(value=None):
    value = value or {}
    if not isinstance(value, dict):
        raise ValueError('region_bandit configuration must be an object')
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown region_bandit settings: ' + str(sorted(unknown)))
    result = deepcopy(DEFAULTS)
    for key, item in value.items():
        if isinstance(result[key], dict):
            if not isinstance(item, dict) or set(item) - set(result[key]):
                raise ValueError('Unknown or invalid region_bandit section: ' + key)
            result[key].update(item)
        else:
            result[key] = deepcopy(item)
    sampling = result['sampling']
    donor = sampling['max_donor_remove_fraction']
    if type(donor) not in (int, float) or not math.isfinite(donor) or not 0 <= donor <= 1:
        raise ValueError('Invalid donor removal fraction')
    if sampling['allocation'] not in ('capacity', 'original_outside'):
        raise ValueError('Unknown exchange allocation')
    if (donor != 1 or sampling['allocation'] != 'capacity') and sampling['mode'] != 'local_edit':
        raise ValueError('Donor protection requires local editing')
    advisor = result['advisor']
    if advisor['decision_mode'] not in ('score_fusion', 'agent'):
        raise ValueError('Unknown decision mode')
    if advisor['decision_mode'] == 'agent' and not advisor['enabled']:
        raise ValueError('Agent decisions require enabled advisor')
    if result['reward']['objective'] not in ('absolute', 'paired_delta'):
        raise ValueError('Unknown reward objective')
    if result['reward']['objective'] == 'paired_delta' and advisor['decision_mode'] != 'agent':
        raise ValueError('Paired delta rewards currently require Agent decisions')
    if sampling['deletion_rule'] not in ('uniform', 'retention_score'):
        raise ValueError('Unknown deletion rule')
    if type(sampling['independent_bandit']) is not bool:
        raise ValueError('independent_bandit must be boolean')
    if sampling['deletion_rule'] != 'uniform' and sampling['mode'] != 'local_edit':
        raise ValueError('Scored exchanges require local_edit')
    if sampling['mode'] not in ('resample', 'local_edit'):
        raise ValueError('Unknown sampling mode')
    if type(sampling['max_replace_fraction']) not in (int, float) or not math.isfinite(sampling['max_replace_fraction']) or not 0 <= sampling['max_replace_fraction'] <= 1:
        raise ValueError('Invalid maximum replacement fraction')
    if type(sampling['resample_control']) is not bool:
        raise ValueError('resample_control must be boolean')
    if result['mode'] not in ('develop_online', 'evaluate_frozen'):
        raise ValueError('Unknown region_bandit mode')
    if result['full_train_methods'] not in ('top3', 'all'):
        raise ValueError('full_train_methods must be top3 or all')
    if result['mode'] == 'evaluate_frozen' and 'baseline_audit' not in value:
        result['baseline_audit'] = True
    for key in ('enabled', 'baseline_audit'):
        if type(result[key]) is not bool:
            raise ValueError(key + ' must be boolean')
    for key in ('coreset_repeats', 'decision_seed'):
        if type(result[key]) is not int or result[key] < (1 if key == 'coreset_repeats' else 0):
            raise ValueError('Invalid ' + key)
    if result['decision_seed'] >= 2**32:
        raise ValueError('decision_seed must fit uint32')
    seeds = result['sampling_seeds']
    if seeds is None:
        seeds = [20261026 + i for i in range(result['coreset_repeats'])]
    if (not isinstance(seeds, list) or len(seeds) != result['coreset_repeats']
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError('sampling_seeds must contain one distinct uint32 seed per repeat')
    # The existing bootstrap samples C0 at this seed before parameter feedback.
    if sampling['independent_bandit'] and (result['coreset_repeats'] != 1 or result['full_train_methods'] != 'all'):
        raise ValueError('Independent historical bandit requires one seed and all five full references')
    if seeds[0] != 20261026 and not sampling['independent_bandit']:
        raise ValueError('reference_seed is fixed at 20261026 by the original sampler')
    result['sampling_seeds'] = seeds
    for section, keys in {
        'regions': ('diagnostic_max_rows', 'max_features', 'max_basic_rules', 'max_conjunction_rules',
                    'min_rows_per_half', 'min_expected_coreset_rows', 'max_behavior_regions', 'max_single_pair_regions', 'max_coverage_regions'),
        'advisor': ('max_history_datasets',), 'policy': ('cold_start_distinct_tasks',),
        'post_full_feedback': ('min_region_rows', 'bootstrap_repeats'),
        'runtime': ('max_wall_seconds', 'infrastructure_retries'),
    }.items():
        for key in keys:
            v = result[section][key]
            if type(v) is not int or v < 0:
                raise ValueError(section + '.' + key + ' must be a nonnegative integer')
    if result['runtime']['max_wall_seconds'] < 1 or result['regions']['diagnostic_max_rows'] < 2:
        raise ValueError('Positive runtime budget and at least two diagnostic rows required')
    if result['regions']['max_rule_conditions'] != 2:
        raise ValueError('This version supports at most two basic conditions')
    fractions = {'regions': ('discover_fraction', 'max_region_fraction', 'jaccard_dedup_threshold'),
                 'quotas': ('class_relative_slack', 'subgroup_relative_slack', 'total_move_fraction'),
                 'policy': ('cold_start_epsilon', 'epsilon')}
    for section, keys in fractions.items():
        for key in keys:
            v = result[section][key]
            if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError(section + '.' + key + ' must be in [0,1]')
    if not 0 < result['regions']['discover_fraction'] < 1:
        raise ValueError('discover_fraction must split both halves')
    for v in result['regions']['numeric_quantiles']:
        if type(v) not in (int, float) or not 0 < v < 1:
            raise ValueError('Invalid numeric quantile')
    if result['actions']['types'] != ['baseline', 'cover', 'focus'] or result['actions']['require_positive_actual_increase'] is not True:
        raise ValueError('This version requires baseline/cover/focus with positive actual increase')
    if result['advisor']['max_attempts'] != 1 or result['advisor']['baseline_score'] != 0:
        raise ValueError('Advisor uses one call and a fixed zero baseline score')
    if result['advisor']['allowed_scores'] != [-1, -.5, 0, .5, 1]:
        raise ValueError('Unsupported API score scale')
    for section, keys in {'quotas': ('solver_time_limit_per_action_seconds',),
                         'actions': ('focus_relative_increase', 'focus_max_percentage_points'),
                         'policy': ('ridge_theta', 'ridge_api'),
                         'reward': ('atol', 'rtol', 'practical_tolerance')}.items():
        for key in keys:
            v = result[section][key]
            if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                raise ValueError('Invalid nonnegative setting: ' + section + '.' + key)
    if any(result['policy'][key] <= 0 for key in ('ridge_theta','ridge_api')):
        raise ValueError('Ridge strengths must be positive')
    pair_weight = result['mcts_rounds']['single_pair_weight']
    if type(pair_weight) not in (int, float) or not math.isfinite(pair_weight) or not 0 < pair_weight <= 1:
        raise ValueError('mcts_rounds.single_pair_weight must be finite and in (0,1]')
    if result['policy']['api_coefficient_bounds'] not in ([0., .2], [0., .5]):
        raise ValueError('Supported API coefficient bounds are [0, 0.2] or [0, 0.5]')
    prior = result['policy']['api_coefficient_prior']
    if type(prior) not in (int, float) or not math.isfinite(prior) or not 0 <= prior <= result['policy']['api_coefficient_bounds'][1]:
        raise ValueError('API coefficient prior must lie within its bounds')
    if result['advisor']['max_history_datasets'] > 5:
        raise ValueError('At most five historical source datasets may enter one API request')
    if (result['memory']['exclude_current_dataset_from_retrieval'] is not True
            or result['memory']['require_compatible_versions'] is not True
            or result['memory']['dataset_total_weight'] != 1.):
        raise ValueError('Source exclusion, version compatibility and unit dataset weight are mandatory')
    for section, keys in {'advisor': ('enabled', 'experience_skills'), 'mcts_rounds': ('enabled',), 'post_full_feedback':
                         ('enabled','predict_core_on_external_validation')}.items():
        if any(type(result[section][key]) is not bool for key in keys):
            raise ValueError('Invalid boolean setting in '+section)
    if not 0 <= result['advisor']['max_retrieval_distance'] <= 1:
        raise ValueError('Retrieval distance must be within [0,1]')
    if result['quotas']['solver_time_limit_per_action_seconds'] <= 0:
        raise ValueError('Solver wall budget must be positive')
    if result['post_full_feedback']['bootstrap_repeats'] < 1 or result['post_full_feedback']['min_region_rows'] < 1:
        raise ValueError('Region diagnostics require positive support and bootstrap counts')
    from .regions import _config as validate_regions
    validate_regions(result)
    return result


def full_train_count(config):
    """Audits/repeated-sampling experiments share a complete five-method bank."""
    return 5 if (config['full_train_methods'] == 'all' or config['baseline_audit']
                 or config['coreset_repeats'] > 1) else 3
