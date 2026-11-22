"""Versioned idea records and a JSON trace assembled from run artifacts."""
import hashlib
import json
import re
import copy
from pathlib import Path


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize_plan(plan, *, budgets=None, mirror_full=False):
    """Move misplaced known idea fields without inventing or overwriting values."""
    result = copy.deepcopy(plan)
    changes = []
    if not isinstance(result, dict) or not isinstance(result.get('strategies'), list):
        raise ValueError('Plan requires a strategies array')
    fields = ('summary', 'change_reason', 'description', 'rationale', 'feature_engineering', 'model',
              'fixed_parameters', 'scaling_rules', 'stage_budgets',
              'stage_parameters', 'adaptation_reason', 'limitations')
    for i, strategy in enumerate(result['strategies']):
        if not isinstance(strategy, dict) or not isinstance(strategy.get('idea'), dict):
            raise ValueError('Each strategy requires an idea object')
        idea = strategy['idea']
        for key in fields:
            if key not in strategy:
                continue
            if key in idea and identity(idea[key]) != identity(strategy[key]):
                raise ValueError(f'Ambiguous idea field at both levels: strategies[{i}].{key}')
            idea[key] = strategy.pop(key)
            changes.append({'from': f'strategies[{i}].{key}',
                            'to': f'strategies[{i}].idea.{key}'})
        if 'summary' in idea:
            if 'description' in idea and idea['description'] != idea['summary']:
                raise ValueError('Conflicting method summary and description')
            idea['description'] = idea.pop('summary')
            idea['plan_format'] = 'brief_v1'
        if idea.get('plan_format') == 'brief_v1':
            if budgets is not None:
                if 'stage_budgets' in idea and idea['stage_budgets'] != budgets:
                    raise ValueError('Explicit stage budgets conflict with runner budgets')
                idea['stage_budgets'] = copy.deepcopy(budgets)
            stages = idea.get('stage_parameters')
            if mirror_full and isinstance(stages, dict):
                if set(stages) == {'full_train'}:
                    stages['coreset'] = copy.deepcopy(stages['full_train'])
                if stages.get('coreset') != stages.get('full_train'):
                    raise ValueError('guarded_once requires identical initial stage parameters')
                rules = idea.get('scaling_rules')
                if rules in (None, [], ''):
                    idea['scaling_rules'] = {}
                elif isinstance(rules, dict) and rules:
                    # Some compact plans describe FE normalization under the
                    # similarly named model-size scaling field.
                    steps = idea.get('feature_engineering')
                    if not (isinstance(steps, list) and all(isinstance(v, str) for v in steps)
                            and all(key in ('StandardScaler', 'RobustScaler', 'MinMaxScaler', 'QuantileTransformer')
                                    and key in ' '.join(steps) and isinstance(value, str)
                                    and not re.search(r'fit_rows|full_train|coreset|样本量|样本数|model__', value, re.I)
                                    for key, value in rules.items())):
                        raise ValueError('guarded_once cannot declare parameter scaling rules')
                    steps.extend(f'{key}: {value}' for key, value in rules.items())
                    idea['scaling_rules'] = {}
                elif isinstance(rules, str) and any(name in rules for name in
                        ('StandardScaler', 'RobustScaler', 'MinMaxScaler', 'QuantileTransformer')):
                    # Preserve an FE explanation only when the declared transformations
                    # corroborate it. Size-dependent instructions belong in a formal rule.
                    steps = idea.get('feature_engineering')
                    names = re.findall(r'[A-Z][A-Za-z]+(?:Scaler|Transformer)|PCA', rules)
                    if (isinstance(steps, list) and all(isinstance(step, str) for step in steps)
                            and all(name in ' '.join(steps) for name in names)
                            and not re.search(r'fit_rows|full_train|coreset|样本量|样本数|model__', rules, re.I)):
                        if rules not in steps:
                            steps.append(rules)
                        idea['scaling_rules'] = {}
    return result, changes


def validate_idea(idea):
    if not isinstance(idea, dict):
        raise ValueError('Plan idea must be an object')
    for key in (('description',) if idea.get('plan_format') == 'brief_v1' else ('description', 'rationale', 'limitations')):
        if not isinstance(idea.get(key), str) or not idea[key].strip():
            raise ValueError('Idea requires nonempty text: ' + key)
    steps = idea.get('feature_engineering')
    if not isinstance(steps, list) or not steps or any(not isinstance(s, str) or not s.strip() for s in steps):
        raise ValueError('Idea requires feature_engineering text steps')
    if not isinstance(idea.get('model'), str) or not idea['model'].strip():
        raise ValueError('Idea requires a model description')
    for key in ('fixed_parameters', 'scaling_rules', 'stage_budgets'):
        if not isinstance(idea.get(key), dict):
            raise ValueError('Idea requires object: ' + key)


def validate_stage_parameters(idea):
    stages=idea.get('stage_parameters')
    if not isinstance(stages,dict) or set(stages)!={'coreset','full_train'}:
        raise ValueError('Idea requires coreset and full_train stage_parameters')
    for stage,parameters in stages.items():
        if not isinstance(parameters,dict):
            raise ValueError('Each stage requires a parameter object; empty means no overrides')
        for key,value in parameters.items():
            if not isinstance(key, str) or '__' not in key or not all(key.split('__')) or key.split('__')[-1] in {'random_state','n_jobs','warm_start','n_estimators','early_stopping','steps','transformers','estimator','base_estimator'}:
                raise ValueError('Stage parameter must use pipeline paths and not override budget controls: '+key)
            # JSON values only; no executable expressions or estimator replacements.
            if not isinstance(value,(str,int,float,bool,list,dict,type(None))):
                raise ValueError('Stage parameters must be JSON values')
    if idea.get('plan_format') != 'brief_v1' and (not isinstance(idea.get('adaptation_reason'), str) or not idea['adaptation_reason'].strip()):
        raise ValueError('Idea requires adaptation_reason')


def describe_eda(eda):
    features = eda['feature_columns']
    missing = {c: v['missing'] for c, v in eda['columns'].items() if v['missing']}
    text = (f"EDA 范围为 P＋Q，排除 V_core、外部 validation 和 test。任务类型：{eda['task_type']}；"
            f"统计行数：{eda.get('train_rows', '未提供')}；目标列：{eda['target']}；有效特征 {len(features)} 个，其中数值特征 "
            f"{len(eda['numeric_columns'])} 个、类别特征 {len(eda['categorical_columns'])} 个。"
            f"非目标字段的缺失数量：{json.dumps(missing, ensure_ascii=False)}。"
            f"排除字段及原因：{json.dumps(eda['excluded_fields'], ensure_ascii=False)}。")
    if 'target_distribution' in eda:
        text += f"类别计数：{json.dumps(eda['target_distribution'], ensure_ascii=False)}。"
    return text


def save_trace(out, manifest):
    """Checkpoint after each stage/call/fit; atomic replacement preserves last snapshot."""
    out = Path(out)
    diagnostics = []
    def read(relative):
        path = out / relative
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text())
            return value
        except (ValueError, OSError) as error:
            diagnostics.append({'artifact': relative, 'error': f'{type(error).__name__}: {error}'})
            return None
    def artifact(relative):
        return relative if (out / relative).exists() else None
    task = manifest.get('task', {})
    methods = []
    n_coresets = manifest.get('configuration', {}).get('n_coresets', 3)
    reused = manifest.get('methods_cache_hit', False)
    for i in range(manifest.get('configuration', {}).get('n_strategies', 3)):
        prefix = f'methods/method_{i}'
        record = read(prefix + '/idea.json')
        if record is None:
            continue
        code = out / prefix / 'method.py'
        executions = []
        for stage, folder, subset in [('proxy', f'proxies/method_{i}', 'P_indices.json')] + [
            ('evaluation', f'evaluations/method_{i}_coreset_{j}', f'coreset_{j}_indices.json') for j in range(n_coresets)
        ] + ([('full_train', f'full_train/method_{i}', 'full_train_indices.json')] if manifest.get('configuration', {}).get('full_train_enabled') else []):
            report = read(folder + ('/evaluation.json' if stage in ('evaluation','full_train') else '/score_report.json'))
            fit_report = read(folder + '/fit_report.json')
            failure = read(folder + '/failure.json')
            if manifest.get('status') == 'failed' and manifest.get('active_execution') == folder:
                failure = failure or {'error': manifest.get('error'), 'stage': manifest.get('failure_stage'), 'origin': 'coordinator'}
            if report is not None or failure is not None or (out / folder).exists():
                executions.append({'stage': stage, 'training_indices': artifact(subset),
                                   'status': 'failed' if failure else 'success' if report else 'incomplete',
                                   'fit_status': 'success' if fit_report and fit_report.get('status') == 'success' else 'unknown',
                                   'fit_report': fit_report, 'report': report, 'failure': failure, 'artifacts': folder,
                                   'reuse': read(folder + '/reuse.json'),
                                   'new_fit': fit_report is not None and read(folder + '/reuse.json') is None})
        metadata = read(prefix + '/method.json') or {}
        repairs = metadata.get('repairs') or []
        repair_path = repairs[-1].get('response_artifact') if repairs else None
        generation = {'origin': 'reused' if reused else 'generated',
            'request_path': artifact(prefix + '/implementation.request.json'),
            'response_path': artifact(prefix + '/implementation.json'),
            'usage': read(prefix + '/implementation.usage.json')}
        if repair_path:
            repair = Path(repair_path).resolve()
            generation.update(request_path=artifact(str(repair.with_suffix('.request.json'))),
                response_path=artifact(str(repair)), usage=read(str(repair.with_suffix('.usage.json'))),
                repair_mode=repairs[-1].get('mode'), effective_implementation_path=artifact(prefix+'/implementation.json'))
        methods.append({'method_id': f'method_{i}', 'idea': record,
                        'implementation': {'version': 1, 'idea_id': record['idea_id'],
                                           'idea_version': record['version'],
                                           'code_path': artifact(prefix + '/method.py'),
                                           'code_sha256': hashlib.sha256(code.read_bytes()).hexdigest() if code.exists() else None,
                                           'configuration': metadata},
                        'generation': generation,
                        'executions': executions})
    edge_comparison = manifest.get('configuration', {}).get('proxy_selection_scheme') == 'edge_mechanism_comparison'
    trace = {'schema_version': '1.1', 'workflow_version': manifest['version'],
             'trace_id': identity({'output': str(out.resolve()), 'task_path': manifest['task_path']}),
             'status': manifest['status'], 'failure_stage': manifest.get('failure_stage'), 'error': manifest.get('error'),
             'dataset': {'dataset_id': task.get('dataset_id') or task.get('task_id') or task.get('name'),
                         'task_path': manifest['task_path'], 'split_id': manifest.get('split_id'),
                         'data_hashes': manifest.get('input_sha256'), 'problem_type': task.get('problem_type'),
                         'metric': task.get('metric')},
             'eda': read('eda.json'), 'training_context': manifest.get('training_context'),
             'parameter_feedback': read('parameter_feedback/frozen_plan.json'),
             'coreset_design': manifest.get('coreset_design') or read('methods/coreset_design.json'),
             'configuration': manifest['configuration'], 'model': manifest['llm_model'],
             'internal_split_id': manifest.get('internal_split_id'),
             'coresets': [{'coreset_id': i, 'source_proxy_method': f'method_{i}' if not edge_comparison and i < manifest.get('configuration', {}).get('n_strategies', 3) else None,
                            'source': 'density_hardness_edge' if edge_comparison else ('proxy_error' if i < manifest.get('configuration', {}).get('n_strategies', 3) else 'gmm_density'),
                            'indices_path': artifact(f'coreset_{i}_indices.json'),
                            'selection': read(f'coreset_{i}_selection.json')} for i in range(n_coresets)],
             'gmm': {'report': read('density_gmm/score_report.json'), 'failure': read('density_gmm/failure.json')},
             'graph': read('graph.json'),
             'reuse_source': manifest.get('reuse_source'),
             'evaluation_set': manifest.get('evaluation_set'),
             'full_train_evaluation_set': manifest.get('full_train_evaluation_set'),
             'full_train_ranking': read('full_train_ranking.json'),
             'stage1_selection': {'summary': read('stage1_selection/summary.json'),
                                  'mcts_admission': read('stage1_selection/mcts_admission.json'),
                                  'full_train_comparison': read('stage1_selection/full_train_comparison.json')},
             'fairness': read('fairness.json'),
             'generation_configuration': manifest.get('generation_configuration'),
             'generation_model': manifest.get('generation_model'),
             'diagnostics': diagnostics,
             'rounds': [{'round': 1, 'plan_path': artifact('methods/plan.json'),
                         'plan_request_path': artifact('methods/plan.request.json'),
                         'plan_usage': read('methods/plan.usage.json'), 'methods': methods,
                         'ranking': read('ranking.json')}],
             'current_run_llm_calls': manifest['llm_calls'], 'timings_seconds': manifest['timings_seconds']}
    temp = out / 'trace.json.tmp'
    temp.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + '\n')
    temp.replace(out / 'trace.json')
    return trace
