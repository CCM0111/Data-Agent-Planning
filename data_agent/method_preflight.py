"""Construct all stage pipelines and validate parameters without fitting data."""
import importlib.util
import json
from pathlib import Path
import sys
import traceback

import joblib

if __package__:
    from .multi_proxy_runtime import validate_code, prepare_pipeline, model_family
    from .multi_proxy_io import atomic_json
    from .model_registry import declared_model_class
    from .parameter_feedback import complete_stage_parameters, validate_estimators
else:
    from multi_proxy_runtime import validate_code, prepare_pipeline, model_family
    from multi_proxy_io import atomic_json
    from model_registry import declared_model_class
    from parameter_feedback import complete_stage_parameters, validate_estimators


def check(context):
    code = Path(context['code'])
    validate_code(code.read_text())
    spec = importlib.util.spec_from_file_location('preflight_generated_method', code)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    output = Path(context['output'])
    reports = []
    stage_parameters = context['idea']['stage_parameters']
    constructed = {}
    if context.get('complete_stage_parameters') or context['task'].get('complete_stage_parameters'):
        # Construct each factory once. Complete the other-stage reset using its
        # own original configuration before applying any new overrides.
        for stage, rows, budget in context['stages']:
            if stage not in ('evaluation', 'full_train'):
                continue
            task = {**context['task'], 'fit_stage': stage, 'fit_rows': rows,
                    'fit_class_counts': context['task']['training_context'].get('proxy_class_counts', []),
                    'stage_parameters': stage_parameters['coreset' if stage == 'evaluation' else 'full_train']}
            constructed[stage] = module.build_method(task, budget, context['seed'])
        if set(constructed) != {'evaluation', 'full_train'}:
            raise ValueError('Stage parameter completion requires both formal stages')
        stage_parameters = complete_stage_parameters(stage_parameters, {
            'coreset': constructed['evaluation'], 'full_train': constructed['full_train']})
    for stage, rows, budget in context['stages']:
        task = {**context['task'], 'fit_stage': stage, 'fit_rows': rows,
                'fit_class_counts': context['task']['training_context'].get('proxy_class_counts', [])}
        if stage != 'proxy':
            task['stage_parameters'] = stage_parameters[
                'coreset' if stage == 'evaluation' else 'full_train']
        if stage == 'full_train':
            task['reference_pipeline_path'] = str(output/'reference.joblib')
            task['expected_model_class'] = reports[-1]['model_class']
        pipeline = constructed[stage] if stage in constructed else module.build_method(task, budget, context['seed'])
        actual, configuration = prepare_pipeline(pipeline, task, budget, context['seed'])
        declared = declared_model_class(context['idea'])
        if declared and actual['model_class'] != declared:
            raise ValueError(f'Implementation class {actual["model_class"]} differs from declared {declared}')
        if model_family(actual['model_class']) != context['family']:
            raise ValueError(f'Implementation model family differs from original idea: '
                             f'{actual["model_class"]} is {model_family(actual["model_class"])}, planned {context["family"]}')
        validate_estimators(pipeline)
        if stage == 'evaluation':
            joblib.dump(pipeline, output/'reference.joblib')
        reports.append({'stage': stage, 'model_class': actual['model_class'],
                        'pipeline_spec': configuration})
    return {'status': 'passed', 'stages': reports, 'stage_parameters': stage_parameters, 'fit_calls': 0,
            'scope': 'Constructor, estimator parameter and declared stage configuration checks only; no fit/predict or validation scores'}


if __name__ == '__main__':
    context = json.loads(Path(sys.argv[1]).read_text())
    try:
        atomic_json(Path(context['output'])/'report.json', check(context))
    except Exception as error:
        atomic_json(Path(context['output'])/'failure.json',
                    {'error': f'{type(error).__name__}: {error}'})
        traceback.print_exc()
        sys.exit(1)
