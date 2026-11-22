"""Validate task-level GMM choices made alongside method stage parameters."""
from dataclasses import replace

GMM_PLAN_INSTRUCTION = '''Alongside the strategies, return one top-level coreset_design object:
{"gmm_components": <integer>, "rationale": <nonempty Chinese explanation>}.
This is one shared task-level GMM, NOT one GMM per candidate method or per class.
Choose K using the supplied training EDA, proxy_train_rows (GMM is fitted on P),
coreset_rows, candidate_rows, feature structure and class counts. Explain why the
chosen granularity is plausible and discuss small-group/coreset quota limitations.
Do not assume K equals the number of candidate methods, classes, or the old default 5.
K must be an integer in [1, min(proxy_train_rows, coreset_rows, candidate_rows)].
Use only provided EDA and sizes; do not claim unobserved clustering quality or use
validation/test results. Put K only in coreset_design, not pipeline stage_parameters.
The same K will be used for subsequent GMM fitting and component grouping.
'''


def validate_gmm_design(design, task_context):
    if not isinstance(design, dict):
        raise ValueError('Plan requires task-level coreset_design with gmm_components and rationale')
    sizes=task_context.get('training_context', {})
    capacities=[sizes.get(k) for k in ('proxy_train_rows','coreset_rows','candidate_rows')]
    if any(type(n) is not int or n < 1 for n in capacities):
        raise ValueError('GMM planning requires positive actual P/coreset/Q sizes')
    k=design.get('gmm_components'); reason=design.get('rationale')
    if type(k) is not int or not 1 <= k <= min(capacities):
        raise ValueError(f'LLM GMM K must be an integer in [1, {min(capacities)}]')
    if not isinstance(reason,str) or not reason.strip():
        raise ValueError('LLM GMM K requires a nonempty EDA/size rationale')
    return {'gmm_components':k,'rationale':reason.strip()}


def apply_gmm_design(cfg, design, task_context):
    design=validate_gmm_design(design, task_context)
    return replace(cfg, gmm_components=design['gmm_components']), design


def validate_reused_gmm(design, report, model_components):
    k=design['gmm_components']
    if report.get('requested_components')!=k or report.get('actual_components')!=k or model_components!=k:
        raise ValueError('Shared GMM component count differs from the task-level LLM plan')
