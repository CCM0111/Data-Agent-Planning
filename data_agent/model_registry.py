"""Supported main estimators shared by generation and budget enforcement.

None controls mean normal sklearn fitting under the runner's wall-clock timeout;
they do not invent an iteration parameter for closed-form/non-iterative models.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ModelRule:
    family: str
    parameter: str | None = None
    budget_key: str | None = None


MODEL_REGISTRY = {}
for prefixes, family, parameter, budget_key in [
    (('RandomForest', 'ExtraTrees'), 'bagging', 'n_estimators', 'max_trees'),
    (('GradientBoosting', 'AdaBoost'), 'boosting', 'n_estimators', 'max_trees'),
    (('HistGradientBoosting',), 'boosting', 'max_iter', 'max_trees'),
    (('MLP',), 'neural', 'max_iter', 'max_epochs'),
    (('SGD',), 'linear', None, None),
    (('KNeighbors',), 'neighbors', None, None),
    (('DecisionTree',), 'tree', None, None),
]:
    for prefix in prefixes:
        for suffix in ('Classifier', 'Regressor'):
            MODEL_REGISTRY[prefix + suffix] = ModelRule(family, parameter, budget_key)
for name in ('LogisticRegression', 'Ridge', 'LinearRegression', 'ElasticNet', 'Lasso', 'BayesianRidge'):
    MODEL_REGISTRY[name] = ModelRule('linear')
for name in ('GaussianNB', 'BernoulliNB'):
    MODEL_REGISTRY[name] = ModelRule('naive_bayes')


def model_contract():
    return ('Supported exact main estimator classes: ' + ', '.join(sorted(MODEL_REGISTRY)) +
            '. Choose a class appropriate for the task. Classification must produce probabilities '
            "(e.g. SGDClassifier loss='log_loss'). BernoulliNB requires binary features or an explicit "
            'binarize threshold. GaussianNB requires dense input. Ensure preprocessing output is compatible. '
            'Family taxonomy: RandomForest and ExtraTrees are bagging (including bootstrap=False); '
            'only a single DecisionTree is tree; AdaBoost and both GradientBoosting variants are boosting. '
            'AdaBoost may contain a DecisionTree of the same task type as its estimator weak learner. '
            'Do not use stacking, voting, calibration, or model-based feature selection which add extra fitted predictors.')


def declared_model_class(idea):
    """Resolve an explicit main class, never guess from a vague family or name."""
    text = idea.get('model', '') if isinstance(idea, dict) else ''
    if not isinstance(text, str):
        return None
    names = re.findall(r'(?<![A-Za-z0-9_])(' + '|'.join(sorted(MODEL_REGISTRY, key=len, reverse=True)) + r')(?![A-Za-z0-9_])', text)
    unique = list(dict.fromkeys(names))
    if len(unique) == 1:
        return unique[0]
    if unique and unique[0] in ('AdaBoostClassifier', 'AdaBoostRegressor'):
        child = 'DecisionTree' + unique[0][8:]
        if set(unique) <= {unique[0], child}:
            return unique[0]
    return None


def normalize_strategy_family(strategy):
    """Correct taxonomy only; retain the explicitly declared estimator and record it."""
    name = declared_model_class(strategy.get('idea'))
    if name and strategy.get('family') != MODEL_REGISTRY[name].family:
        old = strategy['family']
        strategy['family'] = MODEL_REGISTRY[name].family
        return {'field': 'family', 'from': old, 'to': strategy['family'],
                'evidence': 'idea.model', 'declared_model_class': name}
    return None
