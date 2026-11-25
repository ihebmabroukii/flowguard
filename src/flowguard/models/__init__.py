"""Model zoo.

Ours (NumPy, models/scratch/):   softmax_regression, mlp, and the Isolation Forest in anomaly.py
Trivial baselines (ours):        majority (never alerts), random_prior (chance level)
Library benchmarks:              logreg (scikit-learn), random_forest (scikit-learn), xgboost
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flowguard.models.base import FlowClassifier
from flowguard.models.baselines import make_logreg, make_softmax_regression
from flowguard.models.mlp import make_mlp
from flowguard.models.trees import make_random_forest, make_xgboost
from flowguard.models.trivial import make_majority, make_random_prior

CLASSIFIERS: dict[str, Callable[[dict[str, Any], int], FlowClassifier]] = {
    "softmax_regression": make_softmax_regression,
    "mlp": make_mlp,
    "logreg": make_logreg,
    "random_forest": make_random_forest,
    "xgboost": make_xgboost,
    "majority": make_majority,
    "random_prior": make_random_prior,
}
OURS = ("softmax_regression", "mlp")
TRIVIAL = ("majority", "random_prior")
BENCHMARKS = ("logreg", "random_forest", "xgboost")


def build_classifier(name: str, models_cfg: dict[str, Any], seed: int = 42) -> FlowClassifier:
    """Classifier `name` configured from the `models:` section of configs/experiments.yaml."""
    if name not in CLASSIFIERS:
        raise KeyError(f"Unknown classifier {name!r}. Available: {', '.join(CLASSIFIERS)}")
    return CLASSIFIERS[name](models_cfg.get(name) or {}, seed)


__all__ = ["BENCHMARKS", "CLASSIFIERS", "OURS", "TRIVIAL", "FlowClassifier", "build_classifier"]
