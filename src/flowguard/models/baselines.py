"""Linear models.

softmax_regression: our own multi-class logistic regression (models/scratch/softmax.py).
logreg:             scikit-learn LogisticRegression, kept as a benchmark for ours.
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression

from flowguard.models.base import FlowClassifier
from flowguard.models.scratch import SoftmaxRegression


def make_softmax_regression(cfg: dict[str, Any], seed: int = 42) -> FlowClassifier:
    params = dict(cfg)
    max_rows = params.pop("max_train_rows", None)
    max_class_weight = params.pop("max_class_weight", 100)
    return FlowClassifier(
        "softmax_regression",
        SoftmaxRegression(random_state=seed, **params),
        balanced_weights=True,
        max_class_weight=max_class_weight,
        max_train_rows=max_rows,
        uses_eval_set=True,
        seed=seed,
    )


def make_logreg(cfg: dict[str, Any], seed: int = 42) -> FlowClassifier:
    params = dict(cfg)
    max_rows = params.pop("max_train_rows", None)
    estimator = LogisticRegression(
        C=float(params.pop("C", 1.0)),
        max_iter=int(params.pop("max_iter", 2000)),
        class_weight=params.pop("class_weight", "balanced"),
        random_state=seed,
        **params,
    )
    return FlowClassifier("logreg", estimator, max_train_rows=max_rows, seed=seed)
