"""Tree ensembles: Random Forest and XGBoost."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from flowguard.models.base import FlowClassifier


def make_random_forest(cfg: dict[str, Any], seed: int = 42) -> FlowClassifier:
    params = dict(cfg)
    max_rows = params.pop("max_train_rows", None)
    params.setdefault("class_weight", "balanced_subsample")
    params.setdefault("n_jobs", -1)
    estimator = RandomForestClassifier(random_state=seed, **params)
    return FlowClassifier("random_forest", estimator, max_train_rows=max_rows, seed=seed)


def make_xgboost(cfg: dict[str, Any], seed: int = 42) -> FlowClassifier:
    from xgboost import XGBClassifier  # imported here so the rest of the package works without it

    params = dict(cfg)
    max_rows = params.pop("max_train_rows", None)
    max_class_weight = params.pop("max_class_weight", 100)
    params.setdefault("tree_method", "hist")
    params.setdefault("eval_metric", "mlogloss")
    params.setdefault("n_jobs", -1)
    estimator = XGBClassifier(random_state=seed, **params)
    return FlowClassifier(
        "xgboost",
        estimator,
        balanced_weights=True,
        max_class_weight=max_class_weight,
        max_train_rows=max_rows,
        uses_eval_set=True,
        seed=seed,
    )
