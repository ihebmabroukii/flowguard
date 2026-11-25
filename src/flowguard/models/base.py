"""One interface for every classifier: string family labels in, string family labels out."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from flowguard.data.schema import BENIGN


def stratified_cap(y: np.ndarray, max_rows: int, min_per_class: int = 1000, seed: int = 42) -> np.ndarray:
    """Indices of a class-stratified subsample of at most ~max_rows rows.

    Every class keeps at least min(its size, min_per_class) rows, so rare attacks are never lost.
    """
    n = len(y)
    if max_rows is None or n <= max_rows:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    frac = max_rows / n
    picked = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        k = min(len(idx), max(int(np.ceil(len(idx) * frac)), min_per_class))
        picked.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(picked))


class FlowClassifier:
    """Wraps any scikit-learn-style classifier.

    - encodes family names to integers (XGBoost needs 0..k-1)
    - optional balanced sample weights, clipped so a 36-flow class can't dominate the loss
    - optional stratified row cap for slow models (logistic regression)
    - optional validation set for early stopping (XGBoost)
    """

    def __init__(
        self,
        name: str,
        estimator: Any,
        balanced_weights: bool = False,
        max_class_weight: float | None = None,
        max_train_rows: int | None = None,
        uses_eval_set: bool = False,
        seed: int = 42,
    ):
        self.name = name
        self.estimator = estimator
        self.balanced_weights = balanced_weights
        self.max_class_weight = max_class_weight
        self.max_train_rows = max_train_rows
        self.uses_eval_set = uses_eval_set
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: Any, X_val: pd.DataFrame | None = None, y_val: Any = None) -> FlowClassifier:
        y = np.asarray(y, dtype=object)
        self.encoder_ = LabelEncoder().fit(y)
        rows = stratified_cap(y, self.max_train_rows, seed=self.seed)
        X_fit, y_fit = X.iloc[rows], self.encoder_.transform(y[rows])

        kwargs: dict[str, Any] = {}
        if self.balanced_weights:
            weights = compute_sample_weight("balanced", y_fit)
            if self.max_class_weight:
                weights = np.minimum(weights, self.max_class_weight)
            kwargs["sample_weight"] = weights
        if self.uses_eval_set and X_val is not None and y_val is not None:
            y_val = np.asarray(y_val, dtype=object)
            known = np.isin(y_val, self.encoder_.classes_)
            kwargs["eval_set"] = [(X_val[known], self.encoder_.transform(y_val[known]))]
            kwargs["verbose"] = False

        start = time.perf_counter()
        self.estimator.fit(X_fit, y_fit, **kwargs)
        self.fit_seconds_ = time.perf_counter() - start
        self.n_train_rows_ = int(len(rows))
        return self

    @property
    def classes_(self) -> np.ndarray:
        return self.encoder_.classes_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def attack_score(self, X: pd.DataFrame, proba: np.ndarray | None = None) -> np.ndarray:
        """1 - P(Benign): one number per flow, used for PR curves and thresholds."""
        proba = self.predict_proba(X) if proba is None else proba
        if BENIGN not in self.classes_:
            return np.ones(len(proba))
        return 1.0 - proba[:, list(self.classes_).index(BENIGN)]

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "classes": list(self.classes_),
            "n_train_rows": self.n_train_rows_,
            "fit_seconds": round(self.fit_seconds_, 2),
            "params": {k: v for k, v in self.estimator.get_params().items() if _plain(v)},
        }


def _plain(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))
