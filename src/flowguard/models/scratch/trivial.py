"""Trivial baselines: the floor every real model has to clear.

majority      always predicts the most frequent training class. On this data that is Benign,
              so it is "the detector that never alerts". Its accuracy is around 80%, its
              macro-F1 is close to 0.1: this one row shows why the report never uses accuracy.
random_prior  guesses at random, drawing each prediction from the training class frequencies.
              Its macro-F1 and PR-AUC are the chance level for this class balance.

Same interface as our other estimators (integer labels in, FlowClassifier wraps them).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class TrivialClassifier:
    STRATEGIES = ("majority", "random_prior")

    def __init__(self, strategy: str = "majority", random_state: int = 0):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")
        self.strategy = strategy
        self.random_state = random_state

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {"strategy": self.strategy, "random_state": self.random_state}

    def fit(self, X: Any, y: Any, sample_weight: Any = None, **_: Any) -> TrivialClassifier:
        y = np.asarray(y, dtype=np.int64)
        self.n_classes_ = int(y.max()) + 1
        self.prior_ = np.bincount(y, minlength=self.n_classes_) / len(y)  # unweighted on purpose
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        if self.strategy == "majority":
            out = np.zeros((n, self.n_classes_))
            out[:, int(self.prior_.argmax())] = 1.0
            return out
        # One random one-hot row per flow, seeded so that a model scores the same twice.
        rng = np.random.default_rng(self.random_state)
        picks = rng.choice(self.n_classes_, size=n, p=self.prior_)
        return np.eye(self.n_classes_)[picks]

    def predict(self, X: Any) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
