"""Temperature scaling (Guo, Pleiss, Sun and Weinberger, 2017), written from scratch.

A network trained with cross-entropy is often over-confident: when it says 99%, it is right
less than 99% of the time. Temperature scaling fixes this with one parameter T > 0 learned on
the validation split:

    p_T(k | x) = softmax(z(x) / T)_k

T > 1 softens over-confident probabilities, T < 1 sharpens under-confident ones. Dividing every
logit by the same T never changes the argmax, so predictions and macro-F1 are untouched; only
the confidence values move.

T minimizes the validation negative log-likelihood. Written with beta = 1/T,
    NLL(beta) = mean_i [ logsumexp(beta z_i) - beta z_i[y_i] ]
is convex in beta (log-sum-exp of functions linear in beta), so a golden-section search on
beta finds the global minimum without gradients or tuning.

Why it matters here: the hybrid detector thresholds the classifier's confidence. With
saturated probabilities (several benign flows at exactly 1.0) no threshold can keep false
alerts in budget; after scaling, those confidences separate again.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from flowguard.data.schema import BENIGN
from flowguard.models.scratch.core import GradientClassifier, log_softmax, softmax

GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0
PROBA_FLOOR = 1e-12


def logits_of(classifier: Any, X: Any) -> np.ndarray:
    """Raw class scores of a fitted classifier, column order = classifier.classes_.

    Our gradient-trained models expose their real logits. For any other model (trees,
    scikit-learn) log-probabilities are used: softmax(log p) = p, so they play the same role.
    """
    estimator = getattr(classifier, "estimator", None)
    if isinstance(estimator, GradientClassifier):
        return estimator.decision_function(X)
    proba = np.asarray(classifier.predict_proba(X), dtype=np.float64)
    return np.log(np.clip(proba, PROBA_FLOOR, 1.0))


def negative_log_likelihood(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    y = np.asarray(y, dtype=np.int64)
    return float(-log_softmax(logits / temperature)[np.arange(len(y)), y].mean())


def fit_temperature(logits: Any, y: Any, bounds: tuple[float, float] = (0.05, 20.0),
                    tol: float = 1e-5, max_iter: int = 200) -> float:
    """Temperature T minimizing the NLL of integer labels y, searched over T in `bounds`."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if len(y) == 0:
        raise ValueError("No samples to fit the temperature on")
    rows = np.arange(len(y))

    def nll(beta: float) -> float:
        return float(-log_softmax(z * beta)[rows, y].mean())

    lo, hi = 1.0 / bounds[1], 1.0 / bounds[0]
    a, b = hi - GOLDEN * (hi - lo), lo + GOLDEN * (hi - lo)
    fa, fb = nll(a), nll(b)
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        if fa < fb:          # minimum lies in [lo, b]
            hi, b, fb = b, a, fa
            a = hi - GOLDEN * (hi - lo)
            fa = nll(a)
        else:                # minimum lies in [a, hi]
            lo, a, fa = a, b, fb
            b = lo + GOLDEN * (hi - lo)
            fb = nll(b)
    return float(1.0 / ((lo + hi) / 2.0))


def encode(classes: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """(codes, known): integer codes of labels y in `classes`; known = label is one of them."""
    classes = np.asarray(classes, dtype=object)
    y = np.asarray(y, dtype=object)
    lookup = {c: i for i, c in enumerate(classes)}
    known = np.array([v in lookup for v in y], dtype=bool)
    codes = np.array([lookup.get(v, -1) for v in y], dtype=np.int64)
    return codes, known


class TemperatureScaled:
    """A fitted classifier with temperature-scaled probabilities. Same interface as FlowClassifier."""

    def __init__(self, classifier: Any, temperature: float | None = None):
        self.classifier = classifier
        self.temperature_ = temperature

    def fit(self, X_val: Any, y_val: Any) -> TemperatureScaled:
        codes, known = encode(self.classifier.classes_, y_val)
        logits = logits_of(self.classifier, X_val)
        self.temperature_ = fit_temperature(logits[known], codes[known])
        return self

    @property
    def name(self) -> str:
        return f"{getattr(self.classifier, 'name', type(self.classifier).__name__)}+T"

    @property
    def classes_(self) -> np.ndarray:
        return self.classifier.classes_

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.temperature_ is None:
            raise RuntimeError("Call fit() first")
        return softmax(logits_of(self.classifier, X) / self.temperature_)

    def predict(self, X: Any) -> np.ndarray:
        return self.classifier.predict(X)  # argmax is unchanged by temperature

    def attack_score(self, X: Any, proba: np.ndarray | None = None) -> np.ndarray:
        proba = self.predict_proba(X) if proba is None else proba
        classes = list(self.classes_)
        if BENIGN not in classes:
            return np.ones(len(proba))
        return 1.0 - proba[:, classes.index(BENIGN)]


__all__ = ["TemperatureScaled", "encode", "fit_temperature", "logits_of", "negative_log_likelihood"]
