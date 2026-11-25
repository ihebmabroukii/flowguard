"""Statistical rigor for the results: confidence intervals and significance tests, in NumPy.

Bootstrap confidence intervals (Efron and Tibshirani, 1993)
    Macro-F1, benign false-alert rate and detection rate depend on the test set only through
    its confusion counts, the number of flows in each (true family, predicted family) cell.
    Resampling n flows with replacement is therefore the same as drawing the cell counts from
    a multinomial with the observed cell frequencies. That gives an exact row bootstrap in
    microseconds instead of re-scoring millions of rows a thousand times.

    Caveat, stated in the report: the bootstrap treats flows as independent. Flows of one attack
    session are correlated, so the true uncertainty is somewhat larger than these intervals.

McNemar's test (McNemar, 1947; Dietterich, 1998)
    Two models scored on the same test flows. Only the flows where exactly one of them is right
    carry information: b = A right and B wrong, c = A wrong and B right. Under "both models
    have the same error rate", b ~ Binomial(b + c, 1/2).
        b + c <= 50: exact two-sided binomial test
        otherwise:   chi2 = (|b - c| - 1)^2 / (b + c), 1 degree of freedom (continuity corrected)
    With several model pairs, p-values are adjusted with Holm's step-down method (Holm, 1979).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np

from flowguard.data.schema import BENIGN

EXACT_LIMIT = 50


# --------------------------------------------------------------------------- confusion counts

def confusion_counts(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    """(counts, categories): counts[i, j] = flows with true categories[i] predicted categories[j]."""
    t = np.asarray(y_true, dtype=object).astype(str)
    p = np.asarray(y_pred, dtype=object).astype(str)
    if len(t) != len(p):
        raise ValueError("y_true and y_pred have different lengths")
    cats, inverse = np.unique(np.concatenate([t, p]), return_inverse=True)
    k = len(cats)
    ti, pi = inverse[: len(t)], inverse[len(t):]
    counts = np.bincount(ti * k + pi, minlength=k * k).reshape(k, k)
    return counts, cats.astype(object)


def _macro_f1(counts: np.ndarray, label_idx: np.ndarray) -> np.ndarray:
    """Macro-F1 over `label_idx` for one (k, k) or a batch (B, k, k) of confusion matrices.

    F1_k = 2 TP_k / (2 TP_k + FP_k + FN_k) = 2 TP_k / (predicted_k + true_k), 0 when both are 0
    (the same convention as scikit-learn's zero_division=0).
    """
    c = counts.astype(np.float64)
    tp = np.diagonal(c, axis1=-2, axis2=-1)[..., label_idx]
    predicted = c.sum(axis=-2)[..., label_idx]
    true = c.sum(axis=-1)[..., label_idx]
    den = predicted + true
    f1 = np.divide(2 * tp, den, out=np.zeros_like(den), where=den > 0)
    return f1.mean(axis=-1)


def _benign_fpr_per_10k(counts: np.ndarray, cats: np.ndarray) -> np.ndarray:
    if BENIGN not in cats:
        return np.full(counts.shape[:-2], np.nan)
    b = int(np.flatnonzero(cats == BENIGN)[0])
    row = counts[..., b, :].astype(np.float64)
    total = row.sum(axis=-1)
    alerts = total - row[..., b]
    return np.divide(alerts, total, out=np.full_like(total, np.nan), where=total > 0) * 10_000


def _detection_rate(counts: np.ndarray, cats: np.ndarray) -> np.ndarray:
    attack = cats != BENIGN
    c = counts.astype(np.float64)
    rows = c[..., attack, :]
    total = rows.sum(axis=(-1, -2))
    if BENIGN in cats:
        b = int(np.flatnonzero(cats == BENIGN)[0])
        missed = rows[..., b].sum(axis=-1)
    else:
        missed = np.zeros_like(total)
    return np.divide(total - missed, total, out=np.full_like(total, np.nan), where=total > 0)


# --------------------------------------------------------------------------- bootstrap

def _interval(estimate: float, samples: np.ndarray, alpha: float) -> dict[str, float]:
    samples = samples[np.isfinite(samples)]
    if not np.isfinite(estimate) or len(samples) == 0:
        return {"estimate": float(estimate), "low": float("nan"), "high": float("nan"), "std": float("nan")}
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(estimate), "low": float(low), "high": float(high), "std": float(samples.std(ddof=1))}


def bootstrap_ci(
    y_true: Any,
    y_pred: Any,
    labels: Sequence[str] | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile bootstrap intervals for macro-F1, benign false alerts per 10k and detection rate.

    `labels` fixes the families macro-F1 averages over (default: those present in y_true), so
    every resample is scored on the same classes as the point estimate.
    """
    counts, cats = confusion_counts(y_true, y_pred)
    if labels is None:
        labels = sorted(set(np.asarray(y_true, dtype=object).astype(str)))
    labels = [str(lab) for lab in labels]
    lookup = {c: i for i, c in enumerate(cats)}
    label_idx = np.array([lookup[lab] for lab in labels if lab in lookup], dtype=np.int64)
    if len(label_idx) == 0:
        raise ValueError("None of `labels` occurs in y_true or y_pred")

    n = int(counts.sum())
    rng = np.random.default_rng(seed)
    flat = counts.ravel()
    samples = rng.multinomial(n, flat / n, size=n_boot).reshape(n_boot, *counts.shape)

    return {
        "method": "percentile bootstrap over flows (multinomial on confusion cells)",
        "n_boot": int(n_boot),
        "confidence": 1 - alpha,
        "macro_f1": _interval(float(_macro_f1(counts, label_idx)), _macro_f1(samples, label_idx), alpha),
        "benign_fpr_per_10k": _interval(float(_benign_fpr_per_10k(counts, cats)),
                                        _benign_fpr_per_10k(samples, cats), alpha),
        "detection_rate": _interval(float(_detection_rate(counts, cats)), _detection_rate(samples, cats), alpha),
    }


# --------------------------------------------------------------------------- McNemar

def _binomial_two_sided(k: int, n: int) -> float:
    """P(X <= k) * 2 for X ~ Binomial(n, 1/2), k <= n/2, capped at 1. Exact, via log-gamma."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    log_half_n = -n * math.log(2.0)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n
             for i in range(k + 1)]
    top = max(terms)
    tail = math.exp(top) * sum(math.exp(t - top) for t in terms)
    return min(1.0, 2.0 * tail)


def mcnemar(y_true: Any, pred_a: Any, pred_b: Any, binary: bool = False) -> dict[str, Any]:
    """McNemar's test of "models A and B make errors at the same rate" on the same flows.

    binary=False: a prediction is right when it names the true family.
    binary=True:  a prediction is right when it alerts on attacks and stays quiet on benign flows.
    """
    t = np.asarray(y_true, dtype=object)
    a = np.asarray(pred_a, dtype=object)
    b_ = np.asarray(pred_b, dtype=object)
    if binary:
        attack = t != BENIGN
        right_a, right_b = (a != BENIGN) == attack, (b_ != BENIGN) == attack
    else:
        right_a, right_b = a == t, b_ == t
    b = int((right_a & ~right_b).sum())
    c = int((~right_a & right_b).sum())
    n = b + c
    if n <= EXACT_LIMIT:
        method, statistic, p = "exact binomial", float(min(b, c)), _binomial_two_sided(min(b, c), n)
    else:
        statistic = (abs(b - c) - 1) ** 2 / n
        method, p = "chi-squared, continuity corrected", math.erfc(math.sqrt(statistic / 2))
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "statistic": float(statistic),
            "p_value": float(p), "method": method,
            "error_rate_a": float(1 - right_a.mean()) if len(t) else float("nan"),
            "error_rate_b": float(1 - right_b.mean()) if len(t) else float("nan")}


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def pairwise_mcnemar(y_true: Any, predictions: dict[str, Any], binary: bool = False,
                     alpha: float = 0.05) -> list[dict[str, Any]]:
    """McNemar's test for every pair of models, with Holm-adjusted p-values."""
    rows = []
    for name_a, name_b in combinations(predictions, 2):
        rows.append({"model_a": name_a, "model_b": name_b,
                     **mcnemar(y_true, predictions[name_a], predictions[name_b], binary=binary)})
    for row, adj in zip(rows, holm([r["p_value"] for r in rows])):
        row["p_holm"] = adj
        row["significant"] = bool(adj < alpha)
        row["better"] = (row["model_a"] if row["a_right_b_wrong"] > row["a_wrong_b_right"]
                         else row["model_b"] if row["a_wrong_b_right"] > row["a_right_b_wrong"] else "tie")
    return rows


__all__ = ["bootstrap_ci", "confusion_counts", "holm", "mcnemar", "pairwise_mcnemar"]
