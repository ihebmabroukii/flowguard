"""Metrics for an intrusion detector, as defined in the README.

Accuracy is never reported: with ~80% benign flows, a model that always says "benign" scores ~80%.
Labels are family names (strings); "Benign" is the negative class, everything else is an attack.
Unknown-anomaly alerts from the hybrid detector use the label UNKNOWN and count as attacks.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from flowguard.data.schema import BENIGN, UNKNOWN


def _arr(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=object)


def is_attack(labels: Any) -> np.ndarray:
    return _arr(labels) != BENIGN


def benign_fpr_per_10k(y_true: Any, y_pred: Any) -> float:
    """Benign flows that raised an alert, per 10,000 benign flows."""
    benign = ~is_attack(y_true)
    if not benign.any():
        return float("nan")
    return float(is_attack(_arr(y_pred)[benign]).mean() * 10_000)


def detection_rate(y_true: Any, y_pred: Any) -> float:
    """Share of attack flows that raised any alert (right family or not)."""
    attacks = is_attack(y_true)
    if not attacks.any():
        return float("nan")
    return float(is_attack(_arr(y_pred)[attacks]).mean())


def attack_pr_auc(y_true: Any, attack_score: Any) -> float:
    """Average precision of a continuous attack score (e.g. 1 - P(benign)) for attack vs benign."""
    y = is_attack(y_true).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(average_precision_score(y, np.asarray(attack_score, dtype=float)))


def classification_metrics(y_true: Any, y_pred: Any, labels: Sequence[str] | None = None) -> dict[str, Any]:
    """Macro-F1 over the families present in y_true, per-family precision/recall/F1, and alert rates.

    Passing `labels` restricts macro-F1 to those families (P2 scores only the shared ones).
    """
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    if labels is None:
        labels = sorted(set(y_true))
    labels = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        lab: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for lab, p, r, f, s in zip(labels, precision, recall, f1, support)
    }
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "benign_fpr_per_10k": benign_fpr_per_10k(y_true, y_pred),
        "detection_rate": detection_rate(y_true, y_pred),
        # Stored only to show, next to the trivial baselines, why it is not used to rank models.
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
        "per_class": per_class,
        "n": int(len(y_true)),
    }


def confusion_table(y_true: Any, y_pred: Any, labels: Sequence[str] | None = None) -> pd.DataFrame:
    """Rows = true family, columns = predicted family (UNKNOWN added when it was predicted)."""
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=pd.Index(labels, name="true"), columns=pd.Index(labels, name="predicted"))


def unknown_detection(y_true: Any, y_pred: Any, held_out: str) -> dict[str, float]:
    """P3: how the never-seen family was handled."""
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    held = y_true == held_out
    if not held.any():
        raise ValueError(f"No {held_out!r} flows in y_true")
    pred = y_pred[held]
    return {
        "flagged_unknown": float((pred == UNKNOWN).mean()),
        "flagged_any_alert": float(is_attack(pred).mean()),
        "missed_as_benign": float((pred == BENIGN).mean()),
        "benign_fpr_per_10k": benign_fpr_per_10k(y_true, y_pred),
    }


def evasion_rate(y_true: Any, pred_clean: Any, pred_adv: Any) -> dict[str, float]:
    """P4: of the attack flows the model caught, the share that become 'Benign' after perturbation."""
    y_true, pred_clean, pred_adv = _arr(y_true), _arr(pred_clean), _arr(pred_adv)
    caught = is_attack(y_true) & is_attack(pred_clean)
    if not caught.any():
        return {"evasion_rate": float("nan"), "n_caught": 0}
    return {
        "evasion_rate": float((pred_adv[caught] == BENIGN).mean()),
        "n_caught": int(caught.sum()),
    }


def throughput(predict: Callable[[Any], Any], X: Any, repeats: int = 3) -> float:
    """Flows per second for `predict(X)`, best of `repeats` runs."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        predict(X)
        best = min(best, time.perf_counter() - start)
    return float(len(X) / best) if best > 0 else float("inf")


def flatten(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """{'per_class': {'DoS': {'recall': .9}}} -> {'per_class.DoS.recall': .9}, for MLflow."""
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        elif isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            flat[name] = float(value)
    return flat
