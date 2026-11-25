"""Bootstrap intervals and McNemar's test (NumPy only)."""

import math

import numpy as np
import pytest

from flowguard.eval.stats import (
    _macro_f1,
    bootstrap_ci,
    confusion_counts,
    holm,
    mcnemar,
    pairwise_mcnemar,
)

FAMILIES = np.array(["Benign", "DoS", "PortScan"], dtype=object)


def _predictions(n=3000, error=0.1, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(FAMILIES, size=n, p=[0.7, 0.2, 0.1])
    pred = y.copy()
    flip = rng.random(n) < error
    pred[flip] = rng.choice(FAMILIES, size=int(flip.sum()))
    return y, pred


def _macro_f1_by_hand(y, pred, labels):
    scores = []
    for lab in labels:
        tp = ((pred == lab) & (y == lab)).sum()
        den = (pred == lab).sum() + (y == lab).sum()
        scores.append(2 * tp / den if den else 0.0)
    return float(np.mean(scores))


def test_confusion_counts():
    counts, cats = confusion_counts(["a", "a", "b"], ["a", "b", "b"])
    assert list(cats) == ["a", "b"]
    assert counts.tolist() == [[1, 1], [0, 1]]
    with pytest.raises(ValueError):
        confusion_counts(["a"], ["a", "b"])


def test_point_estimates_match_direct_computation():
    y, pred = _predictions()
    pred[:5] = "Unknown"                                   # a label outside the scored families
    ci = bootstrap_ci(y, pred, n_boot=200)
    assert ci["macro_f1"]["estimate"] == pytest.approx(_macro_f1_by_hand(y, pred, FAMILIES))
    benign = y == "Benign"
    assert ci["benign_fpr_per_10k"]["estimate"] == pytest.approx((pred[benign] != "Benign").mean() * 1e4)
    assert ci["detection_rate"]["estimate"] == pytest.approx((pred[~benign] != "Benign").mean())


def test_interval_contains_the_estimate_and_shrinks_with_more_data():
    small = bootstrap_ci(*_predictions(n=500), n_boot=500)["macro_f1"]
    large = bootstrap_ci(*_predictions(n=50_000), n_boot=500)["macro_f1"]
    for ci in (small, large):
        assert ci["low"] <= ci["estimate"] <= ci["high"]
    assert (large["high"] - large["low"]) < (small["high"] - small["low"]) / 3


def test_multinomial_bootstrap_matches_row_bootstrap():
    """Resampling confusion cells is the same distribution as resampling rows."""
    y, pred = _predictions(n=2000, error=0.2)
    fast = bootstrap_ci(y, pred, n_boot=2000, seed=1)["macro_f1"]
    rng = np.random.default_rng(2)
    slow = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        slow.append(_macro_f1_by_hand(y[idx], pred[idx], FAMILIES))
    assert fast["std"] == pytest.approx(np.std(slow, ddof=1), rel=0.1)
    assert fast["low"] == pytest.approx(np.quantile(slow, 0.025), abs=0.005)


def test_bootstrap_is_seeded_and_labels_are_fixed():
    y, pred = _predictions()
    assert bootstrap_ci(y, pred, seed=3) == bootstrap_ci(y, pred, seed=3)
    only_two = bootstrap_ci(y, pred, labels=["Benign", "DoS"])["macro_f1"]["estimate"]
    assert only_two == pytest.approx(_macro_f1_by_hand(y, pred, ["Benign", "DoS"]))
    with pytest.raises(ValueError):
        bootstrap_ci(y, pred, labels=["Nothing"])


def test_macro_f1_batch_shape():
    counts = np.array([[[5, 1], [0, 4]], [[3, 3], [3, 1]]])
    out = _macro_f1(counts, np.array([0, 1]))
    assert out.shape == (2,)


def test_mcnemar_exact_small_counts():
    y = np.array(["Benign"] * 20, dtype=object)
    a = y.copy()
    b = y.copy()
    b[:10] = "DoS"                                   # B wrong on 10 flows where A is right
    r = mcnemar(y, a, b)
    assert (r["a_right_b_wrong"], r["a_wrong_b_right"]) == (10, 0)
    assert r["method"] == "exact binomial"
    assert r["p_value"] == pytest.approx(2 * 0.5 ** 10)
    assert mcnemar(y, a, a)["p_value"] == 1.0


def test_mcnemar_chi_squared_and_symmetry():
    y = np.array(["Benign"] * 1000, dtype=object)
    a, b = y.copy(), y.copy()
    b[:80] = "DoS"
    a[80:120] = "DoS"
    r = mcnemar(y, a, b)
    assert r["method"].startswith("chi-squared")
    stat = (abs(80 - 40) - 1) ** 2 / 120
    assert r["statistic"] == pytest.approx(stat)
    assert r["p_value"] == pytest.approx(math.erfc(math.sqrt(stat / 2)))
    assert mcnemar(y, b, a)["p_value"] == pytest.approx(r["p_value"])


def test_mcnemar_binary_ignores_wrong_family():
    y = np.array(["DoS"] * 30, dtype=object)
    a = np.array(["PortScan"] * 30, dtype=object)  # wrong family, but it does alert
    assert mcnemar(y, a, y, binary=True)["a_wrong_b_right"] == 0
    assert mcnemar(y, a, y, binary=False)["a_wrong_b_right"] == 30


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_pairwise_mcnemar():
    y, good = _predictions(error=0.05, seed=0)
    _, bad = _predictions(error=0.4, seed=0)
    rows = pairwise_mcnemar(y, {"good": good, "bad": bad, "same": good.copy()})
    assert len(rows) == 3
    by_pair = {(r["model_a"], r["model_b"]): r for r in rows}
    assert by_pair[("good", "bad")]["significant"] and by_pair[("good", "bad")]["better"] == "good"
    assert not by_pair[("good", "same")]["significant"]
    assert all(r["p_holm"] >= r["p_value"] for r in rows)
