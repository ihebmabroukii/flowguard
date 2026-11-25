"""Temperature scaling and calibration metrics (NumPy only), and their use in the hybrid."""

import numpy as np
import pytest

from flowguard.data.schema import BENIGN
from flowguard.eval.calibration import (
    brier_score,
    calibration_summary,
    evaluate_calibration,
    expected_calibration_error,
    reliability_bins,
)
from flowguard.models.calibration import (
    TemperatureScaled,
    encode,
    fit_temperature,
    logits_of,
    negative_log_likelihood,
)
from flowguard.models.hybrid import HybridDetector
from flowguard.models.scratch import IsolationForest, SoftmaxRegression
from flowguard.models.scratch.core import softmax


def _calibrated_sample(n=20_000, k=4, seed=0):
    """Logits z and labels drawn from softmax(z): calibrated at T = 1 by construction."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2.0, (n, k))
    p = softmax(z)
    y = (rng.random((n, 1)) > p.cumsum(axis=1)).sum(axis=1)
    return z, y


@pytest.mark.parametrize("true_t", [0.5, 1.0, 4.0])
def test_fit_temperature_recovers_the_true_temperature(true_t):
    z, y = _calibrated_sample()
    # Logits that are true_t times too large: the right correction is T = true_t.
    assert fit_temperature(z * true_t, y) == pytest.approx(true_t, rel=0.08)


def test_temperature_minimizes_nll_and_keeps_argmax():
    z, y = _calibrated_sample(seed=1)
    over = z * 3
    t = fit_temperature(over, y)
    assert negative_log_likelihood(over, y, t) < negative_log_likelihood(over, y, 1.0)
    for other in (t * 0.8, t * 1.25):
        assert negative_log_likelihood(over, y, t) <= negative_log_likelihood(over, y, other) + 1e-9
    assert (softmax(over / t).argmax(1) == over.argmax(1)).all()
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((0, 3)), np.zeros(0))


def test_ece_is_small_when_calibrated_and_large_when_not():
    z, y = _calibrated_sample()
    good = calibration_summary(z, y)
    bad = calibration_summary(z * 5, y)
    assert good["ece"] < 0.02
    assert bad["ece"] > 0.1
    assert bad["mean_confidence"] > bad["accuracy"]          # over-confident
    fixed = calibration_summary(z * 5, y, temperature=fit_temperature(z * 5, y))
    assert fixed["ece"] < 0.02


def test_reliability_bins_and_brier():
    bins = reliability_bins([0.05, 0.95, 0.95, 1.0], [0, 1, 0, 1], n_bins=10)
    assert bins["count"].sum() == 4
    assert bins["count"][0] == 1 and bins["count"][-1] == 3
    assert bins["accuracy"][-1] == pytest.approx(2 / 3)
    assert expected_calibration_error([1.0, 1.0], [1, 1]) == 0.0
    assert brier_score(np.array([[1.0, 0.0], [0.5, 0.5]]), [0, 1]) == pytest.approx(0.25)


class Encoded:
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, X_val=None, y_val=None):
        self.classes_, codes = np.unique(np.asarray(y, dtype=object), return_inverse=True)
        self.estimator.fit(X, codes)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class ProbaOnly:
    """A classifier that only exposes probabilities (like a tree ensemble)."""

    def __init__(self, inner):
        self.inner = inner
        self.classes_ = inner.classes_

    def predict_proba(self, X):
        return self.inner.predict_proba(X)


def _blobs(n, seed, overlap=1.5):
    rng = np.random.default_rng(seed)
    y = rng.choice(np.array(["Benign", "DoS", "PortScan"], dtype=object), n, p=[0.7, 0.2, 0.1])
    centers = {"Benign": (0, 0), "DoS": (overlap, 0), "PortScan": (0, overlap)}
    X = np.array([centers[v] for v in y]) + rng.normal(0, 1, (n, 2))
    return X, y


def test_logits_of_uses_real_logits_or_log_proba():
    X, y = _blobs(2000, 0)
    clf = Encoded(SoftmaxRegression(lr=0.05, epochs=20, batch_size=128)).fit(X, y)
    real = logits_of(clf, X)
    assert np.allclose(real, clf.estimator.decision_function(X))
    from_proba = logits_of(ProbaOnly(clf), X)
    assert np.allclose(softmax(from_proba), clf.predict_proba(X), atol=1e-9)


def test_temperature_scaled_wrapper_and_encode():
    X, y = _blobs(3000, 0)
    X_val, y_val = _blobs(3000, 1)
    clf = Encoded(SoftmaxRegression(lr=0.05, epochs=20, batch_size=128)).fit(X, y)
    scaled = TemperatureScaled(clf).fit(X_val, np.r_[y_val[:-1], ["Botnet"]])   # unknown label is skipped
    assert scaled.temperature_ > 0
    assert (scaled.predict(X_val) == clf.predict(X_val)).all()
    assert np.allclose(scaled.predict_proba(X_val).sum(1), 1)
    assert scaled.attack_score(X_val).shape == (len(X_val),)
    codes, known = encode(["a", "b"], ["b", "c", "a"])
    assert codes.tolist() == [1, -1, 0] and known.tolist() == [True, False, True]


def test_evaluate_calibration_on_an_overconfident_model():
    X, y = _blobs(4000, 0)
    X_val, y_val = _blobs(4000, 1)
    X_te, y_te = _blobs(4000, 2)
    clf = Encoded(SoftmaxRegression(lr=0.05, epochs=20, batch_size=128)).fit(X, y)
    clf.estimator.W_ *= 6          # make it over-confident on purpose
    clf.estimator.b_ *= 6
    rec = evaluate_calibration(clf, X_val, y_val, X_te, y_te)
    assert rec["temperature"] > 2
    assert rec["after"]["ece"] < rec["before"]["ece"]
    assert rec["after"]["nll"] < rec["before"]["nll"]


def test_hybrid_temperature_scaling_unsaturates_benign_confidences():
    """An over-confident classifier puts benign flows at confidence 1.0, so no tau <= 1 fits the
    budget. After temperature scaling, the known-attack branch can alert again."""
    X, y = _blobs(6000, 0, overlap=3.0)
    X_val, y_val = _blobs(6000, 1, overlap=3.0)
    clf = Encoded(SoftmaxRegression(lr=0.05, epochs=30, batch_size=128)).fit(X, y)
    clf.estimator.W_ *= 60
    clf.estimator.b_ *= 60

    def build(ts):
        h = HybridDetector(clf, IsolationForest(n_estimators=50, random_state=0), max_benign_fpr=0.01,
                           temperature_scaling=ts)
        return h.fit(X, y, X_val, y_val, fit_classifier=False)

    raw, scaled = build(False), build(True)
    assert raw.calibration_["temperature"] == 1.0
    assert not raw.calibration_["classifier_can_alert"]
    assert scaled.calibration_["temperature"] > 10
    assert scaled.calibration_["classifier_can_alert"]
    assert scaled.calibration_["benign_fpr_val"] <= 0.01
    pred = scaled.predict(X_val)
    assert (pred[y_val == "DoS"] == "DoS").mean() > 0.5
    assert (pred[y_val == BENIGN] != BENIGN).mean() <= 0.011
