"""Hybrid detector tests.

Most tests use a tiny NumPy-only label adapter around our SoftmaxRegression, so they run
without pandas or scikit-learn. The last one uses the real FlowClassifier + MLP.
"""

import pickle

import numpy as np
import pytest

from flowguard.data.schema import BENIGN, UNKNOWN
from flowguard.models.hybrid import (
    ANOMALY,
    KNOWN,
    NORMAL,
    HybridDetector,
    threshold_for_count,
    threshold_for_fpr,
)
from flowguard.models.scratch import IsolationForest, SoftmaxRegression

CENTERS = {"Benign": (0, 0, 0, 0), "DoS": (6, 0, 0, 0), "PortScan": (0, 6, 0, 0)}
UNSEEN = ("Botnet", (0, 0, 9, 9))  # never in training


class Encoded:
    """String labels <-> integers around one of our estimators (NumPy only)."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.name = type(estimator).__name__

    def fit(self, X, y, X_val=None, y_val=None):
        self.classes_, codes = np.unique(np.asarray(y, dtype=object), return_inverse=True)
        self.estimator.fit(X, codes)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


def flows(spec, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for family, n in spec.items():
        center = CENTERS.get(family, UNSEEN[1])
        X.append(rng.normal(center, 1.0, (n, 4)))
        y += [family] * n
    return np.vstack(X), np.array(y, dtype=object)


TRAIN = {"Benign": 3000, "DoS": 400, "PortScan": 400}


def fitted_hybrid(max_fpr=0.01, share=0.5):
    X, y = flows(TRAIN, seed=0)
    X_val, y_val = flows(TRAIN, seed=1)
    clf = Encoded(SoftmaxRegression(lr=0.05, epochs=40, batch_size=128))
    hybrid = HybridDetector(clf, IsolationForest(n_estimators=100, random_state=0),
                            max_benign_fpr=max_fpr, classifier_fpr_share=share)
    return hybrid.fit(X, y, X_val, y_val), (X_val, y_val)


# ----------------------------------------------------------------------------- thresholds

def test_threshold_for_count_respects_the_budget_with_ties():
    scores = np.array([0.1, 0.5, 0.9, 0.9, 0.9, 0.3])
    for k in range(7):
        t = threshold_for_count(scores, k)
        assert (scores >= t).sum() <= k
    assert (scores >= threshold_for_count(scores, 3)).sum() == 3      # the three tied 0.9s fit exactly
    assert (scores >= threshold_for_count(scores, 2)).sum() == 0      # they can't be split, so none
    assert threshold_for_count(scores, 10) == 0.1                    # budget covers everything


def test_threshold_for_fpr():
    scores = np.linspace(0, 1, 1000)
    t = threshold_for_fpr(scores, 0.01)
    assert (scores >= t).sum() == 10


# ----------------------------------------------------------------------------- behaviour

def test_benign_false_alerts_stay_within_budget_on_validation():
    hybrid, (X_val, y_val) = fitted_hybrid(max_fpr=0.01)
    cal = hybrid.calibration_
    assert cal["alert_budget"] == 30                          # 1% of 3000 benign flows
    assert cal["benign_alerts_classifier"] <= 15             # half the budget for tau
    assert cal["benign_fpr_val"] <= 0.01
    pred = hybrid.predict(X_val)
    assert (pred[y_val == BENIGN] != BENIGN).mean() <= 0.01
    assert cal["classifier_can_alert"]


def test_known_attacks_get_their_family():
    hybrid, (X_val, y_val) = fitted_hybrid()
    pred = hybrid.predict(X_val)
    for family in ("DoS", "PortScan"):
        assert (pred[y_val == family] == family).mean() > 0.9


def test_unseen_attack_is_flagged_unknown():
    hybrid, _ = fitted_hybrid()
    X_new, _ = flows({UNSEEN[0]: 300}, seed=2)
    d = hybrid.decide(X_new)
    # The softmax model only knows Benign/DoS/PortScan, so it can't name this traffic; the
    # anomaly gate catches it because it looks like no benign flow.
    assert (d["prediction"] != BENIGN).mean() > 0.9
    assert (d["prediction"] == UNKNOWN).mean() > 0.5


def test_decide_is_consistent():
    hybrid, (X_val, _) = fitted_hybrid()
    d = hybrid.decide(X_val)
    known = d["reason"] == KNOWN
    assert (d["attack_confidence"][known] >= hybrid.tau_).all()
    assert (d["prediction"][known] == d["likely_family"][known]).all()
    unknown = d["reason"] == ANOMALY
    assert (d["prediction"][unknown] == UNKNOWN).all()
    assert (d["anomaly_score"][unknown] >= hybrid.theta_).all()
    normal = d["reason"] == NORMAL
    assert (d["prediction"][normal] == BENIGN).all()
    assert set(d["likely_family"]) <= {"DoS", "PortScan"}
    assert len({len(v) for v in d.values()}) == 1


def test_a_stricter_budget_means_fewer_benign_alerts():
    loose, (X_val, y_val) = fitted_hybrid(max_fpr=0.05)
    strict, _ = fitted_hybrid(max_fpr=0.001)
    rate = lambda h: (h.predict(X_val)[y_val == BENIGN] != BENIGN).mean()
    assert rate(strict) <= rate(loose)
    assert strict.tau_ >= loose.tau_


def test_attack_score_ranks_attacks_above_benign():
    hybrid, (X_val, y_val) = fitted_hybrid()
    s = hybrid.attack_score(X_val)
    assert np.median(s[y_val != BENIGN]) > np.median(s[y_val == BENIGN])


def test_predict_before_calibration_fails_and_pickle_roundtrip():
    clf = Encoded(SoftmaxRegression())
    with pytest.raises(RuntimeError):
        HybridDetector(clf, IsolationForest()).predict(np.zeros((1, 4)))
    hybrid, (X_val, _) = fitted_hybrid()
    again = pickle.loads(pickle.dumps(hybrid))
    assert (again.predict(X_val) == hybrid.predict(X_val)).all()
    assert again.summary()["calibration"]["tau"] == hybrid.tau_


def test_anomaly_model_sees_only_benign_training_flows():
    seen = {}

    class Spy(IsolationForest):
        def fit(self, X, y=None):
            seen["n"] = len(X)
            return super().fit(X)

    X, y = flows(TRAIN)
    hybrid = HybridDetector(Encoded(SoftmaxRegression(epochs=3)), Spy(n_estimators=5))
    hybrid.fit(X, y, *flows(TRAIN, seed=1))
    assert seen["n"] == TRAIN["Benign"]


def test_with_real_flow_classifier_and_our_mlp():
    pytest.importorskip("sklearn")
    pd = pytest.importorskip("pandas")
    from flowguard.models.mlp import make_mlp

    X, y = flows(TRAIN)
    X_val, y_val = flows(TRAIN, seed=1)
    cols = [f"f{i}" for i in range(4)]
    clf = make_mlp({"hidden": [16], "lr": 0.01, "epochs": 30, "batch_size": 128, "dropout": 0.0})
    hybrid = HybridDetector(clf, IsolationForest(n_estimators=50), max_benign_fpr=0.01)
    hybrid.fit(pd.DataFrame(X, columns=cols), y, pd.DataFrame(X_val, columns=cols), y_val)
    assert hybrid.calibration_["benign_fpr_val"] <= 0.01
    assert hybrid.summary()["classifier"] == "mlp"
