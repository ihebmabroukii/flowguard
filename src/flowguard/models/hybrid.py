"""Hybrid detector: a classifier for known attacks plus an anomaly gate for unknown ones.

Both parts are ours by default: the NumPy MLP (models/scratch/mlp.py) and the NumPy Isolation
Forest (models/scratch/isolation_forest.py), trained on benign flows only.

Decision rule for one flow:
    c = max over attack families of P(family | flow)       (classifier confidence)
    a = Isolation Forest anomaly score of the flow          (0..1, higher = stranger)

    if   c >= tau     -> alert: known attack, family = argmax over attack families
    elif a >= theta   -> alert: Unknown (looks like no benign traffic, matches no known attack)
    else              -> Benign

Calibration (validation flows only; test data is never touched):
    0. Optional temperature scaling (models/calibration.py): one parameter T, fitted on all
       validation flows, makes the classifier's confidences honest before they are thresholded.
       It never changes which family the classifier names.
    The benign false-alert budget `max_benign_fpr` (1 per 1,000 flows by default) is shared.
    tau takes `classifier_fpr_share` of it: the lowest threshold whose benign alert rate fits.
    theta takes what is left, measured on the benign flows that tau did not already flag.
    So on validation, total benign alerts <= max_benign_fpr by construction.

    python -m flowguard.models.hybrid          # after `make train`; writes models/p1/hybrid.joblib
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np

from flowguard.data.schema import BENIGN, UNKNOWN

log = logging.getLogger(__name__)

KNOWN, ANOMALY, NORMAL = "known_attack", "unknown_anomaly", "benign"


def threshold_for_count(scores: Any, max_alerts: int) -> float:
    """Lowest threshold t such that at most `max_alerts` of `scores` satisfy score >= t.

    Ties are respected: if several flows share the cut-off score, all of them stay below t.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    if len(s) == 0:
        return float("-inf")
    k = int(max(0, max_alerts))
    if k >= len(s):
        return float(s[-1])
    # Flag only scores strictly greater than the (k+1)-th largest one.
    return float(np.nextafter(s[k], np.inf))


def threshold_for_fpr(scores: Any, max_rate: float) -> float:
    """Lowest threshold whose alert rate on `scores` is at most `max_rate`."""
    n = len(np.asarray(scores))
    return threshold_for_count(scores, int(np.floor(max_rate * n + 1e-9)))


class HybridDetector:
    """Combine a fitted family classifier and an anomaly model.

    classifier: anything with `classes_` (family names) and `predict_proba(X)`; FlowClassifier fits.
    anomaly:    anything with `fit(X)` and `score_samples(X)` (higher = more anomalous).
    """

    def __init__(self, classifier: Any, anomaly: Any, max_benign_fpr: float = 0.001,
                 classifier_fpr_share: float = 0.5, temperature_scaling: bool = False):
        if not 0 < classifier_fpr_share <= 1:
            raise ValueError("classifier_fpr_share must be in (0, 1]")
        self.classifier = classifier
        self.anomaly = anomaly
        self.max_benign_fpr = max_benign_fpr
        self.classifier_fpr_share = classifier_fpr_share
        self.temperature_scaling = temperature_scaling

    # ------------------------------------------------------------------ training
    def fit_anomaly(self, X_train: Any, y_train: Any) -> HybridDetector:
        """Fit the anomaly model on benign training flows only."""
        benign = np.asarray(y_train, dtype=object) == BENIGN
        if not benign.any():
            raise ValueError("No benign flows to fit the anomaly model on")
        self.anomaly.fit(_rows(X_train, benign))
        return self

    def fit_temperature(self, X_val: Any, y_val: Any) -> float:
        """Step 0: learn T on the validation flows of families the classifier knows."""
        from flowguard.models.calibration import encode, fit_temperature, logits_of

        codes, known = encode(self.classifier.classes_, y_val)
        if not known.any():
            raise ValueError("No validation flows of a family the classifier knows")
        self.temperature_ = fit_temperature(logits_of(self.classifier, _rows(X_val, known)), codes[known])
        return self.temperature_

    def calibrate(self, X_val: Any, y_val: Any) -> HybridDetector:
        """Choose T (optional), then tau and theta on the benign validation flows (module docstring)."""
        self.temperature_ = 1.0
        if self.temperature_scaling:
            self.fit_temperature(X_val, y_val)
        benign = np.asarray(y_val, dtype=object) == BENIGN
        n = int(benign.sum())
        if n == 0:
            raise ValueError("No benign validation flows to calibrate on")
        X_benign = _rows(X_val, benign)
        conf, _ = self._attack_confidence(X_benign)
        budget = int(np.floor(self.max_benign_fpr * n + 1e-9))

        self.tau_ = max(threshold_for_count(conf, int(np.floor(budget * self.classifier_fpr_share))), 0.0)
        flagged_by_classifier = conf >= self.tau_
        remaining = budget - int(flagged_by_classifier.sum())

        anomaly_scores = self.anomaly.score_samples(_rows(X_benign, ~flagged_by_classifier))
        self.theta_ = threshold_for_count(anomaly_scores, remaining)

        self.calibration_ = {
            "temperature": self.temperature_,
            "n_benign_val": n,
            "alert_budget": budget,
            "tau": self.tau_,
            "theta": self.theta_,
            "benign_alerts_classifier": int(flagged_by_classifier.sum()),
            "benign_alerts_anomaly": int((anomaly_scores >= self.theta_).sum()),
        }
        self.calibration_["benign_fpr_val"] = (
            self.calibration_["benign_alerts_classifier"] + self.calibration_["benign_alerts_anomaly"]) / n
        # If the classifier gives some benign validation flow a confidence of exactly 1.0, no
        # threshold <= 1 fits the budget and the known-attack branch can never fire. That is an
        # over-confident classifier, not something to hide: it is flagged here and logged.
        self.calibration_["classifier_can_alert"] = bool(self.tau_ <= 1.0)
        if not self.calibration_["classifier_can_alert"]:
            log.warning("tau > 1: the classifier is fully confident on some benign flow; "
                        "only the anomaly gate can alert. Consider more regularization.")
        return self

    def fit(self, X_train: Any, y_train: Any, X_val: Any, y_val: Any,
            fit_classifier: bool = True) -> HybridDetector:
        if fit_classifier:
            self.classifier.fit(X_train, y_train, X_val, y_val)
        self.fit_anomaly(X_train, y_train)
        return self.calibrate(X_val, y_val)

    # ------------------------------------------------------------------ inference
    @property
    def attack_classes_(self) -> np.ndarray:
        return np.array([c for c in self.classifier.classes_ if c != BENIGN], dtype=object)

    def _attack_confidence(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """(max attack probability, most likely attack family) per flow."""
        temperature = getattr(self, "temperature_", 1.0)
        if temperature == 1.0:
            proba = np.asarray(self.classifier.predict_proba(X))
        else:
            from flowguard.models.calibration import logits_of
            from flowguard.models.scratch.core import softmax

            proba = softmax(logits_of(self.classifier, X) / temperature)
        attack_cols = [i for i, c in enumerate(self.classifier.classes_) if c != BENIGN]
        if not attack_cols:
            return np.zeros(len(proba)), np.full(len(proba), BENIGN, dtype=object)
        attack_proba = proba[:, attack_cols]
        best = attack_proba.argmax(axis=1)
        return attack_proba[np.arange(len(proba)), best], self.attack_classes_[best]

    def decide(self, X: Any) -> dict[str, np.ndarray]:
        """Everything behind each decision, as aligned arrays (the dashboard shows these)."""
        if not hasattr(self, "tau_"):
            raise RuntimeError("Call fit() or calibrate() before predicting")
        conf, family = self._attack_confidence(X)
        anomaly = self.anomaly.score_samples(X)
        known = conf >= self.tau_
        unknown = ~known & (anomaly >= self.theta_)
        prediction = np.where(known, family, np.where(unknown, UNKNOWN, BENIGN)).astype(object)
        reason = np.where(known, KNOWN, np.where(unknown, ANOMALY, NORMAL)).astype(object)
        return {
            "prediction": prediction,
            "reason": reason,
            "attack_confidence": conf,
            "likely_family": family,
            "anomaly_score": anomaly,
        }

    def predict(self, X: Any) -> np.ndarray:
        return self.decide(X)["prediction"]

    def attack_score(self, X: Any) -> np.ndarray:
        """One continuous alert score for PR curves: how far past its own threshold each part is.

        Each part is rescaled so its threshold maps to 1; the larger of the two is used.
        """
        d = self.decide(X)
        c = d["attack_confidence"] / max(self.tau_, 1e-12)
        a = d["anomaly_score"] / max(self.theta_, 1e-12) if np.isfinite(self.theta_) else 0.0
        return np.maximum(c, a)

    def summary(self) -> dict[str, Any]:
        return {
            "model": "hybrid",
            "classifier": getattr(self.classifier, "name", type(self.classifier).__name__),
            "anomaly": type(self.anomaly).__name__,
            "max_benign_fpr": self.max_benign_fpr,
            "classifier_fpr_share": self.classifier_fpr_share,
            "temperature_scaling": self.temperature_scaling,
            "calibration": getattr(self, "calibration_", None),
        }


def _rows(X: Any, mask: np.ndarray) -> Any:
    """Select rows from a DataFrame or an array with a boolean mask."""
    return X[mask] if not hasattr(X, "iloc") else X.iloc[np.flatnonzero(mask)]


# ---------------------------------------------------------------------- CLI

def build_hybrid(exp_cfg: dict[str, Any], classifier: Any, seed: int = 42) -> HybridDetector:
    from flowguard.models.anomaly import make_isolation_forest

    hcfg = exp_cfg.get("hybrid", {})
    anomaly = make_isolation_forest(exp_cfg["models"].get(hcfg.get("anomaly", "isolation_forest"), {}), seed)
    return HybridDetector(classifier, anomaly,
                          max_benign_fpr=float(hcfg.get("max_benign_fpr", 0.001)),
                          classifier_fpr_share=float(hcfg.get("classifier_fpr_share", 0.5)),
                          temperature_scaling=bool(hcfg.get("temperature_scaling", False)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build and calibrate the P1 hybrid detector")
    parser.add_argument("--retrain-classifier", action="store_true",
                        help="train the classifier again instead of loading models/p1/<name>.joblib")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import joblib

    from flowguard.config import Paths, load_config
    from flowguard.data.io import SPLIT, load_p1
    from flowguard.data.schema import FAMILY
    from flowguard.eval.metrics import classification_metrics
    from flowguard.features import pipeline as fp
    from flowguard.models import build_classifier

    paths = Paths.default()
    exp_cfg = load_config("experiments", paths.root)
    seed = exp_cfg.get("seed", 42)
    name = exp_cfg.get("hybrid", {}).get("classifier", "mlp")

    pipe = fp.load(paths.models / fp.PIPELINE_FILE)
    df = load_p1(paths)
    train, val = df[df[SPLIT] == "train"], df[df[SPLIT] == "val"]
    X_train, X_val = fp.transform(pipe, train), fp.transform(pipe, val)
    y_train, y_val = train[FAMILY].to_numpy(), val[FAMILY].to_numpy()

    saved = paths.models / "p1" / f"{name}.joblib"
    reuse = saved.exists() and not args.retrain_classifier
    from flowguard.models.tune import tuned_models_cfg

    classifier = joblib.load(saved) if reuse else build_classifier(name, tuned_models_cfg(exp_cfg, paths), seed)
    log.info("%s classifier %s", "loaded" if reuse else "training", name)

    hybrid = build_hybrid(exp_cfg, classifier, seed)
    hybrid.fit(X_train, y_train, X_val, y_val, fit_classifier=not reuse)
    pred = hybrid.predict(X_val)
    record = {**hybrid.summary(), "validation": classification_metrics(y_val, pred)}

    (paths.models / "p1").mkdir(parents=True, exist_ok=True)
    joblib.dump(hybrid, paths.models / "p1" / "hybrid.joblib")
    paths.tables.mkdir(parents=True, exist_ok=True)
    (paths.tables / "p1_val_hybrid.json").write_text(json.dumps(record, indent=2, default=str))
    log.info("hybrid: T %.3f, tau %.4f, theta %.4f, val benign FPR %.2f/10k, val macro-F1 %.4f",
             hybrid.temperature_, hybrid.tau_, hybrid.theta_, record["calibration"]["benign_fpr_val"] * 10_000,
             record["validation"]["macro_f1"])


if __name__ == "__main__":
    main()
