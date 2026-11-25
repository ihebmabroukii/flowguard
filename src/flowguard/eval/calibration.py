"""Can the models' probabilities be trusted? Reliability diagrams, ECE, and temperature scaling.

For each model, on the P1 test split:
    confidence = the probability of the predicted family, correct = it is the true family.
    A calibrated model is right 90% of the time when it says 90%.

    ECE   expected calibration error: sum over confidence bins of (bin share) * |accuracy - confidence|
    NLL   mean negative log-likelihood of the true family
    Brier mean squared error between the probability vector and the one-hot true family

Each is reported before and after temperature scaling (T fitted on the validation split only,
models/calibration.py). Outputs:
    reports/tables/calibration_<model>.json     ECE, NLL, Brier before/after, T
    reports/tables/reliability_<model>.csv      one row per bin, before and after (the figure's data)

    python -m flowguard.eval.calibration --models softmax_regression mlp xgboost
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np

from flowguard.models.calibration import encode, fit_temperature, logits_of, negative_log_likelihood
from flowguard.models.scratch.core import softmax

log = logging.getLogger(__name__)

N_BINS = 15


# --------------------------------------------------------------------------- metrics (NumPy only)

def reliability_bins(confidence: Any, correct: Any, n_bins: int = N_BINS) -> dict[str, np.ndarray]:
    """Equal-width confidence bins on (0, 1]: count, mean confidence and accuracy per bin."""
    conf = np.asarray(confidence, dtype=np.float64)
    right = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sum_conf = np.bincount(idx, weights=conf, minlength=n_bins)
    sum_right = np.bincount(idx, weights=right, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"bin_low": edges[:-1], "bin_high": edges[1:], "count": count,
                "mean_confidence": np.where(count > 0, sum_conf / count, np.nan),
                "accuracy": np.where(count > 0, sum_right / count, np.nan)}


def expected_calibration_error(confidence: Any, correct: Any, n_bins: int = N_BINS) -> float:
    bins = reliability_bins(confidence, correct, n_bins)
    used = bins["count"] > 0
    weights = bins["count"][used] / bins["count"].sum()
    return float((weights * np.abs(bins["accuracy"][used] - bins["mean_confidence"][used])).sum())


def brier_score(proba: Any, y: Any) -> float:
    p = np.asarray(proba, dtype=np.float64)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(p)), np.asarray(y, dtype=np.int64)] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def calibration_summary(logits: Any, y: Any, temperature: float = 1.0, n_bins: int = N_BINS) -> dict[str, Any]:
    """ECE, NLL, Brier and reliability bins of softmax(logits / T) against integer labels y."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    proba = softmax(z / temperature)
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y
    return {
        "temperature": float(temperature),
        "ece": expected_calibration_error(confidence, correct, n_bins),
        "nll": negative_log_likelihood(z, y, temperature),
        "brier": brier_score(proba, y),
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
        "saturated_share": float((confidence >= 1.0).mean()),  # confidences that are exactly 1.0
        "bins": reliability_bins(confidence, correct, n_bins),
    }


def evaluate_calibration(classifier: Any, X_val: Any, y_val: Any, X_test: Any, y_test: Any,
                         n_bins: int = N_BINS) -> dict[str, Any]:
    """Fit T on validation, report test calibration before and after. Unknown families are skipped."""
    classes = classifier.classes_
    val_codes, val_known = encode(classes, y_val)
    test_codes, test_known = encode(classes, y_test)
    val_logits = logits_of(classifier, X_val)[val_known]
    test_logits = logits_of(classifier, X_test)[test_known]
    temperature = fit_temperature(val_logits, val_codes[val_known])
    return {
        "temperature": temperature,
        "before": calibration_summary(test_logits, test_codes[test_known], 1.0, n_bins),
        "after": calibration_summary(test_logits, test_codes[test_known], temperature, n_bins),
        "n_test": int(test_known.sum()),
    }


# --------------------------------------------------------------------------- CLI

def run(models: list[str] | None = None, paths: Any = None) -> dict[str, dict]:
    import pandas as pd

    from flowguard.data.io import SPLIT, load_p1
    from flowguard.data.schema import FAMILY
    from flowguard.data.split import TEST, VAL
    from flowguard.eval.protocols import load_context, load_model
    from flowguard.features import pipeline as fp

    ctx = load_context(paths, use_mlflow=False)
    models = models or ctx.cfg.get("calibration", {}).get("models", ["softmax_regression", "mlp"])
    df = load_p1(ctx.paths)
    val, test = df[df[SPLIT] == VAL], df[df[SPLIT] == TEST]
    X_val, X_test = fp.transform(ctx.pipe, val), fp.transform(ctx.pipe, test)
    y_val, y_test = val[FAMILY].to_numpy(), test[FAMILY].to_numpy()

    results = {}
    for name in models:
        try:
            model = load_model(ctx.paths, name)
        except FileNotFoundError as err:
            log.warning("calibration for %s skipped: %s", name, err)
            continue
        rec = evaluate_calibration(model, X_val, y_val, X_test, y_test)
        frames = []
        for stage in ("before", "after"):
            bins = pd.DataFrame(rec[stage].pop("bins"))
            bins.insert(0, "stage", stage)
            frames.append(bins)
        pd.concat(frames).to_csv(ctx.paths.tables / f"reliability_{name}.csv", index=False)
        (ctx.paths.tables / f"calibration_{name}.json").write_text(
            json.dumps({"model": name, **rec}, indent=2, default=float))
        results[name] = rec
        log.info("%-18s T = %.3f   ECE %.4f -> %.4f   NLL %.4f -> %.4f   saturated %.4f -> %.4f",
                 name, rec["temperature"], rec["before"]["ece"], rec["after"]["ece"],
                 rec["before"]["nll"], rec["after"]["nll"],
                 rec["before"]["saturated_share"], rec["after"]["saturated_share"])
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reliability diagrams and temperature scaling on P1")
    parser.add_argument("--models", nargs="+", help="saved P1 models (default: config calibration.models)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.models)


if __name__ == "__main__":
    main()
