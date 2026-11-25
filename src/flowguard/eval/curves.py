"""Loss curves and learning curves for our gradient-trained models: the bias/variance discussion.

Loss curves (no retraining)
    Training and validation loss per epoch, read from the models saved by `make train`
    (GradientClassifier.history_). The epoch whose weights were kept is marked: early stopping
    restores the epoch with the lowest validation loss.
    Reading them: both losses high and flat -> underfitting (bias); validation loss turning back
    up while training loss keeps falling -> overfitting (variance).

Learning curves (retraining)
    Each model is retrained on growing stratified fractions of the P1 training split. Training
    and validation macro-F1 are recorded for each size.
    Reading them: a large gap that closes as data grows -> variance, more data helps; both
    curves low and close together -> bias, a bigger model helps more than more data.

Outputs: reports/tables/loss_curve_<model>.csv, reports/tables/learning_curve.csv

    python -m flowguard.eval.curves                  # both
    python -m flowguard.eval.curves --skip-learning  # loss curves only (seconds)
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

TRAIN_SCORE_ROWS = 200_000  # training macro-F1 is measured on at most this many training rows


def loss_curve(model: Any) -> list[dict[str, Any]]:
    """Per-epoch losses of a fitted FlowClassifier around one of our GradientClassifiers."""
    estimator = getattr(model, "estimator", model)
    history = getattr(estimator, "history_", None)
    if not history:
        raise ValueError(f"{getattr(model, 'name', model)} has no training history (not one of our gradient models)")
    monitor = "val_loss" if "val_loss" in history[0] else "train_loss"
    best_epoch = min(history, key=lambda r: r[monitor])["epoch"]
    return [{**row, "kept": row["epoch"] == best_epoch} for row in history]


def learning_curve(name: str, X_train: Any, y_train: Any, X_val: Any, y_val: Any, models_cfg: dict[str, Any],
                   fractions: list[float], seed: int = 42) -> list[dict[str, Any]]:
    from flowguard.eval.metrics import classification_metrics
    from flowguard.models import build_classifier
    from flowguard.models.base import stratified_cap

    y_train = np.asarray(y_train, dtype=object)
    rows = []
    for fraction in sorted(fractions):
        idx = stratified_cap(y_train, int(round(fraction * len(y_train))), seed=seed)
        X_sub, y_sub = X_train.iloc[idx], y_train[idx]
        start = time.perf_counter()
        cfg = {name: {**(models_cfg.get(name) or {}), "max_train_rows": None}}
        model = build_classifier(name, cfg, seed).fit(X_sub, y_sub, X_val, y_val)
        seconds = time.perf_counter() - start
        score_idx = stratified_cap(y_sub, TRAIN_SCORE_ROWS, seed=seed)
        train_f1 = classification_metrics(y_sub[score_idx], model.predict(X_sub.iloc[score_idx]))["macro_f1"]
        val_f1 = classification_metrics(y_val, model.predict(X_val))["macro_f1"]
        rows.append({"model": name, "fraction": fraction, "n_train_rows": len(idx),
                     "train_macro_f1": train_f1, "val_macro_f1": val_f1, "gap": train_f1 - val_f1,
                     "best_val_loss": getattr(model.estimator, "best_loss_", float("nan")),
                     "epochs_run": getattr(model.estimator, "n_epochs_", None), "seconds": round(seconds, 1)})
        log.info("%s on %6.1f%% of train (%d rows): train F1 %.4f, val F1 %.4f",
                 name, 100 * fraction, len(idx), train_f1, val_f1)
    return rows


def run(models: list[str] | None = None, learning: bool = True, paths: Any = None) -> dict[str, Any]:
    import pandas as pd

    from flowguard.config import Paths, load_config
    from flowguard.eval.protocols import load_model
    from flowguard.models.tune import tuned_models_cfg

    paths = paths or Paths.default()
    exp_cfg = load_config("experiments", paths.root)
    ccfg = exp_cfg.get("curves", {})
    models = models or ccfg.get("models", ["softmax_regression", "mlp"])
    paths.tables.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"loss_curves": [], "learning_curve": None}

    for name in models:
        try:
            curve = loss_curve(load_model(paths, name))
        except (FileNotFoundError, ValueError) as err:
            log.warning("loss curve for %s skipped: %s", name, err)
            continue
        pd.DataFrame(curve).to_csv(paths.tables / f"loss_curve_{name}.csv", index=False)
        out["loss_curves"].append(name)

    if learning:
        from flowguard.data.io import SPLIT, load_p1
        from flowguard.data.schema import FAMILY
        from flowguard.data.split import TRAIN, VAL
        from flowguard.features import pipeline as fp

        pipe = fp.load(paths.models / fp.PIPELINE_FILE)
        df = load_p1(paths)
        train, val = df[df[SPLIT] == TRAIN], df[df[SPLIT] == VAL]
        X_train, X_val = fp.transform(pipe, train), fp.transform(pipe, val)
        y_train, y_val = train[FAMILY].to_numpy(), val[FAMILY].to_numpy()
        fractions = [float(f) for f in ccfg.get("fractions", [0.01, 0.03, 0.1, 0.3, 1.0])]
        models_cfg = tuned_models_cfg(exp_cfg, paths)
        rows = [row for name in models for row in
                learning_curve(name, X_train, y_train, X_val, y_val, models_cfg, fractions, int(exp_cfg.get("seed", 42)))]
        pd.DataFrame(rows).to_csv(paths.tables / "learning_curve.csv", index=False)
        out["learning_curve"] = "learning_curve.csv"
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Loss curves and learning curves for our models")
    parser.add_argument("--models", nargs="+", help="default: config curves.models")
    parser.add_argument("--skip-learning", action="store_true", help="only read the saved loss histories")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.models, learning=not args.skip_learning)


if __name__ == "__main__":
    main()
