"""Train the P1 classifiers and score them on the validation split.

Test-set numbers are produced later by eval/protocols.py, so nothing here can leak into model
choices. Each model is saved to models/p1/<name>.joblib, its validation metrics go to
reports/tables/p1_val_<name>.json, and the run is logged to MLflow when it is installed.

    python -m flowguard.models.train                       # every classifier
    python -m flowguard.models.train --models logreg xgboost --no-mlflow
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import joblib
import pandas as pd

from flowguard import tracking
from flowguard.config import Paths, load_config
from flowguard.data.io import SPLIT, load_p1
from flowguard.data.schema import FAMILY
from flowguard.eval.metrics import attack_pr_auc, classification_metrics, flatten
from flowguard.features import pipeline as fp
from flowguard.models import CLASSIFIERS, FlowClassifier, build_classifier
from flowguard.models.tune import tuned_file, tuned_models_cfg

log = logging.getLogger(__name__)


def train_and_validate(
    name: str,
    X_train: pd.DataFrame,
    y_train: Any,
    X_val: pd.DataFrame,
    y_val: Any,
    models_cfg: dict[str, Any],
    seed: int = 42,
) -> tuple[FlowClassifier, dict[str, Any]]:
    model = build_classifier(name, models_cfg, seed)
    model.fit(X_train, y_train, X_val, y_val)
    proba = model.predict_proba(X_val)
    pred = model.classes_[proba.argmax(axis=1)]
    metrics = classification_metrics(y_val, pred)
    metrics["pr_auc"] = attack_pr_auc(y_val, model.attack_score(X_val, proba))
    return model, metrics


def run(model_names: list[str], use_mlflow: bool = True, paths: Paths | None = None) -> dict[str, dict]:
    paths = paths or Paths.default()
    exp_cfg = load_config("experiments", paths.root)
    seed = exp_cfg.get("seed", 42)

    pipe_path = paths.models / fp.PIPELINE_FILE
    if not pipe_path.exists():
        raise FileNotFoundError(f"{pipe_path} not found. Run `python -m flowguard.features.pipeline` first.")
    pipe = fp.load(pipe_path)

    df = load_p1(paths)
    train, val = df[df[SPLIT] == "train"], df[df[SPLIT] == "val"]
    X_train, X_val = fp.transform(pipe, train), fp.transform(pipe, val)
    y_train, y_val = train[FAMILY].to_numpy(), val[FAMILY].to_numpy()
    log.info("train %d rows, validation %d rows, %d features", len(X_train), len(X_val), X_train.shape[1])

    models_cfg = tuned_models_cfg(exp_cfg, paths)
    apply_tuned = bool(exp_cfg.get("tuning", {}).get("apply_best", False))
    mlflow = tracking.get_mlflow(use_mlflow, paths, exp_cfg.get("mlflow_experiment", "flowguard"))
    out_dir = paths.models / "p1"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths.tables.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for name in model_names:
        with tracking.run(mlflow, f"P1-train-{name}", {"protocol": "P1", "stage": "validation", "model": name}):
            log.info("training %s", name)
            model, metrics = train_and_validate(name, X_train, y_train, X_val, y_val, models_cfg, seed)
            joblib.dump(model, out_dir / f"{name}.joblib")
            record = {**model.summary(), "tuned": apply_tuned and tuned_file(paths, name).exists(),
                      "validation": metrics}
            if hasattr(model.estimator, "history_"):
                record["history"] = model.estimator.history_
            (paths.tables / f"p1_val_{name}.json").write_text(json.dumps(record, indent=2, default=str))
            if mlflow:
                mlflow.log_params({k: v for k, v in record["params"].items() if v is not None})
                mlflow.log_metrics(flatten(metrics, "val."))
                mlflow.log_metric("fit_seconds", record["fit_seconds"])
            results[name] = record
            log.info("%s: val macro-F1 %.4f, benign FPR %.1f/10k, %.0fs",
                     name, metrics["macro_f1"], metrics["benign_fpr_per_10k"], record["fit_seconds"])
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", choices=list(CLASSIFIERS), default=list(CLASSIFIERS))
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.models, use_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
