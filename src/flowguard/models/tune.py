"""Hyperparameter random search on the validation split (Bergstra and Bengio, 2012).

Answers "how did you choose the learning rate, the layer sizes, the regularization?".

For each model in `tuning.spaces` (configs/experiments.yaml):
    trial 0     the hand-picked values from `models:` (so the search can only match or beat them)
    trials 1..n values drawn at random from the search space
    every trial trains on a stratified subsample of the P1 training split, stops early on the
    validation loss, and is scored by validation macro-F1. Test data is never used.

Random search beats grid search when only a few hyperparameters matter, which is usual: every
trial tries a new value of each of them instead of repeating grid values.

Outputs:
    reports/tables/tuning_<model>.csv   one row per trial (the table for the report)
    models/tuning/<model>.json          the best parameters; `make train` uses them when
                                        tuning.apply_best is true

    python -m flowguard.models.tune                          # every model in tuning.spaces
    python -m flowguard.models.tune --models mlp --trials 30
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

TUNING_DIR = "tuning"


# --------------------------------------------------------------------------- search space

def sample_value(spec: Any, rng: np.random.Generator) -> Any:
    """One draw from a search-space entry.

        {log_uniform: [lo, hi]}  exp(U(log lo, log hi)), for learning rates and penalties
        {uniform: [lo, hi]}      U(lo, hi)
        {int_uniform: [lo, hi]}  integer in [lo, hi]
        {choice: [a, b, ...]}    one of the listed values
        anything else            used as a fixed value
    """
    if not isinstance(spec, dict) or len(spec) != 1:
        return spec
    (kind, args), = spec.items()
    if kind == "log_uniform":
        lo, hi = float(args[0]), float(args[1])
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    if kind == "uniform":
        return float(rng.uniform(float(args[0]), float(args[1])))
    if kind == "int_uniform":
        return int(rng.integers(int(args[0]), int(args[1]) + 1))
    if kind == "choice":
        return copy.deepcopy(args[int(rng.integers(len(args)))])
    raise ValueError(f"Unknown search-space kind {kind!r}")


def sample_params(space: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return {name: sample_value(spec, rng) for name, spec in space.items()}


def trial_configs(base: dict[str, Any], space: dict[str, Any], n_trials: int, seed: int) -> list[dict[str, Any]]:
    """Trial 0 = the base config, then n_trials - 1 random draws merged over it."""
    rng = np.random.default_rng(seed)
    trials = [copy.deepcopy(base)]
    for _ in range(max(0, n_trials - 1)):
        trials.append({**copy.deepcopy(base), **sample_params(space, rng)})
    return trials


def searched_values(trial: dict[str, Any], space: dict[str, Any]) -> dict[str, Any]:
    """Only the searched hyperparameters of a trial, as plain values for the CSV."""
    return {k: (str(list(trial[k])) if isinstance(trial.get(k), (list, tuple)) else trial.get(k))
            for k in space}


# --------------------------------------------------------------------------- tuned parameters

def tuned_file(paths: Any, name: str) -> Path:
    return paths.models / TUNING_DIR / f"{name}.json"


def tuned_models_cfg(exp_cfg: dict[str, Any], paths: Any) -> dict[str, Any]:
    """The `models:` config with the best tuned parameters merged in, when tuning.apply_best is on."""
    models_cfg = copy.deepcopy(exp_cfg["models"])
    if not exp_cfg.get("tuning", {}).get("apply_best", False):
        return models_cfg
    for name in list(models_cfg):
        path = tuned_file(paths, name)
        if path.exists():
            best = json.loads(path.read_text())["best_params"]
            models_cfg[name] = {**(models_cfg[name] or {}), **best}
            log.info("%s: using tuned parameters from %s: %s", name, path.name, best)
    return models_cfg


# --------------------------------------------------------------------------- search

def search(name: str, X_train: Any, y_train: Any, X_val: Any, y_val: Any, exp_cfg: dict[str, Any],
           n_trials: int | None = None, seed: int = 42) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the random search for one model. Returns (one row per trial, best searched params)."""
    from flowguard.eval.metrics import classification_metrics
    from flowguard.models import build_classifier

    tcfg = exp_cfg.get("tuning", {})
    space = tcfg["spaces"][name]
    n_trials = int(n_trials or tcfg.get("n_trials", 20))
    base = dict(exp_cfg["models"].get(name) or {})
    if tcfg.get("max_train_rows"):
        base["max_train_rows"] = int(tcfg["max_train_rows"])

    rows: list[dict[str, Any]] = []
    best_score, best = -math.inf, {}
    for i, params in enumerate(trial_configs(base, space, n_trials, seed)):
        start = time.perf_counter()
        model = build_classifier(name, {name: params}, seed).fit(X_train, y_train, X_val, y_val)
        pred = model.predict(X_val)
        estimator = model.estimator
        row = {"trial": i, "source": "config" if i == 0 else "random", **searched_values(params, space),
               "val_macro_f1": classification_metrics(y_val, pred)["macro_f1"],
               "best_val_loss": getattr(estimator, "best_loss_", float("nan")),
               "epochs_run": getattr(estimator, "n_epochs_", None),
               "seconds": round(time.perf_counter() - start, 1)}
        rows.append(row)
        log.info("%s trial %2d/%d  val macro-F1 %.4f  %s", name, i + 1, n_trials, row["val_macro_f1"],
                 searched_values(params, space))
        if row["val_macro_f1"] > best_score:
            best_score, best = row["val_macro_f1"], {k: params[k] for k in space if k in params}
    return rows, best


def run(model_names: list[str] | None = None, n_trials: int | None = None, paths: Any = None) -> dict[str, Any]:
    import pandas as pd

    from flowguard.config import Paths, load_config
    from flowguard.data.io import SPLIT, load_p1
    from flowguard.data.schema import FAMILY
    from flowguard.data.split import TRAIN, VAL
    from flowguard.features import pipeline as fp

    paths = paths or Paths.default()
    exp_cfg = load_config("experiments", paths.root)
    seed = int(exp_cfg.get("seed", 42))
    model_names = model_names or list(exp_cfg.get("tuning", {}).get("spaces", {}))
    pipe = fp.load(paths.models / fp.PIPELINE_FILE)
    df = load_p1(paths)
    train, val = df[df[SPLIT] == TRAIN], df[df[SPLIT] == VAL]
    X_train, X_val = fp.transform(pipe, train), fp.transform(pipe, val)
    y_train, y_val = train[FAMILY].to_numpy(), val[FAMILY].to_numpy()

    out: dict[str, Any] = {}
    (paths.models / TUNING_DIR).mkdir(parents=True, exist_ok=True)
    paths.tables.mkdir(parents=True, exist_ok=True)
    for name in model_names:
        rows, best = search(name, X_train, y_train, X_val, y_val, exp_cfg, n_trials, seed)
        table = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
        table.to_csv(paths.tables / f"tuning_{name}.csv", index=False)
        record = {"model": name, "best_params": best, "best_val_macro_f1": float(table["val_macro_f1"].iloc[0]),
                  "config_val_macro_f1": float(table.loc[table["trial"] == 0, "val_macro_f1"].iloc[0]),
                  "n_trials": len(rows)}
        tuned_file(paths, name).write_text(json.dumps(record, indent=2))
        out[name] = record
        log.info("%s: best val macro-F1 %.4f (config %.4f) with %s", name, record["best_val_macro_f1"],
                 record["config_val_macro_f1"], best)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Random hyperparameter search on the validation split")
    parser.add_argument("--models", nargs="+", help="default: every model in tuning.spaces")
    parser.add_argument("--trials", type=int, help="default: tuning.n_trials")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.models, args.trials)


if __name__ == "__main__":
    main()
