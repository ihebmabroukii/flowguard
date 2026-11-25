"""Ablation study: switch off one design choice at a time and measure what it was worth.

Each variant in `ablation.variants` (configs/experiments.yaml) changes one thing against the
full pipeline, then repeats the whole P1 path: clean (when needed) -> temporal split -> fit the
feature pipeline -> train our models -> score the test split.

    features: {...}   overrides of configs/features.yaml. Cleaning keys (deduplicate,
                      drop_conflicting_duplicates, keep_columns) re-clean from data/interim.
                      keep_columns takes columns off the drop list, e.g. Dst Port.
    training: {...}   balanced_weights: false turns class weighting off; any other key overrides
                      the model's hyperparameters (e.g. dropout: 0.0)

The headline row is `no_dedup`: duplicated flows land in both train and test, so the model is
partly scored on flows it has memorized. Next to the scores, every variant reports the share of
test flows whose exact feature vector also appears in its training split, which makes the
leakage visible as a number.

Output: reports/tables/ablation.csv (+ ablation.json)

    python -m flowguard.eval.ablation
    python -m flowguard.eval.ablation --variants full no_dedup --models softmax_regression
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

CLEANING_KEYS = {"deduplicate", "drop_conflicting_duplicates", "keep_columns"}


def variant_features_cfg(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """features.yaml with the variant's overrides; keep_columns are removed from the drop list."""
    cfg = copy.deepcopy(base)
    overrides = dict(overrides or {})
    keep = set(overrides.pop("keep_columns", []) or [])
    cfg.update(overrides)
    cfg["drop"] = [c for c in cfg.get("drop", []) if c not in keep]
    return cfg


def needs_recleaning(spec: dict[str, Any]) -> bool:
    return bool(CLEANING_KEYS & set((spec or {}).get("features", {}) or {}))


def seen_in_train_share(train_features: Any, test_features: Any) -> float:
    """Share of test flows whose exact feature vector also occurs in the training split."""
    import pandas as pd

    if len(test_features) == 0:
        return float("nan")
    train_hash = pd.util.hash_pandas_object(train_features, index=False).to_numpy()
    test_hash = pd.util.hash_pandas_object(test_features, index=False).to_numpy()
    return float(np.isin(test_hash, train_hash).mean())


def run_variant(name: str, spec: dict[str, Any], df: Any, exp_cfg: dict[str, Any], features_cfg: dict[str, Any],
                models: list[str], models_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from flowguard.data.schema import FAMILY, feature_columns
    from flowguard.data.split import TEST, TRAIN, VAL, temporal_split
    from flowguard.eval.metrics import attack_pr_auc, classification_metrics
    from flowguard.eval.stats import bootstrap_ci
    from flowguard.features import pipeline as fp
    from flowguard.models import build_classifier

    seed = int(exp_cfg.get("seed", 42))
    data_cfg, acfg = exp_cfg["data"], exp_cfg.get("ablation", {})
    split = temporal_split(df, float(data_cfg.get("train_fraction", 0.7)),
                           float(data_cfg.get("validation_fraction", 0.15))).to_numpy()
    feats = feature_columns(df)
    train, val, test = df[split == TRAIN], df[split == VAL], df[split == TEST]
    leakage = seen_in_train_share(train[feats], test[feats])

    pipe = fp.fit_pipeline(train, features_cfg, seed)
    X_train, X_val, X_test = (fp.transform(pipe, part) for part in (train, val, test))
    y_train, y_val, y_test = (part[FAMILY].to_numpy() for part in (train, val, test))
    labels = sorted(set(y_test))
    training = dict(spec.get("training") or {})
    balanced = training.pop("balanced_weights", None)

    rows = []
    for model_name in models:
        params = {**(models_cfg.get(model_name) or {}), **training}
        if acfg.get("max_train_rows"):
            params["max_train_rows"] = int(acfg["max_train_rows"])
        model = build_classifier(model_name, {model_name: params}, seed)
        if balanced is not None:
            model.balanced_weights = bool(balanced)
        model.fit(X_train, y_train, X_val, y_val)
        pred = np.asarray(model.predict(X_test), dtype=object)
        m = classification_metrics(y_test, pred, labels=labels)
        ci = bootstrap_ci(y_test, pred, labels=labels, n_boot=int(exp_cfg.get("stats", {}).get("n_boot", 1000)),
                          seed=seed)["macro_f1"]
        rows.append({"variant": name, "model": model_name, "description": spec.get("description", ""),
                     "macro_f1": m["macro_f1"], "macro_f1_low": ci["low"], "macro_f1_high": ci["high"],
                     "benign_fpr_per_10k": m["benign_fpr_per_10k"], "detection_rate": m["detection_rate"],
                     "pr_auc": attack_pr_auc(y_test, model.attack_score(X_test)),
                     "test_seen_in_train": leakage, "n_rows": len(df), "n_features": X_train.shape[1]})
        log.info("ablation %-18s %-18s macro-F1 %.4f [%.4f, %.4f]  test seen in train %.1f%%",
                 name, model_name, m["macro_f1"], ci["low"], ci["high"], 100 * leakage)
    return rows


def run(variants: list[str] | None = None, models: list[str] | None = None, paths: Any = None) -> Any:
    import pandas as pd

    from flowguard.config import Paths, load_config
    from flowguard.data.clean import clean_frame
    from flowguard.data.io import load_processed
    from flowguard.data.labels import LabelMapper
    from flowguard.models.tune import tuned_models_cfg

    paths = paths or Paths.default()
    exp_cfg = load_config("experiments", paths.root)
    base_features = load_config("features", paths.root)
    acfg = exp_cfg.get("ablation", {})
    specs = acfg.get("variants", {"full": {}})
    variants = variants or list(specs)
    models = models or acfg.get("models", ["softmax_regression", "mlp"])
    models_cfg = tuned_models_cfg(exp_cfg, paths)
    unknown = [v for v in variants if v not in specs]
    if unknown:
        raise KeyError(f"Unknown ablation variants {unknown}. Defined: {list(specs)}")

    processed = load_processed("cic-ids2017", paths)
    raw = None
    rows = []
    for name in variants:
        spec = specs[name] or {}
        features_cfg = variant_features_cfg(base_features, spec.get("features", {}))
        if needs_recleaning(spec):
            if raw is None:
                source = paths.interim / "cic-ids2017"
                if not any(source.glob("*.parquet")):
                    raise FileNotFoundError(f"{source} has no Parquet parts; variant {name!r} must re-clean.")
                raw = pd.read_parquet(source)
            mapper = LabelMapper.from_config(load_config("labels", paths.root))
            df, _ = clean_frame(raw, "cic-ids2017", mapper, features_cfg)
        else:
            df = processed
        rows += run_variant(name, spec, df, exp_cfg, features_cfg, models, models_cfg)

    table = pd.DataFrame(rows)
    full = table[table["variant"] == "full"].set_index("model")["macro_f1"]
    table["delta_vs_full"] = table["macro_f1"] - table["model"].map(full)
    paths.tables.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths.tables / "ablation.csv", index=False)
    (paths.tables / "ablation.json").write_text(json.dumps(table.to_dict("records"), indent=2, default=str))
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ablation study over the pipeline's design choices")
    parser.add_argument("--variants", nargs="+", help="default: every variant in ablation.variants")
    parser.add_argument("--models", nargs="+", help="default: ablation.models")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.variants, args.models)


if __name__ == "__main__":
    main()
