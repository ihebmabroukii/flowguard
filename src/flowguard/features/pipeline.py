"""Preprocessing shared by every model and by the online service.

    signed log1p -> median imputation -> drop constant features -> drop correlated features -> scale

It is fitted on the P1 training split only and saved to models/pipeline.joblib. The online
service loads the same file, so live flows are transformed exactly like training flows.
Output is a pandas DataFrame with feature names, which SHAP uses to label its plots.

    python -m flowguard.features.pipeline
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer, StandardScaler

from flowguard.config import Paths, load_config
from flowguard.data.schema import feature_columns

log = logging.getLogger(__name__)

PIPELINE_FILE = "pipeline.joblib"


def signed_log1p(x: Any) -> Any:
    """log1p that keeps the sign: some features use -1 for 'not present' (e.g. init window bytes)."""
    return np.sign(x) * np.log1p(np.abs(x))


class CorrelationFilter(TransformerMixin, BaseEstimator):
    """Drop the later feature of every pair whose absolute Pearson correlation exceeds `threshold`.

    Correlations are computed on at most `max_rows` training rows, which is plenty to estimate
    them and keeps fitting fast on millions of flows.
    """

    def __init__(self, threshold: float = 0.98, max_rows: int = 200_000, random_state: int = 0):
        self.threshold = threshold
        self.max_rows = max_rows
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> CorrelationFilter:
        frame = self._as_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        if len(frame) > self.max_rows:
            frame = frame.sample(self.max_rows, random_state=self.random_state)
        corr = frame.astype("float64").corr().abs().to_numpy()
        np.fill_diagonal(corr, 0.0)
        corr = np.nan_to_num(corr)
        keep: list[int] = []
        dropped: dict[str, str] = {}
        for j in range(corr.shape[0]):
            twin = next((k for k in keep if corr[j, k] > self.threshold), None)
            if twin is None:
                keep.append(j)
            else:
                dropped[str(self.feature_names_in_[j])] = str(self.feature_names_in_[twin])
        self.keep_ = np.asarray(keep, dtype=int)
        self.dropped_ = dropped  # dropped feature -> the kept feature it duplicates
        return self

    def transform(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.keep_]
        return np.asarray(X)[:, self.keep_]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return self.feature_names_in_[self.keep_]

    @staticmethod
    def _as_frame(X: Any) -> pd.DataFrame:
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))


def build_pipeline(cfg: dict[str, Any], seed: int = 42) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if cfg.get("log1p", True):
        steps.append(("log1p", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")))
    steps.append(("impute", SimpleImputer(strategy="median", keep_empty_features=True)))
    steps.append(("variance", VarianceThreshold(threshold=float(cfg.get("min_variance", 0.0)))))
    if cfg.get("max_correlation") is not None:
        steps.append(("correlation", CorrelationFilter(float(cfg["max_correlation"]), random_state=seed)))
    scaler = cfg.get("scaler", "quantile")
    if scaler == "quantile":
        steps.append(("scale", QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                                   subsample=200_000, random_state=seed)))
    elif scaler == "standard":
        steps.append(("scale", StandardScaler()))
    else:
        raise ValueError(f"Unknown scaler {scaler!r} (expected 'quantile' or 'standard')")
    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")
    return pipe


def fit_pipeline(train: pd.DataFrame, cfg: dict[str, Any], seed: int = 42) -> Pipeline:
    """Fit on the feature columns of `train` (metadata columns are ignored)."""
    feats = feature_columns(train)
    pipe = build_pipeline(cfg, seed)
    pipe.fit(train[feats])
    return pipe


def transform(pipe: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted pipeline to any frame that has the training feature columns."""
    missing = [c for c in pipe.feature_names_in_ if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    out = pipe.transform(df[list(pipe.feature_names_in_)])
    return out.astype("float32")


def describe(pipe: Pipeline) -> dict[str, Any]:
    """What the fitted pipeline kept and removed, for the report."""
    names_in = list(pipe.feature_names_in_)
    info: dict[str, Any] = {"n_features_in": len(names_in)}
    variance = pipe.named_steps["variance"]
    info["constant_features"] = [n for n, keep in zip(names_in, variance.get_support()) if not keep]
    if "correlation" in pipe.named_steps:
        info["correlated_features"] = pipe.named_steps["correlation"].dropped_
    info["features_out"] = list(pipe.get_feature_names_out())
    info["n_features_out"] = len(info["features_out"])
    return info


def save(pipe: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load(path: Path) -> Pipeline:
    return joblib.load(path)


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import json

    from flowguard.data.io import SPLIT, load_p1

    paths = Paths.default()
    seed = load_config("experiments", paths.root).get("seed", 42)
    df = load_p1(paths)
    pipe = fit_pipeline(df[df[SPLIT] == "train"], load_config("features", paths.root), seed)
    save(pipe, paths.models / PIPELINE_FILE)
    info = describe(pipe)
    paths.tables.mkdir(parents=True, exist_ok=True)
    (paths.tables / "feature_pipeline.json").write_text(json.dumps(info, indent=2))
    log.info("pipeline: %d features in, %d out (%d constant, %d correlated removed)",
             info["n_features_in"], info["n_features_out"], len(info["constant_features"]),
             len(info.get("correlated_features", {})))


if __name__ == "__main__":
    main()
