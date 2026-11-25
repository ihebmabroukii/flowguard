"""Turn the interim Parquet parts of one dataset into a single clean, labeled table.

Steps, each counted in reports/tables/cleaning_<dataset>.json for the report:
  1. map raw labels to families (attempted policy applied), drop the `drop` family
  2. keep timestamp and raw label as metadata, then drop identifier columns
  3. replace +/-inf with NaN, drop rows whose features are all missing
  4. remove exact duplicate flows, then flows whose features appear under several families

Output: data/processed/<dataset>.parquet with metadata columns first (see schema.py).

    python -m flowguard.data.clean --dataset cic-ids2017
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flowguard.config import DATASETS, Paths, check_dataset, load_config
from flowguard.data.labels import LabelMapper
from flowguard.data.schema import (
    DATASET,
    DAY,
    DROP,
    FAMILY,
    LABEL_RAW,
    META_COLUMNS,
    RAW_ATTEMPTED,
    RAW_LABEL,
    RAW_TIMESTAMP,
    ROW_ID,
    TIMESTAMP,
    feature_columns,
)

log = logging.getLogger(__name__)


def parse_timestamps(values: pd.Series) -> pd.Series:
    """ISO timestamps (improved releases) fast path, any other format as a fallback."""
    parsed = pd.to_datetime(values, errors="coerce", format="ISO8601")
    if len(values) and parsed.isna().mean() > 0.05:
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    return parsed


def clean_frame(
    df: pd.DataFrame,
    dataset: str,
    mapper: LabelMapper,
    features_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pure cleaning function: returns the cleaned frame and a step-by-step report."""
    report: dict[str, Any] = {"dataset": dataset, "rows_in": len(df), "steps": []}

    def step(name: str, frame: pd.DataFrame, before: int) -> None:
        report["steps"].append({"step": name, "removed": before - len(frame), "rows": len(frame)})
        log.info("%s %-34s removed %9d  rows %10d", dataset, name, before - len(frame), len(frame))

    df = df.rename(columns=lambda c: str(c).strip())

    # 1. Labels.
    before = len(df)
    df = df[df[RAW_LABEL].notna()]
    attempted = df[RAW_ATTEMPTED] if RAW_ATTEMPTED in df.columns else None
    families = mapper.map_series(df[RAW_LABEL], attempted)
    labels_text = df[RAW_LABEL].astype(str)
    base_families = labels_text.map(mapper.lookup(labels_text.unique()))
    report["attempted_flows"] = int((families != base_families).sum())
    report["attempted_policy"] = mapper.attempted_policy
    keep = families.ne(DROP).to_numpy()
    df, families = df[keep], families[keep]
    step("labels (missing, dropped families)", df, before)

    # 2. Metadata, then identifiers.
    meta = pd.DataFrame(
        {
            DATASET: dataset,
            DAY: df[DAY].astype(str) if DAY in df.columns else "unknown",
            TIMESTAMP: parse_timestamps(df[RAW_TIMESTAMP]) if RAW_TIMESTAMP in df.columns else pd.NaT,
            LABEL_RAW: df[RAW_LABEL].astype(str),
            FAMILY: families.astype(str),
        },
        index=df.index,
    )
    to_drop = set(features_cfg.get("drop", [])) | {RAW_LABEL, RAW_ATTEMPTED, RAW_TIMESTAMP, DAY}
    features = df.drop(columns=[c for c in df.columns if c in to_drop])
    non_numeric = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        raise TypeError(f"Non-numeric feature columns left after dropping identifiers: {non_numeric}")
    report["dropped_columns"] = sorted(c for c in df.columns if c in to_drop)
    df = pd.concat([meta, features.astype("float32")], axis=1)

    # 3. Infinite and empty values.
    feats = feature_columns(df)
    values = df[feats].to_numpy()
    inf_mask = np.isinf(values)
    report["inf_values_replaced"] = int(inf_mask.sum())
    report["inf_values_by_column"] = {
        col: int(n) for col, n in zip(feats, inf_mask.sum(axis=0)) if n
    }
    if inf_mask.any():
        df[feats] = np.where(inf_mask, np.nan, values).astype("float32")
    before = len(df)
    df = df[df[feats].notna().any(axis=1)]
    step("rows with no feature values", df, before)

    # 4. Duplicates. Hashing the feature vector keeps memory flat on millions of rows.
    #    `deduplicate: false` exists only for the ablation study (eval/ablation.py), which
    #    measures how much duplicated flows across train and test inflate the scores.
    flow_hash = pd.util.hash_pandas_object(df[feats], index=False)
    if features_cfg.get("deduplicate", True):
        before = len(df)
        exact_dup = pd.DataFrame({"h": flow_hash.to_numpy(), "f": df[FAMILY].to_numpy()}).duplicated()
        df, flow_hash = df[~exact_dup.to_numpy()], flow_hash[~exact_dup.to_numpy()]
        step("exact duplicate flows", df, before)

    if features_cfg.get("drop_conflicting_duplicates", True):
        before = len(df)
        n_families = df[FAMILY].groupby(flow_hash.to_numpy()).transform("nunique")
        df = df[(n_families == 1).to_numpy()]
        step("same flow under several families", df, before)

    df = df.reset_index(drop=True)
    df.insert(0, ROW_ID, np.arange(len(df), dtype="int64"))
    df = df[[*META_COLUMNS, *feature_columns(df)]]

    report["rows_out"] = len(df)
    report["n_features"] = len(feature_columns(df))
    report["families"] = df[FAMILY].value_counts().to_dict()
    report["raw_labels"] = df.groupby([FAMILY, LABEL_RAW]).size().rename("rows").reset_index().to_dict("records")
    return df, report


def clean_dataset(dataset: str, paths: Paths | None = None) -> dict[str, Any]:
    check_dataset(dataset)
    paths = paths or Paths.default()
    source = paths.interim / dataset
    if not any(source.glob("*.parquet")):
        raise FileNotFoundError(f"No interim Parquet for {dataset}. Run `python -m flowguard.data.load` first.")

    mapper = LabelMapper.from_config(load_config("labels", paths.root))
    features_cfg = load_config("features", paths.root)
    df = pd.read_parquet(source)
    df, report = clean_frame(df, dataset, mapper, features_cfg)

    out = paths.processed_file(dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    paths.tables.mkdir(parents=True, exist_ok=True)
    (paths.tables / f"cleaning_{dataset}.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("%s: %d rows, %d features -> %s", dataset, len(df), report["n_features"], _rel(out, paths.root))
    return report


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = Paths.default()
    for dataset in DATASETS if args.dataset == "all" else [args.dataset]:
        if args.dataset == "all" and not any((paths.interim / dataset).glob("*.parquet")):
            log.warning("%s: not loaded yet, skipping", dataset)
            continue
        clean_dataset(dataset, paths)


if __name__ == "__main__":
    main()
