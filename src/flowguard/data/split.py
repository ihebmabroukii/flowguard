"""Leakage-free splits for the evaluation protocols.

P1 (in-dataset): within every (day, family) group, flows are ordered by timestamp. The first
`train_fraction` form the training portion and the rest is test; the last
`validation_fraction` of the training portion becomes validation. Every attack therefore
appears in train and test, and the model is always tested on flows that happened after the
ones it learned from.

P2 (cross-dataset): train on CIC-IDS2017, test on the CSE-CIC-IDS2018 families both datasets share.
P3 (leave-one-family-out): P1 split with one attack family removed from train and validation.

    python -m flowguard.data.split      # writes data/processed/cic-ids2017_splits.parquet
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from flowguard.config import Paths, load_config
from flowguard.data.schema import BENIGN, DAY, FAMILY, ROW_ID, SPLIT, TIMESTAMP

log = logging.getLogger(__name__)

TRAIN, VAL, TEST = "train", "val", "test"


def temporal_split(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    group_cols: tuple[str, ...] = (DAY, FAMILY),
) -> pd.Series:
    """'train' / 'val' / 'test' for every row, time-ordered within each group.

    Ties and missing timestamps are ordered by row_id so the result is deterministic.
    Rows with no timestamp sort last and land in test.
    """
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1) and validation_fraction in [0, 1)")
    ordered = df.sort_values([*group_cols, TIMESTAMP, ROW_ID], na_position="last", kind="stable")
    grouped = ordered.groupby(list(group_cols), sort=False, observed=True)
    position = grouped.cumcount().to_numpy()
    size = grouped[ROW_ID].transform("size").to_numpy()
    rel = position / size

    train_cut = train_fraction * (1 - validation_fraction)
    labels = np.where(rel < train_cut, TRAIN, np.where(rel < train_fraction, VAL, TEST))
    return pd.Series(labels, index=ordered.index, name=SPLIT).reindex(df.index)


@dataclass(frozen=True)
class Masks:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def leave_family_out(df: pd.DataFrame, split: pd.Series, family: str) -> Masks:
    """P3: `family` never appears in train or validation; test is P1-test benign plus every
    flow of the held-out family (from any portion, since the model has never seen it)."""
    if family == BENIGN:
        raise ValueError("Leave-one-family-out holds out an attack family, not Benign")
    if family not in set(df[FAMILY]):
        raise ValueError(f"Family {family!r} is not in this dataset")
    fam = df[FAMILY].to_numpy()
    s = split.to_numpy()
    held = fam == family
    return Masks(
        train=(s == TRAIN) & ~held,
        val=(s == VAL) & ~held,
        test=((s == TEST) & (fam == BENIGN)) | held,
    )


def shared_families(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    """P2: families present in both datasets. Only these are scored on the 2018 data."""
    return sorted(set(train_df[FAMILY]) & set(test_df[FAMILY]))


def split_counts(df: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    """Rows per family and split, for the report and as a sanity check."""
    table = pd.crosstab(df[FAMILY], split).reindex(columns=[TRAIN, VAL, TEST], fill_value=0)
    table["total"] = table.sum(axis=1)
    return table.sort_values("total", ascending=False)


def build_p1(paths: Paths | None = None) -> pd.DataFrame:
    paths = paths or Paths.default()
    data_cfg = load_config("experiments", paths.root)["data"]
    source = paths.processed_file("cic-ids2017")
    if not source.exists():
        raise FileNotFoundError(f"{source} not found. Run load and clean first (make data).")
    df = pd.read_parquet(source, columns=[ROW_ID, DAY, TIMESTAMP, FAMILY])
    split = temporal_split(
        df,
        train_fraction=float(data_cfg.get("train_fraction", 0.7)),
        validation_fraction=float(data_cfg.get("validation_fraction", 0.15)),
    )
    out = pd.DataFrame({ROW_ID: df[ROW_ID], SPLIT: split})
    out.to_parquet(paths.split_file("cic-ids2017"), index=False)

    counts = split_counts(df, split)
    paths.tables.mkdir(parents=True, exist_ok=True)
    counts.to_csv(paths.tables / "p1_split_counts.csv")
    log.info("P1 split written\n%s", counts.to_string())
    return out


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_p1()


if __name__ == "__main__":
    main()
