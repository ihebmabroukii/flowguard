"""Read the processed tables and splits produced by `make data`."""

from __future__ import annotations

import pandas as pd

from flowguard.config import Paths, check_dataset
from flowguard.data.schema import ROW_ID
from flowguard.data.split import SPLIT


def load_processed(dataset: str, paths: Paths | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    check_dataset(dataset)
    paths = paths or Paths.default()
    source = paths.processed_file(dataset)
    if not source.exists():
        raise FileNotFoundError(f"{source} not found. Run `make data` first.")
    return pd.read_parquet(source, columns=columns)


def load_p1(paths: Paths | None = None) -> pd.DataFrame:
    """Processed CIC-IDS2017 with a `split` column (train / val / test)."""
    paths = paths or Paths.default()
    df = load_processed("cic-ids2017", paths)
    split_file = paths.split_file("cic-ids2017")
    if not split_file.exists():
        raise FileNotFoundError(f"{split_file} not found. Run `python -m flowguard.data.split` first.")
    splits = pd.read_parquet(split_file)
    df = df.merge(splits, on=ROW_ID, how="inner", validate="one_to_one")
    if len(df) != len(splits):
        raise ValueError("Split file and processed table disagree; rerun `make data`.")
    return df


__all__ = ["SPLIT", "load_p1", "load_processed"]
