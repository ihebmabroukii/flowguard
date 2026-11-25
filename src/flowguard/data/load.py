"""Read the raw CSVs into Parquet parts under data/interim/<dataset>/.

CSVs are streamed in chunks straight out of the downloaded zip, so the multi-GB 2018 files
never have to be extracted or held in memory at once. Each chunk gets its column names
normalized, its numeric columns coerced to float32, an optional per-class sample applied,
and a `day` column taken from the file name.

    python -m flowguard.data.load --dataset cic-ids2017
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import zipfile
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import IO, Any

import numpy as np
import pandas as pd

from flowguard.config import DATASETS, Paths, check_dataset, load_config
from flowguard.data.schema import DAY, RAW_LABEL

log = logging.getLogger(__name__)

Opener = Callable[[], IO[bytes]]


class IncompleteDownloadError(RuntimeError):
    """The dataset archive is missing or truncated."""


def iter_csv_sources(raw_dir: Path, stack: ExitStack) -> Iterator[tuple[str, Opener]]:
    """(day name, opener) for every CSV of a dataset.

    Extracted CSVs win over the zip so nothing is read twice. Inside a zip, macOS
    metadata entries are skipped.
    """
    csv_files = sorted(raw_dir.rglob("*.csv"))
    if csv_files:
        for path in csv_files:
            yield path.stem.lower(), (lambda p=path: open(p, "rb"))
        return

    archives = sorted(raw_dir.glob("*.zip"))
    if not archives:
        raise IncompleteDownloadError(
            f"No CSV or zip files in {raw_dir}. See 'Getting the data' in the README."
        )
    for archive in archives:
        try:
            zf = stack.enter_context(zipfile.ZipFile(archive))
        except zipfile.BadZipFile as exc:
            raise IncompleteDownloadError(
                f"{archive.name} is not a complete zip file (interrupted download?). "
                "Resume it with `curl -L -C -` as shown in the README."
            ) from exc
        for info in zf.infolist():
            name = PurePosixPath(info.filename)
            if info.is_dir() or name.suffix.lower() != ".csv" or "__MACOSX" in name.parts:
                continue
            if name.name.startswith("._"):
                continue
            yield name.stem.lower(), (lambda i=info, z=zf: z.open(i))


def normalize_chunk(chunk: pd.DataFrame, features_cfg: dict[str, Any]) -> pd.DataFrame:
    """Strip column names, apply renames, and coerce every non-text column to float32."""
    chunk = chunk.rename(columns=lambda c: str(c).strip())
    chunk = chunk.rename(columns=features_cfg.get("rename") or {})
    text_cols = set(features_cfg.get("string_columns", []))
    for col in chunk.columns:
        if col in text_cols:
            chunk[col] = chunk[col].astype("string").str.strip()
        else:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float32")
    return chunk


def sample_chunk(
    chunk: pd.DataFrame, benign_fraction: float, attack_fraction: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Keep each benign row with probability benign_fraction and each other row with attack_fraction."""
    if benign_fraction >= 1.0 and attack_fraction >= 1.0:
        return chunk
    benign = chunk[RAW_LABEL].astype("string").str.strip().str.upper().eq("BENIGN").to_numpy()
    keep_prob = np.where(benign, benign_fraction, attack_fraction)
    return chunk[rng.random(len(chunk)) < keep_prob]


def load_dataset(dataset: str, paths: Paths | None = None, seed: int | None = None) -> dict:
    """Convert one dataset to Parquet parts. Returns (and saves) a per-day row count summary."""
    check_dataset(dataset)
    paths = paths or Paths.default()
    features_cfg = load_config("features", paths.root)
    exp_cfg = load_config("experiments", paths.root)
    data_cfg = exp_cfg["data"]
    ds_cfg = data_cfg["datasets"][dataset]
    benign_fraction = float(ds_cfg.get("benign_fraction", 1.0))
    attack_fraction = float(ds_cfg.get("attack_fraction", 1.0))
    chunksize = int(data_cfg.get("chunksize", 500_000))
    rng = np.random.default_rng(exp_cfg.get("seed", 42) if seed is None else seed)
    text_cols = features_cfg.get("string_columns", [])

    out_dir = paths.interim / dataset
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    summary: dict[str, Any] = {
        "dataset": dataset,
        "benign_fraction": benign_fraction,
        "attack_fraction": attack_fraction,
        "days": {},
    }
    with ExitStack() as stack:
        for day, opener in iter_csv_sources(paths.raw / dataset, stack):
            read = kept = 0
            with opener() as fh:
                reader = pd.read_csv(
                    fh,
                    chunksize=chunksize,
                    dtype={c: "string" for c in text_cols},
                    low_memory=False,
                    skipinitialspace=True,
                )
                for part, chunk in enumerate(reader):
                    read += len(chunk)
                    chunk = normalize_chunk(chunk, features_cfg)
                    if RAW_LABEL not in chunk.columns:
                        raise KeyError(f"{dataset}/{day}: no {RAW_LABEL!r} column after renaming")
                    chunk = sample_chunk(chunk, benign_fraction, attack_fraction, rng)
                    chunk = chunk.assign(**{DAY: day})
                    kept += len(chunk)
                    chunk.to_parquet(out_dir / f"{day}-{part:04d}.parquet", index=False)
            summary["days"][day] = {"rows_read": read, "rows_kept": kept}
            log.info("%s %-28s read %10d  kept %10d", dataset, day, read, kept)

    if not summary["days"]:
        raise IncompleteDownloadError(f"No CSV files found for {dataset} in {paths.raw / dataset}")
    summary["rows_read"] = sum(d["rows_read"] for d in summary["days"].values())
    summary["rows_kept"] = sum(d["rows_kept"] for d in summary["days"].values())
    paths.tables.mkdir(parents=True, exist_ok=True)
    (paths.tables / f"load_{dataset}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = Paths.default()
    for dataset in DATASETS if args.dataset == "all" else [args.dataset]:
        raw_dir = paths.raw / dataset
        has_files = any(raw_dir.rglob("*.csv")) or any(raw_dir.glob("*.zip"))
        if args.dataset == "all" and not has_files:
            log.warning("%s: nothing in %s yet, skipping", dataset, raw_dir)
            continue
        summary = load_dataset(dataset, paths)
        log.info("%s: %d of %d rows written", dataset, summary["rows_kept"], summary["rows_read"])


if __name__ == "__main__":
    main()
