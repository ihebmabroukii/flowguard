"""Shared fixtures: a throwaway project folder and synthetic flows with the real CSV header."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flowguard.config import Paths, load_config
from flowguard.data.labels import LabelMapper

REPO = Path(__file__).resolve().parents[1]

# Exact header of the improved CIC-IDS2017 / CSE-CIC-IDS2018 CSVs (89 columns).
HEADER = (
    "id,Flow ID,Src IP,Src Port,Dst IP,Dst Port,Protocol,Timestamp,Flow Duration,Total Fwd Packet,"
    "Total Bwd packets,Total Length of Fwd Packet,Total Length of Bwd Packet,Fwd Packet Length Max,"
    "Fwd Packet Length Min,Fwd Packet Length Mean,Fwd Packet Length Std,Bwd Packet Length Max,"
    "Bwd Packet Length Min,Bwd Packet Length Mean,Bwd Packet Length Std,Flow Bytes/s,Flow Packets/s,"
    "Flow IAT Mean,Flow IAT Std,Flow IAT Max,Flow IAT Min,Fwd IAT Total,Fwd IAT Mean,Fwd IAT Std,"
    "Fwd IAT Max,Fwd IAT Min,Bwd IAT Total,Bwd IAT Mean,Bwd IAT Std,Bwd IAT Max,Bwd IAT Min,"
    "Fwd PSH Flags,Bwd PSH Flags,Fwd URG Flags,Bwd URG Flags,Fwd RST Flags,Bwd RST Flags,"
    "Fwd Header Length,Bwd Header Length,Fwd Packets/s,Bwd Packets/s,Packet Length Min,"
    "Packet Length Max,Packet Length Mean,Packet Length Std,Packet Length Variance,FIN Flag Count,"
    "SYN Flag Count,RST Flag Count,PSH Flag Count,ACK Flag Count,URG Flag Count,CWR Flag Count,"
    "ECE Flag Count,Down/Up Ratio,Average Packet Size,Fwd Segment Size Avg,Bwd Segment Size Avg,"
    "Fwd Bytes/Bulk Avg,Fwd Packet/Bulk Avg,Fwd Bulk Rate Avg,Bwd Bytes/Bulk Avg,Bwd Packet/Bulk Avg,"
    "Bwd Bulk Rate Avg,Subflow Fwd Packets,Subflow Fwd Bytes,Subflow Bwd Packets,Subflow Bwd Bytes,"
    "FWD Init Win Bytes,Bwd Init Win Bytes,Fwd Act Data Pkts,Fwd Seg Size Min,Active Mean,Active Std,"
    "Active Max,Active Min,Idle Mean,Idle Std,Idle Max,Idle Min,ICMP Code,ICMP Type,"
    "Total TCP Flow Time,Label,Attempted Category"
).split(",")
TEXT = {"Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"}
NUMERIC = [c for c in HEADER if c not in TEXT]
# 89 columns minus the 9 dropped in configs/features.yaml.
N_FEATURES = 80


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Paths:
    """Empty copy of the repo layout with the real configs."""
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    shutil.copy(REPO / "pyproject.toml", tmp_path / "pyproject.toml")
    for sub in ("data/raw/cic-ids2017", "data/raw/cse-cic-ids2018", "data/interim", "data/processed"):
        (tmp_path / sub).mkdir(parents=True)
    monkeypatch.setenv("FLOWGUARD_ROOT", str(tmp_path))
    return Paths(tmp_path)


@pytest.fixture
def mapper() -> LabelMapper:
    return LabelMapper.from_config(load_config("labels", REPO))


@pytest.fixture
def features_cfg() -> dict:
    return load_config("features", REPO)


def make_flows(
    labels: list[str] | dict[str, int],
    start: str = "2017-07-03 09:00:00",
    seed: int = 0,
    attempted: int = -1,
) -> pd.DataFrame:
    """Raw-format rows (real header) with random numeric values and increasing timestamps.

    `labels` is either one label per row or {label: count}.
    """
    if isinstance(labels, dict):
        labels = [lab for lab, n in labels.items() for _ in range(n)]
    n = len(labels)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.random((n, len(NUMERIC))) * 1000, columns=NUMERIC).astype("float32")
    df["id"] = np.arange(1, n + 1, dtype="float32")
    df["Attempted Category"] = np.float32(attempted)
    df["Flow ID"] = [f"10.0.0.{i % 250}-10.0.1.1-{40000 + i}-80-6" for i in range(n)]
    df["Src IP"] = [f"10.0.0.{i % 250}" for i in range(n)]
    df["Dst IP"] = "10.0.1.1"
    df["Timestamp"] = (pd.Timestamp(start) + pd.to_timedelta(np.arange(n), unit="s")).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )
    df["Label"] = labels
    return df[HEADER]
