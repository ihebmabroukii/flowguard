import numpy as np
import pandas as pd

from flowguard.data.clean import clean_dataset, clean_frame
from flowguard.data.load import normalize_chunk
from flowguard.data.schema import META_COLUMNS, feature_columns
from tests.conftest import N_FEATURES, make_flows


def _raw(labels, features_cfg, **kw) -> pd.DataFrame:
    df = normalize_chunk(make_flows(labels, **kw), features_cfg)
    return df.assign(day="monday")


def test_identifiers_never_reach_the_features(mapper, features_cfg):
    out, report = clean_frame(_raw({"BENIGN": 20}, features_cfg), "cic-ids2017", mapper, features_cfg)
    feats = feature_columns(out)
    for col in ("id", "Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port", "Timestamp",
                "Label", "Attempted Category", "day"):
        assert col not in feats
    assert len(feats) == N_FEATURES == report["n_features"]
    assert list(out.columns[: len(META_COLUMNS)]) == list(META_COLUMNS)
    assert all(out[c].dtype == "float32" for c in feats)
    assert out["row_id"].tolist() == list(range(len(out)))
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"]) and out["timestamp"].notna().all()


def test_inf_becomes_nan(mapper, features_cfg):
    raw = _raw({"BENIGN": 5}, features_cfg)
    raw.loc[0, "Flow Bytes/s"] = np.inf
    raw.loc[1, "Flow Packets/s"] = -np.inf
    out, report = clean_frame(raw, "cic-ids2017", mapper, features_cfg)
    assert report["inf_values_replaced"] == 2
    assert not np.isinf(out[feature_columns(out)].to_numpy()).any()
    assert out["Flow Bytes/s"].isna().sum() == 1


def test_duplicates_and_conflicts_are_removed(mapper, features_cfg):
    raw = _raw({"BENIGN": 6, "DoS Hulk": 4}, features_cfg)
    feats = [c for c in raw.columns if c not in {"id", "Flow ID", "Src IP", "Dst IP", "Src Port",
                                                 "Dst Port", "Timestamp", "Label",
                                                 "Attempted Category", "day"}]
    raw.loc[1, feats] = raw.loc[0, feats].to_numpy()   # exact duplicate benign flow (other IP/time)
    raw.loc[7, feats] = raw.loc[6, feats].to_numpy()   # exact duplicate DoS flow
    raw.loc[2, feats] = raw.loc[8, feats].to_numpy()   # same flow labeled Benign and DoS
    out, report = clean_frame(raw, "cic-ids2017", mapper, features_cfg)
    removed = {s["step"]: s["removed"] for s in report["steps"]}
    assert removed["exact duplicate flows"] == 2
    assert removed["same flow under several families"] == 2
    assert len(out) == 6
    assert out["family"].value_counts().to_dict() == {"Benign": 4, "DoS": 2}


def test_deduplication_can_be_switched_off_for_the_ablation(mapper, features_cfg):
    raw = _raw({"BENIGN": 6}, features_cfg)
    feats = [c for c in raw.columns if c not in {"id", "Flow ID", "Src IP", "Dst IP", "Src Port",
                                                 "Dst Port", "Timestamp", "Label",
                                                 "Attempted Category", "day"}]
    raw.loc[1, feats] = raw.loc[0, feats].to_numpy()
    cfg = {**features_cfg, "deduplicate": False, "drop_conflicting_duplicates": False}
    out, report = clean_frame(raw, "cic-ids2017", mapper, cfg)
    assert len(out) == 6
    assert "exact duplicate flows" not in {s["step"] for s in report["steps"]}


def test_attempted_and_dropped_labels(mapper, features_cfg):
    raw = _raw(["BENIGN", "DoS Hulk", "DoS Hulk - Attempted", "Heartbleed", "Label"], features_cfg)
    out, report = clean_frame(raw, "cic-ids2017", mapper, features_cfg)
    assert out["family"].tolist() == ["Benign", "DoS", "Benign"]
    assert out["label_raw"].tolist() == ["BENIGN", "DoS Hulk", "DoS Hulk - Attempted"]
    assert report["attempted_flows"] == 1


def test_rows_without_any_feature_are_dropped(mapper, features_cfg):
    raw = _raw({"BENIGN": 3}, features_cfg)
    numeric = raw.select_dtypes("number").columns
    raw.loc[1, numeric] = np.nan  # what a repeated header row looks like after coercion
    out, _ = clean_frame(raw, "cic-ids2017", mapper, features_cfg)
    assert len(out) == 2


def test_clean_dataset_writes_parquet_and_report(project, features_cfg):
    raw = _raw({"BENIGN": 10, "Portscan": 5}, features_cfg)
    (project.interim / "cic-ids2017").mkdir()
    raw.to_parquet(project.interim / "cic-ids2017" / "monday-0000.parquet", index=False)
    report = clean_dataset("cic-ids2017", project)
    out = pd.read_parquet(project.processed_file("cic-ids2017"))
    assert len(out) == 15 == report["rows_out"]
    assert (project.tables / "cleaning_cic-ids2017.json").exists()
