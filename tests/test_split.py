import numpy as np
import pandas as pd
import pytest

from flowguard.data.split import (
    TEST,
    TRAIN,
    VAL,
    build_p1,
    leave_family_out,
    shared_families,
    split_counts,
    temporal_split,
)


def _clean_like(spec: dict[tuple[str, str], int], seed: int = 0) -> pd.DataFrame:
    """Cleaned-format metadata: {(day, family): n_rows}, timestamps shuffled within the frame."""
    rows = []
    for (day, family), n in spec.items():
        base = pd.Timestamp("2017-07-03") + pd.Timedelta(hours=len(rows) % 7)
        rows += [(day, family, base + pd.Timedelta(seconds=i)) for i in range(n)]
    df = pd.DataFrame(rows, columns=["day", "family", "timestamp"])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # input order must not matter
    df.insert(0, "row_id", np.arange(len(df)))
    return df


SPEC = {("monday", "Benign"): 1000, ("wednesday", "Benign"): 800, ("wednesday", "DoS"): 400,
        ("friday", "Benign"): 600, ("friday", "PortScan"): 300, ("friday", "Botnet"): 50}


def test_every_group_is_in_every_split_with_the_right_share():
    df = _clean_like(SPEC)
    split = temporal_split(df, train_fraction=0.7, validation_fraction=0.15)
    assert split.index.equals(df.index)
    for (day, family), group in df.groupby(["day", "family"]):
        shares = split[group.index].value_counts(normalize=True)
        assert shares[TRAIN] == pytest.approx(0.595, abs=0.02), (day, family)
        assert shares[VAL] == pytest.approx(0.105, abs=0.02), (day, family)
        assert shares[TEST] == pytest.approx(0.30, abs=0.02), (day, family)


def test_test_flows_come_after_training_flows():
    df = _clean_like(SPEC)
    split = temporal_split(df)
    for _, group in df.groupby(["day", "family"]):
        s = split[group.index]
        ts = group["timestamp"]
        assert ts[s == TRAIN].max() < ts[s == VAL].min() <= ts[s == VAL].max() < ts[s == TEST].min()


def test_split_is_deterministic():
    df = _clean_like(SPEC)
    assert temporal_split(df).equals(temporal_split(df))


def test_leave_family_out():
    df = _clean_like(SPEC)
    split = temporal_split(df)
    masks = leave_family_out(df, split, "PortScan")
    fam = df["family"].to_numpy()
    assert not (fam[masks.train] == "PortScan").any()
    assert not (fam[masks.val] == "PortScan").any()
    assert (fam[masks.test] == "PortScan").sum() == 300
    assert set(fam[masks.test]) == {"Benign", "PortScan"}
    assert not (masks.train & masks.test).any() and not (masks.val & masks.test).any()
    with pytest.raises(ValueError):
        leave_family_out(df, split, "Benign")


def test_shared_families_and_counts():
    a = _clean_like({("monday", "Benign"): 5, ("friday", "PortScan"): 5, ("friday", "DoS"): 5})
    b = _clean_like({("wed", "Benign"): 5, ("wed", "DoS"): 5, ("wed", "DDoS"): 5})
    assert shared_families(a, b) == ["Benign", "DoS"]
    counts = split_counts(a, temporal_split(a))
    assert list(counts.columns) == [TRAIN, VAL, TEST, "total"]
    assert counts["total"].sum() == 15


def test_build_p1_writes_split_file(project):
    df = _clean_like(SPEC)
    df.insert(1, "dataset", "cic-ids2017")
    df["label_raw"] = df["family"]
    df["Flow Duration"] = np.float32(1.0)
    df.to_parquet(project.processed_file("cic-ids2017"), index=False)
    out = build_p1(project)
    saved = pd.read_parquet(project.split_file("cic-ids2017"))
    assert len(saved) == len(df) and set(saved["split"]) == {TRAIN, VAL, TEST}
    assert out["row_id"].is_unique
    assert (project.tables / "p1_split_counts.csv").exists()
