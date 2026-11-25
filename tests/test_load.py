import json
import zipfile

import pandas as pd
import pytest

from flowguard.data.load import IncompleteDownloadError, load_dataset
from tests.conftest import HEADER, make_flows


def _zip_csvs(path, frames: dict[str, pd.DataFrame]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, frame in frames.items():
            zf.writestr(name, frame.to_csv(index=False))
        zf.writestr("__MACOSX/._monday.csv", "junk")


def test_load_streams_csvs_from_zip(project):
    monday = make_flows({"BENIGN": 30, "DoS Hulk": 10})
    friday = make_flows({"BENIGN": 20, "Portscan": 5}, start="2017-07-07 09:00:00")
    _zip_csvs(project.raw / "cic-ids2017" / "CICIDS2017_improved.zip",
              {"monday.csv": monday, "friday.csv": friday})

    summary = load_dataset("cic-ids2017", project)

    assert summary["days"] == {
        "friday": {"rows_read": 25, "rows_kept": 25},
        "monday": {"rows_read": 40, "rows_kept": 40},
    }
    df = pd.read_parquet(project.interim / "cic-ids2017")
    assert len(df) == 65
    assert set(df["day"]) == {"monday", "friday"}
    assert df["Flow Duration"].dtype == "float32"
    assert str(df["Label"].dtype) == "string"
    assert set(HEADER) <= set(df.columns)
    assert json.loads((project.tables / "load_cic-ids2017.json").read_text())["rows_kept"] == 65


def test_load_prefers_extracted_csv_and_strips_header_spaces(project):
    frame = make_flows({"BENIGN": 5})
    frame.columns = [f" {c}" for c in frame.columns]  # the original 2017 CSVs pad names with spaces
    frame.to_csv(project.raw / "cic-ids2017" / "Tuesday.csv", index=False)
    load_dataset("cic-ids2017", project)
    df = pd.read_parquet(project.interim / "cic-ids2017")
    assert "Flow Duration" in df.columns and set(df["day"]) == {"tuesday"}


def test_load_samples_2018_per_class(project):
    frame = make_flows({"BENIGN": 4000, "DDoS-HOIC": 1000}, start="2018-02-14 09:00:00")
    _zip_csvs(project.raw / "cse-cic-ids2018" / "CSECICIDS2018_improved.zip",
              {"Wednesday-14-02-2018.csv": frame})
    load_dataset("cse-cic-ids2018", project, seed=1)  # benign 5%, attacks 50% by config
    labels = pd.read_parquet(project.interim / "cse-cic-ids2018")["Label"]
    assert 120 < (labels == "BENIGN").sum() < 280
    assert 420 < (labels == "DDoS-HOIC").sum() < 580


def test_truncated_zip_gives_a_clear_error(project):
    (project.raw / "cic-ids2017" / "CICIDS2017_improved.zip").write_bytes(b"PK\x03\x04partial")
    with pytest.raises(IncompleteDownloadError, match="interrupted download"):
        load_dataset("cic-ids2017", project)


def test_missing_data_gives_a_clear_error(project):
    with pytest.raises(IncompleteDownloadError, match="Getting the data"):
        load_dataset("cic-ids2017", project)
