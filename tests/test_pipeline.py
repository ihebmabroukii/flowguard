import numpy as np
import pandas as pd
import pytest

from flowguard.features.pipeline import (
    CorrelationFilter,
    build_pipeline,
    describe,
    fit_pipeline,
    load,
    save,
    signed_log1p,
    transform,
)

CFG = {"log1p": True, "min_variance": 0.0, "max_correlation": 0.98, "scaler": "quantile"}


def _frame(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "row_id": np.arange(n), "family": "Benign",           # metadata must be ignored
        "Flow Duration": rng.exponential(1e6, n),
        "Total Fwd Packet": rng.poisson(20, n).astype(float),
        "FWD Init Win Bytes": rng.choice([-1, 0, 8192, 65535], n).astype(float),
        "Fwd Bytes/Bulk Avg": 0.0,                             # constant
    })
    df["Subflow Fwd Packets"] = df["Total Fwd Packet"]         # exact twin
    df.loc[::50, "Flow Duration"] = np.nan
    return df


def test_signed_log1p_keeps_sign():
    out = signed_log1p(np.array([-1.0, 0.0, np.e - 1]))
    assert out == pytest.approx([-np.log(2), 0.0, 1.0])


def test_pipeline_removes_constant_and_duplicate_features():
    pipe = fit_pipeline(_frame(), CFG)
    info = describe(pipe)
    assert "row_id" not in pipe.feature_names_in_ and "family" not in pipe.feature_names_in_
    assert info["constant_features"] == ["Fwd Bytes/Bulk Avg"]
    assert info["correlated_features"] == {"Subflow Fwd Packets": "Total Fwd Packet"}
    assert info["features_out"] == ["Flow Duration", "Total Fwd Packet", "FWD Init Win Bytes"]


def test_transform_output_is_clean_float32_frame():
    pipe = fit_pipeline(_frame(), CFG)
    unseen = _frame(300, seed=1)
    unseen.loc[0, "Total Fwd Packet"] = 1e12                   # far outside the training range
    out = transform(pipe, unseen)
    assert isinstance(out, pd.DataFrame) and list(out.columns) == describe(pipe)["features_out"]
    assert out.dtypes.eq("float32").all()
    assert np.isfinite(out.to_numpy()).all()
    assert len(out) == 300


def test_transform_reports_missing_columns():
    pipe = fit_pipeline(_frame(), CFG)
    with pytest.raises(KeyError, match="Missing feature columns"):
        transform(pipe, _frame().drop(columns=["Flow Duration"]))


def test_fitting_uses_only_the_given_rows():
    a, b = _frame(seed=0), _frame(seed=1)
    b["Total Fwd Packet"] += 1000                              # a very different distribution
    b["split"] = "test"                                        # join columns are never features
    pipe_a = fit_pipeline(a, CFG)
    pipe_ab = fit_pipeline(pd.concat([a, b]), CFG)
    assert "split" not in pipe_ab.feature_names_in_
    assert not np.allclose(transform(pipe_a, b).to_numpy(), transform(pipe_ab, b).to_numpy())


def test_standard_scaler_option_and_bad_option():
    out = transform(fit_pipeline(_frame(), {**CFG, "scaler": "standard"}), _frame())
    assert abs(float(out["Total Fwd Packet"].mean())) < 1e-3
    with pytest.raises(ValueError):
        build_pipeline({**CFG, "scaler": "minmax"})


def test_correlation_filter_on_numpy():
    x = np.random.default_rng(0).random((100, 2))
    filt = CorrelationFilter(0.9).fit(np.column_stack([x, x[:, 0] * 2]))
    assert filt.keep_.tolist() == [0, 1]


def test_save_and_load_roundtrip(tmp_path):
    pipe = fit_pipeline(_frame(), CFG)
    save(pipe, tmp_path / "models" / "pipeline.joblib")
    again = load(tmp_path / "models" / "pipeline.joblib")
    pd.testing.assert_frame_equal(transform(pipe, _frame()), transform(again, _frame()))
