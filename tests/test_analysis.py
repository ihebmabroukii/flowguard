"""End-to-end runs of the analysis modules on the synthetic project from test_protocols.py:
CIs and McNemar in P1/P2, calibration, curves, ablation, drift, adaptation, and the report."""

import json

import numpy as np
import pandas as pd
import pytest

from tests.test_protocols import ctx  # noqa: F401  (pytest fixture)


def test_p1_and_p2_carry_intervals_and_mcnemar(ctx):  # noqa: F811
    from flowguard.eval.protocols import run_p1, run_p2

    p1 = run_p1(ctx)
    for rec in p1.values():
        ci = rec["metrics"]["ci"]["macro_f1"]
        assert ci["low"] <= rec["metrics"]["macro_f1"] <= ci["high"]
    table = pd.read_csv(ctx.paths.tables / "p1_mcnemar.csv")
    assert set(table["level"]) == {"family", "alert"}
    assert {"p_value", "p_holm", "significant"} <= set(table.columns)
    run_p2(ctx)
    assert (ctx.paths.tables / "p2_mcnemar.csv").exists()


def test_trivial_baselines_score_low_macro_f1_but_high_accuracy(ctx):  # noqa: F811
    from flowguard.eval.protocols import run_p1
    from flowguard.models.train import run as train_run

    train_run(["majority", "random_prior"], use_mlflow=False, paths=ctx.paths)
    rec = run_p1(ctx, ["majority", "random_prior", "softmax_regression"])
    majority = rec["majority"]["metrics"]
    assert majority["benign_fpr_per_10k"] == 0 and majority["detection_rate"] == 0
    assert majority["accuracy"] > 0.5 > majority["macro_f1"]
    assert rec["random_prior"]["metrics"]["macro_f1"] < rec["softmax_regression"]["metrics"]["macro_f1"]


def test_calibration_run_writes_tables(ctx):  # noqa: F811
    from flowguard.eval.calibration import run

    out = run(["softmax_regression"], ctx.paths)
    rec = json.loads((ctx.paths.tables / "calibration_softmax_regression.json").read_text())
    assert rec["temperature"] == pytest.approx(out["softmax_regression"]["temperature"])
    bins = pd.read_csv(ctx.paths.tables / "reliability_softmax_regression.csv")
    assert set(bins["stage"]) == {"before", "after"}


def test_curves(ctx):  # noqa: F811
    from flowguard.eval.curves import run

    out = run(["softmax_regression"], learning=True, paths=ctx.paths)
    loss = pd.read_csv(ctx.paths.tables / "loss_curve_softmax_regression.csv")
    assert loss["kept"].sum() == 1 and {"train_loss", "val_loss"} <= set(loss.columns)
    curve = pd.read_csv(ctx.paths.tables / out["learning_curve"])
    assert curve["n_train_rows"].is_monotonic_increasing
    assert (curve["val_macro_f1"] > 0.5).all()


def test_ablation_without_recleaning(ctx):  # noqa: F811
    from flowguard.eval.ablation import run

    table = run(["full", "no_log1p", "standard_scaler", "no_class_weights"], ["softmax_regression"], ctx.paths)
    assert list(table["variant"]) == ["full", "no_log1p", "standard_scaler", "no_class_weights"]
    assert table.loc[table["variant"] == "full", "delta_vs_full"].iloc[0] == 0
    assert (table["test_seen_in_train"] == 0).all()        # synthetic flows are all distinct
    assert (table["macro_f1_low"] <= table["macro_f1"]).all()


def test_ablation_helpers(features_cfg):
    from flowguard.eval.ablation import needs_recleaning, seen_in_train_share, variant_features_cfg

    cfg = variant_features_cfg(features_cfg, {"keep_columns": ["Dst Port"], "log1p": False})
    assert "Dst Port" not in cfg["drop"] and "Src Port" in cfg["drop"] and cfg["log1p"] is False
    assert "Dst Port" in features_cfg["drop"]                 # base config untouched
    assert needs_recleaning({"features": {"deduplicate": False}})
    assert not needs_recleaning({"features": {"log1p": False}}) and not needs_recleaning({})
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 0.0, 1.0]})
    test = pd.DataFrame({"a": [1.0, 9.0], "b": [0.0, 0.0]})
    assert seen_in_train_share(train, test) == 0.5


def test_drift_run(ctx):  # noqa: F811
    from flowguard.eval.drift import run
    from flowguard.eval.protocols import run_p1, run_p2

    run_p1(ctx)
    run_p2(ctx)
    out = run(ctx.paths)
    features = pd.read_csv(ctx.paths.tables / "drift_features.csv")
    assert {"all flows", "Benign", "DoS"} <= set(features["scope"])
    assert features["ks"].between(0, 1).all()
    assert 0 <= out["domain_auc"] <= 1
    assert (ctx.paths.tables / "drift_vs_recall_drop.csv").exists()


def test_adaptation_run(ctx):  # noqa: F811
    from flowguard.eval.adaptation import refit_statistics, run

    table = run(["softmax_regression", "hybrid"], ctx.paths)
    assert set(table["scaling"]) == {"2017 statistics", "refit on 2018 window"}
    assert len(table) == 4
    # Re-fitting keeps the selected columns, so the models get the inputs they were trained on.
    d18 = pd.read_parquet(ctx.paths.processed_file("cse-cic-ids2018"))
    feats = list(ctx.pipe.feature_names_in_)
    adapted = refit_statistics(ctx.pipe, d18[feats])
    assert list(adapted.transform(d18[feats]).columns) == list(ctx.pipe.transform(d18[feats]).columns)
    assert not np.allclose(adapted.transform(d18[feats]).to_numpy(), ctx.pipe.transform(d18[feats]).to_numpy())


def test_report_includes_the_analysis_figures(ctx):  # noqa: F811
    pytest.importorskip("matplotlib")
    from flowguard.eval import ablation, adaptation, calibration, curves, drift
    from flowguard.eval.protocols import run_p1, run_p2, run_p3
    from flowguard.eval.report import build_report

    run_p1(ctx)
    run_p2(ctx)
    run_p3(ctx)
    calibration.run(["softmax_regression"], ctx.paths)
    curves.run(["softmax_regression"], paths=ctx.paths)
    ablation.run(["full", "no_log1p"], ["softmax_regression"], ctx.paths)
    drift.run(ctx.paths)
    adaptation.run(["softmax_regression", "hybrid"], ctx.paths)
    out = build_report(ctx.paths)
    expected = {"loss_curves.png", "learning_curves.png", "reliability.png", "ablation.png",
                "drift_top_features.png", "p2_adaptation.png"}
    assert expected <= set(out["figures"])
    for name in out["figures"]:
        assert (ctx.paths.figures / name).stat().st_size > 10_000
    assert {"ablation.md", "p1_mcnemar.md", "calibration_summary.md"} <= set(out["tables"])
    results = pd.read_csv(ctx.paths.tables / "results.csv")
    assert {"p1_macro_f1_low", "p1_macro_f1_high"} <= set(results.columns)
