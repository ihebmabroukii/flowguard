"""End-to-end: processed data -> split -> pipeline -> training -> hybrid -> P1, P2, P3 -> report.

Synthetic flows stand in for the datasets: each family is a blob in 6 features. CSE-CIC-IDS2018
gets one family (Infiltration) that CIC-IDS2017 lacks here, to exercise the "2018-only" path.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from flowguard.config import load_config
from flowguard.data.split import temporal_split
from flowguard.features import pipeline as fp

CENTERS = {"Benign": 0.0, "DoS": 6.0, "PortScan": -6.0, "Botnet": 12.0, "Infiltration": -12.0}


def _flows(spec, dataset, seed):
    rng = np.random.default_rng(seed)
    frames = []
    for (day, family), n in spec.items():
        X = rng.normal(CENTERS[family], 1.0, (n, 6)).astype("float32")
        frame = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(6)])
        frame.insert(0, "family", family)
        frame.insert(0, "label_raw", family)
        frame.insert(0, "timestamp", pd.Timestamp("2017-07-03") + pd.to_timedelta(rng.permutation(n), unit="s"))
        frame.insert(0, "day", day)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "dataset", dataset)
    df.insert(0, "row_id", np.arange(len(df)))
    return df


@pytest.fixture
def ctx(project):
    from flowguard.eval.protocols import load_context
    from flowguard.models.hybrid import build_hybrid
    from flowguard.models.train import run as train_run

    cfg_path = project.root / "configs" / "experiments.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["models"]["softmax_regression"].update(batch_size=64, epochs=40, lr=0.05)
    cfg["models"]["isolation_forest"]["n_estimators"] = 30
    cfg["hybrid"].update(classifier="softmax_regression", max_benign_fpr=0.01)
    cfg["protocols"]["P1_in_dataset"] = cfg["protocols"]["P2_cross_dataset"] = ["softmax_regression", "hybrid"]
    cfg["p3"] = {"families": ["PortScan", "Heartbleed"], "max_train_rows": None}
    cfg_path.write_text(yaml.safe_dump(cfg))

    d17 = _flows({("monday", "Benign"): 900, ("wednesday", "Benign"): 600, ("wednesday", "DoS"): 300,
                  ("friday", "Benign"): 600, ("friday", "PortScan"): 300, ("friday", "Botnet"): 150},
                 "cic-ids2017", seed=0)
    d18 = _flows({("wed-14", "Benign"): 800, ("wed-14", "DoS"): 200, ("thu-01", "Infiltration"): 150},
                 "cse-cic-ids2018", seed=1)
    d17.to_parquet(project.processed_file("cic-ids2017"), index=False)
    d18.to_parquet(project.processed_file("cse-cic-ids2018"), index=False)
    split = temporal_split(d17)
    pd.DataFrame({"row_id": d17["row_id"], "split": split}).to_parquet(project.split_file("cic-ids2017"))
    fp.save(fp.fit_pipeline(d17[split == "train"], load_config("features", project.root)),
            project.models / fp.PIPELINE_FILE)

    train_run(["softmax_regression"], use_mlflow=False, paths=project)
    context = load_context(project, use_mlflow=False)
    clf = joblib.load(project.models / "p1" / "softmax_regression.joblib")
    tr, va = d17[split == "train"], d17[split == "val"]
    hybrid = build_hybrid(cfg, clf).fit(fp.transform(context.pipe, tr), tr["family"].to_numpy(),
                                        fp.transform(context.pipe, va), va["family"].to_numpy(),
                                        fit_classifier=False)
    joblib.dump(hybrid, project.models / "p1" / "hybrid.joblib")
    return context


def test_p1_scores_the_test_split(ctx):
    from flowguard.eval.protocols import run_p1

    results = run_p1(ctx)
    assert set(results) == {"softmax_regression", "hybrid"}
    for name, rec in results.items():
        m = rec["metrics"]
        assert m["macro_f1"] > 0.9, name
        assert {"pr_auc", "flows_per_second", "benign_fpr_per_10k", "per_class"} <= set(m)
        assert (ctx.paths.tables / f"p1_test_{name}.json").exists()
        assert (ctx.paths.tables / f"p1_test_{name}_confusion.csv").exists()
    test_rows = (pd.read_parquet(ctx.paths.split_file("cic-ids2017"))["split"] == "test").sum()
    assert results["hybrid"]["test_rows"] == test_rows


def test_p2_scores_shared_families_and_reports_2018_only_ones(ctx):
    from flowguard.eval.protocols import run_p2

    results = run_p2(ctx)
    rec = results["hybrid"]
    assert rec["shared_families"] == ["Benign", "DoS"]
    assert rec["families_only_in_2018"] == ["Infiltration"]
    assert set(rec["metrics"]["per_class"]) == {"Benign", "DoS"}
    assert rec["test_rows"] == 1000
    # the anomaly gate lets the hybrid alert on a family it was never trained on
    assert (rec["metrics"]["unseen_family_detection_rate"]
            >= results["softmax_regression"]["metrics"]["unseen_family_detection_rate"])
    assert rec["metrics"]["unseen_family_detection_rate"] > 0.9


def test_p3_holds_out_a_family_and_skips_unknown_ones(ctx):
    from flowguard.eval.protocols import run_p3

    results = run_p3(ctx)
    assert list(results) == ["PortScan"]                  # Heartbleed is not in the data: skipped
    rec = json.loads((ctx.paths.tables / "p3_PortScan.json").read_text())
    clf, hybrid = rec["softmax_regression"], rec["hybrid"]
    assert "PortScan" not in clf["held_out_predicted_as"]  # it cannot name what it never saw
    assert hybrid["flagged_any_alert"] >= clf["flagged_any_alert"]
    assert hybrid["flagged_any_alert"] > 0.9
    assert rec["hybrid_calibration"]["benign_fpr_val"] <= 0.01
    with pytest.raises(ValueError):
        run_p3(ctx, ["Benign"])


def test_report_builds_tables_and_figures(ctx):
    pytest.importorskip("matplotlib")
    from flowguard.eval.protocols import run_p1, run_p2, run_p3
    from flowguard.eval.report import build_report

    run_p1(ctx)
    run_p2(ctx)
    run_p3(ctx)
    out = build_report(ctx.paths)
    results = pd.read_csv(ctx.paths.tables / "results.csv")
    assert set(results["written_by"]) == {"us"}
    assert {"p1_macro_f1", "p2_macro_f1", "macro_f1_drop"} <= set(results.columns)
    assert (ctx.paths.tables / "results.md").read_text().startswith("| model | written_by |")
    assert set(out["figures"]) == {"p1_vs_p2_macro_f1.png", "recall_by_family.png", "p3_held_out_outcomes.png"}
    for name in out["figures"]:
        assert (ctx.paths.figures / name).stat().st_size > 10_000
