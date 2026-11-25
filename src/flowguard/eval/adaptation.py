"""Unsupervised adaptation to the new network: how much of the P2 drop does re-scaling recover?

Question: part of the P2 drop is that the same kind of traffic has different raw statistics on
another network (other link speeds, other hosts). If so, re-fitting only the *scaling* of the
features on the new network's traffic, without any label, should recover some of the drop.

What is re-fitted: the median imputer and the quantile scaler of the feature pipeline. What is
kept from 2017: which features are used (constant and correlated features stay removed) and every
trained model, untouched. No 2018 label is used anywhere in the adaptation.

Protocol, to keep adaptation and evaluation apart:
    Within every 2018 capture day, flows are ordered by time. The first half is the adaptation
    window (unlabelled traffic observed after deployment), the second half is scored. Grouping
    is by day only, never by family, because family is a label.
    Both the original and the adapted pipeline are scored on the same second half.

Caveat, stated in the report: the 2018 data is sampled while loading (5% of benign flows, 50% of
attack flows), so the adaptation window holds more attacks than live traffic would. That shifts
the fitted quantiles towards attack traffic.

Output: reports/tables/p2_adaptation.csv (+ .json)

    python -m flowguard.eval.adaptation
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

REFIT_STEPS = ("impute", "scale")


def refit_statistics(pipe: Any, X_raw: Any, steps: tuple[str, ...] = REFIT_STEPS) -> Any:
    """Copy of a fitted pipeline whose `steps` are re-fitted on X_raw; the other steps keep their state.

    Feature selection steps (variance, correlation) keep the columns chosen on 2017, so the
    models receive exactly the inputs they were trained on.
    """
    from sklearn.base import clone

    adapted = copy.deepcopy(pipe)
    Xt = X_raw
    for i, (name, step) in enumerate(adapted.steps):
        if name in steps:
            step = clone(step)
            step.set_output(transform="pandas")
            step.fit(Xt)
            adapted.steps[i] = (name, step)
        Xt = step.transform(Xt)
    adapted.set_output(transform="pandas")
    return adapted


def evaluate(model: Any, X: Any, y: np.ndarray, shared: list[str]) -> dict[str, Any]:
    from flowguard.eval.metrics import attack_pr_auc, classification_metrics, detection_rate

    pred = np.asarray(model.predict(X), dtype=object)
    in_shared = np.isin(y, shared)
    m = classification_metrics(y[in_shared], pred[in_shared], labels=shared)
    out = {"macro_f1": m["macro_f1"], "benign_fpr_per_10k": m["benign_fpr_per_10k"],
           "detection_rate": m["detection_rate"],
           "pr_auc": attack_pr_auc(y[in_shared], np.asarray(model.attack_score(X))[in_shared])}
    if (~in_shared).any():
        out["unseen_family_detection_rate"] = detection_rate(y[~in_shared], pred[~in_shared])
    return out


def run(models: list[str] | None = None, paths: Any = None) -> Any:
    import pandas as pd

    from flowguard.data.io import load_processed
    from flowguard.data.schema import DAY, FAMILY
    from flowguard.data.split import TEST, TRAIN, temporal_split
    from flowguard.eval.protocols import load_context, load_model
    from flowguard.models import TRIVIAL

    ctx = load_context(paths, use_mlflow=False)
    acfg = ctx.cfg.get("adaptation", {})
    models = models or acfg.get("models") or [m for m in ctx.cfg["protocols"]["P2_cross_dataset"] if m not in TRIVIAL]
    fraction = float(acfg.get("adaptation_fraction", 0.5))

    train_families = set(load_processed("cic-ids2017", ctx.paths, columns=[FAMILY])[FAMILY])
    tgt = load_processed("cse-cic-ids2018", ctx.paths)
    window = temporal_split(tgt, train_fraction=fraction, validation_fraction=0.0, group_cols=(DAY,)).to_numpy()
    adapt, scored = tgt[window == TRAIN], tgt[window == TEST]
    y = scored[FAMILY].to_numpy()
    shared = sorted(train_families & set(y))
    log.info("adaptation window %d flows (no labels used), scored half %d flows", len(adapt), len(scored))

    feats = list(ctx.pipe.feature_names_in_)
    adapted_pipe = refit_statistics(ctx.pipe, adapt[feats])
    X_original = ctx.pipe.transform(scored[feats]).astype("float32")
    X_adapted = adapted_pipe.transform(scored[feats]).astype("float32")

    rows = []
    for name in models:
        model = load_model(ctx.paths, name)
        for scaling, X in (("2017 statistics", X_original), ("refit on 2018 window", X_adapted)):
            rows.append({"model": name, "scaling": scaling, **evaluate(model, X, y, shared)})
        before, after = rows[-2]["macro_f1"], rows[-1]["macro_f1"]
        log.info("%-18s P2 macro-F1 %.4f -> %.4f (%+.4f)", name, before, after, after - before)

    table = pd.DataFrame(rows)
    p1 = {}
    for name in models:
        path = ctx.paths.tables / f"p1_test_{name}.json"
        if path.exists():
            p1[name] = json.loads(path.read_text())["metrics"]["macro_f1"]
    if p1:
        table["p1_macro_f1"] = table["model"].map(p1)
        base = table[table["scaling"] == "2017 statistics"].set_index("model")["macro_f1"]
        drop = table["model"].map(p1) - table["model"].map(base)
        gain = table["macro_f1"] - table["model"].map(base)
        # Share of the P1 -> P2 drop that adaptation recovers (only meaningful when there is a drop).
        table["drop_recovered"] = np.where(drop > 1e-9, gain / drop, np.nan)
    ctx.paths.tables.mkdir(parents=True, exist_ok=True)
    table.to_csv(ctx.paths.tables / "p2_adaptation.csv", index=False)
    (ctx.paths.tables / "p2_adaptation.json").write_text(json.dumps(
        {"adaptation_rows": int(len(adapt)), "scored_rows": int(len(scored)), "shared_families": shared,
         "refit_steps": list(REFIT_STEPS), "results": table.to_dict("records")}, indent=2, default=str))
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Re-fit the feature scaling on unlabelled 2018 traffic")
    parser.add_argument("--models", nargs="+", help="default: the P2 models")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.models)


if __name__ == "__main__":
    main()
