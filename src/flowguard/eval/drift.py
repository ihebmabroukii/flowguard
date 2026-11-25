"""Why do scores drop on the 2018 network? Feature drift between CIC-IDS2017 and CSE-CIC-IDS2018.

Three measurements, all from scratch:

1. Per-feature two-sample Kolmogorov-Smirnov statistic
       D = max_x |F_2017(x) - F_2018(x)|      (0 = same distribution, 1 = disjoint)
   computed on the raw feature values. D only depends on the ranks of the values, so it is
   the same before and after our log1p and quantile scaling: it measures the shift the model
   actually faces. With millions of flows every p-value is ~0, so D itself is reported.

2. Population Stability Index, the drift measure used in industry model monitoring
       PSI = sum over bins (q_b - p_b) * ln(q_b / p_b)
   with 10 bins at the 2017 deciles. Rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 major.

3. Domain classifier ("adversarial validation"): our own softmax regression is trained to tell
   2017 benign flows from 2018 benign flows. ROC-AUC 0.5 = the networks look identical; close to
   1.0 = every flow betrays which network it came from. Its weights rank the telling features.

Both are computed for all flows and per shared family, then lined up against the per-family
recall drop from P1 to P2: do the families that drift most lose the most recall?

Outputs (reports/tables/): drift_features.csv, drift_by_family.csv, drift_domain_classifier.json,
drift_vs_recall_drop.csv (when P1 and P2 results exist),
drift_summary.json

    python -m flowguard.eval.drift
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

MAX_ROWS = 200_000
PSI_BINS = 10
PSI_EPS = 1e-4


# --------------------------------------------------------------------------- statistics (NumPy only)

def _finite(values: Any) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    return v[np.isfinite(v)]


def ks_statistic(a: Any, b: Any) -> float:
    """Two-sample Kolmogorov-Smirnov D: the largest gap between the two empirical CDFs."""
    a, b = np.sort(_finite(a)), np.sort(_finite(b))
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def psi(expected: Any, actual: Any, n_bins: int = PSI_BINS) -> float:
    """Population Stability Index of `actual` against `expected`, bins at the expected quantiles.

    Repeated quantiles (features with many identical values) are merged into one bin.
    """
    e, a = _finite(expected), _finite(actual)
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, n_bins + 1)[1:-1]))
    # Bin i holds (edge[i-1], edge[i]]: a value equal to an edge stays below it, so even a
    # constant 2017 feature still gets a "larger than every 2017 value" bin.
    p = np.bincount(np.searchsorted(edges, e, side="left"), minlength=len(edges) + 1) / len(e)
    q = np.bincount(np.searchsorted(edges, a, side="left"), minlength=len(edges) + 1) / len(a)
    p, q = np.maximum(p, PSI_EPS), np.maximum(q, PSI_EPS)
    return float(((q - p) * np.log(q / p)).sum())


def roc_auc(y: Any, score: Any) -> float:
    """ROC-AUC as the Mann-Whitney statistic: P(score of a positive > score of a negative), ties count 1/2."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    # average ranks over ties
    _, first, counts = np.unique(sorted_s, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman(x: Any, y: Any) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return float("nan")

    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        _, first, counts = np.unique(v[order], return_index=True, return_counts=True)
        r = np.empty(len(v))
        r[order] = np.repeat(first + (counts + 1) / 2.0, counts)
        return r

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------- tables

def _sample(df: Any, max_rows: int, seed: int) -> Any:
    return df if len(df) <= max_rows else df.sample(max_rows, random_state=seed)


def drift_table(source: Any, target: Any, features: list[str], max_rows: int = MAX_ROWS, seed: int = 0) -> Any:
    """One row per feature: KS D, PSI, and the medians on both networks, sorted by D."""
    import pandas as pd

    s, t = _sample(source, max_rows, seed), _sample(target, max_rows, seed)
    rows = []
    for f in features:
        a, b = s[f].to_numpy(), t[f].to_numpy()
        rows.append({"feature": f, "ks": ks_statistic(a, b), "psi": psi(a, b),
                     "median_2017": float(np.nanmedian(a)) if len(a) else np.nan,
                     "median_2018": float(np.nanmedian(b)) if len(b) else np.nan})
    return pd.DataFrame(rows).sort_values("ks", ascending=False, ignore_index=True)


def domain_classifier(source: Any, target: Any, pipe: Any, seed: int = 0, max_rows: int = MAX_ROWS) -> dict[str, Any]:
    """Train our softmax regression to tell the networks apart; held-out ROC-AUC and top weights."""
    from flowguard.features import pipeline as fp
    from flowguard.models.scratch import SoftmaxRegression

    s, t = _sample(source, max_rows // 2, seed), _sample(target, max_rows // 2, seed)
    X = np.vstack([fp.transform(pipe, s).to_numpy(), fp.transform(pipe, t).to_numpy()])
    y = np.r_[np.zeros(len(s), dtype=np.int64), np.ones(len(t), dtype=np.int64)]
    names = list(pipe.get_feature_names_out())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    cut = len(y) // 2
    fit_idx, test_idx = order[:cut], order[cut:]
    weights = np.where(y[fit_idx] == 1, 0.5 / max(y[fit_idx].mean(), 1e-12), 0.5 / max(1 - y[fit_idx].mean(), 1e-12))
    model = SoftmaxRegression(lr=0.05, epochs=20, batch_size=4096, alpha=1e-4, random_state=seed)
    model.fit(X[fit_idx], y[fit_idx], sample_weight=weights)
    score = model.predict_proba(X[test_idx])[:, 1]
    coef = model.coef_[1] - model.coef_[0]  # log-odds weight towards "2018" (coef_ is classes x features)
    top = np.argsort(-np.abs(coef))[:15]
    return {"roc_auc": roc_auc(y[test_idx], score), "n_fit": int(cut), "n_test": int(len(test_idx)),
            "top_features": [{"feature": names[i], "weight_towards_2018": float(coef[i])} for i in top]}


def recall_drop(tables: Any, model: str) -> Any:
    """Per-family recall P1 and P2 for one model, from reports/tables/p{1,2}_test_<model>.json."""
    import pandas as pd

    p1, p2 = tables / f"p1_test_{model}.json", tables / f"p2_test_{model}.json"
    if not (p1.exists() and p2.exists()):
        return pd.DataFrame()
    r1 = json.loads(p1.read_text())["metrics"]["per_class"]
    r2 = json.loads(p2.read_text())["metrics"]["per_class"]
    rows = [{"family": f, "recall_p1": r1[f]["recall"], "recall_p2": r2[f]["recall"]} for f in r2 if f in r1]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["recall_drop"] = df["recall_p1"] - df["recall_p2"]
    return df


# --------------------------------------------------------------------------- CLI

def run(paths: Any = None) -> dict[str, Any]:
    import pandas as pd

    from flowguard.config import Paths, load_config
    from flowguard.data.io import SPLIT, load_p1, load_processed
    from flowguard.data.schema import BENIGN, FAMILY, feature_columns
    from flowguard.data.split import TRAIN
    from flowguard.features import pipeline as fp

    paths = paths or Paths.default()
    exp_cfg = load_config("experiments", paths.root)
    seed = int(exp_cfg.get("seed", 42))
    max_rows = int(exp_cfg.get("drift", {}).get("max_rows", MAX_ROWS))
    src = load_p1(paths)
    src = src[src[SPLIT] == TRAIN]
    tgt = load_processed("cse-cic-ids2018", paths)
    features = [f for f in feature_columns(src) if f in tgt.columns and f != SPLIT]
    shared = sorted(set(src[FAMILY]) & set(tgt[FAMILY]))

    per_feature = [drift_table(src, tgt, features, max_rows, seed).assign(scope="all flows")]
    summary = []
    for family in shared:
        table = drift_table(src[src[FAMILY] == family], tgt[tgt[FAMILY] == family], features, max_rows, seed)
        per_feature.append(table.assign(scope=family))
        summary.append({"family": family, "mean_ks": float(table["ks"].mean()),
                        "median_ks": float(table["ks"].median()),
                        "features_psi_over_0.25": int((table["psi"] > 0.25).sum()),
                        "top_feature": table["feature"].iloc[0], "top_ks": float(table["ks"].iloc[0]),
                        "rows_2017": int((src[FAMILY] == family).sum()), "rows_2018": int((tgt[FAMILY] == family).sum())})
    features_df = pd.concat(per_feature, ignore_index=True)
    summary_df = pd.DataFrame(summary)
    paths.tables.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(paths.tables / "drift_features.csv", index=False)
    summary_df.to_csv(paths.tables / "drift_by_family.csv", index=False)

    pipe = fp.load(paths.models / fp.PIPELINE_FILE)
    domain = domain_classifier(src[src[FAMILY] == BENIGN], tgt[tgt[FAMILY] == BENIGN], pipe, seed, max_rows)
    (paths.tables / "drift_domain_classifier.json").write_text(json.dumps(domain, indent=2))
    log.info("domain classifier (2017 vs 2018 benign): ROC-AUC %.4f", domain["roc_auc"])

    out: dict[str, Any] = {"domain_auc": domain["roc_auc"]}
    model = exp_cfg.get("hybrid", {}).get("classifier", "mlp")
    drops = recall_drop(paths.tables, model)
    if not drops.empty and not summary_df.empty:
        joined = drops.merge(summary_df[["family", "mean_ks", "features_psi_over_0.25"]], on="family")
        joined.insert(0, "model", model)
        joined.to_csv(paths.tables / "drift_vs_recall_drop.csv", index=False)
        out["spearman_mean_ks_vs_recall_drop"] = spearman(joined["mean_ks"], joined["recall_drop"])
        log.info("Spearman(mean KS, recall drop) over %d families: %.3f",
                 len(joined), out["spearman_mean_ks_vs_recall_drop"])
        out["n_families"] = int(len(joined))
    (paths.tables / "drift_summary.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="Feature drift between CIC-IDS2017 and CSE-CIC-IDS2018").parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
