"""Turn the protocol outputs in reports/tables/ into the report's tables and figures.

Tables (reports/tables/):
    results.csv / results.md      one row per model: P1 and P2 macro-F1, the drop, false alerts, PR-AUC
    recall_by_family.csv          per-family recall, P1 and P2, every model
    p3_summary.csv / p3_summary.md  what happened to each held-out family, classifier alone vs hybrid
    calibration_summary, tuning_summary (.csv/.md); ablation, p2_adaptation, drift_by_family,
    p1_mcnemar, p2_mcnemar (.md versions of the analysis CSVs)
Figures (reports/figures/):
    p1_vs_p2_macro_f1.png         the generalization drop (RQ1 vs RQ2)
    recall_by_family.png          per-family recall heatmaps, P1 and P2 side by side
    p3_held_out_outcomes.png      where never-seen attacks end up (RQ3)
    loss_curves.png, learning_curves.png   bias / variance of our models
    reliability.png               calibration before and after temperature scaling
    tuning.png                    the random search, validation macro-F1 per trial
    ablation.png                  what each design choice is worth
    drift_top_features.png        the features that changed most between the networks (why RQ2 drops)
    p2_adaptation.png             P2 before and after re-fitting the scaling on unlabelled 2018 traffic

Every figure has a table with the same numbers, so nothing relies on color alone.

    python -m flowguard.eval.report
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flowguard.config import Paths
from flowguard.eval.style import GRID, INK, INK_SECONDARY, REFERENCE, SEQUENTIAL, SERIES, SURFACE

log = logging.getLogger(__name__)

OURS = {"softmax_regression", "mlp", "hybrid"}
TRIVIAL = {"majority", "random_prior"}



# --------------------------------------------------------------------------- reading

def _read(tables: Path, pattern: str) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(tables.glob(pattern))]


def results_table(tables: Path) -> pd.DataFrame:
    rows: dict[str, dict[str, Any]] = {}
    for protocol, pattern in (("p1", "p1_test_*.json"), ("p2", "p2_test_*.json")):
        for rec in _read(tables, pattern):
            m = rec["metrics"]
            row = rows.setdefault(rec["model"], {"model": rec["model"]})
            row[f"{protocol}_macro_f1"] = m["macro_f1"]
            ci = m.get("ci", {}).get("macro_f1")
            if ci:
                row[f"{protocol}_macro_f1_low"], row[f"{protocol}_macro_f1_high"] = ci["low"], ci["high"]
            row[f"{protocol}_benign_fpr_per_10k"] = m["benign_fpr_per_10k"]
            row[f"{protocol}_detection_rate"] = m["detection_rate"]
            row[f"{protocol}_pr_auc"] = m.get("pr_auc")
            if protocol == "p1":
                row["flows_per_second"] = m.get("flows_per_second")
                row["p1_accuracy"] = m.get("accuracy")  # shown only next to the trivial baselines
            if "unseen_family_detection_rate" in m:
                row["p2_unseen_family_detection_rate"] = m["unseen_family_detection_rate"]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows.values())
    df.insert(1, "written_by", np.select([df["model"].isin(OURS), df["model"].isin(TRIVIAL)],
                                         ["us", "trivial baseline"], "library"))
    if {"p1_macro_f1", "p2_macro_f1"} <= set(df.columns):
        df["macro_f1_drop"] = df["p1_macro_f1"] - df["p2_macro_f1"]
    order = {"us": 0, "library": 1, "trivial baseline": 2}
    return df.sort_values(["written_by", "model"], key=lambda s: s.map(order) if s.name == "written_by" else s)


def recall_table(tables: Path) -> pd.DataFrame:
    rows = []
    for protocol, pattern in (("P1", "p1_test_*.json"), ("P2", "p2_test_*.json")):
        for rec in _read(tables, pattern):
            for family, m in rec["metrics"]["per_class"].items():
                rows.append({"protocol": protocol, "model": rec["model"], "family": family,
                             "recall": m["recall"], "support": m["support"]})
    return pd.DataFrame(rows)


def p3_table(tables: Path) -> pd.DataFrame:
    rows = []
    for rec in _read(tables, "p3_*.json"):
        for model in (rec["classifier"], "hybrid"):
            o = rec[model]
            named_other = o["flagged_any_alert"] - o["flagged_unknown"]
            rows.append({"held_out": rec["held_out"], "model": model,
                         "flagged_unknown": o["flagged_unknown"],
                         "named_as_other_attack": named_other,
                         "missed_as_benign": o["missed_as_benign"],
                         "any_alert": o["flagged_any_alert"],
                         "benign_fpr_per_10k": o["benign_fpr_per_10k"]})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
    """Plain Markdown table, no extra dependency."""
    def fmt(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


# --------------------------------------------------------------------------- figures

def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(plt: Any, w: float, h: float, ncols: int = 1) -> tuple[Any, Any]:
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=SURFACE, squeeze=False)
    return fig, axes[0]


def plot_p1_vs_p2(results: pd.DataFrame, out: Path, plt: Any) -> Path | None:
    if not {"p1_macro_f1", "p2_macro_f1"} <= set(results.columns):
        return None
    df = results.iloc[::-1]
    y = np.arange(len(df))
    h = 0.36
    fig, (ax,) = _figure(plt, 7.5, 0.55 * len(df) + 1.4)
    _style(ax)
    for offset, col, color, label in ((h / 2, "p1_macro_f1", SERIES[0], "P1  same network (CIC-IDS2017)"),
                                      (-h / 2, "p2_macro_f1", SERIES[1], "P2  new network (CSE-CIC-IDS2018)")):
        xerr = None
        if {f"{col}_low", f"{col}_high"} <= set(df.columns):
            xerr = np.vstack([df[col] - df[f"{col}_low"], df[f"{col}_high"] - df[col]]).clip(min=0)
        ax.barh(y + offset, df[col], height=h - 0.04, color=color, label=label,
                edgecolor=SURFACE, linewidth=2, xerr=xerr,
                error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 2})
    labels = [f"{m}  (ours)" if w == "us" else f"{m}  (trivial)" if w == "trivial baseline" else m
              for m, w in zip(df["model"], df["written_by"])]
    ax.set_yticks(y, labels, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Detection quality drops on an unseen network", loc="left", color=INK, fontsize=12, pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9,
              labelcolor=INK_SECONDARY, handlelength=1, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_recall_heatmaps(recall: pd.DataFrame, out: Path, plt: Any) -> Path | None:
    if recall.empty:
        return None
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("recall", SEQUENTIAL)
    protocols = [p for p in ("P1", "P2") if p in set(recall["protocol"])]
    families = sorted(set(recall["family"]), key=lambda f: (f != "Benign", f))
    models = list(dict.fromkeys(recall["model"]))
    fig, axes = _figure(plt, 4.2 * len(protocols) + 1.2, 0.42 * len(models) + 2.2, ncols=len(protocols))
    image = None
    for ax, protocol in zip(axes, protocols):
        grid = (recall[recall["protocol"] == protocol]
                .pivot_table(index="model", columns="family", values="recall")
                .reindex(index=models, columns=families))
        image = ax.imshow(grid.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(families)), families, rotation=40, ha="right", fontsize=8, color=INK)
        ax.set_yticks(range(len(models)), models if ax is axes[0] else [""] * len(models), fontsize=9, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(protocol, loc="left", color=INK, fontsize=11)
        values = grid.to_numpy(dtype=float)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                if np.isnan(v):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color=INK_SECONDARY)
                else:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="#ffffff" if v > 0.6 else INK)
        ax.set_xticks(np.arange(-0.5, len(families)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(models)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
    fig.suptitle("Recall per attack family", x=0.01, ha="left", color=INK, fontsize=12)
    cbar = fig.colorbar(image, ax=list(axes), fraction=0.025, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, colors=INK_SECONDARY, length=0)
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_p3(p3: pd.DataFrame, out: Path, plt: Any) -> Path | None:
    if p3.empty:
        return None
    from matplotlib.ticker import PercentFormatter

    models = list(dict.fromkeys(p3["model"]))
    families = sorted(set(p3["held_out"]))
    parts = [("flagged_unknown", "Flagged Unknown", SERIES[0]),
             ("named_as_other_attack", "Alert, wrong family", SERIES[1]),
             ("missed_as_benign", "Missed (called benign)", SERIES[2])]
    fig, axes = _figure(plt, 5.0 * len(models), 0.5 * len(families) + 1.8, ncols=len(models))
    for ax, model in zip(axes, models):
        _style(ax)
        d = p3[p3["model"] == model].set_index("held_out").reindex(families)
        left = np.zeros(len(families))
        for col, label, color in parts:
            vals = d[col].to_numpy(dtype=float)
            ax.barh(families, vals, left=left, color=color, label=label, height=0.62,
                    edgecolor=SURFACE, linewidth=2)
            for k, (v, l) in enumerate(zip(vals, left)):
                if v >= 0.12:
                    ax.text(l + v / 2, k, f"{v:.0%}", ha="center", va="center", fontsize=8, color=INK)
            left += vals
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_title(f"{model}{'  (classifier alone)' if model != 'hybrid' else '  (classifier + anomaly gate)'}",
                     loc="left", color=INK, fontsize=10)
        ax.invert_yaxis()
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3, frameon=False, fontsize=9,
                   labelcolor=INK_SECONDARY, handlelength=1)
    fig.suptitle("Attacks never seen in training: what the detector does with them",
                 x=0.01, ha="left", color=INK, fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out



# --------------------------------------------------------------------------- analysis tables

def _csv(tables: Path, name: str) -> pd.DataFrame:
    path = tables / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def calibration_table(tables: Path) -> pd.DataFrame:
    rows = []
    for rec in _read(tables, "calibration_*.json"):
        rows.append({"model": rec["model"], "temperature": rec["temperature"],
                     **{f"{k}_{stage}": rec[stage][k] for stage in ("before", "after")
                        for k in ("ece", "nll", "brier", "saturated_share")}})
    return pd.DataFrame(rows)


def tuning_table(paths: Paths) -> pd.DataFrame:
    folder = paths.models / "tuning"
    rows = []
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        rec = json.loads(path.read_text())
        rows.append({"model": rec["model"], "trials": rec["n_trials"],
                     "config_val_macro_f1": rec["config_val_macro_f1"],
                     "best_val_macro_f1": rec["best_val_macro_f1"],
                     "best_params": json.dumps(rec["best_params"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- analysis figures

def _save_fig(fig: Any, out: Path, plt: Any) -> Path:
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def _line_style(ax: Any) -> None:
    _style(ax)
    ax.grid(axis="both", color=GRID, linewidth=0.8)


def _legend(ax: Any, ncol: int = 2) -> None:
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, frameon=False, fontsize=9,
              labelcolor=INK_SECONDARY, handlelength=1.4, borderaxespad=0.2)


def plot_loss_curves(tables: Path, out: Path, plt: Any) -> Path | None:
    curves = {p.stem.removeprefix("loss_curve_"): pd.read_csv(p) for p in sorted(tables.glob("loss_curve_*.csv"))}
    if not curves:
        return None
    fig, axes = _figure(plt, 4.6 * len(curves), 3.4, ncols=len(curves))
    for ax, (model, df) in zip(axes, curves.items()):
        _line_style(ax)
        ax.plot(df["epoch"], df["train_loss"], color=SERIES[0], linewidth=2, marker="o", markersize=4,
                label="training loss")
        if "val_loss" in df:
            ax.plot(df["epoch"], df["val_loss"], color=SERIES[1], linewidth=2, marker="o", markersize=4,
                    label="validation loss")
        kept = df.loc[df["kept"].astype(bool), "epoch"]
        if len(kept):
            ax.axvline(kept.iloc[0], color=REFERENCE, linewidth=1, linestyle="--")
            ax.annotate("weights kept", (kept.iloc[0], ax.get_ylim()[1]), xytext=(4, -12),
                        textcoords="offset points", fontsize=8, color=INK_SECONDARY)
        ax.set_title(model, loc="left", color=INK, fontsize=10, pad=24)
        ax.set_xlabel("epoch", color=INK_SECONDARY, fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("cross-entropy", color=INK_SECONDARY, fontsize=9)
        _legend(ax)
    fig.suptitle("Loss per epoch (early stopping on validation loss)", x=0.01, ha="left", color=INK,
                 fontsize=12, y=1.06)
    return _save_fig(fig, out, plt)


def plot_learning_curves(curve: pd.DataFrame, out: Path, plt: Any) -> Path | None:
    if curve.empty:
        return None
    models = list(dict.fromkeys(curve["model"]))
    fig, axes = _figure(plt, 4.6 * len(models), 3.4, ncols=len(models))
    for ax, model in zip(axes, models):
        _line_style(ax)
        d = curve[curve["model"] == model].sort_values("n_train_rows")
        ax.plot(d["n_train_rows"], d["train_macro_f1"], color=SERIES[0], linewidth=2, marker="o",
                markersize=5, label="training")
        ax.plot(d["n_train_rows"], d["val_macro_f1"], color=SERIES[1], linewidth=2, marker="o",
                markersize=5, label="validation")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_title(model, loc="left", color=INK, fontsize=10, pad=24)
        ax.set_xlabel("training flows (log scale)", color=INK_SECONDARY, fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("macro-F1", color=INK_SECONDARY, fontsize=9)
        _legend(ax)
    fig.suptitle("Learning curves: does more data help?", x=0.01, ha="left", color=INK, fontsize=12, y=1.06)
    return _save_fig(fig, out, plt)


def plot_reliability(tables: Path, out: Path, plt: Any) -> Path | None:
    files = sorted(tables.glob("reliability_*.csv"))
    if not files:
        return None
    summaries = {rec["model"]: rec for rec in _read(tables, "calibration_*.json")}
    fig, axes = _figure(plt, 3.9 * len(files), 3.9, ncols=len(files))
    for ax, path in zip(axes, files):
        model = path.stem.removeprefix("reliability_")
        d = pd.read_csv(path)
        _line_style(ax)
        ax.plot([0, 1], [0, 1], color=REFERENCE, linewidth=1, linestyle="--", label="perfect calibration")
        for stage, color, label in (("before", SERIES[1], "as trained"), ("after", SERIES[0], "temperature scaled")):
            b = d[(d["stage"] == stage) & (d["count"] > 0)]
            ece = summaries.get(model, {}).get(stage, {}).get("ece")
            ax.plot(b["mean_confidence"], b["accuracy"], color=color, linewidth=2, marker="o", markersize=5,
                    label=label + (f"  (ECE {ece:.3f})" if ece is not None else ""))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        temp = summaries.get(model, {}).get("temperature")
        ax.set_title(model + (f"   T = {temp:.2f}" if temp else ""), loc="left", color=INK, fontsize=10, pad=48)
        ax.set_xlabel("confidence", color=INK_SECONDARY, fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("accuracy in bin", color=INK_SECONDARY, fontsize=9)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, frameon=False, fontsize=8,
                  labelcolor=INK_SECONDARY, handlelength=1.4, borderaxespad=0.2)
    fig.suptitle("Reliability diagrams on the P1 test split", x=0.01, ha="left", color=INK, fontsize=12, y=1.14)
    return _save_fig(fig, out, plt)


def plot_tuning(tables: Path, out: Path, plt: Any) -> Path | None:
    files = sorted(tables.glob("tuning_*.csv"))
    if not files:
        return None
    fig, axes = _figure(plt, 4.6 * len(files), 3.4, ncols=len(files))
    for ax, path in zip(axes, files):
        model = path.stem.removeprefix("tuning_")
        d = pd.read_csv(path).reset_index(drop=True)
        _line_style(ax)
        x = d["lr"] if "lr" in d else d["trial"]
        random = (d["source"] == "random").to_numpy()
        ax.scatter(x[random], d.loc[random, "val_macro_f1"], s=36, color=SERIES[0], edgecolor=SURFACE,
                   linewidth=1.5, label="random trial", zorder=3)
        ax.scatter(x[~random], d.loc[~random, "val_macro_f1"], s=64, marker="D", color=SERIES[1],
                   edgecolor=SURFACE, linewidth=1.5, label="hand-picked config", zorder=4)
        best = int(d["val_macro_f1"].idxmax())
        ax.annotate(f"best {d.loc[best, 'val_macro_f1']:.3f}", (x[best], d.loc[best, "val_macro_f1"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
        if "lr" in d:
            ax.set_xscale("log")
        ax.set_xlabel("learning rate (log scale)" if "lr" in d else "trial", color=INK_SECONDARY, fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("validation macro-F1", color=INK_SECONDARY, fontsize=9)
        ax.set_title(model, loc="left", color=INK, fontsize=10, pad=24)
        _legend(ax)
    fig.suptitle("Random hyperparameter search (validation split)", x=0.01, ha="left", color=INK,
                 fontsize=12, y=1.06)
    return _save_fig(fig, out, plt)


def _grouped_barh(ax: Any, rows: list[str], groups: list[tuple[str, str, Any, Any]]) -> None:
    """Horizontal bars, one group per row, top to bottom; groups = [(label, color, values, xerr or None)]."""
    y = np.arange(len(rows))[::-1]
    h = 0.8 / len(groups)
    for k, (label, color, values, xerr) in enumerate(groups):
        offset = (len(groups) - 1) / 2 * h - k * h
        ax.barh(y + offset, np.asarray(values, dtype=float), height=h - 0.04, color=color, label=label,
                edgecolor=SURFACE, linewidth=2, xerr=xerr,
                error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 2})
    ax.set_yticks(y, rows, color=INK)


def plot_ablation(ablation: pd.DataFrame, out: Path, plt: Any) -> Path | None:
    if ablation.empty:
        return None
    variants = list(dict.fromkeys(ablation["variant"]))
    models = list(dict.fromkeys(ablation["model"]))[: len(SERIES)]
    fig, (ax,) = _figure(plt, 7.5, 0.3 * len(variants) * len(models) + 1.6)
    _style(ax)
    groups = []
    for model, color in zip(models, SERIES):
        d = ablation[ablation["model"] == model].set_index("variant").reindex(variants)
        err = np.vstack([d["macro_f1"] - d["macro_f1_low"], d["macro_f1_high"] - d["macro_f1"]]).clip(min=0)
        groups.append((model, color, d["macro_f1"], err))
    _grouped_barh(ax, variants, groups)
    ax.set_xlim(0, 1)
    ax.set_xlabel("P1 test macro-F1 (95% bootstrap interval)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Ablation: one design choice switched off at a time", loc="left", color=INK, fontsize=12, pad=26)
    _legend(ax, ncol=len(models))
    fig.tight_layout()
    return _save_fig(fig, out, plt)


def plot_drift(drift: pd.DataFrame, out: Path, plt: Any, top: int = 15) -> Path | None:
    if drift.empty:
        return None
    d = drift[drift["scope"] == "all flows"].nlargest(top, "ks").iloc[::-1]
    fig, (ax,) = _figure(plt, 7.5, 0.32 * len(d) + 1.4)
    _style(ax)
    ax.barh(d["feature"], d["ks"], color=SERIES[0], height=0.66, edgecolor=SURFACE, linewidth=2)
    for k, v in enumerate(d["ks"]):
        ax.text(v + 0.01, k, f"{v:.2f}", va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_xlim(0, 1.05)
    ax.tick_params(axis="y", labelsize=8, colors=INK)
    ax.set_xlabel("Kolmogorov-Smirnov D between 2017 and 2018 (0 = same distribution)",
                  color=INK_SECONDARY, fontsize=9)
    ax.set_title(f"The {len(d)} features that changed most between the two networks", loc="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    return _save_fig(fig, out, plt)


def plot_adaptation(adaptation: pd.DataFrame, out: Path, plt: Any) -> Path | None:
    if adaptation.empty:
        return None
    models = list(dict.fromkeys(adaptation["model"]))
    fig, (ax,) = _figure(plt, 7.5, 0.55 * len(models) + 1.6)
    _style(ax)
    groups = []
    for scaling, color in (("2017 statistics", SERIES[1]), ("refit on 2018 window", SERIES[2])):
        d = adaptation[adaptation["scaling"] == scaling].set_index("model").reindex(models)
        groups.append((f"scaling: {scaling}", color, d["macro_f1"], None))
    _grouped_barh(ax, models, groups)
    ax.set_xlim(0, 1)
    ax.set_xlabel("P2 macro-F1 on the second half of each 2018 day", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Re-fitting the scaling on unlabelled 2018 traffic", loc="left", color=INK, fontsize=12, pad=26)
    _legend(ax)
    fig.tight_layout()
    return _save_fig(fig, out, plt)

# --------------------------------------------------------------------------- build

def build_report(paths: Paths | None = None, figures: bool = True) -> dict[str, Any]:
    paths = paths or Paths.default()
    tables, figs = paths.tables, paths.figures
    tables.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"tables": [], "figures": []}

    results, recall, p3 = results_table(tables), recall_table(tables), p3_table(tables)
    if results.empty and p3.empty:
        raise FileNotFoundError("No protocol results in reports/tables/. Run `make eval` first.")
    calibration, tuning = calibration_table(tables), tuning_table(paths)
    ablation, adaptation = _csv(tables, "ablation.csv"), _csv(tables, "p2_adaptation.csv")
    drift, curve = _csv(tables, "drift_features.csv"), _csv(tables, "learning_curve.csv")

    if not results.empty:
        results.to_csv(tables / "results.csv", index=False)
        (tables / "results.md").write_text(to_markdown(results))
        out["tables"] += ["results.csv", "results.md"]
    if not recall.empty:
        recall.to_csv(tables / "recall_by_family.csv", index=False)
        out["tables"].append("recall_by_family.csv")
    if not p3.empty:
        p3.to_csv(tables / "p3_summary.csv", index=False)
        (tables / "p3_summary.md").write_text(to_markdown(p3))
        out["tables"] += ["p3_summary.csv", "p3_summary.md"]
    for stem, df in (("calibration_summary", calibration), ("tuning_summary", tuning)):
        if not df.empty:
            df.to_csv(tables / f"{stem}.csv", index=False)
            (tables / f"{stem}.md").write_text(to_markdown(df))
            out["tables"] += [f"{stem}.csv", f"{stem}.md"]
    for stem in ("ablation", "p2_adaptation", "drift_by_family", "p1_mcnemar", "p2_mcnemar"):
        df = _csv(tables, f"{stem}.csv")
        if not df.empty:
            (tables / f"{stem}.md").write_text(to_markdown(df))
            out["tables"].append(f"{stem}.md")

    if figures:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figs.mkdir(parents=True, exist_ok=True)
        for made in (plot_p1_vs_p2(results, figs / "p1_vs_p2_macro_f1.png", plt) if not results.empty else None,
                     plot_recall_heatmaps(recall, figs / "recall_by_family.png", plt),
                     plot_p3(p3, figs / "p3_held_out_outcomes.png", plt),
                     plot_loss_curves(tables, figs / "loss_curves.png", plt),
                     plot_learning_curves(curve, figs / "learning_curves.png", plt),
                     plot_reliability(tables, figs / "reliability.png", plt),
                     plot_tuning(tables, figs / "tuning.png", plt),
                     plot_ablation(ablation, figs / "ablation.png", plt),
                     plot_drift(drift, figs / "drift_top_features.png", plt),
                     plot_adaptation(adaptation, figs / "p2_adaptation.png", plt)):
            if made:
                out["figures"].append(made.name)
    log.info("report: tables %s, figures %s", out["tables"], out["figures"])
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build report tables and figures from protocol results")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_report(figures=not args.no_figures)


if __name__ == "__main__":
    main()
