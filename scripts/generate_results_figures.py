#!/usr/bin/env python3
"""Generate paper figures for ML results section.

Figures:
  1. Baseline comparison bar chart
  2. Training progression (Phase 1 -> Phase 2)
  3. Per-label confusion matrices
  4. Model comparison with error bars (5-seed)

Usage:
    python scripts/generate_results_figures.py
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_DIR / "paper" / "figures" / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DIMS = ["access_granularity", "metadata_intensity", "parallelism_efficiency",
        "access_pattern", "interface_choice", "file_strategy",
        "throughput_utilization", "healthy"]
DIM_SHORT = ["Granularity", "Metadata", "Parallelism", "Pattern",
             "Interface", "File Strat.", "Throughput", "Healthy"]

# ---------------------------------------------------------------------------
# Style guide constants
# ---------------------------------------------------------------------------
COLORS = {
    "blue":      "#0072B2",
    "orange":    "#E69F00",
    "green":     "#009E73",
    "vermilion": "#D55E00",
    "purple":    "#CC79A7",
    "cyan":      "#56B4E9",
    "yellow":    "#F0E442",
    "gray":      "#BBBBBB",
    "black":     "#000000",
}

PALETTE_8 = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
             "#CC79A7", "#56B4E9", "#F0E442", "#BBBBBB"]

HATCHES = ["", "//", "\\\\", "xx", "..", "++", "oo", "**"]

RCPARAMS_SC2026 = {
    # Fonts — serif to match IEEE body text
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 8,
    "mathtext.fontset": "stix",

    # Axes
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,

    # Ticks
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",

    # Legend
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.5,

    # Grid
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,

    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    # Lines
    "lines.linewidth": 1.0,
    "lines.markersize": 4,

    # Font embedding (CRITICAL)
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style():
    """Apply IEEE publication style."""
    plt.rcParams.update(RCPARAMS_SC2026)


def save_fig(fig, path):
    """Save figure as PDF and PNG."""
    fig.savefig(path, format="pdf")
    png_path = str(path).replace(".pdf", ".png")
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    logger.info("Saved: %s (%d bytes)", path, path.stat().st_size)


def fig_baseline_comparison():
    """Bar chart comparing all methods on GT test set."""
    methods = ["Majority", "LogReg", "Drishti", "Phase 1\n(Heuristic)", "Phase 2\n(Biquality)"]
    micro_f1 = [0.158, 0.284, 0.384, 0.385, 0.923]
    macro_f1 = [0.036, 0.240, 0.283, 0.282, 0.900]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.16, 2.8))
    bars1 = ax.bar(x - width/2, micro_f1, width, label="Micro-F1",
                   color=COLORS["blue"], edgecolor="black", linewidth=0.4, hatch="")
    bars2 = ax.bar(x + width/2, macro_f1, width, label="Macro-F1",
                   color=COLORS["orange"], edgecolor="black", linewidth=0.4, hatch="//")

    ax.set_ylabel("F1 Score")
    ax.set_title("Baseline Comparison on Ground-Truth Test Set (n=436)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6.5)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6.5)

    fig.tight_layout()
    path = FIG_DIR / "fig_baseline_comparison.pdf"
    save_fig(fig, path)


def fig_training_progression():
    """Per-label F1 comparison: Phase 1 vs Phase 2."""
    phase1_f1 = [0.602, 0.464, 0.000, 0.189, 0.609, 0.000, 0.124, 0.262]
    phase2_f1 = [0.926, 1.000, 1.000, 0.690, 0.956, 0.903, 0.899, 0.824]

    x = np.arange(len(DIMS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.16, 2.8))
    bars1 = ax.bar(x - width/2, phase1_f1, width, label="Phase 1 (Heuristic Only)",
                   color=COLORS["vermilion"], edgecolor="black", linewidth=0.4, hatch="//")
    bars2 = ax.bar(x + width/2, phase2_f1, width, label="Phase 2 (Biquality)",
                   color=COLORS["green"], edgecolor="black", linewidth=0.4, hatch="")

    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Dimension Detection Quality: Phase 1 vs Phase 2")
    ax.set_xticks(x)
    ax.set_xticklabels(DIM_SHORT, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Highlight improvement
    for i in range(len(DIMS)):
        if phase2_f1[i] - phase1_f1[i] > 0.3:
            ax.annotate(f"+{phase2_f1[i]-phase1_f1[i]:.2f}",
                        xy=(x[i] + width/2, phase2_f1[i]),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=6.5, color=COLORS["green"],
                        fontweight="bold")

    fig.tight_layout()
    path = FIG_DIR / "fig_training_progression.pdf"
    save_fig(fig, path)


def fig_model_comparison():
    """3 models with 5-seed error bars."""
    models = ["XGBoost", "LightGBM", "Random Forest"]
    micro_means = [0.920, 0.901, 0.897]
    micro_stds = [0.004, 0.006, 0.004]
    macro_means = [0.896, 0.872, 0.877]
    macro_stds = [0.004, 0.010, 0.014]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    bars1 = ax.bar(x - width/2, micro_means, width, yerr=micro_stds,
                   label="Micro-F1", color=COLORS["blue"], capsize=3,
                   edgecolor="black", linewidth=0.4, hatch="")
    bars2 = ax.bar(x + width/2, macro_means, width, yerr=macro_stds,
                   label="Macro-F1", color=COLORS["orange"], capsize=3,
                   edgecolor="black", linewidth=0.4, hatch="//")

    ax.set_ylabel("F1 Score")
    ax.set_title("Model Comparison (5 seeds, mean +/- std)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.8, 0.98)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = FIG_DIR / "fig_model_comparison.pdf"
    save_fig(fig, path)


def fig_confusion_matrices():
    """8 per-label confusion matrices in a 2x4 grid."""
    import yaml

    # Load model and test data
    with open(PROJECT_DIR / "models" / "phase2" / "xgboost_biquality_w100.pkl", "rb") as f:
        models = pickle.load(f)

    with open(PROJECT_DIR / "configs" / "training.yaml") as f:
        config = yaml.safe_load(f)

    prod_feat = pd.read_parquet(PROJECT_DIR / config["paths"]["production_features"])
    exclude = set(config.get("exclude_features", []))
    for col in prod_feat.columns:
        if col.startswith("_") or col.startswith("drishti_"):
            exclude.add(col)
    feature_cols = [c for c in prod_feat.columns if c not in exclude]

    test_feat = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "benchmark" / "test_features.parquet")
    test_labels = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "benchmark" / "test_labels.parquet")

    X_test = []
    for col in feature_cols:
        if col in test_feat.columns:
            X_test.append(test_feat[col].values)
        else:
            X_test.append(np.zeros(len(test_feat)))
    X_test = np.column_stack(X_test).astype(np.float32)
    y_test = test_labels[DIMS].values

    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.5))

    for idx, (dim, ax) in enumerate(zip(DIMS, axes.flat)):
        y_pred = models[dim].predict(X_test)
        cm = confusion_matrix(y_test[:, idx], y_pred, labels=[0, 1])

        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(DIM_SHORT[idx], fontsize=8, fontweight="bold")

        # Labels
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])
        if idx >= 4:
            ax.set_xlabel("Predicted")
        if idx % 4 == 0:
            ax.set_ylabel("True")

    fig.suptitle("Per-Dimension Confusion Matrices (Phase 2 XGBoost, n=436)",
                 fontsize=9, y=1.02)
    fig.tight_layout()

    path = FIG_DIR / "fig_confusion_matrices.pdf"
    save_fig(fig, path)


if __name__ == "__main__":
    apply_style()
    logger.info("Generating results figures...")
    fig_baseline_comparison()
    fig_training_progression()
    fig_model_comparison()
    fig_confusion_matrices()
    logger.info("All figures saved to %s", FIG_DIR)
