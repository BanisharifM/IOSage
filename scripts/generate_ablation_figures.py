#!/usr/bin/env python3
"""Generate ML ablation figures and LaTeX table for the paper.

Figures:
  - fig_lobo.pdf           -- Leave-One-Benchmark-Out grouped bar chart
  - fig_training_data_ablation.pdf -- GT-only vs Biquality comparison

Table:
  - tab_ml_ablations.tex   -- Combined 3-panel ablation table (already written)

Usage:
    python scripts/generate_ablation_figures.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_DIR / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = PROJECT_DIR / "results" / "ml_ablations.json"

# ---------------------------------------------------------------------------
# Colorblind-safe Okabe-Ito palette (same as generate_evaluation_figures.py)
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

# Alias for backward compat
OI = COLORS

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
    """IEEE-compatible publication style."""
    plt.rcParams.update(RCPARAMS_SC2026)


def save_fig(fig, name):
    """Save figure as PDF and PNG."""
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=300)
    plt.close(fig)
    logger.info("Saved: %s  (%d bytes)", pdf, pdf.stat().st_size)
    logger.info("Saved: %s  (%d bytes)", png, png.stat().st_size)


def load_results():
    """Load ablation results from JSON."""
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)


# ===========================================================================
# Figure: Leave-One-Benchmark-Out (LOBO)
# ===========================================================================

def fig_lobo(data):
    """Grouped bar chart: 6 benchmarks, With vs Without Micro-F1."""
    logger.info("Generating LOBO figure...")

    lobo = data["ablation3_lobo"]

    # Order benchmarks by delta (largest drop last for visual impact)
    benchmarks_order = ["IOR", "DLIO", "h5bench", "mdtest", "HACC-IO", "custom"]
    key_map = {
        "IOR": "ior",
        "DLIO": "dlio",
        "h5bench": "h5bench",
        "mdtest": "mdtest",
        "HACC-IO": "hacc_io",
        "custom": "custom",
    }

    with_vals = []
    without_vals = []
    deltas = []
    for bm in benchmarks_order:
        key = key_map[bm]
        with_vals.append(lobo[key]["with_benchmark"]["micro_f1"])
        without_vals.append(lobo[key]["without_benchmark"]["micro_f1"])
        deltas.append(lobo[key]["delta_micro_f1"])

    with_vals = np.array(with_vals)
    without_vals = np.array(without_vals)
    deltas = np.array(deltas)

    x = np.arange(len(benchmarks_order))
    width = 0.34

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    bars_w = ax.bar(
        x - width / 2, with_vals, width,
        color=OI["green"], edgecolor="black", linewidth=0.4,
        hatch="", label="With benchmark", zorder=3,
    )
    bars_wo = ax.bar(
        x + width / 2, without_vals, width,
        color=OI["vermilion"], edgecolor="black", linewidth=0.4,
        hatch="//", label="Without benchmark", zorder=3,
    )

    # Delta annotations above each pair
    for i in range(len(benchmarks_order)):
        y_top = max(with_vals[i], without_vals[i])
        delta_str = f"$-${deltas[i]:.2f}" if deltas[i] > 0 else f"{deltas[i]:.2f}"
        ax.annotate(
            delta_str,
            xy=(x[i], y_top + 0.04),
            ha="center", va="bottom",
            fontsize=5.5, fontweight="bold", color=OI["blue"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks_order, fontsize=7)
    ax.set_ylabel("Micro-F1")
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.8)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save_fig(fig, "fig_lobo")


# ===========================================================================
# Figure: Training Data Ablation (GT-only vs Biquality)
# ===========================================================================

def fig_training_data_ablation(data):
    """Two-bar comparison: GT-only vs Biquality Micro-F1."""
    logger.info("Generating Training Data Ablation figure...")

    biquality_micro = data["ablation2_gt_only"]["biquality"]["micro_f1"]
    gt_only_micro = data["ablation2_gt_only"]["gt_only"]["micro_f1"]
    biquality_macro = data["ablation2_gt_only"]["biquality"]["macro_f1"]
    gt_only_macro = data["ablation2_gt_only"]["gt_only"]["macro_f1"]

    labels = ["GT-only\n(187 samples)", "Biquality\n(91K + 187)"]
    micro_vals = [gt_only_micro, biquality_micro]
    macro_vals = [gt_only_macro, biquality_macro]

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    bars_micro = ax.bar(
        x - width / 2, micro_vals, width,
        color=OI["blue"], edgecolor="black", linewidth=0.4,
        hatch="", label="Micro-F1", zorder=3,
    )
    bars_macro = ax.bar(
        x + width / 2, macro_vals, width,
        color=OI["cyan"], edgecolor="black", linewidth=0.4,
        hatch="//", label="Macro-F1", zorder=3,
    )

    # Value labels on bars
    for bar, val in zip(bars_micro, micro_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom",
            fontsize=6.5, fontweight="bold", color=OI["blue"],
        )
    for bar, val in zip(bars_macro, macro_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom",
            fontsize=6.5, fontweight="bold", color=OI["blue"],
        )

    # Improvement annotation
    micro_improvement = (biquality_micro - gt_only_micro) / gt_only_micro * 100
    macro_improvement = (biquality_macro - gt_only_macro) / gt_only_macro * 100

    # Draw arrow from GT-only micro bar to Biquality micro bar
    ax.annotate(
        f"+{micro_improvement:.1f}%",
        xy=(1 - width / 2, biquality_micro + 0.035),
        xytext=(0.5, biquality_micro + 0.065),
        ha="center", va="bottom",
        fontsize=7, fontweight="bold", color=OI["green"],
        arrowprops=dict(
            arrowstyle="->", color=OI["green"],
            lw=1.0, connectionstyle="arc3,rad=-0.15",
        ),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0.75, 1.0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.legend(loc="lower right", fontsize=6.5, framealpha=0.8)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save_fig(fig, "fig_training_data_ablation")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    apply_style()

    data = load_results()
    logger.info("Loaded ablation results from %s", RESULTS_FILE)

    fig_lobo(data)
    fig_training_data_ablation(data)

    logger.info("All ablation figures saved to %s", FIG_DIR)
    logger.info("LaTeX table at paper/tables/tab_ml_ablations.tex")
