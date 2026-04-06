#!/usr/bin/env python3
"""Generate paper figures for the evaluation / closed-loop / LLM section.

Figures:
  17. fig_closed_loop_speedup.pdf  -- Bar chart of 3 IOR closed-loop pairs
  18. fig_pipeline_walkthrough.pdf  -- End-to-end case study (5 panels)
  19. fig_llm_groundedness.pdf      -- Grouped bars: 3 LLMs x 2 conditions
  20. fig_ablation_trackb.pdf       -- Ablation study bar chart

Usage:
    python scripts/generate_evaluation_figures.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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

# ---------------------------------------------------------------------------
# Colorblind-safe Okabe-Ito palette
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


# ===========================================================================
# Figure 17: Closed-Loop Speedup
# ===========================================================================

def fig_closed_loop_speedup():
    """Bar chart of 3 IOR closed-loop pairs with log-scale y-axis."""
    logger.info("Generating Figure 17: Closed-Loop Speedup...")

    labels = [
        "Access\nGranularity",
        "Throughput\n(fsync)",
        "Access\nPattern",
    ]
    before = np.array([49.24, 499.71, 2211.75])
    after = np.array([2208.43, 3407.57, 3280.81])
    speedups = [44.8, 6.8, 1.5]
    geo_mean = 7.7

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    bars_b = ax.bar(x - width / 2, before, width,
                    color=OI["vermilion"], edgecolor="black", linewidth=0.4,
                    hatch="//", label="Before fix", zorder=3)
    bars_a = ax.bar(x + width / 2, after, width,
                    color=OI["green"], edgecolor="black", linewidth=0.4,
                    hatch="", label="After fix", zorder=3)

    # Log scale
    ax.set_yscale("log")
    ax.set_ylim(10, 10000)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:.1f}"))

    # Geometric mean line
    ax.axhline(geo_mean * before.min(), color=OI["gray"], linestyle=":",
               linewidth=0.6, alpha=0.0)  # hidden helper

    # Speedup annotations
    for i, sp in enumerate(speedups):
        y_top = max(before[i], after[i])
        ax.annotate(
            f"{sp}x",
            xy=(x[i], y_top * 1.25),
            ha="center", va="bottom",
            fontsize=7, fontweight="bold", color=OI["blue"],
        )

    # Geo-mean annotation (dashed horizontal)
    # Compute a representative bandwidth: geo_mean of after values for reference
    ax.axhline(geo_mean, color=OI["blue"], linestyle="--", linewidth=0.8,
               alpha=0.0)  # not meaningful on bandwidth axis

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Write Bandwidth (MiB/s)")
    ax.legend(loc="upper left", fontsize=6.5, framealpha=0.8)
    ax.grid(axis="y", alpha=0.25, which="both")

    # Add text for geo mean
    ax.text(0.98, 0.02, f"Geometric mean: {geo_mean}x",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, style="italic", color=OI["blue"])

    fig.tight_layout()
    save_fig(fig, "fig_closed_loop_speedup")


# ===========================================================================
# Figure 19: LLM Groundedness
# ===========================================================================

def fig_llm_groundedness():
    """Grouped bar chart: 3 LLMs x 2 conditions (with KB vs without KB)."""
    logger.info("Generating Figure 19: LLM Groundedness...")

    llms = ["Claude\nSonnet", "GPT-4o", "Llama-3\n70B"]
    with_kb = [1.0, 0.917, 1.0]
    without_kb = [0.0, 0.0, 0.0]

    x = np.arange(len(llms))
    width = 0.32

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    bars_w = ax.bar(x - width / 2, with_kb, width,
                    color=OI["blue"], edgecolor="black", linewidth=0.4,
                    hatch="", label="With KB", zorder=3)
    bars_wo = ax.bar(x + width / 2, without_kb, width,
                     color=OI["vermilion"], edgecolor="black", linewidth=0.4,
                     hatch="//", label="Without KB", zorder=3)

    # Value labels
    for bar, val in zip(bars_w, with_kb):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.2f}" if val < 1.0 else "1.00",
                ha="center", va="bottom", fontsize=6.5, fontweight="bold",
                color=OI["blue"])
    for bar, val in zip(bars_wo, without_kb):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.03,
                "0.00", ha="center", va="bottom", fontsize=6.5,
                color=OI["vermilion"])

    ax.set_xticks(x)
    ax.set_xticklabels(llms, fontsize=7)
    ax.set_ylabel("Groundedness Score")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="center right", fontsize=6.5, framealpha=0.8)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save_fig(fig, "fig_llm_groundedness")


# ===========================================================================
# Figure 20: Single-shot Ablation
# ===========================================================================

def fig_ablation_trackb():
    """Bar chart: recommendation ablation (fair ablation results)."""
    logger.info("Generating Figure 20: Recommendation Ablation...")

    conditions = ["Full\nPipeline", "w/o ML\nClassifier", "w/o Knowledge\nGrounding",
                  "Detection\nOnly", "LLM Only\n(no ML, no KB)"]
    groundedness = [1.0, 1.0, 0.0, None, 0.0]  # None = N/A
    rec_precision = [0.70, 0.16, 0.67, None, 0.0]
    n_recs = [1.2, 4.0, 2.6, 0.0, 4.1]

    # Colors: green for full, orange for partial, red for broken, gray for N/A
    colors = [OI["green"], OI["orange"], OI["vermilion"],
              OI["gray"], OI["vermilion"]]

    fig, ax1 = plt.subplots(figsize=(3.5, 2.8))

    x = np.arange(len(conditions))
    width = 0.35

    # Hatching patterns per condition for B&W readability
    condition_hatches = ["", "//", "\\\\", "..", "xx"]

    # Plot groundedness bars (left group)
    for i, (g, c) in enumerate(zip(groundedness, colors)):
        if g is not None:
            ax1.bar(x[i] - width/2, g, width, color=c, edgecolor="black",
                    linewidth=0.4, hatch=condition_hatches[i], zorder=3,
                    label="Groundedness" if i == 0 else "")
        else:
            ax1.bar(x[i] - width/2, 0.05, width, color=c, edgecolor="black",
                    linewidth=0.4, hatch=condition_hatches[i], zorder=3, alpha=0.4)
            ax1.text(x[i] - width/2, 0.10, "N/A", ha="center", va="bottom",
                     fontsize=5.5, color=OI["gray"])

    # Plot rec precision bars (right group)
    for i, (rp, c) in enumerate(zip(rec_precision, colors)):
        if rp is not None:
            ax1.bar(x[i] + width/2, rp, width, color=OI["cyan"], edgecolor="black",
                    linewidth=0.4, hatch=condition_hatches[i], zorder=3, alpha=0.7,
                    label="Rec. Precision" if i == 0 else "")
        else:
            ax1.bar(x[i] + width/2, 0.05, width, color=OI["gray"], edgecolor="black",
                    linewidth=0.4, hatch=condition_hatches[i], zorder=3, alpha=0.4)
            ax1.text(x[i] + width/2, 0.10, "N/A", ha="center", va="bottom",
                     fontsize=5.5, color=OI["gray"])

    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=6)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.25)
    ax1.grid(axis="y", alpha=0.25)

    # Secondary axis for number of recommendations
    ax2 = ax1.twinx()
    ax2.plot(x, n_recs, "D-", color=OI["purple"], markersize=4,
             linewidth=1.0, zorder=4)
    ax2.set_ylabel("Avg. Recommendations", color=OI["purple"], fontsize=7)
    ax2.tick_params(axis="y", labelcolor=OI["purple"], labelsize=6.5)
    ax2.set_ylim(0, 5.5)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.5)

    # Legend
    bar_gnd = mpatches.Patch(facecolor=OI["green"], label="Groundedness")
    bar_rp = mpatches.Patch(facecolor=OI["cyan"], alpha=0.7, label="Rec. Precision")
    line_patch = plt.Line2D([0], [0], color=OI["purple"], marker="D",
                            markersize=3, label="Avg. recs")
    ax1.legend(handles=[bar_gnd, bar_rp, line_patch], loc="upper center",
               fontsize=5.5, framealpha=0.8, ncol=3)

    fig.tight_layout()
    save_fig(fig, "fig_ablation_trackb")


# ===========================================================================
# Figure 18: Pipeline Walkthrough (5-panel case study)
# ===========================================================================

def fig_pipeline_walkthrough():
    """Multi-panel end-to-end walkthrough of a single case study."""
    logger.info("Generating Figure 18: Pipeline Walkthrough...")

    fig = plt.figure(figsize=(7.0, 2.4))

    # 5 panels spread horizontally with arrows between them
    n_panels = 5
    panel_w = 0.155
    panel_h = 0.70
    gap = 0.025
    start_x = 0.03
    y_base = 0.18

    panel_positions = []
    for i in range(n_panels):
        px = start_x + i * (panel_w + gap)
        panel_positions.append((px, y_base, panel_w, panel_h))

    # Panel titles
    titles = ["(a) Input", "(b) ML Detection", "(c) KB Evidence",
              "(d) LLM Recommendation", "(e) Validation"]

    # --- Panel 1: Input summary table ---
    ax1 = fig.add_axes(panel_positions[0])
    ax1.axis("off")
    ax1.set_title(titles[0], fontsize=7, fontweight="bold", pad=4)

    table_data = [
        ["avg_write_sz", "64 B"],
        ["small_io_ratio", "1.00"],
        ["write_bw", "8 MB/s"],
        ["file_not_aligned", "819 K"],
        ["total_ops", "819 K"],
    ]
    table = ax1.table(cellText=table_data, colLabels=["Feature", "Value"],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(5.5)
    table.scale(1.0, 1.15)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        if r == 0:
            cell.set_facecolor(OI["blue"])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F5F5F5")

    # --- Panel 2: ML confidence bars ---
    ax2 = fig.add_axes(panel_positions[1])
    ax2.set_title(titles[1], fontsize=7, fontweight="bold", pad=4)

    dims_short = ["Gran.", "Meta.", "Para.", "Patt.", "Intf.", "File", "Thru.", "Heal."]
    confs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    bar_colors = [OI["vermilion"] if c > 0.5 else OI["gray"] for c in confs]
    y_pos = np.arange(len(dims_short))
    ax2.barh(y_pos, confs, height=0.65, color=bar_colors,
             edgecolor="white", linewidth=0.3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(dims_short, fontsize=5)
    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel("Confidence", fontsize=5.5)
    ax2.tick_params(axis="x", labelsize=5)
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.2)

    # --- Panel 3: KB Evidence (top-3 matching entries) ---
    ax3 = fig.add_axes(panel_positions[2])
    ax3.axis("off")
    ax3.set_title(titles[2], fontsize=7, fontweight="bold", pad=4)

    kb_data = [
        ["IOR-AG-001", "64B→1MB", "44.8x"],
        ["IOR-AG-003", "align xfer", "12.1x"],
        ["DLIO-AG-01", "batch sz", "3.2x"],
    ]
    kb_table = ax3.table(cellText=kb_data,
                         colLabels=["KB Entry", "Fix", "Speedup"],
                         loc="center", cellLoc="center")
    kb_table.auto_set_font_size(False)
    kb_table.set_fontsize(5.5)
    kb_table.scale(1.0, 1.15)
    for (r, c), cell in kb_table.get_celld().items():
        cell.set_linewidth(0.3)
        if r == 0:
            cell.set_facecolor(OI["green"])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F5F5F5")

    # --- Panel 4: LLM recommendation text box ---
    ax4 = fig.add_axes(panel_positions[3])
    ax4.axis("off")
    ax4.set_title(titles[3], fontsize=7, fontweight="bold", pad=4)

    rec_text = (
        "Increase transfer size\n"
        "from 64 B to 1 MB.\n\n"
        "Buffer small writes\n"
        "into large chunks.\n\n"
        "KB: ior_small_posix\n"
        "Confidence: high"
    )
    bbox = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                          boxstyle="round,pad=0.05",
                          facecolor="#F0F7FF", edgecolor=OI["blue"],
                          linewidth=0.8, transform=ax4.transAxes)
    ax4.add_patch(bbox)
    ax4.text(0.5, 0.52, rec_text, transform=ax4.transAxes,
             ha="center", va="center", fontsize=5, family="monospace",
             linespacing=1.3)

    # --- Panel 5: Before/after BW bars ---
    ax5 = fig.add_axes(panel_positions[4])
    ax5.set_title(titles[4], fontsize=7, fontweight="bold", pad=4)

    bw_labels = ["Before", "After"]
    bw_vals = [49.24, 2208.43]
    bar_cols = [OI["vermilion"], OI["green"]]
    ax5.bar([0, 1], bw_vals, width=0.6, color=bar_cols,
            edgecolor="black", linewidth=0.4, hatch=["//", ""])
    ax5.set_xticks([0, 1])
    ax5.set_xticklabels(bw_labels, fontsize=5.5)
    ax5.set_ylabel("MiB/s", fontsize=5.5)
    ax5.set_yscale("log")
    ax5.set_ylim(10, 5000)
    ax5.tick_params(axis="y", labelsize=5)
    ax5.grid(axis="y", alpha=0.2, which="both")

    # Speedup annotation
    ax5.annotate("44.8x", xy=(0.5, 2208.43), ha="center", va="bottom",
                 fontsize=7, fontweight="bold", color=OI["blue"])

    # --- Draw arrows between panels ---
    for i in range(n_panels - 1):
        x_start = panel_positions[i][0] + panel_positions[i][2] + 0.003
        x_end = panel_positions[i + 1][0] - 0.003
        x_mid = (x_start + x_end) / 2
        y_mid = y_base + panel_h / 2

        fig.patches.append(FancyArrowPatch(
            (x_start, y_mid), (x_end, y_mid),
            arrowstyle="->,head_width=3,head_length=3",
            color=OI["blue"], linewidth=0.8,
            transform=fig.transFigure, figure=fig,
        ))

    save_fig(fig, "fig_pipeline_walkthrough")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    apply_style()
    fig_closed_loop_speedup()
    fig_llm_groundedness()
    fig_ablation_trackb()
    fig_pipeline_walkthrough()
    logger.info("All 4 evaluation figures saved to %s", FIG_DIR)
