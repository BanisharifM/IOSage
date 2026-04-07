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
    # IEEE guideline: ~9-10pt for readability in two-column format
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 9,
    "mathtext.fontset": "stix",

    # Axes
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,

    # Ticks — increased from 7 to 8 to prevent y-axis overlap
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",

    # Legend
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.handlelength": 1.5,

    # Grid
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,

    # Figure — increased pad_inches to prevent y-axis clipping
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,

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
    width = 0.28

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

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

    # Speedup annotations — use black for readability
    for i, sp in enumerate(speedups):
        y_top = max(before[i], after[i])
        ax.annotate(
            f"{sp}\u00d7",
            xy=(x[i], y_top * 1.3),
            ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Write Bandwidth (MiB/s)", labelpad=8)
    # Legend between title and bars
    ax.legend(loc="upper center", fontsize=7, framealpha=0.95,
              edgecolor="#cccccc", ncol=2, borderpad=0.3,
              bbox_to_anchor=(0.5, 1.08))
    ax.grid(axis="y", alpha=0.25, which="both")

    # Geometric mean as title
    ax.set_title(f"Geometric mean: {geo_mean}\u00d7", fontsize=9,
                 style="italic", pad=8)
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
    """Bar chart: recommendation ablation — matches TABLE V (4 conditions).

    Updated to match paper after SHAP removal (Entry 77) and
    'Detection only' row removal.
    """
    logger.info("Generating Fig 6: Recommendation Ablation...")

    # Must match TABLE V exactly
    conditions = ["Full\nPipeline", "w/o ML\nClassifier", "w/o Knowledge\nGrounding",
                  "w/o ML\nClassifier\nand KB"]
    groundedness = [1.00, 1.00, 0.00, 0.00]
    rec_precision = [0.73, 0.16, 0.67, 0.00]
    n_recs = [1.4, 4.0, 2.6, 4.1]

    fig, ax1 = plt.subplots(figsize=(3.5, 2.6))

    x = np.arange(len(conditions))
    width = 0.32

    # Groundedness bars (teal/green)
    bars_g = ax1.bar(x - width/2, groundedness, width,
                     color=OI["green"], edgecolor="black", linewidth=0.4,
                     label="Groundedness", zorder=3)
    # Rec. Precision bars (vermilion/red)
    bars_r = ax1.bar(x + width/2, rec_precision, width,
                     color=OI["vermilion"], edgecolor="black", linewidth=0.4,
                     hatch="//", label="Rec. Precision", zorder=3)

    # Value labels on bars
    for bar in bars_g:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=6,
                 color="#006644", fontweight="bold")
    for bar in bars_r:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=6,
                 color=OI["vermilion"], fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=6.5)
    ax1.set_ylabel("Score", fontsize=8)
    ax1.set_ylim(0, 1.2)
    ax1.grid(axis="y", alpha=0.25)

    # Secondary axis for number of recommendations
    ax2 = ax1.twinx()
    ax2.plot(x, n_recs, "D-", color="#333333", markersize=5,
             linewidth=1.2, zorder=4, label="# Recs")
    for i, nr in enumerate(n_recs):
        ax2.text(x[i], nr + 0.15, f"{nr:.1f}", ha="center", va="bottom",
                 fontsize=6, color="#333333", fontweight="bold")
    ax2.set_ylabel("# Recommendations", color="#333333", fontsize=7)
    ax2.tick_params(axis="y", labelcolor="#333333", labelsize=6.5)
    ax2.set_ylim(0, 5.5)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.5)

    # Legend inside chart
    bar_gnd = mpatches.Patch(facecolor=OI["green"], label="Groundedness")
    bar_rp = mpatches.Patch(facecolor=OI["vermilion"], hatch="//",
                            label="Rec. Precision")
    line_patch = plt.Line2D([0], [0], color="#333333", marker="D",
                            markersize=4, label="# Recs")
    ax1.legend(handles=[bar_gnd, bar_rp, line_patch], loc="upper right",
               bbox_to_anchor=(0.99, 0.99),
               fontsize=6.5, framealpha=0.95, edgecolor="#cccccc",
               borderpad=0.4)

    # Save as fig_ablation_pipeline (what paper references)
    save_fig(fig, "fig_ablation_pipeline")


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
# Fig A: IOAgent overgeneration — F1 drops with stronger LLMs
# ===========================================================================

def fig_ioagent_overgeneration():
    """IOAgent detection F1 vs IOSage ML detection across 4 LLMs."""
    logger.info("Generating: IOAgent overgeneration comparison...")

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    models = ['GPT-4.1-\nmini', 'GPT-4o', 'Claude\nSonnet', 'Llama-3.1\n70B']
    ioagent_f1 = [0.465, 0.319, 0.305, 0.262]
    x = np.arange(len(models))

    # IOSage constant line
    ax.axhline(y=0.929, color=COLORS["green"], linewidth=2, linestyle='-',
               label='IOSage (ML detection)', zorder=3)
    ax.fill_between([-0.5, len(models) - 0.5], 0.926, 0.932,
                    color=COLORS["green"], alpha=0.1, zorder=1)
    ax.text(len(models) - 0.55, 0.942, 'IOSage: 0.929',
            fontsize=7.5, color=COLORS["green"], fontweight='bold', ha='right')

    # IOAgent bars
    bars = ax.bar(x, ioagent_f1, width=0.55, color=COLORS["vermilion"],
                  edgecolor='white', linewidth=0.5,
                  label='IOAgent (LLM-only)', zorder=2)

    for bar, val in zip(bars, ioagent_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5,
                color=COLORS["vermilion"])

    ax.set_ylabel('Micro-F1')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.legend(loc='center right', frameon=True, framealpha=0.95,
              edgecolor='#cccccc')

    save_fig(fig, "fig_ioagent_overgeneration")


# ===========================================================================
# Fig B: Iterative speedup comparison across 4 LLMs
# ===========================================================================

def fig_iterative_speedup_comparison():
    """Grouped bar chart: iterative speedups across 17 workloads x 4 LLMs."""
    logger.info("Generating: Iterative speedup comparison...")

    fig, ax = plt.subplots(figsize=(7.16, 2.2))

    workloads = [
        'Small\nPOSIX', 'Small\nO_DIR', 'fsync\nheavy', 'Random\naccess',
        'Mis-\naligned',
        'Meta.\nstorm',
        'Small\nrecs', 'Many\nsmall', 'Shuffle\nheavy',
        'Small\naccess', 'Indep\nvs coll',
        'Shared\nfile',
    ]

    claude = [2241, 11.0, 13.0, 15.5, 158.7, 146.6, 1.0, 9.3, 1.2,
              9.6, 8.8, 1.5]
    gpt4o = [4.6, 11.3, 12.2, 12.3, 2.1, 3.5, 1.0, 1.0, 1.0,
             43.0, 3.8, 1.6]
    llama = [4.8, 12.2, 20.0, 1.0, 30.2, 1.3, 1.1, 1.3, 1.0,
             43.1, 1.7, 1.1]
    mini = [3.1, 3.6, 1.5, 1.0, 15.0, 15.6, 1.0, 1.1, 3.1,
            53.9, 3.9, 1.3]

    x = np.arange(len(workloads))
    width = 0.19
    offsets = [-1.5, -0.5, 0.5, 1.5]
    colors = [COLORS["blue"], COLORS["vermilion"], COLORS["green"],
              COLORS["orange"]]
    labels = ['Claude', 'GPT-4o', 'Llama-70B', 'GPT-4.1-mini']
    data = [claude, gpt4o, llama, mini]

    for i, (vals, color, label) in enumerate(zip(data, colors, labels)):
        ax.bar(x + offsets[i] * width, vals, width, label=label,
               color=color, edgecolor='white', linewidth=0.3)

    ax.set_yscale('log')
    ax.set_ylabel('Speedup (log scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, fontsize=6.5)
    ax.axhline(y=1.0, color='black', linewidth=0.5, linestyle=':', alpha=0.4)

    # Legend inside plot, below suite labels
    ax.legend(loc='upper center', ncol=4, frameon=True, framealpha=0.95,
              edgecolor='#cccccc', columnspacing=0.8, fontsize=7,
              bbox_to_anchor=(0.5, 0.85))
    ax.set_ylim(0.8, 5000)

    # Suite separators
    for pos in [4.5, 5.5, 8.5, 10.5]:
        ax.axvline(x=pos, color='gray', linewidth=0.4, linestyle='--',
                   alpha=0.4)

    # Suite labels above legend
    y_label = 3500
    ax.text(2, y_label, 'IOR', fontsize=7, ha='center', color='#666666',
            fontstyle='italic')
    ax.text(5, y_label, 'mdtest', fontsize=6.5, ha='center', color='#666666',
            fontstyle='italic')
    ax.text(7, y_label, 'DLIO', fontsize=7, ha='center', color='#666666',
            fontstyle='italic')
    ax.text(9.5, y_label, 'h5bench', fontsize=6.5, ha='center',
            color='#666666', fontstyle='italic')
    ax.text(11, y_label, 'HACC', fontsize=6.5, ha='center', color='#666666',
            fontstyle='italic')

    save_fig(fig, "fig_iterative_speedup_comparison")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    apply_style()
    fig_closed_loop_speedup()
    fig_llm_groundedness()
    fig_ablation_trackb()
    fig_pipeline_walkthrough()
    fig_ioagent_overgeneration()
    fig_iterative_speedup_comparison()
    logger.info("All 6 evaluation figures saved to %s", FIG_DIR)
