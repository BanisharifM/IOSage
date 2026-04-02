#!/usr/bin/env python3
"""Generate the IOSage system architecture figure (Figure 1).

Double-column width, shows complete pipeline with both operating modes.
Follows figure_style_guide.md conventions.

Usage:
    python scripts/generate_architecture_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_DIR / "paper" / "figures"

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 8,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Colors (Okabe-Ito subset, slightly muted for boxes)
C_INPUT = "#56B4E9"      # cyan - input/data
C_ML = "#0072B2"         # blue - ML components
C_LLM = "#D55E00"        # vermilion - LLM components
C_KB = "#009E73"          # green - knowledge base
C_OUTPUT = "#E69F00"      # orange - outputs
C_LOOP = "#CC79A7"        # purple - feedback loop
C_OFFLINE = "#F0F0F0"     # light gray - offline phase bg
C_ONLINE = "#FAFAFA"      # white-ish - online phase bg
C_TEXT = "#000000"


def draw_box(ax, x, y, w, h, label, sublabel="", color="#0072B2", alpha=0.15,
             fontsize=8, fontweight="bold", sublabel_size=6.5):
    """Draw a rounded box with label and optional sublabel."""
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02",
                          facecolor=color, alpha=alpha,
                          edgecolor=color, linewidth=1.0)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w/2, y + h/2 + 0.015, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, color=C_TEXT)
        ax.text(x + w/2, y + h/2 - 0.02, sublabel,
                ha="center", va="center", fontsize=sublabel_size,
                color="#444444", style="italic")
    else:
        ax.text(x + w/2, y + h/2, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, color=C_TEXT)


def draw_arrow(ax, x1, y1, x2, y2, color="#333333", style="-|>", lw=0.8):
    """Draw an arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def main():
    fig, ax = plt.subplots(figsize=(7.16, 3.2), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # === Phase backgrounds ===
    # Offline (top strip)
    offline_bg = FancyBboxPatch((0.01, 0.72), 0.98, 0.26,
                                 boxstyle="round,pad=0.01",
                                 facecolor=C_OFFLINE, edgecolor="#CCCCCC",
                                 linewidth=0.5, linestyle="--")
    ax.add_patch(offline_bg)
    ax.text(0.03, 0.95, "OFFLINE (Training)", fontsize=7, color="#666666",
            fontweight="bold", style="italic")

    # Online (bottom area)
    online_bg = FancyBboxPatch((0.01, 0.02), 0.98, 0.68,
                                boxstyle="round,pad=0.01",
                                facecolor=C_ONLINE, edgecolor="#CCCCCC",
                                linewidth=0.5, linestyle="--")
    ax.add_patch(online_bg)
    ax.text(0.03, 0.67, "ONLINE (Inference)", fontsize=7, color="#666666",
            fontweight="bold", style="italic")

    # === OFFLINE PHASE (top) ===
    bw = 0.18  # box width
    bh = 0.12  # box height

    # Training data
    draw_box(ax, 0.03, 0.78, 0.20, 0.12, "Training Data",
             "91K heuristic + 187 GT\n(biquality, w=100)", C_INPUT, alpha=0.2)

    # Benchmark sweep
    draw_box(ax, 0.28, 0.78, 0.20, 0.12, "Benchmark Sweep",
             "6 suites, 623 configs\nIOR/mdtest/DLIO/h5bench/\nHACC-IO/custom", C_KB, alpha=0.2)

    # KB construction
    draw_box(ax, 0.53, 0.78, 0.20, 0.12, "Knowledge Base",
             "623 entries: signature\n+ fix + source code", C_KB, alpha=0.25)

    # ML Training
    draw_box(ax, 0.78, 0.78, 0.18, 0.12, "ML Training",
             "XGBoost, 8 dims\n5-seed, F1=0.923", C_ML, alpha=0.2)

    # Arrows: offline
    draw_arrow(ax, 0.23, 0.84, 0.28, 0.84)
    draw_arrow(ax, 0.48, 0.84, 0.53, 0.84)
    draw_arrow(ax, 0.23, 0.82, 0.78, 0.82, color="#999999", lw=0.5)

    # === ONLINE PHASE (bottom) ===

    # Input
    draw_box(ax, 0.03, 0.48, 0.12, 0.13, "Darshan\nLog", "", C_INPUT, alpha=0.25,
             fontsize=8)

    # Step 1: ML Detection
    draw_box(ax, 0.19, 0.48, 0.14, 0.13, "ML Detect",
             "8-dim classifier\nthreshold=0.3", C_ML, alpha=0.2)

    # Step 2: SHAP
    draw_box(ax, 0.37, 0.48, 0.12, 0.13, "SHAP",
             "per-label top-K\nfeature attribution", C_ML, alpha=0.15)

    # Step 3: KB Retrieval
    draw_box(ax, 0.53, 0.48, 0.14, 0.13, "KB Retrieve",
             "similarity search\n623 entries", C_KB, alpha=0.2)

    # Step 4: LLM
    draw_box(ax, 0.71, 0.48, 0.14, 0.13, "LLM Generate",
             "Claude/GPT-4o/Llama\ncode-level fix", C_LLM, alpha=0.2)

    # Arrows: online pipeline
    draw_arrow(ax, 0.15, 0.545, 0.19, 0.545)
    draw_arrow(ax, 0.33, 0.545, 0.37, 0.545)
    draw_arrow(ax, 0.49, 0.545, 0.53, 0.545)
    draw_arrow(ax, 0.67, 0.545, 0.71, 0.545)

    # KB arrow from offline
    draw_arrow(ax, 0.63, 0.78, 0.63, 0.61, color=C_KB, lw=0.7)
    # ML model arrow from offline
    draw_arrow(ax, 0.87, 0.78, 0.26, 0.61, color=C_ML, lw=0.5, style="-|>")

    # === OUTPUT: Two modes ===

    # Single-shot output (right)
    draw_box(ax, 0.88, 0.48, 0.10, 0.13, "Code Fix",
             "grounded rec.\n+ KB citations", C_OUTPUT, alpha=0.25)
    draw_arrow(ax, 0.85, 0.545, 0.88, 0.545)

    # Label: "Single-shot mode"
    ax.text(0.93, 0.44, "Single-shot", fontsize=6, ha="center",
            color=C_OUTPUT, fontweight="bold")

    # === ITERATIVE LOOP (bottom) ===
    loop_y = 0.15

    # Execute on HPC
    draw_box(ax, 0.30, loop_y, 0.16, 0.12, "Execute Fix",
             "SLURM submit\nbenchmark rerun", C_LOOP, alpha=0.15)

    # New Darshan
    draw_box(ax, 0.52, loop_y, 0.14, 0.12, "New Darshan",
             "collect I/O profile\nmeasure speedup", C_LOOP, alpha=0.15)

    # ML Re-detect
    draw_box(ax, 0.72, loop_y, 0.14, 0.12, "ML Re-detect",
             "converged?\nbottleneck < 0.3", C_LOOP, alpha=0.15)

    # Arrows: iterative loop
    # LLM → Execute
    draw_arrow(ax, 0.78, 0.48, 0.38, 0.27, color=C_LOOP, lw=0.9)
    # Execute → New Darshan
    draw_arrow(ax, 0.46, 0.21, 0.52, 0.21, color=C_LOOP)
    # New Darshan → ML Re-detect
    draw_arrow(ax, 0.66, 0.21, 0.72, 0.21, color=C_LOOP)
    # ML Re-detect → back to SHAP (iterate)
    draw_arrow(ax, 0.79, 0.27, 0.43, 0.48, color=C_LOOP, lw=0.9,
               style="-|>")

    # Iterate label on loop arrow
    ax.text(0.60, 0.38, "iterate", fontsize=6, color=C_LOOP,
            fontweight="bold", rotation=40, ha="center")

    # Converge output
    draw_box(ax, 0.88, loop_y, 0.10, 0.12, "Validated\nSpeedup",
             "measured BW\nimprovement", C_OUTPUT, alpha=0.25)
    draw_arrow(ax, 0.86, 0.21, 0.88, 0.21, color=C_LOOP)

    # Label: "Iterative mode"
    ax.text(0.93, 0.11, "Iterative", fontsize=6, ha="center",
            color=C_LOOP, fontweight="bold")

    # === Step numbers ===
    for i, (x, y) in enumerate([(0.19, 0.62), (0.37, 0.62), (0.53, 0.62), (0.71, 0.62)], 1):
        ax.text(x + 0.01, y, f"Step {i}", fontsize=5.5, color="#888888")

    # === Title ===
    ax.text(0.50, 0.995, "IOSage: ML-Guided I/O Bottleneck Detection and Code-Level Recommendation",
            ha="center", va="top", fontsize=9, fontweight="bold")

    # Save
    pdf_path = FIG_DIR / "fig_architecture.pdf"
    png_path = FIG_DIR / "fig_architecture.png"
    fig.savefig(pdf_path, format="pdf", dpi=300)
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path} ({pdf_path.stat().st_size} bytes)")
    print(f"Saved: {png_path} ({png_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
