#!/usr/bin/env python3
"""Generate SC-quality system architecture figure.

Design principles (from STELLAR SC'25, RCACopilot EuroSys'24):
- Fewer, larger boxes with space for icons and arrows
- Numbered stages with circled numbers
- Explicit phase labels with shaded backgrounds
- Feedback loop shown with dashed curved arrow
- Mini-visualizations inside key boxes
- Merged redundant boxes (Feature Ext + Training, Benchmark + KB)
- No in-figure title (LaTeX caption)

Usage:
    python scripts/generate_architecture_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, Arc
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_DIR / "paper" / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 8,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Professional muted palette
BG = {
    "data":  "#FFF3E0", "bd_data":  "#E65100",
    "ml":    "#E3F2FD", "bd_ml":    "#1565C0",
    "kb":    "#E8F5E9", "bd_kb":    "#2E7D32",
    "llm":   "#FFF8E1", "bd_llm":   "#E65100",
    "out":   "#E0F2F1", "bd_out":   "#00695C",
    "loop":  "#F3E5F5", "bd_loop":  "#6A1B9A",
}
COL_ARROW = "#37474F"
COL_TEXT = "#212121"
COL_SUB = "#616161"
COL_LOOP = "#7B1FA2"


def rbox(ax, x, y, w, h, bg, bd, lw=1.0):
    """Rounded rectangle."""
    p = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.015",
        facecolor=bg, edgecolor=bd, linewidth=lw, zorder=2)
    ax.add_patch(p)
    return p


def phase_bg(ax, x, y, w, h, label, bg="#F5F5F5", bd="#BDBDBD"):
    """Phase background with header label."""
    p = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.01",
        facecolor=bg, edgecolor=bd, linewidth=0.7, zorder=0)
    ax.add_patch(p)
    # Header bar
    hh = 0.035
    hdr = FancyBboxPatch((x, y + h - hh), w, hh,
        boxstyle="round,pad=0.003,rounding_size=0.008",
        facecolor=bd, edgecolor=bd, linewidth=0, zorder=1, alpha=0.15)
    ax.add_patch(hdr)
    ax.text(x + w/2, y + h - hh/2, label,
            fontsize=7.5, fontweight="bold", ha="center", va="center",
            color="#424242", zorder=3)


def arr(ax, x1, y1, x2, y2, color=COL_ARROW, lw=1.2, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw), zorder=5)


def darr(ax, x1, y1, x2, y2, color=COL_ARROW, lw=0.8):
    """Dashed arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        linestyle="dashed"), zorder=5)


def carr(ax, x1, y1, x2, y2, color=COL_ARROW, lw=1.0, rad=0.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        connectionstyle=f"arc3,rad={rad}"), zorder=5)


def stage_num(ax, x, y, num, color="#E65100"):
    """Circled stage number."""
    c = Circle((x, y), 0.018, facecolor=color, edgecolor="white",
               linewidth=0.8, zorder=7)
    ax.add_patch(c)
    ax.text(x, y, str(num), fontsize=6.5, fontweight="bold",
            ha="center", va="center", color="white", zorder=8)


def draw_cylinder(ax, cx, cy, w, h, color="#FFCCBC", ec="#E65100"):
    """Database cylinder icon."""
    body = plt.Rectangle((cx - w/2, cy - h/2), w, h,
        facecolor=color, edgecolor=ec, linewidth=0.6, zorder=3)
    ax.add_patch(body)
    top = Ellipse((cx, cy + h/2), w, h * 0.4,
        facecolor=color, edgecolor=ec, linewidth=0.6, zorder=4)
    ax.add_patch(top)
    bot = Arc((cx, cy - h/2), w, h * 0.4, theta1=180, theta2=360,
        edgecolor=ec, linewidth=0.6, zorder=4)
    ax.add_patch(bot)


def draw_tree_icon(ax, cx, cy, s=0.03):
    """Small decision tree icon."""
    nodes = [(cx, cy+s), (cx-s*0.7, cy), (cx+s*0.7, cy),
             (cx-s, cy-s), (cx-s*0.4, cy-s), (cx+s*0.4, cy-s), (cx+s, cy-s)]
    edges = [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]
    for i, j in edges:
        ax.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]],
                color="#1565C0", lw=0.6, zorder=4)
    for i, (nx, ny) in enumerate(nodes):
        ms = 3.5 if i == 0 else 2.5
        c = "#1565C0" if i < 3 else "#90CAF9"
        ax.plot(nx, ny, "o", color=c, markersize=ms, zorder=5,
                markeredgecolor="#1565C0", markeredgewidth=0.3)


def draw_bars_icon(ax, cx, cy, s=0.03):
    """Small horizontal bar chart icon (SHAP-like)."""
    widths = [1.0, 0.7, 0.5, 0.35, 0.2]
    colors = ["#E65100", "#FF6D00", "#FF9100", "#FFB74D", "#FFE0B2"]
    bh = s * 0.28
    for i, (w, c) in enumerate(zip(widths, colors)):
        by = cy + s*0.6 - i * (bh + s*0.12)
        ax.barh(by, w * s * 1.5, height=bh, left=cx - s*0.8,
                color=c, edgecolor="#E65100", linewidth=0.3, zorder=4)


def draw_doc_icon(ax, cx, cy, s=0.025, color="#C8E6C9", ec="#2E7D32"):
    """Small document/page icon."""
    for i in range(3):
        dx = cx - s*0.5 + i*s*0.08
        dy = cy - s*0.5 + i*s*0.35
        rect = plt.Rectangle((dx, dy), s*1.2, s*1.0,
            facecolor=color if i == 2 else "#E8F5E9",
            edgecolor=ec, linewidth=0.4, zorder=3+i)
        ax.add_patch(rect)
        for j in range(2):
            lx = dx + s*0.15
            ly = dy + s*0.2 + j * s*0.35
            ax.plot([lx, lx + s*0.7], [ly, ly],
                    color=ec, lw=0.3, alpha=0.5, zorder=4+i)


def draw_code_icon(ax, cx, cy, s=0.025):
    """Code bracket icon < / >."""
    ax.text(cx, cy, "</  >", fontsize=8, ha="center", va="center",
            fontfamily="monospace", color="#E65100", fontweight="bold", zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(7.16, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # =====================================================================
    # PHASE 1: TRAINING & KB CONSTRUCTION (top)
    # =====================================================================
    phase_bg(ax, 0.01, 0.68, 0.98, 0.31,
             "Training & Knowledge Base Construction")

    # Box A: Darshan Dataset
    rbox(ax, 0.03, 0.72, 0.18, 0.22, BG["data"], BG["bd_data"])
    ax.text(0.12, 0.915, "Darshan Dataset", fontsize=8.5, fontweight="bold",
            ha="center", color=COL_TEXT, zorder=6)
    draw_cylinder(ax, 0.12, 0.80, 0.06, 0.045)
    ax.text(0.12, 0.735, "1.37M logs\nALCF Polaris\n22 months",
            fontsize=5.5, ha="center", color=COL_SUB, linespacing=1.3, zorder=6)

    # Box B: ML Classifier Training (merged: feat ext + biquality)
    rbox(ax, 0.26, 0.72, 0.26, 0.22, BG["ml"], BG["bd_ml"])
    ax.text(0.39, 0.915, "ML Classifier Training", fontsize=8.5,
            fontweight="bold", ha="center", color=COL_TEXT, zorder=6)
    draw_tree_icon(ax, 0.34, 0.81, s=0.032)
    ax.text(0.45, 0.82, r"$\mathbf{x} \in \mathbb{R}^{157}$" + "\n"
            "XGBoost, biquality\n91K heuristic + 187 GT",
            fontsize=5.5, ha="center", va="center", color=COL_SUB,
            linespacing=1.3, zorder=6)
    ax.text(0.39, 0.735, "8 dimensions, 5-seed, F1 = 0.923",
            fontsize=5.5, ha="center", color="#1565C0",
            fontstyle="italic", zorder=6)

    # Box C: Benchmark Knowledge Base (merged: sweep + KB)
    rbox(ax, 0.57, 0.72, 0.26, 0.22, BG["kb"], BG["bd_kb"])
    ax.text(0.70, 0.915, "Benchmark Knowledge Base", fontsize=8.5,
            fontweight="bold", ha="center", color=COL_TEXT, zorder=6)
    draw_doc_icon(ax, 0.64, 0.82, s=0.028)
    ax.text(0.76, 0.82, "6 suites, 623 configs\nDarshan signature\n+ fix + source code",
            fontsize=5.5, ha="center", va="center", color=COL_SUB,
            linespacing=1.3, zorder=6)
    ax.text(0.70, 0.735, "IOR / mdtest / DLIO / h5bench / HACC-IO / custom",
            fontsize=4.5, ha="center", color="#2E7D32",
            fontstyle="italic", zorder=6)

    # Box D: Trained Model output
    rbox(ax, 0.87, 0.735, 0.11, 0.13, BG["out"], BG["bd_out"])
    ax.text(0.925, 0.845, "Trained\nModel", fontsize=7.5,
            fontweight="bold", ha="center", va="center", color=COL_TEXT, zorder=6)
    ax.text(0.925, 0.755, "F1=0.923", fontsize=6, ha="center",
            color="#00695C", fontweight="bold", zorder=6)

    # Phase 1 arrows
    arr(ax, 0.21, 0.83, 0.26, 0.83)
    arr(ax, 0.52, 0.83, 0.57, 0.83)
    arr(ax, 0.83, 0.80, 0.87, 0.80)

    # =====================================================================
    # PHASE 2: INFERENCE PIPELINE (middle)
    # =====================================================================
    phase_bg(ax, 0.01, 0.32, 0.98, 0.33,
             "Inference Pipeline")

    bh2 = 0.22
    r2y = 0.36

    # Input: New Darshan Log
    rbox(ax, 0.03, r2y, 0.10, bh2, BG["data"], BG["bd_data"])
    ax.text(0.08, r2y + bh2 - 0.025, "Darshan\nLog", fontsize=8,
            fontweight="bold", ha="center", va="top", color=COL_TEXT, zorder=6)
    draw_cylinder(ax, 0.08, r2y + 0.07, 0.04, 0.03)

    # Step 1: ML Detect
    stage_num(ax, 0.175, r2y + bh2 + 0.015, 1, "#1565C0")
    rbox(ax, 0.16, r2y, 0.15, bh2, BG["ml"], BG["bd_ml"])
    ax.text(0.235, r2y + bh2 - 0.025, "ML Detect", fontsize=8,
            fontweight="bold", ha="center", va="top", color=COL_TEXT, zorder=6)
    draw_tree_icon(ax, 0.235, r2y + 0.11, s=0.025)
    ax.text(0.235, r2y + 0.02, "8-dim, 3.9 ms",
            fontsize=5.5, ha="center", color=COL_SUB, zorder=6)

    # Step 2: SHAP Attribution
    stage_num(ax, 0.355, r2y + bh2 + 0.015, 2, "#E65100")
    rbox(ax, 0.34, r2y, 0.14, bh2, BG["llm"], BG["bd_llm"])
    ax.text(0.41, r2y + bh2 - 0.025, "SHAP", fontsize=8,
            fontweight="bold", ha="center", va="top", color=COL_TEXT, zorder=6)
    draw_bars_icon(ax, 0.41, r2y + 0.11, s=0.025)
    ax.text(0.41, r2y + 0.02, "per-label top-K",
            fontsize=5.5, ha="center", color=COL_SUB, zorder=6)

    # Step 3: KB Retrieval
    stage_num(ax, 0.525, r2y + bh2 + 0.015, 3, "#2E7D32")
    rbox(ax, 0.51, r2y, 0.14, bh2, BG["kb"], BG["bd_kb"])
    ax.text(0.58, r2y + bh2 - 0.025, "KB Retrieve", fontsize=8,
            fontweight="bold", ha="center", va="top", color=COL_TEXT, zorder=6)
    draw_doc_icon(ax, 0.58, r2y + 0.11, s=0.022)
    ax.text(0.58, r2y + 0.02, "623 entries",
            fontsize=5.5, ha="center", color=COL_SUB, zorder=6)

    # Step 4: LLM Recommendation
    stage_num(ax, 0.695, r2y + bh2 + 0.015, 4, "#E65100")
    rbox(ax, 0.68, r2y, 0.15, bh2, BG["llm"], BG["bd_llm"])
    ax.text(0.755, r2y + bh2 - 0.025, "LLM Generate", fontsize=8,
            fontweight="bold", ha="center", va="top", color=COL_TEXT, zorder=6)
    draw_code_icon(ax, 0.755, r2y + 0.11)
    ax.text(0.755, r2y + 0.02, "Claude / GPT-4o / Llama",
            fontsize=4.5, ha="center", color=COL_SUB, zorder=6)

    # Output: Code Fix
    rbox(ax, 0.86, r2y, 0.13, bh2, BG["out"], BG["bd_out"])
    ax.text(0.925, r2y + bh2 - 0.025, "Code-Level\nFix", fontsize=8,
            fontweight="bold", ha="center", va="top", color=COL_TEXT, zorder=6)
    ax.text(0.925, r2y + 0.08, "grounded\nrecommendation",
            fontsize=5.5, ha="center", va="center", color=COL_SUB,
            linespacing=1.3, zorder=6)
    ax.text(0.925, r2y + 0.02, "single-shot mode",
            fontsize=5, ha="center", color="#00695C",
            fontweight="bold", fontstyle="italic", zorder=6)

    # Phase 2 arrows (with good spacing)
    arr(ax, 0.13, r2y + bh2/2, 0.16, r2y + bh2/2)
    arr(ax, 0.31, r2y + bh2/2, 0.34, r2y + bh2/2)
    arr(ax, 0.48, r2y + bh2/2, 0.51, r2y + bh2/2)
    arr(ax, 0.65, r2y + bh2/2, 0.68, r2y + bh2/2)
    arr(ax, 0.83, r2y + bh2/2, 0.86, r2y + bh2/2)

    # Vertical arrows: trained model → ML Detect, KB → KB Retrieve
    darr(ax, 0.925, 0.735, 0.235, r2y + bh2 + 0.005,
         color="#1565C0", lw=0.7)
    darr(ax, 0.70, 0.72, 0.58, r2y + bh2 + 0.005,
         color="#2E7D32", lw=0.7)

    # =====================================================================
    # PHASE 3: CLOSED-LOOP VALIDATION (bottom)
    # =====================================================================
    phase_bg(ax, 0.10, 0.01, 0.89, 0.28,
             "Closed-Loop Validation")

    bh3 = 0.17
    r3y = 0.05

    # Execute on HPC
    rbox(ax, 0.13, r3y, 0.17, bh3, BG["loop"], BG["bd_loop"])
    ax.text(0.215, r3y + bh3 - 0.02, "Execute on HPC", fontsize=7.5,
            fontweight="bold", ha="center", va="top", color=COL_TEXT, zorder=6)
    ax.text(0.215, r3y + 0.04, "SLURM submit\nbenchmark rerun",
            fontsize=5.5, ha="center", va="center", color=COL_SUB, zorder=6)

    # Collect New Profile
    rbox(ax, 0.35, r3y, 0.17, bh3, BG["loop"], BG["bd_loop"])
    ax.text(0.435, r3y + bh3 - 0.02, "Collect Profile", fontsize=7.5,
            fontweight="bold", ha="center", va="top", color=COL_TEXT, zorder=6)
    ax.text(0.435, r3y + 0.04, "new Darshan log\nmeasure throughput",
            fontsize=5.5, ha="center", va="center", color=COL_SUB, zorder=6)

    # ML Re-detect
    rbox(ax, 0.57, r3y, 0.17, bh3, BG["loop"], BG["bd_loop"])
    ax.text(0.655, r3y + bh3 - 0.02, "ML Re-detect", fontsize=7.5,
            fontweight="bold", ha="center", va="top", color=COL_TEXT, zorder=6)
    ax.text(0.655, r3y + 0.04, "converged?\nall dims < 0.3",
            fontsize=5.5, ha="center", va="center", color=COL_SUB, zorder=6)

    # Validated Result
    rbox(ax, 0.79, r3y, 0.17, bh3, BG["out"], BG["bd_out"])
    ax.text(0.875, r3y + bh3 - 0.02, "Validated Result", fontsize=7.5,
            fontweight="bold", ha="center", va="top", color=COL_TEXT, zorder=6)
    ax.text(0.875, r3y + 0.055, "33/33 runs\n6 benchmarks",
            fontsize=5.5, ha="center", va="center", color=COL_SUB, zorder=6)
    ax.text(0.875, r3y + 0.015, "iterative mode",
            fontsize=5, ha="center", color=COL_LOOP,
            fontweight="bold", fontstyle="italic", zorder=6)

    # Phase 3 arrows
    arr(ax, 0.30, r3y + bh3/2, 0.35, r3y + bh3/2, color=COL_LOOP)
    arr(ax, 0.52, r3y + bh3/2, 0.57, r3y + bh3/2, color=COL_LOOP)
    arr(ax, 0.74, r3y + bh3/2, 0.79, r3y + bh3/2, color=COL_LOOP)

    # LLM → Execute (down from inference to loop)
    carr(ax, 0.72, r2y, 0.25, r3y + bh3,
         color=COL_LOOP, lw=1.2, rad=0.15)

    # ML Re-detect → back to SHAP (iterate loop — dashed, going up)
    carr(ax, 0.655, r3y + bh3, 0.41, r2y,
         color=COL_LOOP, lw=1.2, rad=0.35)

    # "iterate" label on loop arrow with white background
    ax.text(0.50, 0.295, "iterate", fontsize=7.5, color=COL_LOOP,
            fontweight="bold", ha="center", rotation=50, zorder=8,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # =====================================================================
    # Save
    # =====================================================================
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    for fmt in ["pdf", "png"]:
        path = FIG_DIR / f"fig_architecture.{fmt}"
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {path} ({path.stat().st_size:,} bytes)")
    plt.close(fig)


if __name__ == "__main__":
    main()
