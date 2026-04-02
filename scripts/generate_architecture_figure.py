#!/usr/bin/env python3
"""Generate publication-quality system architecture figure.

Clean design with precise positioning — no overlaps.
Three rows: Offline → Online → Iterative.

Usage:
    python scripts/generate_architecture_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
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

C = {
    "bg_data": "#FFF3E0", "bd_data": "#FF8A65",
    "bg_ml":   "#E3F2FD", "bd_ml":   "#64B5F6",
    "bg_kb":   "#E8F5E9", "bd_kb":   "#81C784",
    "bg_llm":  "#FFF8E1", "bd_llm":  "#FFB74D",
    "bg_out":  "#E0F2F1", "bd_out":  "#4DB6AC",
    "bg_loop": "#F3E5F5", "bd_loop": "#BA68C8",
    "arrow":   "#546E7A",
    "loop":    "#7B1FA2",
    "text":    "#212121",
    "sub":     "#757575",
}


class Box:
    """A positioned box for easy arrow connections."""
    def __init__(self, ax, x, y, w, h, title, lines, bg, bd, title_sz=8, line_sz=6):
        self.x, self.y, self.w, self.h = x, y, w, h
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor=bg, edgecolor=bd, linewidth=0.9, zorder=2)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h - 0.02, title,
                fontsize=title_sz, fontweight="bold", ha="center", va="top",
                color=C["text"], zorder=6)
        if lines:
            ax.text(x + w/2, y + h * 0.38, "\n".join(lines),
                    fontsize=line_sz, ha="center", va="center",
                    color=C["sub"], linespacing=1.4, zorder=6)

    @property
    def right(self):
        return self.x + self.w, self.y + self.h / 2

    @property
    def left(self):
        return self.x, self.y + self.h / 2

    @property
    def top(self):
        return self.x + self.w / 2, self.y + self.h

    @property
    def bottom(self):
        return self.x + self.w / 2, self.y


def arr(ax, p1, p2, color=None, lw=1.0):
    color = color or C["arrow"]
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw), zorder=5)


def carr(ax, p1, p2, color=None, lw=0.8, rad=0.2):
    color = color or C["arrow"]
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"), zorder=5)


def section_bg(ax, x, y, w, h, label):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.008",
        facecolor="#FAFAFA", edgecolor="#E0E0E0",
        linewidth=0.5, linestyle=(0, (4, 3)), zorder=0)
    ax.add_patch(patch)
    ax.text(x + 0.008, y + h - 0.008, label,
            fontsize=6.5, color="#9E9E9E", fontweight="bold",
            fontstyle="italic", va="top", zorder=1)


def main():
    fig, ax = plt.subplots(figsize=(7.16, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    bh = 0.19  # standard box height
    gap = 0.018  # gap between boxes

    # =================================================================
    # ROW 1: OFFLINE (y = 0.77)
    # =================================================================
    r1y = 0.77
    section_bg(ax, 0.01, 0.73, 0.98, 0.26,
               "OFFLINE  (Training & Knowledge Base Construction)")

    r1 = []  # store boxes for arrow connections
    x = 0.03
    r1.append(Box(ax, x, r1y, 0.13, bh, "Darshan Logs",
                  ["1.37M jobs", "ALCF Polaris"],
                  C["bg_data"], C["bd_data"]))

    x = 0.03 + 0.13 + gap
    r1.append(Box(ax, x, r1y, 0.145, bh, "Feature Extraction",
                  [r"$\mathbf{x} \in \mathbb{R}^{157}$", "POSIX+MPI-IO+STDIO"],
                  C["bg_ml"], C["bd_ml"]))

    x += 0.145 + gap
    r1.append(Box(ax, x, r1y, 0.155, bh, "Biquality Training",
                  ["91K heuristic + 187 GT", "XGBoost, w=100"],
                  C["bg_ml"], C["bd_ml"]))

    x += 0.155 + gap
    r1.append(Box(ax, x, r1y, 0.155, bh, "Benchmark Sweep",
                  ["6 suites, 623 configs", "IOR/mdtest/DLIO/..."],
                  C["bg_kb"], C["bd_kb"]))

    x += 0.155 + gap
    r1.append(Box(ax, x, r1y, 0.13, bh, "Knowledge Base",
                  ["623 entries", "signature+fix+code"],
                  C["bg_kb"], C["bd_kb"]))

    x += 0.13 + gap
    r1.append(Box(ax, x, r1y, 0.10, bh, "ML Model",
                  ["F1 = 0.923", "8-label, 5-seed"],
                  C["bg_out"], C["bd_out"]))

    # Row 1 arrows: each box.right → next box.left
    for i in range(len(r1) - 1):
        arr(ax, r1[i].right, r1[i+1].left)

    # =================================================================
    # ROW 2: ONLINE PIPELINE (y = 0.42)
    # =================================================================
    r2y = 0.42
    section_bg(ax, 0.01, 0.38, 0.98, 0.32,
               "ONLINE  (Inference Pipeline)")

    r2 = []
    x = 0.03
    r2.append(Box(ax, x, r2y, 0.09, bh, "Darshan\nLog",
                  ["new trace"],
                  C["bg_data"], C["bd_data"], title_sz=7.5))

    x += 0.09 + gap
    r2.append(Box(ax, x, r2y, 0.135, bh, "1. ML Detect",
                  ["8-dim classifier", "threshold = 0.3", "3.9 ms"],
                  C["bg_ml"], C["bd_ml"], title_sz=7.5))

    x += 0.135 + gap
    r2.append(Box(ax, x, r2y, 0.12, bh, "2. SHAP",
                  ["per-label top-K", "feature attribution", "2.7 ms"],
                  C["bg_llm"], C["bd_llm"], title_sz=7.5))

    x += 0.12 + gap
    r2.append(Box(ax, x, r2y, 0.13, bh, "3. KB Retrieve",
                  ["similarity search", "623 entries", "2.8 ms"],
                  C["bg_kb"], C["bd_kb"], title_sz=7.5))

    x += 0.13 + gap
    r2.append(Box(ax, x, r2y, 0.145, bh, "4. LLM Generate",
                  ["Claude/GPT-4o/Llama", "code-level fix", "~14 s"],
                  C["bg_llm"], C["bd_llm"], title_sz=7.5))

    x += 0.145 + gap
    r2.append(Box(ax, x, r2y, 0.155, bh, "Code Fix",
                  ["grounded recommendation", "+ KB citations", "groundedness = 1.0"],
                  C["bg_out"], C["bd_out"], title_sz=7.5))

    # Row 2 arrows
    for i in range(len(r2) - 1):
        arr(ax, r2[i].right, r2[i+1].left)

    # Mode label
    ax.text(r2[5].x + r2[5].w / 2, r2y - 0.025, "single-shot mode",
            fontsize=6, ha="center", color=C["bd_out"],
            fontweight="bold", fontstyle="italic", zorder=6)

    # Vertical arrows: offline model → online ML Detect, offline KB → online KB Retrieve
    carr(ax, r1[2].bottom, r2[1].top, color="#90CAF9", lw=0.6, rad=-0.15)
    ax.text((r1[2].bottom[0] + r2[1].top[0])/2 - 0.04, 0.68,
            "trained\nmodel", fontsize=5, color="#64B5F6", ha="center", zorder=6)

    carr(ax, r1[4].bottom, r2[3].top, color="#81C784", lw=0.6, rad=-0.1)
    ax.text((r1[4].bottom[0] + r2[3].top[0])/2 + 0.02, 0.68,
            "KB", fontsize=5, color="#81C784", ha="center", zorder=6)

    # =================================================================
    # ROW 3: ITERATIVE (y = 0.07)
    # =================================================================
    r3y = 0.07
    section_bg(ax, 0.12, 0.03, 0.87, 0.32,
               "ITERATIVE  (Closed-Loop Validation)")

    r3 = []
    x = 0.15
    r3.append(Box(ax, x, r3y, 0.155, bh, "Execute Fix",
                  ["SLURM submit", "benchmark rerun", "on HPC cluster"],
                  C["bg_loop"], C["bd_loop"]))

    x += 0.155 + gap
    r3.append(Box(ax, x, r3y, 0.155, bh, "New Darshan",
                  ["collect I/O profile", "measure throughput"],
                  C["bg_loop"], C["bd_loop"]))

    x += 0.155 + gap
    r3.append(Box(ax, x, r3y, 0.155, bh, "ML Re-detect",
                  ["converged?", "all dims < 0.3", "or max 5 iters"],
                  C["bg_loop"], C["bd_loop"]))

    x += 0.155 + gap
    r3.append(Box(ax, x, r3y, 0.17, bh, "Validated Speedup",
                  ["measured improvement", "33/33 runs, 6 benchmarks", "geomean 4.52x"],
                  C["bg_out"], C["bd_out"]))

    # Row 3 arrows
    for i in range(len(r3) - 1):
        arr(ax, r3[i].right, r3[i+1].left, color=C["loop"])

    # Mode label
    ax.text(r3[3].x + r3[3].w / 2, r3y - 0.025, "iterative mode",
            fontsize=6, ha="center", color=C["bd_loop"],
            fontweight="bold", fontstyle="italic", zorder=6)

    # LLM Generate → Execute Fix (curved down)
    carr(ax, (r2[4].x + r2[4].w * 0.3, r2[4].y),
         (r3[0].x + r3[0].w * 0.5, r3[0].y + r3[0].h),
         color=C["loop"], lw=1.0, rad=0.2)

    # ML Re-detect → back to SHAP (curved up, the iterate loop)
    carr(ax, (r3[2].x + r3[2].w * 0.5, r3[2].y + r3[2].h),
         (r2[2].x + r2[2].w * 0.5, r2[2].y),
         color=C["loop"], lw=1.0, rad=0.3)

    # "iterate" label on the loop arrow
    ax.text(0.44, 0.345, "iterate", fontsize=7, color=C["loop"],
            fontweight="bold", ha="center", rotation=55, zorder=6,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    # No in-figure title — LaTeX \caption{} handles this per IEEE convention

    # Save
    fig.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.005)
    for fmt in ["pdf", "png"]:
        path = FIG_DIR / f"fig_architecture.{fmt}"
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {path} ({path.stat().st_size:,} bytes)")
    plt.close(fig)


if __name__ == "__main__":
    main()
