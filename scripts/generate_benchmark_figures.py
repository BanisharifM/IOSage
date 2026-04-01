#!/usr/bin/env python3
"""Generate benchmark ground-truth and supplementary figures for SC 2026 paper.

Figures:
  B1. fig_gt_label_distribution.pdf   -- GT label distribution stacked by benchmark
  B3. fig_gt_vs_heuristic.pdf         -- GT vs heuristic label distribution comparison
  B4. fig_domain_shift_tsne.pdf       -- t-SNE: benchmark vs production features
  B5. fig_benchmark_signatures.pdf    -- Per-scenario Darshan signature heatmap
  15. fig_facility_health.pdf         -- Facility-scale bottleneck prevalence (1.37M logs)

Usage:
    python scripts/generate_benchmark_figures.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
FIG_DIR = PROJECT_DIR / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style (from figure_style_guide.md)
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
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 8,
    "mathtext.fontset": "stix",
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

DIMENSION_LABELS = {
    "access_granularity":     "Access\nGranularity",
    "metadata_intensity":     "Metadata\nIntensity",
    "parallelism_efficiency": "Parallelism\nEfficiency",
    "access_pattern":         "Access\nPattern",
    "interface_choice":       "Interface\nChoice",
    "file_strategy":          "File\nStrategy",
    "throughput_utilization":  "Throughput\nUtilization",
    "healthy":                "Healthy",
}

DIMENSION_ORDER = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]

BENCH_COLORS = {
    "IOR": COLORS["blue"],
    "mdtest": COLORS["orange"],
    "DLIO": COLORS["green"],
    "custom": COLORS["vermilion"],
    "h5bench": COLORS["purple"],
    "HACC-IO": COLORS["cyan"],
}

BENCH_HATCHES = {
    "IOR": "",
    "mdtest": "//",
    "DLIO": "\\\\",
    "custom": "xx",
    "h5bench": "..",
    "HACC-IO": "++",
}


def apply_style():
    plt.rcParams.update(RCPARAMS_SC2026)


def save_fig(fig, name, subdir=None):
    out_dir = FIG_DIR / subdir if subdir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=300)
    plt.close(fig)
    logger.info("Saved: %s  (%d bytes)", pdf, pdf.stat().st_size)


# ===========================================================================
# B1: Ground-truth label distribution stacked by benchmark
# ===========================================================================
def fig_gt_label_distribution():
    """Stacked bar chart: GT label counts per dimension, colored by benchmark."""
    logger.info("Generating B1: GT label distribution by benchmark...")

    labels_path = PROJECT_DIR / "data" / "processed" / "benchmark" / "labels.parquet"
    if not labels_path.exists():
        logger.warning("Benchmark labels not found: %s", labels_path)
        return

    df = pd.read_parquet(labels_path)

    # Map lowercase parquet values to display names
    BENCH_NAME_MAP = {
        "ior": "IOR",
        "mdtest": "mdtest",
        "dlio": "DLIO",
        "custom": "custom",
        "h5bench": "h5bench",
        "hacc_io": "HACC-IO",
    }

    bench_col = "benchmark"
    if bench_col not in df.columns:
        logger.warning("No 'benchmark' column in labels. Skipping.")
        return

    # Map to display names
    df["bench_display"] = df[bench_col].map(BENCH_NAME_MAP).fillna(df[bench_col])

    benchmarks = ["IOR", "mdtest", "DLIO", "custom", "h5bench", "HACC-IO"]
    dims = DIMENSION_ORDER

    # Count per benchmark per dimension
    counts = {}
    for bench in benchmarks:
        bench_df = df[df["bench_display"] == bench]
        counts[bench] = []
        for dim in dims:
            if dim in bench_df.columns:
                counts[bench].append(int(bench_df[dim].sum()))
            else:
                counts[bench].append(0)

    fig, ax = plt.subplots(figsize=(7.16, 2.8), constrained_layout=True)
    x = np.arange(len(dims))
    width = 0.65
    bottom = np.zeros(len(dims))

    for bench in benchmarks:
        vals = np.array(counts[bench])
        color = BENCH_COLORS.get(bench, COLORS["gray"])
        hatch = BENCH_HATCHES.get(bench, "")
        ax.bar(x, vals, width, bottom=bottom, label=bench,
               color=color, hatch=hatch, edgecolor="white", linewidth=0.3)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([DIMENSION_LABELS[d] for d in dims], ha="center")
    ax.set_ylabel("Number of samples")
    ax.set_title("Ground-truth label distribution by benchmark suite")
    ax.legend(loc="upper right", ncol=3)

    # Annotate totals
    for i, total in enumerate(bottom):
        if total > 0:
            ax.text(i, total + 2, str(int(total)), ha="center", va="bottom", fontsize=6.5)

    save_fig(fig, "fig_gt_label_distribution")


# ===========================================================================
# B3: Ground-truth vs heuristic label distribution comparison
# ===========================================================================
def fig_gt_vs_heuristic():
    """Side-by-side bar chart: GT vs heuristic label rates."""
    logger.info("Generating B3: GT vs heuristic label comparison...")

    # Heuristic label rates (from paper_materials.md, 131K production logs)
    heuristic_rates = {
        "access_granularity": 30.7,
        "metadata_intensity": 0.8,
        "parallelism_efficiency": 1.6,
        "access_pattern": 5.2,
        "interface_choice": 1.2,
        "file_strategy": 1.4,
        "throughput_utilization": 20.5,
        "healthy": 54.2,
    }

    # GT label rates (from benchmark labels)
    labels_path = PROJECT_DIR / "data" / "processed" / "benchmark" / "labels.parquet"
    if labels_path.exists():
        df = pd.read_parquet(labels_path)
        gt_rates = {}
        gt_counts = {}
        for dim in DIMENSION_ORDER:
            if dim in df.columns:
                gt_rates[dim] = float(df[dim].mean() * 100)
                gt_counts[dim] = int(df[dim].sum())
            else:
                gt_rates[dim] = 0.0
                gt_counts[dim] = 0
        logger.info("  GT rates: %s", {d: f"{r:.1f}%" for d, r in gt_rates.items()})
    else:
        logger.warning("Benchmark labels not found, using estimated rates")
        gt_rates = {d: 10.0 for d in DIMENSION_ORDER}

    dims = DIMENSION_ORDER
    fig, ax = plt.subplots(figsize=(7.16, 2.8), constrained_layout=True)
    x = np.arange(len(dims))
    width = 0.35

    h_vals = [heuristic_rates.get(d, 0) for d in dims]
    g_vals = [gt_rates.get(d, 0) for d in dims]

    bars1 = ax.bar(x - width / 2, h_vals, width, label="Heuristic (131K prod.)",
                   color=COLORS["blue"], hatch="", edgecolor="white", linewidth=0.3)
    bars2 = ax.bar(x + width / 2, g_vals, width, label="Ground-truth (623 bench.)",
                   color=COLORS["orange"], hatch="//", edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([DIMENSION_LABELS[d] for d in dims], ha="center")
    ax.set_ylabel("Positive rate (%)")
    ax.set_title("Label distribution: heuristic (production) vs ground-truth (benchmark)")
    ax.legend(loc="upper right")

    save_fig(fig, "fig_gt_vs_heuristic")


# ===========================================================================
# B4: t-SNE domain shift visualization
# ===========================================================================
def fig_domain_shift_tsne():
    """t-SNE visualization of benchmark vs production feature distributions."""
    logger.info("Generating B4: Domain shift t-SNE...")

    from sklearn.manifold import TSNE

    # Load production features (sample)
    prod_path = PROJECT_DIR / "data" / "processed" / "production" / "features.parquet"
    bench_path = PROJECT_DIR / "data" / "processed" / "benchmark" / "features.parquet"

    if not prod_path.exists() or not bench_path.exists():
        logger.warning("Feature files not found. Skipping t-SNE.")
        return

    prod_df = pd.read_parquet(prod_path)
    bench_df = pd.read_parquet(bench_path)

    # Get shared numeric columns (exclude metadata)
    exclude_prefixes = ("_", "drishti_")
    shared_cols = [c for c in prod_df.columns
                   if c in bench_df.columns
                   and not any(c.startswith(p) for p in exclude_prefixes)
                   and prod_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    if len(shared_cols) < 10:
        logger.warning("Too few shared features (%d). Skipping t-SNE.", len(shared_cols))
        return

    # Sample production data
    n_sample = min(2000, len(prod_df))
    prod_sample = prod_df[shared_cols].sample(n=n_sample, random_state=42).fillna(0)
    bench_sample = bench_df[shared_cols].fillna(0)

    # Apply log1p to reduce scale differences
    combined = pd.concat([prod_sample, bench_sample], ignore_index=True)
    combined = np.log1p(combined.abs()) * np.sign(combined)

    # t-SNE
    logger.info("  Running t-SNE on %d samples (%d features)...", len(combined), len(shared_cols))
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    embedding = tsne.fit_transform(combined.values)

    n_prod = len(prod_sample)
    prod_emb = embedding[:n_prod]
    bench_emb = embedding[n_prod:]

    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    ax.scatter(prod_emb[:, 0], prod_emb[:, 1], s=3, alpha=0.3,
               color=COLORS["blue"], label=f"Production ({n_prod})", rasterized=True)
    ax.scatter(bench_emb[:, 0], bench_emb[:, 1], s=8, alpha=0.7,
               color=COLORS["vermilion"], label=f"Benchmark ({len(bench_sample)})",
               marker="^", edgecolors="none")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Feature distribution: production vs benchmark")
    ax.legend(loc="best", markerscale=2)
    ax.grid(False)

    save_fig(fig, "fig_domain_shift_tsne")


# ===========================================================================
# B5: Per-scenario Darshan signature heatmap
# ===========================================================================
def fig_benchmark_signatures():
    """Heatmap: key Darshan features per benchmark scenario."""
    logger.info("Generating B5: Benchmark signature heatmap...")

    bench_path = PROJECT_DIR / "data" / "processed" / "benchmark" / "features.parquet"
    labels_path = PROJECT_DIR / "data" / "processed" / "benchmark" / "labels.parquet"

    if not bench_path.exists() or not labels_path.exists():
        logger.warning("Benchmark files not found. Skipping signatures.")
        return

    feat_df = pd.read_parquet(bench_path)
    lab_df = pd.read_parquet(labels_path)

    # Key diagnostic features
    key_features = [
        "small_io_ratio", "seq_write_ratio", "seq_read_ratio",
        "metadata_time_ratio", "collective_ratio", "avg_write_size",
        "avg_read_size", "read_ratio", "total_bw_mb_s",
    ]
    available = [f for f in key_features if f in feat_df.columns]

    if len(available) < 5:
        logger.warning("Too few key features available (%d). Skipping.", len(available))
        return

    # Get scenario labels
    scenario_col = None
    for col in ["_scenario", "_job_name", "scenario"]:
        if col in lab_df.columns:
            scenario_col = col
            break

    if scenario_col is None:
        logger.warning("No scenario column found. Skipping signature heatmap.")
        return

    # Group by scenario, compute mean feature values
    combined = feat_df[available].copy()
    combined["scenario"] = lab_df[scenario_col].values

    scenario_means = combined.groupby("scenario")[available].mean()

    # Select representative scenarios (up to 20)
    if len(scenario_means) > 20:
        scenario_means = scenario_means.head(20)

    # Normalize per-column for heatmap display
    normalized = (scenario_means - scenario_means.min()) / (scenario_means.max() - scenario_means.min() + 1e-10)

    fig, ax = plt.subplots(figsize=(7.16, 4.5), constrained_layout=True)
    im = ax.imshow(normalized.values, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([f.replace("_", "\n") for f in available], rotation=0, ha="center")
    ax.set_yticks(range(len(scenario_means)))
    ax.set_yticklabels(scenario_means.index, fontsize=6)
    ax.set_title("Normalized Darshan feature signatures per benchmark scenario")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Normalized value", fontsize=7)

    save_fig(fig, "fig_benchmark_signatures")


# ===========================================================================
# Fig 15: Facility-scale I/O health (1.37M logs)
# ===========================================================================
def fig_facility_health():
    """Bar chart: bottleneck prevalence across 131K cleaned production logs."""
    logger.info("Generating Fig 15: Facility-scale I/O health...")

    labels_path = PROJECT_DIR / "data" / "processed" / "production" / "labels.parquet"
    if not labels_path.exists():
        # Try alternate path
        labels_path = PROJECT_DIR / "data" / "processed" / "labels.parquet"
    if not labels_path.exists():
        logger.warning("Production labels not found. Skipping facility health.")
        return

    df = pd.read_parquet(labels_path)

    dims = DIMENSION_ORDER
    rates = []
    counts = []
    for dim in dims:
        if dim in df.columns:
            rate = float(df[dim].mean() * 100)
            count = int(df[dim].sum())
        else:
            rate = 0.0
            count = 0
        rates.append(rate)
        counts.append(count)

    fig, ax = plt.subplots(figsize=(7.16, 2.4), constrained_layout=True)
    x = np.arange(len(dims))
    bars = ax.bar(x, rates, color=PALETTE_8, edgecolor="white", linewidth=0.3,
                  hatch=[HATCHES[i] for i in range(len(dims))])

    # Annotate with counts
    for i, (rate, count) in enumerate(zip(rates, counts)):
        label = f"{rate:.1f}%\n({count:,})"
        ax.text(i, rate + 1.0, label, ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([DIMENSION_LABELS[d] for d in dims], ha="center")
    ax.set_ylabel("Prevalence (%)")
    ax.set_title(f"I/O bottleneck prevalence across {len(df):,} production Darshan logs (ALCF Polaris)")
    ax.set_ylim(0, max(rates) * 1.25)

    save_fig(fig, "fig_facility_health")


# ===========================================================================
# Main
# ===========================================================================
def main():
    apply_style()
    logger.info("=" * 60)
    logger.info("Generating benchmark ground-truth and supplementary figures")
    logger.info("=" * 60)

    fig_gt_label_distribution()     # B1
    fig_gt_vs_heuristic()           # B3
    fig_domain_shift_tsne()         # B4
    fig_benchmark_signatures()      # B5
    fig_facility_health()           # Fig 15

    logger.info("=" * 60)
    logger.info("All figures generated.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
