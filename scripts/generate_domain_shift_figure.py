#!/usr/bin/env python
"""
Generate domain shift visualization (t-SNE / UMAP) between
Polaris production data and Delta benchmark data.

Addresses reviewer weakness W9: domain shift between training
and evaluation data.

Outputs:
  - paper/figures/fig_domain_shift_source.pdf
  - paper/figures/fig_domain_shift_labels.pdf
  - results/domain_shift_analysis.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Style: IOSage RC-params (Okabe-Ito palette, serif, pdf.fonttype=42)
# ---------------------------------------------------------------------------
RCPARAMS_SC2026 = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
plt.rcParams.update(RCPARAMS_SC2026)

# Okabe-Ito color palette
OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_GREEN = "#009E73"
OI_RED = "#D55E00"
OI_PURPLE = "#CC79A7"
OI_CYAN = "#56B4E9"
OI_YELLOW = "#F0E442"
OI_GRAY = "#999999"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model_features(
    prod_cols: list, bench_cols: list, exclude: list
) -> list:
    """Return the intersection of feature columns, minus excluded ones."""
    # Drop metadata/info columns (start with _) and excluded features
    exclude_set = set(exclude)
    common = sorted(set(prod_cols) & set(bench_cols))
    features = [
        c for c in common
        if not c.startswith("_") and c not in exclude_set
    ]
    return features


def stratified_sample(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    n: int,
    seed: int,
    dimensions: list,
) -> pd.DataFrame:
    """Stratified sample from production data using heuristic labels."""
    rng = np.random.RandomState(seed)
    # Create a composite stratification key from the most common dimensions
    key_cols = [d for d in dimensions if d in labels.columns and d != "healthy"]
    if len(key_cols) == 0:
        return df.sample(n=min(n, len(df)), random_state=seed)

    # Build strata string
    strata = labels[key_cols].astype(str).agg("-".join, axis=1)
    strata_counts = strata.value_counts()

    # Proportional allocation
    fracs = strata_counts / strata_counts.sum()
    alloc = (fracs * n).round().astype(int).clip(lower=1)
    # Adjust to hit exactly n
    diff = n - alloc.sum()
    if diff > 0:
        biggest = alloc.nlargest(abs(diff)).index
        alloc.loc[biggest] += 1
    elif diff < 0:
        biggest = alloc.nlargest(abs(diff)).index
        alloc.loc[biggest] -= 1

    idx_list = []
    for stratum, count in alloc.items():
        mask = strata == stratum
        pool = df.index[mask]
        chosen = rng.choice(pool, size=min(count, len(pool)), replace=False)
        idx_list.extend(chosen)

    # If we're short, fill randomly from remaining
    remaining = n - len(idx_list)
    if remaining > 0:
        leftover = df.index.difference(idx_list)
        extra = rng.choice(leftover, size=min(remaining, len(leftover)), replace=False)
        idx_list.extend(extra)

    return df.loc[idx_list[:n]]


def compute_ks_tests(
    prod_feat: np.ndarray, bench_feat: np.ndarray, feature_names: list
) -> list:
    """Per-feature KS test between production and benchmark."""
    results = []
    for i, name in enumerate(feature_names):
        stat, pval = ks_2samp(prod_feat[:, i], bench_feat[:, i])
        results.append({"feature": name, "ks_statistic": float(stat), "p_value": float(pval)})
    return sorted(results, key=lambda x: -x["ks_statistic"])


def main():
    parser = argparse.ArgumentParser(description="Domain shift visualization")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--prod-features", default="data/processed/production/features.parquet")
    parser.add_argument("--prod-labels", default="data/processed/production/labels.parquet")
    parser.add_argument("--bench-features", default="data/processed/benchmark/features.parquet")
    parser.add_argument("--bench-labels", default="data/processed/benchmark/labels.parquet")
    parser.add_argument("--n-prod-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--out-dir", default="paper/figures")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # --- Load config ---
    cfg = load_config(args.config)
    exclude = cfg.get("exclude_features", [])
    dimensions = cfg.get("dimensions", [])

    # --- Load data ---
    logger.info("Loading production features (%s)", args.prod_features)
    prod_df = pd.read_parquet(args.prod_features)
    logger.info("Production shape: %s", prod_df.shape)

    logger.info("Loading benchmark features (%s)", args.bench_features)
    bench_df = pd.read_parquet(args.bench_features)
    logger.info("Benchmark shape: %s", bench_df.shape)

    prod_labels = pd.read_parquet(args.prod_labels)
    bench_labels = pd.read_parquet(args.bench_labels)

    # --- Feature alignment ---
    features = get_model_features(prod_df.columns.tolist(), bench_df.columns.tolist(), exclude)
    logger.info("Model features: %d", len(features))

    # --- Sample production data (stratified) ---
    logger.info("Stratified sampling %d production samples", args.n_prod_samples)
    prod_sample = stratified_sample(
        prod_df, prod_labels, args.n_prod_samples, args.seed, dimensions
    )
    logger.info("Sampled production: %d rows", len(prod_sample))

    # --- Extract numeric features ---
    prod_feat = prod_sample[features].values.astype(np.float64)
    bench_feat = bench_df[features].values.astype(np.float64)

    # Replace inf/nan
    prod_feat = np.nan_to_num(prod_feat, nan=0.0, posinf=0.0, neginf=0.0)
    bench_feat = np.nan_to_num(bench_feat, nan=0.0, posinf=0.0, neginf=0.0)

    # --- log1p transform (same as training pipeline) ---
    logger.info("Applying log1p transform")
    prod_feat = np.log1p(np.abs(prod_feat)) * np.sign(prod_feat)
    bench_feat = np.log1p(np.abs(bench_feat)) * np.sign(bench_feat)

    # --- KS tests ---
    logger.info("Computing per-feature KS tests")
    ks_results = compute_ks_tests(prod_feat, bench_feat, features)
    median_ks = float(np.median([r["ks_statistic"] for r in ks_results]))
    top10 = ks_results[:10]

    logger.info("Median KS statistic: %.4f", median_ks)
    logger.info("Top 10 features with largest domain shift:")
    for r in top10:
        logger.info("  %s: KS=%.4f (p=%.2e)", r["feature"], r["ks_statistic"], r["p_value"])

    # --- Combine for t-SNE ---
    combined = np.vstack([prod_feat, bench_feat])
    # Standardize for t-SNE
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    n_prod = len(prod_feat)
    n_bench = len(bench_feat)
    source_labels = ["Production"] * n_prod + ["Benchmark"] * n_bench

    # --- Try UMAP first, fall back to t-SNE ---
    try:
        import umap
        logger.info("Running UMAP (n_neighbors=15, min_dist=0.1)")
        reducer = umap.UMAP(
            n_neighbors=15, min_dist=0.1, metric="euclidean",
            random_state=args.seed, n_components=2,
        )
        embedding = reducer.fit_transform(combined_scaled)
        method_name = "UMAP"
    except ImportError:
        logger.info("UMAP not available, using t-SNE (perplexity=%d)", int(args.perplexity))
        tsne = TSNE(
            n_components=2, perplexity=args.perplexity,
            random_state=args.seed, n_iter=1000, init="pca",
            learning_rate="auto",
        )
        embedding = tsne.fit_transform(combined_scaled)
        method_name = "t-SNE"

    emb_prod = embedding[:n_prod]
    emb_bench = embedding[n_prod:]

    # --- Figure 1: Colored by source ---
    os.makedirs(args.out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.scatter(
        emb_prod[:, 0], emb_prod[:, 1],
        c=OI_BLUE, s=6, alpha=0.35, label=f"Production (n={n_prod})",
        edgecolors="none", rasterized=True,
    )
    ax.scatter(
        emb_bench[:, 0], emb_bench[:, 1],
        c=OI_ORANGE, s=18, alpha=0.85, label=f"Benchmark (n={n_bench})",
        edgecolors="none", marker="^", rasterized=True,
    )
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    ax.set_title(f"Domain Shift: Production vs. Benchmark ({method_name})")
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="gray")
    ax.set_xticks([])
    ax.set_yticks([])

    fig1_path = os.path.join(args.out_dir, "fig_domain_shift_source.pdf")
    fig.savefig(fig1_path)
    plt.close(fig)
    logger.info("Saved: %s", fig1_path)

    # --- Figure 2: Benchmark points colored by bottleneck type ---
    # Assign each benchmark sample a "dominant" bottleneck dimension
    bottleneck_dims = [
        d for d in dimensions if d != "healthy" and d in bench_labels.columns
    ]
    bench_label_vals = bench_labels[bottleneck_dims].values  # (623, 7)

    # For each benchmark sample, pick the first active bottleneck or "healthy"
    dim_colors = {
        "access_granularity": OI_ORANGE,
        "metadata_intensity": OI_GREEN,
        "parallelism_efficiency": OI_RED,
        "access_pattern": OI_PURPLE,
        "interface_choice": OI_CYAN,
        "file_strategy": OI_YELLOW,
        "throughput_utilization": "#D55E00",  # vermillion
        "healthy": OI_BLUE,
    }

    bench_dominant = []
    for i in range(n_bench):
        assigned = "healthy"
        for j, d in enumerate(bottleneck_dims):
            if bench_label_vals[i, j] == 1:
                assigned = d
                break
        bench_dominant.append(assigned)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    # Production in gray background
    ax.scatter(
        emb_prod[:, 0], emb_prod[:, 1],
        c=OI_GRAY, s=6, alpha=0.2, label="Production",
        edgecolors="none", rasterized=True,
    )
    # Benchmark by bottleneck type
    unique_dims = sorted(set(bench_dominant))
    dim_short = {
        "access_granularity": "Granularity",
        "metadata_intensity": "Metadata",
        "parallelism_efficiency": "Parallelism",
        "access_pattern": "Pattern",
        "interface_choice": "Interface",
        "file_strategy": "File Strat.",
        "throughput_utilization": "Throughput",
        "healthy": "Healthy",
    }
    for dim in unique_dims:
        mask = [d == dim for d in bench_dominant]
        count = sum(mask)
        if count == 0:
            continue
        ax.scatter(
            emb_bench[mask, 0], emb_bench[mask, 1],
            c=dim_colors.get(dim, "black"), s=18, alpha=0.85,
            label=f"{dim_short.get(dim, dim)} ({count})",
            edgecolors="none", marker="^", rasterized=True,
        )
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    ax.set_title(f"Benchmark Bottleneck Types ({method_name})")
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="gray",
              fontsize=6, ncol=2, columnspacing=0.8, handletextpad=0.3)
    ax.set_xticks([])
    ax.set_yticks([])

    fig2_path = os.path.join(args.out_dir, "fig_domain_shift_labels.pdf")
    fig.savefig(fig2_path)
    plt.close(fig)
    logger.info("Saved: %s", fig2_path)

    # --- Save metrics ---
    os.makedirs(args.results_dir, exist_ok=True)
    metrics = {
        "method": method_name,
        "n_production_samples": n_prod,
        "n_benchmark_samples": n_bench,
        "n_features": len(features),
        "median_ks_statistic": median_ks,
        "top_10_shifted_features": top10,
        "all_ks_results": ks_results,
        "seed": args.seed,
    }
    metrics_path = os.path.join(args.results_dir, "domain_shift_analysis.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics: %s", metrics_path)

    # Summary
    logger.info("=" * 60)
    logger.info("DOMAIN SHIFT ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info("Method: %s", method_name)
    logger.info("Features: %d", len(features))
    logger.info("Production samples: %d  |  Benchmark samples: %d", n_prod, n_bench)
    logger.info("Median KS statistic: %.4f", median_ks)
    n_sig = sum(1 for r in ks_results if r["p_value"] < 0.05)
    logger.info(
        "Features with significant shift (p<0.05): %d / %d (%.1f%%)",
        n_sig, len(features), 100.0 * n_sig / len(features),
    )
    logger.info("Top 5 shifted features:")
    for r in ks_results[:5]:
        logger.info("  %-40s KS=%.4f  p=%.2e", r["feature"], r["ks_statistic"], r["p_value"])


if __name__ == "__main__":
    main()
