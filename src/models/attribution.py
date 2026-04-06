"""
SHAP Feature Attribution for Multi-Label I/O Bottleneck Classifiers.

Computes TreeSHAP values per label dimension and generates paper figures:
1. Feature-label heatmap (mean |SHAP| for top features x 8 labels)
2. Per-label beeswarm plots (SHAP value distribution)
3. Global bar chart (stacked by label contribution)

Usage:
    python -m src.models.attribution
    python -m src.models.attribution --model-path models/phase2/xgboost_biquality_w100.pkl
    python -m src.models.attribution --top-k 20 --output-dir paper/figures/shap
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

DIMENSIONS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]

# Short display names for figures
DIM_SHORT = {
    "access_granularity": "Granularity",
    "metadata_intensity": "Metadata",
    "parallelism_efficiency": "Parallelism",
    "access_pattern": "Pattern",
    "interface_choice": "Interface",
    "file_strategy": "File Strategy",
    "throughput_utilization": "Throughput",
    "healthy": "Healthy",
}


def load_model_and_data(model_path, config_path=None):
    """Load trained models and test data."""
    import yaml

    with open(model_path, "rb") as f:
        models = pickle.load(f)

    config_path = config_path or PROJECT_DIR / "configs" / "training.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load production features for feature column names
    paths = config["paths"]
    prod_feat = pd.read_parquet(PROJECT_DIR / paths["production_features"])
    exclude = set(config.get("exclude_features", []))
    for col in prod_feat.columns:
        if col.startswith("_") or col.startswith("drishti_"):
            exclude.add(col)
    feature_cols = [c for c in prod_feat.columns if c not in exclude]

    # Load benchmark test set
    bench_dir = PROJECT_DIR / "data" / "processed" / "benchmark"
    test_feat = pd.read_parquet(bench_dir / "test_features.parquet")
    test_labels = pd.read_parquet(bench_dir / "test_labels.parquet")

    # Align features
    X_test = []
    for col in feature_cols:
        if col in test_feat.columns:
            X_test.append(test_feat[col].values)
        else:
            X_test.append(np.zeros(len(test_feat)))
    X_test = np.column_stack(X_test).astype(np.float32)
    y_test = test_labels[DIMENSIONS].values.astype(np.float32)

    return models, X_test, y_test, feature_cols, test_labels


def compute_shap_values(models, X, feature_names, max_samples=500):
    """Compute SHAP values for each label dimension using TreeSHAP."""
    n_samples = min(len(X), max_samples)
    X_sample = X[:n_samples]

    shap_dict = {}
    for dim in DIMENSIONS:
        if dim not in models:
            continue
        logger.info("  Computing SHAP for '%s' (%d samples)...", dim, n_samples)
        explainer = shap.TreeExplainer(models[dim])
        sv = explainer.shap_values(X_sample)
        # For binary classifier, shap_values may return list [neg, pos]
        if isinstance(sv, list):
            sv = sv[1]  # positive class
        shap_dict[dim] = sv

    return shap_dict, X_sample


def plot_feature_label_heatmap(shap_dict, feature_names, output_path, top_k=20):
    """Generate heatmap of mean |SHAP| (features x labels).

    This is the KEY figure for the paper: shows which features matter
    for which bottleneck type at a glance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Compute mean |SHAP| per feature per label
    n_features = len(feature_names)
    n_labels = len(DIMENSIONS)
    importance_matrix = np.zeros((n_features, n_labels))

    for j, dim in enumerate(DIMENSIONS):
        if dim in shap_dict:
            importance_matrix[:, j] = np.abs(shap_dict[dim]).mean(axis=0)

    # Select top-K features by max importance across any label
    max_importance = importance_matrix.max(axis=1)
    top_idx = np.argsort(max_importance)[-top_k:][::-1]

    matrix_top = importance_matrix[top_idx]
    names_top = [feature_names[i] for i in top_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix_top, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(n_labels))
    ax.set_xticklabels([DIM_SHORT[d] for d in DIMENSIONS], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(names_top, fontsize=8)

    ax.set_xlabel("Bottleneck Dimension", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_title(f"Mean |SHAP| Value — Top {top_k} Features", fontsize=12)

    plt.colorbar(im, ax=ax, label="Mean |SHAP|", shrink=0.8)
    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(str(output_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap: %s", output_path)

    return importance_matrix, top_idx


def plot_per_label_beeswarm(shap_dict, X_sample, feature_names, output_dir, top_k=15):
    """Generate beeswarm plot for each label dimension."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for dim in DIMENSIONS:
        if dim not in shap_dict:
            continue

        sv = shap_dict[dim]
        explanation = shap.Explanation(
            values=sv,
            data=X_sample,
            feature_names=feature_names,
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.beeswarm(explanation, max_display=top_k, show=False)
        ax.set_title(f"SHAP — {DIM_SHORT[dim]}", fontsize=12)
        plt.tight_layout()

        path = output_dir / f"shap_beeswarm_{dim}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved beeswarm: %s", path)


def plot_global_bar(shap_dict, feature_names, output_path, top_k=20):
    """Stacked bar chart: global feature importance colored by label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_features = len(feature_names)
    contributions = np.zeros((n_features, len(DIMENSIONS)))

    for j, dim in enumerate(DIMENSIONS):
        if dim in shap_dict:
            contributions[:, j] = np.abs(shap_dict[dim]).mean(axis=0)

    # Top K by total importance
    total = contributions.sum(axis=1)
    top_idx = np.argsort(total)[-top_k:]

    # IEEE two-column: single-column width is ~3.5in, full-width is ~7.16in
    fig, ax = plt.subplots(figsize=(7.16, 5.0))

    y_pos = np.arange(top_k)
    bar_height = 0.65
    left = np.zeros(top_k)

    colors = plt.cm.Set2(np.linspace(0, 1, len(DIMENSIONS)))

    for j, dim in enumerate(DIMENSIONS):
        widths = contributions[top_idx, j]
        ax.barh(y_pos, widths, height=bar_height, left=left,
                label=DIM_SHORT[dim],
                color=colors[j], edgecolor="white", linewidth=0.3)
        left += widths

    # Feature names as y-tick labels — larger font for readability
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=10,
                       fontfamily="monospace")
    ax.set_xlabel("Mean |SHAP| Value", fontsize=11)

    # Legend at top, horizontal, outside the bars
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1),
              fontsize=9, ncol=4, frameon=True, framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout(pad=1.5)

    fig.savefig(output_path, dpi=300, bbox_inches="tight",
                pad_inches=0.01)
    fig.savefig(str(output_path).replace(".pdf", ".png"), dpi=300,
                bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    logger.info("Saved global bar: %s", output_path)


def validate_shap_against_domain(shap_dict, feature_names, y_test):
    """Check if SHAP top features match known domain expectations."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("SHAP DOMAIN VALIDATION")
    logger.info("=" * 60)

    expected = {
        "access_granularity": ["small_io_ratio", "small_write_ratio", "avg_write_size",
                                "POSIX_SIZE_WRITE_0_100", "POSIX_SIZE_WRITE_100_1K"],
        "metadata_intensity": ["metadata_time_ratio", "POSIX_F_META_TIME", "opens_per_op",
                                "POSIX_OPENS", "stats_per_op"],
        "access_pattern": ["seq_read_ratio", "seq_write_ratio", "POSIX_SEQ_READS",
                            "POSIX_SEQ_WRITES"],
        "interface_choice": ["collective_ratio", "MPIIO_COLL_WRITES", "MPIIO_INDEP_WRITES",
                              "has_mpiio"],
        "file_strategy": ["num_files", "POSIX_FILENOS", "nprocs"],
        "throughput_utilization": ["total_bw_mb_s", "fsync_ratio", "POSIX_FSYNCS",
                                    "write_bw_mb_s"],
        "healthy": ["total_bw_mb_s", "seq_write_ratio", "small_io_ratio"],
    }

    for dim in DIMENSIONS:
        if dim not in shap_dict:
            continue
        mean_abs = np.abs(shap_dict[dim]).mean(axis=0)
        top_10_idx = np.argsort(mean_abs)[-10:][::-1]
        top_10_names = [feature_names[i] for i in top_10_idx]

        expected_feats = expected.get(dim, [])
        matched = [f for f in expected_feats if f in top_10_names]
        match_rate = len(matched) / max(len(expected_feats), 1)

        logger.info("")
        logger.info("  %s (%.0f%% domain match):", dim, 100 * match_rate)
        logger.info("    Top SHAP:  %s", ", ".join(top_10_names[:5]))
        logger.info("    Expected:  %s", ", ".join(expected_feats[:5]))
        logger.info("    Matched:   %s", ", ".join(matched) if matched else "NONE")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis for multi-label I/O classifiers")
    parser.add_argument("--model-path", default="models/phase2/xgboost_biquality_w100.pkl")
    parser.add_argument("--output-dir", default="paper/figures/shap")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=436)
    args = parser.parse_args()

    output_dir = PROJECT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model and data...")
    models, X_test, y_test, feature_cols, test_labels = load_model_and_data(
        PROJECT_DIR / args.model_path
    )
    logger.info("Model: %d labels, Test: %d samples, Features: %d",
                len(models), len(X_test), len(feature_cols))

    logger.info("")
    logger.info("Computing SHAP values (TreeSHAP)...")
    shap_dict, X_sample = compute_shap_values(
        models, X_test, feature_cols, max_samples=args.max_samples
    )

    logger.info("")
    logger.info("Generating figures...")

    # 1. Feature-label heatmap (MAIN paper figure)
    plot_feature_label_heatmap(
        shap_dict, feature_cols,
        output_dir / "fig_shap_heatmap.pdf",
        top_k=args.top_k,
    )

    # 2. Per-label beeswarm (supplementary)
    plot_per_label_beeswarm(
        shap_dict, X_sample, feature_cols,
        output_dir, top_k=15,
    )

    # 3. Global bar chart
    plot_global_bar(
        shap_dict, feature_cols,
        output_dir / "fig_shap_global_bar.pdf",
        top_k=args.top_k,
    )

    # 4. Domain validation
    validate_shap_against_domain(shap_dict, feature_cols, y_test)

    # Save raw SHAP values for LLM grounding
    shap_path = output_dir / "shap_values.pkl"
    with open(shap_path, "wb") as f:
        pickle.dump({
            "shap_dict": shap_dict,
            "feature_names": feature_cols,
            "X_sample": X_sample,
        }, f)
    logger.info("")
    logger.info("SHAP values saved to %s", shap_path)
    logger.info("All figures saved to %s", output_dir)


if __name__ == "__main__":
    main()
