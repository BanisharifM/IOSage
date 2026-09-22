"""
SHAP Feature Attribution for Multi-Label I/O Bottleneck Classifiers.

Computes TreeSHAP values per label dimension and generates paper figures:
1. Feature-label heatmap (mean |SHAP| for top features x 8 labels)
2. Per-label beeswarm plots (SHAP value distribution)
3. Global bar chart (stacked by label contribution)

The model is a bundle from ``scripts/train_biquality.py``; the samples are
the benchmark test rows of that bundle's run (``splits.npz`` next to it), so
attribution never sees a row the model was fitted on. Healthy is derived from
the seven decisions and has no model, so it has no SHAP values.

Usage:
    python -m src.models.attribution --bundle results/resubmission/training/<run>/xgboost_w100_seed42.pkl \
        --output-dir results/resubmission/shap/<run>
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.biquality import BOTTLENECK_DIMENSIONS, BUNDLE_FORMAT, load_benchmark, load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# The seven modeled labels; healthy is derived and has no attribution
DIMENSIONS = list(BOTTLENECK_DIMENSIONS)

# Short display names for figures
DIM_SHORT = {
    "access_granularity": "Granularity",
    "metadata_intensity": "Metadata",
    "parallelism_efficiency": "Parallelism",
    "access_pattern": "Pattern",
    "interface_choice": "Interface",
    "file_strategy": "File Strategy",
    "throughput_utilization": "Throughput",
}


def load_bundle_and_test_rows(bundle_path):
    """The bundle's models and feature names, and the benchmark test rows of
    its run (from ``splits.npz`` in the same directory)."""
    bundle_path = Path(bundle_path)
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict) or bundle.get("bundle_format") != BUNDLE_FORMAT:
        raise ValueError(f"{bundle_path} is not a model bundle; train with scripts/train_biquality.py")
    splits_path = bundle_path.parent / "splits.npz"
    if not splits_path.exists():
        raise FileNotFoundError(f"{splits_path} missing: the bundle must stay in its run directory")
    splits = np.load(splits_path, allow_pickle=True)
    config = load_config(bundle["config_path"])
    bench = load_benchmark(config, bundle["feature_names"])
    if not np.array_equal(bench.ids, splits["bench_ids"]):
        raise ValueError("benchmark data changed since the run; ids differ from splits.npz")
    test_idx = splits["bench_test"]
    return bundle, bench.X[test_idx], bench.y[test_idx], list(bundle["feature_names"])


def compute_shap_values(models, X, feature_names, max_samples=500):
    """Compute SHAP values for each label dimension using TreeSHAP."""
    n_samples = min(len(X), max_samples)
    X_sample = X[:n_samples]

    shap_dict = {}
    for dim in BOTTLENECK_DIMENSIONS:
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

    # Single-column figure — native size so fonts aren't scaled down
    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    y_pos = np.arange(top_k)
    bar_height = 0.7
    left = np.zeros(top_k)

    colors = plt.cm.Set2(np.linspace(0, 1, len(DIMENSIONS)))

    for j, dim in enumerate(DIMENSIONS):
        widths = contributions[top_idx, j]
        ax.barh(y_pos, widths, height=bar_height, left=left,
                label=DIM_SHORT[dim],
                color=colors[j], edgecolor="white", linewidth=0.2)
        left += widths

    # Tighten right-side whitespace: bound x-axis just past the longest bar
    max_bar_total = float(left.max())
    ax.set_xlim(0, max_bar_total * 1.02)

    # Feature names — readable monospace font (+1pt for legibility)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=7,
                       fontfamily="monospace")
    ax.set_xlabel("Mean |SHAP| Value", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)

    # Legend inside chart — lower-right has space (short bars there)
    ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.01),
              fontsize=6, ncol=2, frameon=True, framealpha=0.95,
              edgecolor="#cccccc", borderpad=0.4,
              columnspacing=0.6, handletextpad=0.3, labelspacing=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    fig.savefig(output_path, dpi=300, bbox_inches="tight",
                pad_inches=0.01)
    fig.savefig(str(output_path).replace(".pdf", ".png"), dpi=300,
                bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    logger.info("Saved global bar: %s", output_path)


# Features a domain expert expects to drive each label, from the label's
# definition (docs/4_reference/IO_Bottleneck_Detection_Guide.md and the
# verification rules in src/data/benchmark_verify.py)
EXPECTED_FEATURES = {
    "access_granularity": ["small_io_ratio", "small_write_ratio", "small_read_ratio", "avg_write_size",
                           "avg_read_size", "POSIX_SIZE_WRITE_0_100", "POSIX_SIZE_WRITE_100_1K",
                           "POSIX_SIZE_READ_0_100", "POSIX_SIZE_READ_100_1K", "medium_write_ratio"],
    "metadata_intensity": ["metadata_time_ratio", "metadata_time_ratio_all", "POSIX_F_META_TIME",
                           "opens_per_op", "POSIX_OPENS", "stats_per_op", "POSIX_STATS"],
    "parallelism_efficiency": ["rank_byte_range_ratio", "top_rank_byte_share", "RANK_BYTES_GINI",
                               "SHARED_BYTE_IMBALANCE", "rank_bytes_cv_all", "io_rank_fraction",
                               "byte_imbalance", "time_imbalance"],
    "access_pattern": ["seq_read_ratio", "seq_write_ratio", "POSIX_SEQ_READS", "POSIX_SEQ_WRITES",
                       "consec_read_ratio", "consec_write_ratio"],
    "interface_choice": ["collective_ratio", "MPIIO_COLL_WRITES", "MPIIO_COLL_READS", "MPIIO_INDEP_WRITES",
                         "MPIIO_INDEP_READS", "has_mpiio", "is_shared_file"],
    "file_strategy": ["num_files", "POSIX_FILENOS", "nprocs", "POSIX_OPENS", "opens_per_mb"],
    "throughput_utilization": ["fsync_ratio", "POSIX_FSYNCS", "total_bw_mb_s", "write_bw_mb_s",
                               "POSIX_MAX_BYTE_READ", "POSIX_MAX_BYTE_WRITTEN"],
}


def validate_shap_against_domain(shap_dict, feature_names, y_test, top_k=10):
    """Per label, on the test samples that carry the label, does the
    attribution point at the features the label is defined by?

    Returns ``{dimension: {...}}`` with the top features over the positive
    samples, the matches with ``EXPECTED_FEATURES``, the sample count, and
    ``status``: ``assessed``, ``no_positive_samples`` or ``not_assessed``
    (no expectation defined). Healthy has no model and is never assessed.
    """
    result = {}
    for i, dim in enumerate(BOTTLENECK_DIMENSIONS):
        positives = np.flatnonzero(y_test[:, i] == 1)
        entry = {"n_positive": int(len(positives))}
        if dim not in EXPECTED_FEATURES:
            entry["status"] = "not_assessed"
        elif len(positives) == 0:
            entry["status"] = "no_positive_samples"
        else:
            mean_abs = np.abs(shap_dict[dim][positives]).mean(axis=0)
            top = [feature_names[j] for j in np.argsort(mean_abs)[-top_k:][::-1]]
            matched = [f for f in EXPECTED_FEATURES[dim] if f in top]
            entry.update(status="assessed", top_features=top, expected=EXPECTED_FEATURES[dim],
                         matched=matched, match_rate=len(matched) / len(EXPECTED_FEATURES[dim]))
        result[dim] = entry
        logger.info("  %-24s %s%s", dim, entry["status"],
                    f" match {entry['match_rate']:.0%} top: {', '.join(entry['top_features'][:5])}"
                    if entry["status"] == "assessed" else "")
    return result


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis for the biquality detector")
    parser.add_argument("--bundle", required=True, help="model bundle inside its run directory")
    parser.add_argument("--output-dir", required=True,
                        help="figures and values go here (never a paper repository)")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle, X_test, y_test, feature_cols = load_bundle_and_test_rows(args.bundle)
    logger.info("Bundle %s seed %s: %d test samples, %d features",
                bundle["model_type"], bundle["seed"], len(X_test), len(feature_cols))

    shap_dict, X_sample = compute_shap_values(bundle["models"], X_test, feature_cols,
                                              max_samples=args.max_samples)
    y_sample = y_test[:len(X_sample)]

    plot_feature_label_heatmap(shap_dict, feature_cols, output_dir / "fig_shap_heatmap.pdf", top_k=args.top_k)
    plot_per_label_beeswarm(shap_dict, X_sample, feature_cols, output_dir, top_k=15)
    plot_global_bar(shap_dict, feature_cols, output_dir / "fig_shap_global_bar.pdf", top_k=args.top_k)

    validation = validate_shap_against_domain(shap_dict, feature_cols, y_sample)
    with open(output_dir / "domain_validation.json", "w") as f:
        json.dump({"bundle": str(args.bundle), "n_samples": int(len(X_sample)), "validation": validation}, f, indent=2)

    with open(output_dir / "shap_values.pkl", "wb") as f:
        pickle.dump({"shap_dict": shap_dict, "feature_names": feature_cols, "X_sample": X_sample,
                     "y_sample": y_sample, "bundle": str(args.bundle)}, f)
    logger.info("SHAP values, figures and domain_validation.json saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
