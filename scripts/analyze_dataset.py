"""
Dataset Characterization Analysis for IOSage
====================================================
Generates figures and statistics from the raw feature parquet file.

Outputs:
  - SC_Draft_Paper/figures/*.png  — Publication-ready figures
  - SC_Draft_Paper/stats.json     — Key numbers for paper_materials.md
  - stdout                        — Summary statistics

Usage::

    python scripts/analyze_dataset.py \
        --input data/processed/raw_features.parquet \
        --output-dir SC_Draft_Paper

    # Quick mode (sample 50K rows for faster iteration)
    python scripts/analyze_dataset.py \
        --input data/processed/raw_features.parquet \
        --output-dir SC_Draft_Paper --sample 50000
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Publication-quality plot defaults
plt.rcParams.update({
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'serif',
})

# ---------------------------------------------------------------------------
# Feature groupings (from feature_extraction.py)
# ---------------------------------------------------------------------------
VOLUME_COLS = [
    'POSIX_BYTES_READ', 'POSIX_BYTES_WRITTEN',
    'MPIIO_BYTES_READ', 'MPIIO_BYTES_WRITTEN',
    'STDIO_BYTES_READ', 'STDIO_BYTES_WRITTEN',
]
OP_COUNT_COLS = [
    'POSIX_READS', 'POSIX_WRITES', 'POSIX_OPENS', 'POSIX_SEEKS',
    'POSIX_STATS', 'POSIX_FSYNCS',
    'MPIIO_INDEP_READS', 'MPIIO_INDEP_WRITES',
    'MPIIO_COLL_READS', 'MPIIO_COLL_WRITES',
    'STDIO_READS', 'STDIO_WRITES',
]
TIMING_COLS = [
    'POSIX_F_READ_TIME', 'POSIX_F_WRITE_TIME', 'POSIX_F_META_TIME',
    'MPIIO_F_READ_TIME', 'MPIIO_F_WRITE_TIME',
    'STDIO_F_READ_TIME', 'STDIO_F_WRITE_TIME',
]
INDICATOR_COLS = [
    'has_posix', 'has_mpiio', 'has_stdio',
    'has_hdf5', 'has_pnetcdf', 'has_apmpi', 'has_heatmap',
    'is_shared_file',
]

# Heuristic signals for AI/ML workloads
# AI workloads tend to: use STDIO (Python), many small reads (data loading),
# no MPI-IO (PyTorch/TF use POSIX), high read:write ratio (inference/training reads)
AI_SIGNAL_MODULES = {'STDIO,APMPI,HEATMAP', 'POSIX,STDIO,APMPI,HEATMAP'}


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_basic_stats(df):
    """Compute basic dataset statistics."""
    stats = {}
    stats['total_rows'] = len(df)
    stats['unique_jobids'] = int(df['_jobid'].nunique()) if '_jobid' in df.columns else 0
    stats['feature_cols'] = len([c for c in df.columns if not c.startswith('_')])
    stats['info_cols'] = len([c for c in df.columns if c.startswith('_')])

    # Date range from _start_time (Unix timestamps)
    if '_start_time' in df.columns and (df['_start_time'] > 0).any():
        valid_ts = df['_start_time'][df['_start_time'] > 0]
        stats['date_min'] = str(pd.Timestamp(valid_ts.min(), unit='s').date())
        stats['date_max'] = str(pd.Timestamp(valid_ts.max(), unit='s').date())
    elif '_source_path' in df.columns:
        # Fallback: extract from file paths
        import re
        pattern = r'Darshan_Logs/(\d{4})/(\d{1,2})/(\d{1,2})/'
        dates = df['_source_path'].str.extract(pattern)
        dates.columns = ['year', 'month', 'day']
        dates = dates.dropna()
        if len(dates) > 0:
            dates = dates.astype(int)
            date_strs = dates.apply(
                lambda r: f"{r['year']:04d}-{r['month']:02d}-{r['day']:02d}",
                axis=1)
            stats['date_min'] = date_strs.min()
            stats['date_max'] = date_strs.max()

    # Job size
    if 'nprocs' in df.columns:
        stats['nprocs_median'] = float(df['nprocs'].median())
        stats['nprocs_mean'] = float(df['nprocs'].mean())
        stats['nprocs_max'] = int(df['nprocs'].max())
        stats['nprocs_p25'] = float(df['nprocs'].quantile(0.25))
        stats['nprocs_p75'] = float(df['nprocs'].quantile(0.75))
        stats['nprocs_p99'] = float(df['nprocs'].quantile(0.99))
        stats['single_proc_pct'] = float(
            (df['nprocs'] == 1).sum() / len(df) * 100)

    # Runtime
    if 'runtime_seconds' in df.columns:
        valid_rt = df['runtime_seconds'][df['runtime_seconds'] > 0]
        stats['runtime_median_min'] = float(valid_rt.median() / 60)
        stats['runtime_mean_min'] = float(valid_rt.mean() / 60)
        stats['runtime_max_hr'] = float(valid_rt.max() / 3600)

    # I/O volume
    total_bytes = (df.get('POSIX_BYTES_READ', 0) + df.get('POSIX_BYTES_WRITTEN', 0)
                   + df.get('STDIO_BYTES_READ', 0) + df.get('STDIO_BYTES_WRITTEN', 0))
    if hasattr(total_bytes, 'sum'):
        stats['total_io_bytes'] = float(total_bytes.sum())
        stats['total_io_tb'] = float(total_bytes.sum() / 1e12)
        stats['zero_io_pct'] = float((total_bytes == 0).sum() / len(df) * 100)

    # Module distribution
    if '_modules' in df.columns:
        mod_counts = df['_modules'].value_counts()
        stats['module_combos'] = {
            str(k): int(v) for k, v in mod_counts.head(15).items()
        }
        stats['has_posix_pct'] = float(
            df['_modules'].str.contains('POSIX').sum() / len(df) * 100)
        stats['has_mpiio_pct'] = float(
            df['_modules'].str.contains('MPI-IO').sum() / len(df) * 100)
        stats['has_stdio_pct'] = float(
            df['_modules'].str.contains('STDIO').sum() / len(df) * 100)

    return stats


def compute_sparsity(df):
    """Compute feature sparsity — what fraction of values are zero per feature."""
    feat_cols = [c for c in df.columns if not c.startswith('_')]
    sparsity = {}
    for c in feat_cols:
        if df[c].dtype in ['float64', 'int64', 'float32', 'int32']:
            zero_frac = (df[c] == 0).sum() / len(df)
            sparsity[c] = float(zero_frac)
    return sparsity


def classify_workload_type(df):
    """Heuristic classification of workload types based on I/O patterns.

    Categories:
      - AI/ML: STDIO-heavy, many small reads, Python-style I/O
      - Traditional HPC: POSIX+MPI-IO, large sequential I/O
      - Metadata-heavy: High open/stat count relative to data volume
      - Minimal I/O: Very little I/O activity
      - Other: Doesn't fit cleanly
    """
    labels = pd.Series('other', index=df.index)

    posix_bytes = df.get('POSIX_BYTES_READ', 0) + df.get('POSIX_BYTES_WRITTEN', 0)
    stdio_bytes = df.get('STDIO_BYTES_READ', 0) + df.get('STDIO_BYTES_WRITTEN', 0)
    total_bytes = posix_bytes + stdio_bytes
    has_mpiio = df.get('has_mpiio', 0) == 1
    has_posix = df.get('has_posix', 0) == 1
    has_stdio = df.get('has_stdio', 0) == 1

    # Minimal I/O: total bytes < 1 MB
    minimal = total_bytes < 1e6
    labels[minimal] = 'minimal_io'

    # AI/ML heuristic: STDIO present, no MPI-IO, POSIX present
    # (Python frameworks use POSIX for data loading + STDIO for logging)
    ai_signal = has_stdio & ~has_mpiio & (df.get('nprocs', 1) >= 1)
    # Additional: small read pattern (many reads, small average size)
    posix_reads = df.get('POSIX_READS', 0)
    posix_bytes_read = df.get('POSIX_BYTES_READ', 0)
    avg_read_size = posix_bytes_read / np.maximum(posix_reads, 1)
    small_read_pattern = (posix_reads > 100) & (avg_read_size < 1e6)  # < 1MB avg

    labels[ai_signal & ~minimal] = 'likely_ai_ml'
    labels[ai_signal & small_read_pattern & ~minimal] = 'likely_ai_ml'

    # Traditional HPC: MPI-IO present, large I/O
    trad_hpc = has_mpiio & (total_bytes > 1e6)
    labels[trad_hpc] = 'traditional_hpc'

    # Metadata-heavy: high opens relative to bytes
    opens = df.get('POSIX_OPENS', 0)
    metadata_heavy = (opens > 1000) & (total_bytes < 1e8) & ~minimal
    labels[metadata_heavy] = 'metadata_heavy'

    return labels


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_temporal_distribution(df, fig_dir):
    """Fig: Logs per month over time.

    Uses _start_time (Unix timestamp) if available and non-zero,
    falls back to extracting dates from _source_path.
    """
    # Try _start_time first (preferred — actual job start time)
    if '_start_time' in df.columns and (df['_start_time'] > 0).any():
        valid = df[df['_start_time'] > 0]
        ts = pd.to_datetime(valid['_start_time'], unit='s')
        ym = ts.dt.to_period('M').astype(str)
        monthly = ym.value_counts().sort_index()
    elif '_source_path' in df.columns:
        # Fallback: extract from file paths
        pattern = r'Darshan_Logs/(\d{4})/(\d{1,2})/'
        dates = df['_source_path'].str.extract(pattern)
        dates.columns = ['year', 'month']
        dates = dates.dropna()
        if len(dates) == 0:
            logger.warning("Could not extract dates from paths")
            return
        dates['year'] = dates['year'].astype(int)
        dates['month'] = dates['month'].astype(int)
        dates['ym'] = dates['year'].astype(str) + '-' + dates['month'].apply(
            lambda x: f'{x:02d}')
        monthly = dates['ym'].value_counts().sort_index()
    else:
        logger.warning("No temporal data available, skipping temporal plot")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(monthly)), monthly.values, color='#2196F3', edgecolor='none')
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly.index, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Darshan Logs')
    ax.set_title('Temporal Distribution of Darshan Logs on Polaris')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    fig.savefig(fig_dir / 'temporal_distribution.png', bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved temporal_distribution.png")


def plot_module_distribution(df, fig_dir):
    """Fig: Module combination frequency (horizontal bar)."""
    if '_modules' not in df.columns:
        return
    mod_counts = df['_modules'].value_counts().head(10)
    # Clean up labels
    labels = [m if m else '(no I/O modules)' for m in mod_counts.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(labels)), mod_counts.values, color='#4CAF50',
                   edgecolor='none')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Number of Logs')
    ax.set_title('Top 10 Darshan Module Combinations')
    ax.invert_yaxis()
    # Add count labels on bars
    for bar, val in zip(bars, mod_counts.values):
        ax.text(bar.get_width() + len(df) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', fontsize=9)
    fig.savefig(fig_dir / 'module_distribution.png', bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved module_distribution.png")


def plot_nprocs_distribution(df, fig_dir):
    """Fig: Job size (nprocs) distribution — log scale."""
    if 'nprocs' not in df.columns:
        return
    valid = df['nprocs'][df['nprocs'] > 0]

    fig, ax = plt.subplots()
    bins = np.logspace(0, np.log10(valid.max() + 1), 50)
    ax.hist(valid, bins=bins, color='#FF9800', edgecolor='none', alpha=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Processes (nprocs)')
    ax.set_ylabel('Count')
    ax.set_title('Job Size Distribution')
    # Mark common sizes
    for nproc in [1, 4, 16, 64, 256, 1024]:
        if nproc <= valid.max():
            ax.axvline(nproc, color='gray', linestyle='--', alpha=0.3)
            ax.text(nproc, ax.get_ylim()[1] * 0.7, str(nproc),
                    rotation=90, fontsize=8, alpha=0.5)
    fig.savefig(fig_dir / 'nprocs_distribution.png', bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved nprocs_distribution.png")


def plot_io_volume_distribution(df, fig_dir):
    """Fig: Total I/O bytes distribution — log scale."""
    total_bytes = (df.get('POSIX_BYTES_READ', 0) + df.get('POSIX_BYTES_WRITTEN', 0)
                   + df.get('STDIO_BYTES_READ', 0) + df.get('STDIO_BYTES_WRITTEN', 0))
    nonzero = total_bytes[total_bytes > 0]

    fig, ax = plt.subplots()
    bins = np.logspace(0, np.log10(nonzero.max() + 1), 60)
    ax.hist(nonzero, bins=bins, color='#9C27B0', edgecolor='none', alpha=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Total I/O Volume (bytes)')
    ax.set_ylabel('Count')
    ax.set_title(f'I/O Volume Distribution (excluding {(total_bytes == 0).sum():,} zero-I/O logs)')
    # Mark size thresholds
    for size, label in [(1e3, '1KB'), (1e6, '1MB'), (1e9, '1GB'), (1e12, '1TB')]:
        if size < nonzero.max():
            ax.axvline(size, color='red', linestyle=':', alpha=0.4)
            ax.text(size, ax.get_ylim()[1] * 0.5, label,
                    rotation=90, fontsize=8, color='red', alpha=0.6)
    fig.savefig(fig_dir / 'io_volume_distribution.png', bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved io_volume_distribution.png")


def plot_runtime_distribution(df, fig_dir):
    """Fig: Job runtime distribution."""
    if 'runtime_seconds' not in df.columns:
        return
    valid = df['runtime_seconds'][df['runtime_seconds'] > 0]

    fig, ax = plt.subplots()
    bins = np.logspace(0, np.log10(valid.max() + 1), 50)
    ax.hist(valid, bins=bins, color='#00BCD4', edgecolor='none', alpha=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Runtime (seconds)')
    ax.set_ylabel('Count')
    ax.set_title('Job Runtime Distribution')
    for sec, label in [(60, '1min'), (3600, '1hr'), (86400, '1day')]:
        if sec < valid.max():
            ax.axvline(sec, color='gray', linestyle='--', alpha=0.4)
            ax.text(sec * 1.1, ax.get_ylim()[1] * 0.5, label,
                    fontsize=9, alpha=0.6)
    fig.savefig(fig_dir / 'runtime_distribution.png', bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved runtime_distribution.png")


def plot_sparsity(sparsity, fig_dir):
    """Fig: Feature sparsity — what % of each feature is zero."""
    sorted_sp = sorted(sparsity.items(), key=lambda x: x[1], reverse=True)
    names = [s[0] for s in sorted_sp]
    values = [s[1] * 100 for s in sorted_sp]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(names)), values, color='#F44336', edgecolor='none',
           alpha=0.7, width=1.0)
    ax.set_xlabel('Feature Index (sorted by sparsity)')
    ax.set_ylabel('Zero-Value Percentage (%)')
    ax.set_title(f'Feature Sparsity ({sum(1 for v in values if v > 95)} features > 95% zero)')
    ax.axhline(95, color='black', linestyle='--', alpha=0.5, label='95% threshold')
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5, label='50% threshold')
    ax.legend()

    # Summary text
    n_dense = sum(1 for v in values if v < 10)
    n_moderate = sum(1 for v in values if 10 <= v < 50)
    n_sparse = sum(1 for v in values if 50 <= v < 95)
    n_very_sparse = sum(1 for v in values if v >= 95)
    ax.text(0.98, 0.95,
            f'Dense (<10%): {n_dense}\nModerate (10-50%): {n_moderate}\n'
            f'Sparse (50-95%): {n_sparse}\nVery sparse (>95%): {n_very_sparse}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.savefig(fig_dir / 'feature_sparsity.png', bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved feature_sparsity.png")


def plot_workload_types(workload_labels, fig_dir):
    """Fig: Workload type distribution (pie + bar)."""
    counts = workload_labels.value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart
    colors = {
        'likely_ai_ml': '#2196F3',
        'traditional_hpc': '#4CAF50',
        'metadata_heavy': '#FF9800',
        'minimal_io': '#9E9E9E',
        'other': '#E0E0E0',
    }
    pie_colors = [colors.get(c, '#E0E0E0') for c in counts.index]
    ax1.pie(counts.values, labels=counts.index, colors=pie_colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
    ax1.set_title('Workload Type Distribution (Heuristic)')

    # Bar chart with counts
    ax2.barh(range(len(counts)), counts.values, color=pie_colors)
    ax2.set_yticks(range(len(counts)))
    ax2.set_yticklabels(counts.index)
    ax2.set_xlabel('Number of Logs')
    ax2.set_title('Workload Types by Count')
    ax2.invert_yaxis()
    for i, (_, val) in enumerate(counts.items()):
        ax2.text(val + len(workload_labels) * 0.01, i, f'{val:,}',
                 va='center', fontsize=9)

    fig.savefig(fig_dir / 'workload_types.png', bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved workload_types.png")


def plot_read_write_balance(df, fig_dir):
    """Fig: Read vs Write volume scatter."""
    reads = df.get('POSIX_BYTES_READ', 0) + df.get('STDIO_BYTES_READ', 0)
    writes = df.get('POSIX_BYTES_WRITTEN', 0) + df.get('STDIO_BYTES_WRITTEN', 0)
    # Filter to nonzero
    mask = (reads > 0) | (writes > 0)
    r = reads[mask]
    w = writes[mask]

    fig, ax = plt.subplots()
    # Sample for plotting if too many points
    if len(r) > 50000:
        idx = np.random.RandomState(42).choice(len(r), 50000, replace=False)
        r = r.iloc[idx]
        w = w.iloc[idx]
    ax.scatter(r + 1, w + 1, alpha=0.05, s=1, c='#1976D2')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Total Bytes Read')
    ax.set_ylabel('Total Bytes Written')
    ax.set_title('Read vs Write Volume')
    # Diagonal line (equal read/write)
    lims = [1, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'r--', alpha=0.3, label='Equal R/W')
    ax.legend()
    fig.savefig(fig_dir / 'read_write_balance.png', bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved read_write_balance.png")


def plot_correlation_heatmap(df, fig_dir):
    """Fig: Feature correlation heatmap (top features only)."""
    # Select a subset of important features to keep readable
    key_features = [
        'nprocs', 'runtime_seconds',
        'POSIX_BYTES_READ', 'POSIX_BYTES_WRITTEN', 'POSIX_READS', 'POSIX_WRITES',
        'POSIX_OPENS', 'POSIX_SEEKS', 'POSIX_STATS',
        'POSIX_SEQ_READS', 'POSIX_SEQ_WRITES',
        'POSIX_F_READ_TIME', 'POSIX_F_WRITE_TIME', 'POSIX_F_META_TIME',
        'MPIIO_BYTES_READ', 'MPIIO_BYTES_WRITTEN',
        'STDIO_BYTES_READ', 'STDIO_BYTES_WRITTEN',
        'has_posix', 'has_mpiio', 'has_stdio',
        'num_files',
    ]
    available = [c for c in key_features if c in df.columns]
    if len(available) < 5:
        return

    corr = df[available].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(available, rotation=90, fontsize=7)
    ax.set_yticklabels(available, fontsize=7)
    ax.set_title('Feature Correlation Matrix (Key Features)')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(fig_dir / 'correlation_heatmap.png', bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Dataset characterization analysis for IOSage paper')
    parser.add_argument('--input', required=True,
                        help='Input parquet file (raw_features.parquet)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory (SC_Draft_Paper)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample N rows for quick iteration')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        stream=sys.stdout)

    output_dir = Path(args.output_dir)
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading %s...", args.input)
    df = pd.read_parquet(args.input)
    logger.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42)
        logger.info("Sampled %d rows for quick analysis", len(df))

    # --- Compute statistics ---
    logger.info("Computing basic statistics...")
    stats = compute_basic_stats(df)

    logger.info("Computing feature sparsity...")
    sparsity = compute_sparsity(df)
    stats['n_features_gt95pct_zero'] = sum(1 for v in sparsity.values() if v > 0.95)
    stats['n_features_gt50pct_zero'] = sum(1 for v in sparsity.values() if v > 0.50)
    stats['n_features_lt10pct_zero'] = sum(1 for v in sparsity.values() if v < 0.10)

    logger.info("Classifying workload types...")
    workload_labels = classify_workload_type(df)
    stats['workload_types'] = {
        str(k): int(v) for k, v in workload_labels.value_counts().items()
    }

    # --- Save stats ---
    stats_path = output_dir / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Saved stats to %s", stats_path)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("  DATASET CHARACTERIZATION SUMMARY")
    print("=" * 70)
    print(f"  Total rows:       {stats['total_rows']:,}")
    print(f"  Unique job IDs:   {stats['unique_jobids']:,}")
    print(f"  Date range:       {stats.get('date_min', '?')} to {stats.get('date_max', '?')}")
    print(f"  Features:         {stats['feature_cols']} + {stats['info_cols']} info")
    print()
    print(f"  nprocs median:    {stats.get('nprocs_median', '?')}")
    print(f"  nprocs max:       {stats.get('nprocs_max', '?')}")
    print(f"  Single-proc jobs: {stats.get('single_proc_pct', '?'):.1f}%")
    print()
    print(f"  Runtime median:   {stats.get('runtime_median_min', '?'):.1f} min")
    print(f"  Runtime max:      {stats.get('runtime_max_hr', '?'):.1f} hours")
    print()
    print(f"  Total I/O volume: {stats.get('total_io_tb', '?'):.1f} TB")
    print(f"  Zero-I/O logs:    {stats.get('zero_io_pct', '?'):.1f}%")
    print()
    print(f"  Has POSIX:        {stats.get('has_posix_pct', '?'):.1f}%")
    print(f"  Has MPI-IO:       {stats.get('has_mpiio_pct', '?'):.1f}%")
    print(f"  Has STDIO:        {stats.get('has_stdio_pct', '?'):.1f}%")
    print()
    print(f"  Sparse features (>95% zero): {stats.get('n_features_gt95pct_zero', '?')}")
    print(f"  Dense features (<10% zero):  {stats.get('n_features_lt10pct_zero', '?')}")
    print()
    print("  Workload types (heuristic):")
    for wt, count in sorted(stats.get('workload_types', {}).items(),
                            key=lambda x: -x[1]):
        pct = count / stats['total_rows'] * 100
        print(f"    {wt:20s}: {count:>10,} ({pct:.1f}%)")
    print("=" * 70)

    # --- Generate figures ---
    logger.info("Generating figures...")
    plot_temporal_distribution(df, fig_dir)
    plot_module_distribution(df, fig_dir)
    plot_nprocs_distribution(df, fig_dir)
    plot_io_volume_distribution(df, fig_dir)
    plot_runtime_distribution(df, fig_dir)
    plot_sparsity(sparsity, fig_dir)
    plot_workload_types(workload_labels, fig_dir)
    plot_read_write_balance(df, fig_dir)
    plot_correlation_heatmap(df, fig_dir)

    logger.info("All figures saved to %s", fig_dir)
    logger.info("Done! Update SC_Draft_Paper/paper_materials.md with stats.json values.")


if __name__ == '__main__':
    main()
