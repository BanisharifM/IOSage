"""
Generate Paper-Ready Figures for SC 2026 Dataset/Preprocessing Section
======================================================================
Produces publication-quality figures for the SC 2026 paper.

Output directory: figures/preprocessing/
All figures are saved as both PDF (for LaTeX) and PNG (for review).

Figure inventory:
  1. fig_data_characterization.pdf  — Multi-panel (3): I/O volume, sparsity, normalization
  2. fig_temporal_split.pdf         — Timeline with train/val/test regions
  3. fig_cleaning_funnel.pdf        — Cleaning pipeline funnel chart
  4. fig_feature_correlation.pdf    — Top-feature correlation heatmap
  5. fig_normalization_effect.pdf   — Before/after normalization for key features

Usage:
    python scripts/generate_paper_figures.py
    python scripts/generate_paper_figures.py --figures 1 3 5    # specific figures
    python scripts/generate_paper_figures.py --output-dir results/figures

Modify STYLE_CONFIG and individual plot functions to adjust appearance.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ===========================================================================
# Style Configuration — modify here to change appearance globally
# ===========================================================================
STYLE_CONFIG = {
    # Figure sizes (width, height) in inches — IEEE column is 3.5in, double is 7in
    'single_col': (3.5, 2.8),
    'double_col': (7.0, 2.8),
    'double_col_tall': (7.0, 4.5),
    'triple_panel': (7.0, 2.4),

    # Font sizes
    'font_size': 8,
    'title_size': 9,
    'label_size': 8,
    'tick_size': 7,
    'legend_size': 7,
    'annotation_size': 6.5,

    # Colors — colorblind-friendly palette (Okabe-Ito)
    'palette': {
        'train': '#0072B2',     # blue
        'val': '#E69F00',       # orange
        'test': '#D55E00',      # vermilion
        'removed': '#BBBBBB',   # gray
        'kept': '#009E73',      # green
        'highlight': '#CC79A7', # pink
        'primary': '#0072B2',
        'secondary': '#E69F00',
        'tertiary': '#D55E00',
    },

    # Grid and spine
    'grid_alpha': 0.3,
    'grid_style': '--',
    'spine_width': 0.5,

    # DPI for raster
    'dpi': 300,
}

# Feature groups and their normalization methods (from preprocessing.yaml)
NORMALIZATION_MAP = {
    'volume': 'log1p + RobustScaler',
    'count': 'log1p + RobustScaler',
    'histogram': 'log1p',
    'top4': 'log1p',
    'timing': 'log1p + RobustScaler',
    'timestamp': 'none',
    'categorical': 'none',
    'rank_id': 'none',
    'conditional_size': 'log1p',
    'indicator': 'none',
    'ratio': 'none',
    'ratio_unbounded': 'log1p',
    'derived_absolute': 'log1p',
    'metadata': 'log1p',
}

# ===========================================================================
# Data Loading
# ===========================================================================

def load_data(data_dir):
    """Load all required data files."""
    data_dir = Path(data_dir)
    data = {}

    logger.info("Loading data files...")
    data['raw'] = pd.read_parquet(data_dir / 'production/raw_features.parquet')
    data['engineered'] = pd.read_parquet(data_dir / 'production/features.parquet')
    data['train'] = pd.read_parquet(data_dir / 'splits' / 'train.parquet')
    data['val'] = pd.read_parquet(data_dir / 'splits' / 'val.parquet')
    data['test'] = pd.read_parquet(data_dir / 'splits' / 'test.parquet')
    data['eda_stats'] = pd.read_parquet(data_dir / 'production/eda/stats.parquet')

    for name, df in data.items():
        logger.info(f"  {name}: {df.shape[0]:,} rows x {df.shape[1]} cols")

    return data


def apply_style():
    """Apply publication style to matplotlib."""
    cfg = STYLE_CONFIG
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': cfg['font_size'],
        'axes.titlesize': cfg['title_size'],
        'axes.labelsize': cfg['label_size'],
        'xtick.labelsize': cfg['tick_size'],
        'ytick.labelsize': cfg['tick_size'],
        'legend.fontsize': cfg['legend_size'],
        'figure.dpi': cfg['dpi'],
        'savefig.dpi': cfg['dpi'],
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'axes.grid': True,
        'grid.alpha': cfg['grid_alpha'],
        'grid.linestyle': cfg['grid_style'],
        'axes.linewidth': cfg['spine_width'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,   # TrueType fonts in PDF (required by IEEE)
        'ps.fonttype': 42,
    })


def save_figure(fig, output_dir, name):
    """Save figure as PDF and PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f'{name}.pdf'
    png_path = output_dir / f'{name}.png'

    fig.savefig(pdf_path, format='pdf')
    fig.savefig(png_path, format='png')
    plt.close(fig)
    logger.info(f"  Saved: {pdf_path}")


# ===========================================================================
# Figure 1: Multi-Panel Data Characterization
# ===========================================================================

def fig_data_characterization(data, output_dir):
    """Three-panel figure: I/O volume distribution, feature sparsity, normalization effect."""
    logger.info("Generating Figure 1: Data Characterization (3 panels)...")
    cfg = STYLE_CONFIG

    fig, axes = plt.subplots(1, 3, figsize=cfg['double_col_tall'])

    df = data['engineered']
    eda = data['eda_stats']
    train_norm = data['train']

    # --- Panel (a): I/O Volume Distribution ---
    ax = axes[0]
    total_bytes = df['POSIX_BYTES_READ'] + df['POSIX_BYTES_WRITTEN']
    # Convert to log10 scale, handle zeros
    log_bytes = np.log10(total_bytes.clip(lower=1))
    ax.hist(log_bytes, bins=80, color=cfg['palette']['primary'], alpha=0.85,
            edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Total I/O Volume (log$_{10}$ bytes)')
    ax.set_ylabel('Job Count')
    ax.set_title('(a) I/O Volume Distribution', fontsize=cfg['title_size'])

    # Add human-readable labels
    byte_labels = {3: '1 KB', 6: '1 MB', 9: '1 GB', 12: '1 TB'}
    for pos, label in byte_labels.items():
        if pos >= log_bytes.min() and pos <= log_bytes.max():
            ax.axvline(pos, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
            ax.text(pos, ax.get_ylim()[1] * 0.92, label,
                    ha='center', fontsize=cfg['annotation_size'],
                    color='gray', style='italic')

    # --- Panel (b): Feature Sparsity Curve ---
    ax = axes[1]
    # Only use non-metadata, non-info features
    feature_cols = [c for c in eda.index
                    if not c.startswith('_')]
    zero_fracs = eda.loc[feature_cols, 'zero_fraction'].sort_values(ascending=False)

    x_range = np.arange(len(zero_fracs))
    ax.fill_between(x_range, zero_fracs.values, alpha=0.3,
                    color=cfg['palette']['primary'])
    ax.plot(x_range, zero_fracs.values, color=cfg['palette']['primary'],
            linewidth=1.0)

    # Threshold lines
    ax.axhline(0.99, color=cfg['palette']['tertiary'], linestyle='--',
               linewidth=0.8, alpha=0.8)
    ax.axhline(0.95, color=cfg['palette']['secondary'], linestyle='--',
               linewidth=0.8, alpha=0.8)

    n_99 = int((zero_fracs > 0.99).sum())
    n_95 = int((zero_fracs > 0.95).sum())
    ax.text(len(zero_fracs) * 0.6, 0.995,
            f'{n_99} features > 99% zero',
            fontsize=cfg['annotation_size'], color=cfg['palette']['tertiary'])
    ax.text(len(zero_fracs) * 0.6, 0.945,
            f'{n_95} features > 95% zero',
            fontsize=cfg['annotation_size'], color=cfg['palette']['secondary'])

    ax.set_xlabel('Feature Index (sorted by sparsity)')
    ax.set_ylabel('Zero Fraction')
    ax.set_title('(b) Feature Sparsity', fontsize=cfg['title_size'])
    ax.set_ylim(-0.05, 1.05)

    # --- Panel (c): Before/After Normalization ---
    ax = axes[2]
    # Pick representative features from different groups
    example_features = ['POSIX_BYTES_WRITTEN', 'POSIX_READS', 'POSIX_F_WRITE_TIME']
    example_labels = ['Bytes Written\n(volume)', 'Read Ops\n(count)', 'Write Time\n(timing)']
    colors = [cfg['palette']['primary'], cfg['palette']['secondary'],
              cfg['palette']['tertiary']]

    # Before: raw values (log1p transform for visualization)
    before_vals = []
    after_vals = []
    for feat in example_features:
        raw = df[feat].values
        norm = train_norm[feat].values
        before_vals.append(np.log1p(raw))
        after_vals.append(norm)

    # Box plots side by side
    positions_before = np.array([1, 4, 7])
    positions_after = np.array([2, 5, 8])

    bp_before = ax.boxplot(
        before_vals, positions=positions_before, widths=0.7,
        patch_artist=True, showfliers=False,
        medianprops=dict(color='black', linewidth=1),
    )
    bp_after = ax.boxplot(
        after_vals, positions=positions_after, widths=0.7,
        patch_artist=True, showfliers=False,
        medianprops=dict(color='black', linewidth=1),
    )

    for i, (box_b, box_a) in enumerate(zip(bp_before['boxes'], bp_after['boxes'])):
        box_b.set_facecolor(colors[i])
        box_b.set_alpha(0.4)
        box_a.set_facecolor(colors[i])
        box_a.set_alpha(0.85)

    ax.set_xticks([1.5, 4.5, 7.5])
    ax.set_xticklabels(example_labels, fontsize=cfg['annotation_size'])
    ax.set_ylabel('Feature Value')
    ax.set_title('(c) Before/After Normalization', fontsize=cfg['title_size'])

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='gray', alpha=0.4, label='Before (log1p)'),
        mpatches.Patch(facecolor='gray', alpha=0.85, label='After (log1p + Robust)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=cfg['annotation_size'], framealpha=0.8)

    fig.tight_layout(w_pad=1.5)
    save_figure(fig, output_dir, 'fig_data_characterization')


# ===========================================================================
# Figure 2: Temporal Split Visualization
# ===========================================================================

def fig_temporal_split(data, output_dir):
    """Timeline showing train/val/test split boundaries with job density."""
    logger.info("Generating Figure 2: Temporal Split...")
    cfg = STYLE_CONFIG

    fig, ax = plt.subplots(figsize=cfg['double_col'])

    df = data['engineered']
    train = data['train']
    val = data['val']
    test = data['test']

    # Convert Unix timestamps to datetime
    train_times = pd.to_datetime(train['_start_time'], unit='s')
    val_times = pd.to_datetime(val['_start_time'], unit='s')
    test_times = pd.to_datetime(test['_start_time'], unit='s')

    # Create weekly bin counts for each split
    all_times = pd.to_datetime(df['_start_time'], unit='s')
    date_range = pd.date_range(all_times.min().normalize(),
                                all_times.max().normalize() + pd.Timedelta(days=7),
                                freq='W')

    train_counts = pd.cut(train_times, bins=date_range).value_counts().sort_index()
    val_counts = pd.cut(val_times, bins=date_range).value_counts().sort_index()
    test_counts = pd.cut(test_times, bins=date_range).value_counts().sort_index()

    # Midpoints for plotting
    midpoints = [interval.mid for interval in train_counts.index]

    # Stacked area plot
    ax.fill_between(midpoints, 0, train_counts.values,
                    color=cfg['palette']['train'], alpha=0.7, label='Train (70%)')
    ax.fill_between(midpoints, 0, val_counts.values,
                    color=cfg['palette']['val'], alpha=0.7, label='Val (15%)')
    ax.fill_between(midpoints, 0, test_counts.values,
                    color=cfg['palette']['test'], alpha=0.7, label='Test (15%)')

    # Add split boundary lines
    train_end = train_times.max()
    val_end = val_times.max()
    ax.axvline(train_end, color='black', linestyle='--', linewidth=0.8, alpha=0.8)
    ax.axvline(val_end, color='black', linestyle='--', linewidth=0.8, alpha=0.8)

    # Annotate boundaries
    y_top = ax.get_ylim()[1] * 0.95
    ax.text(train_end, y_top, f' Train end\n {train_end.strftime("%Y-%m-%d")}',
            fontsize=cfg['annotation_size'], va='top', ha='left')
    ax.text(val_end, y_top, f' Val end\n {val_end.strftime("%Y-%m-%d")}',
            fontsize=cfg['annotation_size'], va='top', ha='left')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    ax.set_xlabel('Date')
    ax.set_ylabel('Jobs per Week')
    ax.set_title('Temporal Data Split', fontsize=cfg['title_size'])
    ax.legend(loc='upper left', fontsize=cfg['legend_size'], framealpha=0.8)

    fig.tight_layout()
    save_figure(fig, output_dir, 'fig_temporal_split')


# ===========================================================================
# Figure 3: Cleaning Pipeline Funnel
# ===========================================================================

def fig_cleaning_funnel(data, output_dir):
    """Horizontal bar chart showing filtering stages and row counts."""
    logger.info("Generating Figure 3: Cleaning Funnel...")
    cfg = STYLE_CONFIG

    fig, ax = plt.subplots(figsize=cfg['single_col'])

    # Cleaning steps with counts (from paper_materials.md Section 1.7.2)
    stages = [
        ('Raw Dataset', 1397216),
        ('Require POSIX', 497327),
        ('Min Duration (1s)', 429835),
        ('Min Bytes (1 KB)', 131151),
    ]

    labels = [s[0] for s in stages]
    counts = [s[1] for s in stages]
    removed = [0] + [stages[i][1] - stages[i+1][1] for i in range(len(stages)-1)]

    y_pos = np.arange(len(stages))[::-1]
    bars = ax.barh(y_pos, counts, height=0.6,
                   color=[cfg['palette']['kept'] if i == len(stages)-1
                          else cfg['palette']['primary'] for i in range(len(stages))],
                   alpha=0.85, edgecolor='white', linewidth=0.5)

    # Add count labels on bars
    for i, (bar, count, rem) in enumerate(zip(bars, counts, removed)):
        # Count inside bar
        x_text = count * 0.5 if count > 200000 else count + 20000
        ha = 'center' if count > 200000 else 'left'
        ax.text(x_text, bar.get_y() + bar.get_height() / 2,
                f'{count:,}', ha=ha, va='center',
                fontsize=cfg['annotation_size'], fontweight='bold',
                color='white' if count > 200000 else 'black')
        # Removed annotation
        if rem > 0:
            pct = rem / stages[0][1] * 100
            ax.text(counts[0] * 1.02, bar.get_y() + bar.get_height() / 2,
                    f'-{rem:,} ({pct:.1f}%)',
                    ha='left', va='center',
                    fontsize=cfg['annotation_size'], color=cfg['palette']['removed'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=cfg['tick_size'])
    ax.set_xlabel('Number of Samples')
    ax.set_title('Data Cleaning Pipeline', fontsize=cfg['title_size'])
    ax.set_xlim(0, counts[0] * 1.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

    fig.tight_layout()
    save_figure(fig, output_dir, 'fig_cleaning_funnel')


# ===========================================================================
# Figure 4: Feature Correlation Heatmap (Top Features)
# ===========================================================================

def fig_feature_correlation(data, output_dir):
    """Correlation heatmap of top-20 most informative features."""
    logger.info("Generating Figure 4: Feature Correlation Heatmap...")
    cfg = STYLE_CONFIG

    train = data['train']

    # Select a representative subset of features (not all 186)
    # Pick features that are most relevant for bottleneck detection
    top_features = [
        'POSIX_BYTES_READ', 'POSIX_BYTES_WRITTEN',
        'POSIX_READS', 'POSIX_WRITES', 'POSIX_OPENS',
        'POSIX_SEQ_READS', 'POSIX_SEQ_WRITES',
        'POSIX_F_READ_TIME', 'POSIX_F_WRITE_TIME', 'POSIX_F_META_TIME',
        'read_ratio', 'seq_read_ratio',
        'small_io_ratio', 'metadata_time_ratio',
        'read_bw_mb_s', 'write_bw_mb_s',
        'rw_ratio', 'opens_per_mb',
        'nprocs', 'runtime_seconds',
    ]
    # Only keep features that exist in the data
    top_features = [f for f in top_features if f in train.columns]

    # Compute Spearman correlation on the train set
    corr = train[top_features].corr(method='spearman')

    fig, ax = plt.subplots(figsize=(cfg['double_col'][0], cfg['double_col'][0] * 0.85))

    # Shorter labels for readability
    short_labels = {
        'POSIX_BYTES_READ': 'Bytes Read',
        'POSIX_BYTES_WRITTEN': 'Bytes Written',
        'POSIX_READS': 'Read Ops',
        'POSIX_WRITES': 'Write Ops',
        'POSIX_OPENS': 'Open Ops',
        'POSIX_SEQ_READS': 'Seq Reads',
        'POSIX_SEQ_WRITES': 'Seq Writes',
        'POSIX_F_READ_TIME': 'Read Time',
        'POSIX_F_WRITE_TIME': 'Write Time',
        'POSIX_F_META_TIME': 'Meta Time',
        'read_ratio': 'Read Ratio',
        'seq_read_ratio': 'Seq Read Ratio',
        'small_io_ratio': 'Small I/O Ratio',
        'metadata_time_ratio': 'Meta Time Ratio',
        'read_bw_mb_s': 'Read BW',
        'write_bw_mb_s': 'Write BW',
        'rw_ratio': 'R/W Ratio',
        'opens_per_mb': 'Opens/MB',
        'nprocs': 'Num Procs',
        'runtime_seconds': 'Runtime',
    }
    display_labels = [short_labels.get(f, f) for f in top_features]

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                xticklabels=display_labels, yticklabels=display_labels,
                cbar_kws={'shrink': 0.8, 'label': 'Spearman rho'},
                ax=ax)

    ax.set_title('Feature Correlation (Spearman, Train Set)',
                 fontsize=cfg['title_size'])
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             fontsize=cfg['annotation_size'])
    plt.setp(ax.get_yticklabels(), rotation=0,
             fontsize=cfg['annotation_size'])

    fig.tight_layout()
    save_figure(fig, output_dir, 'fig_feature_correlation')


# ===========================================================================
# Figure 5: Normalization Effect (Detailed Before/After)
# ===========================================================================

def fig_normalization_effect(data, output_dir):
    """Detailed before/after normalization comparison for 6 representative features."""
    logger.info("Generating Figure 5: Normalization Effect...")
    cfg = STYLE_CONFIG

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0))

    df_raw = data['engineered']
    df_norm = data['train']

    # Representative features from different groups
    features = [
        ('POSIX_BYTES_WRITTEN', 'volume', 'Bytes Written'),
        ('POSIX_READS', 'count', 'Read Operations'),
        ('POSIX_F_WRITE_TIME', 'timing', 'Write Time (s)'),
        ('read_bw_mb_s', 'ratio_unbounded', 'Read Bandwidth'),
        ('POSIX_SIZE_READ_0_100', 'histogram', 'Small Reads (0-100B)'),
        ('read_ratio', 'ratio', 'Read Ratio'),
    ]

    for idx, (feat, group, label) in enumerate(features):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        raw_vals = df_raw[feat].dropna().values
        norm_vals = df_norm[feat].dropna().values

        method = NORMALIZATION_MAP.get(group, 'none')

        if method == 'none':
            # For unmodified features, show raw distribution only
            ax.hist(raw_vals, bins=60, color=cfg['palette']['primary'],
                    alpha=0.85, edgecolor='white', linewidth=0.2, density=True)
            ax.set_title(f'{label}\n({method})', fontsize=cfg['annotation_size'] + 0.5)
        else:
            # Show both raw (log1p for vis) and normalized
            ax.hist(norm_vals, bins=60, color=cfg['palette']['primary'],
                    alpha=0.85, edgecolor='white', linewidth=0.2,
                    density=True, label='Normalized')
            # Overlay raw (log1p transformed) for comparison
            raw_log = np.log1p(raw_vals)
            ax.hist(raw_log, bins=60, color=cfg['palette']['secondary'],
                    alpha=0.35, edgecolor='white', linewidth=0.2,
                    density=True, label='log1p only')
            ax.legend(fontsize=cfg['annotation_size'] - 0.5, framealpha=0.7)
            ax.set_title(f'{label}\n({method})', fontsize=cfg['annotation_size'] + 0.5)

        ax.set_ylabel('Density' if col == 0 else '', fontsize=cfg['annotation_size'])
        ax.tick_params(labelsize=cfg['annotation_size'])

    fig.suptitle('Normalization Effect by Feature Group',
                 fontsize=cfg['title_size'], y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, 'fig_normalization_effect')


# ===========================================================================
# Table Generators (LaTeX output for paper)
# ===========================================================================

def generate_dataset_table(data, output_dir):
    """Generate LaTeX table for dataset summary."""
    logger.info("Generating Table: Dataset Summary (LaTeX)...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = data['raw']
    df_clean = data['engineered']
    eda = data['eda_stats']

    n_raw = len(df_raw)
    n_clean = len(df_clean)
    n_features = len(eda)

    # Count feature groups
    group_counts = eda['feature_group'].value_counts()

    latex = r"""\begin{table}[t]
\centering
\caption{Dataset Summary}
\label{tab:dataset}
\begin{tabular}{lr}
\toprule
\textbf{Property} & \textbf{Value} \\
\midrule
Source system & ALCF Polaris \\
Collection period & Apr 2024 -- Feb 2026 \\
Raw Darshan logs & """ + f"{n_raw:,}" + r""" \\
After cleaning & """ + f"{n_clean:,}" + r""" \\
Retention rate & """ + f"{n_clean/n_raw*100:.1f}" + r"""\% \\
Total features & """ + f"{n_features}" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Feature breakdown:}} \\
~~Raw counters & """ + f"{group_counts.get('volume', 0) + group_counts.get('count', 0) + group_counts.get('histogram', 0) + group_counts.get('top4', 0) + group_counts.get('timing', 0) + group_counts.get('timestamp', 0) + group_counts.get('categorical', 0) + group_counts.get('rank_id', 0) + group_counts.get('conditional_size', 0)}" + r""" \\
~~Indicators & """ + f"{group_counts.get('indicator', 0)}" + r""" \\
~~Derived ratios & """ + f"{group_counts.get('ratio', 0) + group_counts.get('ratio_unbounded', 0) + group_counts.get('derived_absolute', 0)}" + r""" \\
~~Metadata & """ + f"{group_counts.get('metadata', 0)}" + r""" \\
\midrule
Train / Val / Test & """ + f"{len(data['train']):,} / {len(data['val']):,} / {len(data['test']):,}" + r""" \\
Split method & Temporal (70/15/15) \\
\bottomrule
\end{tabular}
\end{table}"""

    table_path = output_dir / 'tab_dataset_summary.tex'
    table_path.write_text(latex)
    logger.info(f"  Saved: {table_path}")


def generate_normalization_table(data, output_dir):
    """Generate LaTeX table for normalization methods per feature group."""
    logger.info("Generating Table: Normalization Methods (LaTeX)...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eda = data['eda_stats']
    group_counts = eda['feature_group'].value_counts()

    rows = []
    for group in ['volume', 'count', 'histogram', 'top4', 'timing',
                   'ratio_unbounded', 'derived_absolute', 'metadata',
                   'conditional_size', 'ratio', 'indicator', 'timestamp',
                   'categorical', 'rank_id']:
        n = group_counts.get(group, 0)
        method = NORMALIZATION_MAP.get(group, 'none')

        # LaTeX-safe method names
        method_tex = method.replace('log1p + RobustScaler', r'$\log(1{+}x)$ + RobustScaler')
        method_tex = method_tex.replace('log1p', r'$\log(1{+}x)$')

        rows.append(f"    {group.replace('_', ' ').title()} & {n} & {method_tex} \\\\")

    latex = r"""\begin{table}[t]
\centering
\caption{Group-Specific Normalization Strategy}
\label{tab:normalization}
\begin{tabular}{lrl}
\toprule
\textbf{Feature Group} & \textbf{\#} & \textbf{Method} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""

    table_path = output_dir / 'tab_normalization.tex'
    table_path.write_text(latex)
    logger.info(f"  Saved: {table_path}")


# ===========================================================================
# Registry of all figures
# ===========================================================================

FIGURE_REGISTRY = {
    1: ('fig_data_characterization', fig_data_characterization),
    2: ('fig_temporal_split', fig_temporal_split),
    3: ('fig_cleaning_funnel', fig_cleaning_funnel),
    4: ('fig_feature_correlation', fig_feature_correlation),
    5: ('fig_normalization_effect', fig_normalization_effect),
}

TABLE_REGISTRY = {
    'dataset': generate_dataset_table,
    'normalization': generate_normalization_table,
}


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate paper-ready figures for SC 2026'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/processed',
        help='Path to processed data directory (default: data/processed)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='paper/figures/preprocessing',
        help='Output directory for figures (default: paper/figures/preprocessing)'
    )
    parser.add_argument(
        '--figures', type=int, nargs='*', default=None,
        help='Specific figure numbers to generate (default: all)'
    )
    parser.add_argument(
        '--tables', action='store_true', default=True,
        help='Also generate LaTeX tables (default: True)'
    )
    parser.add_argument(
        '--no-tables', action='store_false', dest='tables',
        help='Skip LaTeX table generation'
    )
    args = parser.parse_args()

    apply_style()

    data = load_data(args.data_dir)

    # Determine which figures to generate
    fig_ids = args.figures if args.figures else list(FIGURE_REGISTRY.keys())

    for fig_id in fig_ids:
        if fig_id not in FIGURE_REGISTRY:
            logger.warning(f"Unknown figure ID: {fig_id}. Skipping.")
            continue
        name, func = FIGURE_REGISTRY[fig_id]
        func(data, args.output_dir)

    # Generate tables
    if args.tables:
        for name, func in TABLE_REGISTRY.items():
            func(data, args.output_dir)

    logger.info("Done. All figures saved to %s", args.output_dir)


if __name__ == '__main__':
    main()
