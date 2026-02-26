"""
Data Preprocessing Pipeline
============================
Multi-stage pipeline for cleaning, engineering, analyzing, and normalizing
features extracted from Darshan logs.

Pipeline stages (each saves an intermediate parquet):
  Stage 1: Raw extraction (batch_extract.py --raw) -> raw_features.parquet
  Stage 2: Cleaning -> cleaned_features.parquet
  Stage 3: Feature engineering -> engineered_features.parquet
  Stage 4: Statistical analysis (EDA) -> stats report (no parquet)
  Stage 5: Normalization -> normalized_features.parquet

Design principles:
  - NO hardcoded feature exclusions.  All exclusion decisions are driven by
    EDA (Stage 4) and configured in preprocessing.yaml.
  - Group-specific normalization: different transforms for different counter
    types (volume, count, histogram, timing, ratio, indicator).
  - Save intermediate stages: Stage 1 is immutable ground truth; Stage 2 is
    the cleaned baseline for alternative normalization experiments.
  - Fit normalizers on training data only to prevent data leakage.
"""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.feature_extraction import (
    FEATURE_GROUPS,
    _EPS,
    _SENTINEL,
    _compute_derived_features,
    get_info_columns,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 2: Cleaning
# ---------------------------------------------------------------------------

def stage2_clean(df, config):
    """Clean raw features: filter invalid jobs, handle sentinels, add indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Raw features from Stage 1 (``raw_features.parquet``).
    config : dict
        Preprocessing configuration (from ``configs/preprocessing.yaml``).

    Returns
    -------
    pd.DataFrame
        Cleaned features.
    dict
        Cleaning report (counts of removed/modified rows).
    """
    cleaning = config.get('cleaning', {})
    report = {'initial_rows': len(df)}

    # --- Filter: require POSIX module ---
    if cleaning.get('require_posix', True) and 'has_posix' in df.columns:
        mask = df['has_posix'] == 1
        n_dropped = (~mask).sum()
        if n_dropped > 0:
            logger.info("Removed %d jobs without POSIX module", n_dropped)
            df = df[mask].copy()
    report['after_require_posix'] = len(df)

    # --- Filter: minimum duration ---
    min_duration = cleaning.get('min_duration_seconds', 10)
    if 'runtime_seconds' in df.columns:
        mask = df['runtime_seconds'] >= min_duration
        n_dropped = (~mask).sum()
        logger.info("Removed %d jobs with runtime < %d seconds",
                     n_dropped, min_duration)
        df = df[mask].copy()
    report['after_min_duration'] = len(df)

    # --- Filter: minimum total bytes ---
    min_bytes = cleaning.get('min_total_bytes', 4096)
    if 'POSIX_BYTES_READ' in df.columns and 'POSIX_BYTES_WRITTEN' in df.columns:
        total_bytes = df['POSIX_BYTES_READ'] + df['POSIX_BYTES_WRITTEN']
        mask = total_bytes >= min_bytes
        n_dropped = (~mask).sum()
        logger.info("Removed %d jobs with total bytes < %d", n_dropped, min_bytes)
        df = df[mask].copy()
    report['after_min_bytes'] = len(df)

    # --- Filter: minimum I/O operations ---
    min_ops = cleaning.get('min_io_ops', 2)
    if 'POSIX_READS' in df.columns and 'POSIX_WRITES' in df.columns:
        total_ops = df['POSIX_READS'] + df['POSIX_WRITES']
        mask = total_ops >= min_ops
        n_dropped = (~mask).sum()
        logger.info("Removed %d jobs with total ops < %d", n_dropped, min_ops)
        df = df[mask].copy()
    report['after_min_ops'] = len(df)

    # --- Filter: non-negative timing ---
    for col in ['POSIX_F_READ_TIME', 'POSIX_F_WRITE_TIME', 'POSIX_F_META_TIME']:
        if col in df.columns:
            mask = df[col] >= 0
            n_dropped = (~mask).sum()
            if n_dropped > 0:
                logger.info("Removed %d jobs with negative %s", n_dropped, col)
                df = df[mask].copy()

    report['after_timing_filter'] = len(df)

    # --- Handle sentinel values ---
    sentinel_cfg = config.get('sentinel_handling', {})
    rank_replacement = sentinel_cfg.get('replace_negative_rank_with', 0)

    # Integer sentinel -1 replacement for rank-related counters
    rank_int_cols = [
        'POSIX_FASTEST_RANK', 'POSIX_FASTEST_RANK_BYTES',
        'POSIX_SLOWEST_RANK', 'POSIX_SLOWEST_RANK_BYTES',
    ]
    for col in rank_int_cols:
        if col in df.columns:
            mask = df[col] == _SENTINEL
            if mask.any():
                df.loc[mask, col] = rank_replacement

    # Float sentinel 0.0 for rank timing (already 0 for non-shared, keep as is)

    # MMAPS sentinel -1 (overflow clamp)
    if 'POSIX_MMAPS' in df.columns:
        mask = df['POSIX_MMAPS'] == _SENTINEL
        if mask.any():
            df.loc[mask, 'POSIX_MMAPS'] = 0

    report['final_rows'] = len(df)
    report['rows_removed'] = report['initial_rows'] - report['final_rows']
    report['removal_pct'] = (
        100 * report['rows_removed'] / max(report['initial_rows'], 1)
    )

    logger.info(
        "Cleaning complete: %d -> %d rows (removed %d, %.1f%%)",
        report['initial_rows'], report['final_rows'],
        report['rows_removed'], report['removal_pct']
    )
    return df, report


# ---------------------------------------------------------------------------
# Stage 3: Feature Engineering
# ---------------------------------------------------------------------------

def stage3_engineer(df):
    """Compute derived features from cleaned raw counters.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned features from Stage 2.

    Returns
    -------
    pd.DataFrame
        Features with derived columns added.
    """
    logger.info("Computing derived features for %d rows...", len(df))

    info_cols = [c for c in df.columns if c.startswith('_')]

    # Process each row to compute derived features
    derived_rows = []
    for idx, row in df.iterrows():
        f = row.to_dict()
        modules = str(f.get('_modules', '')).split(',')

        # Replace sentinels for computation
        for key in list(f.keys()):
            if not key.startswith('_') and f[key] == _SENTINEL:
                f[key] = 0.0

        _compute_derived_features(f, modules)
        derived_rows.append(f)

    result = pd.DataFrame(derived_rows, index=df.index)
    n_new = len(result.columns) - len(df.columns)
    logger.info("Added %d derived features (total: %d columns)",
                 n_new, len(result.columns))
    return result


# ---------------------------------------------------------------------------
# Stage 4: Statistical Analysis (EDA)
# ---------------------------------------------------------------------------

def compute_statistics(df):
    """Compute comprehensive statistics for EDA.

    Run this on Stage 2 (cleaned) or Stage 3 (engineered) features to
    inform decisions about feature exclusion and normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe (cleaned or engineered).

    Returns
    -------
    pd.DataFrame
        Statistics per feature: count, mean, std, min, max, percentiles,
        skewness, kurtosis, zero_fraction, negative_fraction.
    """
    feature_cols = [c for c in df.columns if not c.startswith('_')]
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])

    stats = pd.DataFrame(index=numeric_df.columns)
    stats['count'] = numeric_df.count()
    stats['mean'] = numeric_df.mean()
    stats['std'] = numeric_df.std()
    stats['min'] = numeric_df.min()
    stats['p01'] = numeric_df.quantile(0.01)
    stats['p05'] = numeric_df.quantile(0.05)
    stats['p25'] = numeric_df.quantile(0.25)
    stats['median'] = numeric_df.median()
    stats['p75'] = numeric_df.quantile(0.75)
    stats['p95'] = numeric_df.quantile(0.95)
    stats['p99'] = numeric_df.quantile(0.99)
    stats['max'] = numeric_df.max()
    stats['skewness'] = numeric_df.skew()
    stats['kurtosis'] = numeric_df.kurtosis()
    stats['zero_fraction'] = (numeric_df == 0).mean()
    stats['negative_fraction'] = (numeric_df < 0).mean()
    stats['nonzero_count'] = (numeric_df != 0).sum()

    # Identify feature group for each column
    group_map = {}
    for group_name, cols in FEATURE_GROUPS.items():
        for col in cols:
            group_map[col] = group_name
    stats['feature_group'] = stats.index.map(
        lambda x: group_map.get(x, 'other')
    )

    logger.info("Computed statistics for %d features", len(stats))
    return stats


def compute_correlation_matrix(df, method='spearman'):
    """Compute feature correlation matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.
    method : str
        Correlation method ('spearman' recommended for heavy-tailed data).

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    feature_cols = [c for c in df.columns if not c.startswith('_')]
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])
    return numeric_df.corr(method=method)


def find_redundant_features(corr_matrix, threshold=0.90):
    """Find pairs of features with correlation above threshold.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix from ``compute_correlation_matrix``.
    threshold : float
        Absolute correlation threshold.

    Returns
    -------
    list[tuple]
        List of (feature_a, feature_b, correlation) tuples.
    """
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val >= threshold:
                pairs.append((cols[i], cols[j], corr_val))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Stage 5: Normalization
# ---------------------------------------------------------------------------

def stage5_normalize(df, config, fit=True, scalers=None):
    """Apply group-specific normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered features (Stage 3 output).
    config : dict
        Preprocessing configuration with normalization settings.
    fit : bool
        If True, fit scalers on this data (training set).
        If False, use pre-fitted scalers (validation/test set).
    scalers : dict, optional
        Pre-fitted scalers keyed by group name.  Required when fit=False.

    Returns
    -------
    pd.DataFrame
        Normalized features.
    dict
        Fitted scalers (save for inference on validation/test).
    """
    from sklearn.preprocessing import RobustScaler

    norm_config = config.get('normalization', {})
    if scalers is None:
        scalers = {}

    df = df.copy()
    feature_cols = [c for c in df.columns if not c.startswith('_')]

    # Build column-to-group mapping
    col_to_group = {}
    for group_name, cols in FEATURE_GROUPS.items():
        for col in cols:
            if col in feature_cols:
                col_to_group[col] = group_name

    # Get normalization method per group from config
    group_methods = {
        'volume': norm_config.get('volume_counters', 'log1p_robust'),
        'count': norm_config.get('count_counters', 'log1p_robust'),
        'histogram': norm_config.get('histogram_counters', 'log1p'),
        'top4': norm_config.get('top4_counters', 'log1p'),
        'timing': norm_config.get('timing_counters', 'log1p_robust'),
        'timestamp': norm_config.get('timestamp_counters', 'none'),
        'categorical': norm_config.get('categorical_counters', 'none'),
        'rank_id': norm_config.get('rank_id_counters', 'none'),
        'conditional_size': norm_config.get('conditional_size_counters', 'log1p'),
        'indicator': norm_config.get('indicator_features', 'none'),
        'ratio': norm_config.get('ratio_features', 'none'),
        'derived_absolute': norm_config.get('derived_absolute', 'log1p'),
        'metadata': norm_config.get('metadata_features', 'log1p'),
    }

    # Apply group-specific normalization
    for group_name, method in group_methods.items():
        cols = [c for c in FEATURE_GROUPS.get(group_name, [])
                if c in df.columns]
        if not cols or method == 'none':
            continue

        if method == 'log1p':
            for col in cols:
                df[col] = df[col].apply(lambda x: math.log1p(max(x, 0)))

        elif method == 'log1p_robust':
            # Step 1: log1p transform
            for col in cols:
                df[col] = df[col].apply(lambda x: math.log1p(max(x, 0)))

            # Step 2: RobustScaler (median + IQR)
            if fit:
                scaler = RobustScaler()
                df[cols] = scaler.fit_transform(df[cols])
                scalers[group_name] = scaler
                logger.info("Fitted RobustScaler for %s (%d features)",
                            group_name, len(cols))
            else:
                scaler = scalers.get(group_name)
                if scaler is not None:
                    df[cols] = scaler.transform(df[cols])
                else:
                    logger.warning("No pre-fitted scaler for %s", group_name)

        elif method == 'log10p1':
            # Legacy: log10(x+1) for backward compatibility
            for col in cols:
                df[col] = df[col].apply(lambda x: math.log10(max(x, 0) + 1))

    logger.info("Normalization complete: %d columns", len(feature_cols))
    return df, scalers


# ---------------------------------------------------------------------------
# Data Splits
# ---------------------------------------------------------------------------

def create_splits(df, config, labels_df=None):
    """Create train/val/test splits.

    Supports both random and temporal splits.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.
    config : dict
        Preprocessing configuration with split settings.
    labels_df : pd.DataFrame, optional
        Label dataframe for stratification.

    Returns
    -------
    dict
        Split indices: ``{'train_idx': array, 'val_idx': array,
        'test_idx': array}`` for simple split, or includes ``'folds'``
        for cross-validation.
    """
    split_config = config.get('splits', {})
    method = split_config.get('method', 'temporal')
    test_fraction = split_config.get('test_fraction', 0.15)
    val_fraction = split_config.get('val_fraction', 0.15)
    seed = config.get('random_seed', 42)

    n = len(df)

    if method == 'temporal' and '_start_time' in df.columns:
        # Sort by start time, split chronologically
        sorted_idx = df['_start_time'].sort_values().index
        n_test = int(n * test_fraction)
        n_val = int(n * val_fraction)

        test_idx = sorted_idx[-n_test:].values
        val_idx = sorted_idx[-(n_test + n_val):-n_test].values
        train_idx = sorted_idx[:-(n_test + n_val)].values

        logger.info(
            "Temporal split: train=%d, val=%d, test=%d",
            len(train_idx), len(val_idx), len(test_idx)
        )
    else:
        # Random split
        rng = np.random.RandomState(seed)
        indices = np.arange(n)
        rng.shuffle(indices)

        n_test = int(n * test_fraction)
        n_val = int(n * val_fraction)
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]

        logger.info(
            "Random split (seed=%d): train=%d, val=%d, test=%d",
            seed, len(train_idx), len(val_idx), len(test_idx)
        )

    return {
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
    }


# ---------------------------------------------------------------------------
# Sparse feature detection (for EDA, not automatic removal)
# ---------------------------------------------------------------------------

def find_sparse_features(df, max_zero_fraction=0.99):
    """Identify features with high zero fraction (candidates for removal).

    This is an analysis function for EDA, not an automatic removal step.
    The final exclusion decision should be made after reviewing statistics
    and domain relevance.

    Parameters
    ----------
    df : pd.DataFrame
    max_zero_fraction : float
        Features with > this fraction of zeros are flagged.

    Returns
    -------
    list[tuple]
        List of (column_name, zero_fraction) for flagged features.
    """
    feature_cols = [c for c in df.columns if not c.startswith('_')]
    sparse = []
    for col in feature_cols:
        if col in df.columns:
            zero_frac = (df[col] == 0).mean()
            if zero_frac > max_zero_fraction:
                sparse.append((col, zero_frac))
    sparse.sort(key=lambda x: x[1], reverse=True)
    return sparse


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------

def load_preprocessing_config(config_path=None):
    """Load preprocessing configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to preprocessing.yaml.  Defaults to
        ``configs/preprocessing.yaml`` relative to project root.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / 'configs' / 'preprocessing.yaml'
        )
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("Config not found: %s, using defaults", config_path)
        return _default_config()
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def _default_config():
    """Return default preprocessing configuration."""
    return {
        'cleaning': {
            'min_duration_seconds': 10,
            'min_total_bytes': 4096,
            'min_io_ops': 2,
            'require_posix': True,
        },
        'sentinel_handling': {
            'replace_negative_rank_with': 0,
        },
        'normalization': {
            'volume_counters': 'log1p_robust',
            'count_counters': 'log1p_robust',
            'histogram_counters': 'log1p',
            'top4_counters': 'log1p',
            'timing_counters': 'log1p_robust',
            'timestamp_counters': 'none',
            'categorical_counters': 'none',
            'rank_id_counters': 'none',
            'conditional_size_counters': 'log1p',
            'indicator_features': 'none',
            'ratio_features': 'none',
            'derived_absolute': 'log1p',
            'metadata_features': 'log1p',
        },
        'splits': {
            'method': 'temporal',
            'test_fraction': 0.15,
            'val_fraction': 0.15,
        },
        'feature_selection': {
            'correlation_threshold': 0.90,
            'min_nonzero_fraction': 0.01,
        },
        'random_seed': 42,
    }
