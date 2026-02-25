"""
Data Preprocessing Pipeline
============================
Cleans, validates, normalizes, and splits the extracted feature data.

Steps:
  1. Filter invalid logs (zero I/O, negative times, short runtime)
  2. Handle missing modules (zero-fill)
  3. Remove sparse features (>95% zeros)
  4. Normalize with StandardScaler
  5. Create train/val/test splits
"""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.feature_extraction import get_feature_names, get_info_columns

logger = logging.getLogger(__name__)


def filter_valid_logs(df, min_total_ops=1, min_runtime=1.0):
    """Remove invalid Darshan logs.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe (one row per job).
    min_total_ops : int
        Minimum total I/O operations (reads + writes).
    min_runtime : float
        Minimum runtime in seconds.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    n_before = len(df)

    # Jobs must have at least some I/O
    if 'POSIX_READS' in df.columns and 'POSIX_WRITES' in df.columns:
        mask_io = (df['POSIX_READS'] + df['POSIX_WRITES']) >= min_total_ops
        df = df[mask_io]

    # Valid timing (non-negative)
    for col in ['POSIX_F_READ_TIME', 'POSIX_F_WRITE_TIME']:
        if col in df.columns:
            df = df[df[col] >= 0]

    # Minimum runtime
    if 'runtime_seconds' in df.columns:
        df = df[df['runtime_seconds'] >= min_runtime]

    n_after = len(df)
    logger.info("Filtered: %d -> %d valid logs (removed %d)",
                n_before, n_after, n_before - n_after)
    return df


def replace_sentinels(df, sentinel=-1, replacement=0.0):
    """Replace Darshan sentinel values with a default.

    Darshan uses -1 for counters that are not available (e.g., rank info
    for single-rank jobs).
    """
    feature_cols = [c for c in df.columns if not c.startswith('_')]
    df[feature_cols] = df[feature_cols].replace(sentinel, replacement)
    return df


def remove_sparse_features(df, max_zero_fraction=0.95, exclude_prefixes=('_',)):
    """Remove features where more than max_zero_fraction values are zero.

    Parameters
    ----------
    df : pd.DataFrame
    max_zero_fraction : float
        Features with > this fraction of zeros are dropped.
    exclude_prefixes : tuple
        Column name prefixes to never drop.

    Returns
    -------
    pd.DataFrame
        Dataframe with sparse features removed.
    list
        Names of removed columns.
    """
    candidates = [c for c in df.columns
                  if not any(c.startswith(p) for p in exclude_prefixes)]
    removed = []
    for col in candidates:
        zero_frac = (df[col] == 0).mean()
        if zero_frac > max_zero_fraction:
            removed.append(col)

    if removed:
        logger.info("Removing %d sparse features (>%.0f%% zeros): %s",
                     len(removed), max_zero_fraction * 100,
                     removed[:10])
        df = df.drop(columns=removed)

    return df, removed


def normalize_features(df, feature_cols=None, scaler=None):
    """Apply StandardScaler normalization.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list, optional
        Columns to normalize.  If None, all non-_ columns are used.
    scaler : StandardScaler, optional
        Pre-fitted scaler for inference.  If None, a new one is fitted.

    Returns
    -------
    pd.DataFrame
        Normalized dataframe.
    StandardScaler
        Fitted scaler (save for inference).
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if not c.startswith('_')]

    if scaler is None:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        logger.info("Fitted StandardScaler on %d features", len(feature_cols))
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
        logger.info("Applied pre-fitted StandardScaler on %d features", len(feature_cols))

    return df, scaler


def create_splits(df, labels_df, n_folds=5, test_fraction=0.2, seed=42):
    """Create train/val/test splits for cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.
    labels_df : pd.DataFrame
        Label dataframe (same index as df). Columns = dimension names.
    n_folds : int
        Number of CV folds.
    test_fraction : float
        Fraction held out as final test set.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{'test_idx': array, 'folds': [(train_idx, val_idx), ...]}``
    """
    rng = np.random.RandomState(seed)
    n = len(df)
    indices = np.arange(n)
    rng.shuffle(indices)

    # Hold-out test set
    n_test = int(n * test_fraction)
    test_idx = indices[:n_test]
    train_val_idx = indices[n_test:]

    # Stratify on the most common label combination (first dimension)
    if labels_df is not None and len(labels_df.columns) > 0:
        y_strat = labels_df.iloc[train_val_idx, 0].values
    else:
        y_strat = np.zeros(len(train_val_idx))

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_fold_idx, val_fold_idx in kf.split(train_val_idx, y_strat):
        folds.append((train_val_idx[train_fold_idx], train_val_idx[val_fold_idx]))

    logger.info("Splits: %d test, %d train/val (%d folds)",
                len(test_idx), len(train_val_idx), n_folds)
    return {'test_idx': test_idx, 'folds': folds}


def full_preprocessing_pipeline(df, config=None):
    """Run the complete preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw feature dataframe from batch extraction.
    config : dict, optional
        Configuration from feature_extraction.yaml.

    Returns
    -------
    pd.DataFrame
        Cleaned, filtered dataframe (NOT normalized — normalization happens
        during training to avoid data leakage from test set).
    list
        Names of removed sparse features.
    """
    if config is None:
        config = {}

    filtering = config.get('filtering', {})
    min_ops = filtering.get('min_total_ops', 1)
    min_runtime = filtering.get('min_runtime_seconds', 1.0)
    max_zero = filtering.get('max_zero_fraction', 0.95)
    sentinel = filtering.get('sentinel_value', -1)
    replacement = filtering.get('replacement_value', 0)

    logger.info("Starting preprocessing pipeline on %d rows", len(df))

    # Step 1: Replace sentinels
    df = replace_sentinels(df, sentinel=sentinel, replacement=replacement)

    # Step 2: Filter invalid logs
    df = filter_valid_logs(df, min_total_ops=min_ops, min_runtime=min_runtime)

    # Step 3: Remove sparse features
    df, removed = remove_sparse_features(df, max_zero_fraction=max_zero)

    logger.info("Preprocessing complete: %d rows, %d columns",
                len(df), len(df.columns))
    return df, removed
