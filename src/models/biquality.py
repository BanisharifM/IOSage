"""
Biquality training: one path from the two data sources to a model bundle.

Framework: biquality learning (Nodet et al., Machine Learning 2023). The
production logs carry heuristic (untrusted) labels, the benchmark logs carry
construction (trusted) labels; both share one feature space and one label
set. Benchmark training rows enter training with a higher sample weight,
validation rows guide choices, and test rows are read only for an explicitly
requested final evaluation.

This module owns everything the audit found duplicated or unguarded across
the older entry points: sample alignment by a unique id, split validation,
grouped benchmark partitions, the feature contract, weighting, fitting with
real early stopping, healthy derived from the seven bottleneck decisions,
group bootstrap intervals, and an immutable run directory with a manifest.
The entry point is ``scripts/train_biquality.py``.
"""

import json
import logging
import os
import pickle
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

from src.data.label_rules import BOTTLENECK_DIMENSIONS, DIMENSION_NAMES
from src.data.feature_extraction import FEATURE_SCHEMA_VERSION, get_feature_names, get_raw_feature_names
from src.utils.artifacts import sha256_file

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
BUNDLE_FORMAT = 4
SUPPORTED_MODELS = ('xgboost', 'lightgbm', 'random_forest')


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path):
    """The training configuration; every key the module reads must exist."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"training config not found: {path}")
    with open(path) as fh:
        config = yaml.safe_load(fh)
    for key in ('paths', 'exclude_features', 'benchmark_split', 'biquality',
                'decision_threshold', 'imbalance', 'early_stopping', 'models', 'evaluation'):
        if key not in config:
            raise KeyError(f"training config lacks '{key}'")
    config['_path'] = str(path)
    return config


def _resolve(config, key):
    return PROJECT_DIR / config['paths'][key]


# ---------------------------------------------------------------------------
# Data: alignment by unique id, splits, feature contract
# ---------------------------------------------------------------------------

@dataclass
class ProductionData:
    X: np.ndarray
    y: np.ndarray            # seven bottleneck labels
    valid: np.ndarray        # target validity for incomplete module records
    ids: np.ndarray          # _source_path
    start_time: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    feature_names: list


@dataclass
class BenchmarkData:
    X: np.ndarray
    y: np.ndarray            # seven bottleneck labels
    valid: np.ndarray        # construction labels are valid for every target
    ids: np.ndarray          # "<benchmark>/<job_id>/<log basename>"
    groups: np.ndarray       # "<benchmark>/<job_id>"
    benchmark: np.ndarray    # benchmark type per row
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _check_schema(df, what):
    if '_schema_version' not in df.columns:
        raise ValueError(f"{what} has no _schema_version column; re-extract")
    versions = set(pd.unique(df['_schema_version']))
    if versions != {FEATURE_SCHEMA_VERSION}:
        raise ValueError(f"{what} schema {sorted(versions)}, expected {FEATURE_SCHEMA_VERSION}")


def _label_matrix(labels, what):
    missing = [name for name in DIMENSION_NAMES if name not in labels.columns]
    if missing:
        raise ValueError(f"{what} lack label columns {missing}")
    values = labels[DIMENSION_NAMES]
    if not values.isin([0, 1]).all().all():
        raise ValueError(f"{what} labels must be binary and complete")
    expected_healthy = (values[BOTTLENECK_DIMENSIONS].sum(axis=1) == 0).astype(int)
    if not np.array_equal(values['healthy'].to_numpy(dtype=int), expected_healthy.to_numpy()):
        raise ValueError(f"{what} healthy labels are inconsistent with bottleneck labels")
    return values[BOTTLENECK_DIMENSIONS].to_numpy(dtype=np.float32)


def _label_validity(labels, what, required):
    columns = [f'valid_{dimension}' for dimension in DIMENSION_NAMES]
    present = [column for column in columns if column in labels.columns]
    if not present:
        if required:
            raise ValueError(f"{what} lack per-dimension validity columns")
        return np.ones((len(labels), len(BOTTLENECK_DIMENSIONS)), dtype=bool)
    if len(present) != len(columns):
        raise ValueError(f"{what} have an incomplete validity contract")
    values = labels[columns]
    if not values.isin([0, 1]).all().all():
        raise ValueError(f"{what} validity columns must be binary and complete")
    return values[[f'valid_{dimension}' for dimension in BOTTLENECK_DIMENSIONS]].to_numpy(
        dtype=bool)


def _finite_matrix(frame, columns, what):
    matrix = frame[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{what} contain a non-finite feature value")
    return matrix


def feature_columns(features, config, feature_set='full'):
    """The ordered feature contract: every non-info, non-heuristic column of
    the production features minus the configured exclusions. ``feature_set``
    ``raw`` keeps only the raw counters, indicators and metadata (the
    "no derived features" ablation)."""
    exclude = set(config['exclude_features'])
    unknown = exclude - set(features.columns)
    if unknown:
        raise ValueError(f"exclude_features names columns that do not exist: {sorted(unknown)}")
    names = [c for c in features.columns
             if not c.startswith('_') and not c.startswith('drishti_') and c not in exclude]
    if feature_set == 'raw':
        raw = set(get_raw_feature_names())
        names = [c for c in names if c in raw]
    elif feature_set != 'full':
        raise ValueError(f"unknown feature set {feature_set!r}")
    return names


def _positions_valid(splits, n):
    missing = [key for key in ('train_idx', 'val_idx', 'test_idx') if key not in splits]
    if missing:
        raise ValueError(f"production split lacks {missing}")
    parts = [np.asarray(splits[k]) for k in ('train_idx', 'val_idx', 'test_idx')]
    if any(not np.issubdtype(part.dtype, np.integer) for part in parts):
        raise ValueError("production split positions must be integers")
    joined = np.concatenate(parts)
    if any(len(p) == 0 for p in parts):
        raise ValueError("a production split partition is empty")
    if len(np.unique(joined)) != n or joined.min() != 0 or joined.max() != n - 1:
        raise ValueError("production split positions must be disjoint and cover every row")
    return parts


def _production_groups(features):
    required = ['_uid', '_jobid', '_start_time', '_source_path']
    missing = [name for name in required if name not in features.columns]
    if missing:
        raise ValueError(f"production features lack grouping columns {missing}")
    return [(uid, jobid) if jobid != 0 else ('path', path)
            for uid, jobid, path in zip(
                features['_uid'], features['_jobid'], features['_source_path'])]


def load_production(config, feature_set='full'):
    """Production features, labels and the temporal split, aligned by ``_source_path``.

    The labels file is joined on ``_source_path`` (unique per log; ``_jobid``
    is not, one SLURM job holds many launches). The split positions from
    preprocessing are checked for coverage, disjointness and time order.
    """
    features = pd.read_parquet(_resolve(config, 'production_features'))
    labels = pd.read_parquet(_resolve(config, 'production_labels'))
    _check_schema(features, 'production features')
    missing_features = [name for name in get_feature_names() if name not in features.columns]
    if missing_features:
        raise ValueError(f"production features lack {len(missing_features)} schema columns, "
                         f"first {missing_features[:5]}")
    for name, df in (('features', features), ('labels', labels)):
        if '_source_path' not in df.columns:
            raise ValueError(f"production {name} lack _source_path")
        if df['_source_path'].duplicated().any():
            raise ValueError(f"production {name} have duplicate _source_path")
    labels = labels.set_index('_source_path')
    if not labels.index.equals(pd.Index(features['_source_path'])):
        if set(labels.index) != set(features['_source_path']):
            raise ValueError("production labels and features hold different samples")
        labels = labels.loc[features['_source_path']]

    names = feature_columns(features, config, feature_set)
    with open(_resolve(config, 'production_splits'), 'rb') as fh:
        splits = pickle.load(fh)
    train_idx, val_idx, test_idx = _positions_valid(splits, len(features))
    start = features['_start_time'].to_numpy()
    group_values = _production_groups(features)
    group_sets = [{group_values[pos] for pos in idx} for idx in (train_idx, val_idx, test_idx)]
    if any(group_sets[i] & group_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("a production job appears in more than one split partition")
    group_first = {}
    for group, timestamp in zip(group_values, start):
        group_first[group] = min(group_first.get(group, timestamp), timestamp)
    first_by_part = [np.asarray([group_first[group] for group in groups]) for groups in group_sets]
    if not (first_by_part[0].max() <= first_by_part[1].min()
            <= first_by_part[1].max() <= first_by_part[2].min()):
        raise ValueError("production job groups are not in time order (train < val < test)")

    aligned_labels = labels.reset_index()
    y = _label_matrix(aligned_labels, 'production labels')
    valid = _label_validity(aligned_labels, 'production labels', required=True)
    X = _finite_matrix(features, names, 'production features')

    return ProductionData(
        X=X, y=y, valid=valid,
        ids=features['_source_path'].to_numpy(), start_time=start,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, feature_names=names)


def grouped_benchmark_split(labels, groups, test_ratio, seed):
    """Development and test rows with whole jobs on one side.

    Iterative stratification (Sechidis et al. 2011, ``iterstrat``) runs on the
    groups, each described by the union of its rows' labels, then the group
    assignment is expanded to rows. No group appears on both sides.
    """
    if not 0 < test_ratio < 1:
        raise ValueError(f"benchmark split ratio must be in (0, 1), got {test_ratio}")
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    if len(unique_groups) < 2:
        raise ValueError("benchmark split needs at least two job groups")
    group_labels = np.zeros((len(unique_groups), labels.shape[1]), dtype=int)
    np.maximum.at(group_labels, inverse, labels.astype(int))
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    dev_groups, test_groups = next(splitter.split(np.zeros((len(unique_groups), 1)), group_labels))
    in_test = np.isin(inverse, test_groups)
    dev_idx, test_idx = np.flatnonzero(~in_test), np.flatnonzero(in_test)
    if set(groups[dev_idx]) & set(groups[test_idx]):
        raise AssertionError("a benchmark job is on both sides of the split")
    if len(dev_idx) == 0 or len(test_idx) == 0:
        raise ValueError("benchmark split left a side empty")
    return dev_idx, test_idx


def grouped_benchmark_partitions(labels, groups, test_ratio, validation_ratio, seed):
    """Group-disjoint benchmark train, validation, and test row positions."""
    remaining, test_idx = grouped_benchmark_split(labels, groups, test_ratio, seed)
    train_local, val_local = grouped_benchmark_split(
        labels[remaining], groups[remaining], validation_ratio, seed + 1)
    train_idx, val_idx = remaining[train_local], remaining[val_local]
    group_sets = [set(groups[idx]) for idx in (train_idx, val_idx, test_idx)]
    if any(group_sets[i] & group_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("benchmark partitions share a job")
    return train_idx, val_idx, test_idx


def load_benchmark(config, feature_names):
    """Benchmark features and labels (row-aligned outputs of
    ``extract_benchmark_features.py``), with the grouped split."""
    features = pd.read_parquet(_resolve(config, 'benchmark_features'))
    labels = pd.read_parquet(_resolve(config, 'benchmark_labels'))
    _check_schema(features, 'benchmark features')
    if len(features) != len(labels):
        raise ValueError("benchmark features and labels differ in length")
    same = (features['_ground_truth_job_id'].astype(str).to_numpy() == labels['job_id'].astype(str).to_numpy())
    if not same.all() or not (features['_benchmark'].to_numpy() == labels['benchmark'].to_numpy()).all():
        raise ValueError("benchmark features and labels are not row-aligned")
    if '_scenario' not in features.columns or 'scenario' not in labels.columns:
        raise ValueError("benchmark features and labels need scenario columns")
    if not (features['_scenario'].astype(str).to_numpy()
            == labels['scenario'].astype(str).to_numpy()).all():
        raise ValueError("benchmark feature and label scenarios are not row-aligned")
    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        raise ValueError(f"benchmark features lack {len(missing)} contract columns, first {missing[:5]}")
    y = _label_matrix(labels, 'benchmark labels')
    valid = _label_validity(labels, 'benchmark labels', required=False)

    groups = (features['_benchmark'] + '/' + features['_ground_truth_job_id'].astype(str)).to_numpy()
    ids = (groups + '/' + features['_source_path'].map(lambda p: Path(p).name)).to_numpy()
    if len(set(ids)) != len(ids):
        raise ValueError("benchmark sample ids are not unique")
    split = config['benchmark_split']
    train_idx, val_idx, test_idx = grouped_benchmark_partitions(
        y, groups, split['test_ratio'], split['validation_ratio'], split['seed'])
    X = _finite_matrix(features, feature_names, 'benchmark features')
    return BenchmarkData(X=X, y=y, valid=valid, ids=ids,
                         groups=groups, benchmark=features['_benchmark'].to_numpy(),
                         train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


# ---------------------------------------------------------------------------
# Fitting and prediction
# ---------------------------------------------------------------------------

def scale_pos_weights(y, max_weight, valid=None):
    """Per-label negative/positive ratio, capped."""
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    if valid is None:
        valid = np.ones_like(y, dtype=bool)
    if valid.shape != y.shape:
        raise ValueError("label validity shape differs from labels")
    weights = []
    for i, dim in enumerate(BOTTLENECK_DIMENSIONS):
        observed = y[valid[:, i], i]
        positives = observed.sum()
        negatives = len(observed) - positives
        if positives == 0 or negatives == 0:
            raise ValueError(f"training rows need both classes for {dim}")
        weights.append(min(negatives / positives, max_weight))
    return weights


def fit_models(X, y, weights, X_val, y_val, model_type, config, seed,
               valid=None, val_valid=None):
    """One binary classifier per bottleneck dimension.

    Tree boosters stop early on the production validation partition and the
    number of rounds they kept is returned per label; the random forest has
    no validation role.
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"model {model_type!r} is not supported; choose from {SUPPORTED_MODELS} "
                         "(the sklearn MLP cannot take sample weights, so a weighted MLP is not offered)")
    if len(X) == 0 or len(X_val) == 0:
        raise ValueError("training and early-stopping rows must be nonempty")
    if X.shape[0] != y.shape[0] or len(weights) != len(X):
        raise ValueError("training features, labels, and weights differ in length")
    if X_val.shape[0] != y_val.shape[0] or X.shape[1] != X_val.shape[1]:
        raise ValueError("validation features and labels differ or feature counts changed")
    if not (np.isfinite(X).all() and np.isfinite(y).all() and np.isfinite(weights).all()
            and np.isfinite(X_val).all() and np.isfinite(y_val).all()):
        raise ValueError("training inputs contain non-finite values")
    if (weights <= 0).any():
        raise ValueError("sample weights must be positive")
    if valid is None:
        valid = np.ones_like(y, dtype=bool)
    if val_valid is None:
        val_valid = np.ones_like(y_val, dtype=bool)
    if valid.shape != y.shape or val_valid.shape != y_val.shape:
        raise ValueError("label validity shapes differ from label matrices")
    if not valid.any(axis=0).all() or not val_valid.any(axis=0).all():
        raise ValueError("every target needs valid training and validation rows")
    params = dict(config['models'][model_type]['params'])
    spw = scale_pos_weights(y, config['imbalance']['max_weight'], valid=valid)
    rounds = int(config['early_stopping']['rounds'])
    models, best_iteration = {}, {}
    for i, dim in enumerate(BOTTLENECK_DIMENSIONS):
        train_rows = valid[:, i]
        validation_rows = val_valid[:, i]
        X_dimension = X[train_rows]
        y_dimension = y[train_rows, i]
        weight_dimension = weights[train_rows]
        X_validation = X_val[validation_rows]
        y_validation = y_val[validation_rows, i]
        if model_type == 'xgboost':
            from xgboost import XGBClassifier
            clf = XGBClassifier(**params, scale_pos_weight=spw[i], random_state=seed,
                                verbosity=0, early_stopping_rounds=rounds)
            clf.fit(X_dimension, y_dimension, sample_weight=weight_dimension,
                    eval_set=[(X_validation, y_validation)], verbose=False)
            best_iteration[dim] = int(clf.best_iteration)
        elif model_type == 'lightgbm':
            import lightgbm
            clf = lightgbm.LGBMClassifier(**params, scale_pos_weight=spw[i], random_state=seed, verbose=-1)
            clf.fit(X_dimension, y_dimension, sample_weight=weight_dimension,
                    eval_set=[(X_validation, y_validation)],
                    callbacks=[lightgbm.early_stopping(rounds, verbose=False)])
            best_iteration[dim] = int(clf.best_iteration_)
        else:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(**params, random_state=seed)
            clf.fit(X_dimension, y_dimension, sample_weight=weight_dimension)
            best_iteration[dim] = int(params['n_estimators'])
        models[dim] = clf
    return models, best_iteration


def validate_bundle(bundle):
    """Validate the fields required for prediction and provenance."""
    if not isinstance(bundle, dict) or bundle.get('bundle_format') != BUNDLE_FORMAT:
        raise ValueError(f"model is not a bundle of format {BUNDLE_FORMAT}")
    required = {
        'feature_schema_version', 'feature_names', 'dimensions', 'bottleneck_dimensions',
        'decision_threshold', 'model_type', 'seed', 'models', 'input_hashes',
        'config', 'config_sha256', 'git_revision', 'git_clean', 'final_evaluation',
        'benchmark_partitions',
    }
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"model bundle lacks fields {sorted(missing)}")
    if bundle['feature_schema_version'] != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"bundle schema {bundle['feature_schema_version']}, "
                         f"code expects {FEATURE_SCHEMA_VERSION}")
    if list(bundle['dimensions']) != list(DIMENSION_NAMES):
        raise ValueError("bundle label order differs from the code")
    if list(bundle['bottleneck_dimensions']) != list(BOTTLENECK_DIMENSIONS):
        raise ValueError("bundle bottleneck order differs from the code")
    if set(bundle['models']) != set(BOTTLENECK_DIMENSIONS):
        raise ValueError("bundle model keys differ from the bottleneck dimensions")
    if len(bundle['feature_names']) != len(set(bundle['feature_names'])):
        raise ValueError("bundle has duplicate feature names")
    if not 0 < float(bundle['decision_threshold']) < 1:
        raise ValueError("bundle decision threshold must be in (0, 1)")
    if not bundle['git_clean']:
        raise ValueError("bundle was not produced from a clean Git worktree")
    if bundle['model_type'] not in SUPPORTED_MODELS:
        raise ValueError(f"bundle has unsupported model type {bundle['model_type']!r}")
    if not isinstance(bundle['seed'], int):
        raise ValueError("bundle seed must be an integer")
    if not isinstance(bundle['final_evaluation'], bool):
        raise ValueError("bundle final_evaluation must be boolean")
    partitions = bundle['benchmark_partitions']
    expected_partitions = {'train', 'validation', 'test'}
    if not isinstance(partitions, dict) or set(partitions) != expected_partitions:
        raise ValueError("bundle benchmark partitions have an invalid contract")
    partition_sets = []
    for name in ('train', 'validation', 'test'):
        values = partitions[name]
        if (not isinstance(values, list) or not all(isinstance(value, str) and value
                                                    for value in values)
                or len(values) != len(set(values))):
            raise ValueError(f"bundle benchmark partition {name} has invalid ids")
        partition_sets.append(set(values))
    if any(partition_sets[i] & partition_sets[j]
           for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("bundle benchmark partitions share sample ids")
    if not partition_sets[0] or not partition_sets[1]:
        raise ValueError("bundle benchmark training and validation partitions must be nonempty")
    if bool(partition_sets[2]) != bundle['final_evaluation']:
        raise ValueError("bundle benchmark test partition differs from final-evaluation mode")
    expected_inputs = {'production_features', 'production_labels', 'production_splits',
                       'benchmark_features', 'benchmark_labels'}
    if set(bundle['input_hashes']) != expected_inputs:
        raise ValueError("bundle input hashes do not cover the five training inputs")
    return bundle


def verify_bundle_inputs(bundle):
    """Check that the configuration and datasets still match the bundle."""
    validate_bundle(bundle)
    if sha256_file(bundle['config_path']) != bundle['config_sha256']:
        raise ValueError("training configuration differs from the bundle hash")
    for name, record in bundle['input_hashes'].items():
        if set(record) != {'path', 'sha256'}:
            raise ValueError(f"bundle input record {name} has an invalid contract")
        if sha256_file(record['path']) != record['sha256']:
            raise ValueError(f"training input {name} differs from the bundle hash")


def load_final_benchmark_test_frames(bundle_path):
    """Load the exact benchmark test rows declared inside a final bundle."""
    bundle_path = Path(bundle_path)
    with bundle_path.open('rb') as handle:
        bundle = pickle.load(handle)
    validate_bundle(bundle)
    verify_bundle_inputs(bundle)
    if not bundle['final_evaluation']:
        raise ValueError("benchmark evaluation requires a final-evaluation bundle")

    splits_path = bundle_path.parent / 'splits.npz'
    if not splits_path.is_file():
        raise FileNotFoundError(f"training split artifact is missing: {splits_path}")
    splits = np.load(splits_path, allow_pickle=False)
    required = {'bench_train', 'bench_val', 'bench_test', 'bench_ids', 'bench_groups'}
    if not required.issubset(splits.files):
        raise ValueError(f"training split artifact lacks {sorted(required - set(splits.files))}")

    config = load_config(bundle['config_path'])
    benchmark = load_benchmark(config, bundle['feature_names'])
    if not np.array_equal(benchmark.ids.astype(str), splits['bench_ids'].astype(str)):
        raise ValueError("benchmark sample ids differ from the training split artifact")
    for split_key, bundle_key in (
            ('bench_train', 'train'), ('bench_val', 'validation'), ('bench_test', 'test')):
        positions = np.asarray(splits[split_key])
        if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer):
            raise ValueError(f"training split {split_key} is not an integer vector")
        if len(positions) and (positions.min() < 0 or positions.max() >= len(benchmark.ids)):
            raise ValueError(f"training split {split_key} has an out-of-range position")
        ids = benchmark.ids[positions].astype(str).tolist()
        if ids != bundle['benchmark_partitions'][bundle_key]:
            raise ValueError(f"training split {split_key} differs from the model bundle")

    features_path = Path(bundle['input_hashes']['benchmark_features']['path'])
    labels_path = Path(bundle['input_hashes']['benchmark_labels']['path'])
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)
    if len(features) != len(benchmark.ids) or len(labels) != len(benchmark.ids):
        raise ValueError("benchmark frame lengths differ from the verified bundle input")
    test_positions = np.asarray(splits['bench_test'])
    test_features = features.iloc[test_positions].reset_index(drop=True)
    test_labels = labels.iloc[test_positions].reset_index(drop=True)
    frame_ids = (
        test_features['_benchmark'].astype(str) + '/'
        + test_features['_ground_truth_job_id'].astype(str) + '/'
        + test_features['_source_path'].map(lambda value: Path(str(value)).name)
    ).tolist()
    if frame_ids != bundle['benchmark_partitions']['test']:
        raise ValueError("selected benchmark test rows differ from the model bundle")
    return bundle, test_features, test_labels, frame_ids


def predict(bundle, X):
    """Probabilities of the seven bottleneck labels and the eight decisions.

    Healthy is derived: it is 1 exactly when no bottleneck probability
    reaches the bundle's decision threshold, so a prediction can never be
    healthy and bottlenecked at once, nor neither.
    """
    validate_bundle(bundle)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("prediction input must be a two-dimensional matrix")
    if X.shape[1] != len(bundle['feature_names']):
        raise ValueError(f"expected {len(bundle['feature_names'])} features, got {X.shape[1]}")
    if not np.isfinite(X).all():
        raise ValueError("prediction input contains non-finite values")
    proba = np.column_stack([bundle['models'][d].predict_proba(X)[:, 1]
                             for d in BOTTLENECK_DIMENSIONS])
    if not np.isfinite(proba).all() or ((proba < 0) | (proba > 1)).any():
        raise ValueError("model returned an invalid probability")
    decisions = (proba >= bundle['decision_threshold']).astype(int)
    healthy = (decisions.sum(axis=1) == 0).astype(int)
    return proba, np.column_stack([decisions, healthy])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true8, y_pred8, groups, config):
    """Micro and macro F1, Hamming loss, per-label scores, and bootstrap
    intervals that resample whole groups (jobs), not rows."""
    boot = config['evaluation']['bootstrap']
    y_true8 = np.asarray(y_true8)
    y_pred8 = np.asarray(y_pred8)
    groups = np.asarray(groups)
    if y_true8.shape != y_pred8.shape or y_true8.ndim != 2 or y_true8.shape[1] != len(DIMENSION_NAMES):
        raise ValueError("evaluation label matrices must have the same eight-column shape")
    if len(groups) != len(y_true8) or len(groups) == 0:
        raise ValueError("evaluation groups must provide one nonempty value per row")
    if not (np.isin(y_true8, [0, 1]).all() and np.isin(y_pred8, [0, 1]).all()):
        raise ValueError("evaluation labels and decisions must be binary")
    if int(boot['n_resamples']) <= 0:
        raise ValueError("bootstrap n_resamples must be positive")
    if not 0 < float(boot['confidence_level']) < 1:
        raise ValueError("bootstrap confidence_level must be in (0, 1)")
    metrics = {
        'micro_f1': float(f1_score(y_true8, y_pred8, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(y_true8, y_pred8, average='macro', zero_division=0)),
        'hamming_loss': float(hamming_loss(y_true8, y_pred8)),
        'n_samples': int(len(y_true8)), 'n_groups': int(len(np.unique(groups))),
        'per_label': {},
    }
    for i, dim in enumerate(DIMENSION_NAMES):
        metrics['per_label'][dim] = {
            'f1': float(f1_score(y_true8[:, i], y_pred8[:, i], zero_division=0)),
            'precision': float(precision_score(y_true8[:, i], y_pred8[:, i], zero_division=0)),
            'recall': float(recall_score(y_true8[:, i], y_pred8[:, i], zero_division=0)),
            'support': int(y_true8[:, i].sum()),
        }
    rng = np.random.RandomState(boot['seed'])
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    rows_of = [np.flatnonzero(inverse == g) for g in range(len(unique_groups))]
    micro, macro = [], []
    for _ in range(int(boot['n_resamples'])):
        chosen = rng.choice(len(unique_groups), len(unique_groups), replace=True)
        idx = np.concatenate([rows_of[g] for g in chosen])
        micro.append(f1_score(y_true8[idx], y_pred8[idx], average='micro', zero_division=0))
        macro.append(f1_score(y_true8[idx], y_pred8[idx], average='macro', zero_division=0))
    alpha = (1 - boot['confidence_level']) / 2 * 100
    metrics['micro_f1_ci'] = [float(np.percentile(micro, alpha)), float(np.percentile(micro, 100 - alpha))]
    metrics['macro_f1_ci'] = [float(np.percentile(macro, alpha)), float(np.percentile(macro, 100 - alpha))]
    return metrics


def with_healthy(y7):
    """The eight-column label matrix: healthy is the complement of the seven."""
    return np.column_stack([y7, (y7.sum(axis=1) == 0).astype(y7.dtype)])


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def _git_revision():
    out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_DIR, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else 'unknown'


def _git_state():
    revision = _git_revision()
    if revision == 'unknown':
        raise RuntimeError("cannot identify the Git revision")
    status = subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=all'], cwd=PROJECT_DIR,
        capture_output=True, text=True)
    if status.returncode != 0:
        raise RuntimeError(f"cannot inspect Git state: {status.stderr.strip()}")
    return revision, not bool(status.stdout.strip())


def _input_hashes(config):
    keys = ('production_features', 'production_labels', 'production_splits',
            'benchmark_features', 'benchmark_labels')
    return {key: {'path': str(_resolve(config, key)), 'sha256': sha256_file(_resolve(config, key))}
            for key in keys}


def train_run(config, model_type, seeds, clean_weight, run_dir, feature_set='full',
              use_production=True, hold_out_benchmark=None, final_evaluation=False):
    """Train ``model_type`` for every seed and write an immutable run directory.

    Protocol: production training rows (weight 1) plus benchmark training
    rows (weight ``clean_weight``); early stopping on the production
    validation rows; separate benchmark validation metrics for choices. Test
    metrics are computed only when ``final_evaluation`` is true, after every
    choice is fixed.

    Ablations, recorded in the manifest: ``feature_set='raw'`` drops the
    derived features; ``use_production=False`` trains on the benchmark
    training rows alone and early-stops on benchmark validation;
    ``hold_out_benchmark`` removes one benchmark type from the training rows
    and, for a final run, reports the test metrics on that
    type's rows as well (``test_excluded_benchmark``).
    """
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"run directory exists, runs are immutable: {run_dir}")
    seeds = list(seeds)
    if not seeds or len(seeds) != len(set(seeds)) or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must be a nonempty list of unique integers")
    if clean_weight <= 0:
        raise ValueError("clean_weight must be positive")
    if not 0 < float(config['decision_threshold']) < 1:
        raise ValueError("decision_threshold must be in (0, 1)")
    revision, git_clean = _git_state()
    if not git_clean:
        raise RuntimeError("training requires a clean Git worktree")
    input_hashes = _input_hashes(config)
    config_sha256 = sha256_file(config['_path'])

    prod = load_production(config, feature_set)
    bench = load_benchmark(config, prod.feature_names)
    if hold_out_benchmark is not None and hold_out_benchmark not in set(bench.benchmark):
        raise ValueError(f"no benchmark rows of type {hold_out_benchmark!r}")
    train_idx = bench.train_idx
    if hold_out_benchmark is not None:
        train_idx = train_idx[bench.benchmark[train_idx] != hold_out_benchmark]
    if len(train_idx) == 0:
        raise ValueError("benchmark filtering left no training rows")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = run_dir.parent / f'.{run_dir.name}.incomplete.{os.getpid()}.{time.time_ns()}'
    work_dir.mkdir()
    prod_train = prod.train_idx if use_production else prod.train_idx[:0]
    if use_production:
        X_train = np.vstack([prod.X[prod_train], bench.X[train_idx]])
        y_train = np.vstack([prod.y[prod_train], bench.y[train_idx]])
        valid_train = np.vstack([prod.valid[prod_train], bench.valid[train_idx]])
        weights = np.concatenate([np.ones(len(prod_train)), np.full(len(train_idx), clean_weight)])
        X_early, y_early = prod.X[prod.val_idx], prod.y[prod.val_idx]
        valid_early = prod.valid[prod.val_idx]
        early_stopping_source = 'production_validation'
    else:
        X_train, y_train = bench.X[train_idx], bench.y[train_idx]
        valid_train = bench.valid[train_idx]
        weights = np.ones(len(train_idx))
        X_early, y_early = bench.X[bench.val_idx], bench.y[bench.val_idx]
        valid_early = bench.valid[bench.val_idx]
        early_stopping_source = 'benchmark_validation'
    excluded_test = (bench.test_idx[bench.benchmark[bench.test_idx] == hold_out_benchmark]
                 if hold_out_benchmark is not None else None)
    logger.info("Training rows: %d production (w=1) + %d benchmark train (w=%.0f); "
                "benchmark validation %d; test %s; feature set %s (%d features)",
                len(prod_train), len(train_idx), clean_weight, len(bench.val_idx),
                len(bench.test_idx) if final_evaluation else 'not evaluated',
                feature_set, len(prod.feature_names))

    np.savez(work_dir / 'splits.npz', prod_train=prod_train, prod_val=prod.val_idx,
             prod_test=prod.test_idx, bench_train=train_idx, bench_val=bench.val_idx,
             bench_test=bench.test_idx if final_evaluation else np.array([], dtype=int),
             bench_ids=bench.ids.astype(str), bench_groups=bench.groups.astype(str))
    benchmark_partitions = {
        'train': bench.ids[train_idx].astype(str).tolist(),
        'validation': bench.ids[bench.val_idx].astype(str).tolist(),
        'test': (bench.ids[bench.test_idx].astype(str).tolist()
                 if final_evaluation else []),
    }

    y_val8 = with_healthy(bench.y[bench.val_idx])
    y_test8 = with_healthy(bench.y[bench.test_idx]) if final_evaluation else None
    resolved_config = {k: v for k, v in config.items() if not k.startswith('_')}
    per_seed = {}
    for seed in seeds:
        t0 = time.time()
        models, best_iteration = fit_models(
            X_train, y_train, weights, X_early, y_early, model_type, config, seed,
            valid=valid_train, val_valid=valid_early)
        bundle = {
            'bundle_format': BUNDLE_FORMAT, 'feature_schema_version': FEATURE_SCHEMA_VERSION,
            'feature_names': list(prod.feature_names), 'dimensions': list(DIMENSION_NAMES),
            'bottleneck_dimensions': list(BOTTLENECK_DIMENSIONS),
            'decision_threshold': float(config['decision_threshold']),
            'model_type': model_type, 'seed': seed, 'clean_weight': clean_weight,
            'best_iteration': best_iteration, 'models': models,
            'config_path': config['_path'], 'config': resolved_config,
            'config_sha256': config_sha256, 'input_hashes': input_hashes,
            'git_revision': revision, 'git_clean': git_clean,
            'final_evaluation': bool(final_evaluation),
            'benchmark_partitions': benchmark_partitions,
            'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        validate_bundle(bundle)
        _, val_pred = predict(bundle, bench.X[bench.val_idx])
        per_seed[seed] = {
            'validation': evaluate(y_val8, val_pred, bench.groups[bench.val_idx], config),
            'best_iteration': best_iteration, 'fit_seconds': round(time.time() - t0, 1),
        }
        if final_evaluation:
            _, test_pred = predict(bundle, bench.X[bench.test_idx])
            per_seed[seed]['test'] = evaluate(
                y_test8, test_pred, bench.groups[bench.test_idx], config)
        if final_evaluation and excluded_test is not None and len(excluded_test):
            _, excluded_pred = predict(bundle, bench.X[excluded_test])
            per_seed[seed]['test_excluded_benchmark'] = evaluate(
                with_healthy(bench.y[excluded_test]), excluded_pred,
                bench.groups[excluded_test], config)
        weight_tag = format(float(clean_weight), 'g').replace('.', 'p')
        with open(work_dir / f'{model_type}_w{weight_tag}_seed{seed}.pkl', 'wb') as fh:
            pickle.dump(bundle, fh)
        message = "seed %d: validation micro-F1 %.4f"
        values = [seed, per_seed[seed]['validation']['micro_f1']]
        if final_evaluation:
            message += ", test micro-F1 %.4f [%.4f, %.4f], macro %.4f"
            values.extend([per_seed[seed]['test']['micro_f1'],
                           *per_seed[seed]['test']['micro_f1_ci'],
                           per_seed[seed]['test']['macro_f1']])
        logger.info(message, *values)

    final_revision, final_clean = _git_state()
    if final_revision != revision or not final_clean:
        raise RuntimeError("Git revision or worktree state changed during the run")
    if (_input_hashes(config) != input_hashes
            or sha256_file(config['_path']) != config_sha256):
        raise RuntimeError("a training input or configuration changed during the run")

    summary = {}
    metric_splits = ['validation'] + (['test'] if final_evaluation else [])
    for split in metric_splits:
        for metric in ('micro_f1', 'macro_f1', 'hamming_loss'):
            values = [per_seed[s][split][metric] for s in seeds]
            summary[f'{split}_{metric}_mean'] = float(np.mean(values))
            summary[f'{split}_{metric}_std'] = float(np.std(values))
    manifest = {
        'model_type': model_type, 'seeds': seeds, 'clean_weight': clean_weight,
        'feature_set': feature_set, 'use_production': use_production,
        'hold_out_benchmark': hold_out_benchmark, 'final_evaluation': bool(final_evaluation),
        'early_stopping_source': early_stopping_source,
        'feature_names': list(prod.feature_names), 'config': resolved_config,
        'config_path': config['_path'], 'config_sha256': config_sha256,
        'git_revision': revision, 'git_clean': git_clean,
        'feature_schema_version': FEATURE_SCHEMA_VERSION, 'n_features': len(prod.feature_names),
        'inputs': input_hashes,
        'sizes': {'prod_train': int(len(prod_train)), 'prod_val': int(len(prod.val_idx)),
                  'prod_test': int(len(prod.test_idx)), 'bench_train': int(len(train_idx)),
                  'bench_validation': int(len(bench.val_idx)),
                  'bench_test': int(len(bench.test_idx)) if final_evaluation else 0,
                  'bench_test_excluded': int(len(excluded_test))
                  if final_evaluation and excluded_test is not None else 0},
        'summary': summary, 'per_seed': {str(s): per_seed[s] for s in seeds},
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(work_dir / 'manifest.json', 'w') as fh:
        json.dump(manifest, fh, indent=2)
    os.rename(work_dir, run_dir)
    logger.info("Run written: %s (validation micro-F1 %.4f +/- %.4f over %d seeds)", run_dir,
                summary['validation_micro_f1_mean'], summary['validation_micro_f1_std'], len(seeds))
    return manifest
