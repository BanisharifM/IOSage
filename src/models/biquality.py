"""
Biquality training: one path from the two data sources to a model bundle.

Framework: biquality learning (Nodet et al., Machine Learning 2023). The
production logs carry heuristic (untrusted) labels, the benchmark logs carry
construction (trusted) labels; both share one feature space and one label
set. The benchmark development rows enter training with a higher sample
weight; the benchmark test rows are evaluated once, after every choice.

This module owns everything the audit found duplicated or unguarded across
the older entry points: sample alignment by a unique id, split validation,
grouped benchmark partitions, the feature contract, weighting, fitting with
real early stopping, healthy derived from the seven bottleneck decisions,
group bootstrap intervals, and an immutable run directory with a manifest.
The entry point is ``scripts/train_biquality.py``.
"""

import hashlib
import json
import logging
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

from src.data.benchmark_verify import BOTTLENECK_DIMENSIONS, DIMENSION_NAMES
from src.data.feature_extraction import FEATURE_SCHEMA_VERSION, get_raw_feature_names

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
BUNDLE_FORMAT = 1
SUPPORTED_MODELS = ('xgboost', 'lightgbm', 'random_forest')


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path):
    """The training configuration; every key the module reads must exist."""
    path = Path(path)
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
    ids: np.ndarray          # "<benchmark>/<job_id>/<log basename>"
    groups: np.ndarray       # "<benchmark>/<job_id>"
    benchmark: np.ndarray    # benchmark type per row
    dev_idx: np.ndarray
    test_idx: np.ndarray


def _check_schema(df, what):
    if '_schema_version' not in df.columns:
        raise ValueError(f"{what} has no _schema_version column; re-extract")
    versions = set(pd.unique(df['_schema_version']))
    if versions != {FEATURE_SCHEMA_VERSION}:
        raise ValueError(f"{what} schema {sorted(versions)}, expected {FEATURE_SCHEMA_VERSION}")


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
    parts = [np.asarray(splits[k]) for k in ('train_idx', 'val_idx', 'test_idx')]
    joined = np.concatenate(parts)
    if any(len(p) == 0 for p in parts):
        raise ValueError("a production split partition is empty")
    if len(np.unique(joined)) != n or joined.min() != 0 or joined.max() != n - 1:
        raise ValueError("production split positions must be disjoint and cover every row")
    return parts


def load_production(config, feature_set='full'):
    """Production features, labels and the temporal split, aligned by ``_source_path``.

    The labels file is joined on ``_source_path`` (unique per log; ``_jobid``
    is not, one SLURM job holds many launches). The split positions from
    preprocessing are checked for coverage, disjointness and time order.
    """
    features = pd.read_parquet(_resolve(config, 'production_features'))
    labels = pd.read_parquet(_resolve(config, 'production_labels'))
    _check_schema(features, 'production features')
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
    if not (start[train_idx].max() <= start[val_idx].min() <= start[val_idx].max()
            <= start[test_idx].min()):
        raise ValueError("production split is not in time order (train < val < test)")

    return ProductionData(
        X=features[names].to_numpy(dtype=np.float32),
        y=labels[BOTTLENECK_DIMENSIONS].to_numpy(dtype=np.float32),
        ids=features['_source_path'].to_numpy(), start_time=start,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, feature_names=names)


def grouped_benchmark_split(labels, groups, test_ratio, seed):
    """Development and test rows with whole jobs on one side.

    Iterative stratification (Sechidis et al. 2011, ``iterstrat``) runs on the
    groups, each described by the union of its rows' labels, then the group
    assignment is expanded to rows. No group appears on both sides.
    """
    unique_groups, inverse = np.unique(groups, return_inverse=True)
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
    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        raise ValueError(f"benchmark features lack {len(missing)} contract columns, first {missing[:5]}")
    if (labels[DIMENSION_NAMES].sum(axis=1) == 0).any():
        raise ValueError("a benchmark row has no label")
    healthy_and_bottleneck = (labels['healthy'] == 1) & (labels[BOTTLENECK_DIMENSIONS].sum(axis=1) > 0)
    if healthy_and_bottleneck.any():
        raise ValueError("a benchmark row is healthy and bottlenecked at once")

    groups = (features['_benchmark'] + '/' + features['_ground_truth_job_id'].astype(str)).to_numpy()
    ids = (groups + '/' + features['_source_path'].map(lambda p: Path(p).name)).to_numpy()
    if len(set(ids)) != len(ids):
        raise ValueError("benchmark sample ids are not unique")
    y = labels[BOTTLENECK_DIMENSIONS].to_numpy(dtype=np.float32)
    split = config['benchmark_split']
    dev_idx, test_idx = grouped_benchmark_split(y, groups, split['test_ratio'], split['seed'])
    return BenchmarkData(X=features[feature_names].to_numpy(dtype=np.float32), y=y, ids=ids,
                         groups=groups, benchmark=features['_benchmark'].to_numpy(),
                         dev_idx=dev_idx, test_idx=test_idx)


# ---------------------------------------------------------------------------
# Fitting and prediction
# ---------------------------------------------------------------------------

def scale_pos_weights(y, max_weight):
    """Per-label negative/positive ratio, capped."""
    return [min((len(y) - y[:, i].sum()) / max(y[:, i].sum(), 1), max_weight)
            for i in range(y.shape[1])]


def fit_models(X, y, weights, X_val, y_val, model_type, config, seed):
    """One binary classifier per bottleneck dimension.

    Tree boosters stop early on the production validation partition and the
    number of rounds they kept is returned per label; the random forest has
    no validation role.
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"model {model_type!r} is not supported; choose from {SUPPORTED_MODELS} "
                         "(the sklearn MLP cannot take sample weights, so a weighted MLP is not offered)")
    params = dict(config['models'][model_type]['params'])
    spw = scale_pos_weights(y, config['imbalance']['max_weight'])
    rounds = int(config['early_stopping']['rounds'])
    models, best_iteration = {}, {}
    for i, dim in enumerate(BOTTLENECK_DIMENSIONS):
        if model_type == 'xgboost':
            from xgboost import XGBClassifier
            clf = XGBClassifier(**params, scale_pos_weight=spw[i], random_state=seed,
                                verbosity=0, early_stopping_rounds=rounds)
            clf.fit(X, y[:, i], sample_weight=weights, eval_set=[(X_val, y_val[:, i])], verbose=False)
            best_iteration[dim] = int(clf.best_iteration)
        elif model_type == 'lightgbm':
            import lightgbm
            clf = lightgbm.LGBMClassifier(**params, scale_pos_weight=spw[i], random_state=seed, verbose=-1)
            clf.fit(X, y[:, i], sample_weight=weights, eval_set=[(X_val, y_val[:, i])],
                    callbacks=[lightgbm.early_stopping(rounds, verbose=False)])
            best_iteration[dim] = int(clf.best_iteration_)
        else:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(**params, random_state=seed)
            clf.fit(X, y[:, i], sample_weight=weights)
            best_iteration[dim] = int(params['n_estimators'])
        models[dim] = clf
    return models, best_iteration


def predict(bundle, X):
    """Probabilities of the seven bottleneck labels and the eight decisions.

    Healthy is derived: it is 1 exactly when no bottleneck probability
    reaches the bundle's decision threshold, so a prediction can never be
    healthy and bottlenecked at once, nor neither.
    """
    if X.shape[1] != len(bundle['feature_names']):
        raise ValueError(f"expected {len(bundle['feature_names'])} features, got {X.shape[1]}")
    proba = np.column_stack([bundle['models'][d].predict_proba(X)[:, 1] for d in BOTTLENECK_DIMENSIONS])
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

def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _git_revision():
    out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_DIR, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else 'unknown'


def train_run(config, model_type, seeds, clean_weight, run_dir, feature_set='full',
              use_production=True, hold_out_benchmark=None):
    """Train ``model_type`` for every seed and write an immutable run directory.

    Protocol: production training rows (weight 1) plus benchmark development
    rows (weight ``clean_weight``); early stopping on the production
    validation rows; benchmark development metrics for choices; benchmark
    test metrics once per seed, reported as mean and standard deviation
    over seeds. Nothing is chosen on the test rows.

    Ablations, recorded in the manifest: ``feature_set='raw'`` drops the
    derived features; ``use_production=False`` trains on the benchmark
    development rows alone; ``hold_out_benchmark`` removes one benchmark
    type from the development rows and reports the test metrics on that
    type's rows as well (``test_excluded_benchmark``).
    """
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"run directory exists, runs are immutable: {run_dir}")
    prod = load_production(config, feature_set)
    bench = load_benchmark(config, prod.feature_names)
    if hold_out_benchmark is not None and hold_out_benchmark not in set(bench.benchmark):
        raise ValueError(f"no benchmark rows of type {hold_out_benchmark!r}")
    run_dir.mkdir(parents=True)

    dev_idx = bench.dev_idx
    if hold_out_benchmark is not None:
        dev_idx = dev_idx[bench.benchmark[dev_idx] != hold_out_benchmark]
    prod_train = prod.train_idx if use_production else prod.train_idx[:0]
    X_train = np.vstack([prod.X[prod_train], bench.X[dev_idx]])
    y_train = np.vstack([prod.y[prod_train], bench.y[dev_idx]])
    weights = np.concatenate([np.ones(len(prod_train)), np.full(len(dev_idx), clean_weight)])
    X_val, y_val = prod.X[prod.val_idx], prod.y[prod.val_idx]
    excluded_test = (bench.test_idx[bench.benchmark[bench.test_idx] == hold_out_benchmark]
                 if hold_out_benchmark is not None else None)
    logger.info("Training rows: %d production (w=1) + %d benchmark dev (w=%.0f); validation %d; "
                "benchmark test %d rows in %d jobs; feature set %s (%d features)",
                len(prod_train), len(dev_idx), clean_weight, len(prod.val_idx), len(bench.test_idx),
                len(np.unique(bench.groups[bench.test_idx])), feature_set, len(prod.feature_names))

    np.savez(run_dir / 'splits.npz', prod_train=prod_train, prod_val=prod.val_idx,
             prod_test=prod.test_idx, bench_dev=dev_idx, bench_test=bench.test_idx,
             bench_ids=bench.ids, bench_groups=bench.groups)

    y_dev8 = with_healthy(bench.y[dev_idx])
    y_test8 = with_healthy(bench.y[bench.test_idx])
    per_seed = {}
    for seed in seeds:
        t0 = time.time()
        models, best_iteration = fit_models(X_train, y_train, weights, X_val, y_val, model_type, config, seed)
        bundle = {
            'bundle_format': BUNDLE_FORMAT, 'feature_schema_version': FEATURE_SCHEMA_VERSION,
            'feature_names': list(prod.feature_names), 'dimensions': list(DIMENSION_NAMES),
            'bottleneck_dimensions': list(BOTTLENECK_DIMENSIONS),
            'decision_threshold': float(config['decision_threshold']),
            'model_type': model_type, 'seed': seed, 'clean_weight': clean_weight,
            'best_iteration': best_iteration, 'models': models,
            'config_path': config['_path'], 'git_revision': _git_revision(),
            'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        _, dev_pred = predict(bundle, bench.X[dev_idx])
        _, test_pred = predict(bundle, bench.X[bench.test_idx])
        per_seed[seed] = {
            'dev': evaluate(y_dev8, dev_pred, bench.groups[dev_idx], config),
            'test': evaluate(y_test8, test_pred, bench.groups[bench.test_idx], config),
            'best_iteration': best_iteration, 'fit_seconds': round(time.time() - t0, 1),
        }
        if excluded_test is not None and len(excluded_test):
            _, held_pred = predict(bundle, bench.X[excluded_test])
            per_seed[seed]['test_excluded_benchmark'] = evaluate(with_healthy(bench.y[excluded_test]), held_pred,
                                                       bench.groups[excluded_test], config)
        with open(run_dir / f'{model_type}_w{int(clean_weight)}_seed{seed}.pkl', 'wb') as fh:
            pickle.dump(bundle, fh)
        logger.info("seed %d: dev micro-F1 %.4f, test micro-F1 %.4f [%.4f, %.4f], macro %.4f",
                    seed, per_seed[seed]['dev']['micro_f1'], per_seed[seed]['test']['micro_f1'],
                    *per_seed[seed]['test']['micro_f1_ci'], per_seed[seed]['test']['macro_f1'])

    summary = {}
    for split in ('dev', 'test'):
        for metric in ('micro_f1', 'macro_f1', 'hamming_loss'):
            values = [per_seed[s][split][metric] for s in seeds]
            summary[f'{split}_{metric}_mean'] = float(np.mean(values))
            summary[f'{split}_{metric}_std'] = float(np.std(values))
    manifest = {
        'model_type': model_type, 'seeds': list(seeds), 'clean_weight': clean_weight,
        'feature_set': feature_set, 'use_production': use_production,
        'hold_out_benchmark': hold_out_benchmark, 'feature_names': list(prod.feature_names),
        'config': {k: v for k, v in config.items() if not k.startswith('_')},
        'config_path': config['_path'], 'git_revision': _git_revision(),
        'feature_schema_version': FEATURE_SCHEMA_VERSION, 'n_features': len(prod.feature_names),
        'inputs': {k: {'path': str(_resolve(config, k)), 'sha256': _sha256(_resolve(config, k))}
                   for k in ('production_features', 'production_labels', 'production_splits',
                             'benchmark_features', 'benchmark_labels')},
        'sizes': {'prod_train': int(len(prod_train)), 'prod_val': int(len(prod.val_idx)),
                  'prod_test': int(len(prod.test_idx)), 'bench_dev': int(len(dev_idx)),
                  'bench_test': int(len(bench.test_idx)),
                  'bench_test_excluded': int(len(excluded_test)) if excluded_test is not None else 0},
        'summary': summary, 'per_seed': {str(s): per_seed[s] for s in seeds},
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(run_dir / 'manifest.json', 'w') as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Run written: %s (test micro-F1 %.4f +/- %.4f over %d seeds)", run_dir,
                summary['test_micro_f1_mean'], summary['test_micro_f1_std'], len(seeds))
    return manifest
