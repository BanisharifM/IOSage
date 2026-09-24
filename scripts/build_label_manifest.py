#!/usr/bin/env python3
"""
Build the benchmark label manifest, the single source of constructed labels.

One row per benchmark sample (one log for IOR, mdtest, h5bench and HACC-IO;
one job for the per-process DLIO and custom runs). Two label sources exist:

- ``slurm_out``: the ``Label:`` line of the job's SLURM stdout in
  ``data/benchmark_results/<benchmark>/<scenario>_<jobid>.out``.
- ``step_mapping``: the April 2026 IOR boost job 17310653, whose 66 steps
  were mapped to labels by SLURM step times in
  ``scripts/run_boost_experiment.py`` and recorded in
  ``results/boost_experiment/new_gt/{new_features,new_labels}.parquet``
  (row i of the one is row i of the other).

A log with neither source gets ``source=none`` and is excluded downstream;
it stays in the manifest so the exclusion is explicit. A job with more than
one stdout file or more than one Label line is an error.

Usage:
    python scripts/build_label_manifest.py [--output data/benchmark_labels/manifest.csv]
"""

import argparse
import glob
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from src.data.benchmark_logs import (  # noqa: E402
    AGGREGATED_BENCHMARKS, MANIFEST_COLUMNS, PER_RANK_BENCHMARKS,
    group_logs_by_job, job_id_of)
from src.data.benchmark_verify import BOTTLENECK_DIMENSIONS, DIMENSION_NAMES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = PROJECT_DIR / "configs" / "benchmarks.yaml"

# These completed jobs changed more factors than their recorded label vector
# describes. They remain listed in the manifest with source=none and cannot
# enter training. The corrected generators apply only to new runs.
INVALID_SCENARIO_PREFIXES = {
    "h5bench": {
        "h5b_indep_small_n": "small requests and single-OST placement were not labeled",
        "h5b_indep_interleaved_": "interface case used the single-OST directory",
        "h5b_interleaved_access_": "healthy label used the single-OST directory",
    },
    "hacc_io": {
        "hacc_posix_shared_large_": "interface case used the single-OST directory",
        "hacc_posix_shared_small_p": "interface case used the single-OST directory",
        "hacc_posix_shared_single_ost_": "duplicate workload carried an incomplete label vector",
    },
}


def label_string_to_dims(label_str):
    """``'access_pattern=1,file_strategy=1'`` to the eight binary columns;
    healthy is 1 exactly when no bottleneck dimension is."""
    dims = {d: 0 for d in DIMENSION_NAMES}
    for part in label_str.split(","):
        key, _, val = part.strip().partition("=")
        if key not in DIMENSION_NAMES:
            raise ValueError(f"unknown dimension {key!r} in {label_str!r}")
        dims[key] = int(val)
    dims["healthy"] = int(sum(dims[d] for d in BOTTLENECK_DIMENSIONS) == 0)
    return dims


def slurm_out_label(results_dir, job_id):
    """``(scenario, label string)`` from the job's stdout, or None when the
    job has no stdout file. More than one file or Label line is an error."""
    outs = glob.glob(os.path.join(results_dir, f"*_{job_id}.out"))
    if not outs:
        return None
    if len(outs) > 1:
        raise ValueError(f"job {job_id}: {len(outs)} stdout files: {outs}")
    labels = [line.strip().split("Label:", 1)[1].strip()
              for line in open(outs[0]) if line.strip().startswith("Label:")]
    if len(labels) != 1:
        raise ValueError(f"job {job_id}: {len(labels)} Label lines in {outs[0]}")
    scenario = os.path.basename(outs[0])[: -len(f"_{job_id}.out")]
    return scenario, labels[0]


def invalid_scenario_reason(benchmark, scenario):
    for prefix, reason in INVALID_SCENARIO_PREFIXES.get(benchmark, {}).items():
        if scenario.startswith(prefix):
            return reason
    return None


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def load_benchmark_config(path):
    """Load the benchmark inventory consumed by this generator."""
    with open(path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError(f"unsupported benchmark config: {path}")
    required = {"paths", "benchmarks", "expected_counts"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"benchmark config lacks {sorted(missing)}")
    supported = AGGREGATED_BENCHMARKS | PER_RANK_BENCHMARKS
    configured = config["benchmarks"]
    if len(configured) != len(set(configured)) or set(configured) != supported:
        raise ValueError("configured benchmarks must list every supported benchmark exactly once")
    if set(config["expected_counts"]) != supported:
        raise ValueError("expected_counts must cover every supported benchmark")
    return config


def boost_rows(features_path, labels_path):
    """Rows of the boost job from the recorded step mapping."""
    if not features_path.exists() or not labels_path.exists():
        logger.warning("boost artifacts not found, no step_mapping rows")
        return {}
    feats = pd.read_parquet(features_path, columns=["_source_path"])
    labs = pd.read_parquet(labels_path)
    if len(feats) != len(labs):
        raise ValueError("boost features and labels differ in length")
    rows = {}
    for src, (_, lab) in zip(feats["_source_path"], labs.iterrows()):
        dims = {d: int(lab[d]) for d in DIMENSION_NAMES}
        rows[os.path.basename(src)] = (str(lab["job_id"]), str(lab["scenario"]), dims)
    return rows


def build(log_base, results_base, benchmarks, boost_features, boost_labels):
    boost = boost_rows(boost_features, boost_labels)
    rows = []
    for bench in benchmarks:
        log_dir = os.path.join(log_base, bench)
        results_dir = os.path.join(results_base, bench)
        if bench in PER_RANK_BENCHMARKS:
            samples = [(job, "", files) for job, files in sorted(group_logs_by_job(log_dir, bench).items())]
        else:
            samples = [(job_id_of(f), os.path.basename(f), [f])
                       for f in sorted(glob.glob(os.path.join(log_dir, "*.darshan")))]
        for job_id, log_file, files in samples:
            row = {"benchmark": bench, "job_id": job_id, "log_file": log_file}
            found = slurm_out_label(results_dir, job_id) if job_id else None
            if found is not None:
                scenario, label_str = found
                invalid_reason = invalid_scenario_reason(bench, scenario)
                if invalid_reason:
                    row.update(scenario=scenario, source="none", note=invalid_reason,
                               **{d: 0 for d in DIMENSION_NAMES})
                else:
                    row.update(scenario=scenario, source="slurm_out", note="",
                               **label_string_to_dims(label_str))
            elif log_file in boost:
                _, scenario, dims = boost[log_file]
                row.update(scenario=scenario, source="step_mapping",
                           note="SLURM step times, scripts/run_boost_experiment.py", **dims)
            else:
                row.update(scenario="", source="none",
                           note="no SLURM stdout and no step mapping", **{d: 0 for d in DIMENSION_NAMES})
            rows.append(row)
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--log-dir")
    parser.add_argument("--results-dir")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_benchmark_config(args.config)
    paths = config["paths"]
    log_dir = Path(args.log_dir).resolve() if args.log_dir else _project_path(paths["log_dir"])
    results_dir = (Path(args.results_dir).resolve() if args.results_dir
                   else _project_path(paths["results_dir"]))
    output = Path(args.output).resolve() if args.output else _project_path(paths["output"])
    manifest = build(
        log_dir,
        results_dir,
        config["benchmarks"],
        _project_path(paths["boost_features"]),
        _project_path(paths["boost_labels"]),
    )
    actual_counts = {}
    for (benchmark, source), count in manifest.groupby(["benchmark", "source"]).size().items():
        actual_counts.setdefault(benchmark, {})[source] = int(count)
    if actual_counts != config["expected_counts"]:
        raise RuntimeError(
            "generated benchmark inventory differs from configs/benchmarks.yaml: "
            f"actual={actual_counts}, expected={config['expected_counts']}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = output.with_name(output.name + ".manifest.json")
    existing = [str(path) for path in (output, provenance_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to replace label manifest artifacts: {existing}")
    token = f"{os.getpid()}.{time.time_ns()}"
    output_tmp = output.with_name(output.name + f".tmp.{token}")
    manifest.to_csv(output_tmp, index=False)
    os.rename(output_tmp, output)

    def sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    provenance = {
        "schema_version": 1,
        "status": "passed",
        "config": {"path": str(Path(args.config).resolve()), "sha256": sha256(args.config)},
        "script": {"path": str(Path(__file__).resolve()),
                   "sha256": sha256(Path(__file__).resolve())},
        "inputs": {
            "log_dir": str(log_dir.resolve()),
            "results_dir": str(results_dir.resolve()),
            "boost_features": str(_project_path(paths["boost_features"]).resolve()),
            "boost_labels": str(_project_path(paths["boost_labels"]).resolve()),
        },
        "counts": actual_counts,
        "output": {"path": str(output), "sha256": sha256(output), "rows": len(manifest)},
    }
    provenance_tmp = provenance_path.with_name(provenance_path.name + f".tmp.{token}")
    with open(provenance_tmp, "x") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.rename(provenance_tmp, provenance_path)
    summary = manifest.groupby(["benchmark", "source"]).size()
    logger.info("Manifest written: %s (%d rows)\n%s", output, len(manifest), summary.to_string())
    logger.info("Provenance: %s", provenance_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
