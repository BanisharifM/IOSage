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
import logging
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from src.data.benchmark_logs import (  # noqa: E402
    AGGREGATED_BENCHMARKS, DEFAULT_MANIFEST, MANIFEST_COLUMNS, PER_RANK_BENCHMARKS,
    group_logs_by_job, job_id_of)
from src.data.benchmark_verify import BOTTLENECK_DIMENSIONS, DIMENSION_NAMES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BOOST_FEATURES = PROJECT_DIR / "results" / "boost_experiment" / "new_gt" / "new_features.parquet"
BOOST_LABELS = PROJECT_DIR / "results" / "boost_experiment" / "new_gt" / "new_labels.parquet"


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


def boost_rows():
    """Rows of the boost job from the recorded step mapping."""
    if not BOOST_FEATURES.exists() or not BOOST_LABELS.exists():
        logger.warning("boost artifacts not found, no step_mapping rows")
        return {}
    feats = pd.read_parquet(BOOST_FEATURES, columns=["_source_path"])
    labs = pd.read_parquet(BOOST_LABELS)
    if len(feats) != len(labs):
        raise ValueError("boost features and labels differ in length")
    rows = {}
    for src, (_, lab) in zip(feats["_source_path"], labs.iterrows()):
        dims = {d: int(lab[d]) for d in DIMENSION_NAMES}
        rows[os.path.basename(src)] = (str(lab["job_id"]), str(lab["scenario"]), dims)
    return rows


def build(log_base, results_base):
    boost = boost_rows()
    rows = []
    for bench in sorted(AGGREGATED_BENCHMARKS | PER_RANK_BENCHMARKS):
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
                row.update(scenario=scenario, source="slurm_out", note="", **label_string_to_dims(label_str))
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
    parser.add_argument("--log-dir", default=str(PROJECT_DIR / "data" / "benchmark_logs"))
    parser.add_argument("--results-dir", default=str(PROJECT_DIR / "data" / "benchmark_results"))
    parser.add_argument("--output", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()

    manifest = build(args.log_dir, args.results_dir)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    summary = manifest.groupby(["benchmark", "source"]).size()
    logger.info("Manifest written: %s (%d rows)\n%s", args.output, len(manifest), summary.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
