#!/usr/bin/env python3
"""
Build the benchmark label manifest, the single source of constructed labels.

One row per benchmark sample (one log for IOR, mdtest, h5bench and HACC-IO;
one job for the per-process DLIO and custom runs). Two label sources exist:

- ``slurm_out``: the ``Label:`` line of the job's SLURM stdout in
  ``data/benchmark_results/<benchmark>/<scenario>_<jobid>.out``.
- ``step_mapping``: the April 2026 IOR boost job 17310653, whose 66 steps
  were mapped to labels by SLURM step times using the source at Git commit
  ``83083ed:scripts/run_boost_experiment.py`` and recorded in
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
import json
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from src.data.benchmark_logs import (  # noqa: E402
    AGGREGATED_BENCHMARKS, MANIFEST_COLUMNS, PER_RANK_BENCHMARKS,
    VALIDITY_COLUMNS, group_logs_by_job, job_id_of)
from src.data.label_rules import (  # noqa: E402
    BOTTLENECK_DIMENSIONS, DIMENSION_NAMES, LABEL_DEFINITIONS_PATH,
)
from src.utils.artifacts import sha256_file, write_atomic  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = PROJECT_DIR / "configs" / "benchmarks.yaml"
EXCLUDED_LABEL = "classifier_excluded"

def label_string_to_dims(label_str):
    """``'access_pattern=1,file_strategy=1'`` to the binary label columns;
    healthy is 1 exactly when no bottleneck dimension is."""
    dims = {d: 0 for d in DIMENSION_NAMES}
    stated_healthy = None
    for part in label_str.split(","):
        key, _, val = part.strip().partition("=")
        if key not in DIMENSION_NAMES:
            raise ValueError(f"unknown dimension {key!r} in {label_str!r}")
        if val not in {"0", "1"}:
            raise ValueError(f"dimension {key!r} is not binary in {label_str!r}")
        if key == "healthy":
            stated_healthy = int(val)
        else:
            dims[key] = int(val)
    derived_healthy = int(sum(dims[d] for d in BOTTLENECK_DIMENSIONS) == 0)
    if stated_healthy is not None and stated_healthy != derived_healthy:
        raise ValueError(f"healthy is inconsistent with problem labels in {label_str!r}")
    dims["healthy"] = derived_healthy
    return dims


def slurm_out_label(results_dir, job_id):
    """``(scenario, label string)`` from the job's stdout, or None when the
    job has no stdout file. More than one file or Label line is an error."""
    outs = glob.glob(os.path.join(results_dir, f"*_{job_id}.out"))
    if not outs:
        return None
    if len(outs) > 1:
        raise ValueError(f"job {job_id}: {len(outs)} stdout files: {outs}")
    with open(outs[0]) as handle:
        labels = [line.strip().split("Label:", 1)[1].strip()
                  for line in handle if line.strip().startswith("Label:")]
    if len(labels) != 1:
        raise ValueError(f"job {job_id}: {len(labels)} Label lines in {outs[0]}")
    scenario = os.path.basename(outs[0])[: -len(f"_{job_id}.out")]
    return scenario, labels[0]


def _contract(positives=(), negatives=()):
    """Create labels and validity for explicitly controlled targets."""
    positives = set(positives)
    negatives = set(negatives)
    overlap = positives & negatives
    if overlap:
        raise ValueError(f"targets cannot be both positive and negative: {sorted(overlap)}")
    unknown = (positives | negatives) - set(BOTTLENECK_DIMENSIONS)
    if unknown:
        raise ValueError(f"unknown contract targets: {sorted(unknown)}")
    labels = {dimension: int(dimension in positives) for dimension in BOTTLENECK_DIMENSIONS}
    labels["healthy"] = int(not positives)
    validity = {
        f"valid_{dimension}": int(dimension in positives or dimension in negatives)
        for dimension in BOTTLENECK_DIMENSIONS
    }
    validity["valid_healthy"] = int(all(
        validity[f"valid_{dimension}"] for dimension in BOTTLENECK_DIMENSIONS
    ))
    return labels, validity


def _scenario_rank(scenario):
    match = re.search(r"_n(\d+)(?:_|$)", scenario)
    return int(match.group(1)) if match else None


def apply_manifest_policy(benchmark, scenario, dimensions=None):
    """Return the audited target contract for one constructed sample.

    The stdout label is accepted as provenance but is not treated as a full
    negative vector. Only targets controlled by the generator are valid.
    """
    del dimensions
    positive = ()
    negative = ()
    reason = ""

    if benchmark == "custom":
        if scenario.startswith("custom_imbalance_"):
            positive = ("parallelism_efficiency",)
        elif scenario.startswith("custom_balanced_"):
            negative = ("parallelism_efficiency",)
    elif benchmark == "dlio":
        reason = "DLIO aggregate counters do not distinguish the registered scenario target"
    elif benchmark == "h5bench":
        rank = _scenario_rank(scenario)
        if scenario.startswith("h5b_collective_small_"):
            positive = ("access_granularity",)
            negative = ("interface_choice",)
        elif scenario.startswith("h5b_collective_large_healthy_"):
            negative = ("access_granularity", "metadata_intensity", "interface_choice")
        elif scenario.startswith("h5b_indep_large_healthy_"):
            negative = ("access_granularity",)
        elif scenario.startswith("h5b_interleaved_access_"):
            negative = ("access_pattern",)
        elif scenario.startswith("h5b_indep_small_interleaved_") and rank == 64:
            positive = ("access_granularity", "interface_choice")
        elif scenario.startswith("h5b_indep_small_") \
                and not scenario.startswith("h5b_indep_small_single_ost_"):
            positive = ("access_granularity", "interface_choice")
        else:
            reason = "scenario does not meet a registered target rule or has a storage-layout confound"
    elif benchmark == "hacc_io":
        if scenario.startswith("hacc_fpp_healthy_"):
            negative = ("file_strategy",)
        else:
            reason = "HACC construction does not isolate a registered target"
    elif benchmark == "ior":
        if scenario.startswith(("ior_small_posix_", "ior_small_direct_")):
            positive = ("access_granularity",)
        elif scenario.startswith("ior_misaligned_"):
            positive = ("access_granularity", "request_alignment")
        elif scenario.startswith("ior_random_small_"):
            positive = ("access_granularity", "access_pattern")
        elif scenario.startswith("ior_random_posix_"):
            positive = ("access_pattern",)
        elif scenario.startswith("ior_interface_mpiio_indep_"):
            positive = ("interface_choice",)
        elif scenario.startswith("ior_fsync_per_write_"):
            positive = ("throughput_utilization",)
        elif scenario.startswith("ior_healthy_collective_"):
            negative = ("access_granularity", "metadata_intensity", "interface_choice")
        elif scenario.startswith("ior_healthy_posix_fpp_"):
            negative = ("access_granularity", "metadata_intensity", "access_pattern",
                        "file_strategy", "throughput_utilization")
        elif scenario.startswith("ior_healthy_large_seq_"):
            negative = ("access_granularity", "metadata_intensity", "access_pattern",
                        "file_strategy", "throughput_utilization")
        elif scenario.startswith("ior_e2e_mpiio_coll_"):
            negative = ("interface_choice",)
        elif scenario.startswith("ior_io500_hard_"):
            positive = ("access_granularity",)
        elif scenario.startswith("17310653."):
            step = int(scenario.rsplit(".", 1)[1])
            if step < 24:
                positive = ("access_pattern",)
            elif step < 42:
                reason = "file-per-process run has too few files for the many-small-files class"
            elif step < 60:
                positive = ("throughput_utilization",)
            else:
                negative = ("access_granularity", "access_pattern", "request_alignment",
                            "interface_choice", "throughput_utilization")
        else:
            reason = "scenario has no audited target contract"
    elif benchmark == "mdtest":
        if scenario.startswith(("mdtest_meta_shared_", "mdtest_meta_unique_configured_",
                                "mdtest_deep_tree_", "mdtest_io500_easy_")):
            positive = ("metadata_intensity",)
        elif scenario.startswith("mdtest_meta_unique_"):
            reason = "stored run has a partial POSIX record set"
        elif scenario.startswith("mdtest_io500_hard_"):
            positive = ("metadata_intensity", "file_strategy")
        elif scenario.startswith("mdtest_fpp_explosion_"):
            positive = ("file_strategy",)
        elif scenario.startswith("mdtest_healthy_"):
            negative = ("metadata_intensity",)
        else:
            reason = "metadata construction is not a stable registered control"

    if not positive and not negative:
        return None, None, reason or "scenario has no audited target contract"
    labels, validity = _contract(positive, negative)
    controlled = ",".join(sorted(set(positive) | set(negative)))
    return labels, validity, f"audited targets: {controlled}"


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def _recorded_path(value):
    """Use a repository-relative path when the artifact is inside the project."""
    path = Path(value).resolve()
    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


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
        raise FileNotFoundError(
            f"boost artifacts are required: {features_path}, {labels_path}"
        )
    feats = pd.read_parquet(features_path, columns=["_source_path"])
    labs = pd.read_parquet(labels_path)
    if len(feats) != len(labs):
        raise ValueError("boost features and labels differ in length")
    rows = {}
    for src, (_, lab) in zip(feats["_source_path"], labs.iterrows()):
        dims = {d: int(lab[d]) if d in lab else 0 for d in DIMENSION_NAMES}
        dims["healthy"] = int(not any(dims[d] for d in BOTTLENECK_DIMENSIONS))
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
                if label_str == EXCLUDED_LABEL:
                    dims, validity = None, None
                    note = "generator marks scenario as excluded from classifier ground truth"
                else:
                    recorded_dims = label_string_to_dims(label_str)
                    dims, validity, note = apply_manifest_policy(
                        bench, scenario, recorded_dims
                    )
                if dims is None:
                    row.update(scenario=scenario, source="none", note=note,
                               **{d: 0 for d in DIMENSION_NAMES},
                               **{d: 0 for d in VALIDITY_COLUMNS})
                else:
                    row.update(scenario=scenario, source="slurm_out", note=note,
                               **dims, **validity)
            elif log_file in boost:
                _, scenario, dims = boost[log_file]
                dims, validity, note = apply_manifest_policy(bench, scenario, dims)
                if dims is None:
                    row.update(scenario=scenario, source="none", note=note,
                               **{d: 0 for d in DIMENSION_NAMES},
                               **{d: 0 for d in VALIDITY_COLUMNS})
                else:
                    provenance = (
                        "SLURM step mapping from results/boost_experiment/new_gt; "
                        "source at Git commit 83083ed:scripts/run_boost_experiment.py"
                    )
                    row.update(scenario=scenario, source="step_mapping",
                               note="; ".join(filter(None, (provenance, note))),
                               **dims, **validity)
            else:
                row.update(scenario="", source="none",
                           note="no SLURM stdout and no step mapping",
                           **{d: 0 for d in DIMENSION_NAMES},
                           **{d: 0 for d in VALIDITY_COLUMNS})
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
    write_atomic(output, lambda path: manifest.to_csv(path, index=False))

    provenance = {
        "schema_version": 1,
        "status": "passed",
        "config": {"path": _recorded_path(args.config), "sha256": sha256_file(args.config)},
        "label_definitions": {
            "path": _recorded_path(LABEL_DEFINITIONS_PATH),
            "sha256": sha256_file(LABEL_DEFINITIONS_PATH),
        },
        "script": {"path": _recorded_path(__file__),
                   "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": {
            "log_dir": _recorded_path(log_dir),
            "results_dir": _recorded_path(results_dir),
            "boost_features": _recorded_path(_project_path(paths["boost_features"])),
            "boost_labels": _recorded_path(_project_path(paths["boost_labels"])),
        },
        "counts": actual_counts,
        "output": {"path": _recorded_path(output),
                   "sha256": sha256_file(output), "rows": len(manifest)},
    }
    def write_provenance(path):
        with path.open("x") as handle:
            json.dump(provenance, handle, indent=2, sort_keys=True)
            handle.write("\n")

    write_atomic(provenance_path, write_provenance)
    summary = manifest.groupby(["benchmark", "source"]).size()
    logger.info("Manifest written: %s (%d rows)\n%s", output, len(manifest), summary.to_string())
    logger.info("Provenance: %s", provenance_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
