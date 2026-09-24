"""
Benchmark Darshan log layout
============================
How the benchmark sweeps lay out their Darshan logs, shared by feature
extraction and by ground-truth verification.

Compiled MPI benchmarks (IOR, mdtest, h5bench, HACC-IO) write one log per
launch, and one SLURM job may hold several launches (IOR write and read
phases), each of which is a sample. The Python benchmarks (DLIO, the custom
mpi4py script) run with ``DARSHAN_ENABLE_NONMPI=1`` and write one log per
process; the logs of one job are merged by ``parse_benchmark_job`` into one
sample. The SLURM job id in the file name (``_id<jobid>-``) identifies the
job.
"""

import glob
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.data.label_rules import BOTTLENECK_DIMENSIONS, DIMENSION_NAMES, LABEL_DEFINITIONS_PATH
from src.data.parse_darshan import parse_benchmark_job, parse_darshan_log
from src.utils.artifacts import sha256_file, write_atomic
from src.utils.isolation import ChildFailure, run_in_child

logger = logging.getLogger(__name__)

# Benchmark types that produce per-process logs (Python + LD_PRELOAD)
PER_RANK_BENCHMARKS = frozenset({"dlio", "custom"})

# Benchmark types that produce one log per launch (compiled MPI)
AGGREGATED_BENCHMARKS = frozenset({"ior", "mdtest", "h5bench", "hacc_io"})

_JOB_ID_PATTERN = re.compile(r"_id(\d+)-")


def job_id_of(path):
    """SLURM job id encoded in a benchmark log name, or None."""
    match = _JOB_ID_PATTERN.search(os.path.basename(path))
    return match.group(1) if match else None


def group_logs_by_job(log_dir, bench_type):
    """Group the ``.darshan`` files of ``log_dir`` by SLURM job id.

    DLIO's startup probes (lscpu, uname) are left out. A file without a job
    id in its name forms a group of its own, keyed by its base name.

    Returns
    -------
    dict
        ``{job_id: [paths]}`` with the paths sorted.
    """
    jobs = defaultdict(list)
    for fpath in sorted(glob.glob(os.path.join(log_dir, "*.darshan"))):
        basename = os.path.basename(fpath)
        if bench_type == "dlio" and ("_lscpu_" in basename or "_uname_" in basename):
            continue
        jobs[job_id_of(fpath) or basename].append(fpath)
    return dict(jobs)


def iter_benchmark_samples(bench_type, log_dir, timeout=None):
    """Yield ``(job_id, files, parsed, error)`` for every benchmark sample.

    A sample is one merged job for the per-process benchmarks and one log
    for the others. Each sample is parsed in a disposable child process, so
    a crash inside the Darshan library or a timeout (``timeout`` seconds,
    None for no limit) is reported like any other parse failure: ``parsed``
    is None and ``error`` holds the cause (the exception text of the strict
    parser, the fatal signal, or the timeout).
    """
    if bench_type in PER_RANK_BENCHMARKS:
        groups = sorted(group_logs_by_job(log_dir, bench_type).items())
    else:
        groups = [(job_id_of(f), [f])
                  for f in sorted(glob.glob(os.path.join(log_dir, "*.darshan")))]

    for job_id, files in groups:
        error = None
        try:
            if bench_type in PER_RANK_BENCHMARKS:
                parsed = run_in_child(parse_benchmark_job, files, timeout=timeout)
            else:
                parsed = run_in_child(parse_darshan_log, files[0], strict=True, timeout=timeout)
        except ChildFailure as exc:
            error = str(exc)
            logger.error("Cannot parse %s job %s (%d files): %s",
                         bench_type, job_id, len(files), error)
            parsed = None
        yield job_id, files, parsed, error


# ---------------------------------------------------------------------------
# Label manifest: the single source of the constructed labels
# ---------------------------------------------------------------------------

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "benchmark_labels" / "manifest.csv"
MANIFEST_KEYS = ["benchmark", "job_id", "log_file"]
VALIDITY_COLUMNS = [f"valid_{dimension}" for dimension in DIMENSION_NAMES]
MANIFEST_COLUMNS = (
    MANIFEST_KEYS + ["scenario", "source", "note"]
    + DIMENSION_NAMES + VALIDITY_COLUMNS
)
EXCLUDED_SOURCE = "none"
LABEL_SOURCES = {"slurm_out", "step_mapping", EXCLUDED_SOURCE}
VERIFICATION_COLUMNS = {
    "benchmark", "job_id", "first_file", "scenario", "source", "labels", "status"
}
VERIFICATION_SIDECAR_VERSION = 1


def load_manifest(path=DEFAULT_MANIFEST):
    """Read the label manifest and check its contract.

    One row per sample: ``log_file`` is the log's base name for the
    benchmarks with one log per launch and empty for the per-process
    benchmarks (one row per job). ``source`` says where the label came from;
    ``none`` marks a log that exists but has no label and is excluded.
    """
    manifest = pd.read_csv(path, dtype={"job_id": str, "log_file": str, "note": str},
                           keep_default_na=False)
    missing = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"manifest {path} lacks columns {missing}")
    bad = set(manifest["source"]) - LABEL_SOURCES
    if bad:
        raise ValueError(f"manifest {path} has unknown sources {sorted(bad)}")
    if manifest.duplicated(MANIFEST_KEYS).any():
        dup = manifest[manifest.duplicated(MANIFEST_KEYS, keep=False)].iloc[0]
        raise ValueError(f"manifest {path} has duplicate keys, first: {dup[MANIFEST_KEYS].tolist()}")
    labeled = manifest[manifest["source"] != "none"]
    if not labeled[DIMENSION_NAMES].isin([0, 1]).all().all():
        raise ValueError(f"manifest {path} has non-binary labels")
    if not labeled[VALIDITY_COLUMNS].isin([0, 1]).all().all():
        raise ValueError(f"manifest {path} has non-binary validity values")
    valid_problem = [f"valid_{dimension}" for dimension in BOTTLENECK_DIMENSIONS]
    if (labeled[valid_problem].sum(axis=1) == 0).any():
        raise ValueError(f"manifest {path} has labeled rows without a valid problem target")
    for dimension in BOTTLENECK_DIMENSIONS:
        invalid_positive = (
            (labeled[dimension] == 1) & (labeled[f"valid_{dimension}"] == 0)
        )
        if invalid_positive.any():
            raise ValueError(f"manifest {path} labels invalid targets for {dimension}")
    expected_healthy = (labeled[BOTTLENECK_DIMENSIONS].sum(axis=1) == 0).astype(int)
    if not (labeled["healthy"] == expected_healthy).all():
        raise ValueError(f"manifest {path} has inconsistent healthy labels")
    expected_valid_healthy = labeled[valid_problem].all(axis=1).astype(int)
    if not (labeled["valid_healthy"] == expected_valid_healthy).all():
        raise ValueError(f"manifest {path} has inconsistent healthy validity")
    return manifest


def manifest_label_string(row):
    """The labels of a manifest row over its valid targets, as the report writes them."""
    return ",".join(
        f"{dimension}={int(row[dimension])}"
        for dimension in DIMENSION_NAMES if int(row[f"valid_{dimension}"])
    )


def sidecar_path(report_path):
    """The provenance file written next to a verification report."""
    report_path = Path(report_path)
    return report_path.with_name(report_path.name + ".manifest.json")


def write_verification_sidecar(report_path, manifest_path, extra=None):
    """Bind a written report to the exact manifest and label definitions it verified."""
    report_path = Path(report_path)
    payload = {
        "schema_version": VERIFICATION_SIDECAR_VERSION,
        "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "label_manifest": {"path": str(Path(manifest_path)), "sha256": sha256_file(manifest_path)},
        "label_definitions": {"path": str(LABEL_DEFINITIONS_PATH),
                              "sha256": sha256_file(LABEL_DEFINITIONS_PATH)},
    }
    payload.update(extra or {})

    def writer(path):
        with path.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    return write_atomic(sidecar_path(report_path), writer)


def _require_sidecar(report_path, manifest_path):
    """The report must carry a sidecar that names the current manifest and definitions."""
    path = sidecar_path(report_path)
    if not path.is_file():
        raise ValueError(f"verification report {report_path} has no provenance sidecar {path}")
    sidecar = json.loads(path.read_text())
    if sidecar.get("schema_version") != VERIFICATION_SIDECAR_VERSION:
        raise ValueError(f"verification sidecar {path} has an unsupported schema")
    expected = {
        "report": sha256_file(report_path),
        "label_manifest": sha256_file(manifest_path),
        "label_definitions": sha256_file(LABEL_DEFINITIONS_PATH),
    }
    for key, digest in expected.items():
        recorded = sidecar.get(key, {}).get("sha256")
        if recorded != digest:
            raise ValueError(
                f"verification report {report_path} was produced for a different {key} "
                f"(sidecar sha256 {recorded}, current {digest})")
    return sidecar


def validate_verification_report(manifest, report_path, manifest_path, bench_types=None):
    """Require one passing verification row for every labeled manifest row.

    ``manifest`` is ``load_manifest(manifest_path)``. The report's sidecar
    must name this manifest and the current label definitions by hash, and
    every row's scenario, source and label string must equal the manifest's.
    """
    _require_sidecar(report_path, manifest_path)
    report = pd.read_csv(report_path, dtype=str, keep_default_na=False)
    missing_columns = VERIFICATION_COLUMNS - set(report.columns)
    if missing_columns:
        raise ValueError(f"verification report {report_path} lacks columns "
                         f"{sorted(missing_columns)}")
    unknown_benchmarks = set(report['benchmark']) - set(manifest['benchmark'])
    if unknown_benchmarks:
        raise ValueError(f"verification report has unknown benchmarks {sorted(unknown_benchmarks)}")
    selected = set(bench_types) if bench_types is not None else set(manifest['benchmark'])
    expected = manifest[manifest['benchmark'].isin(selected)].copy()
    actual = report[report['benchmark'].isin(selected)].copy()

    expected['_sample_file'] = expected.apply(
        lambda row: row['log_file'] if row['benchmark'] in AGGREGATED_BENCHMARKS else '', axis=1)
    actual['_sample_file'] = actual.apply(
        lambda row: row['first_file'] if row['benchmark'] in AGGREGATED_BENCHMARKS else '', axis=1)
    keys = ['benchmark', 'job_id', '_sample_file']
    if actual.duplicated(keys).any():
        row = actual[actual.duplicated(keys, keep=False)].iloc[0]
        raise ValueError(f"verification report has duplicate sample {row[keys].tolist()}")

    expected_keys = set(map(tuple, expected[keys].to_numpy()))
    actual_keys = set(map(tuple, actual[keys].to_numpy()))
    if expected_keys != actual_keys:
        raise ValueError(
            f"verification report sample set differs from manifest: "
            f"missing={len(expected_keys - actual_keys)}, extra={len(actual_keys - expected_keys)}")

    expected['_labels'] = expected.apply(manifest_label_string, axis=1)
    joined = expected.merge(actual, on=keys, suffixes=('_manifest', '_report'), validate='one_to_one')
    mismatched = joined[
        (joined['scenario_manifest'] != joined['scenario_report'])
        | (joined['source_manifest'] != joined['source_report'])
    ]
    if not mismatched.empty:
        row = mismatched.iloc[0]
        raise ValueError(f"verification metadata differs for {row[keys].tolist()}")
    relabeled = joined[joined['_labels'] != joined['labels']]
    if not relabeled.empty:
        row = relabeled.iloc[0]
        raise ValueError(
            f"verification report labels differ from the manifest for {row[keys].tolist()}: "
            f"report {row['labels']!r}, manifest {row['_labels']!r}")
    excluded = joined['source_manifest'] == EXCLUDED_SOURCE
    bad_excluded = joined[excluded & (joined['status'] != 'excluded')]
    if not bad_excluded.empty:
        raise ValueError("a manifest exclusion is not marked excluded in the verification report")
    bad_labeled = joined[~excluded & (joined['status'] != 'pass')]
    if not bad_labeled.empty:
        counts = bad_labeled['status'].value_counts().to_dict()
        first = bad_labeled.iloc[0]
        raise ValueError(
            f"{len(bad_labeled)} labeled samples did not pass verification {counts}; "
            f"first: {first[keys].tolist()}")
    return {'labeled_pass': int((~excluded).sum()), 'excluded': int(excluded.sum())}


def manifest_row(manifest, bench_type, job_id, files):
    """The manifest row of one sample; ``row["source"] == EXCLUDED_SOURCE``
    marks a sample the manifest excludes (its scenario and note are kept).

    Raises ``KeyError`` when the sample has no row and ``ValueError`` when
    it has more than one, so an unlisted log can never become a sample.
    """
    key = os.path.basename(files[0]) if bench_type in AGGREGATED_BENCHMARKS else ""
    rows = manifest[(manifest["benchmark"] == bench_type)
                    & (manifest["job_id"] == str(job_id))
                    & (manifest["log_file"] == key)]
    if len(rows) == 0:
        raise KeyError(f"no manifest row for {bench_type} job {job_id} {key or '(per-process job)'}")
    if len(rows) > 1:
        raise ValueError(f"{len(rows)} manifest rows for {bench_type} job {job_id} {key}")
    return rows.iloc[0]
