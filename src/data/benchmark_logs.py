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
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import darshan
import pandas as pd

from src.data.benchmark_verify import DIMENSION_NAMES
from src.data.parse_darshan import parse_benchmark_job, parse_darshan_log

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


def iter_benchmark_samples(bench_type, log_dir):
    """Yield ``(job_id, files, parsed)`` for every sample of a benchmark.

    A sample is one merged job for the per-process benchmarks and one log
    for the others. ``parsed`` is None when the sample could not be parsed;
    the error is logged with the job id so the caller can count it.
    """
    if bench_type in PER_RANK_BENCHMARKS:
        groups = sorted(group_logs_by_job(log_dir, bench_type).items())
    else:
        groups = [(job_id_of(f), [f])
                  for f in sorted(glob.glob(os.path.join(log_dir, "*.darshan")))]

    for job_id, files in groups:
        try:
            if bench_type in PER_RANK_BENCHMARKS:
                parsed = parse_benchmark_job(files)
            else:
                parsed = parse_darshan_log(files[0])
                if parsed is None:
                    raise ValueError("parse_darshan_log returned None")
        except ValueError as exc:
            logger.error("Cannot parse %s job %s (%d files): %s",
                         bench_type, job_id, len(files), exc)
            parsed = None
        yield job_id, files, parsed


# ---------------------------------------------------------------------------
# Label manifest: the single source of the constructed labels
# ---------------------------------------------------------------------------

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "benchmark_labels" / "manifest.csv"
MANIFEST_KEYS = ["benchmark", "job_id", "log_file"]
MANIFEST_COLUMNS = MANIFEST_KEYS + ["scenario", "source", "note"] + DIMENSION_NAMES
LABEL_SOURCES = {"slurm_out", "step_mapping", "none"}


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
    if (labeled[DIMENSION_NAMES].sum(axis=1) == 0).any():
        raise ValueError(f"manifest {path} has labeled rows without any dimension")
    return manifest


def manifest_row(manifest, bench_type, job_id, files):
    """The manifest row of one sample, or None when the row says ``none``.

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
    row = rows.iloc[0]
    return None if row["source"] == "none" else row


def posix_file_facts(files):
    """Per-file facts the verification rules need from the raw POSIX records:
    ``offsets`` (record id to the highest byte offset read or written) and
    ``data_files`` (distinct files with any bytes moved, standard streams
    left out)."""
    offsets = {}
    data_files = set()
    for path in files:
        report = darshan.DarshanReport(str(path), read_all=False)
        if "POSIX" not in report.modules:
            continue
        report.mod_read_all_records("POSIX")
        df = report.records["POSIX"].to_df()["counters"]
        names = getattr(report, "name_records", {}) or {}
        per_file = df.groupby("id").agg(offset=("POSIX_MAX_BYTE_WRITTEN", "max"),
                                        offset_r=("POSIX_MAX_BYTE_READ", "max"),
                                        bytes=("POSIX_BYTES_READ", "sum"), written=("POSIX_BYTES_WRITTEN", "sum"))
        for rid, row in per_file.iterrows():
            rid = int(rid)
            offsets[rid] = max(offsets.get(rid, 0), int(row["offset"]), int(row["offset_r"]))
            if row["bytes"] + row["written"] > 0 and names.get(rid) not in ("<STDIN>", "<STDOUT>", "<STDERR>"):
                data_files.add(rid)
    return {"offsets": offsets, "data_files": len(data_files)}
