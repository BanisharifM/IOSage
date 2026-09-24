"""Closed-loop objective: wall time with a work-invariance guard.

Why this module exists (CODE_FIXES A1/A2, CLOSED_LOOP_RECOMPUTE.md section 6):
the closed loop used to score an iteration by the Darshan write-bandwidth ratio
(POSIX_BYTES_WRITTEN / POSIX_F_WRITE_TIME).  That ratio is built from
rank-summed cumulative counters, so a fix that removes bytes (NetCDF NOFILL on the
E2E kernel: 0.67x by bandwidth, 6.4x by wall time) looks like a regression and a
proposal that skips work (mdtest writing 100x fewer bytes) looks like a 130x win.

What replaces it, following the measurement literature:
* the cost being optimised is execution time, and speedup is a ratio of times
  (Hoefler and Belli, SC'15, Rule 1; Section 2.1.1);
* the wall time of a job is the Darshan job ``run_time``: darshan-core reduces
  ``start_time`` with MPI_MIN and ``end_time`` with MPI_MAX over the ranks, so it
  is the elapsed time of the instrumented process group.  Multi-phase jobs
  (h5bench write then read, DLIO datagen then training) produce several logs;
  concurrent logs are one phase (longest rank wins), sequential logs add up;
* nondeterministic measurements are summarised by the median with the observed
  spread reported next to it, not by a single run (Hoefler and Belli Rules 5-8;
  Kalibera and Jones, effect-size confidence intervals);
* the work must be the same before a ratio of times means anything: bytes moved
  (from every log of the job) or the benchmark's configured work must agree with
  the baseline within ``work_tolerance``.

Bandwidth is kept as a secondary, recorded quantity only.
"""

from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import Iterable

from ..data.parse_darshan import parse_darshan_log

logger = logging.getLogger(__name__)

# Fraction by which bytes (or configured work) may differ from the baseline
# before the iteration is rejected as "changed the workload".
DEFAULT_WORK_TOLERANCE = 0.25
# Wall-time speedup below this is a regression that triggers rollback
# (time-based counterpart of the old 0.9x bandwidth rule).
DEFAULT_REGRESSION_FACTOR = 0.9
# Minimum relative gain over the baseline's own spread before a candidate is
# accepted as an improvement; guards against accepting run-to-run noise.
DEFAULT_MIN_GAIN = 0.05
# Confidence level of the median's interval used to compare two configurations
# (Hoefler & Belli SC'15, Rules 5-7). 0.90 with 8 runs uses the 2nd and 7th order statistic
# (coverage 93.0%), so one outlier on either side cannot move the interval; this is the
# 8-run / 90% practice of STELLAR. Fewer than 5 runs cannot reach 90% at all.
DEFAULT_CONFIDENCE = 0.90
# Darshan's default MAX_RECORDS per module per rank; a job whose opens reach
# 2 x this per rank has truncated counters (A3).
DARSHAN_DEFAULT_MAX_RECORDS = 1024


def darshan_summary(path):
    """Job metadata and the POSIX counters the objective needs, for one log."""
    parsed = parse_darshan_log(str(path))
    if parsed is None:
        return None
    job, c = parsed["job"], parsed["counters"]
    return {
        "path": str(path),
        "name": Path(path).name,
        "start": float(job.get("start_time", 0) or 0),
        "end": float(job.get("end_time", 0) or 0),
        "runtime": float(job.get("runtime", 0.0) or 0.0),
        "nprocs": int(job.get("nprocs", 1) or 1),
        "bytes_written": float(c.get("POSIX_BYTES_WRITTEN", 0) or 0),
        "bytes_read": float(c.get("POSIX_BYTES_READ", 0) or 0),
        "writes": float(c.get("POSIX_WRITES", 0) or 0),
        "reads": float(c.get("POSIX_READS", 0) or 0),
        "opens": float(c.get("POSIX_OPENS", 0) or 0),
        "write_time": float(c.get("POSIX_F_WRITE_TIME", 0) or 0),
        "meta_time": float(c.get("POSIX_F_META_TIME", 0) or 0),
    }


def phase_walltime(summaries):
    """Sum the elapsed interval union represented by all logs.

    Logs whose start falls inside an earlier log's window are concurrent ranks
    of the same phase (DLIO per-rank NONMPI logs); logs that start after the
    previous phase ended are a new sequential phase (h5bench write then read).
    Returns (walltime_s, n_phases).
    """
    phases = []
    for s in sorted(summaries, key=lambda x: (x["start"], x["runtime"])):
        start = float(s["start"])
        end = start + float(s["runtime"])
        if phases and start <= phases[-1]["end"]:
            phases[-1]["end"] = max(phases[-1]["end"], end)
        else:
            phases.append({"start": start, "end": end})
    return sum(p["end"] - p["start"] for p in phases), len(phases)


def select_primary_log(summaries, benchmark_type=None):
    """The log whose counters feed the classifier (A2).

    h5bench: the write-phase log (its name carries ``h5bench_write``); DLIO: the
    per-rank log that moved the most bytes (the training phase); otherwise the
    earliest-started log with the most bytes written.
    """
    if not summaries:
        return None
    if benchmark_type == "h5bench":
        for s in summaries:
            if "h5bench_write" in s["name"]:
                return s
    if benchmark_type == "dlio":
        return max(summaries, key=lambda s: s["bytes_read"] + s["bytes_written"])
    return sorted(summaries, key=lambda s: (s["start"], -s["bytes_written"]))[0]


# Parameters that define the amount of work a benchmark does.  The closed loop
# may change how I/O is done, never how much: a proposal that touches one of
# these is rejected before it runs (the mdtest "130x" came from cutting
# items_per_rank/write_bytes; computation_time would fake DLIO speedups).
WORK_PARAMS = {
    "ior": ("block_size", "segments"),
    "mdtest": ("items_per_rank", "write_bytes", "files_only"),
    "hacc_io": ("num_particles"),
    "custom": ("base_size_mb",),
    "h5bench": ("DIM_1", "DIM_2", "DIM_3", "TIMESTEPS"),
    "dlio": ("num_files_train", "num_samples_per_file", "record_length", "epochs",
             "computation_time"),
}


def _size_bytes(v):
    """IOR-style size strings ('1m', '64k', '2g') or numbers -> float value.

    Floats are kept as floats (truncating would make computation_time 0.1
    equal to 0.0); size suffixes expand to bytes.
    """
    if v is None:
        return 0.0
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    t = str(v).strip().lower()
    mult = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}
    if t and t[-1] in mult:
        return float(t[:-1]) * mult[t[-1]]
    return float(t)


def work_params_changed(benchmark_type, baseline_config, new_config):
    """Names of work-defining parameters whose value differs from the baseline."""
    keys = WORK_PARAMS.get(benchmark_type, ())
    if isinstance(keys, str):
        keys = (keys,)
    changed = []
    for k in keys:
        b, n = (baseline_config or {}).get(k), (new_config or {}).get(k)
        if b is None or n is None:
            continue
        try:
            same = abs(_size_bytes(b) - _size_bytes(n)) <= 1e-9 * max(1.0, abs(_size_bytes(b)))
        except (TypeError, ValueError):
            same = str(b) == str(n)
        if not same:
            changed.append(k)
    return changed


def configured_work(workload, config, nprocs, benchmark_type=None):
    """Logical work implied by the benchmark configuration, independent of
    how the I/O is done.

    This, not bytes seen by Darshan, is the invariant of a fair speedup: a
    correct fix may move fewer bytes through the I/O stack (NetCDF NOFILL on
    the E2E kernel writes 0.43x the bytes for the same output; compression
    shrinks files) while the amount of data the application produces is
    unchanged.  Only the ratio between two configurations matters, so unit
    constants are irrelevant.  Returns (work, items) or (None, None) when the
    suite has no closed form here.
    """
    if config is None:
        return None, None
    bt = benchmark_type or (workload or "").split("_")[0]
    if bt == "hacc":
        bt = "hacc_io"
    try:
        if bt == "mdtest":
            items = int(config.get("items_per_rank", 0))
            wb = int(config.get("write_bytes", 0))
            return items * max(wb, 1) * nprocs, items * nprocs
        if bt == "dlio":
            nf = int(config.get("num_files_train", 0))
            ns = int(config.get("num_samples_per_file", 1))
            rl = int(config.get("record_length", 0))
            ep = int(config.get("epochs", 1))
            return nf * ns * rl * ep, nf
        if bt == "ior":
            return _size_bytes(config.get("block_size")) * float(config.get("segments", 1)) * nprocs, None
        if bt == "hacc_io":
            return int(config.get("num_particles", 0)) * nprocs, None
        if bt == "custom":
            return float(config.get("base_size_mb", 0)) * nprocs, None
        if bt == "h5bench":
            dims = 1
            for k in ("DIM_1", "DIM_2", "DIM_3"):
                dims *= int(config.get(k, 1) or 1)
            return dims * int(config.get("TIMESTEPS", 1) or 1) * nprocs, None
    except (TypeError, ValueError):
        return None, None
    return None, None


def job_measurement(log_paths: Iterable, benchmark_type=None, workload=None, config=None):
    """Wall time and work of one SLURM job from all of its Darshan logs.

    Returns None when no log parses.  ``walltime_s`` is the phase-summed job
    runtime; ``bytes_total`` is written + read over every log; ``primary_log``
    is the log to use for classifier features.
    """
    summaries = [s for s in (darshan_summary(p) for p in log_paths) if s]
    if not summaries:
        return None
    walltime, n_phases = phase_walltime(summaries)
    primary = select_primary_log(summaries, benchmark_type)
    agg = {k: sum(s[k] for s in summaries)
           for k in ("bytes_written", "bytes_read", "writes", "reads", "opens",
                     "write_time", "meta_time")}
    nprocs = max(s["nprocs"] for s in summaries)
    cap_hit = any(s["opens"] >= 2 * DARSHAN_DEFAULT_MAX_RECORDS * s["nprocs"]
                  for s in summaries)
    cfg_bytes, cfg_files = configured_work(workload, config, nprocs, benchmark_type)
    m = {
        "walltime_s": walltime,
        "n_phases": n_phases,
        "n_logs": len(summaries),
        "primary_log": primary["path"],
        "nprocs": nprocs,
        "bytes_total": agg["bytes_written"] + agg["bytes_read"],
        "configured_bytes": cfg_bytes,
        "configured_files": cfg_files,
        "darshan_record_cap_hit": cap_hit,
        "write_bw_mb_s": (agg["bytes_written"] / agg["write_time"] / 1e6)
        if agg["write_time"] > 0 else 0.0,
        **agg,
    }
    if cap_hit:
        logger.warning("darshan_record_cap_hit: opens per rank reached the "
                       "Darshan MAX_RECORDS default; counters are truncated")
    return m


def work_ratio(new, base):
    """(ratio, source): configured work when both sides have it, else bytes.

    Only a ``configured`` ratio can reject an iteration; a ``darshan`` bytes
    ratio is recorded as advisory, because bytes moved are not work (see
    ``configured_work``).
    """
    if new.get("configured_bytes") and base.get("configured_bytes"):
        return new["configured_bytes"] / base["configured_bytes"], "configured"
    if base.get("bytes_total"):
        return new.get("bytes_total", 0.0) / base["bytes_total"], "darshan"
    return None, "unknown"


def median_ci(values, confidence=DEFAULT_CONFIDENCE):
    """Distribution-free confidence interval for the median from order statistics.

    For n sorted runs, the interval [x(k), x(n+1-k)] covers the true median with probability
    1 - 2 * P(Binomial(n, 1/2) <= k-1), whatever the distribution (no normality assumed,
    Hoefler & Belli Rule 6). The largest k whose coverage still reaches ``confidence`` gives
    the narrowest valid interval. If even [min, max] falls short (n too small), that interval
    is returned with ``valid`` False and its true coverage, so the caller can refuse to decide.
    """
    from math import comb
    vals = sorted(v for v in values if v and v > 0)
    n = len(vals)
    if n == 0:
        return None
    def coverage(k):
        return 1.0 - 2.0 * sum(comb(n, i) for i in range(k)) / 2.0 ** n
    best_k = 1
    for k in range(1, n // 2 + 1):
        if coverage(k) >= confidence:
            best_k = k
    cov = coverage(best_k) if n > 1 else 0.0
    return {"lower": vals[best_k - 1], "upper": vals[n - best_k], "coverage": round(cov, 4),
            "rank": best_k, "n": n, "valid": n > 1 and cov >= confidence}


def aggregate_repeats(measurements, confidence=DEFAULT_CONFIDENCE):
    """Median wall time over repeated runs with its confidence interval (Rules 5-8).

    ``ci_lower``/``ci_upper`` bound the median (see ``median_ci``); ``rel_mad`` is the
    outlier-resistant dispersion. ``spread_rel`` = (max - min) / median is kept as a diagnostic only: on a
    shared filesystem a single burst from another user inflates it by an order of magnitude.
    """
    ms = [m for m in measurements if m and m.get("walltime_s", 0) > 0]
    if not ms:
        return None
    wt = [m["walltime_s"] for m in ms]
    med = statistics.median(wt)
    ci = median_ci(wt, confidence)
    agg = dict(ms[0])            # counters/work from the first run
    agg.update({
        "walltime_s": med,
        "walltime_runs_s": wt,
        "walltime_min_s": min(wt),
        "walltime_max_s": max(wt),
        "spread_rel": (max(wt) - min(wt)) / med if med > 0 else 0.0,
        "rel_mad": _rel_mad(wt),
        "ci_lower_s": ci["lower"], "ci_upper_s": ci["upper"],
        "ci_coverage": ci["coverage"], "ci_valid": ci["valid"],
        "n_repeats": len(wt),
        "write_bw_mb_s": statistics.median(m["write_bw_mb_s"] for m in ms),
        "bytes_total": statistics.median(m["bytes_total"] for m in ms),
    })
    return agg


def _rel_mad(values):
    """Median absolute deviation relative to the median.

    Used instead of (max - min) / median because a single outlier run, which a shared
    filesystem produces regularly, moves the range by an order of magnitude while the
    median absolute deviation stays close to the typical run-to-run difference.
    """
    vals = [v for v in values if v and v > 0]
    if len(vals) < 2:
        return 0.0
    med = statistics.median(vals)
    if med <= 0:
        return 0.0
    return statistics.median([abs(v - med) for v in vals]) / med


def evaluate_candidate(base, new, best_speedup, work_tolerance=DEFAULT_WORK_TOLERANCE,
                       regression_factor=DEFAULT_REGRESSION_FACTOR, min_gain=DEFAULT_MIN_GAIN):
    """Score an iteration against the baseline and the best so far.

    Returns a dict with ``speedup`` (median wall-time ratio), ``bw_speedup``
    (secondary), ``work_ratio``/``work_source``, and the decision flags
    ``rejected_work_changed``, ``accepted`` (new best) and ``regression``.
    """
    speedup = base["walltime_s"] / new["walltime_s"] if new["walltime_s"] > 0 else 0.0
    bw_speedup = (new["write_bw_mb_s"] / base["write_bw_mb_s"]
                  if base.get("write_bw_mb_s") else None)
    ratio, source = work_ratio(new, base)
    work_changed = source == "configured" and abs(ratio - 1.0) > work_tolerance
    bytes_ratio = (new.get("bytes_total", 0.0) / base["bytes_total"]) if base.get("bytes_total") else None
    # Decision (Rule 7): compare the two medians through their confidence intervals. The
    # candidate is a new best only if its whole interval lies below the baseline's, i.e. even
    # its slow end beats the baseline's fast end, and the gain exceeds ``min_gain``. When either
    # side has too few runs for a valid interval, nothing is accepted on timing evidence alone.
    both_valid = bool(base.get("ci_valid")) and bool(new.get("ci_valid"))
    if both_valid:
        separated_faster = new["ci_upper_s"] < base["ci_lower_s"]
        separated_slower = new["ci_lower_s"] > base["ci_upper_s"]
        speedup_lo = base["ci_lower_s"] / new["ci_upper_s"] if new["ci_upper_s"] > 0 else 0.0
        speedup_hi = base["ci_upper_s"] / new["ci_lower_s"] if new["ci_lower_s"] > 0 else 0.0
        basis = "median_ci"
    else:
        separated_faster = separated_slower = False
        speedup_lo = speedup_hi = None
        basis = "insufficient_runs"
    noise = max(base.get("rel_mad", 0.0), min_gain)
    accepted = ((not work_changed) and separated_faster
                and speedup > best_speedup and speedup > 1.0 + min_gain)
    regression = work_changed or separated_slower or (not both_valid and speedup < regression_factor)
    if work_changed:
        verdict = "rejected_work_changed"
    elif accepted:
        verdict = "faster"
    elif separated_slower:
        verdict = "slower"
    elif separated_faster:
        verdict = "faster_not_best"
    else:
        verdict = "no_significant_change" if both_valid else "insufficient_runs"
    return {
        "speedup": round(speedup, 3),
        "bw_speedup": round(bw_speedup, 3) if bw_speedup is not None else None,
        "work_ratio": round(ratio, 3) if ratio is not None else None,
        "work_source": source,
        "bytes_ratio": round(bytes_ratio, 3) if bytes_ratio is not None else None,
        "bytes_ratio_flag": bytes_ratio is not None and abs(bytes_ratio - 1.0) > work_tolerance,
        "noise_margin": round(noise, 3),
        "decision_basis": basis,
        "verdict": verdict,
        "speedup_ci": [round(speedup_lo, 3), round(speedup_hi, 3)] if speedup_lo is not None else None,
        "rejected_work_changed": work_changed,
        "accepted": accepted,
        "regression": regression,
    }
