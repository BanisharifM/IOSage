"""
Recompute iterative closed-loop speedups under a wall-time objective.

Reads the canonical trackc_*.json histories (unchanged), re-locates every
baseline / iteration Darshan log through the retained SLURM .out files, and
recomputes per-run speedups under three definitions side by side:

  bw        : Darshan write_bw_mb_s ratio (what the paper's Table VII used)
  walltime  : Darshan job runtime ratio (max end - min start across all logs
              belonging to the SLURM job, so multi-phase jobs are spanned)
  tool      : the benchmark's own reported figure from stdout, when present

It also records byte, operation, and open counts per run so that the
"same work" invariant can be checked, and flags any iteration where the
bandwidth and wall-time metrics disagree on the keep/rollback decision.

Nothing under results/boost_experiment/full_evaluation/iterative/ is modified.

Usage:
    python scripts/recompute_iterative_walltime.py
    python scripts/recompute_iterative_walltime.py --output-dir <dir>
"""

import argparse
import csv
import glob
import json
import logging
import math
import os
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOCAL_PKGS = PROJECT_DIR / ".local_pkgs"
if LOCAL_PKGS.exists():
    sys.path.insert(0, str(LOCAL_PKGS))
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.data.parse_darshan import parse_darshan_log  # noqa: E402
from src.data.feature_extraction import extract_raw_features  # noqa: E402
from src.data.preprocessing import stage3_engineer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("darshan").setLevel(logging.ERROR)

TRACKC_DIR = PROJECT_DIR / "results" / "boost_experiment" / "full_evaluation" / "iterative"
OUT_DIR_DEFAULT = TRACKC_DIR.parent / "iterative_walltime"
SLURM_DIR = PROJECT_DIR / "results" / "iterative"
DARSHAN_DIR = PROJECT_DIR / "data" / "benchmark_logs" / "iterative"

# JSON model field -> token used in SLURM job names
MODEL_SHORT = {
    "claude-sonnet": "claude",
    "gpt-4o": "gpt-4o",
    "llama-70b": "llama",
    "gpt-4.1-mini": "gpt-4.1-mini",
}

# Tolerance for matching a recorded bandwidth to a recomputed one
BW_REL_TOL = 0.02
# Work-invariance tolerance (bytes / ops / opens ratio outside this is flagged)
WORK_TOL = 0.25
# Executor's regression rule (iterative_optimizer.run_optimization)
REGRESSION_FACTOR = 0.9
# Darshan default MAX_RECORDS per module per rank; the IOR/mdtest SLURM path
# never raises it, so jobs touching more files than this are truncated.
DARSHAN_DEFAULT_MAX_RECORDS = 1024


def configured_work(workload, config, nprocs):
    """Work implied by the benchmark configuration, independent of Darshan.

    Used where Darshan counters are unreliable (record cap). Returns
    (bytes, files) or (None, None) when the suite has no closed form here.
    """
    if config is None:
        return None, None
    try:
        if workload.startswith("mdtest"):
            items = int(config.get("items_per_rank", 0))
            wb = int(config.get("write_bytes", 0))
            return items * wb * nprocs, items * nprocs
        if workload.startswith("dlio"):
            # training-phase bytes read = files x samples x record x epochs
            nf = int(config.get("num_files_train", 0))
            ns = int(config.get("num_samples_per_file", 1))
            rl = int(config.get("record_length", 0))
            ep = int(config.get("epochs", 1))
            return nf * ns * rl * ep, nf
    except (TypeError, ValueError):
        return None, None
    return None, None


def parse_out_file(path):
    """Extract tool-reported figures and script timestamps from a SLURM .out."""
    txt = Path(path).read_text(errors="replace")
    info = {"out_path": str(path)}
    m = re.search(r"Max Write:\s+([\d.]+) MiB/sec", txt)
    if m:
        info["tool_ior_write_mibs"] = float(m.group(1))
    m = re.search(r"File creation\s*:\s*([\d.]+)", txt)
    if m:
        info["tool_mdtest_create_ops_s"] = float(m.group(1))
    m = re.search(r"WRITE Checkpoint Perf:\s*([\d.]+)\s*BW\[MB/s\].*?([\d.]+)\s*MaxTime\[sec\]", txt)
    if m:
        info["tool_hacc_write_mbs"] = float(m.group(1))
        info["tool_hacc_write_maxtime_s"] = float(m.group(2))
    m = re.search(r"Observed completion time:\s*([\d.]+)\s*s", txt)
    if m:
        info["tool_h5bench_write_wall_s"] = float(m.group(1))
    m = re.search(r"Observed read completion time:\s*([\d.]+)\s*s", txt)
    if m:
        info["tool_h5bench_read_wall_s"] = float(m.group(1))
    m = re.search(r"SYNC Observed write rate:\s*([\d.]+)\s*(MB|KB|GB)/s", txt)
    if m:
        scale = {"KB": 1e-3, "MB": 1.0, "GB": 1e3}[m.group(2)]
        info["tool_h5bench_write_mbs"] = float(m.group(1)) * scale
    return info


def darshan_summary(path):
    """Job metadata and key counters for one Darshan log."""
    parsed = parse_darshan_log(path)
    if parsed is None:
        return None
    job, c = parsed["job"], parsed["counters"]
    return {
        "path": str(path),
        "start": float(job.get("start_time", 0)),
        "end": float(job.get("end_time", 0)),
        "runtime": float(job.get("runtime", 0.0)),
        "nprocs": int(job.get("nprocs", 1)),
        "bytes_written": float(c.get("POSIX_BYTES_WRITTEN", 0)),
        "bytes_read": float(c.get("POSIX_BYTES_READ", 0)),
        "writes": float(c.get("POSIX_WRITES", 0)),
        "reads": float(c.get("POSIX_READS", 0)),
        "opens": float(c.get("POSIX_OPENS", 0)),
        "write_time": float(c.get("POSIX_F_WRITE_TIME", 0)),
        "meta_time": float(c.get("POSIX_F_META_TIME", 0)),
    }


def executor_bw(path):
    """Replicate IterativeExecutor.extract_features + the optimizer's bw pick."""
    parsed = parse_darshan_log(path)
    if parsed is None:
        return None
    df = stage3_engineer(pd.DataFrame([extract_raw_features(parsed)]))
    f = df.iloc[0].to_dict()
    return f.get("write_bw_mb_s", 0) or f.get("total_bw_mb_s", 0.001)


def job_logs(job_id):
    """All Darshan logs for a SLURM job, newest-mtime first (executor order)."""
    return sorted(glob.glob(str(DARSHAN_DIR / f"*id{job_id}*")),
                  key=os.path.getmtime, reverse=True)


def candidate_jobs(workload, model_short, stage):
    """SLURM job ids whose .out matches this (workload, model, stage)."""
    pat = str(SLURM_DIR / f"iter_{workload}_r0_{model_short}_{stage}_*.out")
    jobs = []
    for out in glob.glob(pat):
        m = re.search(r"_(\d+)\.out$", out)
        if m:
            jobs.append((m.group(1), out))
    return jobs


def locate_stage(workload, model_short, stage, recorded_bw):
    """Find the SLURM job whose executor-selected log reproduces recorded_bw."""
    matches = []
    for job_id, out in candidate_jobs(workload, model_short, stage):
        logs = job_logs(job_id)
        if not logs:
            continue
        bw = executor_bw(logs[0])
        if bw is None:
            continue
        rel = abs(bw - recorded_bw) / max(abs(recorded_bw), 1e-9)
        matches.append((rel, job_id, out, logs, bw))
    if not matches:
        raise ValueError(f"{workload}/{model_short}/{stage}: no candidate jobs")
    accepted = [match for match in matches if match[0] <= BW_REL_TOL]
    if not accepted:
        best = min(matches, key=lambda match: match[0])
        raise ValueError(
            f"{workload}/{model_short}/{stage}: no job within "
            f"{100 * BW_REL_TOL:.1f}%; nearest is {100 * best[0]:.1f}% "
            f"(job {best[1]})")
    if len(accepted) != 1:
        raise ValueError(
            f"{workload}/{model_short}/{stage}: {len(accepted)} jobs fall within "
            f"{100 * BW_REL_TOL:.1f}%; identity is ambiguous")
    rel, job_id, out, logs, bw = accepted[0]
    return {"job_id": job_id, "out": out, "logs": logs,
            "recomputed_bw": bw, "bw_match_rel_err": rel}


def phase_walltime(summaries):
    """Sum, over sequential phases, of the longest float runtime per phase.

    Logs whose start falls inside an earlier log's window are concurrent
    ranks of the same phase (DLIO per-rank NONMPI logs); logs that start
    after the previous phase ended are a new sequential phase (h5bench
    write then read, DLIO datagen then training).
    """
    phases = []
    for s in sorted(summaries, key=lambda x: (x["start"], -x["runtime"])):
        if phases and s["start"] < phases[-1]["end"]:
            phases[-1]["runtime"] = max(phases[-1]["runtime"], s["runtime"])
            phases[-1]["end"] = max(phases[-1]["end"], s["start"] + s["runtime"])
        else:
            phases.append({"start": s["start"], "end": s["start"] + s["runtime"],
                           "runtime": s["runtime"]})
    return sum(p["runtime"] for p in phases), len(phases)


def stage_measurements(loc):
    """Wall time and work counters across every log of the job."""
    summaries = [s for s in (darshan_summary(p) for p in loc["logs"]) if s]
    if not summaries:
        return None
    starts = [s["start"] for s in summaries if s["start"] > 0]
    ends = [s["end"] for s in summaries if s["end"] > 0]
    span = (max(ends) - min(starts)) if starts and ends else None
    walltime, n_phases = phase_walltime(summaries)
    primary = summaries[0]  # executor-selected log
    agg = {k: sum(s[k] for s in summaries)
           for k in ("bytes_written", "bytes_read", "writes", "reads", "opens",
                     "write_time", "meta_time")}
    out = parse_out_file(loc["out"])
    cap_hit = (not Path(loc["out"]).name.startswith("iter_dlio")) and any(
        s["opens"] >= 2 * DARSHAN_DEFAULT_MAX_RECORDS * s["nprocs"] for s in summaries)
    return {
        "job_id": loc["job_id"],
        "n_logs": len(summaries),
        "darshan_record_cap_hit": cap_hit,
        "walltime_s": walltime,
        "n_phases": n_phases,
        "walltime_span_int_s": span,
        "primary_runtime_s": primary["runtime"],
        "primary_log": Path(primary["path"]).name,
        "nprocs": primary["nprocs"],
        "recomputed_bw": loc["recomputed_bw"],
        "bw_match_rel_err": loc["bw_match_rel_err"],
        **agg,
        **{k: v for k, v in out.items() if k != "out_path"},
        "out_file": Path(loc["out"]).name,
    }


def tool_speedup(base, it):
    """Benchmark-reported speedup where the tool prints a usable figure."""
    for key, higher_is_better in (
        ("tool_ior_write_mibs", True),
        ("tool_mdtest_create_ops_s", True),
        ("tool_hacc_write_mbs", True),
        ("tool_h5bench_write_wall_s", False),
    ):
        b, n = base.get(key), it.get(key)
        if b and n:
            return (n / b) if higher_is_better else (b / n), key
    return None, None


def ratio(a, b):
    return (a / b) if (a and b) else None


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def recompute_run(hist, workloads_cfg):
    workload, model = hist["workload"], hist["model"]
    hist["config_baseline"] = (workloads_cfg.get(workload) or {}).get("bad_config")
    ms = MODEL_SHORT[model]
    rec = {
        "workload": workload, "model": model,
        "paper_status": hist.get("final_status"),
        "paper_best_iteration": hist.get("best_iteration"),
        "paper_best_speedup_bw": hist.get("best_speedup"),
        "iterations": [],
    }
    if hist.get("final_status") == "already_healthy" or "baseline_bw" not in hist:
        rec["note"] = "excluded by classifier (already_healthy); no iterations"
        return rec

    base_loc = locate_stage(workload, ms, "baseline", hist["baseline_bw"])
    base = stage_measurements(base_loc)
    rec["baseline"] = base

    best_wall, best_wall_iter = 1.0, -1
    disagreements = []
    for it in hist.get("iterations", []):
        k = it["iteration"]
        row = {"iteration": k, "executed": it.get("executed", False),
               "paper_speedup_bw": it.get("speedup"),
               "validated_config": it.get("validated_config")}
        if it.get("executed") and it.get("new_bw") is not None:
            loc = locate_stage(workload, ms, f"i{k}", it["new_bw"])
            if loc:
                m = stage_measurements(loc)
                if m is None:
                    raise ValueError(
                        f"{workload}/{model}/i{k}: matched job has no parsed logs")
                row.update(m)
                row["speedup_walltime"] = ratio(base["walltime_s"], m["walltime_s"])
                row["speedup_walltime_span_int"] = ratio(base["walltime_span_int_s"], m["walltime_span_int_s"])
                row["speedup_primary_runtime"] = ratio(base["primary_runtime_s"], m["primary_runtime_s"])
                row["speedup_tool"], row["tool_metric"] = tool_speedup(base, m)
                row["bytes_written_ratio"] = ratio(m["bytes_written"], base["bytes_written"])
                row["bytes_total_ratio"] = ratio(m["bytes_written"] + m["bytes_read"],
                                                 base["bytes_written"] + base["bytes_read"])
                row["ops_ratio"] = ratio(m["writes"] + m["reads"], base["writes"] + base["reads"])
                row["opens_ratio"] = ratio(m["opens"], base["opens"])
                r = row["bytes_total_ratio"]
                row["work_source"] = "darshan"
                cb, cf = configured_work(workload, it.get("validated_config"), m["nprocs"])
                bb, bf = configured_work(workload, hist["config_baseline"], base["nprocs"]) \
                    if "config_baseline" in hist else (None, None)
                if cb is not None and bb:
                    r = cb / bb
                    row["bytes_configured_ratio"] = r
                    row["files_configured_ratio"] = (cf / bf) if bf else None
                    row["work_source"] = "configured"
                row["work_changed"] = r is not None and (r < 1 - WORK_TOL or r > 1 + WORK_TOL)
                # Decision under each metric, using the executor's own rule
                bw_s = it.get("speedup") or 0
                w_s = row["speedup_walltime"] or 0
                bw_regress = bw_s < REGRESSION_FACTOR
                w_regress = w_s < REGRESSION_FACTOR
                if bw_regress != w_regress:
                    disagreements.append({"iteration": k, "bw": bw_s, "walltime": w_s})
                if w_s > best_wall:
                    best_wall, best_wall_iter = w_s, k
        rec["iterations"].append(row)

    rec["best_speedup_walltime"] = best_wall
    rec["best_iteration_walltime"] = best_wall_iter
    rec["best_iteration_agrees"] = (best_wall_iter == hist.get("best_iteration"))
    rec["decision_disagreements"] = disagreements
    # Work invariance at the paper's chosen best iteration
    pb = next((r for r in rec["iterations"] if r["iteration"] == hist.get("best_iteration")), None)
    if hist.get("best_iteration", -1) == -1:
        rec["note"] = "loop kept the baseline config (no iteration improved bw)"
        rec["paper_best_walltime_speedup"] = 1.0
        rec["paper_best_tool_speedup"] = 1.0
        rec["paper_best_bytes_ratio"] = 1.0
        rec["paper_best_ops_ratio"] = 1.0
        rec["paper_best_opens_ratio"] = 1.0
        rec["paper_best_work_changed"] = False
        rec["paper_best_primary_log"] = base["primary_log"]
    elif pb:
        rec["paper_best_walltime_speedup"] = pb.get("speedup_walltime")
        rec["paper_best_tool_speedup"] = pb.get("speedup_tool")
        rec["paper_best_bytes_ratio"] = pb.get("bytes_written_ratio")
        rec["paper_best_ops_ratio"] = pb.get("ops_ratio")
        rec["paper_best_opens_ratio"] = pb.get("opens_ratio")
        rec["paper_best_work_changed"] = pb.get("work_changed")
        rec["paper_best_work_source"] = pb.get("work_source")
        rec["paper_best_bytes_configured_ratio"] = pb.get("bytes_configured_ratio")
        rec["paper_best_files_configured_ratio"] = pb.get("files_configured_ratio")
        rec["paper_best_record_cap_hit"] = pb.get("darshan_record_cap_hit") or base.get("darshan_record_cap_hit")
        rec["paper_best_primary_log"] = pb.get("primary_log")
        rec["paper_best_walltime_span_int"] = pb.get("speedup_walltime_span_int")
        rec["paper_best_n_phases"] = pb.get("n_phases")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--output-dir", default=str(OUT_DIR_DEFAULT))
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")

    with open(PROJECT_DIR / "configs" / "iterative.yaml") as fh:
        workloads_cfg = yaml.safe_load(fh)["workloads"]
    files = sorted(TRACKC_DIR.glob("trackc_*.json"))
    if not files:
        raise FileNotFoundError(f"no trackc result files in {TRACKC_DIR}")
    logger.info("Recomputing %d runs from %s", len(files), TRACKC_DIR)
    records = []
    for f in files:
        hist = json.load(open(f))
        if isinstance(hist, list):
            hist = hist[0]
        logger.info("%s", f.name)
        rec = recompute_run(hist, workloads_cfg)
        rec["source_file"] = f.name
        records.append(rec)

    out_dir.mkdir(parents=True)
    for rec in records:
        output_name = rec["source_file"].replace("trackc_", "walltime_")
        with open(out_dir / output_name, "w") as fh:
            json.dump(rec, fh, indent=2, default=str)

    # Summary table
    rows = []
    for r in records:
        rows.append({
            "workload": r["workload"], "model": r["model"], "status": r["paper_status"],
            "paper_bw_speedup": r.get("paper_best_speedup_bw"),
            "walltime_at_paper_best": r.get("paper_best_walltime_speedup"),
            "tool_at_paper_best": r.get("paper_best_tool_speedup"),
            "best_walltime_speedup": r.get("best_speedup_walltime"),
            "best_iter_bw": r.get("paper_best_iteration"),
            "best_iter_walltime": r.get("best_iteration_walltime"),
            "best_iter_agrees": r.get("best_iteration_agrees"),
            "n_decision_disagreements": len(r.get("decision_disagreements", [])),
            "bytes_ratio": r.get("paper_best_bytes_ratio"),
            "ops_ratio": r.get("paper_best_ops_ratio"),
            "opens_ratio": r.get("paper_best_opens_ratio"),
            "work_changed": r.get("paper_best_work_changed"),
            "work_source": r.get("paper_best_work_source"),
            "bytes_configured_ratio": r.get("paper_best_bytes_configured_ratio"),
            "files_configured_ratio": r.get("paper_best_files_configured_ratio"),
            "record_cap_hit": r.get("paper_best_record_cap_hit"),
            "guarded_walltime": (1.0 if r.get("paper_best_work_changed") else r.get("paper_best_walltime_speedup")),
            "n_phases": r.get("paper_best_n_phases"),
            "primary_log": r.get("paper_best_primary_log"),
            "note": r.get("note", ""),
        })
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Per-model geomeans, same active-set rule as the paper (exclude already_healthy)
    per_model = {}
    for r in rows:
        if r["status"] == "already_healthy" or r["paper_bw_speedup"] is None:
            continue
        d = per_model.setdefault(r["model"], {"bw": [], "wall": [], "guarded": [], "bw_gt1": [], "wall_gt1": [], "n": 0, "n_rejected": 0})
        d["n"] += 1
        d["bw"].append(r["paper_bw_speedup"])
        d["wall"].append(r["walltime_at_paper_best"])
        d["guarded"].append(r["guarded_walltime"])
        if r["work_changed"]:
            d["n_rejected"] += 1
        if r["paper_bw_speedup"] and r["paper_bw_speedup"] > 1.0:
            d["bw_gt1"].append(r["paper_bw_speedup"])
            d["wall_gt1"].append(r["walltime_at_paper_best"])
    geo = {m: {"n_active": d["n"],
               "geomean_bw_all_active": geomean(d["bw"]),
               "geomean_walltime_all_active": geomean(d["wall"]),
               "geomean_walltime_guarded": geomean(d["guarded"]),
               "n_rejected_by_work_guard": d["n_rejected"],
               "n_bw_gt1": len(d["bw_gt1"]),
               "geomean_bw_gt1_only": geomean(d["bw_gt1"]),
               "geomean_walltime_same_subset": geomean(d["wall_gt1"])}
           for m, d in per_model.items()}
    with open(out_dir / "geomeans.json", "w") as fh:
        json.dump(geo, fh, indent=2)

    logger.info("Wrote %d run files, summary.csv, geomeans.json -> %s", len(records), out_dir)
    for m, g in geo.items():
        logger.info("%-14s n=%2d  bw=%.2fx  walltime=%.2fx  guarded=%.2fx (rejected %d) | paper-subset n=%d bw=%.2fx",
                    m, g["n_active"], g["geomean_bw_all_active"] or 0, g["geomean_walltime_all_active"] or 0,
                    g["geomean_walltime_guarded"] or 0, g["n_rejected_by_work_guard"],
                    g["n_bw_gt1"], g["geomean_bw_gt1_only"] or 0)


if __name__ == "__main__":
    raise SystemExit(main())
