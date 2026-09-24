"""Verify application smoke runs: scheduler state, run-script exit line, and Darshan logs.

For every job listed in a TSV file (app <TAB> jobid <TAB> sbatch arguments) this script checks:
  1. sacct reports COMPLETED with exit code 0:0;
  2. the job's stdout (path taken from the #SBATCH --output line of the run script) contains
     at least one "exit N" line (optionally "<phase> exit N"), all with N = 0, and at least
     one "darshan: <path>" line;
  3. every Darshan log named there lies under the expected log root, opens with PyDarshan,
     has no module with the partial flag set (record cap hit), and reports the bytes written
     per module.

Usage:
    python scripts/verify_app_smoke_runs.py --jobs results/apps/smoke_2026-09-21_jobs.tsv \
        --log-root data/benchmark_logs/apps --output results/apps/smoke_2026-09-21_verify.json
"""
import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("verify_app_smoke_runs")
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from src.utils.artifacts import sha256_file  # noqa: E402

BYTE_COUNTERS = {
    "POSIX": ("POSIX_BYTES_WRITTEN", "POSIX_BYTES_READ"),
    "MPI-IO": ("MPIIO_BYTES_WRITTEN", "MPIIO_BYTES_READ"),
    "STDIO": ("STDIO_BYTES_WRITTEN", "STDIO_BYTES_READ"),
}


def sacct_state(jobid):
    """Return (state, exit_code, elapsed, nodelist) of the batch allocation."""
    cmd = ["sacct", "-j", str(jobid), "-X", "-n", "-P", "-o", "State,ExitCode,Elapsed,NodeList"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    if not out:
        return ("UNKNOWN", "", "", "")
    parts = out.splitlines()[0].split("|")
    return tuple(parts + [""] * (4 - len(parts)))


def stdout_path(script, jobid):
    """Resolve the #SBATCH --output pattern of a run script for one job id."""
    for line in Path(script).read_text().splitlines():
        m = re.match(r"#SBATCH\s+--output=(\S+)", line)
        if m:
            return Path(m.group(1).replace("%j", str(jobid)))
    return None


def inspect_darshan(path, log_root):
    """Open one Darshan log and summarize modules, partial flags and byte totals."""
    import darshan  # imported here so --help works without PyDarshan

    info = {"path": str(path), "under_log_root": str(path).startswith(str(log_root))}
    report = darshan.DarshanReport(str(path), read_all=True)
    info["nprocs"] = int(report.metadata["job"]["nprocs"])
    info["run_time_s"] = round(float(report.metadata["job"].get("run_time", 0.0)), 2)
    info["modules"] = sorted(report.modules.keys())
    info["partial_modules"] = sorted(
        name for name, mod in report.modules.items() if mod.get("partial_flag")
    )
    totals = {}
    for mod, (wcol, rcol) in BYTE_COUNTERS.items():
        if mod in report.records:
            counters = report.records[mod].to_df()["counters"]
            totals[mod] = {
                "written_mb": round(float(counters[wcol].clip(lower=0).sum()) / 1e6, 1),
                "read_mb": round(float(counters[rcol].clip(lower=0).sum()) / 1e6, 1),
                "records": int(len(counters)),
            }
    info["bytes"] = totals
    return info


def verify_job(app, jobid, args_str, log_root):
    result = {"app": app, "jobid": jobid, "checks": {}, "problems": []}
    state, exit_code, elapsed, nodes = sacct_state(jobid)
    result.update({"state": state, "exit_code": exit_code, "elapsed": elapsed, "nodes": nodes})
    result["checks"]["scheduler_completed"] = state == "COMPLETED" and exit_code == "0:0"
    if not result["checks"]["scheduler_completed"]:
        result["problems"].append(f"sacct state {state} exit {exit_code}")

    script = next((tok for tok in args_str.split() if tok.endswith(".slurm")), None)
    out_path = stdout_path(script, jobid) if script else None
    result["stdout"] = str(out_path) if out_path else None
    text = out_path.read_text(errors="replace") if out_path and out_path.exists() else ""
    result["checks"]["stdout_found"] = bool(text)
    result["checks"]["env_ok_line"] = "env.sh ok:" in text
    # Run scripts print "exit N ..." once, or "<phase> exit N ..." per phase (openPMD: write, read).
    exit_codes = [int(c) for c in re.findall(r"^(?:\w+ )?exit (\d+)\b", text, flags=re.M)]
    result["exit_codes"] = exit_codes
    result["checks"]["exit_zero_line"] = bool(exit_codes) and all(c == 0 for c in exit_codes)
    if not text:
        result["problems"].append("stdout file missing or empty")
    elif not exit_codes:
        result["problems"].append("no 'exit N' line in stdout")
    elif any(exit_codes):
        result["problems"].append(f"non-zero exit line(s) in stdout: {exit_codes}")
    if text and not result["checks"]["env_ok_line"]:
        result["problems"].append("env.sh check line missing")

    log_paths = [Path(p) for p in re.findall(r"^darshan: (\S+\.darshan)\s*$", text, flags=re.M)]
    result["checks"]["darshan_log_reported"] = bool(log_paths)
    if not log_paths:
        result["problems"].append("no Darshan log reported")
    result["darshan"] = []
    for path in log_paths:
        try:
            info = inspect_darshan(path, log_root)
        except Exception as exc:  # report, do not hide
            info = {"path": str(path), "error": repr(exc)}
            result["problems"].append(f"cannot open {path.name}: {exc!r}")
        else:
            if not info["under_log_root"]:
                result["problems"].append(f"log outside {log_root}: {path}")
            if info["partial_modules"]:
                result["problems"].append(f"record cap hit in {info['partial_modules']}")
            if not any(v["written_mb"] or v["read_mb"] for v in info["bytes"].values()):
                result["problems"].append(f"no bytes recorded in {path.name}")
        result["darshan"].append(info)
    result["passed"] = not result["problems"]
    return result


def move_job_files(result, trash_root):
    """Move the resolved evidence of one passed job into a job-specific directory."""
    import shutil

    destination = trash_root / str(result["jobid"])
    if destination.exists():
        raise FileExistsError(f"archive destination already exists: {destination}")
    destination.mkdir(parents=True)
    moved = []
    out_path = Path(result["stdout"]) if result.get("stdout") else None
    candidates = []
    if out_path:
        candidates += [out_path, out_path.with_suffix(".err")]
    candidates += [Path(d["path"]) for d in result.get("darshan", []) if d.get("path")]
    for path in candidates:
        if path.is_file():
            target = destination / path.name
            record = {"original": str(path), "moved_to": str(target), "sha256": sha256_file(path)}
            shutil.move(str(path), str(target))
            moved.append(record)
            if result.get("stdout") == str(path):
                result["stdout"] = str(target)
            for darshan_info in result.get("darshan", []):
                if darshan_info.get("path") == str(path):
                    darshan_info["path"] = str(target)
    if out_path:
        log_dir = out_path.parent / f"logs_{result['jobid']}"
        if log_dir.is_dir():
            target = destination / log_dir.name
            hashes = {str(path.relative_to(log_dir)): sha256_file(path)
                      for path in log_dir.rglob("*") if path.is_file()}
            shutil.move(str(log_dir), str(target))
            moved.append({"original": str(log_dir), "moved_to": str(target), "files": hashes})
    return moved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", required=True, help="TSV: app, jobid, sbatch arguments")
    parser.add_argument("--log-root", required=True, help="Directory that must contain the Darshan logs")
    parser.add_argument("--output", required=True, help="Path of the JSON report")
    parser.add_argument("--archive-passed", action="store_true",
                        help="Move evidence from passed jobs into a job-specific local archive")
    parser.add_argument("--trash-root", default=str(PROJECT_DIR / ".codex-trash" / "app_smoke"),
                        help="Root directory used by --archive-passed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log_root = Path(args.log_root).resolve()
    results = []
    for line in Path(args.jobs).read_text().splitlines():
        if not line.strip():
            continue
        app, jobid, args_str = (line.split("\t") + ["", ""])[:3]
        res = verify_job(app, jobid, args_str, log_root)
        results.append(res)
        logger.info("%-9s %s %s %s %s", app, jobid, res["state"], "PASS" if res["passed"] else "FAIL",
                    "; ".join(res["problems"]))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"log_root": str(log_root), "results": results}, indent=2))
    if args.archive_passed:
        trash_root = Path(args.trash_root).resolve()
        for res in results:
            if res["passed"]:
                res["archived"] = move_job_files(res, trash_root)
                logger.info("%-9s %s archived %d item(s)", res["app"], res["jobid"],
                            len(res["archived"]))
        Path(args.output).write_text(json.dumps({"log_root": str(log_root), "results": results}, indent=2))
    n_pass = sum(r["passed"] for r in results)
    logger.info("%d of %d jobs passed; report written to %s", n_pass, len(results), args.output)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
