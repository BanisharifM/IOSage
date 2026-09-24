"""Run one scored series of an application from its preregistration (TASK_01, phase 4).

For every repeat k the registered cases run in ``case_order`` (problem first, then fix),
each as one job of the application's tracked run script, submitted through
``IterativeExecutor.submit_and_wait`` (a batch driver must not call sbatch with an inherited
SLURM environment). After the traced repeats one uninstrumented control per case runs.
Every job appends its typed row to ``<attempt>/manifest.jsonl``; the driver records every
submission and outcome in ``<attempt>/series.json``. A failed job stops the series: the
attempt keeps its rows, and a repaired experiment starts as a new named attempt.

Usage (as a batch driver):
    sbatch scripts/measurement_study/run_study.slurm scripts/apps/run_series.py \
        --cases configs/app_cases.yaml --app nek5000 --attempt attempt_2026-09-24_a
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.apps import app_cases, manifest  # noqa: E402
from src.llm.iterative_executor import IterativeExecutor  # noqa: E402

logger = logging.getLogger("run_series")


def plan(doc, app):
    """Ordered list of (case, role, kind, repeat, control) for one series."""
    spec = app_cases.application(doc, app)
    repeats = doc["protocol"]["repeats"]
    order = spec["case_order"]
    jobs = [(case, spec["cases"][case]["role"], spec["cases"][case]["kind"], k, False)
            for k in range(repeats) for case in order]
    jobs += [(case, spec["cases"][case]["role"], spec["cases"][case]["kind"], 0, True) for case in order]
    return jobs


def row_for(manifest_path, job_id):
    if not Path(manifest_path).exists():
        return None
    for row in manifest.read_rows(manifest_path):
        if row.get("jobid") == str(job_id):
            return row
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default=str(PROJECT_DIR / "configs" / "app_cases.yaml"))
    parser.add_argument("--app", required=True)
    parser.add_argument("--attempt", required=True, help="new attempt name; its directory must not exist")
    parser.add_argument("--results-root", default=str(PROJECT_DIR / "results" / "apps"))
    parser.add_argument("--config", default=str(PROJECT_DIR / "configs" / "iterative.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="write the plan and submit nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    doc = app_cases.load(args.cases)
    spec = app_cases.application(doc, args.app)
    script = PROJECT_DIR / spec["script"]
    if not script.is_file():
        logger.error("run script not found: %s", script)
        sys.exit(2)
    attempt_dir = Path(args.results_root) / args.app / args.attempt
    if attempt_dir.exists():
        logger.error("attempt directory exists, choose a new attempt name: %s", attempt_dir)
        sys.exit(2)
    jobs_dir = attempt_dir / "jobs"
    logs_dir = attempt_dir / "logs"
    jobs_dir.mkdir(parents=True)
    logs_dir.mkdir()
    manifest_path = attempt_dir / "manifest.jsonl"
    series_path = attempt_dir / "series.json"
    (attempt_dir / "equivalence.json").write_text(json.dumps({"pairs": app_cases.pairs(doc, args.app)}, indent=2))
    (attempt_dir / "cases_registered.yaml").write_text(Path(args.cases).read_text())
    schedule = plan(doc, args.app)
    state = {"app": args.app, "attempt": args.attempt, "cases_file": str(Path(args.cases).resolve()),
             "script": str(script), "manifest": str(manifest_path),
             "protocol": doc["protocol"], "plan": [dict(case=c, role=r, kind=k, repeat=n, control=ctl)
                                                    for c, r, k, n, ctl in schedule],
             "jobs": [], "status": "planned"}
    series_path.write_text(json.dumps(state, indent=2))
    if args.dry_run:
        logger.info("dry run: %d jobs planned, nothing submitted (%s)", len(schedule), series_path)
        return
    executor = IterativeExecutor(yaml.safe_load(open(args.config)))
    time_limit = spec["resources"].get("time_limit")
    state["status"] = "running"
    for case, role, kind, repeat, control in schedule:
        export = (f"--export=CASE_ID={case},CASE_ROLE={role},CASE_KIND={kind},REPEAT={repeat},"
                  f"ATTEMPT={args.attempt},MANIFEST={manifest_path},EVIDENCE_DIR={logs_dir}"
                  + (",NODARSHAN=1" if control else ""))
        sbatch_args = [export, f"--output={jobs_dir}/%x_%j.out", f"--error={jobs_dir}/%x_%j.err"]
        if time_limit:
            sbatch_args.append(f"--time={time_limit}")
        if spec["resources"].get("partition"):
            sbatch_args.append(f"--partition={spec['resources']['partition']}")
        record = {"case": case, "role": role, "kind": kind, "repeat": repeat, "control": control,
                  "submitted_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job_id": None, "outcome": None}
        state["jobs"].append(record)
        series_path.write_text(json.dumps(state, indent=2))
        job_id = executor.submit_and_wait(str(script), poll_interval=10, sbatch_args=sbatch_args,
                                          script_args=app_cases.case_args(doc, args.app, case))
        record["job_id"] = job_id
        if job_id is None:
            record["outcome"] = "job_failed_or_not_completed"
        else:
            time.sleep(5)
            row = row_for(manifest_path, job_id)
            if row is None:
                record["outcome"] = "no_manifest_row"
            elif row["rc"] != 0:
                record["outcome"] = f"exit_code_{row['rc']}"
            elif not row["correctness"]["pass"] or not row["io_validation"]["pass"]:
                record["outcome"] = "checks_failed"
            else:
                record["outcome"] = "ok"
                record["wall_s"] = row["wall_s"]
        series_path.write_text(json.dumps(state, indent=2))
        logger.info("%s %s repeat %d%s: job %s %s", case, role, repeat, " control" if control else "",
                    job_id, record["outcome"])
        if record["outcome"] != "ok":
            state["status"] = "stopped_on_failure"
            series_path.write_text(json.dumps(state, indent=2))
            logger.error("series stopped after a failed job; start a new named attempt after the repair")
            sys.exit(1)
    state["status"] = "complete"
    series_path.write_text(json.dumps(state, indent=2))
    logger.info("series complete: %d jobs, manifest %s", len(schedule), manifest_path)


if __name__ == "__main__":
    main()
