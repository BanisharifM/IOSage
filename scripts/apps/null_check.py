"""Null check of the decision rule on one application (EXECUTION_ROADMAP S0.10).

Runs the unchanged case twice, as arms A and A' with identical arguments, round-robin
(A, A', A, A', ...), each run as one job of the application's run script. Wall times come
from the manifest rows that apps/common/run_lib.sh writes. The rule
(aggregate_repeats + evaluate_candidate) must return no_significant_change for A vs A'.

Usage (as a batch driver, see scripts/measurement_study/run_study.slurm):
    sbatch scripts/measurement_study/run_study.slurm scripts/apps/null_check.py \
        --app nek5000 --script /work/hdd/bdau/mbanisharifdehkordi/apps/nek5000/runs/run_turbChannel.slurm \
        --case N1 --repeats 11 --output results/apps/null_check/nek5000_N1.json -- 200 50 1 yes
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

from src.llm.closed_loop_metrics import aggregate_repeats, evaluate_candidate  # noqa: E402
from src.llm.iterative_executor import IterativeExecutor  # noqa: E402

logger = logging.getLogger("null_check")


def correctness_passed(row):
    """Return true only for a named correctness check with a boolean pass."""
    correctness = row.get("correctness")
    return (isinstance(correctness, dict)
            and isinstance(correctness.get("check"), str)
            and bool(correctness["check"].strip())
            and correctness.get("pass") is True)


def manifest_row(manifest, job_id):
    """The row of one job id in a manifest.jsonl, or None."""
    if not manifest.exists():
        return None
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        if row.get("jobid") == str(job_id):
            return row
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", required=True, help="Application name used by iosage_manifest")
    parser.add_argument("--script", required=True, help="The application's run script")
    parser.add_argument("--case", required=True, help="Case id of the unchanged configuration")
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--time-limit", default=None, help="sbatch --time override, e.g. 00:10:00")
    parser.add_argument("--manifest-root", default=str(PROJECT_DIR / "results" / "apps" / "null_check"),
                        help="MANIFEST_ROOT passed to the job (kept apart from the scored runs)")
    parser.add_argument("--config", default=str(PROJECT_DIR / "configs" / "iterative.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("script_args", nargs="*", help="Positional arguments of the run script (after --)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")

    script = Path(args.script).resolve()
    if not script.is_file():
        logger.error("run script not found: %s", script)
        sys.exit(2)
    manifest = Path(args.manifest_root) / args.app / "manifest.jsonl"
    executor = IterativeExecutor(yaml.safe_load(open(args.config)))
    extra = [f"--time={args.time_limit}"] if args.time_limit else []

    arms = {"A": {"rows": [], "job_ids": [], "failed": []}, "A_prime": {"rows": [], "job_ids": [], "failed": []}}
    for k in range(args.repeats):
        for arm, a in arms.items():
            export = (f"--export=CASE_ID={args.case},CASE_ROLE=null_{arm},CASE_KIND=null_check,"
                      f"REPEAT={k},MANIFEST_ROOT={args.manifest_root}")
            job_id = executor.submit_and_wait(str(script), poll_interval=10,
                                              sbatch_args=[export, *extra], script_args=args.script_args)
            if job_id is None:
                logger.warning("%s round %d: job failed", arm, k)
                a["failed"].append(k)
                continue
            time.sleep(5)   # the manifest line is written by the job just before it ends
            row = manifest_row(manifest, job_id)
            if row is None or row.get("rc") != 0 or row.get("wall_s") is None:
                logger.warning("%s round %d: no usable manifest row for job %s (%s)", arm, k, job_id, row)
                a["failed"].append(k)
                continue
            if not correctness_passed(row):
                logger.warning("%s round %d: correctness check missing or failed (job %s)", arm, k, job_id)
                a["failed"].append(k)
                continue
            a["rows"].append(row)
            a["job_ids"].append(job_id)
            logger.info("%s round %d: %.1f s (job %s)", arm, k, row["wall_s"], job_id)

    incomplete = {arm: len(a["rows"]) for arm, a in arms.items()
                  if len(a["rows"]) != args.repeats}
    if incomplete:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"app": args.app, "case": args.case, "repeats": args.repeats,
                       "status": "incomplete",
                       "arms": {arm: {"successful_runs": len(a["rows"]),
                                      "job_ids": a["job_ids"], "failed_rounds": a["failed"]}
                                for arm, a in arms.items()}}, f, indent=2)
        logger.error("required run counts not met: %s", incomplete)
        raise SystemExit(1)

    agg = {}
    for arm, a in arms.items():
        agg[arm] = aggregate_repeats(
            [{"walltime_s": r["wall_s"], "write_bw_mb_s": 0.0, "bytes_total": r["out_bytes"]} for r in a["rows"]],
            args.confidence)
    null = evaluate_candidate(agg["A"], agg["A_prime"], best_speedup=1.0)
    null_ok = null["verdict"] == "no_significant_change"
    for arm, g in agg.items():
        logger.info("%-8s median %.1f s, CI [%.1f, %.1f], relMAD %.1f%%, range %.0f%%, runs %s", arm,
                    g["walltime_s"], g["ci_lower_s"], g["ci_upper_s"], 100 * g["rel_mad"],
                    100 * g["spread_rel"], [round(w, 1) for w in g["walltime_runs_s"]])
    logger.info("NULL A vs A': speedup %.2fx, CI %s, verdict %s -> %s", null["speedup"],
                null["speedup_ci"], null["verdict"], "PASS" if null_ok else "FAIL")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"app": args.app, "script": str(script), "script_args": args.script_args, "case": args.case,
                   "repeats": args.repeats, "confidence": args.confidence, "manifest": str(manifest),
                   "arms": {arm: {"job_ids": a["job_ids"], "failed_rounds": a["failed"],
                                  "runs_s": [round(w, 2) for w in agg[arm]["walltime_runs_s"]],
                                  "median_s": round(agg[arm]["walltime_s"], 2),
                                  "ci_s": [round(agg[arm]["ci_lower_s"], 2), round(agg[arm]["ci_upper_s"], 2)],
                                  "ci_coverage": agg[arm]["ci_coverage"], "rel_mad": round(agg[arm]["rel_mad"], 3),
                                  "range_rel": round(agg[arm]["spread_rel"], 3)} for arm, a in arms.items()},
                   "null_test": {**null, "pass": null_ok}}, f, indent=2, default=str)
    logger.info("wrote %s", out)
    sys.exit(0 if null_ok else 1)


if __name__ == "__main__":
    main()
