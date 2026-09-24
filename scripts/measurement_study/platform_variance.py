"""Run-to-run variance of one benchmark configuration on several storage targets.

The closed loop decides on wall time, so its verdicts are only as good as the repeatability
of a single configuration. This script runs the unchanged baseline of one workload N times
on each storage target, strictly sequentially (the study must not interfere with itself),
and reports the median with two dispersion measures: the range, which one outlier
dominates, and the relative median absolute deviation, which it does not.

Usage:
    python scripts/measurement_study/platform_variance.py --workload ior_fsync_heavy \
        --repeats 7 --targets work_hdd work_nvme --output results/measurement_study/x.json
"""
import argparse
import copy
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.llm.benchmark_command_builder import BenchmarkCommandBuilder  # noqa: E402
from src.llm.closed_loop_metrics import _rel_mad, job_measurement  # noqa: E402
from src.llm.iterative_executor import IterativeExecutor  # noqa: E402

logger = logging.getLogger("platform_variance")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", required=True, help="IOR workload name from the config")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--targets", nargs="+", default=["work_hdd", "work_nvme"],
                        help="Keys of measurement_study.targets in the config")
    parser.add_argument("--config", default=str(PROJECT_DIR / "configs" / "iterative.yaml"))
    parser.add_argument("--output", required=True, help="JSON result path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")

    cfg = yaml.safe_load(open(args.config))
    targets = cfg.get("measurement_study", {}).get("targets", {})
    workload = cfg["workloads"][args.workload]
    if workload.get("benchmark", "ior") != "ior":
        raise SystemExit("this study drives IOR workloads only")

    results = {"workload": args.workload, "repeats": args.repeats,
               "started": time.strftime("%Y-%m-%d %H:%M:%S"), "targets": {}}
    incomplete = False
    for name in args.targets:
        base_dir = targets[name]
        tcfg = copy.deepcopy(cfg)
        tcfg["slurm"]["scratch_dir"] = base_dir          # executor and builder agree on the path
        executor = IterativeExecutor(tcfg)
        builder = BenchmarkCommandBuilder(config_path=args.config, scratch_dir=base_dir)
        _, params, _ = builder.validate_ior_params(workload["bad_config"])
        job = f"mstudy_{name}_{args.workload}"
        cmd = builder.build_ior_command(params, output_dir=f"{base_dir}/{job}")
        script = executor.generate_slurm_script(job, cmd, "ior")
        walls, job_ids, failed = [], [], []
        for i in range(args.repeats):
            # Submit through the executor: it strips the submitter's SLURM_* step options, which
            # otherwise carry over into the job and stall srun when this driver is itself a batch job.
            job_id = executor.submit_and_wait(script, poll_interval=10)
            if job_id is None:
                logger.warning("%s run %d: job failed", name, i)
                failed.append({"run": i, "reason": "job_failed"})
                continue
            time.sleep(5)  # let the Darshan log land
            logs = executor.find_darshan_logs(job_id)
            if not logs:
                logger.warning("%s run %d: no Darshan log (job %s)", name, i, job_id)
                failed.append({"run": i, "job_id": job_id, "reason": "darshan_log_missing"})
                continue
            wall = job_measurement(logs, "ior", args.workload, params)["walltime_s"]
            walls.append(wall)
            job_ids.append(job_id)
            logger.info("%s run %d: %.1f s (job %s)", name, i, wall, job_id)
        if len(walls) != args.repeats:
            incomplete = True
            results["targets"][name] = {
                "dir": base_dir, "status": "incomplete", "job_ids": job_ids,
                "runs_s": [round(w, 2) for w in walls], "failed_attempts": failed,
            }
            logger.error("%s has %d/%d required runs", name, len(walls), args.repeats)
        else:
            med = statistics.median(walls)
            results["targets"][name] = {
                "dir": base_dir, "job_ids": job_ids, "runs_s": [round(w, 2) for w in walls],
                "median_s": round(med, 2),
                "range_rel": round((max(walls) - min(walls)) / med, 3),
                "rel_mad": round(_rel_mad(walls), 3),
            }
            logger.info("== %s: median %.1f s, range %.0f%%, relMAD %.1f%%", name, med,
                        (max(walls) - min(walls)) / med * 100, _rel_mad(walls) * 100)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:                # save after every target
            json.dump(results, f, indent=2)
    logger.info("wrote %s", args.output)
    if incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
