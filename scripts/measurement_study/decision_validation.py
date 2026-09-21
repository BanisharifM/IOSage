"""Validate the closed loop's accept/reject rule on real runs.

Three arms are run round-robin (A, A', B, A, A', B, ...), so all share one time window:
  A   the workload's baseline configuration
  A'  the same configuration again (null change: the rule must NOT call it faster or slower)
  B   the workload's known good configuration (real change: the rule must call it faster)
Both guarantees are reported; the script exits non-zero if either fails.

Usage:
    python scripts/measurement_study/decision_validation.py --workload ior_fsync_heavy \
        --target work_nvme --repeats 8 --output results/measurement_study/x.json
"""
import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.llm.benchmark_command_builder import BenchmarkCommandBuilder  # noqa: E402
from src.llm.closed_loop_metrics import aggregate_repeats, evaluate_candidate, job_measurement  # noqa: E402
from src.llm.iterative_executor import IterativeExecutor  # noqa: E402

logger = logging.getLogger("decision_validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--target", default="work_nvme", help="Key of measurement_study.targets")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--fix", nargs="*", default=None, metavar="KEY=VALUE",
                        help="Build arm B as the baseline plus these overrides instead of the workload's "
                             "known_good_config (use when that config does not keep the work identical)")
    parser.add_argument("--config", default=str(PROJECT_DIR / "configs" / "iterative.yaml"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(open(args.config))
    base_dir = cfg["measurement_study"]["targets"][args.target]
    workload = cfg["workloads"][args.workload]
    tcfg = copy.deepcopy(cfg)
    tcfg["slurm"]["scratch_dir"] = base_dir
    executor = IterativeExecutor(tcfg)
    builder = BenchmarkCommandBuilder(config_path=args.config, scratch_dir=base_dir)

    if args.fix:
        fixed = dict(workload["bad_config"])
        fixed.update(dict(kv.split("=", 1) for kv in args.fix))
    else:
        fixed = workload["known_good_config"]
    arms = {}
    for arm, params in (("A", workload["bad_config"]), ("A_prime", workload["bad_config"]), ("B", fixed)):
        _, sanitized, _ = builder.validate_ior_params(params)
        job = f"dval_{arm}_{args.workload}"
        cmd = builder.build_ior_command(sanitized, output_dir=f"{base_dir}/{job}")
        arms[arm] = {"params": sanitized, "script": executor.generate_slurm_script(job, cmd, "ior"),
                     "measurements": [], "job_ids": []}

    for k in range(args.repeats):
        for arm, a in arms.items():
            # Through the executor, which strips the submitter's SLURM_* step options (see
            # IterativeExecutor.submit_and_wait); a raw sbatch from a batch driver stalls srun.
            job_id = executor.submit_and_wait(a["script"], poll_interval=10)
            if job_id is None:
                logger.warning("%s round %d: job failed", arm, k)
                continue
            time.sleep(5)
            logs = executor.find_darshan_logs(job_id)
            if not logs:
                logger.warning("%s round %d: no Darshan log (job %s)", arm, k, job_id)
                continue
            m = job_measurement(logs, "ior", args.workload, a["params"])
            a["measurements"].append(m)
            a["job_ids"].append(job_id)
            logger.info("%s round %d: %.1f s (job %s)", arm, k, m["walltime_s"], job_id)

    agg = {arm: aggregate_repeats(a["measurements"], args.confidence) for arm, a in arms.items()}
    null = evaluate_candidate(agg["A"], agg["A_prime"], best_speedup=1.0)
    real = evaluate_candidate(agg["A"], agg["B"], best_speedup=1.0)
    null_ok = null["verdict"] == "no_significant_change"
    real_ok = real["verdict"] == "faster"
    for arm, g in agg.items():
        logger.info("%-8s median %.1f s, CI [%.1f, %.1f], relMAD %.1f%%, range %.0f%%, runs %s", arm,
                    g["walltime_s"], g["ci_lower_s"], g["ci_upper_s"], 100 * g["rel_mad"],
                    100 * g["spread_rel"], [round(w, 1) for w in g["walltime_runs_s"]])
    logger.info("NULL  A vs A': speedup %.2fx, CI %s, verdict %s -> %s", null["speedup"],
                null["speedup_ci"], null["verdict"], "PASS" if null_ok else "FAIL")
    logger.info("REAL  A vs B : speedup %.2fx, CI %s, verdict %s -> %s", real["speedup"],
                real["speedup_ci"], real["verdict"], "PASS" if real_ok else "FAIL")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"workload": args.workload, "target": args.target, "dir": base_dir,
                   "repeats": args.repeats, "confidence": args.confidence,
                   "arms": {arm: {"job_ids": arms[arm]["job_ids"],
                                  "runs_s": [round(w, 2) for w in g["walltime_runs_s"]],
                                  "median_s": round(g["walltime_s"], 2),
                                  "ci_s": [round(g["ci_lower_s"], 2), round(g["ci_upper_s"], 2)],
                                  "ci_coverage": g["ci_coverage"], "rel_mad": round(g["rel_mad"], 3),
                                  "range_rel": round(g["spread_rel"], 3)} for arm, g in agg.items()},
                   "null_test": {**null, "pass": null_ok}, "real_test": {**real, "pass": real_ok}},
                  f, indent=2, default=str)
    logger.info("wrote %s", args.output)
    sys.exit(0 if (null_ok and real_ok) else 1)


if __name__ == "__main__":
    main()
