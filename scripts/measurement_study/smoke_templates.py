"""Run one baseline job for every benchmark type and fail if any does not complete.

Files of passing runs are moved to `.codex-trash/smoke/` afterwards (use --keep to leave them
in place); a failing type keeps its job script, stdout/stderr and Darshan logs for diagnosis.

Meant to be run from a batch driver (scripts/measurement_study/run_study.slurm): that is the
situation in which a job template inherits the driver's SLURM_* variables, so it checks the
templates under the same conditions as the sweep.

Usage: python scripts/measurement_study/smoke_templates.py [--types ior mdtest ...]
"""
import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.llm.benchmark_command_builder import BenchmarkCommandBuilder  # noqa: E402
from src.llm.iterative_executor import IterativeExecutor  # noqa: E402

logger = logging.getLogger("smoke_templates")

REPRESENTATIVE = {"ior": "ior_small_posix", "mdtest": "mdtest_metadata_storm",
                  "hacc_io": "hacc_posix_shared_small", "custom": "custom_load_imbalance",
                  "h5bench": "h5bench_small_access", "dlio": "dlio_small_records"}


def build(executor, builder, btype, params, job):
    scratch = f"{executor.scratch_dir}/{job}"
    kwargs = {}
    if btype == "hacc_io":
        _, s, _ = builder.validate_hacc_params(params)
        cmd = builder.build_hacc_command(s, output_dir=scratch); kwargs["hacc_config"] = s
    elif btype == "custom":
        _, s, _ = builder.validate_custom_params(params)
        cmd = builder.build_custom_command(s, output_dir=scratch)
    elif btype == "h5bench":
        _, s, _ = builder.validate_h5bench_params(params)
        w, r, _ = builder.build_h5bench_config(s, output_dir=scratch,
                                               config_path=f"{executor.results_dir}/{job}_config.json")
        cmd = (w, r); kwargs["h5bench_config"] = s
    elif btype == "dlio":
        _, s, _ = builder.validate_dlio_params(params)
        cmd = builder.build_dlio_command(s, data_dir=scratch); kwargs["dlio_config"] = s
    elif btype == "mdtest":
        cmd = builder.build_mdtest_command(params, output_dir=scratch)
    else:
        _, s, _ = builder.validate_ior_params(params)
        cmd = builder.build_ior_command(s, output_dir=scratch)
    return executor.generate_slurm_script(job, cmd, btype, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--types", nargs="+", default=list(REPRESENTATIVE))
    parser.add_argument("--config", default=str(PROJECT_DIR / "configs" / "iterative.yaml"))
    parser.add_argument("--keep", action="store_true",
                        help="Leave job scripts, stdout/stderr and Darshan logs of passing types in place "
                             "(default: moved to .codex-trash/smoke/; a failing type always keeps its files)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(open(args.config))
    executor = IterativeExecutor(cfg)
    builder = BenchmarkCommandBuilder(config_path=args.config)
    scripts = {t: build(executor, builder, t, cfg["workloads"][REPRESENTATIVE[t]]["bad_config"], f"smoke_{t}")
               for t in args.types}
    # Timing is not measured here, so the jobs may run side by side.
    with ThreadPoolExecutor(max_workers=len(scripts)) as pool:
        futures = {t: pool.submit(executor.submit_and_wait, s, 3600, 15) for t, s in scripts.items()}
        results = {t: f.result() for t, f in futures.items()}
    failed = []
    for t, job_id in results.items():
        logs = executor.find_darshan_logs(job_id) if job_id else []
        ok = bool(job_id) and bool(logs)
        logger.info("%-8s job %s  darshan logs %d  -> %s", t, job_id, len(logs), "PASS" if ok else "FAIL")
        if not ok:
            failed.append(t)
        elif not args.keep:
            # A passing smoke run is not evidence of anything later, so it leaves nothing in the
            # working tree. Files are moved to the trash folder, never deleted (no-delete policy).
            stale = [Path(p) for p in logs]
            stale += list(Path(executor.results_dir).glob(f"smoke_{t}_{job_id}.*"))
            stale += [Path(executor.results_dir) / f"smoke_{t}.slurm",
                      Path(executor.results_dir) / f"smoke_{t}_config.json"]
            trash = PROJECT_DIR / ".codex-trash" / "smoke" / f"{t}_{job_id}"
            moved = 0
            for f in stale:
                if f.is_file():
                    trash.mkdir(parents=True, exist_ok=True)
                    f.rename(trash / f.name)
                    moved += 1
            logger.info("%-8s moved %d files of the passing run to %s", t, moved, trash)
    if failed:
        logger.error("failed benchmark types: %s", failed)
        sys.exit(1)
    logger.info("all %d benchmark types completed with Darshan logs", len(results))


if __name__ == "__main__":
    main()
