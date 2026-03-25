"""
IOPrescriber Step 5: Closed-Loop Validation on Delta.

Validates LLM recommendations by:
1. Running the "bad" benchmark config → collect Darshan log
2. Running the "fixed" benchmark config → collect Darshan log
3. Parsing both logs → extract features
4. Computing actual speedup (bandwidth improvement)

This is the "killer metric" for SC acceptance.
AIIO showed 1.8x-146x, STELLAR showed near-optimal in <5 attempts.

For IOR benchmarks, the "fix" is already known:
  - small_posix (-t 64) → fix: -t 1048576 (increase transfer size)
  - interface_posix_shared → fix: -a MPIIO -c (use collective I/O)
  - fsync_per_write → fix: remove -Y (no per-write fsync)

The LLM recommendation is validated by checking it matches the known fix.

Usage:
    validator = Validator()
    result = validator.validate_fix("ior_small_posix", bad_config, good_config)
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# Known benchmark fix pairs for closed-loop validation
# Each entry: bad config, good config, expected bottleneck, expected fix description
VALIDATION_PAIRS = {
    "ior_small_posix": {
        "description": "Small POSIX writes (64B) → Large writes (1MB)",
        "bottleneck": "access_granularity",
        "bad_config": {
            "api": "POSIX", "transfer_size": "64", "block_size": "1M",
            "segments": "100", "file_per_proc": True,
            "extra_flags": "-e -C -w -r",
        },
        "good_config": {
            "api": "POSIX", "transfer_size": "1048576", "block_size": "100M",
            "segments": "10", "file_per_proc": True,
            "extra_flags": "-e -C -w -r",
        },
        "expected_fix": "Increase transfer size from 64B to 1MB",
    },
    "ior_interface_posix_shared": {
        "description": "POSIX on shared file → MPI-IO collective",
        "bottleneck": "interface_choice",
        "bad_config": {
            "api": "POSIX", "transfer_size": "1048576", "block_size": "100M",
            "segments": "10", "file_per_proc": False,
            "extra_flags": "-e -C -w -r",
        },
        "good_config": {
            "api": "MPIIO", "transfer_size": "1048576", "block_size": "100M",
            "segments": "10", "file_per_proc": False,
            "extra_flags": "-c -w -r",  # -c = collective
        },
        "expected_fix": "Use MPI-IO collective instead of POSIX for shared file",
    },
    "ior_fsync_heavy": {
        "description": "fsync after every write → fsync only at end",
        "bottleneck": "throughput_utilization",
        "bad_config": {
            "api": "POSIX", "transfer_size": "65536", "block_size": "100M",
            "segments": "10", "file_per_proc": True,
            "extra_flags": "-e -C -Y -w -r",  # -Y = fsync per write
        },
        "good_config": {
            "api": "POSIX", "transfer_size": "65536", "block_size": "100M",
            "segments": "10", "file_per_proc": True,
            "extra_flags": "-e -C -w -r",  # no -Y
        },
        "expected_fix": "Remove per-write fsync (-Y flag)",
    },
    "ior_random_to_sequential": {
        "description": "Random access → Sequential access",
        "bottleneck": "access_pattern",
        "bad_config": {
            "api": "POSIX", "transfer_size": "65536", "block_size": "100M",
            "segments": "10", "file_per_proc": True,
            "extra_flags": "-e -C -z -w -r",  # -z = random
        },
        "good_config": {
            "api": "POSIX", "transfer_size": "65536", "block_size": "100M",
            "segments": "10", "file_per_proc": True,
            "extra_flags": "-e -C -w -r",  # no -z = sequential
        },
        "expected_fix": "Use sequential access instead of random (-z flag)",
    },
    # --- mdtest pairs (metadata workloads) ---
    "mdtest_metadata_storm": {
        "description": "Shared directory MDS contention → Unique directory per rank",
        "bottleneck": "metadata_intensity",
        "benchmark": "mdtest",
        "bad_config": {
            "items_per_rank": 10000, "write_bytes": 100,
            "files_only": True, "unique_dir": False,
        },
        "good_config": {
            "items_per_rank": 10000, "write_bytes": 100,
            "files_only": True, "unique_dir": True,
        },
        "expected_fix": "Use unique directory per rank (-u) to reduce MDS contention",
        "metric": "create_rate_ops_sec",
    },
    "mdtest_fpp_explosion": {
        "description": "Excessive file-per-process (640K files) → Fewer files (6.4K)",
        "bottleneck": "file_strategy",
        "benchmark": "mdtest",
        "bad_config": {
            "items_per_rank": 5000, "write_bytes": 4096,
            "files_only": True, "unique_dir": True,
        },
        "good_config": {
            "items_per_rank": 50, "write_bytes": 4096,
            "files_only": True, "unique_dir": True,
        },
        "expected_fix": "Reduce files per rank from 5000 to 50 to lower metadata overhead",
        "metric": "create_rate_ops_sec",
        "ntasks_override": 128,
        "nodes_override": 1,
    },
    # --- IOR pairs at scale ---
    "ior_collective_vs_independent": {
        "description": "Independent MPI-IO → Collective MPI-IO at 64-rank scale",
        "bottleneck": "interface_choice",
        "bad_config": {
            "api": "MPIIO", "transfer_size": "1048576", "block_size": "100M",
            "segments": "10", "file_per_proc": False,
            "extra_flags": "-e -C -w -r",  # no -c = independent
        },
        "good_config": {
            "api": "MPIIO", "transfer_size": "1048576", "block_size": "100M",
            "segments": "10", "file_per_proc": False,
            "extra_flags": "-c -w -r",  # -c = collective
        },
        "expected_fix": "Enable collective MPI-IO (-c) for shared-file access at scale",
        "ntasks_override": 64,
        "nodes_override": 4,
    },
    "ior_small_to_large_direct": {
        "description": "Small O_DIRECT transfers (4KB) → Large O_DIRECT (1MB)",
        "bottleneck": "access_granularity",
        "bad_config": {
            "api": "POSIX", "transfer_size": "4096", "block_size": "1M",
            "segments": "100", "file_per_proc": True,
            "extra_flags": "-e -C -w -r -O useO_DIRECT=1",
        },
        "good_config": {
            "api": "POSIX", "transfer_size": "1048576", "block_size": "100M",
            "segments": "10", "file_per_proc": True,
            "extra_flags": "-e -C -w -r -O useO_DIRECT=1",
        },
        "expected_fix": "Increase transfer size from 4KB to 1MB with O_DIRECT",
    },
}


class Validator:
    """Closed-loop validation: run benchmark before/after, measure speedup."""

    def __init__(self, account="bdau-delta-cpu", partition="cpu",
                 nodes=1, ntasks=16, walltime="00:30:00",
                 scratch_dir=None, darshan_log_dir=None):
        self.account = account
        self.partition = partition
        self.nodes = nodes
        self.ntasks = ntasks
        self.walltime = walltime
        self.scratch_dir = scratch_dir or "/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/ioprescriber"
        self.darshan_log_dir = str(PROJECT_DIR / "data" / "benchmark_logs" / "ioprescriber")
        self.darshan_lib = "/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so"
        self.results_dir = str(PROJECT_DIR / "results" / "closed_loop")

        os.makedirs(self.scratch_dir, exist_ok=True)
        os.makedirs(self.darshan_log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        logger.info("Validator initialized: account=%s, ntasks=%d, scratch=%s",
                    account, ntasks, self.scratch_dir)

    def _build_ior_command(self, config):
        """Build IOR command from config dict."""
        cmd = "ior"
        cmd += f" -a {config['api']}"
        cmd += f" -t {config['transfer_size']}"
        cmd += f" -b {config['block_size']}"
        cmd += f" -s {config['segments']}"
        if config.get("file_per_proc"):
            cmd += " -F"
        cmd += f" {config.get('extra_flags', '')}"
        cmd += f" -o {self.scratch_dir}/ior_test_file"
        return cmd

    def _build_mdtest_command(self, config):
        """Build mdtest command from config dict."""
        cmd = "mdtest"
        cmd += f" -n {config['items_per_rank']}"
        if config.get("write_bytes", 0) > 0:
            cmd += f" -w {config['write_bytes']}"
            cmd += f" -e {config.get('read_bytes', config['write_bytes'])}"
        if config.get("files_only"):
            cmd += " -F"
        if config.get("unique_dir"):
            cmd += " -u"
        if config.get("stagger_shift"):
            cmd += " -N -1"
        if config.get("tree_depth"):
            cmd += f" -z {config['tree_depth']}"
        if config.get("branching"):
            cmd += f" -b {config['branching']}"
        cmd += f" -d {self.scratch_dir}/mdtest_out"
        return cmd

    def _detect_benchmark_type(self, pair_name):
        """Determine benchmark type from pair config or name."""
        pair = VALIDATION_PAIRS.get(pair_name, {})
        if pair.get("benchmark") == "mdtest":
            return "mdtest"
        if pair_name.startswith("mdtest_"):
            return "mdtest"
        return "ior"

    def _generate_slurm_script(self, job_name, bench_command, benchmark_type="ior",
                               nodes=None, ntasks=None):
        """Generate SLURM batch script for IOR or mdtest benchmark.

        Args:
            job_name: SLURM job name
            bench_command: Full benchmark command string
            benchmark_type: 'ior' or 'mdtest' (controls module loading, cleanup)
            nodes: Override node count (default: self.nodes)
            ntasks: Override task count (default: self.ntasks)
        """
        job_nodes = nodes or self.nodes
        job_ntasks = ntasks or self.ntasks

        # mdtest is bundled with IOR module on Delta
        module_load = "module load ior/3.3.0-gcc13.3.1"

        if benchmark_type == "mdtest":
            cleanup_cmd = f"rm -rf {self.scratch_dir}/mdtest_out* 2>/dev/null || true"
            scratch_setup = f"mkdir -p {self.scratch_dir}/mdtest_out"
        else:
            cleanup_cmd = f"rm -f {self.scratch_dir}/ior_test_file* 2>/dev/null || true"
            scratch_setup = f"mkdir -p {self.scratch_dir}"

        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={self.partition}
#SBATCH --account={self.account}
#SBATCH --nodes={job_nodes}
#SBATCH --ntasks={job_ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time={self.walltime}
#SBATCH --output={self.results_dir}/{job_name}_%j.out
#SBATCH --error={self.results_dir}/{job_name}_%j.err

{module_load}

export DARSHAN_LOGPATH="{self.darshan_log_dir}"
mkdir -p "${{DARSHAN_LOGPATH}}"

# Cleanup on exit
cleanup() {{ {cleanup_cmd}; }}
trap cleanup EXIT

echo "============================================================"
echo "IOPrescriber Closed-Loop Validation"
echo "Job: {job_name}"
echo "Benchmark: {benchmark_type}"
echo "Command: {bench_command}"
echo "Nodes: {job_nodes}, Tasks: {job_ntasks}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

{scratch_setup}

# Run benchmark with Darshan instrumentation
srun --cpu-bind=none --export=ALL,LD_PRELOAD={self.darshan_lib} \\
    {bench_command}

echo ""
echo "Exit code: $?"
echo "Completed: $(date)"

# Check for Darshan log
LOGS=$(ls -t ${{DARSHAN_LOGPATH}}/*.darshan 2>/dev/null | head -1)
if [ -n "$LOGS" ]; then
    echo "Darshan log: $LOGS"
else
    echo "WARNING: No Darshan log found"
fi
"""
        script_path = Path(self.results_dir) / f"{job_name}.slurm"
        with open(script_path, "w") as f:
            f.write(script)
        os.chmod(script_path, 0o755)
        return str(script_path)

    def submit_and_wait(self, script_path, timeout_seconds=1800):
        """Submit SLURM job and wait for completion.

        Returns job_id or None if failed.
        """
        # Submit
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error("sbatch failed: %s", result.stderr)
            return None

        job_id = result.stdout.strip().split()[-1]
        logger.info("  Submitted job %s", job_id)

        # Wait for completion
        t0 = time.time()
        while time.time() - t0 < timeout_seconds:
            check = subprocess.run(
                ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
                capture_output=True, text=True,
            )
            states = [s.strip() for s in check.stdout.strip().split("\n") if s.strip()]
            if any(s in ["COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"] for s in states):
                final_state = states[0] if states else "UNKNOWN"
                logger.info("  Job %s finished: %s (%.0fs)",
                            job_id, final_state, time.time() - t0)
                return job_id if final_state == "COMPLETED" else None
            time.sleep(30)

        logger.error("  Job %s timed out after %ds", job_id, timeout_seconds)
        return None

    def find_darshan_log(self, job_id):
        """Find the Darshan log for a completed job."""
        import glob
        pattern = os.path.join(self.darshan_log_dir, f"*id{job_id}*")
        logs = sorted(glob.glob(pattern))
        return logs[0] if logs else None

    def parse_darshan_for_metrics(self, darshan_path):
        """Parse Darshan log and extract key I/O metrics."""
        import sys
        sys.path.insert(0, str(PROJECT_DIR))
        from src.data.parse_darshan import parse_darshan_log
        from src.data.feature_extraction import extract_raw_features
        from src.data.preprocessing import stage3_engineer
        import pandas as pd

        parsed = parse_darshan_log(darshan_path)
        if parsed is None:
            return None

        raw = extract_raw_features(parsed)
        df = pd.DataFrame([raw])
        df = stage3_engineer(df)
        features = df.iloc[0].to_dict()

        # Extract key metrics
        metrics = {
            "total_bw_mb_s": features.get("total_bw_mb_s", 0),
            "write_bw_mb_s": features.get("write_bw_mb_s", 0),
            "read_bw_mb_s": features.get("read_bw_mb_s", 0),
            "avg_write_size": features.get("avg_write_size", 0),
            "seq_write_ratio": features.get("seq_write_ratio", 0),
            "small_io_ratio": features.get("small_io_ratio", 0),
            "metadata_time_ratio": features.get("metadata_time_ratio", 0),
            "POSIX_BYTES_WRITTEN": features.get("POSIX_BYTES_WRITTEN", 0),
            "POSIX_WRITES": features.get("POSIX_WRITES", 0),
            "runtime_seconds": features.get("runtime_seconds", 0),
        }
        return metrics, features

    def validate_pair(self, pair_name, submit_jobs=True):
        """Run closed-loop validation for a known bad/good pair.

        Args:
            pair_name: key from VALIDATION_PAIRS
            submit_jobs: if True, submit to SLURM. If False, look for existing logs.

        Returns:
            dict with before/after metrics and speedup
        """
        if pair_name not in VALIDATION_PAIRS:
            raise ValueError(f"Unknown pair: {pair_name}. Available: {list(VALIDATION_PAIRS.keys())}")

        pair = VALIDATION_PAIRS[pair_name]
        benchmark_type = self._detect_benchmark_type(pair_name)
        logger.info("")
        logger.info("=" * 60)
        logger.info("CLOSED-LOOP VALIDATION: %s", pair_name)
        logger.info("  %s", pair["description"])
        logger.info("  Expected bottleneck: %s", pair["bottleneck"])
        logger.info("  Benchmark type: %s", benchmark_type)
        logger.info("=" * 60)

        # Build command based on benchmark type
        if benchmark_type == "mdtest":
            bad_cmd = self._build_mdtest_command(pair["bad_config"])
            good_cmd = self._build_mdtest_command(pair["good_config"])
        else:
            bad_cmd = self._build_ior_command(pair["bad_config"])
            good_cmd = self._build_ior_command(pair["good_config"])

        # Per-pair node/task overrides (for multi-node experiments)
        pair_nodes = pair.get("nodes_override", None)
        pair_ntasks = pair.get("ntasks_override", None)

        logger.info("  Bad command:  %s", bad_cmd)
        logger.info("  Good command: %s", good_cmd)
        if pair_nodes or pair_ntasks:
            logger.info("  Override: nodes=%s, ntasks=%s", pair_nodes, pair_ntasks)

        result = {
            "pair_name": pair_name,
            "description": pair["description"],
            "bottleneck": pair["bottleneck"],
            "expected_fix": pair["expected_fix"],
            "benchmark_type": benchmark_type,
            "bad_command": bad_cmd,
            "good_command": good_cmd,
        }
        if pair.get("metric"):
            result["metric"] = pair["metric"]

        if submit_jobs:
            # Submit bad config
            logger.info("  Submitting 'bad' config...")
            bad_script = self._generate_slurm_script(
                f"iop_bad_{pair_name}", bad_cmd,
                benchmark_type=benchmark_type,
                nodes=pair_nodes, ntasks=pair_ntasks,
            )
            bad_job = self.submit_and_wait(bad_script)

            if bad_job:
                bad_log = self.find_darshan_log(bad_job)
                if bad_log:
                    bad_metrics, bad_features = self.parse_darshan_for_metrics(bad_log)
                    result["bad_metrics"] = bad_metrics
                    result["bad_job_id"] = bad_job
                    result["bad_darshan"] = bad_log
                    logger.info("  Bad BW: %.2f MB/s", bad_metrics["total_bw_mb_s"])

            # Submit good config
            logger.info("  Submitting 'good' config...")
            good_script = self._generate_slurm_script(
                f"iop_good_{pair_name}", good_cmd,
                benchmark_type=benchmark_type,
                nodes=pair_nodes, ntasks=pair_ntasks,
            )
            good_job = self.submit_and_wait(good_script)

            if good_job:
                good_log = self.find_darshan_log(good_job)
                if good_log:
                    good_metrics, good_features = self.parse_darshan_for_metrics(good_log)
                    result["good_metrics"] = good_metrics
                    result["good_job_id"] = good_job
                    result["good_darshan"] = good_log
                    logger.info("  Good BW: %.2f MB/s", good_metrics["total_bw_mb_s"])

            # Compute speedup
            if "bad_metrics" in result and "good_metrics" in result:
                bad_bw = result["bad_metrics"]["total_bw_mb_s"] or 0.001
                good_bw = result["good_metrics"]["total_bw_mb_s"] or 0.001
                speedup = round(good_bw / bad_bw, 2)
                result["speedup"] = speedup
                result["status"] = "validated"
                logger.info("")
                logger.info("  SPEEDUP: %.2fx (%s → %s MB/s)",
                            speedup, f"{bad_bw:.1f}", f"{good_bw:.1f}")
            else:
                result["status"] = "incomplete"
                logger.warning("  Validation incomplete — missing job results")

        else:
            result["status"] = "dry_run"
            logger.info("  Dry run — no jobs submitted")

        return result

    def validate_all(self, submit_jobs=True):
        """Run closed-loop validation on all known pairs."""
        all_results = {}
        for pair_name in VALIDATION_PAIRS:
            result = self.validate_pair(pair_name, submit_jobs=submit_jobs)
            all_results[pair_name] = result

        # Save results
        results_path = Path(self.results_dir) / "closed_loop_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info("Results saved: %s", results_path)

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("CLOSED-LOOP VALIDATION SUMMARY")
        logger.info("=" * 60)
        for name, r in all_results.items():
            speedup = r.get("speedup", "N/A")
            status = r.get("status", "unknown")
            logger.info("  %-30s speedup=%-8s status=%s", name, f"{speedup}x", status)
        logger.info("=" * 60)

        return all_results
