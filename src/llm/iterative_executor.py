"""
SLURM execution wrapper for Track C iterative optimization.

Handles: SLURM script generation, job submission, polling, Darshan log
discovery, and feature extraction for the iterative feedback loop.

Refactored from src/ioprescriber/validator.py for reusable iteration support.
"""

import glob
import json
import logging
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


class IterativeExecutor:
    """Execute benchmark commands via SLURM and parse Darshan results."""

    def __init__(self, config):
        """Initialize from iterative.yaml slurm section.

        Args:
            config: dict from yaml.safe_load(iterative.yaml)["slurm"]
        """
        slurm_cfg = config.get("slurm", config)
        self.account = slurm_cfg.get("account", "bdau-delta-cpu")
        self.partition = slurm_cfg.get("partition", "cpu")
        self.nodes = slurm_cfg.get("nodes", 1)
        self.ntasks = slurm_cfg.get("ntasks", 16)
        self.walltime = slurm_cfg.get("walltime", "00:10:00")
        self.scratch_dir = slurm_cfg.get(
            "scratch_dir",
            "/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative",
        )
        self.darshan_log_dir = str(
            PROJECT_DIR / slurm_cfg.get("darshan_log_dir", "data/benchmark_logs/iterative")
        )
        self.darshan_lib = slurm_cfg.get(
            "darshan_lib",
            "/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so",
        )
        self.results_dir = str(PROJECT_DIR / "results" / "iterative")

        os.makedirs(self.scratch_dir, exist_ok=True)
        os.makedirs(self.darshan_log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        logger.info(
            "IterativeExecutor: account=%s, partition=%s, ntasks=%d",
            self.account, self.partition, self.ntasks,
        )

    def generate_slurm_script(self, job_name, benchmark_command, benchmark_type="ior"):
        """Generate a SLURM batch script for a benchmark run.

        Args:
            job_name: unique job identifier
            benchmark_command: the IOR/mdtest command to run
            benchmark_type: 'ior' or 'mdtest'

        Returns:
            path to the generated .slurm script
        """
        module_load = "module load ior/3.3.0-gcc13.3.1"

        # Use per-job scratch directory to prevent file conflicts between concurrent runs
        job_scratch = f"{self.scratch_dir}/{job_name}"

        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={self.partition}
#SBATCH --account={self.account}
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks={self.ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output={self.results_dir}/{job_name}_%j.out
#SBATCH --error={self.results_dir}/{job_name}_%j.err

{module_load}

export DARSHAN_LOGPATH="{self.darshan_log_dir}"
mkdir -p "${{DARSHAN_LOGPATH}}"

# Fix SLURM env var conflicts on Delta
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null
export SLURM_CPUS_PER_TASK=1

# Per-job scratch to avoid file conflicts between concurrent runs
JOB_SCRATCH="{job_scratch}"
mkdir -p "$JOB_SCRATCH"

# Cleanup benchmark files on exit
cleanup() {{ rm -rf "$JOB_SCRATCH" 2>/dev/null || true; }}
trap cleanup EXIT

echo "============================================================"
echo "Track C Iterative Optimization - Benchmark Execution"
echo "Job: {job_name}"
echo "Command: {benchmark_command}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

# Run benchmark with Darshan instrumentation
srun --cpu-bind=none --export=ALL,LD_PRELOAD={self.darshan_lib} \\
    {benchmark_command}

EXIT_CODE=$?
echo ""
echo "Exit code: $EXIT_CODE"
echo "Completed: $(date)"

# Report Darshan log location
LOGS=$(ls -t ${{DARSHAN_LOGPATH}}/*.darshan 2>/dev/null | head -1)
if [ -n "$LOGS" ]; then
    echo "Darshan log: $LOGS"
else
    echo "WARNING: No Darshan log found"
fi

exit $EXIT_CODE
"""
        script_path = Path(self.results_dir) / f"{job_name}.slurm"
        with open(script_path, "w") as f:
            f.write(script)
        os.chmod(script_path, 0o755)
        return str(script_path)

    def submit_and_wait(self, script_path, timeout_seconds=7200, poll_interval=30):
        """Submit SLURM job and wait for completion.

        Args:
            script_path: path to .slurm script
            timeout_seconds: max wait time (default 10 min for benchmarks)
            poll_interval: seconds between sacct polls

        Returns:
            job_id string if successful, None if failed
        """
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error("sbatch failed: %s", result.stderr.strip())
            return None

        job_id = result.stdout.strip().split()[-1]
        logger.info("  Submitted SLURM job %s", job_id)

        t0 = time.time()
        while time.time() - t0 < timeout_seconds:
            try:
                check = subprocess.run(
                    ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
                    capture_output=True, text=True, timeout=30,
                )
                states = [s.strip() for s in check.stdout.strip().split("\n") if s.strip()]
                terminal = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"}
                if any(s in terminal for s in states):
                    final_state = states[0] if states else "UNKNOWN"
                    elapsed = time.time() - t0
                    logger.info("  Job %s: %s (%.0fs)", job_id, final_state, elapsed)
                    return job_id if final_state == "COMPLETED" else None
            except subprocess.TimeoutExpired:
                pass
            time.sleep(poll_interval)

        logger.error("  Job %s timed out after %ds", job_id, timeout_seconds)
        # Try to cancel the timed-out job
        subprocess.run(["scancel", job_id], capture_output=True)
        return None

    def find_darshan_log(self, job_id):
        """Find the Darshan log file for a completed SLURM job.

        Args:
            job_id: SLURM job ID string

        Returns:
            path to .darshan file or None
        """
        pattern = os.path.join(self.darshan_log_dir, f"*id{job_id}*")
        logs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if logs:
            logger.info("  Darshan log: %s", Path(logs[0]).name)
            return logs[0]
        logger.warning("  No Darshan log found for job %s", job_id)
        return None

    def extract_features(self, darshan_path):
        """Parse Darshan log and extract full feature vector + key metrics.

        Args:
            darshan_path: path to .darshan file

        Returns:
            (metrics_dict, full_features_dict) or (None, None) on failure
        """
        import sys
        sys.path.insert(0, str(PROJECT_DIR))

        try:
            from src.data.parse_darshan import parse_darshan_log
            from src.data.feature_extraction import extract_raw_features
            from src.data.preprocessing import stage3_engineer
            import pandas as pd

            parsed = parse_darshan_log(darshan_path)
            if parsed is None:
                logger.error("  Failed to parse Darshan log: %s", darshan_path)
                return None, None

            raw = extract_raw_features(parsed)
            df = pd.DataFrame([raw])
            df = stage3_engineer(df)
            features = df.iloc[0].to_dict()

            # Key metrics for reporting
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
                "POSIX_FSYNCS": features.get("POSIX_FSYNCS", 0),
                "runtime_seconds": features.get("runtime_seconds", 0),
                "nprocs": features.get("nprocs", 0),
            }
            return metrics, features

        except Exception as e:
            logger.error("  Feature extraction failed: %s", e)
            return None, None

    def parse_ior_output(self, job_id):
        """Parse IOR stdout to extract write/read bandwidth.

        Returns dict with ior_write_bw_mibs, ior_read_bw_mibs, or None.
        """
        out_pattern = os.path.join(self.results_dir, f"*_{job_id}.out")
        out_files = glob.glob(out_pattern)
        if not out_files:
            return None

        result = {}
        with open(out_files[0]) as f:
            for line in f:
                if "Max Write:" in line:
                    try:
                        result["ior_write_bw_mibs"] = float(line.split()[2])
                    except (IndexError, ValueError):
                        pass
                if "Max Read:" in line:
                    try:
                        result["ior_read_bw_mibs"] = float(line.split()[2])
                    except (IndexError, ValueError):
                        pass
        return result if result else None

    def execute_benchmark(self, benchmark_command, job_name, benchmark_type="ior"):
        """Full execution cycle: generate script -> submit -> wait -> parse.

        Args:
            benchmark_command: IOR/mdtest command string
            job_name: unique identifier
            benchmark_type: 'ior' or 'mdtest'

        Returns:
            dict with:
                success: bool
                job_id: str or None
                metrics: dict of Darshan-extracted metrics or None
                features: dict of full feature vector or None
                ior_output: dict of IOR-reported BW or None
                darshan_path: str or None
                elapsed_s: float
        """
        t0 = time.time()
        result = {
            "success": False,
            "job_id": None,
            "metrics": None,
            "features": None,
            "ior_output": None,
            "darshan_path": None,
            "elapsed_s": 0,
        }

        # Generate and submit
        script = self.generate_slurm_script(job_name, benchmark_command, benchmark_type)
        job_id = self.submit_and_wait(script)

        if not job_id:
            result["elapsed_s"] = time.time() - t0
            return result

        result["job_id"] = job_id

        # Find and parse Darshan log
        darshan_path = self.find_darshan_log(job_id)
        if darshan_path:
            result["darshan_path"] = darshan_path
            metrics, features = self.extract_features(darshan_path)
            if metrics:
                result["metrics"] = metrics
                result["features"] = features
                result["success"] = True

        # Also parse IOR stdout for write BW
        if benchmark_type == "ior":
            ior_out = self.parse_ior_output(job_id)
            if ior_out:
                result["ior_output"] = ior_out

        result["elapsed_s"] = time.time() - t0
        return result
