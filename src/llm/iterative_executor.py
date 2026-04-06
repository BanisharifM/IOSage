"""
SLURM execution wrapper for IOSage iterative closed-loop optimization.

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

    def generate_slurm_script(self, job_name, benchmark_command, benchmark_type="ior",
                             hacc_config=None, h5bench_config=None, dlio_config=None):
        """Generate a SLURM batch script for a benchmark run.

        Args:
            job_name: unique job identifier
            benchmark_command: the IOR/mdtest/HACC-IO/custom command to run
                For h5bench: (write_cmd, read_cmd) tuple
                For DLIO: (datagen_cmd, training_cmd) tuple
            benchmark_type: 'ior', 'mdtest', 'hacc_io', 'custom', 'h5bench', or 'dlio'
            hacc_config: dict with collective_buffering key (for HACC-IO)
            h5bench_config: dict with COLLECTIVE_DATA key (for h5bench)
            dlio_config: dict with DLIO params (for DLIO)

        Returns:
            path to the generated .slurm script
        """
        # Use per-job scratch directory to prevent file conflicts between concurrent runs.
        # Include $SLURM_JOB_ID to ensure uniqueness when the same workload runs
        # concurrently with different LLM models (e.g., 3 LLMs × same baseline).
        # Without this, the EXIT cleanup trap from the first job to finish
        # deletes the scratch dir while other jobs are still using it.
        job_scratch = f"{self.scratch_dir}/{job_name}_${{SLURM_JOB_ID}}"

        if benchmark_type == "h5bench":
            script = self._generate_h5bench_slurm(
                job_name, benchmark_command, job_scratch, h5bench_config or {}
            )
        elif benchmark_type == "dlio":
            script = self._generate_dlio_slurm(
                job_name, benchmark_command, job_scratch, dlio_config or {}
            )
        elif benchmark_type == "hacc_io":
            script = self._generate_hacc_slurm(
                job_name, benchmark_command, job_scratch, hacc_config or {}
            )
        elif benchmark_type == "custom":
            script = self._generate_custom_slurm(
                job_name, benchmark_command, job_scratch
            )
        else:
            # IOR / mdtest
            module_load = "module load ior/3.3.0-gcc13.3.1"
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
echo "IOSage Iterative Optimization - Benchmark Execution"
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
LOGS=$(ls -t ${{DARSHAN_LOGPATH}}/*id${{SLURM_JOB_ID}}* 2>/dev/null | head -1)
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

    def _generate_hacc_slurm(self, job_name, benchmark_command, job_scratch,
                             hacc_config):
        """Generate SLURM script for HACC-IO benchmark."""
        cb = hacc_config.get("collective_buffering", "disabled")
        if cb == "enabled":
            hints_line = 'export MPICH_MPIIO_HINTS="*:romio_cb_write=enable:romio_ds_write=disable"'
        else:
            hints_line = (
                'export MPICH_MPIIO_HINTS='
                '"*:romio_cb_write=disable:romio_cb_read=disable'
                ':romio_ds_write=disable:romio_ds_read=disable"'
            )

        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={self.partition}
#SBATCH --account={self.account}
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks={self.ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output={self.results_dir}/{job_name}_%j.out
#SBATCH --error={self.results_dir}/{job_name}_%j.err

module load PrgEnv-gnu/8.6.0 2>/dev/null || true

export DARSHAN_LOGPATH="{self.darshan_log_dir}"
mkdir -p "${{DARSHAN_LOGPATH}}"

# ROMIO collective buffering control
{hints_line}
export MPICH_MPIIO_HINTS_DISPLAY=1

# Fix SLURM env var conflicts on Delta
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null
export SLURM_CPUS_PER_TASK=1

# Per-job scratch to avoid file conflicts between concurrent runs
JOB_SCRATCH="{job_scratch}"
mkdir -p "$JOB_SCRATCH"

# Cleanup HACC-IO checkpoint files on exit
cleanup() {{ rm -f "$JOB_SCRATCH"/hacc_checkpoint* 2>/dev/null; rm -rf "$JOB_SCRATCH" 2>/dev/null || true; }}
trap cleanup EXIT

echo "============================================================"
echo "IOSage Iterative Optimization - HACC-IO Execution"
echo "Job: {job_name}"
echo "Command: {benchmark_command}"
echo "Collective buffering: {cb}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

# Run HACC-IO with Darshan instrumentation
srun --cpu-bind=none --export=ALL,LD_PRELOAD={self.darshan_lib} \\
    {benchmark_command}

EXIT_CODE=$?
echo ""
echo "Exit code: $EXIT_CODE"
echo "Completed: $(date)"

# Cleanup checkpoint data files
rm -f "$JOB_SCRATCH"/hacc_checkpoint* 2>/dev/null || true

# Report Darshan log location
LOGS=$(ls -t ${{DARSHAN_LOGPATH}}/*id${{SLURM_JOB_ID}}* 2>/dev/null | head -1)
if [ -n "$LOGS" ]; then
    echo "Darshan log: $LOGS"
else
    echo "WARNING: No Darshan log found"
fi

exit $EXIT_CODE
"""

    def _generate_custom_slurm(self, job_name, benchmark_command, job_scratch):
        """Generate SLURM script for custom (load_imbalance) benchmark."""
        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={self.partition}
#SBATCH --account={self.account}
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks={self.ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output={self.results_dir}/{job_name}_%j.out
#SBATCH --error={self.results_dir}/{job_name}_%j.err

export DARSHAN_LOGPATH="{self.darshan_log_dir}"
export DARSHAN_ENABLE_NONMPI=1
export DARSHAN_MODMEM=4
mkdir -p "${{DARSHAN_LOGPATH}}"

# Fix SLURM env var conflicts on Delta
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null
export SLURM_CPUS_PER_TASK=1

# Per-job scratch to avoid file conflicts between concurrent runs
JOB_SCRATCH="{job_scratch}"
mkdir -p "$JOB_SCRATCH"

# Cleanup on exit
cleanup() {{ rm -rf "$JOB_SCRATCH" 2>/dev/null || true; }}
trap cleanup EXIT

echo "============================================================"
echo "IOSage Iterative Optimization - Custom Benchmark Execution"
echo "Job: {job_name}"
echo "Command: {benchmark_command}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

# Run custom benchmark with Darshan instrumentation
srun --cpu-bind=none --export=ALL,LD_PRELOAD={self.darshan_lib} \\
    {benchmark_command}

EXIT_CODE=$?
echo ""
echo "Exit code: $EXIT_CODE"
echo "Completed: $(date)"

# Report Darshan log location
LOGS=$(ls -t ${{DARSHAN_LOGPATH}}/*id${{SLURM_JOB_ID}}* 2>/dev/null | head -1)
if [ -n "$LOGS" ]; then
    echo "Darshan log: $LOGS"
else
    echo "WARNING: No Darshan log found"
fi

exit $EXIT_CODE
"""

    def _generate_h5bench_slurm(self, job_name, benchmark_commands, job_scratch,
                                h5bench_config):
        """Generate SLURM script for h5bench benchmark (write + read phases).

        Args:
            job_name: unique job identifier
            benchmark_commands: (write_cmd, read_cmd) tuple
            job_scratch: per-job scratch directory
            h5bench_config: dict with COLLECTIVE_DATA key
        """
        write_cmd, read_cmd = benchmark_commands
        coll = h5bench_config.get("COLLECTIVE_DATA", "NO")
        if coll == "YES":
            hints_line = 'export MPICH_MPIIO_HINTS="*:romio_cb_write=enable:romio_ds_write=disable"'
        else:
            hints_line = (
                'export MPICH_MPIIO_HINTS='
                '"*:romio_cb_write=disable:romio_cb_read=disable'
                ':romio_ds_write=disable:romio_ds_read=disable"'
            )

        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={self.partition}
#SBATCH --account={self.account}
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks={self.ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output={self.results_dir}/{job_name}_%j.out
#SBATCH --error={self.results_dir}/{job_name}_%j.err

module load PrgEnv-gnu/8.6.0 cray-hdf5-parallel/1.14.3.5 2>/dev/null || true

export DARSHAN_LOGPATH="{self.darshan_log_dir}"
mkdir -p "${{DARSHAN_LOGPATH}}"

# HDF5 parallel library path
export LD_LIBRARY_PATH="/opt/cray/pe/hdf5-parallel/1.14.3.5/gnu/12.2/lib:${{LD_LIBRARY_PATH:-}}"

# ROMIO collective buffering control
{hints_line}
export MPICH_MPIIO_HINTS_DISPLAY=1

# Fix SLURM env var conflicts on Delta
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null
export SLURM_CPUS_PER_TASK=1

# Per-job scratch to avoid file conflicts between concurrent runs
JOB_SCRATCH="{job_scratch}"
mkdir -p "$JOB_SCRATCH"

# Cleanup h5bench data files on exit
cleanup() {{ rm -f "$JOB_SCRATCH"/h5bench_output.h5* "$JOB_SCRATCH"/output.csv 2>/dev/null; rm -rf "$JOB_SCRATCH" 2>/dev/null || true; }}
trap cleanup EXIT

echo "============================================================"
echo "IOSage Iterative Optimization - h5bench Execution"
echo "Job: {job_name}"
echo "Collective data: {coll}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

# Pre-flight checks
if [ -x "{write_cmd.split()[0]}" ]; then
    echo "  h5bench_write: OK"
else
    echo "  ERROR: h5bench_write not found"
    exit 1
fi

# === h5bench WRITE phase ===
echo ""
echo "=== h5bench WRITE phase ==="
echo "Command: {write_cmd}"
srun --cpu-bind=none --export=ALL,LD_PRELOAD={self.darshan_lib} \\
    {write_cmd}

WRITE_RC=$?
echo ""
echo "Write completed at $(date), exit code: ${{WRITE_RC}}"

# === h5bench READ phase ===
H5_FILE=$(echo "{read_cmd}" | awk '{{print $NF}}')
if [ -f "$H5_FILE" ]; then
    echo ""
    echo "=== h5bench READ phase ==="
    echo "Command: {read_cmd}"
    srun --cpu-bind=none --export=ALL,LD_PRELOAD={self.darshan_lib} \\
        {read_cmd}

    READ_RC=$?
    echo ""
    echo "Read completed at $(date), exit code: ${{READ_RC}}"
else
    echo "WARNING: HDF5 output file not found, skipping read phase"
fi

echo ""
echo "Exit code (write): $WRITE_RC"
echo "Completed: $(date)"

# Report Darshan log location
LOGS=$(ls -t ${{DARSHAN_LOGPATH}}/*id${{SLURM_JOB_ID}}* 2>/dev/null | head -1)
if [ -n "$LOGS" ]; then
    echo "Darshan log: $LOGS"
else
    echo "WARNING: No Darshan log found"
fi

exit $WRITE_RC
"""

    def _generate_dlio_slurm(self, job_name, benchmark_commands, job_scratch,
                             dlio_config):
        """Generate SLURM script for DLIO benchmark (datagen + training phases).

        Args:
            job_name: unique job identifier
            benchmark_commands: (datagen_cmd, training_cmd) tuple
            job_scratch: per-job scratch directory
            dlio_config: dict with DLIO params
        """
        datagen_cmd, training_cmd = benchmark_commands

        # DLIO requires PyTorch which needs CUDA — must use GPU partition
        gpu_partition = "gpuA100x4"
        gpu_account = "bdau-delta-gpu"

        # Use 4 ranks for DLIO (not 16). The original benchmark sweep that
        # successfully generated 60 GT logs used 4 ranks on CPU. With 16 ranks
        # and tiny records (64 bytes), PyTorch distributed training hangs on
        # inter-rank synchronization barriers because the I/O is negligible
        # relative to synchronization overhead.
        dlio_ntasks = 4

        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={gpu_partition}
#SBATCH --account={gpu_account}
#SBATCH --nodes=1
#SBATCH --ntasks={dlio_ntasks}
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=128g
#SBATCH --time=01:00:00
#SBATCH --output={self.results_dir}/{job_name}_%j.out
#SBATCH --error={self.results_dir}/{job_name}_%j.err

export DARSHAN_LOGPATH="{self.darshan_log_dir}"
export DARSHAN_MODMEM=16
export DARSHAN_ENABLE_NONMPI=1
# Darshan config: exclude Python library paths, increase record limit to 8192
# Without this, Python's 500+ .so/.pyc imports fill all 1024 POSIX record slots
# before DLIO reads any actual data files, making the Darshan log useless.
export DARSHAN_CONFIG_PATH="{PROJECT_DIR}/configs/darshan_dlio.conf"
mkdir -p "${{DARSHAN_LOGPATH}}"

# Fix PyTorch CUDA library conflict on Delta.
# The system anaconda3 ships libnvJitLink.so.12.1.105 (CUDA 12.1) which only
# provides symbols up to __nvJitLinkCreate_12_1.  PyTorch 2.8+cu128 installs
# nvidia-cusparse-cu12 12.5 which requires __nvJitLinkCreate_12_8 from the
# pip-installed nvidia-nvjitlink-cu12 12.8.  Because /sw/external/python/
# anaconda3/lib appears early in LD_LIBRARY_PATH, the old library is loaded
# first and the import fails with "undefined symbol __nvJitLinkCreate_12_8".
# Prepending the pip-installed NVIDIA lib paths ensures the correct versions
# are found before the stale system copies.
NVIDIA_LIB_BASE="/projects/bdau/envs/sc2026/lib/python3.9/site-packages/nvidia"
NVIDIA_LIBS=""
for subdir in "$NVIDIA_LIB_BASE"/*/lib; do
    [ -d "$subdir" ] && NVIDIA_LIBS="${{NVIDIA_LIBS:+$NVIDIA_LIBS:}}$subdir"
done
export LD_LIBRARY_PATH="${{NVIDIA_LIBS:+$NVIDIA_LIBS:}}${{LD_LIBRARY_PATH:-}}"

# Fix SLURM env var conflicts on Delta
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null
export SLURM_CPUS_PER_TASK=4

# Per-job scratch to avoid file conflicts between concurrent runs
JOB_SCRATCH="{job_scratch}"
mkdir -p "$JOB_SCRATCH"

# Cleanup DLIO data on exit
cleanup() {{ rm -rf "$JOB_SCRATCH" 2>/dev/null || true; }}
trap cleanup EXIT

echo "============================================================"
echo "IOSage Iterative Optimization - DLIO Execution"
echo "Job: {job_name}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

# Step 1: Generate data (no Darshan needed for data gen)
echo ""
echo "=== DLIO Data Generation Phase ==="
srun --cpu-bind=none --export=ALL \\
    {datagen_cmd}

echo "Data generation complete at $(date)"

# Step 2: Run training (this generates the Darshan log we want)
echo ""
echo "=== DLIO Training Phase (with Darshan) ==="
srun --cpu-bind=none --export=ALL,LD_PRELOAD={self.darshan_lib} \\
    {training_cmd}

EXIT_CODE=$?
echo ""
echo "DLIO training complete at $(date), exit code: $EXIT_CODE"

# Report Darshan log location
LOGS=$(ls -t ${{DARSHAN_LOGPATH}}/*id${{SLURM_JOB_ID}}* 2>/dev/null | head -1)
if [ -n "$LOGS" ]; then
    echo "Darshan log: $LOGS"
else
    echo "WARNING: No Darshan log found"
fi

exit $EXIT_CODE
"""

    def submit_and_wait(self, script_path, timeout_seconds=7200, poll_interval=30):
        """Submit SLURM job and wait for completion.

        IMPORTANT: Unsets conflicting SLURM env vars BEFORE sbatch to prevent
        'step creation disabled' errors on Delta. The parent session (Claude Code)
        sets SLURM_CPUS_PER_TASK=8 and SLURM_TRES_PER_TASK=cpu=8 which conflict
        with the child job's --cpus-per-task=1.

        Args:
            script_path: path to .slurm script
            timeout_seconds: max wait time (default 2h)
            poll_interval: seconds between sacct polls

        Returns:
            job_id string if successful, None if failed
        """
        # Clean SLURM env vars that the parent session sets (e.g., Claude Code job
        # sets SLURM_CPUS_PER_TASK=8, SLURM_TRES_PER_TASK=cpu=8). These get inherited
        # by sbatch and cause "step creation disabled" errors when the child job's
        # --cpus-per-task=1 conflicts with the inherited value.
        clean_env = os.environ.copy()
        # Strip PYTHONPATH: .local_pkgs has numpy 1.24.3 which shadows conda's
        # numpy 1.26.4 and breaks TensorFlow/DLIO (missing np.dtypes)
        clean_env.pop("PYTHONPATH", None)
        for var in ["SLURM_CPUS_PER_TASK", "SLURM_TRES_PER_TASK",
                     "SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU", "SLURM_MEM_PER_NODE",
                     "SLURM_CPU_BIND", "SLURM_CPU_BIND_LIST", "SLURM_CPU_BIND_TYPE",
                     "SLURM_CPU_BIND_VERBOSE", "SLURM_DISTRIBUTION",
                     "SLURM_JOB_CPUS_PER_NODE", "SLURM_NTASKS", "SLURM_NPROCS",
                     "SLURM_NNODES", "SLURM_NODELIST", "SLURM_JOB_NODELIST",
                     "SLURM_STEP_NODELIST", "SLURM_TASKS_PER_NODE",
                     "SLURM_JOB_NUM_NODES", "SLURM_STEP_NUM_TASKS",
                     "SLURM_STEP_NUM_NODES", "SLURM_STEP_TASKS_PER_NODE"]:
            clean_env.pop(var, None)

        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True, text=True,
            env=clean_env,
        )
        if result.returncode != 0:
            logger.error("sbatch failed: %s", result.stderr.strip())
            return None

        job_id = result.stdout.strip().split()[-1]
        logger.info("  Submitted SLURM job %s", job_id)

        t0 = time.time()
        while time.time() - t0 < timeout_seconds:
            try:
                # Query ONLY the batch step (JobID.batch) to get the overall job state.
                # Without this filter, sacct returns states for ALL srun sub-steps
                # (e.g., step .0 = COMPLETED for data gen, step .1 = RUNNING for training).
                # Seeing COMPLETED from a sub-step would cause premature return.
                check = subprocess.run(
                    ["sacct", "-j", job_id, "--format=JobID,State",
                     "--noheader", "--parsable2"],
                    capture_output=True, text=True, timeout=30,
                )
                # Find the main job state (line where JobID is exactly the job_id,
                # not a sub-step like job_id.0 or job_id.batch)
                main_state = None
                for line in check.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.strip().split("|")
                    if len(parts) >= 2 and parts[0].strip() == job_id:
                        main_state = parts[1].strip()
                        break
                if main_state is None:
                    # Fallback: look for .batch step
                    for line in check.stdout.strip().split("\n"):
                        parts = line.strip().split("|")
                        if len(parts) >= 2 and parts[0].strip() == f"{job_id}.batch":
                            main_state = parts[1].strip()
                            break
                if main_state is None:
                    time.sleep(poll_interval)
                    continue

                terminal = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"}
                if main_state in terminal:
                    elapsed = time.time() - t0
                    logger.info("  Job %s: %s (%.0fs)", job_id, main_state, elapsed)
                    return job_id if main_state == "COMPLETED" else None
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

    def parse_hacc_output(self, job_id):
        """Parse HACC-IO stdout to extract write timing and throughput.

        Returns dict with hacc_write_time_s, hacc_total_bytes, or None.
        """
        out_pattern = os.path.join(self.results_dir, f"*_{job_id}.out")
        out_files = glob.glob(out_pattern)
        if not out_files:
            return None

        import re
        result = {}
        with open(out_files[0]) as f:
            for line in f:
                # HACC-IO prints lines like "Checkpoint write: X seconds"
                if "write" in line.lower() and "second" in line.lower():
                    match = re.search(r"(\d+\.?\d*)\s*second", line.lower())
                    if match:
                        result["hacc_write_time_s"] = float(match.group(1))
                if "total" in line.lower() and "byte" in line.lower():
                    match = re.search(r"(\d+)\s*byte", line.lower())
                    if match:
                        result["hacc_total_bytes"] = int(match.group(1))
        return result if result else None

    def parse_custom_output(self, job_id):
        """Parse custom load_imbalance stdout.

        Returns dict with custom_runtime_s, or None.
        """
        out_pattern = os.path.join(self.results_dir, f"*_{job_id}.out")
        out_files = glob.glob(out_pattern)
        if not out_files:
            return None

        import re
        result = {}
        with open(out_files[0]) as f:
            for line in f:
                if "runtime" in line.lower() or "elapsed" in line.lower():
                    match = re.search(r"(\d+\.?\d*)\s*(?:s|sec)", line.lower())
                    if match:
                        result["custom_runtime_s"] = float(match.group(1))
        return result if result else None

    def parse_h5bench_output(self, job_id):
        """Parse h5bench stdout to extract write/read rates.

        Returns dict with h5bench_write_rate, h5bench_read_rate, or None.
        """
        out_pattern = os.path.join(self.results_dir, f"*_{job_id}.out")
        out_files = glob.glob(out_pattern)
        if not out_files:
            return None

        import re
        result = {}
        with open(out_files[0]) as f:
            for line in f:
                # h5bench writes lines like "Write rate: X MB/s"
                if "write" in line.lower() and ("rate" in line.lower() or "mb/s" in line.lower()):
                    match = re.search(r"(\d+\.?\d*)\s*(?:mb/s|mib/s)", line.lower())
                    if match:
                        result["h5bench_write_rate"] = float(match.group(1))
                if "read" in line.lower() and ("rate" in line.lower() or "mb/s" in line.lower()):
                    match = re.search(r"(\d+\.?\d*)\s*(?:mb/s|mib/s)", line.lower())
                    if match:
                        result["h5bench_read_rate"] = float(match.group(1))
                # Also check for "completed" lines with timing
                if "write completed" in line.lower():
                    match = re.search(r"exit code:\s*(\d+)", line.lower())
                    if match:
                        result["h5bench_write_rc"] = int(match.group(1))
        return result if result else None

    def parse_dlio_output(self, job_id):
        """Parse DLIO stdout to extract training throughput.

        Returns dict with dlio_throughput_samples_s, or None.
        """
        out_pattern = os.path.join(self.results_dir, f"*_{job_id}.out")
        out_files = glob.glob(out_pattern)
        if not out_files:
            return None

        import re
        result = {}
        with open(out_files[0]) as f:
            for line in f:
                # DLIO prints throughput as "Throughput: X samples/s"
                if "throughput" in line.lower():
                    match = re.search(r"(\d+\.?\d*)\s*(?:samples?/s|it/s)", line.lower())
                    if match:
                        result["dlio_throughput_samples_s"] = float(match.group(1))
                # Also capture training time
                if "training" in line.lower() and "complete" in line.lower():
                    pass  # timing from exit code line
                if "epoch" in line.lower() and "time" in line.lower():
                    match = re.search(r"(\d+\.?\d*)\s*(?:s|sec)", line.lower())
                    if match:
                        result["dlio_epoch_time_s"] = float(match.group(1))
        return result if result else None

    def execute_benchmark(self, benchmark_command, job_name, benchmark_type="ior",
                          hacc_config=None, h5bench_config=None, dlio_config=None):
        """Full execution cycle: generate script -> submit -> wait -> parse.

        Args:
            benchmark_command: IOR/mdtest/HACC-IO/custom command string,
                or (write_cmd, read_cmd) tuple for h5bench,
                or (datagen_cmd, training_cmd) tuple for DLIO
            job_name: unique identifier
            benchmark_type: 'ior', 'mdtest', 'hacc_io', 'custom', 'h5bench', or 'dlio'
            hacc_config: dict with collective_buffering key (for HACC-IO)
            h5bench_config: dict with COLLECTIVE_DATA key (for h5bench)
            dlio_config: dict with DLIO params (for DLIO)

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

        # Set appropriate timeout per benchmark type
        if benchmark_type == "dlio":
            timeout = 36000  # 10 hours
        elif benchmark_type == "h5bench":
            timeout = 7200   # 2 hours
        else:
            timeout = 7200   # default 2 hours

        # Generate and submit
        script = self.generate_slurm_script(
            job_name, benchmark_command, benchmark_type,
            hacc_config=hacc_config, h5bench_config=h5bench_config,
            dlio_config=dlio_config,
        )
        job_id = self.submit_and_wait(script, timeout_seconds=timeout)

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

        # Also parse benchmark stdout for write BW
        if benchmark_type == "ior":
            ior_out = self.parse_ior_output(job_id)
            if ior_out:
                result["ior_output"] = ior_out
        elif benchmark_type == "hacc_io":
            hacc_out = self.parse_hacc_output(job_id)
            if hacc_out:
                result["ior_output"] = hacc_out  # Reuse field for consistency
        elif benchmark_type == "custom":
            custom_out = self.parse_custom_output(job_id)
            if custom_out:
                result["ior_output"] = custom_out
        elif benchmark_type == "h5bench":
            h5_out = self.parse_h5bench_output(job_id)
            if h5_out:
                result["ior_output"] = h5_out
        elif benchmark_type == "dlio":
            dlio_out = self.parse_dlio_output(job_id)
            if dlio_out:
                result["ior_output"] = dlio_out

        result["elapsed_s"] = time.time() - t0
        return result
