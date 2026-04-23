#!/bin/bash
#SBATCH --job-name=iter_hacc_fpp_small_r0_llama_baseline
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_hacc_fpp_small_r0_llama_baseline_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_hacc_fpp_small_r0_llama_baseline_%j.err

module load PrgEnv-gnu/8.6.0 2>/dev/null || true

export DARSHAN_LOGPATH="/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_logs/iterative"
mkdir -p "${DARSHAN_LOGPATH}"

# ROMIO collective buffering control
export MPICH_MPIIO_HINTS="*:romio_cb_write=disable:romio_cb_read=disable:romio_ds_write=disable:romio_ds_read=disable"
export MPICH_MPIIO_HINTS_DISPLAY=1

# Fix SLURM env var conflicts on Delta
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null
export SLURM_CPUS_PER_TASK=1

# Per-job scratch to avoid file conflicts between concurrent runs
JOB_SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_hacc_fpp_small_r0_llama_baseline"
mkdir -p "$JOB_SCRATCH"

# Cleanup HACC-IO checkpoint files on exit
cleanup() { rm -f "$JOB_SCRATCH"/hacc_checkpoint* 2>/dev/null; rm -rf "$JOB_SCRATCH" 2>/dev/null || true; }
trap cleanup EXIT

echo "============================================================"
echo "IOSage Iterative Optimization - HACC-IO Execution"
echo "Job: iter_hacc_fpp_small_r0_llama_baseline"
echo "Command: /work/hdd/bdau/mbanisharifdehkordi/hacc-io/hacc_io_fpp 50 /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_hacc_fpp_small_r0_llama_baseline/hacc_checkpoint"
echo "Collective buffering: disabled"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

# Run HACC-IO with Darshan instrumentation
srun --cpu-bind=none --export=ALL,LD_PRELOAD=/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so \
    /work/hdd/bdau/mbanisharifdehkordi/hacc-io/hacc_io_fpp 50 /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_hacc_fpp_small_r0_llama_baseline/hacc_checkpoint

EXIT_CODE=$?
echo ""
echo "Exit code: $EXIT_CODE"
echo "Completed: $(date)"

# Cleanup checkpoint data files
rm -f "$JOB_SCRATCH"/hacc_checkpoint* 2>/dev/null || true

# Report Darshan log location
LOGS=$(ls -t ${DARSHAN_LOGPATH}/*id${SLURM_JOB_ID}* 2>/dev/null | head -1)
if [ -n "$LOGS" ]; then
    echo "Darshan log: $LOGS"
else
    echo "WARNING: No Darshan log found"
fi

exit $EXIT_CODE
