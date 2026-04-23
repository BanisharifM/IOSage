#!/bin/bash
#SBATCH --job-name=iter_ior_fsync_heavy_r0_gpt-4.1-mini_baseline
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_ior_fsync_heavy_r0_gpt-4.1-mini_baseline_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_ior_fsync_heavy_r0_gpt-4.1-mini_baseline_%j.err

module load ior/3.3.0-gcc13.3.1

export DARSHAN_LOGPATH="/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_logs/iterative"
mkdir -p "${DARSHAN_LOGPATH}"

# Fix SLURM env var conflicts on Delta
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null
export SLURM_CPUS_PER_TASK=1

# Per-job scratch to avoid file conflicts between concurrent runs
JOB_SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_ior_fsync_heavy_r0_gpt-4.1-mini_baseline"
mkdir -p "$JOB_SCRATCH"

# Cleanup benchmark files on exit
cleanup() { rm -rf "$JOB_SCRATCH" 2>/dev/null || true; }
trap cleanup EXIT

echo "============================================================"
echo "IOSage Iterative Optimization - Benchmark Execution"
echo "Job: iter_ior_fsync_heavy_r0_gpt-4.1-mini_baseline"
echo "Command: ior -a POSIX -t 65536 -b 104857600 -s 10 -F -e -C -Y -w -r -o /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_ior_fsync_heavy_r0_gpt-4.1-mini_baseline/ior_test_file"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

# Run benchmark with Darshan instrumentation
srun --cpu-bind=none --export=ALL,LD_PRELOAD=/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so \
    ior -a POSIX -t 65536 -b 104857600 -s 10 -F -e -C -Y -w -r -o /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_ior_fsync_heavy_r0_gpt-4.1-mini_baseline/ior_test_file

EXIT_CODE=$?
echo ""
echo "Exit code: $EXIT_CODE"
echo "Completed: $(date)"

# Report Darshan log location
LOGS=$(ls -t ${DARSHAN_LOGPATH}/*id${SLURM_JOB_ID}* 2>/dev/null | head -1)
if [ -n "$LOGS" ]; then
    echo "Darshan log: $LOGS"
else
    echo "WARNING: No Darshan log found"
fi

exit $EXIT_CODE
