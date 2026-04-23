#!/bin/bash
#SBATCH --job-name=iter_custom_load_imbalance_r0_gpt-4o_baseline
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_custom_load_imbalance_r0_gpt-4o_baseline_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_custom_load_imbalance_r0_gpt-4o_baseline_%j.err

export DARSHAN_LOGPATH="/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_logs/iterative"
export DARSHAN_ENABLE_NONMPI=1
export DARSHAN_MODMEM=4
mkdir -p "${DARSHAN_LOGPATH}"

# Fix SLURM env var conflicts on Delta
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null
export SLURM_CPUS_PER_TASK=1

# Per-job scratch to avoid file conflicts between concurrent runs
JOB_SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_custom_load_imbalance_r0_gpt-4o_baseline"
mkdir -p "$JOB_SCRATCH"

# Cleanup on exit
cleanup() { rm -rf "$JOB_SCRATCH" 2>/dev/null || true; }
trap cleanup EXIT

echo "============================================================"
echo "IOSage Iterative Optimization - Custom Benchmark Execution"
echo "Job: iter_custom_load_imbalance_r0_gpt-4o_baseline"
echo "Command: /projects/bdau/envs/sc2026/bin/python /work/hdd/bdau/mbanisharifdehkordi/SC_2026/benchmarks/custom/load_imbalance.py --imbalance-factor 10.0 --base-size-mb 10 --output-dir /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_custom_load_imbalance_r0_gpt-4o_baseline"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

# Run custom benchmark with Darshan instrumentation
srun --cpu-bind=none --export=ALL,LD_PRELOAD=/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so \
    /projects/bdau/envs/sc2026/bin/python /work/hdd/bdau/mbanisharifdehkordi/SC_2026/benchmarks/custom/load_imbalance.py --imbalance-factor 10.0 --base-size-mb 10 --output-dir /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_custom_load_imbalance_r0_gpt-4o_baseline

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
