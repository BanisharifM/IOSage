#!/bin/bash
#SBATCH --job-name=iter_ior_random_access_r0_i3
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_ior_random_access_r0_i3_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_ior_random_access_r0_i3_%j.err

module load ior/3.3.0-gcc13.3.1

export DARSHAN_LOGPATH="/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_logs/iterative"
mkdir -p "${DARSHAN_LOGPATH}"

# Fix SLURM env var conflicts on Delta
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK 2>/dev/null

# Per-job scratch to avoid file conflicts between concurrent runs
JOB_SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_ior_random_access_r0_i3"
mkdir -p "$JOB_SCRATCH"

# Cleanup benchmark files on exit
cleanup() { rm -rf "$JOB_SCRATCH" 2>/dev/null || true; }
trap cleanup EXIT

echo "============================================================"
echo "Track C Iterative Optimization - Benchmark Execution"
echo "Job: iter_ior_random_access_r0_i3"
echo "Command: ior -a MPIIO -t 1048576 -b 104857600 -s 10 -F -e -C -w -r -c -F -o /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_ior_random_access_r0_i3/ior_test_file"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

# Run benchmark with Darshan instrumentation
srun --cpu-bind=none --export=ALL,LD_PRELOAD=/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so \
    ior -a MPIIO -t 1048576 -b 104857600 -s 10 -F -e -C -w -r -c -F -o /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_ior_random_access_r0_i3/ior_test_file

EXIT_CODE=$?
echo ""
echo "Exit code: $EXIT_CODE"
echo "Completed: $(date)"

# Report Darshan log location
LOGS=$(ls -t ${DARSHAN_LOGPATH}/*.darshan 2>/dev/null | head -1)
if [ -n "$LOGS" ]; then
    echo "Darshan log: $LOGS"
else
    echo "WARNING: No Darshan log found"
fi

exit $EXIT_CODE
