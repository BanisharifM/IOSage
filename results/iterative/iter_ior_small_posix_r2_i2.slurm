#!/bin/bash
#SBATCH --job-name=iter_ior_small_posix_r2_i2
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:10:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_ior_small_posix_r2_i2_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_ior_small_posix_r2_i2_%j.err

module load ior/3.3.0-gcc13.3.1

export DARSHAN_LOGPATH="/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_logs/iterative"
mkdir -p "${DARSHAN_LOGPATH}"

# Cleanup benchmark files on exit
cleanup() { rm -f /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/ior_test_file* /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/mdtest_dir* 2>/dev/null || true; }
trap cleanup EXIT

echo "============================================================"
echo "Track C Iterative Optimization - Benchmark Execution"
echo "Job: iter_ior_small_posix_r2_i2"
echo "Command: ior -a MPIIO -t 1048576 -b 1048576 -s 100 -F -e -C -w -r -o /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/ior_test_file"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

mkdir -p /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative

# Run benchmark with Darshan instrumentation
srun --cpu-bind=none --export=ALL,LD_PRELOAD=/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so \
    ior -a MPIIO -t 1048576 -b 1048576 -s 100 -F -e -C -w -r -o /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/ior_test_file

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
