#!/bin/bash
#SBATCH --job-name=good_ior_collective_vs_independent
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=results/closed_loop/good_ior_collective_vs_independent_%j.out
#SBATCH --error=results/closed_loop/good_ior_collective_vs_independent_%j.err

module load ior/3.3.0-gcc13.3.1
export DARSHAN_LOGPATH="/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_logs/ioprescriber"
mkdir -p "${DARSHAN_LOGPATH}"
cleanup() { rm -f /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/ioprescriber/ior_test_file* 2>/dev/null || true; }
trap cleanup EXIT

echo "Pair: ior_collective_vs_independent (GOOD)"
echo "Command: ior -a MPIIO -t 1048576 -b 100M -s 10 -c -w -r -o /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/ioprescriber/ior_test_file"
echo "Date: $(date)"
echo "Nodes: 4, Tasks: 64"

mkdir -p /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/ioprescriber

srun --cpu-bind=none --export=ALL,LD_PRELOAD=/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so \
    ior -a MPIIO -t 1048576 -b 100M -s 10 -c -w -r -o /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/ioprescriber/ior_test_file

echo "Exit code: $?"
echo "Completed: $(date)"
ls -t ${DARSHAN_LOGPATH}/*.darshan 2>/dev/null | head -1
