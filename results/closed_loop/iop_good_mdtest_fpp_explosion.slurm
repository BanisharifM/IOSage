#!/bin/bash
#SBATCH --job-name=good_mdtest_fpp_explosion
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=results/closed_loop/good_mdtest_fpp_explosion_%j.out
#SBATCH --error=results/closed_loop/good_mdtest_fpp_explosion_%j.err

module load ior/3.3.0-gcc13.3.1
export DARSHAN_LOGPATH="/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_logs/ioprescriber"
mkdir -p "${DARSHAN_LOGPATH}"
cleanup() { rm -rf /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/ioprescriber/mdtest_out* 2>/dev/null || true; }
trap cleanup EXIT

echo "Pair: mdtest_fpp_explosion (GOOD)"
echo "Command: mdtest -n 50 -w 4096 -e 4096 -F -u -d /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/ioprescriber/mdtest_out"
echo "Date: $(date)"
echo "Nodes: 1, Tasks: 128"

mkdir -p /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/ioprescriber/mdtest_out

srun --cpu-bind=none --export=ALL,LD_PRELOAD=/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so \
    mdtest -n 50 -w 4096 -e 4096 -F -u -d /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/ioprescriber/mdtest_out

echo "Exit code: $?"
echo "Completed: $(date)"
ls -t ${DARSHAN_LOGPATH}/*.darshan 2>/dev/null | head -1
