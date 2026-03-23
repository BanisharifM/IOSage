#!/bin/bash
#SBATCH --job-name=iop_good_ior_interface_posix_shared
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/closed_loop/iop_good_ior_interface_posix_shared_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/closed_loop/iop_good_ior_interface_posix_shared_%j.err

module load ior/3.3.0-gcc13.3.1

export DARSHAN_LOGPATH="/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_logs/ioprescriber"
mkdir -p "${DARSHAN_LOGPATH}"

# Cleanup on exit
cleanup() { rm -f /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/ioprescriber/ior_test_file* 2>/dev/null || true; }
trap cleanup EXIT

echo "============================================================"
echo "IOPrescriber Closed-Loop Validation"
echo "Job: iop_good_ior_interface_posix_shared"
echo "Command: ior -a MPIIO -t 1048576 -b 100M -s 10 -c -w -r -o /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/ioprescriber/ior_test_file"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

mkdir -p /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/ioprescriber

# Run IOR with Darshan instrumentation
srun --cpu-bind=none --export=ALL,LD_PRELOAD=/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so \
    ior -a MPIIO -t 1048576 -b 100M -s 10 -c -w -r -o /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/ioprescriber/ior_test_file

echo ""
echo "Exit code: $?"
echo "Completed: $(date)"

# Check for Darshan log
LOGS=$(ls -t ${DARSHAN_LOGPATH}/*.darshan 2>/dev/null | head -1)
if [ -n "$LOGS" ]; then
    echo "Darshan log: $LOGS"
else
    echo "WARNING: No Darshan log found"
fi
