#!/bin/bash
#SBATCH --job-name=iter_h5bench_indep_vs_coll_r0_llama_i2
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_h5bench_indep_vs_coll_r0_llama_i2_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_h5bench_indep_vs_coll_r0_llama_i2_%j.err

module load PrgEnv-gnu/8.6.0 cray-hdf5-parallel/1.14.3.5 2>/dev/null || true

export DARSHAN_LOGPATH="/work/hdd/bdau/mbanisharifdehkordi/SC_2026/data/benchmark_logs/iterative"
mkdir -p "${DARSHAN_LOGPATH}"

# HDF5 parallel library path
export LD_LIBRARY_PATH="/opt/cray/pe/hdf5-parallel/1.14.3.5/gnu/12.2/lib:${LD_LIBRARY_PATH:-}"

# ROMIO collective buffering control
export MPICH_MPIIO_HINTS="*:romio_cb_write=enable:romio_ds_write=disable"
export MPICH_MPIIO_HINTS_DISPLAY=1

# Fix SLURM env var conflicts on Delta
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null
export SLURM_CPUS_PER_TASK=1

# Per-job scratch to avoid file conflicts between concurrent runs
JOB_SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_h5bench_indep_vs_coll_r0_llama_i2"
mkdir -p "$JOB_SCRATCH"

# Cleanup h5bench data files on exit
cleanup() { rm -f "$JOB_SCRATCH"/h5bench_output.h5* "$JOB_SCRATCH"/output.csv 2>/dev/null; rm -rf "$JOB_SCRATCH" 2>/dev/null || true; }
trap cleanup EXIT

echo "============================================================"
echo "IOSage Iterative Optimization - h5bench Execution"
echo "Job: iter_h5bench_indep_vs_coll_r0_llama_i2"
echo "Collective data: YES"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

# Pre-flight checks
if [ -x "/work/hdd/bdau/mbanisharifdehkordi/h5bench/build/h5bench_write" ]; then
    echo "  h5bench_write: OK"
else
    echo "  ERROR: h5bench_write not found"
    exit 1
fi

# === h5bench WRITE phase ===
echo ""
echo "=== h5bench WRITE phase ==="
echo "Command: /work/hdd/bdau/mbanisharifdehkordi/h5bench/build/h5bench_write /work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_h5bench_indep_vs_coll_r0_llama_i2_config.json.write /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_h5bench_indep_vs_coll_r0_llama_i2/h5bench_output.h5"
srun --cpu-bind=none --export=ALL,LD_PRELOAD=/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so \
    /work/hdd/bdau/mbanisharifdehkordi/h5bench/build/h5bench_write /work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_h5bench_indep_vs_coll_r0_llama_i2_config.json.write /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_h5bench_indep_vs_coll_r0_llama_i2/h5bench_output.h5

WRITE_RC=$?
echo ""
echo "Write completed at $(date), exit code: ${WRITE_RC}"

# === h5bench READ phase ===
H5_FILE=$(echo "/work/hdd/bdau/mbanisharifdehkordi/h5bench/build/h5bench_read /work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_h5bench_indep_vs_coll_r0_llama_i2_config.json.read /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_h5bench_indep_vs_coll_r0_llama_i2/h5bench_output.h5" | awk '{print $NF}')
if [ -f "$H5_FILE" ]; then
    echo ""
    echo "=== h5bench READ phase ==="
    echo "Command: /work/hdd/bdau/mbanisharifdehkordi/h5bench/build/h5bench_read /work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_h5bench_indep_vs_coll_r0_llama_i2_config.json.read /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_h5bench_indep_vs_coll_r0_llama_i2/h5bench_output.h5"
    srun --cpu-bind=none --export=ALL,LD_PRELOAD=/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so \
        /work/hdd/bdau/mbanisharifdehkordi/h5bench/build/h5bench_read /work/hdd/bdau/mbanisharifdehkordi/SC_2026/results/iterative/iter_h5bench_indep_vs_coll_r0_llama_i2_config.json.read /work/hdd/bdau/mbanisharifdehkordi/bench_scratch/iterative/iter_h5bench_indep_vs_coll_r0_llama_i2/h5bench_output.h5

    READ_RC=$?
    echo ""
    echo "Read completed at $(date), exit code: ${READ_RC}"
else
    echo "WARNING: HDF5 output file not found, skipping read phase"
fi

echo ""
echo "Exit code (write): $WRITE_RC"
echo "Completed: $(date)"

# Report Darshan log location
LOGS=$(ls -t ${DARSHAN_LOGPATH}/*id${SLURM_JOB_ID}* 2>/dev/null | head -1)
if [ -n "$LOGS" ]; then
    echo "Darshan log: $LOGS"
else
    echo "WARNING: No Darshan log found"
fi

exit $WRITE_RC
