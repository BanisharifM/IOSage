#!/bin/bash
#SBATCH --job-name=e2e_baseline
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=results/closed_loop/e2e_baseline_%j.out
#SBATCH --error=results/closed_loop/e2e_baseline_%j.err

module load gcc-native/13.2 cray-mpich/8.1.32 cray-hdf5-parallel/1.14.3.5 cray-netcdf-hdf5parallel/4.9.0.17

unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null
export SLURM_CPUS_PER_TASK=1
export PMI_NO_PREINITIALIZE=1
export PMI_NO_FORK=1

PROJECT_DIR="/work/hdd/bdau/mbanisharifdehkordi/SC_2026"
E2E_DIR="${PROJECT_DIR}/benchmarks/e2e"
DARSHAN_LIB="/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so"
export DARSHAN_LOGPATH="${PROJECT_DIR}/data/benchmark_logs/e2e"
SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/e2e/baseline"

mkdir -p "${DARSHAN_LOGPATH}" "${SCRATCH}"
cd "${SCRATCH}"

echo "E2E BASELINE (default NetCDF-4) - $(date)"
echo "Grid: 8x8x1, local 32x32x128, global 256x256x128"
echo "Host: $(hostname), Job: $SLURM_JOB_ID"

rm -f "${SCRATCH}/e2e_baseline.nc4"

srun --cpu-bind=none --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \
    "${E2E_DIR}/write_3d_nc4" \
    "${SCRATCH}/e2e_baseline" \
    8 8 1 32 32 128

echo "Exit code: $?"
echo "Completed: $(date)"
# Find THIS job's Darshan log (not just the latest in the directory)
ls -la ${DARSHAN_LOGPATH}/*id${SLURM_JOB_ID}* 2>/dev/null || echo "WARNING: No Darshan log found for job $SLURM_JOB_ID"
