#!/bin/bash
#SBATCH --job-name=e2e_optimized
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/resubmission/e2e_optimized_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/resubmission/e2e_optimized_%j.err
#SBATCH --export=NONE

set -euo pipefail

unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null
export SLURM_CPUS_PER_TASK=1
export PMI_NO_PREINITIALIZE=1
export PMI_NO_FORK=1

PROJECT_DIR="/work/hdd/bdau/mbanisharifdehkordi/IOSage"
E2E_DIR="${PROJECT_DIR}/benchmarks/e2e"
DARSHAN_LIB="/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so"
export DARSHAN_LOGPATH="${PROJECT_DIR}/data/benchmark_logs/e2e"
SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/e2e/optimized_${SLURM_JOB_ID}"
RESULTS_DIR="${PROJECT_DIR}/results/resubmission/closed_loop/${SLURM_JOB_ID}"

mkdir -p "${DARSHAN_LOGPATH}" "${SCRATCH}" "${RESULTS_DIR}"
source "${E2E_DIR}/modules.sh"
source "${E2E_DIR}/job_lib.sh"
e2e_build
BIN="${E2E_DIR}/build/write_3d_nc4_ultra_optimized"
MANIFEST="${RESULTS_DIR}/e2e_optimized_${SLURM_JOB_ID}.tsv"
cleanup() { rm -rf "${SCRATCH}"; }
trap cleanup EXIT
cd "${SCRATCH}"

echo "E2E ULTRA-OPTIMIZED - $(date)"
echo "Grid: 8x8x1, local 32x32x128, global 256x256x128"
echo "Host: $(hostname), Job: $SLURM_JOB_ID"

rm -f "${SCRATCH}/e2e_optimized.nc4"

e2e_run_and_record ultra_optimized netcdf_disable_fill_process_chunks_no_sync \
    "$BIN" "${SCRATCH}/e2e_optimized" "$MANIFEST" 8 8 1 32 32 128
[[ $(wc -l < "$MANIFEST") -eq 2 ]] || { echo "optimized manifest is incomplete" >&2; exit 92; }
echo "Completed: $(date)"
cat "$MANIFEST"
