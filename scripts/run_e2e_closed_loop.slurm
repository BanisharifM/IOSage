#!/bin/bash
#SBATCH --job-name=e2e_closed_loop
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/resubmission/e2e_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/IOSage/results/resubmission/e2e_%j.err
#SBATCH --export=NONE

set -euo pipefail

# =============================================================================
# E2E Closed-Loop Validation: Pathological vs Ultra-Optimized NetCDF-4 Write
# =============================================================================
# Source: Jay Lofstead's E2E benchmark (HPDC 2011), adapted for parallel NetCDF-4
#
# Two configurations:
#   1. Pathological: NC_FILL + 1x1x64 chunk sizes + nc_sync after every variable
#   2. Ultra-optimized: NC_NOFILL + process-matched chunks + independent I/O + no sync
#
# Grid: 8x8x1 process decomposition, 32x32x128 local grid per process
# 10 double-precision 3D variables per run
# Data volume: 10 vars * 8 bytes * 256 * 256 * 128 = ~640 MB per run
#
# Run from project root:
#   sbatch scripts/run_e2e_closed_loop.slurm
# =============================================================================

echo "============================================================"
echo "E2E Closed-Loop Validation"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "============================================================"

# --- Environment ---
PROJECT_DIR="/work/hdd/bdau/mbanisharifdehkordi/IOSage"
E2E_DIR="${PROJECT_DIR}/benchmarks/e2e"
DARSHAN_LIB="/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so"
DARSHAN_LOGPATH="${PROJECT_DIR}/data/benchmark_logs/e2e"
SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/e2e/closed_loop_${SLURM_JOB_ID}"
RESULTS_DIR="${PROJECT_DIR}/results/resubmission/closed_loop/${SLURM_JOB_ID}"

mkdir -p "${DARSHAN_LOGPATH}" "${RESULTS_DIR}" "${SCRATCH}"
source "${E2E_DIR}/modules.sh"
source "${E2E_DIR}/job_lib.sh"
e2e_build
MANIFEST="${RESULTS_DIR}/e2e_closed_loop_${SLURM_JOB_ID}.tsv"
CORRECTNESS_MANIFEST="${RESULTS_DIR}/e2e_closed_loop_correctness_${SLURM_JOB_ID}.tsv"
cleanup() { rm -rf "${SCRATCH}"; }
trap cleanup EXIT
cd "${SCRATCH}"

# --- Darshan environment ---
export DARSHAN_LOGPATH="${DARSHAN_LOGPATH}"
export DXT_ENABLE_IO_TRACE=1
export DARSHAN_DISABLE_SHARED_REDUCTION=1

# Grid parameters
# Process decomposition: 8 x 8 x 1 = 64 processes
NPX=8
NPY=8
NPZ=1
# Local grid per process: 32 x 32 x 128 (proper scale, matching AIIO evaluation)
NDX=32
NDY=32
NDZ=128

# Global grid: 256 x 256 x 128
# Per variable: 256*256*128*8 bytes = 64 MB
# 10 variables: 640 MB total
# Note: Pathological config may take 1-4 hours due to extreme chunk fragmentation.
# This is expected and demonstrates the severity of the bottleneck.

echo ""
echo "Grid: ${NPX}x${NPY}x${NPZ} processes, ${NDX}x${NDY}x${NDZ} local"
echo "Global: $((NPX*NDX))x$((NPY*NDY))x$((NPZ*NDZ))"
echo "Data volume: ~640 MB per run (10 variables, double precision)"
echo ""

# =============================================================================
# Run 1: Pathological configuration
# =============================================================================
echo "============================================================"
echo "Phase 1: PATHOLOGICAL configuration"
echo "  - NC_FILL enabled (double-write penalty)"
echo "  - 1x1x64 chunk sizes (extreme fragmentation)"
echo "  - nc_sync + MPI_Barrier after every variable"
echo "  - Collective I/O"
echo "Start: $(date)"
echo "============================================================"

rm -f "${SCRATCH}/e2e_pathological.nc4"

# Fix SLURM env var conflicts (Delta inherits conflicting vars from parent sessions)
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK SLURM_CPUS_PER_TASK 2>/dev/null
export SLURM_CPUS_PER_TASK=1
export SLURM_CPUS_PER_TASK=1

e2e_run_and_record pathological none "${E2E_DIR}/build/write_3d_nc4_pathological" \
    "${SCRATCH}/e2e_pathological" "$MANIFEST" \
    ${NPX} ${NPY} ${NPZ} ${NDX} ${NDY} ${NDZ}
echo "Pathological finished: $(date)"

echo ""

# Small barrier between runs
sleep 5

# =============================================================================
# Run 2: Ultra-optimized configuration
# =============================================================================
echo "============================================================"
echo "Phase 2: ULTRA-OPTIMIZED configuration"
echo "  - NC_NOFILL (skip fill-value initialization)"
echo "  - Process-matched chunk sizes (${NDX}x${NDY}x${NDZ})"
echo "  - No nc_sync or barriers between writes"
echo "  - Independent I/O (better for aligned access)"
echo "  - 128 MB chunk cache"
echo "Start: $(date)"
echo "============================================================"

rm -f "${SCRATCH}/e2e_ultra_optimized.nc4"

e2e_run_and_record ultra_optimized netcdf_disable_fill_process_chunks_no_sync \
    "${E2E_DIR}/build/write_3d_nc4_ultra_optimized" \
    "${SCRATCH}/e2e_ultra_optimized" "$MANIFEST" \
    ${NPX} ${NPY} ${NPZ} ${NDX} ${NDY} ${NDZ}
echo "Ultra-optimized finished: $(date)"

echo ""

# =============================================================================
# Run 3: Baseline configuration (default NetCDF behavior)
# =============================================================================
echo "============================================================"
echo "Phase 3: BASELINE configuration (default NetCDF-4)"
echo "  - Default fill behavior"
echo "  - Default chunking (library-chosen)"
echo "  - Collective I/O"
echo "  - No explicit sync"
echo "Start: $(date)"
echo "============================================================"

rm -f "${SCRATCH}/e2e_baseline.nc4"

e2e_run_and_record baseline none "${E2E_DIR}/build/write_3d_nc4" \
    "${SCRATCH}/e2e_baseline" "$MANIFEST" \
    ${NPX} ${NPY} ${NPZ} ${NDX} ${NDY} ${NDZ}
echo "Baseline finished: $(date)"

e2e_verify_pair pathological ultra_optimized \
    "${SCRATCH}/e2e_pathological.nc4" "${SCRATCH}/e2e_ultra_optimized.nc4" \
    "$CORRECTNESS_MANIFEST"
e2e_verify_pair baseline ultra_optimized \
    "${SCRATCH}/e2e_baseline.nc4" "${SCRATCH}/e2e_ultra_optimized.nc4" \
    "$CORRECTNESS_MANIFEST"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Output format: filename nprocs ndx ndy ndz size_GB time_sec throughput_GBps"
echo ""
echo "Run manifest: ${MANIFEST}"
[[ $(wc -l < "$MANIFEST") -eq 4 ]] || { echo "closed-loop manifest is incomplete" >&2; exit 92; }
[[ $(wc -l < "$CORRECTNESS_MANIFEST") -eq 3 ]] || { echo "correctness manifest is incomplete" >&2; exit 92; }
cat "$MANIFEST"
cat "$CORRECTNESS_MANIFEST"
echo ""
echo "Job completed: $(date)"
echo "============================================================"
