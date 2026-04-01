#!/bin/bash
#SBATCH --job-name=e2e_closed_loop
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=results/closed_loop/e2e_%j.out
#SBATCH --error=results/closed_loop/e2e_%j.err

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
module load gcc-native/13.2 cray-mpich/8.1.32 cray-hdf5-parallel/1.14.3.5 cray-netcdf-hdf5parallel/4.9.0.17

PROJECT_DIR="/work/hdd/bdau/mbanisharifdehkordi/SC_2026"
E2E_DIR="${PROJECT_DIR}/benchmarks/e2e"
DARSHAN_LIB="/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so"
DARSHAN_LOGPATH="${PROJECT_DIR}/data/benchmark_logs/e2e"
SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch/e2e"
RESULTS_DIR="${PROJECT_DIR}/results/closed_loop"

mkdir -p "${DARSHAN_LOGPATH}" "${RESULTS_DIR}" "${SCRATCH}"
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

srun --cpu-bind=none --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \
    "${E2E_DIR}/write_3d_nc4_pathological" \
    "${SCRATCH}/e2e_pathological" \
    ${NPX} ${NPY} ${NPZ} ${NDX} ${NDY} ${NDZ}

PATHOLOGICAL_EXIT=$?
echo "Pathological exit code: ${PATHOLOGICAL_EXIT}"
echo "Pathological finished: $(date)"

# Capture Darshan log
PATHO_LOG=$(ls -t ${DARSHAN_LOGPATH}/*.darshan 2>/dev/null | head -1)
if [ -n "${PATHO_LOG}" ]; then
    echo "Pathological Darshan log: ${PATHO_LOG}"
    cp "${PATHO_LOG}" "${DARSHAN_LOGPATH}/e2e_pathological_${SLURM_JOB_ID}.darshan" 2>/dev/null
else
    echo "WARNING: No Darshan log found for pathological run"
fi

# Clean up data file
rm -f "${SCRATCH}/e2e_pathological.nc4"

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

srun --cpu-bind=none --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \
    "${E2E_DIR}/write_3d_nc4_ultra_optimized" \
    "${SCRATCH}/e2e_ultra_optimized" \
    ${NPX} ${NPY} ${NPZ} ${NDX} ${NDY} ${NDZ}

OPTIMIZED_EXIT=$?
echo "Ultra-optimized exit code: ${OPTIMIZED_EXIT}"
echo "Ultra-optimized finished: $(date)"

# Capture Darshan log
OPT_LOG=$(ls -t ${DARSHAN_LOGPATH}/*.darshan 2>/dev/null | head -1)
if [ -n "${OPT_LOG}" ]; then
    echo "Ultra-optimized Darshan log: ${OPT_LOG}"
    cp "${OPT_LOG}" "${DARSHAN_LOGPATH}/e2e_ultra_optimized_${SLURM_JOB_ID}.darshan" 2>/dev/null
else
    echo "WARNING: No Darshan log found for ultra-optimized run"
fi

# Clean up data file
rm -f "${SCRATCH}/e2e_ultra_optimized.nc4"

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

srun --cpu-bind=none --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \
    "${E2E_DIR}/write_3d_nc4" \
    "${SCRATCH}/e2e_baseline" \
    ${NPX} ${NPY} ${NPZ} ${NDX} ${NDY} ${NDZ}

BASELINE_EXIT=$?
echo "Baseline exit code: ${BASELINE_EXIT}"
echo "Baseline finished: $(date)"

# Capture Darshan log
BASE_LOG=$(ls -t ${DARSHAN_LOGPATH}/*.darshan 2>/dev/null | head -1)
if [ -n "${BASE_LOG}" ]; then
    echo "Baseline Darshan log: ${BASE_LOG}"
    cp "${BASE_LOG}" "${DARSHAN_LOGPATH}/e2e_baseline_${SLURM_JOB_ID}.darshan" 2>/dev/null
else
    echo "WARNING: No Darshan log found for baseline run"
fi

# Clean up data file
rm -f "${SCRATCH}/e2e_baseline.nc4"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Output format: filename nprocs ndx ndy ndz size_GB time_sec throughput_GBps"
echo ""
echo "Darshan logs saved to: ${DARSHAN_LOGPATH}"
ls -la ${DARSHAN_LOGPATH}/e2e_*_${SLURM_JOB_ID}.darshan 2>/dev/null
echo ""
echo "Job completed: $(date)"
echo "============================================================"
