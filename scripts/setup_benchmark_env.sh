#!/bin/bash
# =============================================================================
# Setup Benchmark Environment on Delta
# =============================================================================
# Run ONCE before submitting benchmark jobs.
# This script:
#   1. Verifies Delta environment (modules, Darshan, Lustre)
#   2. Creates benchmark directories with controlled Lustre striping
#   3. Installs mpi4py from source (required for cray-mpich compatibility)
#   4. Installs DLIO benchmark
#   5. Validates the complete setup
# =============================================================================
set -euo pipefail

# --- Configuration ---
PROJECT_DIR="/work/hdd/bdau/mbanisharifdehkordi/SC_2026"
BENCH_SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch"
DARSHAN_LIB="/sw/spack/deltacpu-2022-03/apps/darshan-runtime/3.3.1-gcc-11.2.0-7tis4xp/lib/libdarshan.so"
DARSHAN_PARSER="/sw/spack/deltacpu-2022-03/apps/darshan-util/3.3.1-gcc-11.2.0-vq4wq2e/bin/darshan-parser"
PYTHON_BIN="/projects/bdau/envs/sc2026/bin/python"
PIP_BIN="/projects/bdau/envs/sc2026/bin/pip"
LOG_DIR="${PROJECT_DIR}/data/benchmark_logs"
RESULTS_DIR="${PROJECT_DIR}/data/benchmark_results"

echo "============================================================"
echo "Phase 2: Benchmark Environment Setup"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"

# --- Step 1: Verify modules ---
echo ""
echo "[1/6] Verifying modules..."
module load ior/3.3.0-gcc13.3.1 2>/dev/null || { echo "ERROR: Cannot load ior module"; exit 1; }
IOR_PATH=$(which ior 2>/dev/null) || { echo "ERROR: ior not found after module load"; exit 1; }
MDTEST_PATH=$(which mdtest 2>/dev/null) || { echo "ERROR: mdtest not found after module load"; exit 1; }
echo "  IOR:    ${IOR_PATH}"
echo "  mdtest: ${MDTEST_PATH}"

# Verify cray-mpich
if ! module list 2>&1 | grep -q "cray-mpich"; then
    echo "ERROR: cray-mpich not loaded. Load PrgEnv-gnu first."
    exit 1
fi
echo "  cray-mpich: loaded"

# --- Step 2: Verify Darshan ---
echo ""
echo "[2/6] Verifying Darshan..."
if [ ! -f "${DARSHAN_LIB}" ]; then
    echo "ERROR: Darshan runtime not found at ${DARSHAN_LIB}"
    echo "  Search for it: find /sw -name 'libdarshan.so' 2>/dev/null"
    exit 1
fi
echo "  Runtime: ${DARSHAN_LIB}"

if [ ! -f "${DARSHAN_PARSER}" ]; then
    echo "ERROR: darshan-parser not found at ${DARSHAN_PARSER}"
    exit 1
fi
echo "  Parser:  ${DARSHAN_PARSER}"

# Verify Darshan lib is compatible (64-bit, not stripped)
file "${DARSHAN_LIB}" | grep -q "ELF 64-bit" || { echo "ERROR: Darshan lib is not 64-bit ELF"; exit 1; }
echo "  Darshan 3.3.1 verified OK"

# --- Step 3: Create benchmark directories with controlled Lustre striping ---
echo ""
echo "[3/6] Creating benchmark directories with Lustre stripe overrides..."

# Log directories
mkdir -p "${LOG_DIR}"/{ior,mdtest,dlio,custom}
mkdir -p "${RESULTS_DIR}"/{ior,mdtest,dlio,custom}
echo "  Log dir:     ${LOG_DIR}"
echo "  Results dir: ${RESULTS_DIR}"

# Scratch directories with controlled striping (overrides PFL)
mkdir -p "${BENCH_SCRATCH}"

# Bottleneck directory: single OST (limited bandwidth by design)
BOTTLENECK_DIR="${BENCH_SCRATCH}/bottleneck"
mkdir -p "${BOTTLENECK_DIR}"
lfs setstripe -c 1 -S 1M -p ddn_hdd "${BOTTLENECK_DIR}" 2>/dev/null || echo "  WARN: lfs setstripe failed for bottleneck dir (may need compute node)"
echo "  Bottleneck dir: ${BOTTLENECK_DIR} (stripe_count=1, pool=ddn_hdd)"

# Healthy directory: all 12 HDD OSTs (maximum bandwidth)
HEALTHY_DIR="${BENCH_SCRATCH}/healthy"
mkdir -p "${HEALTHY_DIR}"
lfs setstripe -c -1 -S 1M -p ddn_hdd "${HEALTHY_DIR}" 2>/dev/null || echo "  WARN: lfs setstripe failed for healthy dir (may need compute node)"
echo "  Healthy dir:    ${HEALTHY_DIR} (stripe_count=-1, all 12 HDD OSTs)"

# Medium directory: 4 OSTs
MEDIUM_DIR="${BENCH_SCRATCH}/medium"
mkdir -p "${MEDIUM_DIR}"
lfs setstripe -c 4 -S 1M -p ddn_hdd "${MEDIUM_DIR}" 2>/dev/null || echo "  WARN: lfs setstripe failed for medium dir (may need compute node)"
echo "  Medium dir:     ${MEDIUM_DIR} (stripe_count=4)"

# mdtest scratch (will create many small files)
MDTEST_DIR="${BENCH_SCRATCH}/mdtest"
mkdir -p "${MDTEST_DIR}"
lfs setstripe -c 1 -S 1M "${MDTEST_DIR}" 2>/dev/null || true
echo "  mdtest dir:     ${MDTEST_DIR} (stripe_count=1)"

# DLIO data directory
DLIO_DIR="${BENCH_SCRATCH}/dlio"
mkdir -p "${DLIO_DIR}"
echo "  DLIO dir:       ${DLIO_DIR}"

# Verify striping
echo ""
echo "  Verifying stripe settings..."
for dir in "${BOTTLENECK_DIR}" "${HEALTHY_DIR}" "${MEDIUM_DIR}"; do
    STRIPE_COUNT=$(lfs getstripe -c "${dir}" 2>/dev/null || echo "unknown")
    echo "    ${dir}: stripe_count=${STRIPE_COUNT}"
done

# --- Step 4: Install mpi4py from source ---
echo ""
echo "[4/6] Installing mpi4py from source (cray-mpich compatibility)..."

# Check if mpi4py already works
if ${PYTHON_BIN} -c "from mpi4py import MPI; print('mpi4py OK:', MPI.Get_library_version()[:50])" 2>/dev/null; then
    echo "  mpi4py already installed and working"
else
    echo "  mpi4py not found or incompatible, building from source..."
    # Cray compiler wrappers for MPI
    export MPICC=cc
    export MPICXX=CC
    ${PIP_BIN} install --no-binary mpi4py mpi4py 2>&1 | tail -5
    # Verify
    if ${PYTHON_BIN} -c "from mpi4py import MPI" 2>/dev/null; then
        echo "  mpi4py installed successfully"
    else
        echo "  WARNING: mpi4py installation may have failed."
        echo "  Try manually: MPICC=cc ${PIP_BIN} install --no-binary mpi4py mpi4py"
    fi
fi

# --- Step 5: Install DLIO benchmark ---
echo ""
echo "[5/6] Installing DLIO benchmark..."

if ${PYTHON_BIN} -c "import dlio_benchmark" 2>/dev/null; then
    echo "  DLIO already installed"
else
    echo "  Installing DLIO benchmark..."
    ${PIP_BIN} install dlio-benchmark 2>&1 | tail -5
    if ${PYTHON_BIN} -c "import dlio_benchmark" 2>/dev/null; then
        echo "  DLIO installed successfully"
    else
        echo "  WARNING: DLIO installation may have failed."
        echo "  DLIO benchmarks will be skipped. IOR/mdtest are sufficient for Phase 1."
    fi
fi

# --- Step 6: Validation Summary ---
echo ""
echo "[6/6] Environment Validation Summary"
echo "============================================================"
echo ""

# Check everything
ERRORS=0

echo "Component              Status"
echo "---------------------  ------"

# IOR
if which ior >/dev/null 2>&1; then
    echo "IOR 3.3.0              OK"
else
    echo "IOR 3.3.0              FAIL"
    ERRORS=$((ERRORS + 1))
fi

# mdtest
if which mdtest >/dev/null 2>&1; then
    echo "mdtest                 OK"
else
    echo "mdtest                 FAIL"
    ERRORS=$((ERRORS + 1))
fi

# Darshan runtime
if [ -f "${DARSHAN_LIB}" ]; then
    echo "Darshan runtime        OK"
else
    echo "Darshan runtime        FAIL"
    ERRORS=$((ERRORS + 1))
fi

# Darshan parser
if [ -f "${DARSHAN_PARSER}" ]; then
    echo "Darshan parser         OK"
else
    echo "Darshan parser         FAIL"
    ERRORS=$((ERRORS + 1))
fi

# Python
if [ -f "${PYTHON_BIN}" ]; then
    echo "Python 3.9             OK"
else
    echo "Python 3.9             FAIL"
    ERRORS=$((ERRORS + 1))
fi

# mpi4py
if ${PYTHON_BIN} -c "from mpi4py import MPI" 2>/dev/null; then
    echo "mpi4py                 OK"
else
    echo "mpi4py                 WARN (needed for DLIO/custom only)"
fi

# DLIO
if ${PYTHON_BIN} -c "import dlio_benchmark" 2>/dev/null; then
    echo "DLIO benchmark         OK"
else
    echo "DLIO benchmark         WARN (can install later)"
fi

# Lustre
if lfs getstripe "${BOTTLENECK_DIR}" >/dev/null 2>&1; then
    echo "Lustre striping        OK"
else
    echo "Lustre striping        WARN (lfs may need compute node)"
fi

# srun
if which srun >/dev/null 2>&1; then
    echo "SLURM srun             OK"
else
    echo "SLURM srun             FAIL"
    ERRORS=$((ERRORS + 1))
fi

echo ""
if [ ${ERRORS} -eq 0 ]; then
    echo "All critical components OK. Ready for benchmarking."
else
    echo "ERROR: ${ERRORS} critical component(s) missing. Fix before proceeding."
    exit 1
fi

echo ""
echo "Next steps:"
echo "  1. Run smoke test:  bash benchmarks/smoke_test.sh"
echo "  2. Submit IOR sweep: sbatch benchmarks/ior/run_ior_sweep.sh"
echo "  3. Monitor: squeue -u \$USER"
echo ""
echo "Setup complete at $(date)"
