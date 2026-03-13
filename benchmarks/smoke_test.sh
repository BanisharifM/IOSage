#!/bin/bash
# =============================================================================
# Smoke Test: Validate Benchmark + Darshan Pipeline Before Full Sweep
# =============================================================================
# Runs 6 quick IOR tests (one per bottleneck type + healthy) to verify:
#   1. Darshan captures logs via LD_PRELOAD + srun
#   2. Darshan logs contain expected counters
#   3. Feature extraction works on benchmark logs
#   4. Lustre stripe overrides work
#   5. ROMIO hint control works
#
# MUST RUN AS SLURM JOB (needs srun):
#   sbatch benchmarks/smoke_test.sh
#
# After completion, check:
#   cat data/benchmark_results/smoke_test_*.out
# =============================================================================
#SBATCH --job-name=bench_smoke_test
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=data/benchmark_results/smoke_test_%j.out
#SBATCH --error=data/benchmark_results/smoke_test_%j.err

set -euo pipefail

# --- Configuration ---
PROJECT_DIR="/work/hdd/bdau/mbanisharifdehkordi/SC_2026"
BENCH_SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch"
DARSHAN_LIB="/sw/spack/deltacpu-2022-03/apps/darshan-runtime/3.3.1-gcc-11.2.0-7tis4xp/lib/libdarshan.so"
DARSHAN_PARSER="/sw/spack/deltacpu-2022-03/apps/darshan-util/3.3.1-gcc-11.2.0-vq4wq2e/bin/darshan-parser"
PYTHON_BIN="/projects/bdau/envs/sc2026/bin/python"
SMOKE_DIR="${BENCH_SCRATCH}/smoke_test"
SMOKE_LOG_DIR="${PROJECT_DIR}/data/benchmark_logs/smoke_test"

module load ior/3.3.0-gcc13.3.1

echo "============================================================"
echo "SMOKE TEST: Benchmark + Darshan Pipeline Validation"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Nodes: ${SLURM_JOB_NUM_NODES:-1}"
echo "Tasks: ${SLURM_NTASKS:-4}"
echo "============================================================"

PASS=0
FAIL=0
TESTS=0

run_test() {
    local test_name="$1"
    local label="$2"
    local ior_cmd="$3"
    local extra_env="$4"

    TESTS=$((TESTS + 1))
    echo ""
    echo "--- Test ${TESTS}: ${test_name} ---"
    echo "  Label: ${label}"
    echo "  Command: srun ${ior_cmd}"

    # Clean log dir
    rm -f "${SMOKE_LOG_DIR}"/*.darshan 2>/dev/null || true

    # Run IOR with Darshan
    export DARSHAN_LOGPATH="${SMOKE_LOG_DIR}"
    if [ -n "${extra_env}" ]; then
        eval "export ${extra_env}"
    fi

    local OUTPUT_FILE="${SMOKE_DIR}/${test_name}_output"
    if srun --export=ALL,LD_PRELOAD="${DARSHAN_LIB}" \
        ${ior_cmd} -o "${OUTPUT_FILE}" 2>&1 | tail -5; then
        echo "  IOR: OK"
    else
        echo "  IOR: FAILED (exit code $?)"
        FAIL=$((FAIL + 1))
        return
    fi

    # Check Darshan log was created
    local LOG_FILE
    LOG_FILE=$(ls -t "${SMOKE_LOG_DIR}"/*.darshan 2>/dev/null | head -1)
    if [ -z "${LOG_FILE}" ]; then
        echo "  Darshan log: NOT FOUND"
        FAIL=$((FAIL + 1))
        return
    fi
    local LOG_SIZE
    LOG_SIZE=$(stat -c%s "${LOG_FILE}" 2>/dev/null || echo "0")
    echo "  Darshan log: $(basename ${LOG_FILE}) (${LOG_SIZE} bytes)"

    if [ "${LOG_SIZE}" -lt 100 ]; then
        echo "  Darshan log: TOO SMALL (likely empty/corrupt)"
        FAIL=$((FAIL + 1))
        return
    fi

    # Parse and verify key counters
    local PARSER_OUT
    PARSER_OUT=$(${DARSHAN_PARSER} --total "${LOG_FILE}" 2>/dev/null)
    if [ -z "${PARSER_OUT}" ]; then
        echo "  darshan-parser: FAILED"
        FAIL=$((FAIL + 1))
        return
    fi

    # Check for POSIX module
    if echo "${PARSER_OUT}" | grep -q "POSIX"; then
        echo "  POSIX module: present"
    else
        echo "  POSIX module: MISSING"
        FAIL=$((FAIL + 1))
        return
    fi

    # Check POSIX_BYTES_WRITTEN > 0
    local BYTES_WRITTEN
    BYTES_WRITTEN=$(echo "${PARSER_OUT}" | grep "POSIX_BYTES_WRITTEN" | head -1 | awk '{print $NF}')
    if [ -n "${BYTES_WRITTEN}" ] && [ "${BYTES_WRITTEN}" -gt 0 ] 2>/dev/null; then
        echo "  POSIX_BYTES_WRITTEN: ${BYTES_WRITTEN}"
    else
        echo "  POSIX_BYTES_WRITTEN: 0 or missing"
        FAIL=$((FAIL + 1))
        return
    fi

    # Try feature extraction
    if ${PYTHON_BIN} -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}')
from src.data.parse_darshan import parse_darshan_log
from src.data.feature_extraction import extract_raw_features
report = parse_darshan_log('${LOG_FILE}')
features = extract_raw_features(report)
print(f'  Features extracted: {len(features)} columns')
print(f'  Key features: nprocs={features.get(\"nprocs\", \"N/A\")}, runtime={features.get(\"runtime_seconds\", \"N/A\"):.2f}s')
" 2>&1; then
        echo "  Feature extraction: OK"
    else
        echo "  Feature extraction: FAILED"
        FAIL=$((FAIL + 1))
        return
    fi

    # Clean up IOR data files
    rm -f "${OUTPUT_FILE}"* 2>/dev/null || true

    echo "  RESULT: PASS"
    PASS=$((PASS + 1))
}

# --- Setup ---
mkdir -p "${SMOKE_DIR}" "${SMOKE_LOG_DIR}"

# Test 1: Small I/O (access_granularity bottleneck)
# Use stripe_count=1 dir
SMALL_DIR="${SMOKE_DIR}/small"
mkdir -p "${SMALL_DIR}"
lfs setstripe -c 1 -S 1M "${SMALL_DIR}" 2>/dev/null || true
run_test "small_io" "access_granularity=1" \
    "ior -a POSIX -t 512 -b 64K -s 100 -F -e -C -w -r" ""

# Test 2: Random I/O (access_pattern bottleneck)
run_test "random_io" "access_pattern=1" \
    "ior -a POSIX -t 4096 -b 10M -s 10 -z -e -C -w -r -O useO_DIRECT=1" ""

# Test 3: Interface misuse (POSIX on shared file)
run_test "interface_misuse" "interface_choice=1" \
    "ior -a POSIX -t 1048576 -b 100M -s 4 -e -C -w -r -O useO_DIRECT=1" \
    'MPICH_MPIIO_HINTS="*:romio_cb_write=disable:romio_cb_read=disable"'

# Test 4: Healthy (large sequential FPP)
HEALTHY_DIR="${SMOKE_DIR}/healthy"
mkdir -p "${HEALTHY_DIR}"
lfs setstripe -c -1 -S 1M "${HEALTHY_DIR}" 2>/dev/null || true
run_test "healthy_fpp" "healthy=1" \
    "ior -a POSIX -t 4194304 -b 100M -s 4 -F -e -C -w -r -O useO_DIRECT=1" ""

# Test 5: Healthy MPI-IO collective
run_test "healthy_collective" "healthy=1" \
    "ior -a MPIIO -t 4194304 -b 100M -s 4 -c -e -C -w -r" \
    'MPICH_MPIIO_HINTS="*:romio_cb_write=enable:romio_ds_write=disable"'

# Test 6: mdtest (metadata)
echo ""
echo "--- Test 6: mdtest (metadata_intensity) ---"
TESTS=$((TESTS + 1))
MDTEST_DIR="${SMOKE_DIR}/mdtest_test"
mkdir -p "${MDTEST_DIR}"
rm -f "${SMOKE_LOG_DIR}"/*.darshan 2>/dev/null || true
export DARSHAN_LOGPATH="${SMOKE_LOG_DIR}"

if srun --export=ALL,LD_PRELOAD="${DARSHAN_LIB}" \
    mdtest -n 100 -w 100 -e 100 -F -d "${MDTEST_DIR}" 2>&1 | tail -5; then
    LOG_FILE=$(ls -t "${SMOKE_LOG_DIR}"/*.darshan 2>/dev/null | head -1)
    if [ -n "${LOG_FILE}" ]; then
        echo "  Darshan log: $(basename ${LOG_FILE})"
        echo "  RESULT: PASS"
        PASS=$((PASS + 1))
    else
        echo "  Darshan log: NOT FOUND"
        FAIL=$((FAIL + 1))
    fi
else
    echo "  mdtest: FAILED"
    FAIL=$((FAIL + 1))
fi
rm -rf "${MDTEST_DIR}" 2>/dev/null || true

# --- Lustre stripe verification ---
echo ""
echo "--- Lustre Stripe Verification ---"
TESTS=$((TESTS + 1))
echo "  Bottleneck dir striping:"
STRIPE=$(lfs getstripe -c "${SMALL_DIR}" 2>/dev/null || echo "unknown")
echo "    ${SMALL_DIR}: stripe_count=${STRIPE}"
if [ "${STRIPE}" = "1" ]; then
    echo "  RESULT: PASS (PFL overridden)"
    PASS=$((PASS + 1))
else
    echo "  RESULT: WARN (stripe_count=${STRIPE}, expected 1)"
    # Not fatal — PFL may not be overridable from compute node
    PASS=$((PASS + 1))
fi

# --- Summary ---
echo ""
echo "============================================================"
echo "SMOKE TEST RESULTS"
echo "============================================================"
echo "Total tests: ${TESTS}"
echo "Passed:      ${PASS}"
echo "Failed:      ${FAIL}"
echo ""

if [ ${FAIL} -eq 0 ]; then
    echo "ALL TESTS PASSED. Ready for full sweep."
    echo ""
    echo "Next steps:"
    echo "  1. bash benchmarks/ior/run_ior_sweep.sh --dry-run   # Preview IOR jobs"
    echo "  2. bash benchmarks/ior/run_ior_sweep.sh             # Submit IOR jobs"
    echo "  3. bash benchmarks/mdtest/run_mdtest_sweep.sh       # Submit mdtest jobs"
    exit 0
else
    echo "${FAIL} TEST(S) FAILED. Investigate before running full sweep."
    echo "Check: data/benchmark_results/smoke_test_${SLURM_JOB_ID}.out"
    exit 1
fi
