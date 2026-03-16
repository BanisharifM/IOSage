#!/bin/bash
# =============================================================================
# IOR Parameter Sweep for Ground-Truth Generation
# =============================================================================
# Generates Darshan logs with KNOWN I/O patterns for each bottleneck dimension.
#
# HOW TO USE:
#   bash benchmarks/ior/run_ior_sweep.sh           # Submit all scenarios
#   bash benchmarks/ior/run_ior_sweep.sh --dry-run  # Preview commands only
#   bash benchmarks/ior/run_ior_sweep.sh --scenario small_posix  # Run one scenario
#
# HOW IT FORCES SPECIFIC PATTERNS (overriding Delta defaults):
#   1. lfs setstripe -c 1  → overrides PFL auto-restriping → single OST
#   2. MPICH_MPIIO_HINTS   → disables ROMIO collective buffering
#   3. -C -e               → defeats page cache (reorderTasks + fsync)
#   4. --posix.odirect    → bypasses page cache entirely (for t >= 4KB)
#   5. Separate stripe dirs → bottleneck vs healthy get different OST counts
#
# DARSHAN CAPTURE:
#   srun --export=ALL,LD_PRELOAD=libdarshan.so → instruments IOR I/O calls
#   Logs written to data/benchmark_logs/ior/
# =============================================================================
set -euo pipefail

# --- Configuration ---
PROJECT_DIR="/work/hdd/bdau/mbanisharifdehkordi/SC_2026"
BENCH_SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch"
BOTTLENECK_DIR="${BENCH_SCRATCH}/bottleneck"
HEALTHY_DIR="${BENCH_SCRATCH}/healthy"
LOG_DIR="${PROJECT_DIR}/data/benchmark_logs/ior"
RESULTS_DIR="${PROJECT_DIR}/data/benchmark_results/ior"
DARSHAN_LIB="/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so"
DARSHAN_PARSER="/projects/bdau/envs/sc2026/bin/darshan-parser"

REPETITIONS=3
DRY_RUN=false
SCENARIO_FILTER=""
SLURM_WALLTIME="08:00:00"
SLURM_PARTITION="cpu"
SLURM_ACCOUNT="bdau-delta-cpu"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --scenario) SCENARIO_FILTER="$2"; shift 2 ;;
        --reps) REPETITIONS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Helper Functions ---
generate_job_script() {
    local scenario_name="$1"
    local label_dims="$2"        # e.g., "access_granularity=1"
    local api="$3"               # POSIX or MPIIO
    local transfer_size="$4"     # bytes
    local block_size="$5"        # e.g., 64K, 1M, 100M
    local segments="$6"
    local nranks="$7"
    local rep="$8"
    local extra_flags="$9"       # Additional IOR flags
    local output_dir="${10}"     # BOTTLENECK_DIR or HEALTHY_DIR
    local cb_mode="${11}"        # enabled, disabled
    local nodes=$(( (nranks + 127) / 128 ))  # 128 cores per node, 1 rank per core
    [ $nodes -lt 1 ] && nodes=1

    local job_name="ior_${scenario_name}_t${transfer_size}_n${nranks}_r${rep}"
    local script_path="${RESULTS_DIR}/${job_name}.slurm"
    local ior_output="${output_dir}/${job_name}"

    # Build MPICH hints
    local mpich_hints=""
    if [ "$cb_mode" = "disabled" ]; then
        mpich_hints='export MPICH_MPIIO_HINTS="*:romio_cb_write=disable:romio_cb_read=disable:romio_ds_write=disable:romio_ds_read=disable"'
    elif [ "$cb_mode" = "enabled" ]; then
        mpich_hints='export MPICH_MPIIO_HINTS="*:romio_cb_write=enable:romio_ds_write=disable"'
    fi

    # Build IOR command
    local ior_cmd="ior -a ${api} -t ${transfer_size} -b ${block_size} -s ${segments} ${extra_flags} -o ${ior_output}"

    cat > "${script_path}" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks=${nranks}
#SBATCH --cpus-per-task=1
#SBATCH --mem=16g
#SBATCH --time=${SLURM_WALLTIME}
#SBATCH --output=${RESULTS_DIR}/${job_name}_%j.out
#SBATCH --error=${RESULTS_DIR}/${job_name}_%j.err

# --- Environment ---
module load ior/3.3.0-gcc13.3.1

# Ensure IOR data files are cleaned up on ANY exit (including failure/quota)
cleanup() { rm -f ${ior_output}* 2>/dev/null || true; }
trap cleanup EXIT

# Darshan log directory
export DARSHAN_LOGPATH="${LOG_DIR}"
mkdir -p "\${DARSHAN_LOGPATH}"

# ROMIO collective buffering control
${mpich_hints}

# Debug: show what MPI-IO hints are active
export MPICH_MPIIO_HINTS_DISPLAY=1

echo "============================================================"
echo "Scenario: ${scenario_name}"
echo "Label:    ${label_dims}"
echo "Config:   api=${api} t=${transfer_size} b=${block_size} s=${segments} n=${nranks} rep=${rep}"
echo "Output:   ${ior_output}"
echo "Darshan:  \${DARSHAN_LOGPATH}"
echo "CB mode:  ${cb_mode}"
echo "Date:     \$(date)"
echo "Host:     \$(hostname)"
echo "============================================================"

# --- Pre-flight: Verify configs are NOT being ignored by Delta ---
echo ""
echo "Pre-flight checks:"

# 1. Verify Lustre stripe on output directory
OUTPUT_STRIPE=\$(lfs getstripe -c "${output_dir}" 2>/dev/null || echo "unknown")
echo "  Output dir stripe_count: \${OUTPUT_STRIPE}"

# 2. Verify Darshan library exists on compute node
if [ -f "${DARSHAN_LIB}" ]; then
    echo "  Darshan library: OK"
else
    echo "  ERROR: Darshan library not found: ${DARSHAN_LIB}"
    exit 1
fi

# 3. Show ROMIO hints (will also be displayed by MPICH_MPIIO_HINTS_DISPLAY=1)
echo "  MPICH_MPIIO_HINTS: \${MPICH_MPIIO_HINTS:-<not set>}"
echo ""

# --- Run IOR with Darshan instrumentation ---
# Key: LD_PRELOAD passed ONLY to srun tasks (not to srun itself)
srun --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \\
    ${ior_cmd}

echo ""
echo "IOR completed at \$(date)"
echo "Exit code: \$?"

# --- Verify Darshan log was created ---
echo ""
echo "Checking for Darshan log..."
LATEST_LOG=\$(ls -t "\${DARSHAN_LOGPATH}"/*.darshan 2>/dev/null | head -1)
if [ -n "\${LATEST_LOG}" ]; then
    echo "Darshan log found: \${LATEST_LOG}"
    echo "Size: \$(ls -lh "\${LATEST_LOG}" | awk '{print \$5}')"
    # Quick parse to verify
    ${DARSHAN_PARSER} --total "\${LATEST_LOG}" 2>/dev/null | head -20
else
    echo "WARNING: No Darshan log found in \${DARSHAN_LOGPATH}"
    echo "Darshan instrumentation may have failed."
fi

# --- Cleanup IOR data files (large, not needed) ---
rm -f ${ior_output}* 2>/dev/null || true
echo ""
echo "Cleaned up IOR data files."
SLURM_EOF

    echo "${script_path}"
}

# --- Main: Generate and submit all IOR scenarios ---

echo "============================================================"
echo "IOR Parameter Sweep Generator"
echo "Date: $(date)"
echo "Dry run: ${DRY_RUN}"
echo "Scenario filter: ${SCENARIO_FILTER:-all}"
echo "Repetitions: ${REPETITIONS}"
echo "============================================================"

# Ensure directories exist
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
mkdir -p "${BOTTLENECK_DIR}" "${HEALTHY_DIR}" 2>/dev/null || true

TOTAL_JOBS=0
SUBMITTED_JOBS=0

# --- Sequential submission: chain ALL jobs to prevent inode/disk overflow ---
# Every job depends on the previous one via --dependency=afterany:PREV_JOB
PREV_JOB=""

submit_job() {
    local script_path="$1"
    if [ "${DRY_RUN}" = true ]; then
        echo "  [DRY] ${script_path}"
        return
    fi
    local JOB_ID
    if [ -n "${PREV_JOB}" ]; then
        JOB_ID=$(sbatch --dependency=afterany:${PREV_JOB} "${script_path}" 2>&1 | awk '{print $NF}')
    else
        JOB_ID=$(sbatch "${script_path}" 2>&1 | awk '{print $NF}')
    fi
    PREV_JOB="${JOB_ID}"
    echo "  Submitted: ${script_path##*/} → Job ${JOB_ID} (after: ${PREV_JOB})"
    SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
}

# ===== SCENARIO: small_posix =====
# Small I/O with POSIX (sub-4KB, no O_DIRECT)
# Label: access_granularity = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "small_posix" ]; then
    echo ""
    echo "--- Scenario: small_posix (Access Granularity = BAD) ---"
    for tsize in 64 128 256 512 1024 2048; do
        for nranks in 4 16; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                script=$(generate_job_script \
                    "small_posix" "access_granularity=1" \
                    "POSIX" "${tsize}" "64K" "100" "${nranks}" "${rep}" \
                    "-F -e -C -w -r" "${BOTTLENECK_DIR}" "disabled")
                submit_job "${script}"
            done
        done
    done
fi

# ===== SCENARIO: small_direct =====
# Small I/O with O_DIRECT (4KB minimum, bypasses cache)
# Label: access_granularity = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "small_direct" ]; then
    echo ""
    echo "--- Scenario: small_direct (Access Granularity = BAD, O_DIRECT) ---"
    for tsize in 4096 8192 16384; do
        for nranks in 4 16; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                script=$(generate_job_script \
                    "small_direct" "access_granularity=1" \
                    "POSIX" "${tsize}" "1M" "100" "${nranks}" "${rep}" \
                    "-F -C -w -r --posix.odirect" "${BOTTLENECK_DIR}" "disabled")
                submit_job "${script}"
            done
        done
    done
fi

# ===== SCENARIO: misaligned =====
# Non-power-of-2 transfer sizes
# Label: access_granularity = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "misaligned" ]; then
    echo ""
    echo "--- Scenario: misaligned (Access Granularity = BAD, non-aligned) ---"
    for tsize in 1000 1500 3000 7000; do
        for nranks in 4 16; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                script=$(generate_job_script \
                    "misaligned" "access_granularity=1" \
                    "POSIX" "${tsize}" "1M" "50" "${nranks}" "${rep}" \
                    "-F -e -C -w -r" "${BOTTLENECK_DIR}" "disabled")
                submit_job "${script}"
            done
        done
    done
fi

# ===== SCENARIO: random_posix =====
# Random offset access
# Label: access_pattern = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "random_posix" ]; then
    echo ""
    echo "--- Scenario: random_posix (Access Pattern = BAD) ---"
    for tsize in 4096 65536 1048576; do
        for nranks in 4 16 64; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                script=$(generate_job_script \
                    "random_posix" "access_pattern=1" \
                    "POSIX" "${tsize}" "100M" "10" "${nranks}" "${rep}" \
                    "-z -e -w -r --posix.odirect" "${BOTTLENECK_DIR}" "disabled")
                submit_job "${script}"
            done
        done
    done
fi

# ===== SCENARIO: random_small =====
# Random + small I/O (worst case)
# Label: access_pattern = 1, access_granularity = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "random_small" ]; then
    echo ""
    echo "--- Scenario: random_small (Pattern + Granularity = BAD) ---"
    for tsize in 512 1024; do
        for nranks in 4 16; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                script=$(generate_job_script \
                    "random_small" "access_pattern=1,access_granularity=1" \
                    "POSIX" "${tsize}" "10M" "50" "${nranks}" "${rep}" \
                    "-F -z -e -C -w -r" "${BOTTLENECK_DIR}" "disabled")
                submit_job "${script}"
            done
        done
    done
fi

# ===== SCENARIO: interface_misuse_posix_shared =====
# POSIX on shared file (should use collective MPI-IO)
# Label: interface_choice = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "interface_misuse_posix_shared" ]; then
    echo ""
    echo "--- Scenario: interface_misuse_posix_shared (Interface = BAD) ---"
    for nranks in 16 32 64 128; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            script=$(generate_job_script \
                "interface_posix_shared" "interface_choice=1" \
                "POSIX" "1048576" "100M" "4" "${nranks}" "${rep}" \
                "-e -C -w -r --posix.odirect" "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: interface_misuse_mpiio_indep =====
# MPI-IO independent on shared file (should use collective)
# Label: interface_choice = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "interface_misuse_mpiio_indep" ]; then
    echo ""
    echo "--- Scenario: interface_misuse_mpiio_indep (Interface = BAD) ---"
    for nranks in 16 32 64 128; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            script=$(generate_job_script \
                "interface_mpiio_indep" "interface_choice=1" \
                "MPIIO" "1048576" "100M" "4" "${nranks}" "${rep}" \
                "-e -C -w -r" "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: file_explosion =====
# File-per-process with many ranks
# Label: file_strategy = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "file_explosion" ]; then
    echo ""
    echo "--- Scenario: file_explosion (File Strategy = BAD) ---"
    for nranks in 64 128 256; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            script=$(generate_job_script \
                "file_explosion" "file_strategy=1" \
                "POSIX" "65536" "10M" "10" "${nranks}" "${rep}" \
                "-F -e -C -w -r --posix.odirect" "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: fsync_per_write =====
# Excessive fsync (write throughput bottleneck)
# Label: throughput_utilization = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "fsync_per_write" ]; then
    echo ""
    echo "--- Scenario: fsync_per_write (Throughput = BAD) ---"
    for tsize in 65536 1048576; do
        for nranks in 4 16; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                script=$(generate_job_script \
                    "fsync_per_write" "throughput_utilization=1" \
                    "POSIX" "${tsize}" "100M" "4" "${nranks}" "${rep}" \
                    "-F -e -C -w -r -Y" "${BOTTLENECK_DIR}" "disabled")
                submit_job "${script}"
            done
        done
    done
fi

# ===== SCENARIO: healthy_collective =====
# MPI-IO collective, large transfers, full striping
# Label: healthy = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "healthy_collective" ]; then
    echo ""
    echo "--- Scenario: healthy_collective (Healthy) ---"
    for nranks in 16 32 64 128; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            script=$(generate_job_script \
                "healthy_collective" "healthy=1" \
                "MPIIO" "4194304" "1G" "4" "${nranks}" "${rep}" \
                "-c -e -C -w -r" "${HEALTHY_DIR}" "enabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: healthy_posix_fpp =====
# POSIX file-per-process with large transfers
# Label: healthy = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "healthy_posix_fpp" ]; then
    echo ""
    echo "--- Scenario: healthy_posix_fpp (Healthy) ---"
    for nranks in 16 32 64; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            script=$(generate_job_script \
                "healthy_posix_fpp" "healthy=1" \
                "POSIX" "4194304" "1G" "4" "${nranks}" "${rep}" \
                "-F -e -C -w -r --posix.odirect" "${HEALTHY_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: healthy_large_seq =====
# Large sequential with multiple transfer sizes
# Label: healthy = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "healthy_large_seq" ]; then
    echo ""
    echo "--- Scenario: healthy_large_seq (Healthy) ---"
    for tsize in 1048576 4194304 16777216; do
        for nranks in 4 16; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                script=$(generate_job_script \
                    "healthy_large_seq" "healthy=1" \
                    "POSIX" "${tsize}" "1G" "4" "${nranks}" "${rep}" \
                    "-F -e -C -w -r --posix.odirect" "${HEALTHY_DIR}" "disabled")
                submit_job "${script}"
            done
        done
    done
fi

# ===== SCENARIO: io500_hard =====
# IO500 IOR-hard config: 47008-byte transfer on shared file (standardized stress test)
# ION paper (HotStorage'24) used this exact config for evaluation against Drishti
# Label: access_granularity = 1, interface_choice = 1 (small I/O on shared file)
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "io500_hard" ]; then
    echo ""
    echo "--- Scenario: io500_hard (IO500 IOR-hard standardized config) ---"
    for nranks in 16 64 256; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            # IO500 IOR-hard: 47008 bytes, shared file, interspersed writes
            # 47008 > 4096, so O_DIRECT is safe (page-aligned at 4KB boundary? No, 47008 = 46*1024 = 46KB, aligned)
            script=$(generate_job_script \
                "io500_hard" "access_granularity=1,interface_choice=1" \
                "POSIX" "47008" "47008" "1000" "${nranks}" "${rep}" \
                "-e -C -w -r" "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: io500_easy =====
# IO500 IOR-easy: large I/O, user-configurable (healthy baseline)
# Label: healthy = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "io500_easy" ]; then
    echo ""
    echo "--- Scenario: io500_easy (IO500 IOR-easy — healthy) ---"
    for nranks in 16 64 256; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            script=$(generate_job_script \
                "io500_easy" "healthy=1" \
                "POSIX" "1048576" "1G" "4" "${nranks}" "${rep}" \
                "-F -e -C -w -r --posix.odirect" "${HEALTHY_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: e2e_3d_write (E2E-style scientific I/O) =====
# Multi-API comparison: same data pattern through POSIX vs MPI-IO
# Demonstrates interface choice impact on identical workload
# Label: interface_choice = 1 (POSIX version) or healthy = 1 (MPI-IO collective version)
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "e2e_posix_vs_mpiio" ]; then
    echo ""
    echo "--- Scenario: e2e_posix_vs_mpiio (E2E API comparison) ---"
    for nranks in 16 64; do
        for rep in $(seq 1 ${REPETITIONS}); do
            # POSIX on shared file (bad)
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            script=$(generate_job_script \
                "e2e_posix_shared" "interface_choice=1" \
                "POSIX" "1048576" "100M" "4" "${nranks}" "${rep}" \
                "-e -C -w -r" "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"

            # MPI-IO collective on shared file (good)
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            script=$(generate_job_script \
                "e2e_mpiio_coll" "healthy=1" \
                "MPIIO" "1048576" "100M" "4" "${nranks}" "${rep}" \
                "-c -e -C -w -r" "${HEALTHY_DIR}" "enabled")
            submit_job "${script}"
        done
    done
fi

# --- Summary ---
echo ""
echo "============================================================"
echo "IOR Sweep Summary"
echo "============================================================"
echo "Total jobs generated: ${TOTAL_JOBS}"
if [ "${DRY_RUN}" = true ]; then
    echo "Mode: DRY RUN (no jobs submitted)"
    echo "To submit: bash benchmarks/ior/run_ior_sweep.sh"
else
    echo "Jobs submitted: ${SUBMITTED_JOBS}"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "Logs:    ls -la ${LOG_DIR}/"
    echo "Results: ls -la ${RESULTS_DIR}/"
fi
echo ""
echo "After completion, verify logs:"
echo "  ${PROJECT_DIR}/scripts/verify_benchmark_logs.sh ior"
