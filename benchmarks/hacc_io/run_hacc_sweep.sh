#!/bin/bash
# =============================================================================
# HACC-IO Parameter Sweep for Ground-Truth Generation
# =============================================================================
# Generates Darshan logs with KNOWN I/O patterns using the CORAL HACC-IO
# benchmark (cosmology checkpoint I/O kernel).
#
# HACC-IO writes 9 arrays per rank (xx,yy,zz,vx,vy,vz,phi,pid,mask)
# = 38 bytes per particle per rank.
#
# Three executables cover different I/O strategies:
#   hacc_io_posix_shared  — POSIX on shared file (interface misuse for parallel)
#   hacc_io_mpiio_shared  — MPI-IO on shared file (proper parallel I/O)
#   hacc_io_fpp           — POSIX file-per-process (one file per rank)
#
# HOW IT FORCES SPECIFIC PATTERNS (overriding Delta defaults):
#   1. lfs setstripe -c 1  → overrides PFL auto-restriping → single OST
#   2. lfs setstripe -c -1 → all 12 HDD OSTs for healthy benchmarks
#   3. MPICH_MPIIO_HINTS   → controls ROMIO collective buffering
#   4. Output cleanup       → rm checkpoint files immediately after Darshan capture
#   5. Separate stripe dirs → bottleneck vs healthy get different OST counts
#
# HOW TO USE:
#   bash benchmarks/hacc_io/run_hacc_sweep.sh           # Submit all scenarios
#   bash benchmarks/hacc_io/run_hacc_sweep.sh --dry-run  # Preview only
#   bash benchmarks/hacc_io/run_hacc_sweep.sh --scenario posix_shared_small
#
# =============================================================================
set -euo pipefail

# --- Configuration ---
PROJECT_DIR="/work/hdd/bdau/mbanisharifdehkordi/SC_2026"
HACC_BUILD="/work/hdd/bdau/mbanisharifdehkordi/hacc-io"
BENCH_SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch"
BOTTLENECK_DIR="${BENCH_SCRATCH}/bottleneck"
HEALTHY_DIR="${BENCH_SCRATCH}/healthy"
LOG_DIR="${PROJECT_DIR}/data/benchmark_logs/hacc_io"
RESULTS_DIR="${PROJECT_DIR}/data/benchmark_results/hacc_io"
DARSHAN_LIB="/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so"
DARSHAN_PARSER="/projects/bdau/envs/sc2026/bin/darshan-parser"

REPETITIONS=3
DRY_RUN=false
SCENARIO_FILTER=""
SLURM_WALLTIME="01:00:00"
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
    local label_dims="$2"           # e.g., "interface_choice=1,access_granularity=1"
    local executable="$3"           # hacc_io_posix_shared, hacc_io_mpiio_shared, hacc_io_fpp
    local particles_per_rank="$4"   # integer: data_per_rank = particles * 38 bytes
    local nranks="$5"
    local rep="$6"
    local output_dir="$7"           # BOTTLENECK_DIR or HEALTHY_DIR
    local cb_mode="$8"              # enabled, disabled, default
    local nodes=$(( (nranks + 127) / 128 ))
    [ $nodes -lt 1 ] && nodes=1

    local job_name="hacc_${scenario_name}_p${particles_per_rank}_n${nranks}_r${rep}"
    local script_path="${RESULTS_DIR}/${job_name}.slurm"
    local hacc_output="${output_dir}/${job_name}_checkpoint"

    # Calculate expected data size for logging
    local data_per_rank=$(( particles_per_rank * 38 ))
    local total_data=$(( data_per_rank * nranks ))

    # Build MPICH hints
    local mpich_hints=""
    if [ "$cb_mode" = "disabled" ]; then
        mpich_hints='export MPICH_MPIIO_HINTS="*:romio_cb_write=disable:romio_cb_read=disable:romio_ds_write=disable:romio_ds_read=disable"'
    elif [ "$cb_mode" = "enabled" ]; then
        mpich_hints='export MPICH_MPIIO_HINTS="*:romio_cb_write=enable:romio_ds_write=disable"'
    fi

    cat > "${script_path}" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks=${nranks}
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=${SLURM_WALLTIME}
#SBATCH --output=${RESULTS_DIR}/${job_name}_%j.out
#SBATCH --error=${RESULTS_DIR}/${job_name}_%j.err

# --- Environment ---
module load PrgEnv-gnu/8.6.0 2>/dev/null || true

# Cleanup checkpoint files on ANY exit
cleanup() { rm -f "${hacc_output}"* 2>/dev/null || true; }
trap cleanup EXIT

# Darshan log directory
export DARSHAN_LOGPATH="${LOG_DIR}"
mkdir -p "\${DARSHAN_LOGPATH}"

# ROMIO collective buffering control
${mpich_hints}

# Debug: show what MPI-IO hints are active
export MPICH_MPIIO_HINTS_DISPLAY=1

echo "============================================================"
echo "Scenario:    ${scenario_name}"
echo "Label:       ${label_dims}"
echo "Executable:  ${executable}"
echo "Particles:   ${particles_per_rank} per rank (${data_per_rank} bytes/rank)"
echo "Ranks:       ${nranks} on ${nodes} node(s)"
echo "Total data:  ${total_data} bytes"
echo "Output:      ${hacc_output}"
echo "Darshan:     \${DARSHAN_LOGPATH}"
echo "CB mode:     ${cb_mode}"
echo "Rep:         ${rep}"
echo "Date:        \$(date)"
echo "Host:        \$(hostname)"
echo "============================================================"

# --- Pre-flight checks ---
echo ""
echo "Pre-flight checks:"

# 1. Verify Lustre stripe on output directory
OUTPUT_STRIPE=\$(lfs getstripe -c "${output_dir}" 2>/dev/null || echo "unknown")
echo "  Output dir stripe_count: \${OUTPUT_STRIPE}"

# 2. Verify HACC-IO executable
if [ -x "${HACC_BUILD}/${executable}" ]; then
    echo "  Executable: OK (${executable})"
else
    echo "  ERROR: Executable not found: ${HACC_BUILD}/${executable}"
    exit 1
fi

# 3. Verify Darshan library
if [ -f "${DARSHAN_LIB}" ]; then
    echo "  Darshan library: OK"
else
    echo "  ERROR: Darshan library not found: ${DARSHAN_LIB}"
    exit 1
fi

echo "  MPICH_MPIIO_HINTS: \${MPICH_MPIIO_HINTS:-<not set>}"
echo ""

# --- Run HACC-IO with Darshan instrumentation ---
srun --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \\
    ${HACC_BUILD}/${executable} ${particles_per_rank} ${hacc_output}

HACC_RC=\$?
echo ""
echo "HACC-IO completed at \$(date), exit code: \${HACC_RC}"

# --- Verify Darshan log was created ---
echo ""
echo "Checking for Darshan log..."
LATEST_LOG=\$(ls -t "\${DARSHAN_LOGPATH}"/*.darshan 2>/dev/null | head -1)
if [ -n "\${LATEST_LOG}" ]; then
    echo "Darshan log found: \${LATEST_LOG}"
    echo "Size: \$(ls -lh "\${LATEST_LOG}" | awk '{print \$5}')"
    ${DARSHAN_PARSER} --total "\${LATEST_LOG}" 2>/dev/null | head -20
else
    echo "WARNING: No Darshan log found in \${DARSHAN_LOGPATH}"
fi

# --- Cleanup checkpoint data files ---
rm -f "${hacc_output}"* 2>/dev/null || true
echo ""
echo "Cleaned up HACC-IO checkpoint files."
SLURM_EOF

    echo "${script_path}"
}

# --- Main: Generate and submit all HACC-IO scenarios ---

echo "============================================================"
echo "HACC-IO Parameter Sweep Generator"
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

# Sequential submission to prevent disk overflow
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
    echo "  Submitted: ${script_path##*/} → Job ${JOB_ID}"
    SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
}

# =========================================================================
# SINGLE-LABEL SCENARIOS
# =========================================================================

# ===== SCENARIO: posix_shared_large =====
# POSIX I/O on shared file with large data — interface_choice bottleneck
# HACC-IO uses POSIX read/write on a shared file, which is suboptimal for
# parallel I/O. Should use MPI-IO collective.
# 1M particles/rank × 38 bytes = 38 MB/rank
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "posix_shared_large" ]; then
    echo ""
    echo "--- Scenario: posix_shared_large (Interface Choice = BAD) ---"
    for nranks in 16 32 64; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            script=$(generate_job_script \
                "posix_shared_large" "interface_choice=1" \
                "hacc_io_posix_shared" "1000000" "${nranks}" "${rep}" \
                "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: fpp_many_ranks =====
# File-per-process with many ranks — file_strategy bottleneck
# Each rank creates its own file → metadata overhead, many small files
# 100K particles/rank × 38 bytes = 3.8 MB/rank (moderate per file)
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "fpp_many_ranks" ]; then
    echo ""
    echo "--- Scenario: fpp_many_ranks (File Strategy = BAD) ---"
    for nranks in 32 64 128; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            script=$(generate_job_script \
                "fpp_many_ranks" "file_strategy=1" \
                "hacc_io_fpp" "100000" "${nranks}" "${rep}" \
                "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: posix_shared_single_ost =====
# POSIX on shared file, single OST — throughput_utilization bottleneck
# All ranks write to one file on one OST → bandwidth ceiling
# 500K particles/rank × 38 bytes = 19 MB/rank, many ranks contending
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "posix_shared_single_ost" ]; then
    echo ""
    echo "--- Scenario: posix_shared_single_ost (Throughput = BAD) ---"
    for nranks in 32 64 128; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            script=$(generate_job_script \
                "posix_shared_single_ost" "throughput_utilization=1" \
                "hacc_io_posix_shared" "500000" "${nranks}" "${rep}" \
                "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: mpiio_collective_healthy =====
# MPI-IO collective on shared file, full striping — healthy baseline
# Proper parallel I/O: MPI-IO with collective buffering on all OSTs
# 1M particles/rank × 38 bytes = 38 MB/rank
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "mpiio_collective_healthy" ]; then
    echo ""
    echo "--- Scenario: mpiio_collective_healthy (Healthy) ---"
    for nranks in 16 32 64 128; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            script=$(generate_job_script \
                "mpiio_collective_healthy" "healthy=1" \
                "hacc_io_mpiio_shared" "1000000" "${nranks}" "${rep}" \
                "${HEALTHY_DIR}" "enabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: fpp_healthy =====
# File-per-process with few ranks, large data — healthy baseline
# Moderate file count, large per-file data, full striping
# 2M particles/rank × 38 bytes = 76 MB/rank
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "fpp_healthy" ]; then
    echo ""
    echo "--- Scenario: fpp_healthy (Healthy) ---"
    for nranks in 4 8 16; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            script=$(generate_job_script \
                "fpp_healthy" "healthy=1" \
                "hacc_io_fpp" "2000000" "${nranks}" "${rep}" \
                "${HEALTHY_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# =========================================================================
# MULTI-LABEL SCENARIOS (multiple bottlenecks per job)
# =========================================================================

# ===== SCENARIO: posix_shared_small =====
# POSIX on shared file with small particles — interface_choice + access_granularity
# Small per-rank data on shared file without collective I/O
# 100 particles/rank × 38 bytes = 3,800 bytes/rank (tiny I/O)
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "posix_shared_small" ]; then
    echo ""
    echo "--- Scenario: posix_shared_small (Interface + Granularity = BAD) ---"
    for nranks in 16 32 64; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            script=$(generate_job_script \
                "posix_shared_small" "access_granularity=1,interface_choice=1" \
                "hacc_io_posix_shared" "100" "${nranks}" "${rep}" \
                "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: posix_shared_small_single_ost =====
# POSIX shared file, small data, single OST — 3 simultaneous bottlenecks
# interface_choice + access_granularity + throughput_utilization
# 200 particles/rank × 38 bytes = 7,600 bytes/rank
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "posix_shared_small_single_ost" ]; then
    echo ""
    echo "--- Scenario: posix_shared_small_single_ost (Interface + Granularity + Throughput = BAD) ---"
    for nranks in 32 64 128; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            script=$(generate_job_script \
                "posix_shared_small_single_ost" "access_granularity=1,interface_choice=1,throughput_utilization=1" \
                "hacc_io_posix_shared" "200" "${nranks}" "${rep}" \
                "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: fpp_small_many =====
# File-per-process with small particles and many ranks
# file_strategy + access_granularity
# 50 particles/rank × 38 bytes = 1,900 bytes/rank (tiny files, many of them)
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "fpp_small_many" ]; then
    echo ""
    echo "--- Scenario: fpp_small_many (File Strategy + Granularity = BAD) ---"
    for nranks in 32 64 128; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            script=$(generate_job_script \
                "fpp_small_many" "file_strategy=1,access_granularity=1" \
                "hacc_io_fpp" "50" "${nranks}" "${rep}" \
                "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: posix_shared_many_single_ost =====
# POSIX shared file, many ranks, single OST — interface_choice + throughput_utilization
# 500K particles/rank × 38 bytes = 19 MB/rank, all through one OST
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "posix_shared_many_single_ost" ]; then
    echo ""
    echo "--- Scenario: posix_shared_many_single_ost (Interface + Throughput = BAD) ---"
    for nranks in 32 64 128; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            script=$(generate_job_script \
                "posix_shared_many_single_ost" "interface_choice=1,throughput_utilization=1" \
                "hacc_io_posix_shared" "500000" "${nranks}" "${rep}" \
                "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# --- Summary ---
echo ""
echo "============================================================"
echo "HACC-IO Sweep Summary"
echo "============================================================"
echo "Total jobs generated: ${TOTAL_JOBS}"
if [ "${DRY_RUN}" = true ]; then
    echo "Mode: DRY RUN (no jobs submitted)"
    echo "To submit: bash benchmarks/hacc_io/run_hacc_sweep.sh"
else
    echo "Jobs submitted: ${SUBMITTED_JOBS}"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "Logs:    ls -la ${LOG_DIR}/"
    echo "Results: ls -la ${RESULTS_DIR}/"
fi
echo ""
echo "Scenario breakdown:"
echo "  Single-label:"
echo "    posix_shared_large         → interface_choice=1"
echo "    fpp_many_ranks             → file_strategy=1"
echo "    posix_shared_single_ost    → throughput_utilization=1"
echo "    mpiio_collective_healthy   → healthy=1"
echo "    fpp_healthy                → healthy=1"
echo "  Multi-label:"
echo "    posix_shared_small         → access_granularity=1, interface_choice=1"
echo "    posix_shared_small_1ost    → access_granularity=1, interface_choice=1, throughput_utilization=1"
echo "    fpp_small_many             → file_strategy=1, access_granularity=1"
echo "    posix_shared_many_1ost     → interface_choice=1, throughput_utilization=1"
echo ""
echo "After completion, run feature extraction:"
echo "  python scripts/extract_benchmark_features.py --bench-type hacc_io"
