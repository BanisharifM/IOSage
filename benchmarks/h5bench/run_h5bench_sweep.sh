#!/bin/bash
# =============================================================================
# h5bench Parameter Sweep for Ground-Truth Generation
# =============================================================================
# Generates Darshan logs with KNOWN HDF5 I/O patterns using LBNL h5bench.
#
# h5bench covers HDF5-specific bottleneck dimensions:
#   - interface_choice: independent vs collective HDF5 I/O
#   - access_pattern: contiguous vs interleaved memory/file layout
#   - access_granularity: small vs large DIM sizes
#   - throughput_utilization: single OST vs full striping
#   - healthy: collective + large + contiguous
#
# h5bench_write/read take a flat key=value config file (not JSON).
# The JSON format is only used by the Python h5bench driver.
#
# HOW IT FORCES SPECIFIC PATTERNS (overriding Delta defaults):
#   1. lfs setstripe -c 1  → overrides PFL auto-restriping → single OST
#   2. lfs setstripe -c -1 → all 12 HDD OSTs for healthy benchmarks
#   3. MPICH_MPIIO_HINTS   → controls ROMIO collective buffering
#   4. COLLECTIVE_DATA     → YES/NO controls HDF5 collective I/O
#   5. MEM_PATTERN/FILE_PATTERN → CONTIG/INTERLEAVED controls access pattern
#
# HOW TO USE:
#   bash benchmarks/h5bench/run_h5bench_sweep.sh           # Submit all
#   bash benchmarks/h5bench/run_h5bench_sweep.sh --dry-run  # Preview only
#   bash benchmarks/h5bench/run_h5bench_sweep.sh --scenario indep_small
#
# =============================================================================
set -euo pipefail

# --- Configuration ---
PROJECT_DIR="/work/hdd/bdau/mbanisharifdehkordi/SC_2026"
H5BENCH_BUILD="/work/hdd/bdau/mbanisharifdehkordi/h5bench/build"
BENCH_SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch"
BOTTLENECK_DIR="${BENCH_SCRATCH}/bottleneck"
HEALTHY_DIR="${BENCH_SCRATCH}/healthy"
LOG_DIR="${PROJECT_DIR}/data/benchmark_logs/h5bench"
RESULTS_DIR="${PROJECT_DIR}/data/benchmark_results/h5bench"
DARSHAN_LIB="/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so"
DARSHAN_PARSER="/projects/bdau/envs/sc2026/bin/darshan-parser"
HDF5_PARALLEL="/opt/cray/pe/hdf5-parallel/1.14.3.5/gnu/12.2"

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
generate_h5bench_config() {
    # h5bench_write/read expect flat KEY=VALUE config files
    # (the JSON format is only used by the Python h5bench driver)
    local config_path="$1"
    local output_dir="$2"
    local collective_data="$3"     # YES or NO
    local collective_meta="$4"     # YES or NO
    local mem_pattern="$5"         # CONTIG or INTERLEAVED
    local file_pattern="$6"        # CONTIG or INTERLEAVED
    local dim1="$7"                # elements per rank (8 bytes each for double)
    local timesteps="$8"           # number of write iterations
    local read_option="${9:-FULL}" # read option (FULL by default)

    # Write config (used by h5bench_write)
    cat > "${config_path}.write" << CONFIG_EOF
MEM_PATTERN=${mem_pattern}
FILE_PATTERN=${file_pattern}
TIMESTEPS=${timesteps}
DELAYED_CLOSE_TIMESTEPS=0
COLLECTIVE_DATA=${collective_data}
COLLECTIVE_METADATA=${collective_meta}
EMULATED_COMPUTE_TIME_PER_TIMESTEP=0 s
NUM_DIMS=1
DIM_1=${dim1}
DIM_2=1
DIM_3=1
CSV_FILE=output.csv
CONFIG_EOF

    # Read config (used by h5bench_read)
    cat > "${config_path}.read" << CONFIG_EOF
MEM_PATTERN=${mem_pattern}
FILE_PATTERN=${file_pattern}
READ_OPTION=${read_option}
TIMESTEPS=${timesteps}
DELAYED_CLOSE_TIMESTEPS=0
COLLECTIVE_DATA=${collective_data}
COLLECTIVE_METADATA=${collective_meta}
EMULATED_COMPUTE_TIME_PER_TIMESTEP=0 s
NUM_DIMS=1
DIM_1=${dim1}
DIM_2=1
DIM_3=1
CSV_FILE=output.csv
CONFIG_EOF
}

generate_job_script() {
    local scenario_name="$1"
    local label_dims="$2"
    local config_path="$3"
    local nranks="$4"
    local rep="$5"
    local output_dir="$6"
    local cb_mode="$7"              # enabled, disabled, default
    local nodes=$(( (nranks + 127) / 128 ))
    [ $nodes -lt 1 ] && nodes=1

    local job_name="h5b_${scenario_name}_n${nranks}_r${rep}"
    local script_path="${RESULTS_DIR}/${job_name}.slurm"

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
module load PrgEnv-gnu/8.6.0 cray-hdf5-parallel/1.14.3.5 2>/dev/null || true

# Cleanup h5bench data files on ANY exit
cleanup() { rm -f "${output_dir}/h5bench_output.h5"* "${output_dir}/output.csv" 2>/dev/null || true; }
trap cleanup EXIT

# Darshan log directory
export DARSHAN_LOGPATH="${LOG_DIR}"
mkdir -p "\${DARSHAN_LOGPATH}"

# HDF5 parallel library path (ensure runtime finds parallel version)
export LD_LIBRARY_PATH="${HDF5_PARALLEL}/lib:\${LD_LIBRARY_PATH:-}"

# ROMIO collective buffering control
${mpich_hints}
export MPICH_MPIIO_HINTS_DISPLAY=1

echo "============================================================"
echo "Scenario:    ${scenario_name}"
echo "Label:       ${label_dims}"
echo "Config:      ${config_path}"
echo "Ranks:       ${nranks} on ${nodes} node(s)"
echo "Output dir:  ${output_dir}"
echo "Darshan:     \${DARSHAN_LOGPATH}"
echo "CB mode:     ${cb_mode}"
echo "Rep:         ${rep}"
echo "Date:        \$(date)"
echo "Host:        \$(hostname)"
echo "============================================================"

# --- Pre-flight checks ---
echo ""
echo "Pre-flight checks:"
OUTPUT_STRIPE=\$(lfs getstripe -c "${output_dir}" 2>/dev/null || echo "unknown")
echo "  Output dir stripe_count: \${OUTPUT_STRIPE}"

if [ -x "${H5BENCH_BUILD}/h5bench_write" ]; then
    echo "  h5bench_write: OK"
else
    echo "  ERROR: h5bench_write not found at ${H5BENCH_BUILD}"
    exit 1
fi

if [ -f "${DARSHAN_LIB}" ]; then
    echo "  Darshan library: OK"
else
    echo "  ERROR: Darshan library not found"
    exit 1
fi

echo "  MPICH_MPIIO_HINTS: \${MPICH_MPIIO_HINTS:-<not set>}"
echo ""

# --- Run h5bench write with Darshan ---
echo "=== h5bench WRITE phase ==="
srun --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \\
    ${H5BENCH_BUILD}/h5bench_write ${config_path}.write ${output_dir}/h5bench_output.h5

WRITE_RC=\$?
echo ""
echo "Write completed at \$(date), exit code: \${WRITE_RC}"

# --- Run h5bench read with Darshan ---
# Read uses same HDF5 file created by write
if [ -f "${output_dir}/h5bench_output.h5" ]; then
    echo ""
    echo "=== h5bench READ phase ==="
    srun --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \\
        ${H5BENCH_BUILD}/h5bench_read ${config_path}.read ${output_dir}/h5bench_output.h5

    READ_RC=\$?
    echo ""
    echo "Read completed at \$(date), exit code: \${READ_RC}"
else
    echo "WARNING: HDF5 output file not found, skipping read phase"
fi

# --- Verify Darshan log ---
echo ""
echo "Checking for Darshan log..."
LATEST_LOG=\$(ls -t "\${DARSHAN_LOGPATH}"/*.darshan 2>/dev/null | head -1)
if [ -n "\${LATEST_LOG}" ]; then
    echo "Darshan log found: \${LATEST_LOG}"
    echo "Size: \$(ls -lh "\${LATEST_LOG}" | awk '{print \$5}')"
    ${DARSHAN_PARSER} --total "\${LATEST_LOG}" 2>/dev/null | head -20
else
    echo "WARNING: No Darshan log found"
fi

# --- Cleanup HDF5 data files ---
rm -f "${output_dir}/h5bench_output.h5"* "${output_dir}/output.csv" 2>/dev/null || true
echo ""
echo "Cleaned up h5bench data files."
SLURM_EOF

    echo "${script_path}"
}

# --- Main ---
echo "============================================================"
echo "h5bench Parameter Sweep Generator"
echo "Date: $(date)"
echo "Dry run: ${DRY_RUN}"
echo "Scenario filter: ${SCENARIO_FILTER:-all}"
echo "Repetitions: ${REPETITIONS}"
echo "============================================================"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
mkdir -p "${BOTTLENECK_DIR}" "${HEALTHY_DIR}" 2>/dev/null || true

TOTAL_JOBS=0
SUBMITTED_JOBS=0
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

# ===== SCENARIO: indep_small =====
# HDF5 independent I/O with small data — interface_choice bottleneck
# Each rank writes independently (no collective), small arrays
# 1024 elements × 8 bytes = 8 KB per rank per timestep
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "indep_small" ]; then
    echo ""
    echo "--- Scenario: indep_small (Interface Choice = BAD) ---"
    for nranks in 16 32 64; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            config="${RESULTS_DIR}/config_indep_small_n${nranks}_r${rep}.json"
            generate_h5bench_config "${config}" "${BOTTLENECK_DIR}" \
                "NO" "NO" "CONTIG" "CONTIG" "1024" "10"
            script=$(generate_job_script \
                "indep_small" "interface_choice=1" \
                "${config}" "${nranks}" "${rep}" "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: interleaved_access =====
# Interleaved memory+file pattern — access_pattern bottleneck
# Non-contiguous access forces HDF5 to do strided I/O
# 262144 elements × 8 bytes = 2 MB per rank per timestep
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "interleaved_access" ]; then
    echo ""
    echo "--- Scenario: interleaved_access (Access Pattern = BAD) ---"
    for nranks in 16 32 64; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            config="${RESULTS_DIR}/config_interleaved_n${nranks}_r${rep}.json"
            generate_h5bench_config "${config}" "${BOTTLENECK_DIR}" \
                "YES" "YES" "INTERLEAVED" "INTERLEAVED" "262144" "10"
            script=$(generate_job_script \
                "interleaved_access" "access_pattern=1" \
                "${config}" "${nranks}" "${rep}" "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: collective_small =====
# Collective HDF5 I/O but very small data — access_granularity bottleneck
# 128 elements × 8 bytes = 1 KB per rank per timestep
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "collective_small" ]; then
    echo ""
    echo "--- Scenario: collective_small (Access Granularity = BAD) ---"
    for nranks in 16 32 64; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            config="${RESULTS_DIR}/config_coll_small_n${nranks}_r${rep}.json"
            generate_h5bench_config "${config}" "${BOTTLENECK_DIR}" \
                "YES" "YES" "CONTIG" "CONTIG" "128" "20"
            script=$(generate_job_script \
                "collective_small" "access_granularity=1" \
                "${config}" "${nranks}" "${rep}" "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: collective_large_healthy =====
# Collective HDF5 I/O, large data, full striping — healthy baseline
# 4194304 elements × 8 bytes = 32 MB per rank per timestep
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "collective_large_healthy" ]; then
    echo ""
    echo "--- Scenario: collective_large_healthy (Healthy) ---"
    for nranks in 16 32 64 128; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            config="${RESULTS_DIR}/config_coll_large_n${nranks}_r${rep}.json"
            generate_h5bench_config "${config}" "${HEALTHY_DIR}" \
                "YES" "YES" "CONTIG" "CONTIG" "4194304" "5"
            script=$(generate_job_script \
                "collective_large_healthy" "healthy=1" \
                "${config}" "${nranks}" "${rep}" "${HEALTHY_DIR}" "enabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: indep_large_healthy =====
# Independent HDF5 I/O, large data (fewer ranks = less contention)
# 4194304 elements × 8 bytes = 32 MB per rank per timestep
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "indep_large_healthy" ]; then
    echo ""
    echo "--- Scenario: indep_large_healthy (Healthy) ---"
    for nranks in 4 8 16; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            config="${RESULTS_DIR}/config_indep_large_n${nranks}_r${rep}.json"
            generate_h5bench_config "${config}" "${HEALTHY_DIR}" \
                "NO" "NO" "CONTIG" "CONTIG" "4194304" "5"
            script=$(generate_job_script \
                "indep_large_healthy" "healthy=1" \
                "${config}" "${nranks}" "${rep}" "${HEALTHY_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# =========================================================================
# MULTI-LABEL SCENARIOS
# =========================================================================

# ===== SCENARIO: indep_small_interleaved =====
# Independent + interleaved + small data: 3 bottlenecks
# interface_choice + access_pattern + access_granularity
# 256 elements × 8 bytes = 2 KB per rank per timestep
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "indep_small_interleaved" ]; then
    echo ""
    echo "--- Scenario: indep_small_interleaved (Interface + Pattern + Granularity = BAD) ---"
    for nranks in 16 32 64; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            config="${RESULTS_DIR}/config_indep_small_inter_n${nranks}_r${rep}.json"
            generate_h5bench_config "${config}" "${BOTTLENECK_DIR}" \
                "NO" "NO" "INTERLEAVED" "INTERLEAVED" "256" "20"
            script=$(generate_job_script \
                "indep_small_interleaved" "access_granularity=1,interface_choice=1,access_pattern=1" \
                "${config}" "${nranks}" "${rep}" "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: indep_interleaved =====
# Independent + interleaved: 2 bottlenecks
# interface_choice + access_pattern (but large enough data to not be granularity issue)
# 524288 elements × 8 bytes = 4 MB per rank per timestep
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "indep_interleaved" ]; then
    echo ""
    echo "--- Scenario: indep_interleaved (Interface + Pattern = BAD) ---"
    for nranks in 16 32 64; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            config="${RESULTS_DIR}/config_indep_inter_n${nranks}_r${rep}.json"
            generate_h5bench_config "${config}" "${BOTTLENECK_DIR}" \
                "NO" "NO" "INTERLEAVED" "INTERLEAVED" "524288" "10"
            script=$(generate_job_script \
                "indep_interleaved" "interface_choice=1,access_pattern=1" \
                "${config}" "${nranks}" "${rep}" "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# ===== SCENARIO: indep_small_single_ost =====
# Independent + small + single OST: interface_choice + access_granularity + throughput_utilization
# 512 elements × 8 bytes = 4 KB per rank per timestep, on 1 OST
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "indep_small_single_ost" ]; then
    echo ""
    echo "--- Scenario: indep_small_single_ost (Interface + Granularity + Throughput = BAD) ---"
    for nranks in 32 64 128; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            config="${RESULTS_DIR}/config_indep_small_1ost_n${nranks}_r${rep}.json"
            generate_h5bench_config "${config}" "${BOTTLENECK_DIR}" \
                "NO" "NO" "CONTIG" "CONTIG" "512" "20"
            script=$(generate_job_script \
                "indep_small_single_ost" "access_granularity=1,interface_choice=1,throughput_utilization=1" \
                "${config}" "${nranks}" "${rep}" "${BOTTLENECK_DIR}" "disabled")
            submit_job "${script}"
        done
    done
fi

# --- Summary ---
echo ""
echo "============================================================"
echo "h5bench Sweep Summary"
echo "============================================================"
echo "Total jobs generated: ${TOTAL_JOBS}"
if [ "${DRY_RUN}" = true ]; then
    echo "Mode: DRY RUN (no jobs submitted)"
    echo "To submit: bash benchmarks/h5bench/run_h5bench_sweep.sh"
else
    echo "Jobs submitted: ${SUBMITTED_JOBS}"
fi
echo ""
echo "Scenario breakdown:"
echo "  Single-label:"
echo "    indep_small              → interface_choice=1"
echo "    interleaved_access       → access_pattern=1"
echo "    collective_small         → access_granularity=1"
echo "    collective_large_healthy → healthy=1"
echo "    indep_large_healthy      → healthy=1"
echo "  Multi-label:"
echo "    indep_small_interleaved  → access_granularity=1, interface_choice=1, access_pattern=1"
echo "    indep_interleaved        → interface_choice=1, access_pattern=1"
echo "    indep_small_single_ost   → access_granularity=1, interface_choice=1, throughput_utilization=1"
echo ""
echo "After completion, run feature extraction:"
echo "  python scripts/extract_benchmark_features.py --bench-type h5bench"
