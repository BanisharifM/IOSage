#!/bin/bash
# =============================================================================
# mdtest Parameter Sweep for Ground-Truth Generation
# =============================================================================
# Generates Darshan logs with KNOWN metadata-intensive patterns.
#
# HOW TO USE:
#   bash benchmarks/mdtest/run_mdtest_sweep.sh           # Submit all
#   bash benchmarks/mdtest/run_mdtest_sweep.sh --dry-run  # Preview only
#   bash benchmarks/mdtest/run_mdtest_sweep.sh --scenario metadata_storm_shared
#
# KEY mdtest FLAGS:
#   -n COUNT    Files per MPI rank
#   -w BYTES    Write bytes per file on creation
#   -e BYTES    Read bytes per file
#   -F          File-only mode (skip dir tests)
#   -u          Unique directory per rank (reduces MDS contention)
#   -z DEPTH    Directory tree depth
#   -b BRANCH   Branching factor per level
#   -N SHIFT    Stagger stat/read by N ranks (defeats MDS cache)
#   -R          Random stat order
# =============================================================================
set -euo pipefail

# --- Configuration ---
PROJECT_DIR="/work/hdd/bdau/mbanisharifdehkordi/SC_2026"
BENCH_SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch"
MDTEST_DIR="${BENCH_SCRATCH}/mdtest"
LOG_DIR="${PROJECT_DIR}/data/benchmark_logs/mdtest"
RESULTS_DIR="${PROJECT_DIR}/data/benchmark_results/mdtest"
DARSHAN_LIB="/sw/spack/deltacpu-2022-03/apps/darshan-runtime/3.3.1-gcc-11.2.0-7tis4xp/lib/libdarshan.so"
DARSHAN_PARSER="/sw/spack/deltacpu-2022-03/apps/darshan-util/3.3.1-gcc-11.2.0-vq4wq2e/bin/darshan-parser"

REPETITIONS=3
DRY_RUN=false
SCENARIO_FILTER=""
SLURM_WALLTIME="02:00:00"
SLURM_PARTITION="cpu"
SLURM_ACCOUNT="bdau-delta-cpu"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --scenario) SCENARIO_FILTER="$2"; shift 2 ;;
        --reps) REPETITIONS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

generate_mdtest_job() {
    local scenario_name="$1"
    local label_dims="$2"
    local nranks="$3"
    local items="$4"
    local write_bytes="$5"
    local read_bytes="$6"
    local rep="$7"
    local extra_flags="$8"
    local nodes=$(( (nranks + 127) / 128 ))
    [ $nodes -lt 1 ] && nodes=1

    local job_name="mdtest_${scenario_name}_n${items}_r${nranks}_rep${rep}"
    local script_path="${RESULTS_DIR}/${job_name}.slurm"
    local test_dir="${MDTEST_DIR}/${job_name}"

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

module load ior/3.3.0-gcc13.3.1

export DARSHAN_LOGPATH="${LOG_DIR}"
mkdir -p "\${DARSHAN_LOGPATH}"

echo "============================================================"
echo "Scenario: ${scenario_name}"
echo "Label:    ${label_dims}"
echo "Config:   items=${items} w=${write_bytes} e=${read_bytes} n=${nranks} rep=${rep}"
echo "Test dir: ${test_dir}"
echo "Date:     \$(date)"
echo "Host:     \$(hostname)"
echo "============================================================"

# --- Pre-flight: Verify configs are NOT being ignored by Delta ---
echo ""
echo "Pre-flight checks:"

# Verify Darshan library exists on compute node
if [ -f "${DARSHAN_LIB}" ]; then
    echo "  Darshan library: OK"
else
    echo "  ERROR: Darshan library not found: ${DARSHAN_LIB}"
    exit 1
fi

# Create test directory (single-stripe for metadata focus)
mkdir -p "${test_dir}"

# Verify stripe on test directory
TEST_STRIPE=\$(lfs getstripe -c "${test_dir}" 2>/dev/null || echo "unknown")
echo "  Test dir stripe_count: \${TEST_STRIPE}"
echo ""

srun --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \\
    mdtest -n ${items} -w ${write_bytes} -e ${read_bytes} \\
    ${extra_flags} \\
    -d "${test_dir}"

echo ""
echo "mdtest completed at \$(date), exit code: \$?"

# Verify Darshan log
LATEST_LOG=\$(ls -t "\${DARSHAN_LOGPATH}"/*.darshan 2>/dev/null | head -1)
if [ -n "\${LATEST_LOG}" ]; then
    echo "Darshan log: \${LATEST_LOG} (\$(ls -lh "\${LATEST_LOG}" | awk '{print \$5}'))"
else
    echo "WARNING: No Darshan log found"
fi

# Cleanup test files
rm -rf "${test_dir}" 2>/dev/null || true
SLURM_EOF

    echo "${script_path}"
}

echo "============================================================"
echo "mdtest Parameter Sweep Generator"
echo "Date: $(date)"
echo "Dry run: ${DRY_RUN}"
echo "============================================================"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${MDTEST_DIR}" 2>/dev/null || true
TOTAL_JOBS=0
SUBMITTED_JOBS=0

# ===== metadata_storm_shared =====
# Label: metadata_intensity = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "metadata_storm_shared" ]; then
    echo ""
    echo "--- Scenario: metadata_storm_shared ---"
    for items in 1000 5000 10000 50000; do
        for nranks in 4 16; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                script=$(generate_mdtest_job \
                    "meta_shared" "metadata_intensity=1" \
                    "${nranks}" "${items}" "100" "100" "${rep}" \
                    "-F")
                if [ "${DRY_RUN}" = true ]; then
                    echo "  [DRY] ${script}"
                else
                    JOB_ID=$(sbatch "${script}" 2>&1 | awk '{print $NF}')
                    echo "  Submitted: Job ${JOB_ID}"
                    SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
                fi
            done
        done
    done
fi

# ===== metadata_storm_unique =====
# Label: metadata_intensity = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "metadata_storm_unique" ]; then
    echo ""
    echo "--- Scenario: metadata_storm_unique ---"
    for items in 5000 10000 50000; do
        for nranks in 4 16 64; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                script=$(generate_mdtest_job \
                    "meta_unique" "metadata_intensity=1" \
                    "${nranks}" "${items}" "0" "0" "${rep}" \
                    "-F -u")
                if [ "${DRY_RUN}" = true ]; then
                    echo "  [DRY] ${script}"
                else
                    JOB_ID=$(sbatch "${script}" 2>&1 | awk '{print $NF}')
                    echo "  Submitted: Job ${JOB_ID}"
                    SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
                fi
            done
        done
    done
fi

# ===== metadata_cross_node =====
# Label: metadata_intensity = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "metadata_cross_node" ]; then
    echo ""
    echo "--- Scenario: metadata_cross_node ---"
    for items in 5000 10000; do
        for nranks in 32 64 128; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                script=$(generate_mdtest_job \
                    "meta_cross" "metadata_intensity=1" \
                    "${nranks}" "${items}" "64" "64" "${rep}" \
                    "-F -N ${nranks}")
                if [ "${DRY_RUN}" = true ]; then
                    echo "  [DRY] ${script}"
                else
                    JOB_ID=$(sbatch "${script}" 2>&1 | awk '{print $NF}')
                    echo "  Submitted: Job ${JOB_ID}"
                    SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
                fi
            done
        done
    done
fi

# ===== file_per_process_explosion =====
# Label: file_strategy = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "file_per_process_explosion" ]; then
    echo ""
    echo "--- Scenario: file_per_process_explosion ---"
    for items in 100 500 1000 5000; do
        for nranks in 32 64 128; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                script=$(generate_mdtest_job \
                    "fpp_explosion" "file_strategy=1" \
                    "${nranks}" "${items}" "4096" "4096" "${rep}" \
                    "-F -u")
                if [ "${DRY_RUN}" = true ]; then
                    echo "  [DRY] ${script}"
                else
                    JOB_ID=$(sbatch "${script}" 2>&1 | awk '{print $NF}')
                    echo "  Submitted: Job ${JOB_ID}"
                    SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
                fi
            done
        done
    done
fi

# ===== deep_tree =====
# Label: metadata_intensity = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "deep_tree" ]; then
    echo ""
    echo "--- Scenario: deep_tree ---"
    for depth in 3 5 10; do
        for branch in 2 5; do
            for nranks in 4 16; do
                for rep in $(seq 1 ${REPETITIONS}); do
                    TOTAL_JOBS=$((TOTAL_JOBS + 1))
                    script=$(generate_mdtest_job \
                        "deep_tree_d${depth}_b${branch}" "metadata_intensity=1" \
                        "${nranks}" "100" "0" "0" "${rep}" \
                        "-u -z ${depth} -b ${branch}")
                    if [ "${DRY_RUN}" = true ]; then
                        echo "  [DRY] ${script}"
                    else
                        JOB_ID=$(sbatch "${script}" 2>&1 | awk '{print $NF}')
                        echo "  Submitted: Job ${JOB_ID}"
                        SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
                    fi
                done
            done
        done
    done
fi

# ===== healthy_metadata =====
# Label: healthy = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "healthy_metadata" ]; then
    echo ""
    echo "--- Scenario: healthy_metadata ---"
    for items in 10 50; do
        for nranks in 4 16; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                script=$(generate_mdtest_job \
                    "healthy" "healthy=1" \
                    "${nranks}" "${items}" "1048576" "1048576" "${rep}" \
                    "-F -u")
                if [ "${DRY_RUN}" = true ]; then
                    echo "  [DRY] ${script}"
                else
                    JOB_ID=$(sbatch "${script}" 2>&1 | awk '{print $NF}')
                    echo "  Submitted: Job ${JOB_ID}"
                    SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
                fi
            done
        done
    done
fi

# ===== io500_mdtest_hard =====
# IO500 mdtest-hard: 3901-byte files in SHARED directory (standardized stress test)
# ION paper used this config. Single shared dir = maximum MDS contention
# Label: metadata_intensity = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "io500_mdtest_hard" ]; then
    echo ""
    echo "--- Scenario: io500_mdtest_hard (IO500 standardized) ---"
    for items in 1000 5000 10000; do
        for nranks in 16 64 256; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                # IO500 mdtest-hard: shared dir (-S implied by no -u), 3901 bytes per file
                script=$(generate_mdtest_job \
                    "io500_hard" "metadata_intensity=1" \
                    "${nranks}" "${items}" "3901" "3901" "${rep}" \
                    "-F")
                if [ "${DRY_RUN}" = true ]; then
                    echo "  [DRY] ${script}"
                else
                    JOB_ID=$(sbatch "${script}" 2>&1 | awk '{print $NF}')
                    echo "  Submitted: Job ${JOB_ID}"
                    SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
                fi
            done
        done
    done
fi

# ===== io500_mdtest_easy =====
# IO500 mdtest-easy: unique dirs, empty files (healthy metadata)
# Label: healthy = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "io500_mdtest_easy" ]; then
    echo ""
    echo "--- Scenario: io500_mdtest_easy (IO500 standardized — healthy) ---"
    for items in 100 500; do
        for nranks in 16 64; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                script=$(generate_mdtest_job \
                    "io500_easy" "healthy=1" \
                    "${nranks}" "${items}" "0" "0" "${rep}" \
                    "-F -u")
                if [ "${DRY_RUN}" = true ]; then
                    echo "  [DRY] ${script}"
                else
                    JOB_ID=$(sbatch "${script}" 2>&1 | awk '{print $NF}')
                    echo "  Submitted: Job ${JOB_ID}"
                    SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
                fi
            done
        done
    done
fi

echo ""
echo "============================================================"
echo "mdtest Sweep Summary"
echo "Total jobs: ${TOTAL_JOBS}"
if [ "${DRY_RUN}" = true ]; then
    echo "Mode: DRY RUN"
else
    echo "Submitted: ${SUBMITTED_JOBS}"
fi
echo "============================================================"
