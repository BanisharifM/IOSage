#!/bin/bash
# =============================================================================
# DLIO Parameter Sweep for Ground-Truth Generation
# =============================================================================
# Generates Darshan logs with KNOWN ML I/O patterns.
# DLIO emulates DL training I/O: data loading, shuffling, checkpointing.
#
# PREREQUISITES:
#   - mpi4py built from source against cray-mpich
#   - dlio-benchmark installed
#   - Run scripts/setup_benchmark_env.sh first
#
# HOW TO USE:
#   bash benchmarks/dlio/run_dlio_sweep.sh           # Submit all
#   bash benchmarks/dlio/run_dlio_sweep.sh --dry-run  # Preview only
# =============================================================================
set -euo pipefail

PROJECT_DIR="/work/hdd/bdau/mbanisharifdehkordi/SC_2026"
BENCH_SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch"
DLIO_DIR="${BENCH_SCRATCH}/dlio"
LOG_DIR="${PROJECT_DIR}/data/benchmark_logs/dlio"
RESULTS_DIR="${PROJECT_DIR}/data/benchmark_results/dlio"
DARSHAN_LIB="/work/hdd/bdau/mbanisharifdehkordi/darshan-install/lib/libdarshan.so"
PYTHON_BIN="/projects/bdau/envs/sc2026/bin/python"
DLIO_BIN="/projects/bdau/envs/sc2026/bin/dlio_benchmark"

REPETITIONS=3
DRY_RUN=false
SCENARIO_FILTER=""
SLURM_WALLTIME="08:00:00"
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

generate_dlio_job() {
    local scenario_name="$1"
    local label_dims="$2"
    local nranks="$3"
    local rep="$4"
    local dlio_overrides="$5"  # Hydra-style overrides
    local nodes=$(( (nranks + 127) / 128 ))
    [ $nodes -lt 1 ] && nodes=1

    local job_name="dlio_${scenario_name}_n${nranks}_rep${rep}"
    local script_path="${RESULTS_DIR}/${job_name}.slurm"
    local data_dir="${DLIO_DIR}/${job_name}"

    cat > "${script_path}" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks=${nranks}
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=${SLURM_WALLTIME}
#SBATCH --output=${RESULTS_DIR}/${job_name}_%j.out
#SBATCH --error=${RESULTS_DIR}/${job_name}_%j.err

export DARSHAN_LOGPATH="${LOG_DIR}"
export DARSHAN_MODMEM=4
export DARSHAN_ENABLE_NONMPI=1
mkdir -p "\${DARSHAN_LOGPATH}" "${data_dir}"

echo "============================================================"
echo "Scenario: ${scenario_name}"
echo "Label:    ${label_dims}"
echo "Ranks:    ${nranks}, Rep: ${rep}"
echo "Data dir: ${data_dir}"
echo "Date:     \$(date)"
echo "============================================================"

# Step 1: Generate data (no Darshan needed for data gen)
srun --export=ALL \\
    ${DLIO_BIN} \\
    workload=unet3d \\
    ++workload.workflow.generate_data=True \\
    ++workload.workflow.train=False \\
    ++workload.dataset.data_folder=${data_dir} \\
    ${dlio_overrides}

echo "Data generation complete at \$(date)"

# Step 2: Run training (this is what generates the Darshan log we want)
srun --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \\
    ${DLIO_BIN} \\
    workload=unet3d \\
    ++workload.workflow.generate_data=False \\
    ++workload.workflow.train=True \\
    ++workload.dataset.data_folder=${data_dir} \\
    ${dlio_overrides}

echo ""
echo "DLIO training complete at \$(date), exit code: \$?"

# Verify Darshan log
LATEST_LOG=\$(ls -t "\${DARSHAN_LOGPATH}"/*.darshan 2>/dev/null | head -1)
if [ -n "\${LATEST_LOG}" ]; then
    echo "Darshan log: \${LATEST_LOG} (\$(ls -lh "\${LATEST_LOG}" | awk '{print \$5}'))"
else
    echo "WARNING: No Darshan log found"
fi

# Cleanup DLIO data
rm -rf "${data_dir}" 2>/dev/null || true
SLURM_EOF

    echo "${script_path}"
}

echo "============================================================"
echo "DLIO Parameter Sweep Generator"
echo "Date: $(date)"
echo "============================================================"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${DLIO_DIR}" 2>/dev/null || true
TOTAL_JOBS=0
SUBMITTED_JOBS=0

# ===== small_records =====
# Label: access_granularity = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "small_records" ]; then
    echo ""
    echo "--- Scenario: small_records ---"
    for rl in 64 256 1024 4096; do
        for nranks in 4 8; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                overrides="++workload.dataset.record_length=${rl}"
                overrides+=" ++workload.dataset.num_files_train=1000"
                overrides+=" ++workload.dataset.num_samples_per_file=1"
                overrides+=" ++workload.reader.batch_size=1"
                overrides+=" ++workload.reader.read_threads=1"
                overrides+=" ++workload.train.computation_time=0.01"
                overrides+=" ++workload.train.epochs=2"
                overrides+=" ++workload.train.seed=42"
                overrides+=" ++workload.dataset.format=npz"

                script=$(generate_dlio_job \
                    "small_rl${rl}" "access_granularity=1" \
                    "${nranks}" "${rep}" "${overrides}")
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

# ===== checkpoint_burst =====
# Label: throughput_utilization = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "checkpoint_burst" ]; then
    echo ""
    echo "--- Scenario: checkpoint_burst ---"
    for ms in 100000000 500000000 1000000000; do
        for nranks in 4 8; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                overrides="++workload.dataset.record_length=65536"
                overrides+=" ++workload.dataset.num_files_train=100"
                overrides+=" ++workload.dataset.num_samples_per_file=10"
                overrides+=" ++workload.reader.batch_size=8"
                overrides+=" ++workload.reader.read_threads=4"
                overrides+=" ++workload.train.computation_time=0.5"
                overrides+=" ++workload.train.epochs=5"
                overrides+=" ++workload.train.seed=42"
                overrides+=" ++workload.workflow.checkpoint=True"
                overrides+=" ++workload.checkpoint.epochs_between_checkpoints=1"
                overrides+=" ++workload.checkpoint.model_size=${ms}"
                overrides+=" ++workload.dataset.format=npz"

                script=$(generate_dlio_job \
                    "ckpt_ms${ms}" "throughput_utilization=1" \
                    "${nranks}" "${rep}" "${overrides}")
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

# ===== healthy_ml =====
# Label: healthy = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "healthy_ml" ]; then
    echo ""
    echo "--- Scenario: healthy_ml ---"
    for rl in 1048576 4194304; do
        for nranks in 4 8; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                overrides="++workload.dataset.record_length=${rl}"
                overrides+=" ++workload.dataset.num_files_train=100"
                overrides+=" ++workload.dataset.num_samples_per_file=10"
                overrides+=" ++workload.reader.batch_size=32"
                overrides+=" ++workload.reader.read_threads=8"
                overrides+=" ++workload.train.computation_time=1.0"
                overrides+=" ++workload.train.epochs=3"
                overrides+=" ++workload.train.seed=42"
                overrides+=" ++workload.dataset.format=npz"

                script=$(generate_dlio_job \
                    "healthy_rl${rl}" "healthy=1" \
                    "${nranks}" "${rep}" "${overrides}")
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

# ===== shuffle_heavy =====
# Label: access_pattern = 1
if [ -z "${SCENARIO_FILTER}" ] || [ "${SCENARIO_FILTER}" = "shuffle_heavy" ]; then
    echo ""
    echo "--- Scenario: shuffle_heavy ---"
    for nranks in 4 8; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            overrides="++workload.dataset.record_length=65536"
            overrides+=" ++workload.dataset.num_files_train=5000"
            overrides+=" ++workload.dataset.num_samples_per_file=1"
            overrides+=" ++workload.reader.batch_size=1"
            overrides+=" ++workload.reader.read_threads=1"
            overrides+=" ++workload.reader.sample_shuffle=random"
            overrides+=" ++workload.reader.file_shuffle=random"
            overrides+=" ++workload.train.computation_time=0.01"
            overrides+=" ++workload.train.epochs=2"
            overrides+=" ++workload.train.seed=42"
            overrides+=" ++workload.dataset.format=npz"

            script=$(generate_dlio_job \
                "shuffle" "access_pattern=1" \
                "${nranks}" "${rep}" "${overrides}")
            if [ "${DRY_RUN}" = true ]; then
                echo "  [DRY] ${script}"
            else
                JOB_ID=$(sbatch "${script}" 2>&1 | awk '{print $NF}')
                echo "  Submitted: Job ${JOB_ID}"
                SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
            fi
        done
    done
fi

echo ""
echo "============================================================"
echo "DLIO Sweep Summary"
echo "Total jobs: ${TOTAL_JOBS}"
if [ "${DRY_RUN}" = true ]; then
    echo "Mode: DRY RUN"
else
    echo "Submitted: ${SUBMITTED_JOBS}"
fi
echo "============================================================"
