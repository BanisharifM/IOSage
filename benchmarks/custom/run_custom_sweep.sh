#!/bin/bash
# =============================================================================
# Custom mpi4py Benchmark Sweep for Ground-Truth Generation
# =============================================================================
# Generates load-imbalanced I/O patterns not easily created by IOR/mdtest.
# Label: parallelism_efficiency = 1 (by construction)
#
# PREREQUISITES: mpi4py built from source against cray-mpich
# =============================================================================
set -euo pipefail

PROJECT_DIR="/work/hdd/bdau/mbanisharifdehkordi/SC_2026"
BENCH_SCRATCH="/work/hdd/bdau/mbanisharifdehkordi/bench_scratch"
LOG_DIR="${PROJECT_DIR}/data/benchmark_logs/custom"
RESULTS_DIR="${PROJECT_DIR}/data/benchmark_results/custom"
DARSHAN_LIB="/sw/spack/deltacpu-2022-03/apps/darshan-runtime/3.3.1-gcc-11.2.0-7tis4xp/lib/libdarshan.so"
PYTHON_BIN="/projects/bdau/envs/sc2026/bin/python"
SCRIPT="${PROJECT_DIR}/benchmarks/custom/load_imbalance.py"

REPETITIONS=3
DRY_RUN=false
SLURM_WALLTIME="01:00:00"
SLURM_PARTITION="cpu"
SLURM_ACCOUNT="bdau-delta-cpu"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --reps) REPETITIONS="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "Custom Benchmark Sweep Generator"
echo "Date: $(date)"
echo "============================================================"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" 2>/dev/null || true
TOTAL_JOBS=0
SUBMITTED_JOBS=0

# ===== Load Imbalance Scenarios =====
for factor in 2 5 10 20; do
    for base_mb in 10 50; do
        for nranks in 4 16 64; do
            for rep in $(seq 1 ${REPETITIONS}); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
                job_name="custom_imbalance_f${factor}_mb${base_mb}_n${nranks}_r${rep}"
                nodes=$(( (nranks + 127) / 128 ))
                [ $nodes -lt 1 ] && nodes=1
                output_dir="${BENCH_SCRATCH}/custom/${job_name}"

                script_path="${RESULTS_DIR}/${job_name}.slurm"
                cat > "${script_path}" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks=${nranks}
#SBATCH --cpus-per-task=1
#SBATCH --mem=32g
#SBATCH --time=${SLURM_WALLTIME}
#SBATCH --output=${RESULTS_DIR}/${job_name}_%j.out
#SBATCH --error=${RESULTS_DIR}/${job_name}_%j.err

export DARSHAN_LOGPATH="${LOG_DIR}"
export DARSHAN_ENABLE_NONMPI=1
mkdir -p "\${DARSHAN_LOGPATH}" "${output_dir}"

echo "Custom imbalance: factor=${factor}, base=${base_mb}MB, ranks=${nranks}, rep=${rep}"
echo "Date: \$(date)"

srun --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \\
    ${PYTHON_BIN} ${SCRIPT} \\
    --imbalance-factor ${factor} \\
    --base-size-mb ${base_mb} \\
    --output-dir "${output_dir}" \\
    --seed $((42 + rep))

echo "Complete at \$(date), exit: \$?"

LATEST_LOG=\$(ls -t "\${DARSHAN_LOGPATH}"/*.darshan 2>/dev/null | head -1)
[ -n "\${LATEST_LOG}" ] && echo "Darshan log: \${LATEST_LOG}" || echo "WARNING: No Darshan log"

rm -rf "${output_dir}" 2>/dev/null || true
SLURM_EOF

                if [ "${DRY_RUN}" = true ]; then
                    echo "  [DRY] ${script_path}"
                else
                    JOB_ID=$(sbatch "${script_path}" 2>&1 | awk '{print $NF}')
                    echo "  Submitted: Job ${JOB_ID}"
                    SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
                fi
            done
        done
    done
done

# ===== Healthy Balanced =====
echo ""
echo "--- Healthy balanced ---"
for base_mb in 50 100; do
    for nranks in 4 16 64; do
        for rep in $(seq 1 ${REPETITIONS}); do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            job_name="custom_balanced_mb${base_mb}_n${nranks}_r${rep}"
            nodes=$(( (nranks + 127) / 128 ))
            [ $nodes -lt 1 ] && nodes=1
            output_dir="${BENCH_SCRATCH}/custom/${job_name}"

            script_path="${RESULTS_DIR}/${job_name}.slurm"
            cat > "${script_path}" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks=${nranks}
#SBATCH --cpus-per-task=1
#SBATCH --mem=32g
#SBATCH --time=${SLURM_WALLTIME}
#SBATCH --output=${RESULTS_DIR}/${job_name}_%j.out
#SBATCH --error=${RESULTS_DIR}/${job_name}_%j.err

export DARSHAN_LOGPATH="${LOG_DIR}"
export DARSHAN_ENABLE_NONMPI=1
mkdir -p "\${DARSHAN_LOGPATH}" "${output_dir}"

echo "Balanced I/O: base=${base_mb}MB, ranks=${nranks}, rep=${rep}"

srun --export=ALL,LD_PRELOAD=${DARSHAN_LIB} \\
    ${PYTHON_BIN} ${SCRIPT} \\
    --imbalance-factor 1.0 \\
    --base-size-mb ${base_mb} \\
    --output-dir "${output_dir}" \\
    --seed $((42 + rep))

echo "Complete at \$(date)"
rm -rf "${output_dir}" 2>/dev/null || true
SLURM_EOF

            if [ "${DRY_RUN}" = true ]; then
                echo "  [DRY] ${script_path}"
            else
                JOB_ID=$(sbatch "${script_path}" 2>&1 | awk '{print $NF}')
                echo "  Submitted: Job ${JOB_ID}"
                SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
            fi
        done
    done
done

echo ""
echo "============================================================"
echo "Custom Sweep Summary: ${TOTAL_JOBS} total"
if [ "${DRY_RUN}" = true ]; then echo "Mode: DRY RUN"; else echo "Submitted: ${SUBMITTED_JOBS}"; fi
echo "============================================================"
