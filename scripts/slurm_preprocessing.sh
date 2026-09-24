#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120g
#SBATCH --time=02:00:00
#SBATCH --account=bdau-delta-cpu
#SBATCH --export=NONE
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/IOSage/logs/slurm/preprocess_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/IOSage/logs/slurm/preprocess_%j.err

# ==============================================================
# Preprocessing Pipeline (Stages 2-5)
# ==============================================================
# Single-node job: 152 MB parquet fits in memory.
# 16 CPUs / 120 GB is sufficient — most work is vectorized pandas/numpy.
#
# Usage:
#   sbatch scripts/slurm_preprocessing.sh
#   sbatch scripts/slurm_preprocessing.sh --start-stage 3  # resume from stage 3
# ==============================================================

set -euo pipefail

# --- Configuration ---
source /etc/profile
PROJECT_DIR="/work/hdd/bdau/mbanisharifdehkordi/IOSage"
IOSAGE_ENV="/work/nvme/bdau/mbanisharifdehkordi/envs/iosage"
PYTHON="${IOSAGE_ENV}/bin/python"
INPUT="${PROJECT_DIR}/data/processed/resubmission/production/raw_features.parquet"
OUTPUT_DIR="${PROJECT_DIR}/data/processed/resubmission/production"
CONFIG="${PROJECT_DIR}/configs/preprocessing.yaml"
export PYTHONNOUSERSITE=1
cd "$PROJECT_DIR"

# Pass through any extra args (e.g., --start-stage 3, --sample 10000)
EXTRA_ARGS=("$@")

# --- Setup ---
echo "================================================================="
echo "  Preprocessing Pipeline"
echo "================================================================="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "  Memory:    ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "  Start:     $(date)"
echo "  Input:     ${INPUT}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  Config:    ${CONFIG}"
printf '  Extra:'
printf ' %q' "${EXTRA_ARGS[@]}"
printf '\n'
echo "================================================================="

# Create log directory
mkdir -p "${PROJECT_DIR}/logs/slurm"

# Verify input exists
if [ ! -f "${INPUT}" ]; then
    echo "ERROR: Input file not found: ${INPUT}"
    exit 1
fi
echo "Input file: $(ls -lh ${INPUT})"

# --- Run Pipeline ---
"${PYTHON}" scripts/run_preprocessing.py \
    --input "${INPUT}" \
    --output-dir "${OUTPUT_DIR}" \
    --config "${CONFIG}" \
    "${EXTRA_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "================================================================="
echo "  Preprocessing Done: $(date)"
echo "  Exit code: ${EXIT_CODE}"
echo "================================================================="

# Summary of output files
echo ""
echo "Output files:"
ls -lh "${OUTPUT_DIR}"/cleaned_features.parquet \
       "${OUTPUT_DIR}"/features.parquet \
       "${OUTPUT_DIR}"/normalized_features.parquet \
       "${OUTPUT_DIR}"/eda_stats.parquet \
       "${OUTPUT_DIR}"/eda_report.json \
       "${OUTPUT_DIR}"/scalers.pkl \
       "${OUTPUT_DIR}"/split_indices.pkl \
       "${OUTPUT_DIR}"/preprocessing_manifest.json \
       "${OUTPUT_DIR}"/splits/*.parquet \
       2>/dev/null || echo "(some files may not exist yet)"

exit ${EXIT_CODE}
