#!/bin/bash
# =============================================================================
# SLURM Array Job: Batch Feature Extraction from 1.37M Darshan Logs
# =============================================================================
# Architecture:
#   Step 1: Discover and split files into N chunks (one per array task)
#   Step 2: SLURM array job: each task processes its chunk with 100 workers
#   Step 3: Merge job validates every requested path and combines all chunks
#
# Each array task runs batch_extract.py which:
#   - Uses multiprocessing.Pool with imap_unordered (lazy, memory-efficient)
#   - Recycles workers every 500 tasks (maxtasksperchild, prevents C leaks)
#   - Has per-file 120s timeout via signal.alarm (prevents hung PyDarshan)
#   - Writes atomic sub-chunk files with _part_ prefix (no SLURM collision)
#   - Supports checkpoint/resume (skip completed sub-chunks on restart)
#   - Keeps one error CSV per attempt for diagnosis
#
# Submit:
#   bash scripts/slurm_batch_extract.sh            # writes under data/processed/resubmission/production
#   OUTPUT_DIR=<dir> bash scripts/slurm_batch_extract.sh
#   RESUME=1 OUTPUT_DIR=<dir> bash scripts/slurm_batch_extract.sh
#   DEPENDENCY=afterany:<jobid> bash scripts/slurm_batch_extract.sh   # start after timed runs end
#
# Monitor:
#   squeue -u $USER
#   ls -la $OUTPUT_DIR/chunks/
#   tail -f $OUTPUT_DIR/logs/extract_JOBID_0.out
#   wc -l $OUTPUT_DIR/chunks/*_errors.csv
#
# A new run refuses prior file lists. RESUME=1 reuses their exact request sets.
# =============================================================================

set -euo pipefail

# --- Configuration ---
IOSAGE_ENV="${IOSAGE_ENV:-/work/nvme/bdau/mbanisharifdehkordi/envs/iosage}"
PYTHON="${IOSAGE_ENV}/bin/python"
PROJECT_DIR=/work/hdd/bdau/mbanisharifdehkordi/IOSage
INPUT_DIR="${INPUT_DIR:-${PROJECT_DIR}/Darshan_Logs}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/data/processed/resubmission/production}"
CHUNK_DIR="${OUTPUT_DIR}/chunks"
FILELIST_DIR="${OUTPUT_DIR}/filelists"
FINAL_OUTPUT="${OUTPUT_DIR}/raw_features.parquet"
LOG_DIR="${OUTPUT_DIR}/logs"
# Reading 1.4M small files from 20 nodes loads the metadata servers of the shared file
# system; never run this next to timed benchmark or application runs.
DEPENDENCY="${DEPENDENCY:-}"
RESUME="${RESUME:-0}"

N_CHUNKS=20              # Number of array tasks (one per node)
WORKERS_PER_TASK=100     # Workers per node (128 CPUs, leave 28 for OS/IO)
TIMEOUT_PER_FILE=120     # Seconds before killing a stuck file
CHUNK_SIZE=10000         # Rows per internal sub-chunk file
RETRY_WORKERS=4          # Reduce storage contention for files that exceeded the first limit
RETRY_TIMEOUT=1800       # Slow large logs get a full 30 minutes on retry

echo "================================================================="
echo "  Batch Feature Extraction $(date)"
echo "================================================================="
echo "Input:       ${INPUT_DIR}"
echo "Output:      ${FINAL_OUTPUT}"
echo "Array tasks: ${N_CHUNKS} (1 node each)"
echo "Workers:     ${WORKERS_PER_TASK} per task"
echo "Timeout:     ${TIMEOUT_PER_FILE}s per file"
echo ""

# --- Step 1: Create or reuse file lists ---
if [ -e "${FINAL_OUTPUT}" ]; then
    echo "ERROR: final output already exists: ${FINAL_OUTPUT}"
    exit 1
fi
mkdir -p "${FILELIST_DIR}" "${CHUNK_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"

ALLFILES="${FILELIST_DIR}/all_darshan_files.txt"
if [ "${RESUME}" = "1" ]; then
    if ! ls "${FILELIST_DIR}"/chunk_[0-9][0-9][0-9].txt >/dev/null 2>&1; then
        echo "ERROR: RESUME=1 requires existing chunk file lists in ${FILELIST_DIR}"
        exit 1
    fi
    echo "[Step 1] Reusing existing file lists"
    TOTAL=$(cat "${FILELIST_DIR}"/chunk_[0-9][0-9][0-9].txt | wc -l)
else
    if ls "${FILELIST_DIR}"/chunk_*.txt >/dev/null 2>&1; then
        echo "ERROR: file lists already exist; set RESUME=1 or choose another OUTPUT_DIR"
        exit 1
    fi
    echo "[Step 1] Discovering .darshan files..."
    # Use lfs find on Lustre for faster metadata lookup, fall back to GNU find
    if lfs find "${INPUT_DIR}" -name "*.darshan" -type f > "${ALLFILES}" 2>/dev/null; then
        echo "  Used lfs find (Lustre-optimized)"
    else
        echo "  lfs find unavailable, using GNU find"
        find "${INPUT_DIR}" -name "*.darshan" -type f > "${ALLFILES}"
    fi
    TOTAL=$(wc -l < "${ALLFILES}")
    echo "  Found ${TOTAL} .darshan files"
    if [ "${TOTAL}" -eq 0 ]; then
        echo "ERROR: No .darshan files found in ${INPUT_DIR}"
        exit 1
    fi
    LINES_PER_CHUNK=$(( (TOTAL + N_CHUNKS - 1) / N_CHUNKS ))
    echo "  Splitting into ${N_CHUNKS} chunks of ~${LINES_PER_CHUNK} files each"
    split -l "${LINES_PER_CHUNK}" -d -a 3 "${ALLFILES}" "${FILELIST_DIR}/chunk_"
    for f in "${FILELIST_DIR}"/chunk_*; do
        if [[ ! "$f" == *.txt ]]; then
            mv "$f" "${f}.txt"
        fi
    done
    for f in "${FILELIST_DIR}"/chunk_*.txt; do
        shuf "$f" -o "$f"
    done
    echo "  Shuffled file lists for MDT load balancing"
fi

# Count actual chunks created (may be < N_CHUNKS if fewer files)
ACTUAL_CHUNKS=$(ls "${FILELIST_DIR}"/chunk_*.txt 2>/dev/null | wc -l)
echo "  Created ${ACTUAL_CHUNKS} file lists:"

for f in "${FILELIST_DIR}"/chunk_*.txt; do
    n=$(wc -l < "$f")
    echo "    $(basename "$f"): ${n} files"
done

# --- Step 2: Submit array job ---
echo ""
echo "[Step 2] Submitting SLURM array job (0-$((ACTUAL_CHUNKS - 1)))..."

ARRAY_JOBID=$(sbatch --parsable ${DEPENDENCY:+--dependency=$DEPENDENCY} <<SBATCH
#!/bin/bash
#SBATCH --job-name=darshan_extract
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --array=0-$((ACTUAL_CHUNKS - 1))
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=240g
#SBATCH --time=2-00:00:00
#SBATCH --export=NONE
#SBATCH --output=${LOG_DIR}/extract_%A_%a.out
#SBATCH --error=${LOG_DIR}/extract_%A_%a.err
source /etc/profile
export PYTHONNOUSERSITE=1

# ---- Per-task header ----
echo "================================================================="
echo "  Extract Chunk \${SLURM_ARRAY_TASK_ID} / $((ACTUAL_CHUNKS - 1))"
echo "================================================================="
echo "Job ID:    \${SLURM_JOB_ID} (array task \${SLURM_ARRAY_TASK_ID})"
echo "Node:      \$(hostname)"
echo "CPUs:      \${SLURM_CPUS_PER_TASK}"
echo "Memory:    \${SLURM_MEM_PER_NODE:-240g}"
echo "Start:     \$(date)"
echo "Workers:   ${WORKERS_PER_TASK}"
echo "Timeout:   ${TIMEOUT_PER_FILE}s per file"
echo ""

cd ${PROJECT_DIR}

# Map task ID to chunk file (zero-padded 3 digits)
CHUNK_ID=\$(printf "%03d" \${SLURM_ARRAY_TASK_ID})
FILELIST="${FILELIST_DIR}/chunk_\${CHUNK_ID}.txt"
CHUNK_OUTPUT="${CHUNK_DIR}/chunk_\${CHUNK_ID}.parquet"

if [ ! -f "\${FILELIST}" ]; then
    echo "ERROR: File list not found: \${FILELIST}"
    exit 1
fi

N_FILES=\$(wc -l < "\${FILELIST}")
echo "Processing \${N_FILES} files from \${FILELIST}"
echo "Output: \${CHUNK_OUTPUT}"
echo ""

# Run extraction. Exit 3 means the saved parts are valid but some paths need
# another attempt. --no-shuffle keeps the already shuffled request order.
set +e
${PYTHON} -m src.data.batch_extract \
    --file-list "\${FILELIST}" \
    --output "\${CHUNK_OUTPUT}" \
    --workers ${WORKERS_PER_TASK} \
    --chunk-size ${CHUNK_SIZE} \
    --timeout ${TIMEOUT_PER_FILE} \
    --no-shuffle \
    --log-level INFO
EXIT_CODE=\$?
set -e

if [ "\${EXIT_CODE}" -eq 3 ]; then
    echo "Retrying remaining paths with ${RETRY_WORKERS} workers and ${RETRY_TIMEOUT}s timeout"
    set +e
    ${PYTHON} -m src.data.batch_extract \
        --file-list "\${FILELIST}" \
        --output "\${CHUNK_OUTPUT}" \
        --workers ${RETRY_WORKERS} \
        --chunk-size ${CHUNK_SIZE} \
        --timeout ${RETRY_TIMEOUT} \
        --no-shuffle \
        --log-level INFO
    EXIT_CODE=\$?
    set -e
fi

echo ""
echo "================================================================="
echo "  Chunk \${SLURM_ARRAY_TASK_ID} finished (exit code: \${EXIT_CODE})"
echo "  End: \$(date)"
echo "================================================================="

echo "  Attempt reports: ${CHUNK_DIR}/chunk_\${CHUNK_ID}_attempt_*_errors.csv"

exit \${EXIT_CODE}
SBATCH
)

echo "  Array job submitted: ${ARRAY_JOBID}"

# --- Step 3: Submit validation and merge job ---
echo ""
echo "[Step 3] Submitting validation and merge job (afterany:${ARRAY_JOBID})..."

MERGE_JOBID=$(sbatch --parsable --dependency=afterany:${ARRAY_JOBID} <<SBATCH
#!/bin/bash
#SBATCH --job-name=merge_features
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=${LOG_DIR}/merge_%j.out
#SBATCH --error=${LOG_DIR}/merge_%j.err
source /etc/profile
export PYTHONNOUSERSITE=1

echo "================================================================="
echo "  Merge Feature Chunks"
echo "================================================================="
echo "Job ID:    \${SLURM_JOB_ID}"
echo "Node:      \$(hostname)"
echo "Start:     \$(date)"
echo "Expected:  ${ACTUAL_CHUNKS} chunks"
echo ""

cd ${PROJECT_DIR}

${PYTHON} - <<PYEOF
from src.data.batch_extract import merge_extraction_chunks

summary = merge_extraction_chunks(
    '${CHUNK_DIR}', '${FILELIST_DIR}', '${FINAL_OUTPUT}', ${ACTUAL_CHUNKS})
print(f'Merge complete: {summary}')
PYEOF
MERGE_RC=\$?
echo "merge rc=\${MERGE_RC}"
[ "\${MERGE_RC}" -ne 0 ] && exit "\${MERGE_RC}"

echo ""
echo "================================================================="
echo "  Merge Done \$(date)"
echo "================================================================="
SBATCH
)

echo "  Merge job submitted: ${MERGE_JOBID} (depends on ${ARRAY_JOBID})"

# --- Summary ---
echo ""
echo "================================================================="
echo "  Submission Summary"
echo "================================================================="
echo "Total files:  ${TOTAL}"
echo "Array job:    ${ARRAY_JOBID} (${ACTUAL_CHUNKS} tasks, 128 CPUs + 240GB each)"
echo "Merge job:    ${MERGE_JOBID} (publishes only after complete validation)"
echo "Output:       ${FINAL_OUTPUT}"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  ls -la ${CHUNK_DIR}/"
echo "  tail -f ${LOG_DIR}/extract_${ARRAY_JOBID}_0.out"
echo "  wc -l ${CHUNK_DIR}/*_errors.csv"
echo ""
echo "Post-completion validation:"
echo "  ${PYTHON} -c \"import pandas as pd; df = pd.read_parquet('${FINAL_OUTPUT}'); print(df.shape); print(df.dtypes)\""
