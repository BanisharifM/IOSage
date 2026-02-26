#!/bin/bash
# =============================================================================
# SLURM Array Job: Batch Feature Extraction from 1.37M Darshan Logs
# =============================================================================
# Architecture:
#   Step 1: Discover and split files into N chunks (one per array task)
#   Step 2: SLURM array job — each task processes its chunk with 100 workers
#   Step 3: Merge job (afterany) — combines all chunk parquets into one file
#
# Each array task runs batch_extract.py which:
#   - Uses multiprocessing.Pool with imap_unordered (lazy, memory-efficient)
#   - Recycles workers every 500 tasks (maxtasksperchild, prevents C leaks)
#   - Has per-file 120s timeout via signal.alarm (prevents hung PyDarshan)
#   - Writes atomic sub-chunk files with _part_ prefix (no SLURM collision)
#   - Supports checkpoint/resume (skip completed sub-chunks on restart)
#   - Logs errors to CSV for post-hoc diagnosis
#
# Submit:
#   bash scripts/slurm_batch_extract.sh
#
# Monitor:
#   squeue -u $USER
#   ls -la data/processed/chunks/
#   tail -f logs/slurm/extract_JOBID_0.out
#   wc -l data/processed/chunks/*_errors.csv
# =============================================================================

set -euo pipefail

# --- Configuration ---
PYTHON=/projects/bdau/envs/sc2026/bin/python
PROJECT_DIR=/work/hdd/bdau/mbanisharifdehkordi/SC_2026
INPUT_DIR="${PROJECT_DIR}/Darshan_Logs"
OUTPUT_DIR="${PROJECT_DIR}/data/processed"
CHUNK_DIR="${OUTPUT_DIR}/chunks"
FILELIST_DIR="${PROJECT_DIR}/data/filelists"
FINAL_OUTPUT="${OUTPUT_DIR}/raw_features.parquet"
LOG_DIR="${PROJECT_DIR}/logs/slurm"

N_CHUNKS=20              # Number of array tasks (one per node)
WORKERS_PER_TASK=100     # Workers per node (128 CPUs, leave 28 for OS/IO)
TIMEOUT_PER_FILE=120     # Seconds before killing a stuck file
CHUNK_SIZE=10000         # Rows per internal sub-chunk file

echo "================================================================="
echo "  Batch Feature Extraction — $(date)"
echo "================================================================="
echo "Input:       ${INPUT_DIR}"
echo "Output:      ${FINAL_OUTPUT}"
echo "Array tasks: ${N_CHUNKS} (1 node each)"
echo "Workers:     ${WORKERS_PER_TASK} per task"
echo "Timeout:     ${TIMEOUT_PER_FILE}s per file"
echo ""

# --- Step 1: Create file lists ---
mkdir -p "${FILELIST_DIR}" "${CHUNK_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[Step 1] Discovering .darshan files..."

ALLFILES="${FILELIST_DIR}/all_darshan_files.txt"

# Use lfs find on Lustre for faster metadata lookup, fall back to GNU find
if lfs find "${INPUT_DIR}" -name "*.darshan" -type f > "${ALLFILES}" 2>/dev/null; then
    echo "  Used lfs find (Lustre-optimized)"
else
    echo "  lfs find unavailable, using GNU find"
    find "${INPUT_DIR}" -name "*.darshan" -type f > "${ALLFILES}"
fi

TOTAL=$(wc -l < "${ALLFILES}")
echo "  Found ${TOTAL} .darshan files"

# Zero-file guard
if [ "${TOTAL}" -eq 0 ]; then
    echo "ERROR: No .darshan files found in ${INPUT_DIR}"
    echo "Check that Darshan logs have been unpacked."
    exit 1
fi

# Split into N chunks
LINES_PER_CHUNK=$(( (TOTAL + N_CHUNKS - 1) / N_CHUNKS ))
echo "  Splitting into ${N_CHUNKS} chunks of ~${LINES_PER_CHUNK} files each"

# Clean old file lists
rm -f "${FILELIST_DIR}"/chunk_*.txt

split -l "${LINES_PER_CHUNK}" -d -a 3 "${ALLFILES}" "${FILELIST_DIR}/chunk_"

# Rename split output to .txt
for f in "${FILELIST_DIR}"/chunk_*; do
    if [[ ! "$f" == *.txt ]]; then
        mv "$f" "${f}.txt"
    fi
done

# Shuffle each chunk for Lustre MDT load balancing
# (Files are organized by date/user — sequential access hammers same MDT)
for f in "${FILELIST_DIR}"/chunk_*.txt; do
    shuf "$f" -o "$f"
done
echo "  Shuffled file lists for MDT load balancing"

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

ARRAY_JOBID=$(sbatch --parsable <<SBATCH
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
#SBATCH --output=${LOG_DIR}/extract_%A_%a.out
#SBATCH --error=${LOG_DIR}/extract_%A_%a.err

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

# Run extraction
# --no-shuffle because we pre-shuffled the file lists above
${PYTHON} -m src.data.batch_extract \
    --file-list "\${FILELIST}" \
    --output "\${CHUNK_OUTPUT}" \
    --workers ${WORKERS_PER_TASK} \
    --chunk-size ${CHUNK_SIZE} \
    --timeout ${TIMEOUT_PER_FILE} \
    --raw \
    --no-shuffle \
    --log-level INFO

EXIT_CODE=\$?

echo ""
echo "================================================================="
echo "  Chunk \${SLURM_ARRAY_TASK_ID} finished (exit code: \${EXIT_CODE})"
echo "  End: \$(date)"
echo "================================================================="

# Report error count if error file exists
ERROR_FILE="${CHUNK_DIR}/chunk_\${CHUNK_ID}_errors.csv"
if [ -f "\${ERROR_FILE}" ]; then
    N_ERRORS=\$(( \$(wc -l < "\${ERROR_FILE}") - 1 ))  # minus header
    echo "  Errors: \${N_ERRORS} files (see \${ERROR_FILE})"
fi

exit \${EXIT_CODE}
SBATCH
)

echo "  Array job submitted: ${ARRAY_JOBID}"

# --- Step 3: Submit merge job (runs after array, even if some tasks fail) ---
echo ""
echo "[Step 3] Submitting merge job (afterany:${ARRAY_JOBID})..."

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
#SBATCH --output=${LOG_DIR}/merge_%j.out
#SBATCH --error=${LOG_DIR}/merge_%j.err

echo "================================================================="
echo "  Merge Feature Chunks"
echo "================================================================="
echo "Job ID:    \${SLURM_JOB_ID}"
echo "Node:      \$(hostname)"
echo "Start:     \$(date)"
echo "Expected:  ${ACTUAL_CHUNKS} chunks"
echo ""

cd ${PROJECT_DIR}

${PYTHON} -c "
import pandas as pd
from pathlib import Path
import sys
import time

chunk_dir = Path('${CHUNK_DIR}')
output_path = Path('${FINAL_OUTPUT}')
expected_chunks = ${ACTUAL_CHUNKS}

# Find all chunk parquets (not _part_ files, not _errors files)
chunks = sorted(chunk_dir.glob('chunk_[0-9][0-9][0-9].parquet'))
found = len(chunks)

print(f'Expected chunks: {expected_chunks}')
print(f'Found chunks:    {found}')

# Report missing chunks
if found < expected_chunks:
    found_ids = {int(c.stem.split('_')[1]) for c in chunks}
    missing = [i for i in range(expected_chunks) if i not in found_ids]
    print(f'WARNING: Missing {len(missing)} chunks: {missing}')
    print('Check error logs for these tasks.')

if found == 0:
    print('ERROR: No chunk files found! All tasks may have failed.')
    sys.exit(1)

# Read and merge
print()
t0 = time.time()
dfs = []
total_rows = 0
for cp in chunks:
    df = pd.read_parquet(cp)
    total_rows += len(df)
    dfs.append(df)
    size_mb = cp.stat().st_size / 1e6
    print(f'  {cp.name}: {len(df):>8,} rows  ({size_mb:.1f} MB)')

merged = pd.concat(dfs, ignore_index=True)

# Atomic write
tmp_path = output_path.with_suffix('.parquet.tmp')
merged.to_parquet(tmp_path, index=False, engine='pyarrow')
import os
os.rename(tmp_path, output_path)
elapsed = time.time() - t0

print()
print('=' * 60)
print(f'Merged: {len(merged):,} rows x {len(merged.columns)} columns')
print(f'Output: {output_path}')
print(f'Size:   {output_path.stat().st_size / 1e9:.2f} GB')
print(f'Time:   {elapsed:.1f}s')
print()

# Feature/info column summary
feat_cols = [c for c in merged.columns if not c.startswith('_')]
info_cols = [c for c in merged.columns if c.startswith('_')]
print(f'Feature columns: {len(feat_cols)}')
print(f'Info columns:    {len(info_cols)}')

# Module combinations
if '_modules' in merged.columns:
    print()
    print('Module combinations (top 10):')
    for mod, cnt in merged['_modules'].value_counts().head(10).items():
        print(f'  {mod}: {cnt:,}')

# Duplicate check
if '_jobid' in merged.columns:
    n_unique = merged['_jobid'].nunique()
    n_total = len(merged)
    if n_unique < n_total:
        n_dup = n_total - n_unique
        print(f'WARNING: {n_dup:,} duplicate job IDs ({n_dup/n_total*100:.2f}%)')
    else:
        print(f'OK: All {n_unique:,} job IDs are unique')

# NaN check
nan_cols = merged.columns[merged.isna().all()].tolist()
if nan_cols:
    print(f'WARNING: {len(nan_cols)} all-NaN columns: {nan_cols[:10]}')
else:
    print('OK: No all-NaN columns')

# Error file summary
error_files = sorted(chunk_dir.glob('chunk_*_errors.csv'))
total_errors = 0
for ef in error_files:
    n = sum(1 for _ in open(ef)) - 1  # minus header
    if n > 0:
        total_errors += n
if total_errors > 0:
    print(f'Total extraction errors across all tasks: {total_errors:,}')

# Success rate
print()
success_rate = len(merged) / (len(merged) + total_errors) * 100 if (len(merged) + total_errors) > 0 else 0
print(f'Overall success rate: {success_rate:.1f}%')
print(f'Chunks processed: {found}/{expected_chunks}')
"

echo ""
echo "================================================================="
echo "  Merge Done — \$(date)"
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
echo "Merge job:    ${MERGE_JOBID} (afterany — runs even if some tasks fail)"
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
