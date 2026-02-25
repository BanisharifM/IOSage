#!/bin/bash
# =============================================================================
# Unpack Darshan Log Tarballs
# =============================================================================
# Extracts all logs.tar.gz files from the Darshan log collection.
# Each day directory (YYYY/MM/DD/) contains a logs.tar.gz that gets
# extracted in-place, producing .darshan files alongside the tarball.
#
# Usage:
#   # Unpack everything:
#   bash scripts/unpack_darshan_logs.sh
#
#   # Unpack specific year:
#   bash scripts/unpack_darshan_logs.sh --dir Darshan_Logs/2024
#
#   # Unpack with more workers:
#   bash scripts/unpack_darshan_logs.sh --workers 32
#
#   # Delete tarballs after successful extraction:
#   bash scripts/unpack_darshan_logs.sh --delete-tarballs
#
#   # Count stats only (no extraction):
#   bash scripts/unpack_darshan_logs.sh --stats-only
# =============================================================================

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/Darshan_Logs"
WORKERS=16
DELETE_TARBALLS=false
STATS_ONLY=false
LOG_FILE="${PROJECT_DIR}/logs/unpack_darshan.log"

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --dir) LOG_DIR="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --delete-tarballs) DELETE_TARBALLS=true; shift ;;
        --stats-only) STATS_ONLY=true; shift ;;
        --log-file) LOG_FILE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--dir DIR] [--workers N] [--delete-tarballs] [--stats-only]"
            echo "  --dir DIR          Directory to search for logs.tar.gz (default: Darshan_Logs)"
            echo "  --workers N        Parallel extraction workers (default: 16)"
            echo "  --delete-tarballs  Delete tarballs after successful extraction"
            echo "  --stats-only       Print statistics only, do not extract"
            echo "  --log-file FILE    Path to log file"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# --- Validate ---
if [[ ! -d "$LOG_DIR" ]]; then
    log "ERROR: Directory not found: $LOG_DIR"
    exit 1
fi

# --- Find tarballs ---
log "Searching for logs.tar.gz in ${LOG_DIR}..."
TARBALLS=()
while IFS= read -r -d '' f; do
    TARBALLS+=("$f")
done < <(find "$LOG_DIR" -name "logs.tar.gz" -print0 | sort -z)

TOTAL=${#TARBALLS[@]}
log "Found ${TOTAL} tarballs"

if [[ $TOTAL -eq 0 ]]; then
    log "No tarballs found. Check if download completed."
    exit 0
fi

# --- Statistics ---
if $STATS_ONLY; then
    TOTAL_SIZE=0
    EXTRACTED=0
    NOT_EXTRACTED=0

    for tarball in "${TARBALLS[@]}"; do
        dir=$(dirname "$tarball")
        size=$(stat -c%s "$tarball" 2>/dev/null || echo 0)
        TOTAL_SIZE=$((TOTAL_SIZE + size))

        # Check if already extracted (at least one .darshan file exists)
        darshan_count=$(find "$dir" -maxdepth 1 -name "*.darshan" 2>/dev/null | head -1 | wc -l)
        if [[ $darshan_count -gt 0 ]]; then
            EXTRACTED=$((EXTRACTED + 1))
        else
            NOT_EXTRACTED=$((NOT_EXTRACTED + 1))
        fi
    done

    SIZE_GB=$(echo "scale=2; $TOTAL_SIZE / 1073741824" | bc)
    log "Statistics:"
    log "  Total tarballs:     ${TOTAL}"
    log "  Already extracted:  ${EXTRACTED}"
    log "  Need extraction:    ${NOT_EXTRACTED}"
    log "  Total tarball size: ${SIZE_GB} GB"
    exit 0
fi

# --- Extract ---
log "Starting extraction with ${WORKERS} parallel workers..."
START_TIME=$(date +%s)
SUCCESS=0
FAIL=0
SKIP=0

extract_one() {
    local tarball="$1"
    local dir
    dir=$(dirname "$tarball")

    # Skip if already extracted
    local darshan_count
    darshan_count=$(find "$dir" -maxdepth 1 -name "*.darshan" 2>/dev/null | head -1 | wc -l)
    if [[ $darshan_count -gt 0 ]]; then
        echo "SKIP:${tarball}"
        return 0
    fi

    # Extract
    if cd "$dir" && tar -xzf logs.tar.gz 2>/dev/null; then
        echo "OK:${tarball}"
    else
        echo "FAIL:${tarball}"
        return 1
    fi
}

export -f extract_one

# Run parallel extraction
printf '%s\0' "${TARBALLS[@]}" | xargs -0 -P "$WORKERS" -I{} bash -c 'extract_one "$@"' _ {} 2>&1 | while IFS= read -r line; do
    case "$line" in
        OK:*)   SUCCESS=$((SUCCESS + 1)) ;;
        SKIP:*) SKIP=$((SKIP + 1)) ;;
        FAIL:*) FAIL=$((FAIL + 1)); log "FAILED: ${line#FAIL:}" ;;
    esac

    DONE=$((SUCCESS + SKIP + FAIL))
    if (( DONE % 50 == 0 || DONE == TOTAL )); then
        ELAPSED=$(( $(date +%s) - START_TIME ))
        RATE=$(echo "scale=1; $DONE / ($ELAPSED + 1)" | bc)
        ETA=$(echo "scale=0; ($TOTAL - $DONE) / ($RATE + 0.001)" | bc 2>/dev/null || echo "?")
        log "Progress: ${DONE}/${TOTAL} (extracted=${SUCCESS}, skipped=${SKIP}, failed=${FAIL}) | ${RATE}/s | ETA: ${ETA}s"
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
log "Extraction complete in ${ELAPSED}s"

# --- Optional: Delete tarballs ---
if $DELETE_TARBALLS; then
    log "Deleting tarballs to free space..."
    for tarball in "${TARBALLS[@]}"; do
        rm -f "$tarball"
    done
    log "Deleted ${TOTAL} tarballs"
fi

# --- Count final .darshan files ---
log "Counting .darshan files..."
DARSHAN_COUNT=$(find "$LOG_DIR" -name "*.darshan" | wc -l)
log "Total .darshan files: ${DARSHAN_COUNT}"

# --- Summary ---
log ""
log "============================================"
log "  Unpack Summary"
log "============================================"
log "  Tarballs processed: ${TOTAL}"
log "  .darshan files:     ${DARSHAN_COUNT}"
log "  Time:               ${ELAPSED}s"
log "  Log directory:      ${LOG_DIR}"
log "============================================"
