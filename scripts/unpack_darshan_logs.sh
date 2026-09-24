#!/usr/bin/env bash
# Extract each logs.tar.gz through a verified member inventory.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/Darshan_Logs"
WORKERS=16
STATS_ONLY=false
LOG_FILE="${PROJECT_DIR}/logs/unpack_darshan.log"
IOSAGE_ENV="${IOSAGE_ENV:-/work/nvme/bdau/mbanisharifdehkordi/envs/iosage}"
PYTHON="${PYTHON:-${IOSAGE_ENV}/bin/python}"

need_value() {
    [[ $# -ge 2 && -n "$2" ]] || { echo "ERROR: $1 needs a value" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir) need_value "$@"; LOG_DIR="$2"; shift 2 ;;
        --workers) need_value "$@"; WORKERS="$2"; shift 2 ;;
        --stats-only) STATS_ONLY=true; shift ;;
        --log-file) need_value "$@"; LOG_FILE="$2"; shift 2 ;;
        --delete-tarballs)
            echo "ERROR: archive deletion is not supported; retain the source inventory" >&2
            exit 2
            ;;
        -h|--help)
            echo "Usage: $0 [--dir DIR] [--workers N] [--stats-only] [--log-file FILE]"
            exit 0
            ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --workers must be positive" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "ERROR: Python is not executable: $PYTHON" >&2; exit 1; }
[[ -d "$LOG_DIR" ]] || { echo "ERROR: directory not found: $LOG_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$LOG_FILE")" "${PROJECT_DIR}/.codex-trash/unpack_staging"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

TARBALLS=()
while IFS= read -r -d '' path; do TARBALLS+=("$path"); done \
    < <(find "$LOG_DIR" -type f -name logs.tar.gz -print0 | sort -z)
TOTAL=${#TARBALLS[@]}
[[ "$TOTAL" -gt 0 ]] || { log "ERROR: no logs.tar.gz archives under $LOG_DIR"; exit 1; }
log "Found ${TOTAL} archives under ${LOG_DIR}"

if "$STATS_ONLY"; then
    COMPLETE=0
    INCOMPLETE=0
    INVALID=0
    for tarball in "${TARBALLS[@]}"; do
        set +e
        "$PYTHON" "$SCRIPT_DIR/archive_inventory.py" "$tarball" --output-dir "$(dirname "$tarball")" \
            --staging-root "${PROJECT_DIR}/.codex-trash/unpack_staging" >/dev/null
        rc=$?
        set -e
        case "$rc" in
            0) COMPLETE=$((COMPLETE + 1)) ;;
            3) INCOMPLETE=$((INCOMPLETE + 1)) ;;
            *) INVALID=$((INVALID + 1)) ;;
        esac
    done
    log "Inventory: complete=${COMPLETE}, incomplete=${INCOMPLETE}, invalid=${INVALID}"
    [[ "$INVALID" -eq 0 ]]
    exit
fi

extract_one() {
    local tarball=$1 output rc
    set +e
    output=$("$PYTHON" "$SCRIPT_DIR/archive_inventory.py" "$tarball" \
        --output-dir "$(dirname "$tarball")" \
        --staging-root "${PROJECT_DIR}/.codex-trash/unpack_staging" --extract 2>&1)
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
        printf 'FAIL:%s:%s\n' "$tarball" "$output"
        return 1
    fi
    if [[ "$output" == *'"status": "extracted"'* ]]; then
        printf 'OK:%s:%s\n' "$tarball" "$output"
    else
        printf 'SKIP:%s:%s\n' "$tarball" "$output"
    fi
}
export -f extract_one
export PYTHON SCRIPT_DIR PROJECT_DIR

STATUS_FILE=$(mktemp "${PROJECT_DIR}/.codex-trash/unpack_status.XXXXXX")
START_TIME=$(date +%s)
set +e
printf '%s\0' "${TARBALLS[@]}" | xargs -0 -P "$WORKERS" -I{} bash -c 'extract_one "$1"' _ {} \
    >"$STATUS_FILE" 2>&1
XARGS_RC=$?
set -e
SUCCESS=$(awk -F: '$1=="OK" {n++} END {print n+0}' "$STATUS_FILE")
SKIP=$(awk -F: '$1=="SKIP" {n++} END {print n+0}' "$STATUS_FILE")
FAIL=$(awk -F: '$1=="FAIL" {n++} END {print n+0}' "$STATUS_FILE")
while IFS= read -r line; do
    [[ "$line" == FAIL:* ]] && log "$line"
done < "$STATUS_FILE"
DONE=$((SUCCESS + SKIP + FAIL))
ELAPSED=$(( $(date +%s) - START_TIME ))
log "Processed=${DONE}/${TOTAL}, extracted=${SUCCESS}, verified=${SKIP}, failed=${FAIL}, seconds=${ELAPSED}"
[[ "$XARGS_RC" -eq 0 && "$DONE" -eq "$TOTAL" && "$FAIL" -eq 0 ]] || exit 1
DARSHAN_COUNT=$(find "$LOG_DIR" -type f -name '*.darshan' | wc -l)
[[ "$DARSHAN_COUNT" -gt 0 ]] || { log "ERROR: no Darshan logs after extraction"; exit 1; }
log "Verified extraction complete: ${DARSHAN_COUNT} Darshan files"
