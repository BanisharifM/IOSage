#!/usr/bin/env bash
# Verify the declared Darshan collection against every archive member.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/Darshan_Logs"
YEARS="2024,2025,2026"
IOSAGE_ENV="${IOSAGE_ENV:-/work/nvme/bdau/mbanisharifdehkordi/envs/iosage}"
PYTHON="${PYTHON:-${IOSAGE_ENV}/bin/python}"

need_value() {
    [[ $# -ge 2 && -n "$2" ]] || { echo "ERROR: $1 needs a value" >&2; exit 2; }
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir) need_value "$@"; LOG_DIR=$2; shift 2 ;;
        --years) need_value "$@"; YEARS=$2; shift 2 ;;
        --python) need_value "$@"; PYTHON=$2; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--dir DIR] [--years YYYY[,YYYY...]] [--python PATH]"
            exit 0
            ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ -x "$PYTHON" ]] || { echo "ERROR: Python is not executable: $PYTHON" >&2; exit 1; }
[[ -d "$LOG_DIR" ]] || { echo "ERROR: collection directory not found: $LOG_DIR" >&2; exit 1; }
IFS=',' read -r -a EXPECTED_YEARS <<< "$YEARS"
[[ "${#EXPECTED_YEARS[@]}" -gt 0 ]] || { echo "ERROR: --years is empty" >&2; exit 2; }
for year in "${EXPECTED_YEARS[@]}"; do
    [[ "$year" =~ ^[0-9]{4}$ ]] || { echo "ERROR: invalid year: $year" >&2; exit 2; }
    [[ -d "$LOG_DIR/$year" ]] || { echo "ERROR: expected year directory missing: $LOG_DIR/$year" >&2; exit 1; }
    count=$(find "$LOG_DIR/$year" -type f -name logs.tar.gz | wc -l)
    [[ "$count" -gt 0 ]] || { echo "ERROR: no archives found for year $year" >&2; exit 1; }
done

TARBALLS=()
while IFS= read -r -d '' path; do TARBALLS+=("$path"); done \
    < <(find "$LOG_DIR" -type f -name logs.tar.gz -print0 | sort -z)
[[ "${#TARBALLS[@]}" -gt 0 ]] || { echo "ERROR: collection has no archives" >&2; exit 1; }

MEMBERS=0
for tarball in "${TARBALLS[@]}"; do
    set +e
    output=$("$PYTHON" "$SCRIPT_DIR/archive_inventory.py" "$tarball" \
        --output-dir "$(dirname "$tarball")" 2>&1)
    rc=$?
    set -e
    [[ "$rc" -eq 0 ]] || { echo "ERROR: archive inventory failed for $tarball: $output" >&2; exit 1; }
    count=$("$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["members"])' <<< "$output")
    MEMBERS=$((MEMBERS + count))
done

DARSHAN_COUNT=$(find "$LOG_DIR" -type f -name '*.darshan' | wc -l)
[[ "$DARSHAN_COUNT" -eq "$MEMBERS" ]] || {
    echo "ERROR: collection count differs from archive inventory: files=$DARSHAN_COUNT members=$MEMBERS" >&2
    exit 1
}

mapfile -t SAMPLE_LOGS < <(find "$LOG_DIR" -type f -name '*.darshan' | sort | head -n 3)
[[ "${#SAMPLE_LOGS[@]}" -gt 0 ]] || { echo "ERROR: no Darshan files available for parser validation" >&2; exit 1; }
for log_file in "${SAMPLE_LOGS[@]}"; do
    "$PYTHON" - "$log_file" <<'PY'
import sys
import darshan
path = sys.argv[1]
report = darshan.DarshanReport(path, read_all=False)
if not report.modules:
    raise ValueError(f"Darshan log has no modules: {path}")
PY
done

printf 'Verification passed: years=%s archives=%d members=%d files=%d parsed=%d\n' \
    "$YEARS" "${#TARBALLS[@]}" "$MEMBERS" "$DARSHAN_COUNT" "${#SAMPLE_LOGS[@]}"
