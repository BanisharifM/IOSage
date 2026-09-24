#!/usr/bin/env bash
# Submit one bounded Globus transfer from the Polaris Darshan collection.

set -euo pipefail
SOURCE_EP="f3b540be-4761-4e95-9703-8a4de7574036"
DEST_EP="7e936164-de58-4e3d-85da-21aa23c07169"
DEST_BASE="/work/hdd/bdau/mbanisharifdehkordi/IOSage/Darshan_Logs"
YEAR=""
MONTH=""
ALL=false
DRY_RUN=false
SYNC=false
PYTHON="${PYTHON:-python3}"

need_value() {
    [[ $# -ge 2 && -n "$2" ]] || { echo "ERROR: $1 needs a value" >&2; exit 2; }
}
usage() {
    echo "Usage: $0 (--all | --year YYYY [--month MM]) [--dry-run] [--sync]"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) ALL=true; shift ;;
        --year) need_value "$@"; YEAR=$2; shift 2 ;;
        --month) need_value "$@"; MONTH=$2; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --sync) SYNC=true; shift ;;
        --dest-ep) need_value "$@"; DEST_EP=$2; shift 2 ;;
        --dest-base) need_value "$@"; DEST_BASE=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if "$ALL" && [[ -n "$YEAR$MONTH" ]]; then
    echo "ERROR: --all cannot be combined with --year or --month" >&2
    exit 2
fi
if ! "$ALL" && [[ -z "$YEAR" ]]; then
    echo "ERROR: select --all or --year; --month requires --year" >&2
    exit 2
fi
if [[ -n "$YEAR" && ! "$YEAR" =~ ^(2024|2025|2026)$ ]]; then
    echo "ERROR: --year must be 2024, 2025, or 2026" >&2
    exit 2
fi
if [[ -n "$MONTH" ]]; then
    [[ "$MONTH" =~ ^(0?[1-9]|1[0-2])$ ]] || { echo "ERROR: --month must be 1 through 12" >&2; exit 2; }
    MONTH=$((10#$MONTH))
fi
[[ "$DEST_BASE" == /* ]] || { echo "ERROR: --dest-base must be an absolute path" >&2; exit 2; }
command -v globus >/dev/null || { echo "ERROR: globus CLI not found" >&2; exit 1; }
command -v "$PYTHON" >/dev/null || { echo "ERROR: Python not found: $PYTHON" >&2; exit 1; }

if [[ -n "$MONTH" ]]; then
    SOURCE_PATH="/${YEAR}/${MONTH}/"
    DEST_PATH="${DEST_BASE}/${YEAR}/${MONTH}/"
    LABEL="Darshan-${YEAR}-${MONTH}"
elif [[ -n "$YEAR" ]]; then
    SOURCE_PATH="/${YEAR}/"
    DEST_PATH="${DEST_BASE}/${YEAR}/"
    LABEL="Darshan-${YEAR}"
else
    SOURCE_PATH="/"
    DEST_PATH="${DEST_BASE}/"
    LABEL="Darshan-Full-Collection"
fi

if ! globus whoami >/dev/null 2>&1; then
    echo "ERROR: Globus authentication is required" >&2
    exit 4
fi
printf 'Source: %s:%s\nDestination: %s:%s\nLabel: %s\n' \
    "$SOURCE_EP" "$SOURCE_PATH" "$DEST_EP" "$DEST_PATH" "$LABEL"

if "$DRY_RUN"; then
    listing=$(globus ls -l "${SOURCE_EP}:${SOURCE_PATH}")
    sed -n '1,50p' <<< "$listing"
    exit 0
fi

TRANSFER_OPTS=(--recursive --label "$LABEL" --preserve-timestamp --notify succeeded)
"$SYNC" && TRANSFER_OPTS+=(--sync-level size)
set +e
TASK_OUTPUT=$(globus transfer "${SOURCE_EP}:${SOURCE_PATH}" "${DEST_EP}:${DEST_PATH}" \
    "${TRANSFER_OPTS[@]}" --format json 2>&1)
SUBMIT_RC=$?
set -e
if [[ "$SUBMIT_RC" -ne 0 ]]; then
    echo "ERROR: transfer submission failed: $TASK_OUTPUT" >&2
    exit "$SUBMIT_RC"
fi
set +e
TASK_ID=$("$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["task_id"])' <<< "$TASK_OUTPUT" 2>/dev/null)
PARSE_RC=$?
set -e
[[ "$PARSE_RC" -eq 0 && -n "$TASK_ID" ]] || { echo "ERROR: transfer response has no task_id" >&2; exit 1; }
printf 'Task ID: %s\n' "$TASK_ID"

set +e
globus task wait "$TASK_ID" --polling-interval 60 --timeout 172800 --timeout-exit-code 50
WAIT_RC=$?
TASK_STATE=$(globus task show "$TASK_ID" --format json 2>&1)
SHOW_RC=$?
set -e
[[ "$SHOW_RC" -eq 0 ]] || { echo "ERROR: cannot read terminal task state: $TASK_STATE" >&2; exit "$SHOW_RC"; }
STATUS=$("$PYTHON" -c 'import json,sys; print(json.load(sys.stdin).get("status", ""))' <<< "$TASK_STATE")
case "$STATUS" in
    SUCCEEDED)
        [[ "$WAIT_RC" -eq 0 ]] || echo "Task succeeded after wait returned status $WAIT_RC"
        echo "Transfer succeeded: $TASK_ID"
        ;;
    ACTIVE)
        echo "Transfer remains active after the wait timeout: $TASK_ID" >&2
        exit 50
        ;;
    FAILED)
        echo "ERROR: transfer failed: $TASK_ID" >&2
        exit 1
        ;;
    *)
        echo "ERROR: unexpected terminal task status '$STATUS' for $TASK_ID" >&2
        exit 1
        ;;
esac
