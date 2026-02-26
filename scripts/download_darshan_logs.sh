#!/bin/bash
# =============================================================================
# Download Darshan Logs from Globus
# =============================================================================
# Downloads the ALCF Polaris Darshan Log Collection from Globus to Delta.
#
# Prerequisites:
#   1. globus login --no-local-server
#   2. globus session consent for ACCESS Delta (see below)
#
# Usage:
#   # Full download (all years):
#   bash scripts/download_darshan_logs.sh
#
#   # Download specific year:
#   bash scripts/download_darshan_logs.sh --year 2024
#
#   # Download specific month:
#   bash scripts/download_darshan_logs.sh --year 2025 --month 6
#
#   # Dry run (list what would be transferred):
#   bash scripts/download_darshan_logs.sh --dry-run
#
# Globus Consent Setup (run once before first transfer):
#   globus session consent \
#     'urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/7e936164-de58-4e3d-85da-21aa23c07169/data_access]' \
#     --no-local-server
# =============================================================================

set -euo pipefail

# --- Configuration ---
SOURCE_EP="f3b540be-4761-4e95-9703-8a4de7574036"  # ALCF Polaris Darshan Collection
DEST_EP="7e936164-de58-4e3d-85da-21aa23c07169"    # ACCESS Delta
DEST_BASE="/work/hdd/bdau/mbanisharifdehkordi/SC_2026/Darshan_Logs"

# --- Parse Arguments ---
YEAR=""
MONTH=""
DRY_RUN=false
SYNC=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --year) YEAR="$2"; shift 2 ;;
        --month) MONTH="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --sync) SYNC=true; shift ;;
        --dest-ep) DEST_EP="$2"; shift 2 ;;
        --dest-base) DEST_BASE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--year YYYY] [--month MM] [--dry-run] [--sync]"
            echo "  --year YYYY     Download only this year"
            echo "  --month MM      Download only this month (requires --year)"
            echo "  --dry-run       List files without transferring"
            echo "  --sync          Only transfer new/changed files"
            echo "  --dest-ep UUID  Override destination endpoint"
            echo "  --dest-base DIR Override destination base path"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Verify Login ---
echo "Checking Globus authentication..."
if ! globus whoami &>/dev/null; then
    echo "ERROR: Not logged in. Run: globus login --no-local-server"
    exit 1
fi
echo "Authenticated as: $(globus whoami)"

# --- Determine Source Path ---
if [[ -n "$YEAR" && -n "$MONTH" ]]; then
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

echo ""
echo "Transfer Configuration:"
echo "  Source:      ${SOURCE_EP}:${SOURCE_PATH}"
echo "  Destination: ${DEST_EP}:${DEST_PATH}"
echo "  Label:       ${LABEL}"
echo ""

# --- Dry Run: List source contents ---
if $DRY_RUN; then
    echo "DRY RUN: Listing source contents..."
    globus ls -l "${SOURCE_EP}:${SOURCE_PATH}" | head -50
    echo ""
    echo "To download, run without --dry-run"
    exit 0
fi

# --- Build Transfer Options ---
TRANSFER_OPTS=(
    --recursive
    --label "$LABEL"
    --preserve-timestamp
    --notify succeeded  # notify on success only
)

if $SYNC; then
    TRANSFER_OPTS+=(--sync-level size)
    echo "Sync mode: only transferring new/changed files"
fi

# --- Submit Transfer ---
echo "Submitting Globus transfer..."
TASK_OUTPUT=$(globus transfer \
    "${SOURCE_EP}:${SOURCE_PATH}" \
    "${DEST_EP}:${DEST_PATH}" \
    "${TRANSFER_OPTS[@]}" \
    --format json 2>&1)

TASK_ID=$(echo "$TASK_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['task_id'])" 2>/dev/null)

if [[ -z "$TASK_ID" ]]; then
    echo "ERROR: Transfer submission failed:"
    echo "$TASK_OUTPUT"
    echo ""
    echo "If you see a consent error, run:"
    echo "  globus session consent 'urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/${DEST_EP}/data_access]' --no-local-server"
    exit 1
fi

echo ""
echo "Transfer submitted successfully!"
echo "  Task ID: ${TASK_ID}"
echo ""
echo "Monitor with:"
echo "  globus task show ${TASK_ID}"
echo "  globus task wait ${TASK_ID} --polling-interval 60"
echo ""

# --- Wait for Transfer ---
echo "Waiting for transfer to complete (this may take hours for the full collection)..."
echo "You can safely Ctrl+C and check later with: globus task show ${TASK_ID}"
echo ""

globus task wait "${TASK_ID}" --polling-interval 60 --timeout 172800 || {
    echo "Transfer still in progress. Check status with:"
    echo "  globus task show ${TASK_ID}"
    exit 0
}

echo ""
echo "Transfer complete!"
globus task show "${TASK_ID}"
