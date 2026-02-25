#!/bin/bash
# =============================================================================
# Verify Darshan Log Download and Extraction
# =============================================================================
# Checks completeness of the downloaded and extracted Darshan log collection.
#
# Usage:
#   bash scripts/verify_darshan_logs.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/Darshan_Logs"

echo "============================================"
echo "  Darshan Log Collection Verification"
echo "============================================"
echo "Directory: ${LOG_DIR}"
echo "Date: $(date)"
echo ""

# --- Check directory structure ---
echo "--- Directory Structure ---"
for year in 2024 2025 2026; do
    if [[ -d "${LOG_DIR}/${year}" ]]; then
        months=$(find "${LOG_DIR}/${year}" -mindepth 1 -maxdepth 1 -type d | wc -l)
        days=$(find "${LOG_DIR}/${year}" -mindepth 2 -maxdepth 2 -type d | wc -l)
        echo "  ${year}: ${months} months, ${days} day directories"
    else
        echo "  ${year}: NOT FOUND"
    fi
done
echo ""

# --- Count tarballs ---
echo "--- Tarballs ---"
TARBALL_COUNT=$(find "$LOG_DIR" -name "logs.tar.gz" | wc -l)
TARBALL_SIZE=$(find "$LOG_DIR" -name "logs.tar.gz" -exec du -cb {} + 2>/dev/null | tail -1 | cut -f1)
TARBALL_SIZE_GB=$(echo "scale=2; ${TARBALL_SIZE:-0} / 1073741824" | bc)
echo "  Total tarballs:   ${TARBALL_COUNT}"
echo "  Total size:       ${TARBALL_SIZE_GB} GB"
echo ""

# --- Count .darshan files ---
echo "--- Darshan Logs ---"
DARSHAN_COUNT=$(find "$LOG_DIR" -name "*.darshan" | wc -l)
DARSHAN_SIZE=$(find "$LOG_DIR" -name "*.darshan" -exec du -cb {} + 2>/dev/null | tail -1 | cut -f1)
DARSHAN_SIZE_GB=$(echo "scale=2; ${DARSHAN_SIZE:-0} / 1073741824" | bc)
echo "  Total .darshan files: ${DARSHAN_COUNT}"
echo "  Total size:           ${DARSHAN_SIZE_GB} GB"
echo ""

# --- Per-year breakdown ---
echo "--- Per-Year Breakdown ---"
for year in 2024 2025 2026; do
    if [[ -d "${LOG_DIR}/${year}" ]]; then
        count=$(find "${LOG_DIR}/${year}" -name "*.darshan" | wc -l)
        tarballs=$(find "${LOG_DIR}/${year}" -name "logs.tar.gz" | wc -l)
        size=$(du -sh "${LOG_DIR}/${year}" 2>/dev/null | cut -f1)
        echo "  ${year}: ${count} logs, ${tarballs} tarballs, ${size} total"
    fi
done
echo ""

# --- Check for empty day directories (missing tarballs) ---
echo "--- Completeness Check ---"
EMPTY_DAYS=0
DAYS_WITHOUT_DARSHAN=0
for day_dir in $(find "$LOG_DIR" -mindepth 3 -maxdepth 3 -type d 2>/dev/null | sort); do
    if [[ ! -f "${day_dir}/logs.tar.gz" ]]; then
        darshan_in_dir=$(find "$day_dir" -name "*.darshan" 2>/dev/null | wc -l)
        if [[ $darshan_in_dir -eq 0 ]]; then
            EMPTY_DAYS=$((EMPTY_DAYS + 1))
        fi
    else
        darshan_in_dir=$(find "$day_dir" -name "*.darshan" 2>/dev/null | wc -l)
        if [[ $darshan_in_dir -eq 0 ]]; then
            DAYS_WITHOUT_DARSHAN=$((DAYS_WITHOUT_DARSHAN + 1))
        fi
    fi
done
echo "  Empty day dirs (no tarball, no logs): ${EMPTY_DAYS}"
echo "  Days with tarball but no extracted logs: ${DAYS_WITHOUT_DARSHAN}"

if [[ $DAYS_WITHOUT_DARSHAN -gt 0 ]]; then
    echo "  WARNING: ${DAYS_WITHOUT_DARSHAN} days need extraction. Run:"
    echo "    bash scripts/unpack_darshan_logs.sh"
fi
echo ""

# --- Sample a few logs with PyDarshan ---
echo "--- Sample Log Validation ---"
SAMPLE_LOGS=$(find "$LOG_DIR" -name "*.darshan" 2>/dev/null | shuf -n 3)
if [[ -n "$SAMPLE_LOGS" ]]; then
    for log_file in $SAMPLE_LOGS; do
        echo "  Testing: ${log_file}"
        /sw/external/python/anaconda3/bin/python -c "
import darshan
try:
    r = darshan.DarshanReport('${log_file}', read_all=False)
    mods = list(r.modules.keys()) if hasattr(r, 'modules') else ['unknown']
    print(f'    OK: modules={mods}')
except Exception as e:
    print(f'    FAIL: {e}')
" 2>&1
    done
else
    echo "  No .darshan files found to test"
fi
echo ""

# --- Disk usage summary ---
echo "--- Disk Usage ---"
du -sh "${LOG_DIR}" 2>/dev/null
df -h /work/ | tail -1

echo ""
echo "============================================"
echo "  Verification Complete"
echo "============================================"
