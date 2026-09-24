#!/bin/bash
# ==============================================================
# Compile IOSage Paper (LaTeX → PDF)
# ==============================================================
# Usage: bash scripts/compile_paper.sh
#        PAPER_DIR=/path/to/paper bash scripts/compile_paper.sh
#
# Does: pdflatex → bibtex → pdflatex × 2
# Checks: compilation errors, missing references
# Archives build artifacts under .codex-trash and keeps the new PDF
# ==============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
# Default: the active paper. Override with PAPER_DIR (absolute or root-relative).
PAPER_DIR="${PAPER_DIR:-papers/IPDPS_2027}"
case "${PAPER_DIR}" in /*) ;; *) PAPER_DIR="${PROJECT_ROOT}/${PAPER_DIR}" ;; esac
MAIN="main"

PAPER_DIR="$(cd "${PAPER_DIR}" && pwd -P)"
FROZEN_DIR="$(cd "${PROJECT_ROOT}/papers/SC_2026" && pwd -P)"
if [ "${PAPER_DIR}" = "${FROZEN_DIR}" ]; then
    echo "ERROR: refusing to compile inside the frozen SC 2026 repository" >&2
    exit 2
fi

BUILD_TAG="$(date -u +%Y%m%dT%H%M%SZ)_$$"
ARCHIVE_DIR="${PROJECT_ROOT}/.codex-trash/paper_build/${BUILD_TAG}"
mkdir -p "${ARCHIVE_DIR}/before" "${ARCHIVE_DIR}/after"
MARKER="${ARCHIVE_DIR}/build_started"
touch "${MARKER}"

archive_files() {
    local destination=$1
    shift
    local path
    for path in "$@"; do
        if [ -e "${path}" ]; then
            mv "${path}" "${destination}/"
        fi
    done
}

cd "${PAPER_DIR}"

BUILD_FILES=("${MAIN}.aux" "${MAIN}.log" "${MAIN}.bbl" "${MAIN}.blg"
             "${MAIN}.out" "${MAIN}.synctex.gz")
archive_files "${ARCHIVE_DIR}/before" "${MAIN}.pdf" "${BUILD_FILES[@]}"

on_exit() {
    local status=$?
    if [ "${status}" -ne 0 ]; then
        archive_files "${ARCHIVE_DIR}/after" "${MAIN}.pdf" "${BUILD_FILES[@]}"
        echo "Build evidence archived at ${ARCHIVE_DIR}" >&2
    fi
}
trap on_exit EXIT

echo "Compiling ${MAIN}.tex..."

pdflatex -halt-on-error -interaction=nonstopmode "${MAIN}.tex"

bibtex "${MAIN}"

pdflatex -halt-on-error -interaction=nonstopmode "${MAIN}.tex"
pdflatex -halt-on-error -interaction=nonstopmode "${MAIN}.tex"

# Check for errors in log
if grep -q "^!" "${MAIN}.log"; then
    echo "ERROR: LaTeX compilation errors found:"
    grep "^!" "${MAIN}.log"
    exit 1
fi

if grep -Eq 'LaTeX Warning: (Reference|Citation).*undefined|There were undefined references' "${MAIN}.log"; then
    echo "ERROR: unresolved references or citations" >&2
    grep -E 'LaTeX Warning: (Reference|Citation).*undefined|There were undefined references' "${MAIN}.log" >&2
    exit 1
fi

# Report PDF info
if [ -s "${MAIN}.pdf" ] && [ "${MAIN}.pdf" -nt "${MARKER}" ]; then
    PAGES=$(pdfinfo "${MAIN}.pdf" | awk '/^Pages:/ {print $2}')
    [ -n "${PAGES}" ] || { echo "ERROR: PDF page count is missing" >&2; exit 1; }
    SIZE=$(ls -lh "${MAIN}.pdf" | awk '{print $5}')
    echo "OK: ${MAIN}.pdf (${PAGES} pages, ${SIZE})"
else
    echo "ERROR: PDF not generated"
    exit 1
fi

archive_files "${ARCHIVE_DIR}/after" "${BUILD_FILES[@]}"
trap - EXIT

echo "Build evidence archived at ${ARCHIVE_DIR}"
