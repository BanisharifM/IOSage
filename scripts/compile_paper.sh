#!/bin/bash
# ==============================================================
# Compile SC 2026 Paper (LaTeX → PDF)
# ==============================================================
# Usage: bash scripts/compile_paper.sh
#
# Does: pdflatex → bibtex → pdflatex × 2
# Checks: compilation errors, missing references
# Cleans: removes build artifacts, keeps PDF
# ==============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
PAPER_DIR="${PROJECT_ROOT}/paper"
MAIN="main"

cd "${PAPER_DIR}"

echo "Compiling ${MAIN}.tex..."

# Pass 1: Initial compilation (may fail on first pass, that's normal)
pdflatex -interaction=nonstopmode "${MAIN}.tex" > /dev/null 2>&1 || true

# Pass 2: Bibliography
bibtex "${MAIN}" > /dev/null 2>&1 || true

# Pass 3-4: Resolve references
pdflatex -interaction=nonstopmode "${MAIN}.tex" > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode "${MAIN}.tex" > /dev/null 2>&1 || true

# Check for errors in log
if grep -q "^!" "${MAIN}.log"; then
    echo "ERROR: LaTeX compilation errors found:"
    grep "^!" "${MAIN}.log"
    exit 1
fi

# Check for undefined references
UNDEF=$(grep -c "LaTeX Warning.*undefined" "${MAIN}.log" 2>/dev/null || true)
UNDEF=${UNDEF:-0}
if [ "${UNDEF}" -gt 0 ]; then
    echo "WARNING: ${UNDEF} undefined reference(s)"
    grep "LaTeX Warning.*undefined" "${MAIN}.log"
fi

# Report PDF info
if [ -f "${MAIN}.pdf" ]; then
    PAGES=$(pdfinfo "${MAIN}.pdf" 2>/dev/null | grep Pages | awk '{print $2}' || echo "?")
    SIZE=$(ls -lh "${MAIN}.pdf" | awk '{print $5}')
    echo "OK: ${MAIN}.pdf (${PAGES} pages, ${SIZE})"
else
    echo "ERROR: PDF not generated"
    exit 1
fi

# Clean build artifacts
rm -f "${MAIN}.aux" "${MAIN}.log" "${MAIN}.bbl" "${MAIN}.blg" "${MAIN}.out" "${MAIN}.synctex.gz"

echo "Done."
