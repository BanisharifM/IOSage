#!/usr/bin/env bash
# ==============================================================
# Sync paper/ to SC_2026_Paper repo (linked to Overleaf)
# ==============================================================
# Usage: bash scripts/sync_paper_to_overleaf.sh [commit_message]
#
# Copies LaTeX-relevant files from paper/ to a local clone of
# SC_2026_Paper, commits, and pushes. Overleaf then pulls from
# GitHub to see the updates.
#
# One-time setup:
#   git clone git@github.com:BanisharifM/SC_2026_Paper.git \
#       /work/hdd/bdau/mbanisharifdehkordi/SC_2026_Paper
# ==============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN_REPO="$(dirname "${SCRIPT_DIR}")"
PAPER_SRC="${MAIN_REPO}/paper"
PAPER_REPO="${MAIN_REPO}/../SC_2026_Paper"
COMMIT_MSG="${1:-Sync paper from SC_2026 main repo}"

# Check paper repo exists
if [ ! -d "${PAPER_REPO}/.git" ]; then
    echo "ERROR: Paper repo not found at ${PAPER_REPO}"
    echo "Run: git clone git@github.com:BanisharifM/SC_2026_Paper.git ${PAPER_REPO}"
    exit 1
fi

# Pull latest from Overleaf (avoid clobbering edits)
echo "Pulling latest from SC_2026_Paper..."
git -C "${PAPER_REPO}" pull origin main || git -C "${PAPER_REPO}" pull origin master || true

# Sync only LaTeX-relevant files using rsync
# --delete removes files in dest that are no longer in source
echo "Syncing files..."
rsync -av --delete \
    --include="main.tex" \
    --include="references.bib" \
    --include="IEEEtran.cls" \
    --include="IEEEtran.bst" \
    --include="README.md" \
    --include=".gitignore" \
    --include="sections/" \
    --include="sections/*.tex" \
    --include="figures/" \
    --include="figures/**" \
    --include="tables/" \
    --include="tables/*.tex" \
    --exclude="main.pdf" \
    --exclude="paper_materials.md" \
    --exclude="stats.json" \
    --exclude="*.aux" \
    --exclude="*.log" \
    --exclude="*.out" \
    --exclude="*.bbl" \
    --exclude="*.blg" \
    --exclude="*.synctex.gz" \
    --exclude="*" \
    "${PAPER_SRC}/" "${PAPER_REPO}/"

# Commit and push if there are changes
cd "${PAPER_REPO}"
git add -A
if git diff --cached --quiet; then
    echo "No changes to sync."
else
    git commit -m "${COMMIT_MSG}"
    # Try main first, fall back to master
    git push origin main 2>/dev/null || git push origin master
    echo "Synced and pushed to SC_2026_Paper."
fi
