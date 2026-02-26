#!/usr/bin/env bash
# ==============================================================
# Pull Overleaf changes back into SC_2026/paper/
# ==============================================================
# Usage: bash scripts/pull_from_overleaf.sh
#
# After editing on Overleaf:
# 1. In Overleaf: Menu > GitHub > Push to GitHub
# 2. Run this script to pull .tex/.bib changes back
#
# NOTE: Only .tex and .bib files are pulled back. Figures and
# tables are always authored in the main repo.
# ==============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN_REPO="$(dirname "${SCRIPT_DIR}")"
PAPER_SRC="${MAIN_REPO}/paper"
PAPER_REPO="${MAIN_REPO}/../SC_2026_Paper"

# Check paper repo exists
if [ ! -d "${PAPER_REPO}/.git" ]; then
    echo "ERROR: Paper repo not found at ${PAPER_REPO}"
    echo "Run: git clone git@github.com:BanisharifM/SC_2026_Paper.git ${PAPER_REPO}"
    exit 1
fi

# Pull latest from GitHub (Overleaf pushes here)
echo "Pulling latest from SC_2026_Paper..."
git -C "${PAPER_REPO}" pull origin main 2>/dev/null || git -C "${PAPER_REPO}" pull origin master

# Copy .tex and .bib files back (figures/tables stay as-is)
echo "Copying .tex and .bib files to paper/..."
rsync -av \
    --include="*.tex" \
    --include="*.bib" \
    --include="sections/" \
    --include="sections/*.tex" \
    --include="tables/" \
    --include="tables/*.tex" \
    --exclude="*" \
    "${PAPER_REPO}/" "${PAPER_SRC}/"

echo "Done. Review changes in paper/ before committing to SC_2026."
echo "  git diff paper/"
