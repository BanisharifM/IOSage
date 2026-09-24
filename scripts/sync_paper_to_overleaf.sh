#!/usr/bin/env bash
# Validate and review active-paper changes before a separate commit and push.

set -euo pipefail

if [ "$#" -ne 0 ]; then
    echo "Usage: bash scripts/sync_paper_to_overleaf.sh" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PAPER_REPO="$PROJECT_ROOT/papers/IPDPS_2027"

if [ ! -d "$PAPER_REPO/.git" ]; then
    echo "ERROR: active paper repository not found at $PAPER_REPO" >&2
    exit 1
fi

PAPER_DIR="$PAPER_REPO" bash "$PROJECT_ROOT/scripts/compile_paper.sh"

git -C "$PAPER_REPO" status --short
git -C "$PAPER_REPO" diff --check
git -C "$PAPER_REPO" diff -- \
    main.tex references.bib IEEEtran.cls IEEEtran.bst \
    sections figures tables

echo "Review the diff above, then commit and push from $PAPER_REPO when authorized."
