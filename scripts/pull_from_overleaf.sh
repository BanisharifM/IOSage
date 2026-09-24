#!/usr/bin/env bash
# Update the active IPDPS 2027 paper repository from its main remote.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PAPER_REPO="$PROJECT_ROOT/papers/IPDPS_2027"

if [ ! -d "$PAPER_REPO/.git" ]; then
    echo "ERROR: active paper repository not found at $PAPER_REPO" >&2
    exit 1
fi
if [ -n "$(git -C "$PAPER_REPO" status --porcelain)" ]; then
    echo "ERROR: active paper repository has local changes; review them before pulling" >&2
    git -C "$PAPER_REPO" status --short >&2
    exit 1
fi

git -C "$PAPER_REPO" pull --ff-only origin main
git -C "$PAPER_REPO" status --short
echo "Active paper repository updated: $PAPER_REPO"
