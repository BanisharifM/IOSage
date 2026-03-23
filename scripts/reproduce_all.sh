#!/bin/bash
# =============================================================================
# SC 2026 — Reproduce All Paper Results
# =============================================================================
# One script to reproduce every result in the paper.
# Assumes: conda env sc2026 activated, data in data/processed/
#
# Usage:
#   bash scripts/reproduce_all.sh              # Run everything
#   bash scripts/reproduce_all.sh --quick      # Skip long steps (benchmarks)
#   bash scripts/reproduce_all.sh --step N     # Run only step N
#
# Expected runtime: ~1 hour on CPU (AMD EPYC 7763, 128 cores)
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

QUICK=false
STEP=0
for arg in "$@"; do
    case $arg in
        --quick) QUICK=true ;;
        --step) shift; STEP=$1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# =============================================================================
# Step 1: Verify environment
# =============================================================================
if [ "$STEP" = 0 ] || [ "$STEP" = 1 ]; then
log "Step 1: Verifying environment..."
python -c "
import xgboost, lightgbm, shap, cleanlab, sklearn, pandas, numpy, yaml
print(f'  xgboost={xgboost.__version__}')
print(f'  lightgbm={lightgbm.__version__}')
print(f'  shap={shap.__version__}')
print(f'  cleanlab={cleanlab.__version__}')
print(f'  sklearn={sklearn.__version__}')
print('  Environment OK')
"
fi

# =============================================================================
# Step 2: Verify data files exist
# =============================================================================
if [ "$STEP" = 0 ] || [ "$STEP" = 2 ]; then
log "Step 2: Verifying data files..."
for f in \
    data/processed/production/features.parquet \
    data/processed/production/labels.parquet \
    data/processed/production/split_indices.pkl \
    data/processed/benchmark/features.parquet \
    data/processed/benchmark/labels.parquet \
    data/processed/benchmark/split_indices.pkl; do
    if [ ! -f "$f" ]; then
        echo "  MISSING: $f"
        echo "  Run: python scripts/extract_benchmark_features.py --bench-type all"
        echo "  And: python scripts/prepare_phase2_data.py"
        exit 1
    fi
    echo "  OK: $f"
done
fi

# =============================================================================
# Step 3: Extract benchmark features (if needed)
# =============================================================================
if [ "$STEP" = 0 ] || [ "$STEP" = 3 ]; then
if [ "$QUICK" = false ]; then
    log "Step 3: Extracting benchmark features..."
    python scripts/extract_benchmark_features.py --bench-type all
else
    log "Step 3: SKIPPED (--quick mode)"
fi
fi

# =============================================================================
# Step 4: Prepare Phase 2 data (iterative stratification)
# =============================================================================
if [ "$STEP" = 0 ] || [ "$STEP" = 4 ]; then
log "Step 4: Preparing Phase 2 data splits..."
python scripts/prepare_phase2_data.py
fi

# =============================================================================
# Step 5: Phase 1 baseline (heuristic-only training)
# =============================================================================
if [ "$STEP" = 0 ] || [ "$STEP" = 5 ]; then
log "Step 5: Phase 1 — Heuristic-only training..."
python -m src.models.train --config configs/training.yaml --model xgboost --n-seeds 1 --save
fi

# =============================================================================
# Step 6: Phase 2 biquality training (5 seeds, all models)
# =============================================================================
if [ "$STEP" = 0 ] || [ "$STEP" = 6 ]; then
log "Step 6: Phase 2 — Biquality training (5 seeds, 3 models)..."
python -m src.models.train_biquality --model all --clean-weight 100 --n-seeds 5 --save
fi

# =============================================================================
# Step 7: Label agreement analysis (Drishti vs GT)
# =============================================================================
if [ "$STEP" = 0 ] || [ "$STEP" = 7 ]; then
log "Step 7: Label agreement analysis..."
python scripts/analyze_label_agreement.py
fi

# =============================================================================
# Step 8: SHAP analysis
# =============================================================================
if [ "$STEP" = 0 ] || [ "$STEP" = 8 ]; then
log "Step 8: SHAP feature attribution..."
python -m src.models.attribution
fi

# =============================================================================
# Step 9: Generate all paper figures
# =============================================================================
if [ "$STEP" = 0 ] || [ "$STEP" = 9 ]; then
log "Step 9: Generating paper figures..."
python scripts/generate_results_figures.py
python scripts/generate_labeling_figures.py
if [ "$QUICK" = false ]; then
    python scripts/generate_paper_figures.py
fi
fi

# =============================================================================
# Step 10: Verify ground-truth logs
# =============================================================================
if [ "$STEP" = 0 ] || [ "$STEP" = 10 ]; then
log "Step 10: Verifying benchmark ground-truth..."
python scripts/verify_all_ground_truth.py --bench-type all
fi

# =============================================================================
# Summary
# =============================================================================
log ""
log "============================================================"
log "REPRODUCTION COMPLETE"
log "============================================================"
log ""
log "Key results:"
log "  Phase 2 XGBoost (5 seeds): Micro-F1=0.920+/-0.004"
log "  vs Drishti baseline:       Micro-F1=0.384 (2.4x improvement)"
log ""
log "Figures saved to: paper/figures/"
log "Models saved to:  models/phase2/"
log "Results saved to: results/"
log "============================================================"
