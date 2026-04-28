#!/bin/bash
# =============================================================================
# IOSage — Reproduce All Paper Results
# =============================================================================
# One script to reproduce every result in the paper.
# Assumes: conda env sc2026 activated, data in data/processed/
#
# Usage:
#   bash scripts/reproduce_all.sh              # Run everything
#   bash scripts/reproduce_all.sh --quick      # Skip long steps (benchmarks)
#   bash scripts/reproduce_all.sh --step N     # Run only step N
#
# Step numbering matches the AD appendix (appendix_ad.pdf):
#   T1 (--step 1-5): feature extraction + preprocessing
#   T2 (--step 6):   ML training (biquality)
#   T3 (--step 7):   SHAP feature attribution
#   T4 (--step 8):   LLM recommendation evaluation
#   T5 (--step 9):   Figure and table generation
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
# Step 5: Heuristic labeling + Phase 1 baseline (heuristic-only training)
# =============================================================================
if [ "$STEP" = 0 ] || [ "$STEP" = 5 ]; then
log "Step 5: Heuristic labeling + Phase 1 baseline..."
python -m src.models.train --config configs/training.yaml --model xgboost --n-seeds 1 --save
fi

# =============================================================================
# Step 6: Phase 2 biquality training (5 seeds, all models)  [AD: T2]
# =============================================================================
if [ "$STEP" = 0 ] || [ "$STEP" = 6 ]; then
log "Step 6: Phase 2 — Biquality training (5 seeds, 4 model families)..."
python -m src.models.train_biquality --model all --clean-weight 100 --n-seeds 5 --save
fi

# =============================================================================
# Step 7: SHAP feature attribution  [AD: T3]
# =============================================================================
if [ "$STEP" = 0 ] || [ "$STEP" = 7 ]; then
log "Step 7: SHAP feature attribution..."
python -m src.models.attribution
fi

# =============================================================================
# Step 8: LLM recommendation evaluation  [AD: T4]
# =============================================================================
# Reproduces Section VI-C (LLM quality on 488 traces) and Table V
# (recommendation ablation, n=8). LLM cache (data/llm_cache/) is consulted
# automatically when present; live inference requires OPENROUTER_API_KEY.
if [ "$STEP" = 0 ] || [ "$STEP" = 8 ]; then
log "Step 8: LLM recommendation evaluation..."
if [ "$QUICK" = true ]; then
    log "  Quick mode: 1 run per workload, using cached outputs in data/llm_cache/"
    python scripts/run_llm_evaluation.py --n-runs 1
    python scripts/run_fair_ablation.py
else
    log "  Full mode: 5 runs per workload (live inference if cache miss)"
    python scripts/run_llm_evaluation.py --n-runs 5
    python scripts/run_fair_ablation.py
fi
fi

# =============================================================================
# Step 9: Generate all paper figures and tables  [AD: T5]
# =============================================================================
if [ "$STEP" = 0 ] || [ "$STEP" = 9 ]; then
log "Step 9: Generating paper figures and tables..."
python scripts/generate_results_figures.py
python scripts/generate_labeling_figures.py
if [ "$QUICK" = false ]; then
    python scripts/generate_paper_figures.py
fi
fi

# =============================================================================
# Step 10: Auxiliary checks (label agreement, ground-truth verification)
# =============================================================================
if [ "$STEP" = 0 ] || [ "$STEP" = 10 ]; then
log "Step 10: Auxiliary checks..."
python scripts/analyze_label_agreement.py
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
log "  Phase 2 XGBoost (5 seeds): Micro-F1=0.929+/-0.003"
log "  vs Drishti baseline:       Micro-F1=0.364 (2.6x improvement)"
log "  vs WisIO baseline:         Micro-F1=0.320 (2.9x improvement)"
log "  vs IOAgent baseline:       Micro-F1=0.331 (2.8x improvement)"
log ""
log "Figures saved to: paper/figures/"
log "Models saved to:  models/phase2/"
log "Results saved to: results/"
log "============================================================"
