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
# Step 6: IOSage biquality training  [AD: T2]
# =============================================================================
# IOSage = XGBoost + biquality learning (91K heuristic + 201 GT, w=100).
# Trains all four model families (XGBoost, LightGBM, Random Forest, MLP) with
# biquality (Table III rows 1, 3, 4, 5). Reads benchmark splits from
# results/boost_experiment/new_splits/ (201 dev / 488 test, the paper's split).
if [ "$STEP" = 0 ] || [ "$STEP" = 6 ]; then
log "Step 6: IOSage biquality training (5 seeds, 4 model families)..."
N_SEEDS=5
[ "$QUICK" = true ] && N_SEEDS=1
python results/boost_experiment/scripts/train_biquality_boost.py \
    --model all --clean-weight 100 --n-seeds $N_SEEDS --save
log "  Also: GT-only XGBoost ablation (Table III row 2, baseline at 0.909)"
python results/boost_experiment/gt_only_xgboost_5seeds/train_gt_only_single_seed.py \
    --n-seeds $N_SEEDS || log "  (gt-only ablation skipped)"
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
# Reproduces Section VI-C (LLM quality on 488 traces, Table V LLM quality),
# Table V (n=8 recommendation ablation), and Tables VIII-IX (TraceBench
# cross-system). LLM cache (data/llm_cache/) is consulted automatically;
# live inference requires OPENROUTER_API_KEY.
if [ "$STEP" = 0 ] || [ "$STEP" = 8 ]; then
log "Step 8: LLM recommendation evaluation..."
N_RUNS=5
[ "$QUICK" = true ] && N_RUNS=1
log "  (a) Full 488-trace LLM quality (4 LLMs, Section VI-C)..."
for model in claude-sonnet-4 gpt-4o gpt-4.1-mini llama-3.1-70b-instruct; do
    python results/boost_experiment/iosage_full488_no_leak/run_iosage_no_leak.py \
        --model "$model" \
        --output-dir "results/boost_experiment/iosage_full488_no_leak/output_${model}" \
        || log "  (${model} skipped — likely missing API key, cache may not cover this run)"
done
log "  (b) Recommendation ablation (Table V, n=8 workloads)..."
python results/boost_experiment/scripts/run_fair_ablation.py
log "  (c) TraceBench cross-system evaluation (Tables VIII-IX)..."
python results/boost_experiment/scripts/run_tracebench_full_evaluation.py \
    || log "  (TraceBench skipped — requires data/external/tracebench/)"
fi

# =============================================================================
# Step 9: Generate all paper figures and tables  [AD: T5]
# =============================================================================
# Produces Figures 2-5 and the LaTeX tables that read from the JSON outputs
# of steps 6-8.
if [ "$STEP" = 0 ] || [ "$STEP" = 9 ]; then
log "Step 9: Generating paper figures and tables..."
python scripts/generate_results_figures.py
python scripts/generate_labeling_figures.py
python scripts/generate_evaluation_figures.py
if [ "$QUICK" = false ]; then
    python scripts/generate_paper_figures.py
fi
log "  Computing final aggregated metrics..."
python results/boost_experiment/scripts/compute_final_metrics.py \
    || log "  (compute_final_metrics skipped)"
fi

# =============================================================================
# Step 10: Auxiliary checks (latency, ML ablations, weight sensitivity)
# =============================================================================
if [ "$STEP" = 0 ] || [ "$STEP" = 10 ]; then
log "Step 10: Auxiliary measurements..."
python results/boost_experiment/scripts/measure_latency.py \
    || log "  (latency measurement skipped)"
python results/boost_experiment/scripts/run_ml_ablations.py \
    || log "  (ML ablations skipped)"
python results/boost_experiment/scripts/run_weight_sensitivity.py \
    || log "  (weight sensitivity skipped)"
fi

# =============================================================================
# Summary
# =============================================================================
log ""
log "============================================================"
log "REPRODUCTION COMPLETE"
log "============================================================"
log ""
log "Key results (DIOBench 488-sample test set, 5 seeds):"
log "  IOSage:                         Micro-F1=0.929+/-0.003"
log "  XGBoost (GT-only ablation):     Micro-F1=0.909+/-0.001"
log "  LightGBM:                       Micro-F1=0.925"
log "  Random Forest:                  Micro-F1=0.916"
log "  MLP:                            Micro-F1=0.767"
log ""
log "External baselines:"
log "  Drishti:  Micro-F1=0.364 (IOSage 2.6x higher)"
log "  WisIO:    Micro-F1=0.320 (IOSage 2.9x higher)"
log "  IOAgent:  Micro-F1=0.331 (IOSage 2.8x higher)"
log ""
log "Figures saved to: paper/figures/"
log "Models saved to:  models/phase2/"
log "Results saved to: results/"
log "============================================================"
