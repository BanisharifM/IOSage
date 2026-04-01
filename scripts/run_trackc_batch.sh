#!/bin/bash
# Iterative batch sweep + ablation runner
# Run from project root: bash scripts/run_trackc_batch.sh
set -e

source .env
PYTHON=/projects/bdau/envs/sc2026/bin/python
OUT=results/iterative

# ---- Phase 2: Remaining sweeps ----
echo "=== Phase 2: Sweep runs ==="

# ior_small_direct (3 models x 5 runs)
for MODEL in claude-sonnet gpt-4o llama-70b; do
    SLUG=$(echo $MODEL | tr '-' '_' | tr '.' '_')
    echo "--- ior_small_direct x $MODEL ---"
    $PYTHON -m src.llm.iterative_optimizer \
        --workload ior_small_direct --model $MODEL --max-iterations 5 --n-runs 5 \
        --output $OUT/sweep_ior_small_direct_${SLUG}.json 2>&1 | tee $OUT/sweep_ior_small_direct_${SLUG}.log
done

# ior_random_access (3 models x 5 runs)
for MODEL in claude-sonnet gpt-4o llama-70b; do
    SLUG=$(echo $MODEL | tr '-' '_' | tr '.' '_')
    echo "--- ior_random_access x $MODEL ---"
    $PYTHON -m src.llm.iterative_optimizer \
        --workload ior_random_access --model $MODEL --max-iterations 5 --n-runs 5 \
        --output $OUT/sweep_ior_random_access_${SLUG}.json 2>&1 | tee $OUT/sweep_ior_random_access_${SLUG}.log
done

# ior_misaligned (3 models x 5 runs)
for MODEL in claude-sonnet gpt-4o llama-70b; do
    SLUG=$(echo $MODEL | tr '-' '_' | tr '.' '_')
    echo "--- ior_misaligned x $MODEL ---"
    $PYTHON -m src.llm.iterative_optimizer \
        --workload ior_misaligned --model $MODEL --max-iterations 5 --n-runs 5 \
        --output $OUT/sweep_ior_misaligned_${SLUG}.json 2>&1 | tee $OUT/sweep_ior_misaligned_${SLUG}.log
done

# ---- Phase 3: Ablation studies (on 3 representative workloads with Claude) ----
echo "=== Phase 3: Ablation studies ==="

for W in ior_small_posix ior_fsync_heavy ior_small_direct; do
    echo "--- Ablation: no_ml on $W ---"
    $PYTHON -m src.llm.iterative_optimizer \
        --workload $W --model claude-sonnet --max-iterations 5 --n-runs 3 --no-ml \
        --output $OUT/ablation_no_ml_${W}.json 2>&1 | tee $OUT/ablation_no_ml_${W}.log

    echo "--- Ablation: no_kb on $W ---"
    $PYTHON -m src.llm.iterative_optimizer \
        --workload $W --model claude-sonnet --max-iterations 5 --n-runs 3 --no-kb \
        --output $OUT/ablation_no_kb_${W}.json 2>&1 | tee $OUT/ablation_no_kb_${W}.log

    echo "--- Ablation: no_shap on $W ---"
    $PYTHON -m src.llm.iterative_optimizer \
        --workload $W --model claude-sonnet --max-iterations 5 --n-runs 3 --no-shap \
        --output $OUT/ablation_no_shap_${W}.json 2>&1 | tee $OUT/ablation_no_shap_${W}.log

    echo "--- Ablation: single_shot on $W ---"
    $PYTHON -m src.llm.iterative_optimizer \
        --workload $W --model claude-sonnet --max-iterations 1 --n-runs 3 \
        --output $OUT/ablation_single_shot_${W}.json 2>&1 | tee $OUT/ablation_single_shot_${W}.log

    echo "--- Ablation: no_feedback on $W ---"
    $PYTHON -m src.llm.iterative_optimizer \
        --workload $W --model claude-sonnet --max-iterations 5 --n-runs 3 --no-feedback \
        --output $OUT/ablation_no_feedback_${W}.json 2>&1 | tee $OUT/ablation_no_feedback_${W}.log
done

# ---- Phase 4: Aggregate results ----
echo "=== Phase 4: Aggregate results ==="
$PYTHON scripts/aggregate_trackc_results.py

echo "=== Iterative batch complete ==="
