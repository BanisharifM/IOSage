#!/bin/bash
# Generate 5 SLURM scripts (one per seed) from slurm_template.sh and submit them.
# Each job runs ~5-15 minutes; all 5 run in parallel for shortest wall time.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS=(42 123 456 789 1024)

mkdir -p split_slurms logs output

for seed in "${SEEDS[@]}"; do
    out="split_slurms/slurm_seed${seed}.sh"
    sed "s/__SEED__/${seed}/g" slurm_template.sh > "$out"
    chmod +x "$out"
    echo "Submitting seed=${seed}..."
    sbatch "$out"
done

echo ""
echo "All 5 jobs submitted. Watch with: squeue -u \$USER -n gt_only_seed_42,gt_only_seed_123,gt_only_seed_456,gt_only_seed_789,gt_only_seed_1024"
echo "After all complete: python aggregate_5seeds.py"
