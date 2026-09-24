#!/usr/bin/env bash
# Run the maintained resubmission pipeline with explicit artifact gates.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/work/nvme/bdau/mbanisharifdehkordi/envs/iosage/bin/python}"
QUICK=false
STEP=0
RUN_ID="repro_$(date -u +%Y%m%dT%H%M%SZ)"

usage() {
    echo "Usage: bash scripts/reproduce_all.sh [--quick] [--step 1-10] [--run-id NAME]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --quick)
            QUICK=true
            shift
            ;;
        --step)
            [ "$#" -ge 2 ] || { echo "ERROR: --step needs a value" >&2; exit 2; }
            STEP=$2
            shift 2
            ;;
        --run-id)
            [ "$#" -ge 2 ] || { echo "ERROR: --run-id needs a value" >&2; exit 2; }
            RUN_ID=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$STEP" in
    0|1|2|3|4|5|6|7|8|9|10) ;;
    *) echo "ERROR: --step must be an integer from 1 through 10" >&2; exit 2 ;;
esac
case "$RUN_ID" in
    ""|*/*|.|..) echo "ERROR: --run-id must be one directory name" >&2; exit 2 ;;
esac

RUN_ROOT="$PROJECT_DIR/results/resubmission/reproduction/$RUN_ID"
if [ -e "$RUN_ROOT" ]; then
    echo "ERROR: reproduction run already exists: $RUN_ROOT" >&2
    exit 1
fi
mkdir -p "$RUN_ROOT"
RUN_LOG="$RUN_ROOT/stages.tsv"
printf 'step\tstatus\tartifact\n' > "$RUN_LOG"

NTHREADS="${NTHREADS:-8}"
export OMP_NUM_THREADS="$NTHREADS"
export OPENBLAS_NUM_THREADS="$NTHREADS"
export MKL_NUM_THREADS="$NTHREADS"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="$RUN_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"

selected() { [ "$STEP" -eq 0 ] || [ "$STEP" -eq "$1" ]; }
require_file() { [ -s "$1" ] || { echo "ERROR: required file missing or empty: $1" >&2; exit 1; }; }
record() { printf '%s\tpassed\t%s\n' "$1" "$2" >> "$RUN_LOG"; }

if selected 1; then
    "$PYTHON_BIN" - <<'PY'
import cleanlab, lightgbm, numpy, pandas, shap, sklearn, xgboost, yaml
print("environment imports passed")
PY
    record 1 "$PYTHON_BIN"
fi

if selected 2; then
    require_file data/processed/resubmission/production/raw_features.parquet
    require_file data/benchmark_labels/manifest.csv
    [ -d data/benchmark_logs ] || { echo "ERROR: benchmark log directory is missing" >&2; exit 1; }
    record 2 data/processed/resubmission/production/raw_features.parquet
fi

MANIFEST="${LABEL_MANIFEST:-$RUN_ROOT/label_manifest.csv}"
VERIFY_REPORT="${VERIFICATION_REPORT:-$RUN_ROOT/verification.csv}"
BENCH_OUTPUT="${BENCH_DATA_DIR:-$RUN_ROOT/data/benchmark}"
PROD_OUTPUT="${PROD_DATA_DIR:-$RUN_ROOT/data/production}"
if selected 3; then
    "$PYTHON_BIN" scripts/build_label_manifest.py --output "$MANIFEST"
    require_file "$MANIFEST"
    require_file "$MANIFEST.manifest.json"
    "$PYTHON_BIN" scripts/verify_all_ground_truth.py --manifest "$MANIFEST" --report "$VERIFY_REPORT"
    require_file "$VERIFY_REPORT"
    record 3 "$VERIFY_REPORT"
fi

if selected 4; then
    require_file "$MANIFEST"
    require_file "$VERIFY_REPORT"
    "$PYTHON_BIN" scripts/extract_benchmark_features.py \
        --manifest "$MANIFEST" --verification-report "$VERIFY_REPORT" \
        --output-dir "$BENCH_OUTPUT" --bench-type all
    require_file "$BENCH_OUTPUT/features.parquet"
    require_file "$BENCH_OUTPUT/labels.parquet"
    require_file "$BENCH_OUTPUT/dataset_manifest.json"
    record 4 "$BENCH_OUTPUT/dataset_manifest.json"
fi

if selected 5; then
    "$PYTHON_BIN" scripts/run_preprocessing.py \
        --input data/processed/resubmission/production/raw_features.parquet \
        --output-dir "$PROD_OUTPUT"
    "$PYTHON_BIN" -m src.data.drishti_labeling \
        --features "$PROD_OUTPUT/features.parquet" \
        --output "$PROD_OUTPUT/labels.parquet"
    require_file "$PROD_OUTPUT/features.parquet"
    require_file "$PROD_OUTPUT/labels.parquet"
    require_file "$PROD_OUTPUT/split_indices.pkl"
    require_file "$PROD_OUTPUT/preprocessing_manifest.json"
    require_file "$PROD_OUTPUT/labels.parquet.manifest.json"
    record 5 "$PROD_OUTPUT/labels.parquet.manifest.json"
fi

TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training_resubmission.yaml}"
RUN_TRAINING_CONFIG="$RUN_ROOT/training_config.yaml"
TRAIN_RUN_DIR="$RUN_ROOT/training/$RUN_ID"
if selected 6; then
    require_file "$PROD_OUTPUT/features.parquet"
    require_file "$PROD_OUTPUT/labels.parquet"
    require_file "$PROD_OUTPUT/split_indices.pkl"
    require_file "$PROD_OUTPUT/preprocessing_manifest.json"
    require_file "$PROD_OUTPUT/labels.parquet.manifest.json"
    require_file "$BENCH_OUTPUT/features.parquet"
    require_file "$BENCH_OUTPUT/labels.parquet"
    require_file "$BENCH_OUTPUT/dataset_manifest.json"
    "$PYTHON_BIN" - "$TRAINING_CONFIG" "$RUN_TRAINING_CONFIG" \
        "$PROD_OUTPUT" "$BENCH_OUTPUT" "$RUN_ROOT/training" <<'PY'
import sys
from pathlib import Path
import yaml

source, target, production, benchmark, runs = map(Path, sys.argv[1:])
with source.open() as handle:
    config = yaml.safe_load(handle)
config["paths"].update({
    "production_features": str((production / "features.parquet").resolve()),
    "production_labels": str((production / "labels.parquet").resolve()),
    "production_splits": str((production / "split_indices.pkl").resolve()),
    "benchmark_features": str((benchmark / "features.parquet").resolve()),
    "benchmark_labels": str((benchmark / "labels.parquet").resolve()),
    "runs_dir": str(runs.resolve()),
})
config["reproduction_source_config"] = str(source.resolve())
with target.open("x") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
    seeds=(42 123 456 789 1024)
    if [ "$QUICK" = true ]; then seeds=(42); fi
    "$PYTHON_BIN" scripts/train_biquality.py \
        --config "$RUN_TRAINING_CONFIG" --model xgboost --seeds "${seeds[@]}" \
        --run-id "$RUN_ID" --final-evaluation
    require_file "$TRAIN_RUN_DIR/manifest.json"
    record 6 "$TRAIN_RUN_DIR/manifest.json"
fi

MODEL_BUNDLE="${MODEL_BUNDLE:-$TRAIN_RUN_DIR/xgboost_w100_seed42.pkl}"
if selected 7; then
    require_file "$MODEL_BUNDLE"
    SHAP_OUTPUT="$RUN_ROOT/shap"
    "$PYTHON_BIN" -m src.models.attribution --bundle "$MODEL_BUNDLE" --output-dir "$SHAP_OUTPUT"
    require_file "$SHAP_OUTPUT/domain_validation.json"
    require_file "$SHAP_OUTPUT/shap_values.pkl"
    record 7 "$SHAP_OUTPUT/domain_validation.json"
fi

if selected 8; then
    require_file "$MODEL_BUNDLE"
    [ -n "${KNOWLEDGE_BASE:-}" ] || { echo "ERROR: KNOWLEDGE_BASE is required for step 8" >&2; exit 2; }
    require_file "$KNOWLEDGE_BASE"
    runs=5
    if [ "$QUICK" = true ]; then runs=1; fi
    "$PYTHON_BIN" scripts/run_llm_evaluation.py \
        --model-bundle "$MODEL_BUNDLE" --knowledge-base "$KNOWLEDGE_BASE" \
        --n-runs "$runs" --output-dir "$RUN_ROOT/llm_evaluation"
    "$PYTHON_BIN" scripts/run_fair_ablation.py \
        --model-bundle "$MODEL_BUNDLE" --knowledge-base "$KNOWLEDGE_BASE" \
        --output-dir "$RUN_ROOT/ablation"
    "$PYTHON_BIN" scripts/run_tracebench_full_evaluation.py \
        --model-bundle "$MODEL_BUNDLE" --knowledge-base "$KNOWLEDGE_BASE" \
        --output-dir "$RUN_ROOT/tracebench"
    require_file "$RUN_ROOT/llm_evaluation/evaluation_summary.json"
    require_file "$RUN_ROOT/llm_evaluation/evaluation_manifest.json"
    require_file "$RUN_ROOT/ablation/fair_ablation_summary.json"
    require_file "$RUN_ROOT/ablation/fair_ablation_manifest.json"
    require_file "$RUN_ROOT/tracebench/tracebench_full_evaluation.json"
    record 8 "$RUN_ROOT/llm_evaluation/evaluation_summary.json"
fi

if selected 9; then
    echo "ERROR: paper figure generation is blocked until the generators consume one validated result manifest; see REPRO-004" >&2
    exit 3
fi

if selected 10; then
    [ -n "${ITERATIVE_RESULTS_DIR:-}" ] || { echo "ERROR: ITERATIVE_RESULTS_DIR is required for step 10" >&2; exit 2; }
    "$PYTHON_BIN" scripts/aggregate_trackc_results.py \
        --results-dir "$ITERATIVE_RESULTS_DIR" \
        --output "$RUN_ROOT/iterative_summary.json"
    require_file "$RUN_ROOT/iterative_summary.json"
    record 10 "$RUN_ROOT/iterative_summary.json"
fi

find "$RUN_ROOT" -type f ! -name artifacts.sha256 -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/artifacts.sha256"
require_file "$RUN_ROOT/artifacts.sha256"
echo "Requested reproduction steps completed. Evidence: $RUN_ROOT"
