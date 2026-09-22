#!/usr/bin/env python3
"""
Train the biquality detector and write an immutable run directory.

Usage:
    python scripts/train_biquality.py --run-id 2026-09-22_xgboost_w100
    python scripts/train_biquality.py --model lightgbm --seeds 42 --clean-weight 10 --run-id ...
    python scripts/train_biquality.py --feature-set raw --run-id ...            # no derived features
    python scripts/train_biquality.py --no-production --run-id ...             # benchmark dev only
    python scripts/train_biquality.py --hold-out-benchmark custom --run-id ... # leave one benchmark out

The run directory (``paths.runs_dir/<run-id>`` from the config) receives one
model bundle per seed, ``splits.npz`` and ``manifest.json`` (resolved config,
input hashes, git revision, development and test metrics per seed and their
mean and standard deviation). An existing run directory is never overwritten.
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from src.models.biquality import PROJECT_DIR as _PD, SUPPORTED_MODELS, load_config, train_run  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Train the biquality detector")
    parser.add_argument("--config", default=str(PROJECT_DIR / "configs" / "training_resubmission.yaml"))
    parser.add_argument("--model", default="xgboost", choices=SUPPORTED_MODELS)
    parser.add_argument("--seeds", type=int, nargs="+", help="default: biquality.seeds of the config")
    parser.add_argument("--clean-weight", type=float, help="default: biquality.clean_weight of the config")
    parser.add_argument("--run-id", required=True, help="name of the run directory under paths.runs_dir")
    parser.add_argument("--feature-set", default="full", choices=["full", "raw"])
    parser.add_argument("--no-production", action="store_true", help="train on benchmark dev rows only")
    parser.add_argument("--hold-out-benchmark", help="benchmark type removed from the dev rows")
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = args.seeds or config["biquality"]["seeds"]
    clean_weight = args.clean_weight if args.clean_weight is not None else config["biquality"]["clean_weight"]
    run_dir = _PD / config["paths"]["runs_dir"] / args.run_id
    train_run(config, args.model, seeds, clean_weight, run_dir, feature_set=args.feature_set,
              use_production=not args.no_production, hold_out_benchmark=args.hold_out_benchmark)
    return 0


if __name__ == "__main__":
    sys.exit(main())
