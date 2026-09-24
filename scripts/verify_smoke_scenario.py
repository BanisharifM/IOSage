#!/usr/bin/env python3
"""Verify one smoke-test Darshan log against its declared benchmark label."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.data.benchmark_verify import verify_benchmark_log  # noqa: E402
from src.data.label_rules import DIMENSION_NAMES  # noqa: E402
from src.data.parse_darshan import parse_darshan_log  # noqa: E402
from src.data.preprocessing import engineer_one, load_preprocessing_config  # noqa: E402


def parse_labels(text):
    labels = dict.fromkeys(DIMENSION_NAMES, 0)
    for item in text.split(","):
        name, separator, value = item.partition("=")
        if not separator or name not in labels or value != "1":
            raise ValueError(f"invalid label item: {item}")
        labels[name] = 1
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    log_path = Path(args.log).resolve()
    labels = parse_labels(args.labels)
    parsed = parse_darshan_log(log_path, strict=True)
    config = load_preprocessing_config()
    features = engineer_one(parsed, config=config)
    passed, report = verify_benchmark_log(features, labels, config['cleaning'])
    result = {"log": str(log_path), "labels": labels, **report}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
