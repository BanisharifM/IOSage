"""Run the ML detector on one or more Darshan logs and print/save the predictions.

Usage:
    python scripts/detect_darshan_logs.py --model <model.pkl> [--threshold 0.3]
        [--output results/detect.json] log1.darshan [log2.darshan ...]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.ioprescriber.detector import Detector  # noqa: E402

logger = logging.getLogger("detect_darshan_logs")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", help="Darshan log files")
    parser.add_argument("--model", required=True, help="Path to the classifier pickle")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection threshold (default 0.3)")
    parser.add_argument("--output", help="Write predictions as JSON to this path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    detector = Detector(model_path=args.model, threshold=args.threshold)
    results = []
    for log in args.logs:
        predictions, detected, _ = detector.detect_from_darshan(log)
        probs = {k: round(float(v), 3) for k, v in sorted(predictions.items(), key=lambda kv: -kv[1])}
        logger.info("%s -> detected=%s probs=%s", Path(log).name, detected, probs)
        results.append({"log": str(log), "detected": list(detected), "probabilities": probs})

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"model": args.model, "threshold": args.threshold, "results": results}, f, indent=2)
        logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
