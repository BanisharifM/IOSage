#!/usr/bin/env python3
"""Recompute recommendation grounding from typed claims and measured KB fixes."""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.ioprescriber.contracts import score_grounding, validate_knowledge_base

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def iter_results(document):
    """Yield evaluation result objects from the supported container layouts."""
    if isinstance(document, list):
        yield from document
        return
    if isinstance(document, dict):
        for value in document.values():
            if not isinstance(value, list):
                raise ValueError("evaluation result mapping values must be arrays")
            yield from value
        return
    raise ValueError("evaluation results must be an array or model-to-array mapping")


def verify_document(results, knowledge_base):
    entries = validate_knowledge_base(knowledge_base)
    entries_by_id = {entry["entry_id"]: entry for entry in entries}
    checked = []
    absent = 0
    failures = []
    for index, result in enumerate(iter_results(results)):
        if not isinstance(result, dict):
            failures.append(f"result {index}: record is not an object")
            continue
        if "error" in result:
            failures.append(f"result {index}: evaluation error: {result['error']}")
            continue
        recommendation = result.get("recommendation")
        parsed = recommendation.get("parsed") if isinstance(recommendation, dict) else None
        if parsed is None:
            absent += 1
            continue
        retrieval = result.get("retrieval")
        if not isinstance(retrieval, dict) or not isinstance(retrieval.get("entries"), list):
            failures.append(f"result {index}: retrieval evidence is missing")
            continue
        matches = []
        for item in retrieval["entries"]:
            entry_id = item.get("entry_id") if isinstance(item, dict) else None
            if entry_id not in entries_by_id:
                failures.append(f"result {index}: unknown retrieved entry {entry_id!r}")
                continue
            matches.append({"entry": entries_by_id[entry_id]})
        try:
            score = score_grounding(parsed, matches)
        except ValueError as exc:
            failures.append(f"result {index}: {exc}")
            continue
        stored = recommendation.get("groundedness")
        if stored != score:
            failures.append(f"result {index}: stored grounding differs from recomputation")
        checked.append({
            "index": index,
            "workload": result.get("workload"),
            "model": result.get("model"),
            "score": score,
        })
    return {
        "schema_version": 1,
        "checked_results": len(checked),
        "absent_recommendations": absent,
        "failed_results": len(failures),
        "failures": failures,
        "results": checked,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--knowledge-base", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results_path = Path(args.results).resolve()
    kb_path = Path(args.knowledge_base).resolve()
    report = verify_document(json.loads(results_path.read_text()),
                             json.loads(kb_path.read_text()))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    logger.info("Checked %d results; %d absent; %d failed",
                report["checked_results"], report["absent_recommendations"],
                report["failed_results"])
    return 1 if report["failures"] or report["absent_recommendations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
