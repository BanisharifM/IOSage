"""Compatibility helpers for the maintained recommendation pipeline.

New callers use :mod:`src.ioprescriber.recommender` and
:mod:`src.ioprescriber.pipeline`. This module retains only the measured-KB
loader and the retrieval helper used by one analysis script.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.ioprescriber.contracts import (
    score_grounding,
    validate_kb_entry,
    validate_knowledge_base,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


def load_knowledge_base(kb_path=None):
    """Load and validate a schema 2 measured-evidence knowledge base."""
    path = Path(kb_path) if kb_path else (
        PROJECT_DIR / "data" / "knowledge_base" / "knowledge_base_full.json"
    )
    with open(path) as handle:
        document = json.load(handle)
    return validate_knowledge_base(document)


def retrieve_relevant_entries(kb_entries, detected_dims, darshan_signature, top_k=3):
    """Return measured entries ranked by label count and counter similarity."""
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    candidates = []
    for entry in kb_entries:
        validate_kb_entry(entry)
        shared = sorted(set(entry["bottleneck_labels"]) & set(detected_dims))
        if not shared:
            continue
        similarity = 0.0
        count = 0
        for name, query_value in darshan_signature.items():
            if name not in entry["darshan_signature"]:
                continue
            query = float(query_value or 0.0)
            evidence = float(entry["darshan_signature"][name] or 0.0)
            if query == 0.0 and evidence == 0.0:
                continue
            similarity += min(abs(query), abs(evidence)) / max(
                abs(query), abs(evidence), 1e-9
            )
            count += 1
        candidates.append({
            "entry": entry,
            "shared_labels": shared,
            "matched_dims": shared,
            "n_matched": len(shared),
            "similarity": similarity / count if count else 0.0,
        })
    candidates.sort(
        key=lambda item: (item["n_matched"], item["similarity"]), reverse=True
    )
    return candidates[:top_k]


def check_groundedness(response_text, kb_entries):
    """Apply the maintained typed grounding scorer to a JSON response."""
    parsed = json.loads(response_text)
    matches = [{"entry": entry} for entry in kb_entries]
    return score_grounding(parsed, matches)


def build_structured_prompt(*_args, **_kwargs):
    raise RuntimeError(
        "retired prompt builder; use src.ioprescriber.recommender.Recommender"
    )


def call_llm(*_args, **_kwargs):
    raise RuntimeError(
        "retired LLM client; use src.ioprescriber.recommender.Recommender"
    )


def recommend_for_sample(*_args, **_kwargs):
    raise RuntimeError(
        "retired recommendation path; use src.ioprescriber.pipeline.IOPrescriber"
    )


def main():
    raise SystemExit(
        "use python -m src.ioprescriber.pipeline with a final model bundle "
        "and a schema 2 measured-evidence knowledge base"
    )


if __name__ == "__main__":
    main()
