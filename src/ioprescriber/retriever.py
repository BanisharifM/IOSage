"""
IOPrescriber Step 3: Knowledge Base Retrieval (RAG).

Retrieves the most relevant benchmark entries from the KB for detected
bottleneck types. Returns entries WITH source code snippets and verified
fix patterns.

Retrieval strategy:
  1. Filter by bottleneck type (exact match)
  2. Rank by Darshan feature similarity (cosine on key counters)
  3. Return top-K with full source code + fix patterns

Input: detected dimensions + Darshan features
Output: list of KB entries with source code and fixes
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


class Retriever:
    """Benchmark Knowledge Base retriever with source code."""

    def __init__(self, kb_path=None, top_k=3):
        kb_path = kb_path or PROJECT_DIR / "data" / "knowledge_base" / "knowledge_base_full.json"
        with open(kb_path) as f:
            self.kb = json.load(f)
        self.top_k = top_k

        # Build index by dimension for fast filtering
        self.by_dim = {}
        for entry in self.kb:
            for dim in entry["bottleneck_labels"]:
                if dim not in self.by_dim:
                    self.by_dim[dim] = []
                self.by_dim[dim].append(entry)

        logger.info("Retriever loaded: %d KB entries, %d dimensions indexed",
                    len(self.kb), len(self.by_dim))

    def _feature_similarity(self, query_sig, entry_sig):
        """Compute feature similarity between query and KB entry."""
        if not query_sig or not entry_sig:
            return 0.0

        similarity = 0.0
        n_common = 0
        for key in query_sig:
            if key in entry_sig:
                qv = float(query_sig[key]) if query_sig[key] else 0
                ev = float(entry_sig[key]) if entry_sig[key] else 0
                if qv != 0 or ev != 0:
                    ratio = min(abs(qv), abs(ev)) / max(abs(qv), abs(ev), 1e-9)
                    similarity += ratio
                    n_common += 1

        return similarity / max(n_common, 1)

    def retrieve(self, detected_dims, darshan_features, top_k=None):
        """Retrieve matching KB entries with source code evidence.

        Args:
            detected_dims: list of detected bottleneck dimension names
            darshan_features: dict of Darshan feature values from the job
            top_k: number of entries to return (default: self.top_k)

        Returns:
            list of dicts with: entry, similarity, matched_dims
        """
        top_k = top_k or self.top_k

        # Build query signature from key features
        key_features = [
            "avg_write_size", "avg_read_size", "small_io_ratio",
            "seq_write_ratio", "seq_read_ratio", "metadata_time_ratio",
            "collective_ratio", "total_bw_mb_s", "fsync_ratio",
            "nprocs", "POSIX_BYTES_WRITTEN", "POSIX_BYTES_READ",
            "POSIX_WRITES", "POSIX_READS", "POSIX_FSYNCS",
            "MPIIO_COLL_WRITES", "MPIIO_INDEP_WRITES",
        ]
        query_sig = {}
        for f in key_features:
            val = darshan_features.get(f, 0)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                query_sig[f] = float(val)

        # Collect candidates from all detected dimensions
        candidates = []
        seen_ids = set()

        for dim in detected_dims:
            if dim not in self.by_dim:
                continue
            for entry in self.by_dim[dim]:
                if entry["entry_id"] in seen_ids:
                    continue
                seen_ids.add(entry["entry_id"])

                shared_labels = set(entry["bottleneck_labels"]) & set(detected_dims)
                similarity = self._feature_similarity(
                    query_sig, entry.get("darshan_signature", {})
                )

                candidates.append({
                    "entry": entry,
                    "similarity": round(similarity, 4),
                    "matched_dims": list(shared_labels),
                    "n_matched": len(shared_labels),
                })

        # Sort: most matched dims first, then highest similarity
        candidates.sort(key=lambda x: (x["n_matched"], x["similarity"]), reverse=True)

        results = candidates[:top_k]

        logger.info("Retrieved %d/%d KB entries for dims=%s",
                    len(results), len(candidates), detected_dims)

        return results

    def get_fix_for_dimension(self, dimension):
        """Get the canonical fix pattern for a specific dimension."""
        entries = self.by_dim.get(dimension, [])
        if not entries:
            return None

        # Return the first entry's fix (they all share the same fix pattern)
        fixes = entries[0].get("fixes", [])
        for fix in fixes:
            if fix.get("dimension") == dimension:
                return fix
        return fixes[0] if fixes else None
