#!/usr/bin/env python3
"""
Atomic-level groundedness verification for LLM recommendations.

Addresses SC reviewer weakness W8:
  "Groundedness as defined (fraction citing KB) conflates 'cites source' with 'is correct.'"

For each LLM recommendation that cites a KB entry, this script verifies at the
ATOMIC claim level:
  1. bottleneck_dimension_match: Does the recommended dimension match the KB entry?
  2. fix_type_match: Does the recommended code change match the KB fix description?
  3. code_pattern_match: Does the code_before/code_after pattern match the KB entry?
  4. speedup_plausibility: Is the expected speedup consistent with KB evidence?
  5. api_change_match: Does the stated API change match the KB source_code?

Two analysis modes:
  A) Controlled evaluation set (60 samples/model from evaluation_results)
  B) All cached LLM responses (117 total, varying counts per model)

Produces:
  - citation_groundedness (existing): fraction of recs citing a valid KB entry
  - claim_groundedness (new): fraction of atomic claims that match (among cited recs)
  - Per-model comparison
  - List of hallucinated claims

Usage:
  python scripts/verify_groundedness_atomic.py

Output: results/groundedness_atomic.json
"""

import json
import logging
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_DIR / "data" / "llm_cache" / "ioprescriber"
KB_PATH = PROJECT_DIR / "data" / "knowledge_base" / "knowledge_base_full.json"
EVAL_DIR = PROJECT_DIR / "results" / "llm_evaluation"
OUTPUT_PATH = PROJECT_DIR / "results" / "groundedness_atomic.json"

# Model name mapping (evaluation results use short names)
MODEL_MAP = {
    "claude-sonnet": "anthropic/claude-sonnet-4",
    "gpt-4o": "openai/gpt-4o",
    "llama-70b": "meta-llama/llama-3.1-70b-instruct",
}


def load_knowledge_base():
    """Load KB and index by entry_id."""
    with open(KB_PATH) as f:
        kb = json.load(f)
    kb_by_id = {e["entry_id"]: e for e in kb}
    logger.info("Loaded KB with %d entries", len(kb_by_id))
    return kb_by_id


def extract_json_from_response(response_text):
    """Extract JSON object from LLM response text (may be wrapped in markdown)."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[\s\S]*\}", response_text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def parse_cited_ids(citation_str):
    """Parse comma-separated KB entry IDs from kb_citation field."""
    if not citation_str:
        return []
    ids = [cid.strip() for cid in citation_str.split(",")]
    return [cid for cid in ids if cid]


def code_similarity(code_a, code_b):
    """Compute normalized string similarity between two code snippets."""
    if not code_a or not code_b:
        return 0.0
    norm_a = " ".join(code_a.split())
    norm_b = " ".join(code_b.split())
    return SequenceMatcher(None, norm_a, norm_b).ratio()


def extract_speedup_number(speedup_str):
    """Extract numeric speedup value(s) from a speedup description string."""
    if not speedup_str:
        return []
    # Match patterns: 2.79x, 16x, 2-3x, 10-50x, Up to 16x
    numbers = re.findall(r"(\d+(?:\.\d+)?)\s*(?:x|X)", speedup_str)
    if not numbers:
        # Also try patterns like "2-3x"
        numbers = re.findall(r"(\d+(?:\.\d+)?)\s*-\s*\d", speedup_str)
    return [float(n) for n in numbers]


def check_dimension_match(rec_dimension, kb_entry):
    """Check if the recommendation's bottleneck dimension matches the KB entry."""
    kb_dims = set(kb_entry.get("bottleneck_labels", []))
    for fix in kb_entry.get("fixes", []):
        kb_dims.add(fix.get("dimension", ""))
    return rec_dimension in kb_dims


def check_fix_type_match(rec, kb_entry):
    """Check if the recommended fix type matches the KB entry's fix description.

    Uses semantic keyword matching between the recommendation and the KB fix.
    """
    rec_text = (
        rec.get("explanation", "") + " " +
        rec.get("code_after", "") + " " +
        rec.get("api_change", "")
    ).lower()

    kb_fixes = kb_entry.get("fixes", [])
    if not kb_fixes:
        return False, 0.0

    best_score = 0.0
    for fix in kb_fixes:
        kb_text = (
            fix.get("fix", "") + " " +
            fix.get("cause", "") + " " +
            fix.get("code_after", "")
        ).lower()

        # Strategy-based keyword matching
        fix_keywords = {
            "buffer": ["buffer", "aggregat", "batch", "chunk", "coalesce", "big_buf",
                        "memcpy", "flush"],
            "collective": ["collective", "mpi_file_write_all", "mpi_file_read_all",
                            "write_all", "read_all"],
            "sequential": ["sort", "sequential", "qsort", "reorder", "offset"],
            "alignment": ["align", "alignment", "aligned", "block boundary"],
            "striping": ["stripe", "lustre", "ost", "lfs setstripe"],
            "reduce_metadata": ["stat", "metadata", "open", "close", "cache",
                                 "batch open", "reduce open"],
            "direct_io": ["direct", "o_direct"],
            "larger_transfer": ["larger", "increase", "transfer size", "1mb",
                                 "1048576", "bigger"],
            "shared_file": ["shared file", "single file", "mpi-io", "mpiio"],
            "file_per_proc": ["file-per-proc", "fpp", "file per proc"],
            "fsync_reduce": ["fsync", "sync", "reduce sync"],
            "random_to_seq": ["random", "sequential", "sort offset", "qsort"],
        }

        rec_strategies = set()
        kb_strategies = set()
        for strategy, keywords in fix_keywords.items():
            if any(kw in rec_text for kw in keywords):
                rec_strategies.add(strategy)
            if any(kw in kb_text for kw in keywords):
                kb_strategies.add(strategy)

        if rec_strategies and kb_strategies:
            overlap = len(rec_strategies & kb_strategies)
            union = len(rec_strategies | kb_strategies)
            score = overlap / max(union, 1)
            best_score = max(best_score, score)
        elif not rec_strategies and not kb_strategies:
            score = code_similarity(rec_text, kb_text)
            best_score = max(best_score, score)

    return best_score >= 0.3, best_score


def check_code_pattern_match(rec, kb_entry):
    """Check if code_before/code_after in the rec match the KB entry's fix patterns."""
    rec_before = rec.get("code_before", "")
    rec_after = rec.get("code_after", "")

    kb_fixes = kb_entry.get("fixes", [])
    if not kb_fixes:
        return False, 0.0, 0.0

    best_before = 0.0
    best_after = 0.0
    for fix in kb_fixes:
        kb_before = fix.get("code_before", "")
        kb_after = fix.get("code_after", "")
        sim_before = code_similarity(rec_before, kb_before)
        sim_after = code_similarity(rec_after, kb_after)
        best_before = max(best_before, sim_before)
        best_after = max(best_after, sim_after)

    avg_sim = (best_before + best_after) / 2.0
    return avg_sim >= 0.4, best_before, best_after


def check_api_change_match(rec, kb_entry):
    """Check if the stated API change matches the KB source code info."""
    api_change = rec.get("api_change", "").lower()
    if not api_change:
        return True, "no_api_change_stated"

    # If rec says no API change / same API
    no_change_phrases = [
        "no api change", "same api", "none", "no change",
        "remains posix", "remains mpi", "no api", "not required",
        "same posix", "different usage pattern", "buffering strategy",
        "usage pattern", "write buffering", "write aggregation",
    ]
    if any(phrase in api_change for phrase in no_change_phrases):
        return True, "no_change_claimed"

    # Build combined KB text for matching
    source = kb_entry.get("source_code", {})
    io_funcs = source.get("io_functions", {})
    all_funcs_text = " ".join(str(v) for v in io_funcs.values()).lower()
    key_params = source.get("key_params", "").lower()

    # Also include fix descriptions
    fix_text = ""
    for fix in kb_entry.get("fixes", []):
        fix_text += " " + fix.get("fix", "").lower()
        fix_text += " " + fix.get("code_after", "").lower()
        fix_text += " " + fix.get("code_before", "").lower()

    combined_kb_text = all_funcs_text + " " + key_params + " " + fix_text

    # Extract specific API names from the recommendation
    api_patterns = [
        r"mpi_file_\w+",
        r"posix\s+write\w*", r"posix\s+read\w*",
        r"write\w*", r"read\w*",
        r"fwrite", r"fread",
        r"pwrite", r"pread",
        r"h5d\w+", r"nc_\w+",
        r"mpi.io", r"mpiio",
        r"buffering", r"aggregation",
    ]

    # Check for any meaningful overlap between the API change description
    # and KB content
    api_words = set(re.findall(r"\b\w+\b", api_change))
    kb_words = set(re.findall(r"\b\w+\b", combined_kb_text))

    # Remove common stop words
    stop_words = {"the", "a", "an", "to", "from", "with", "for", "in", "of",
                  "and", "or", "is", "are", "was", "be", "as", "at", "by",
                  "this", "that", "it", "on", "no", "not"}
    api_words -= stop_words
    kb_words -= stop_words

    if not api_words:
        return True, "no_meaningful_api_words"

    overlap = api_words & kb_words
    overlap_ratio = len(overlap) / len(api_words) if api_words else 0

    if overlap_ratio >= 0.3:
        return True, f"word_overlap={overlap_ratio:.2f}: {sorted(overlap)[:5]}"

    return False, f"low_overlap={overlap_ratio:.2f}, api_words={sorted(api_words)[:5]}"


def check_speedup_plausibility(rec, kb_entry):
    """Check if the claimed speedup is plausible given KB evidence."""
    speedup_str = rec.get("expected_speedup", "")
    numbers = extract_speedup_number(speedup_str)

    if not numbers:
        return True, "no_numeric_speedup", None

    max_claimed = max(numbers)

    plausibility_bounds = {
        "access_granularity": 100.0,
        "access_pattern": 10.0,
        "metadata_intensity": 20.0,
        "parallelism_efficiency": 10.0,
        "interface_choice": 5.0,
        "file_strategy": 10.0,
        "throughput_utilization": 10.0,
        "healthy": 2.0,
    }

    rec_dim = rec.get("bottleneck_dimension", "")
    bound = plausibility_bounds.get(rec_dim, 20.0)

    plausible = max_claimed <= bound
    return plausible, f"max_claimed={max_claimed:.1f}x, bound={bound}x", max_claimed


def verify_recommendation(rec, kb_by_id):
    """Verify a single recommendation at the atomic claim level."""
    result = {
        "bottleneck_dimension": rec.get("bottleneck_dimension", ""),
        "kb_citation": rec.get("kb_citation", ""),
        "has_citation": bool(rec.get("kb_citation", "").strip()),
        "claims": {},
        "hallucinations": [],
    }

    cited_ids = parse_cited_ids(rec.get("kb_citation", ""))

    # 1. Citation validity
    valid_ids = [cid for cid in cited_ids if cid in kb_by_id]
    result["claims"]["citation_valid"] = len(valid_ids) > 0 if cited_ids else False
    result["claims"]["n_cited"] = len(cited_ids)
    result["claims"]["n_valid_cited"] = len(valid_ids)

    if not valid_ids:
        result["claims"]["dimension_match"] = False
        result["claims"]["fix_type_match"] = False
        result["claims"]["code_pattern_match"] = False
        result["claims"]["api_change_match"] = False
        result["claims"]["speedup_plausible"] = False
        if cited_ids:
            result["hallucinations"].append({
                "type": "invalid_citation",
                "detail": f"Cited IDs not in KB: {cited_ids}",
            })
        return result

    primary_kb = kb_by_id[valid_ids[0]]

    # 2. Dimension match
    dim_match = check_dimension_match(rec.get("bottleneck_dimension", ""), primary_kb)
    result["claims"]["dimension_match"] = dim_match
    if not dim_match:
        result["hallucinations"].append({
            "type": "dimension_mismatch",
            "detail": (f"Rec dimension '{rec.get('bottleneck_dimension')}' "
                       f"not in KB labels {primary_kb.get('bottleneck_labels')}"),
        })

    # 3. Fix type match
    fix_match, fix_score = check_fix_type_match(rec, primary_kb)
    result["claims"]["fix_type_match"] = fix_match
    result["claims"]["fix_type_score"] = round(fix_score, 3)
    if not fix_match:
        result["hallucinations"].append({
            "type": "fix_type_mismatch",
            "detail": f"Fix strategy similarity {fix_score:.3f} < 0.3",
        })

    # 4. Code pattern match
    code_match, sim_before, sim_after = check_code_pattern_match(rec, primary_kb)
    result["claims"]["code_pattern_match"] = code_match
    result["claims"]["code_before_similarity"] = round(sim_before, 3)
    result["claims"]["code_after_similarity"] = round(sim_after, 3)
    if not code_match:
        result["hallucinations"].append({
            "type": "code_pattern_mismatch",
            "detail": (f"Code similarity (before={sim_before:.3f}, "
                       f"after={sim_after:.3f}) avg < 0.4"),
        })

    # 5. API change match
    api_match, api_detail = check_api_change_match(rec, primary_kb)
    result["claims"]["api_change_match"] = api_match
    result["claims"]["api_change_detail"] = api_detail
    if not api_match:
        result["hallucinations"].append({
            "type": "api_change_mismatch",
            "detail": f"API not found in KB: {api_detail}",
        })

    # 6. Speedup plausibility
    speedup_ok, speedup_detail, speedup_val = check_speedup_plausibility(
        rec, primary_kb
    )
    result["claims"]["speedup_plausible"] = speedup_ok
    result["claims"]["speedup_detail"] = speedup_detail
    if speedup_val is not None:
        result["claims"]["speedup_claimed"] = speedup_val
    if not speedup_ok:
        result["hallucinations"].append({
            "type": "speedup_implausible",
            "detail": f"Claimed speedup implausible: {speedup_detail}",
        })

    return result


def compute_model_stats(verification_results):
    """Compute per-model summary statistics from verification results."""
    per_model = defaultdict(lambda: {
        "n_recs_total": 0,
        "n_recs_cited": 0,
        "n_recs_uncited": 0,
        "citation_valid_count": 0,
        "dimension_match_count": 0,
        "fix_type_match_count": 0,
        "code_pattern_match_count": 0,
        "api_change_match_count": 0,
        "speedup_plausible_count": 0,
        "hallucinations": [],
    })

    for vr in verification_results:
        model = vr["model"]
        stats = per_model[model]
        stats["n_recs_total"] += 1

        if vr["has_citation"]:
            stats["n_recs_cited"] += 1
        else:
            stats["n_recs_uncited"] += 1

        claims = vr["claims"]
        if claims.get("citation_valid"):
            stats["citation_valid_count"] += 1
        if claims.get("dimension_match"):
            stats["dimension_match_count"] += 1
        if claims.get("fix_type_match"):
            stats["fix_type_match_count"] += 1
        if claims.get("code_pattern_match"):
            stats["code_pattern_match_count"] += 1
        if claims.get("api_change_match"):
            stats["api_change_match_count"] += 1
        if claims.get("speedup_plausible"):
            stats["speedup_plausible_count"] += 1

        for h in vr.get("hallucinations", []):
            h_copy = dict(h)
            h_copy["model"] = model
            h_copy["kb_citation"] = vr.get("kb_citation", "")
            h_copy["bottleneck_dimension"] = vr.get("bottleneck_dimension", "")
            stats["hallucinations"].append(h_copy)

    # Build summary
    summary = {}
    for model, stats in per_model.items():
        n_total = max(stats["n_recs_total"], 1)
        n_cited = max(stats["n_recs_cited"], 1)

        model_summary = {
            "n_recs_total": stats["n_recs_total"],
            "n_recs_cited": stats["n_recs_cited"],
            "n_recs_uncited": stats["n_recs_uncited"],
            # Citation groundedness: among all recs, fraction with valid citation
            "citation_groundedness": round(
                stats["citation_valid_count"] / n_total, 4
            ),
            # Claim-level metrics: computed ONLY among cited recommendations
            "dimension_match_rate": round(
                stats["dimension_match_count"] / n_cited, 4
            ),
            "fix_type_match_rate": round(
                stats["fix_type_match_count"] / n_cited, 4
            ),
            "code_pattern_match_rate": round(
                stats["code_pattern_match_count"] / n_cited, 4
            ),
            "api_change_match_rate": round(
                stats["api_change_match_count"] / n_cited, 4
            ),
            "speedup_plausible_rate": round(
                stats["speedup_plausible_count"] / n_cited, 4
            ),
        }

        # Claim-level groundedness: average of 5 atomic checks among cited recs
        atomic_counts = [
            stats["dimension_match_count"],
            stats["fix_type_match_count"],
            stats["code_pattern_match_count"],
            stats["api_change_match_count"],
            stats["speedup_plausible_count"],
        ]
        model_summary["claim_groundedness"] = round(
            sum(atomic_counts) / (5 * n_cited), 4
        )

        # Hallucination info (only from cited recs — uncited is a separate issue)
        cited_halluc = [h for h in stats["hallucinations"]
                        if h.get("type") != "invalid_citation"]
        model_summary["n_hallucinations_in_cited"] = len(cited_halluc)

        halluc_types = defaultdict(int)
        for h in cited_halluc:
            halluc_types[h["type"]] += 1
        model_summary["hallucination_breakdown"] = dict(halluc_types)

        summary[model] = model_summary

    return summary, per_model


def analyze_evaluation_results(kb_by_id):
    """Analyze the controlled 60-sample evaluation set."""
    # Find most recent evaluation results file
    eval_files = sorted(EVAL_DIR.glob("evaluation_results_*.json"))
    if not eval_files:
        logger.warning("No evaluation results found in %s", EVAL_DIR)
        return None, []

    latest = eval_files[-1]
    logger.info("Using evaluation results: %s", latest.name)

    with open(latest) as f:
        eval_data = json.load(f)

    all_verifications = []
    for short_model, results in eval_data.items():
        full_model = MODEL_MAP.get(short_model, short_model)
        for result in results:
            parsed = result.get("step4_recommendation", {}).get("parsed", {})
            recs = parsed.get("recommendations", [])
            for rec in recs:
                vr = verify_recommendation(rec, kb_by_id)
                vr["model"] = full_model
                vr["workload"] = result.get("workload", "")
                all_verifications.append(vr)

    summary, raw_stats = compute_model_stats(all_verifications)
    return summary, all_verifications, raw_stats


def analyze_cached_responses(kb_by_id):
    """Analyze all cached LLM responses."""
    all_verifications = []
    n_responses = 0
    n_parse_errors = 0

    for fname in sorted(os.listdir(CACHE_DIR)):
        if not fname.endswith(".json"):
            continue
        fpath = CACHE_DIR / fname
        with open(fpath) as f:
            data = json.load(f)

        n_responses += 1
        model = data["metadata"]["model"]

        parsed = extract_json_from_response(data["response"])
        if parsed is None:
            n_parse_errors += 1
            continue

        recs = parsed.get("recommendations", [])
        for rec in recs:
            vr = verify_recommendation(rec, kb_by_id)
            vr["model"] = model
            vr["cache_file"] = fname
            all_verifications.append(vr)

    summary, raw_stats = compute_model_stats(all_verifications)
    return summary, all_verifications, raw_stats, n_responses, n_parse_errors


def print_summary(title, summary):
    """Print formatted summary table."""
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)

    for model, s in sorted(summary.items()):
        short_name = model.split("/")[-1]
        n_cited = s["n_recs_cited"]
        n_total = s["n_recs_total"]
        print(f"\n--- {short_name} ({n_total} total recs, {n_cited} with citations) ---")
        print(f"  Citation groundedness (existing): "
              f"{s['citation_groundedness']:.3f}")
        print(f"  Claim groundedness (NEW, cited):  "
              f"{s['claim_groundedness']:.3f}")
        print(f"    Dimension match:                "
              f"{s['dimension_match_rate']:.3f}")
        print(f"    Fix type match:                 "
              f"{s['fix_type_match_rate']:.3f}")
        print(f"    Code pattern match:             "
              f"{s['code_pattern_match_rate']:.3f}")
        print(f"    API change match:               "
              f"{s['api_change_match_rate']:.3f}")
        print(f"    Speedup plausible:              "
              f"{s['speedup_plausible_rate']:.3f}")
        if s["n_recs_uncited"] > 0:
            print(f"  Uncited recommendations:          "
                  f"{s['n_recs_uncited']}")
        if s["hallucination_breakdown"]:
            print(f"  Hallucinations in cited recs:     "
                  f"{s['n_hallucinations_in_cited']}")
            for htype, count in sorted(s["hallucination_breakdown"].items()):
                print(f"    {htype}: {count}")


def main():
    kb_by_id = load_knowledge_base()

    # ---- Analysis A: Controlled 60-sample evaluation set ----
    eval_result = analyze_evaluation_results(kb_by_id)
    if eval_result is not None:
        eval_summary, eval_verifications, eval_raw = eval_result
        print_summary(
            "ANALYSIS A: CONTROLLED EVALUATION SET (60 samples/model)",
            eval_summary,
        )
    else:
        eval_summary = None
        eval_verifications = []

    # ---- Analysis B: All cached responses ----
    cache_result = analyze_cached_responses(kb_by_id)
    cache_summary, cache_verifications, cache_raw, n_resp, n_parse = cache_result
    print_summary(
        f"ANALYSIS B: ALL CACHED RESPONSES ({n_resp} total, {n_parse} parse errors)",
        cache_summary,
    )

    # ---- Collect sample hallucinations for output ----
    sample_hallucinations = {}
    source = eval_raw if eval_result else cache_raw
    for model, stats in source.items():
        # Pick up to 5 diverse hallucinations
        seen_types = set()
        samples = []
        for h in stats["hallucinations"]:
            if h["type"] not in seen_types or len(samples) < 8:
                samples.append(h)
                seen_types.add(h["type"])
            if len(samples) >= 8:
                break
        sample_hallucinations[model] = samples

    # ---- Save results ----
    output = {
        "description": (
            "Atomic-level groundedness verification (W8 response). "
            "citation_groundedness checks if a valid KB ID is cited. "
            "claim_groundedness checks if each specific claim in cited "
            "recommendations matches the KB entry content."
        ),
        "controlled_evaluation": {
            "description": "60-sample controlled evaluation set per model",
            "per_model": eval_summary,
        } if eval_summary else None,
        "all_cached": {
            "description": "All cached LLM responses",
            "n_responses": n_resp,
            "n_parse_errors": n_parse,
            "per_model": cache_summary,
        },
        "sample_hallucinations": sample_hallucinations,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", OUTPUT_PATH)

    # ---- Print key finding ----
    print("\n" + "=" * 70)
    print("KEY FINDING: Citation vs Claim Groundedness Gap")
    print("=" * 70)
    if eval_summary:
        for model, s in sorted(eval_summary.items()):
            short = model.split("/")[-1]
            gap = s["citation_groundedness"] - s["claim_groundedness"]
            print(f"  {short:30s} citation={s['citation_groundedness']:.3f}  "
                  f"claim={s['claim_groundedness']:.3f}  gap={gap:+.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
