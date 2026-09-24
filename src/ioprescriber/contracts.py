"""Versioned contracts shared by the detection and recommendation paths."""

from __future__ import annotations

import math
import statistics

from src.data.benchmark_verify import BOTTLENECK_DIMENSIONS


PIPELINE_RESULT_VERSION = 1
RECOMMENDATION_SCHEMA_VERSION = 1
QUALITATIVE_SCHEMA_VERSION = 1
KB_SCHEMA_VERSION = 2
ALLOWED_DIMENSIONS = set(BOTTLENECK_DIMENSIONS)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_kb_entry(entry):
    """Reject KB rows that lack traceable measured fix evidence."""
    _require(isinstance(entry, dict), "KB entry must be an object")
    for key in ("entry_id", "sample_id", "job_group", "benchmark", "scenario"):
        _require(isinstance(entry.get(key), str) and entry[key], f"KB entry missing {key}")
    labels = entry.get("bottleneck_labels")
    _require(isinstance(labels, list) and labels, "KB entry needs bottleneck labels")
    _require(set(labels) <= ALLOWED_DIMENSIONS, "KB entry has an unknown bottleneck label")
    signature = entry.get("darshan_signature")
    _require(isinstance(signature, dict) and signature, "KB entry needs a Darshan signature")
    _require(all(isinstance(v, (int, float)) and math.isfinite(v)
                 for v in signature.values()), "KB signature values must be finite numbers")
    source = entry.get("source_code")
    _require(isinstance(source, dict), "KB entry needs source provenance")
    for key in ("repository", "revision", "path"):
        _require(isinstance(source.get(key), str) and source[key],
                 f"KB source provenance missing {key}")
    fixes = entry.get("fixes")
    _require(isinstance(fixes, list) and fixes, "KB entry needs at least one measured fix")
    seen = set()
    for fix in fixes:
        _require(isinstance(fix, dict), "KB fix must be an object")
        for key in ("fix_id", "dimension", "description", "code_before",
                    "code_after", "api_change"):
            _require(isinstance(fix.get(key), str) and fix[key], f"KB fix missing {key}")
        _require(fix["fix_id"] not in seen, "duplicate fix_id in KB entry")
        seen.add(fix["fix_id"])
        _require(fix["dimension"] in labels, "KB fix dimension is not an entry label")
        measurement = fix.get("measurement")
        _require(isinstance(measurement, dict), "KB fix needs measurement provenance")
        _require(measurement.get("metric") == "walltime_speedup",
                 "KB measurement metric must be walltime_speedup")
        speedup = measurement.get("speedup")
        _require(isinstance(speedup, (int, float)) and math.isfinite(speedup) and speedup > 0,
                 "KB measured speedup must be a positive finite number")
        _require(measurement.get("accepted") is True,
                 "KB measurement must be accepted by the timing decision")
        _require(measurement.get("correctness_passed") is True,
                 "KB measurement needs a passed correctness check")
        _require(measurement.get("source_revision") == source["revision"],
                 "KB measurement and source revisions differ")
        for key in ("before_job_ids", "after_job_ids", "result_paths"):
            values = measurement.get(key)
            _require(isinstance(values, list) and values and
                     all(isinstance(v, str) and v for v in values),
                     f"KB measurement needs nonempty {key}")
        for key in ("before_commands", "after_commands"):
            values = measurement.get(key)
            _require(isinstance(values, list) and values and
                     all(isinstance(v, str) and v for v in values),
                     f"KB measurement needs nonempty {key}")
        for key in ("before_walltime_s", "after_walltime_s"):
            values = measurement.get(key)
            _require(isinstance(values, list) and values and all(
                isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                for v in values), f"KB measurement needs positive finite {key}")
        _require(len(measurement["before_job_ids"]) ==
                 len(measurement["before_walltime_s"]) ==
                 len(measurement["before_commands"]),
                 "KB before-run provenance lengths differ")
        _require(len(measurement["after_job_ids"]) ==
                 len(measurement["after_walltime_s"]) ==
                 len(measurement["after_commands"]),
                 "KB after-run provenance lengths differ")
        computed_speedup = (statistics.median(measurement["before_walltime_s"]) /
                            statistics.median(measurement["after_walltime_s"]))
        _require(math.isclose(float(speedup), computed_speedup,
                              rel_tol=0.0, abs_tol=1e-12),
                 "KB speedup differs from recorded median wall times")
        _require(isinstance(measurement.get("correctness_result_path"), str) and
                 measurement["correctness_result_path"],
                 "KB measurement needs a correctness result path")
    return entry


def validate_knowledge_base(document):
    """Validate a complete KB document and return its entry list."""
    _require(isinstance(document, dict), "knowledge base must be an object")
    _require(document.get("schema_version") == KB_SCHEMA_VERSION,
             f"knowledge base schema must be {KB_SCHEMA_VERSION}")
    allowed = document.get("allowed_sample_ids")
    _require(isinstance(allowed, list) and allowed, "knowledge base needs allowed sample IDs")
    _require(len(allowed) == len(set(allowed)), "knowledge base has duplicate allowed sample IDs")
    allowed_groups = document.get("allowed_job_groups")
    _require(isinstance(allowed_groups, list) and allowed_groups,
             "knowledge base needs allowed job groups")
    _require(len(allowed_groups) == len(set(allowed_groups)),
             "knowledge base has duplicate allowed job groups")
    entries = document.get("entries")
    _require(isinstance(entries, list) and entries, "knowledge base has no entries")
    entry_ids = set()
    sample_ids = set()
    for entry in entries:
        validate_kb_entry(entry)
        _require(entry["entry_id"] not in entry_ids, "knowledge base has duplicate entry IDs")
        entry_ids.add(entry["entry_id"])
        sample_ids.add(entry["sample_id"])
    _require(sample_ids <= set(allowed), "knowledge base contains a disallowed sample ID")
    _require({entry["job_group"] for entry in entries} <= set(allowed_groups),
             "knowledge base contains a disallowed job group")
    return entries


def validate_recommendation(response):
    """Validate the JSON response before caching, scoring, or execution."""
    _require(isinstance(response, dict), "recommendation response must be an object")
    _require(response.get("schema_version") == RECOMMENDATION_SCHEMA_VERSION,
             f"recommendation schema must be {RECOMMENDATION_SCHEMA_VERSION}")
    _require(isinstance(response.get("diagnosis"), str) and response["diagnosis"],
             "recommendation diagnosis must be a nonempty string")
    recommendations = response.get("recommendations")
    _require(isinstance(recommendations, list), "recommendations must be a list")
    for rec in recommendations:
        _require(isinstance(rec, dict), "each recommendation must be an object")
        for key in ("explanation", "code_before", "code_after", "kb_citation",
                    "evidence_fix_id", "api_change"):
            _require(isinstance(rec.get(key), str) and rec[key],
                     f"recommendation missing {key}")
        _require(isinstance(rec.get("priority"), int) and rec["priority"] > 0,
                 "recommendation priority must be a positive integer")
        _require(rec.get("bottleneck_dimension") in ALLOWED_DIMENSIONS,
                 "recommendation has an unknown bottleneck dimension")
        speedup = rec.get("expected_speedup")
        _require(isinstance(speedup, (int, float)) and math.isfinite(speedup) and speedup > 0,
                 "expected_speedup must be a positive finite number")
        _require(rec.get("confidence") in {"high", "medium", "low"},
                 "recommendation confidence is invalid")
    return response


def validate_qualitative_response(response):
    """Validate ablation output that has no KB and may make no numeric claim."""
    _require(isinstance(response, dict), "qualitative response must be an object")
    _require(response.get("schema_version") == QUALITATIVE_SCHEMA_VERSION,
             f"qualitative schema must be {QUALITATIVE_SCHEMA_VERSION}")
    _require(isinstance(response.get("diagnosis"), str) and response["diagnosis"],
             "qualitative diagnosis must be a nonempty string")
    recommendations = response.get("recommendations")
    _require(isinstance(recommendations, list), "qualitative recommendations must be a list")
    for rec in recommendations:
        _require(isinstance(rec, dict), "qualitative recommendation must be an object")
        for key in ("explanation", "code_before", "code_after", "api_change"):
            _require(isinstance(rec.get(key), str) and rec[key],
                     f"qualitative recommendation missing {key}")
        _require(isinstance(rec.get("priority"), int) and rec["priority"] > 0,
                 "qualitative priority must be a positive integer")
        _require(rec.get("bottleneck_dimension") in ALLOWED_DIMENSIONS,
                 "qualitative recommendation has an unknown bottleneck dimension")
        _require(rec.get("confidence") in {"high", "medium", "low"},
                 "qualitative confidence is invalid")
        _require("expected_speedup" not in rec and "kb_citation" not in rec,
                 "qualitative output cannot claim KB support or expected speedup")
    return response


def score_grounding(response, retrieved):
    """Compare each typed claim with the exact cited measured fix."""
    validate_recommendation(response)
    entries = {match["entry"]["entry_id"]: match["entry"] for match in retrieved}
    details = []
    for rec in response["recommendations"]:
        entry = entries.get(rec["kb_citation"])
        fix = None
        if entry:
            fix = next((f for f in entry["fixes"]
                        if f["fix_id"] == rec["evidence_fix_id"]), None)
        checks = {
            "citation": entry is not None,
            "diagnosis": bool(entry and rec["bottleneck_dimension"] in entry["bottleneck_labels"]),
            "action": bool(fix and rec["code_before"] == fix["code_before"] and
                           rec["code_after"] == fix["code_after"]),
            "api": bool(fix and rec["api_change"] == fix["api_change"]),
            "measurement": bool(fix and math.isclose(
                float(rec["expected_speedup"]), float(fix["measurement"]["speedup"]),
                rel_tol=0.0, abs_tol=1e-12)),
        }
        details.append({"priority": rec["priority"], "checks": checks,
                        "grounded": all(checks.values())})
    components = {
        key: (sum(d["checks"][key] for d in details) / len(details) if details else 0.0)
        for key in ("citation", "diagnosis", "action", "api", "measurement")
    }
    grounded = sum(d["grounded"] for d in details)
    return {
        "groundedness_score": grounded / len(details) if details else 0.0,
        "n_recommendations": len(details),
        "n_grounded": grounded,
        "n_ungrounded": len(details) - grounded,
        "component_scores": components,
        "details": details,
    }


def validate_pipeline_result(result):
    """Validate the sole public result layout returned by IOPrescriber."""
    _require(isinstance(result, dict), "pipeline result must be an object")
    _require(result.get("schema_version") == PIPELINE_RESULT_VERSION,
             f"pipeline result schema must be {PIPELINE_RESULT_VERSION}")
    required = {"schema_version", "workload", "pipeline_latency_ms", "detection",
                "attribution", "retrieval", "recommendation"}
    _require(set(result) == required, "pipeline result fields do not match the contract")
    _require(isinstance(result["detection"].get("predictions"), dict),
             "pipeline predictions must be an object")
    _require(set(result["detection"]) == {"predictions", "detected"},
             "pipeline detection fields do not match the contract")
    _require(isinstance(result["detection"].get("detected"), list),
             "pipeline detected dimensions must be a list")
    _require(isinstance(result["attribution"], dict), "pipeline attribution must be an object")
    _require(isinstance(result["retrieval"].get("entries"), list),
             "pipeline retrieval entries must be a list")
    _require(set(result["retrieval"]) == {"n_entries", "entries"},
             "pipeline retrieval fields do not match the contract")
    _require(result["retrieval"]["n_entries"] == len(result["retrieval"]["entries"]),
             "pipeline retrieval count differs from its entries")
    _require(isinstance(result["recommendation"], dict),
             "pipeline recommendation must be an object")
    _require(set(result["recommendation"]) == {"parsed", "groundedness", "metadata"},
             "pipeline recommendation fields do not match the contract")
    return result
