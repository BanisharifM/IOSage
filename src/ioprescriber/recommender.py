"""
IOPrescriber Step 4: LLM Code Recommendation via OpenRouter.

Generates specific, grounded code-level I/O optimization recommendations
using structured prompts with ML detection + SHAP features + KB evidence.

Supports multiple LLMs via OpenRouter:
  - anthropic/claude-sonnet-4 (best structured output)
  - openai/gpt-4o (strong code generation)
  - meta-llama/llama-3.1-70b-instruct (open-source, reproducible)

All inputs/outputs cached as JSON for reproducibility.

Input: ML predictions + SHAP attributions + KB entries + Darshan summary
Output: Structured JSON with diagnosis + code fixes + expected speedup
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path

from src.ioprescriber.contracts import (
    QUALITATIVE_SCHEMA_VERSION,
    RECOMMENDATION_SCHEMA_VERSION,
    score_grounding,
    validate_qualitative_response,
    validate_recommendation,
)

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_SCHEMA_VERSION = 2
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1"


# OpenRouter model IDs
MODELS = {
    "claude-sonnet": "anthropic/claude-sonnet-4",
    "gpt-4o": "openai/gpt-4o",
    "llama-70b": "meta-llama/llama-3.1-70b-instruct",
}

DIM_DESCRIPTIONS = {
    "access_granularity": "Many requests no larger than 1 MiB",
    "metadata_intensity": "Metadata calls consume at least half of recorded I/O time",
    "parallelism_efficiency": "Uneven I/O load distribution across MPI ranks",
    "access_pattern": "Many POSIX requests are nonsequential",
    "request_alignment": "Many POSIX requests begin at file-misaligned offsets",
    "interface_choice": "Many independent MPI-IO calls occur without collective calls",
    "file_strategy": "More than 1000 small data files are accessed",
    "throughput_utilization": "A synchronous durability call occurs after each write",
    "healthy": "All registered patterns are absent and observable",
}


class Recommender:
    """LLM-based code recommendation with OpenRouter multi-model support."""

    def __init__(self, model="claude-sonnet", temperature=0.0, max_tokens=2000,
                 cache_dir=None):
        self.model_key = model
        self.model_id = MODELS.get(model, model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir or str(PROJECT_DIR / "data" / "llm_cache" / "ioprescriber")

        # Verify API key
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not set. LLM calls will fail.")

        logger.info("Recommender initialized: model=%s, temp=%.1f, cache=%s",
                    self.model_id, temperature, self.cache_dir)

    def build_prompt(self, predictions, detected_dims, shap_features,
                      kb_entries, darshan_summary):
        """Build structured prompt with all pipeline context.

        Returns (system_prompt, user_prompt)
        """
        system_prompt = """You are an HPC I/O performance expert. You analyze Darshan profiling data
and provide specific, actionable code-level optimization recommendations.

RULES:
1. Every recommendation MUST be grounded in the benchmark evidence provided.
2. Include specific before/after code snippets showing the exact change.
3. Cite the KB entry ID for each recommendation.
4. Copy the numeric measured speedup from the cited fix exactly.
5. Do NOT fabricate performance numbers or API calls that don't exist.
6. Prioritize recommendations by expected impact (highest speedup first).
7. Use standard HPC I/O APIs: POSIX (read/write), MPI-IO (MPI_File_*), HDF5 (H5D*).
"""

        # ML detection
        detection_lines = []
        for dim in detected_dims:
            conf = predictions.get(dim, 0)
            desc = DIM_DESCRIPTIONS.get(dim, "")
            detection_lines.append(f"  - **{dim}** (confidence={conf:.2f}): {desc}")
        detection_str = "\n".join(detection_lines)

        # SHAP features
        shap_str = ""
        for dim, features in shap_features.items():
            if not features:
                continue
            shap_str += f"\n  {dim}:\n"
            for f in features[:5]:
                shap_str += (f"    - {f['feature']} = {f['value']:.4f} "
                            f"(|SHAP|={f['abs_importance']:.4f}, {f['direction']})\n")

        # KB evidence with source code
        kb_str = ""
        for i, match in enumerate(kb_entries[:3]):
            e = match["entry"]
            kb_str += f"\n  --- KB Entry {i+1} (ID: {e['entry_id']}) ---\n"
            kb_str += f"  Benchmark: {e['benchmark']} | Scenario: {e['scenario']}\n"
            kb_str += f"  Bottleneck: {', '.join(e['bottleneck_labels'])}\n"

            # Source code reference
            src = e.get("source_code", {})
            kb_str += (f"  Source: {src['repository']} at {src['revision']}\n"
                       f"  Path: {src['path']}\n")

            # Measured fixes
            for fix in e.get("fixes", [])[:1]:
                measurement = fix["measurement"]
                kb_str += f"  Fix ID: {fix['fix_id']}\n"
                kb_str += f"  Fix: {fix['description']}\n"
                kb_str += f"  API change: {fix['api_change']}\n"
                kb_str += f"  Code BEFORE:\n    {fix['code_before']}\n"
                kb_str += f"  Code AFTER:\n    {fix['code_after']}\n"
                kb_str += f"  Measured wall-time speedup: {measurement['speedup']}\n"
                kb_str += f"  Before jobs: {measurement['before_job_ids']}\n"
                kb_str += f"  After jobs: {measurement['after_job_ids']}\n"

        # Darshan summary
        summary_str = "\n".join(
            f"  {k}: {v}" for k, v in darshan_summary.items() if v and v != 0
        )

        user_prompt = f"""Analyze this HPC job's I/O behavior and provide code-level optimization recommendations.

## Detected Bottlenecks (ML classifier):
{detection_str}

## Key Contributing Features (SHAP per-label attribution):
{shap_str}

## Benchmark Evidence (verified Knowledge Base with source code):
{kb_str}

## Job Darshan Summary:
{summary_str}

## Task:
1. Explain what I/O problems this job has (grounded in Darshan values).
2. For each detected bottleneck, provide a specific code-level fix with before/after code.
3. Copy `expected_speedup`, code, API change, entry ID, and fix ID from one measured fix.
4. Prioritize by expected impact.

Respond in JSON:
{{
  "schema_version": {RECOMMENDATION_SCHEMA_VERSION},
  "diagnosis": "plain language explanation of I/O problems",
  "recommendations": [
    {{
      "priority": 1,
      "bottleneck_dimension": "dimension_name",
      "explanation": "what is wrong and why",
      "code_before": "the problematic I/O code pattern",
      "code_after": "the optimized I/O code",
      "expected_speedup": 1.25,
      "kb_citation": "entry_id from KB",
      "evidence_fix_id": "fix_id from the cited entry",
      "confidence": "high/medium/low",
      "api_change": "e.g., POSIX write -> MPI_File_write_all"
    }}
  ]
}}
"""
        return system_prompt, user_prompt

    def _cache_request(self, system_prompt, user_prompt, cache_namespace,
                       response_contract="measured-v1"):
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "response_schema_version": RECOMMENDATION_SCHEMA_VERSION,
            "response_contract": response_contract,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "user_prompt_sha256": hashlib.sha256(user_prompt.encode()).hexdigest(),
            "model": self.model_id,
            "resolved_model": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "endpoint": OPENROUTER_ENDPOINT,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "namespace": cache_namespace,
        }

    def call_llm(self, system_prompt, user_prompt, cache_namespace=None,
                 bypass_cache=False, validator=validate_recommendation,
                 response_contract="measured-v1"):
        """Call LLM via OpenRouter with caching."""
        request_contract = self._cache_request(
            system_prompt, user_prompt, cache_namespace, response_contract)
        cache_key = hashlib.sha256(
            json.dumps(request_contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        cache_path = Path(self.cache_dir) / f"{cache_key}.json"

        if cache_path.exists() and not bypass_cache:
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("request") != request_contract:
                raise ValueError(f"cache request contract mismatch: {cache_path}")
            parsed, error = self.parse_response(cached.get("response", ""), validator=validator)
            if error:
                raise ValueError(f"cached response violates the schema: {error}")
            logger.info("  Cache hit: %s", cache_path.name[:16])
            source = cached.get("metadata", {})
            metadata = {
                "model": source.get("model", self.model_id),
                "resolved_model": source.get("resolved_model"),
                "cache_hit": True,
                "api_latency_ms": 0.0,
                "tokens_input": 0,
                "tokens_output": 0,
                "request_id": None,
                "cache_source_request_id": source.get("request_id"),
            }
            return cached["response"], metadata

        # API call
        from openai import OpenAI
        client = OpenAI(
            base_url=OPENROUTER_ENDPOINT,
            api_key=self.api_key,
        )

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        text = response.choices[0].message.content
        parsed, error = self.parse_response(text, validator=validator)
        if error:
            raise ValueError(f"LLM response violates the recommendation schema: {error}")
        resolved_model = getattr(response, "model", None)
        if resolved_model != self.model_id:
            raise ValueError(
                f"provider resolved {self.model_id} to {resolved_model}; use an exact model ID")
        metadata = {
            "model": self.model_id,
            "resolved_model": resolved_model,
            "cache_hit": False,
            "api_latency_ms": round(latency_ms, 1),
            "tokens_input": getattr(response.usage, "prompt_tokens", 0),
            "tokens_output": getattr(response.usage, "completion_tokens", 0),
            "request_id": getattr(response, "id", None),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        # Cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "request": request_contract,
                "response": text,
                "metadata": metadata,
            }, f, indent=2)
        logger.info("  LLM response: %d chars, %d tokens, %.0fms",
                    len(text), metadata["tokens_input"] + metadata["tokens_output"],
                    latency_ms)

        return text, metadata

    def parse_response(self, response_text, validator=validate_recommendation):
        """Parse LLM JSON response, handle malformed output gracefully."""
        text = response_text.strip()

        # Extract JSON from markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1].strip()

        try:
            parsed = json.loads(text)
            validator(parsed)
            return parsed, None
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("  JSON parse failed: %s", str(e)[:100])
            return None, str(e)

    def check_groundedness(self, parsed_response, kb_entries):
        """Score diagnosis, action, API, and numeric support independently."""
        return score_grounding(parsed_response, kb_entries)

    def recommend(self, predictions, detected_dims, shap_features,
                   kb_entries, darshan_summary, cache_namespace=None,
                   bypass_cache=False):
        """Generate recommendation and check groundedness.

        Returns:
            recommendation: parsed JSON or None
            groundedness: dict with scores
            metadata: LLM call metadata
            raw_response: raw text
        """
        sys_p, usr_p = self.build_prompt(
            predictions, detected_dims, shap_features,
            kb_entries, darshan_summary,
        )

        raw_response, metadata = self.call_llm(
            sys_p, usr_p, cache_namespace=cache_namespace,
            bypass_cache=bypass_cache)
        parsed, parse_error = self.parse_response(raw_response)
        groundedness = self.check_groundedness(parsed, kb_entries)

        if parse_error:
            logger.warning("  Parse error: %s", parse_error[:80])
        else:
            logger.info("  Groundedness: %.2f (%d/%d citations verified)",
                        groundedness["groundedness_score"],
                        groundedness["n_grounded"],
                        groundedness["n_recommendations"])

        return parsed, groundedness, metadata, raw_response

    def recommend_without_evidence(self, detected_dims, darshan_summary,
                                   shap_features=None,
                                   cache_namespace=None, bypass_cache=False):
        """Generate qualitative ablation output without numeric or KB claims."""
        system_prompt = """You analyze HPC I/O counters without benchmark evidence.
Do not state an expected speedup, benchmark measurement, or citation. Return
only the requested JSON object."""
        user_prompt = f"""Detected dimensions: {json.dumps(detected_dims)}
Darshan summary: {json.dumps(darshan_summary, sort_keys=True)}
SHAP attribution: {json.dumps(shap_features or {}, sort_keys=True)}

Respond as:
{{
  "schema_version": {QUALITATIVE_SCHEMA_VERSION},
  "diagnosis": "nonempty explanation",
  "recommendations": [
    {{
      "priority": 1,
      "bottleneck_dimension": "one detected bottleneck dimension",
      "explanation": "qualitative explanation",
      "code_before": "problematic pattern",
      "code_after": "proposed pattern",
      "confidence": "high, medium, or low",
      "api_change": "qualitative API change"
    }}
  ]
}}
"""
        raw, metadata = self.call_llm(
            system_prompt, user_prompt, cache_namespace=cache_namespace,
            bypass_cache=bypass_cache, validator=validate_qualitative_response,
            response_contract="qualitative-v1")
        parsed, error = self.parse_response(raw, validator=validate_qualitative_response)
        if error:
            raise ValueError(error)
        return parsed, metadata, raw
