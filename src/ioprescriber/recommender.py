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

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# Add local packages
LOCAL_PKGS = PROJECT_DIR / ".local_pkgs"
if LOCAL_PKGS.exists():
    import sys
    sys.path.insert(0, str(LOCAL_PKGS))

# OpenRouter model IDs
MODELS = {
    "claude-sonnet": "anthropic/claude-sonnet-4",
    "gpt-4o": "openai/gpt-4o",
    "llama-70b": "meta-llama/llama-3.1-70b-instruct",
}

DIM_DESCRIPTIONS = {
    "access_granularity": "I/O operations with very small transfer sizes (<1MB), causing excessive syscall overhead",
    "metadata_intensity": "Excessive file metadata operations (open/stat/close) relative to data I/O",
    "parallelism_efficiency": "Uneven I/O load distribution across MPI ranks",
    "access_pattern": "Random (non-sequential) file access, defeating OS read-ahead and storage prefetching",
    "interface_choice": "Using suboptimal I/O interface (POSIX instead of MPI-IO collective for shared files)",
    "file_strategy": "Suboptimal file strategy (file-per-process explosion or shared-file contention)",
    "throughput_utilization": "Throughput below achievable (excessive sync, single-OST, redundant traffic)",
    "healthy": "No significant I/O bottleneck detected",
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
4. Quantify expected improvement using benchmark measurements only.
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
            if src.get("repo"):
                kb_str += f"  Source: {src['repo']} ({src.get('language', '?')})\n"
            if src.get("io_functions"):
                for api, code in list(src["io_functions"].items())[:2]:
                    kb_str += f"  Code ({api}): {code}\n"

            # Fix patterns
            for fix in e.get("fixes", [])[:1]:
                kb_str += f"  Cause: {fix.get('cause', 'N/A')}\n"
                kb_str += f"  Fix: {fix.get('fix', 'N/A')}\n"
                if fix.get("code_before"):
                    kb_str += f"  Code BEFORE:\n    {fix['code_before']}\n"
                if fix.get("code_after"):
                    kb_str += f"  Code AFTER:\n    {fix['code_after']}\n"

        # Darshan summary
        summary_str = "\n".join(
            f"  {k}: {v}" for k, v in darshan_summary.items() if v and v != 0
        )

        user_prompt = f"""Analyze this HPC job's I/O behavior and provide code-level optimization recommendations.

## Detected Bottlenecks (ML classifier, Micro-F1=0.923):
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
3. Estimate expected improvement based ONLY on benchmark evidence (cite KB entry IDs).
4. Prioritize by expected impact.

Respond in JSON:
{{
  "diagnosis": "plain language explanation of I/O problems",
  "recommendations": [
    {{
      "priority": 1,
      "bottleneck_dimension": "dimension_name",
      "explanation": "what is wrong and why",
      "code_before": "the problematic I/O code pattern",
      "code_after": "the optimized I/O code",
      "expected_speedup": "Nx based on KB evidence",
      "kb_citation": "entry_id from KB",
      "confidence": "high/medium/low",
      "api_change": "e.g., POSIX write -> MPI_File_write_all"
    }}
  ],
  "overall_expected_improvement": "estimated total speedup range"
}}
"""
        return system_prompt, user_prompt

    def call_llm(self, system_prompt, user_prompt):
        """Call LLM via OpenRouter with caching."""
        # Cache check
        cache_key = hashlib.md5(
            (system_prompt + user_prompt + self.model_id).encode()
        ).hexdigest()
        cache_path = Path(self.cache_dir) / f"{cache_key}.json"

        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            logger.info("  Cache hit: %s", cache_path.name[:16])
            return cached["response"], cached.get("metadata", {})

        # API call
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
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
        metadata = {
            "model": self.model_id,
            "latency_ms": round(latency_ms, 1),
            "tokens_input": getattr(response.usage, "prompt_tokens", 0),
            "tokens_output": getattr(response.usage, "completion_tokens", 0),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        # Cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({
                "response": text,
                "metadata": metadata,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }, f, indent=2)
        logger.info("  LLM response: %d chars, %d tokens, %.0fms",
                    len(text), metadata["tokens_input"] + metadata["tokens_output"],
                    latency_ms)

        return text, metadata

    def parse_response(self, response_text):
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
            return json.loads(text), None
        except json.JSONDecodeError as e:
            logger.warning("  JSON parse failed: %s", str(e)[:100])
            return None, str(e)

    def check_groundedness(self, parsed_response, kb_entries):
        """Check how many recommendations cite valid KB entries."""
        if not parsed_response or "recommendations" not in parsed_response:
            return {"groundedness_score": 0.0, "n_recommendations": 0, "n_grounded": 0}

        kb_ids = {m["entry"]["entry_id"] for m in kb_entries}
        recs = parsed_response["recommendations"]
        grounded = sum(1 for r in recs if r.get("kb_citation", "") in kb_ids)

        return {
            "groundedness_score": grounded / max(len(recs), 1),
            "n_recommendations": len(recs),
            "n_grounded": grounded,
            "n_ungrounded": len(recs) - grounded,
        }

    def recommend(self, predictions, detected_dims, shap_features,
                   kb_entries, darshan_summary):
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

        raw_response, metadata = self.call_llm(sys_p, usr_p)
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
