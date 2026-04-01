"""
Track C: ML-Guided Iterative LLM Code Optimization for HPC I/O.

Architecture (research-backed -- STELLAR/PerfCoder/POLO/Self-Refine):
  1. ML Classifier detects bottleneck type + SHAP features
  2. KB Retriever finds matching benchmark fix patterns
  3. LLM generates optimized benchmark configuration
  4. BenchmarkCommandBuilder validates config (safety layer)
  5. IterativeExecutor runs on HPC via SLURM, collects Darshan log
  6. ML re-classifies (did the fix work?)
  7. If improved AND < max_iterations: iterate with Darshan feedback
  8. If regression: rollback to best-so-far, try different strategy
  9. If converged (all bottleneck confidences < threshold): stop

Key design decisions (research-backed):
  - 3-5 iterations max (STELLAR converges in 5)
  - Explicit rollback on regression (PerfCodeGen best practice)
  - KB grounding reduces hallucination (ECO: 4x better than raw prompting)
  - ML planner + LLM optimizer pattern (PerfCoder: 4.82x vs 1.96x standalone)
  - Temperature=0 for deterministic code generation (PerfCodeGen protocol)
  - SHAP attribution in prompt (domain-specific feature guidance)
  - Per-iteration cost/latency tracking

References:
  - STELLAR (SC'25): Darshan feedback loop for storage tuning, 5 iterations
  - PerfCoder (2025): ML planner guiding LLM achieves 4.82x vs 1.96x alone
  - ECO (2025): Better prompting with actionable guidance = 7.81x vs 1.99x
  - Self-Refine (NeurIPS'23): iterative self-improvement foundation
  - POLO (IJCAI'25): profiling-guided code optimization

Usage:
    # Single workload test
    python -m src.llm.iterative_optimizer --workload ior_small_posix --model claude-sonnet

    # Full sweep (all workloads, all models)
    python -m src.llm.iterative_optimizer --sweep --n-runs 5

    # Single workload, dry run (no SLURM execution)
    python -m src.llm.iterative_optimizer --workload ior_small_posix --dry-run
"""

import json
import logging
import os
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

LOCAL_PKGS = Path(__file__).resolve().parent.parent.parent / ".local_pkgs"
if LOCAL_PKGS.exists():
    sys.path.insert(0, str(LOCAL_PKGS))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

DIMENSIONS = [
    "access_granularity", "metadata_intensity", "parallelism_efficiency",
    "access_pattern", "interface_choice", "file_strategy",
    "throughput_utilization", "healthy",
]

# OpenRouter model IDs (same as Track B recommender)
MODELS = {
    "claude-sonnet": "anthropic/claude-sonnet-4",
    "gpt-4o": "openai/gpt-4o",
    "llama-70b": "meta-llama/llama-3.1-70b-instruct",
}

# Per-model pricing (USD per million tokens)
MODEL_COSTS = {
    "anthropic/claude-sonnet-4": {"input": 3.0, "output": 15.0},
    "openai/gpt-4o": {"input": 2.50, "output": 10.0},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.59, "output": 0.79},
}

DIM_DESCRIPTIONS = {
    "access_granularity": "I/O with very small transfer sizes, causing excessive syscall overhead",
    "metadata_intensity": "Excessive file metadata operations relative to data I/O",
    "parallelism_efficiency": "Uneven I/O load distribution across MPI ranks",
    "access_pattern": "Random file access, defeating read-ahead and prefetching",
    "interface_choice": "Using suboptimal I/O interface for the access pattern",
    "file_strategy": "Suboptimal file strategy (FPP explosion or shared-file contention)",
    "throughput_utilization": "Throughput below achievable (excessive sync, poor config)",
    "healthy": "No significant I/O bottleneck detected",
}


class IterativeOptimizer:
    """ML-guided iterative LLM code optimization with Darshan feedback.

    The core loop:
        detect(darshan) -> explain(shap) -> retrieve(kb) ->
        generate_config(llm) -> validate(builder) -> execute(slurm) ->
        re_detect(darshan) -> evaluate -> iterate_or_stop
    """

    def __init__(self, config_path=None, model="claude-sonnet",
                 max_iterations=5, temperature=0.0, cache_dir=None,
                 use_ml=True, use_shap=True, use_kb=True,
                 use_feedback=True, dry_run=False):
        """Initialize the iterative optimizer.

        Args:
            config_path: path to iterative.yaml
            model: LLM model key (claude-sonnet, gpt-4o, llama-70b)
            max_iterations: maximum refinement iterations
            temperature: LLM temperature (0.0 for deterministic)
            cache_dir: LLM response cache directory
            use_ml: if False, skip ML detection (ablation A1)
            use_shap: if False, skip SHAP features (ablation A3)
            use_kb: if False, skip KB retrieval (ablation A2)
            use_feedback: if False, omit previous Darshan from prompt (ablation A5)
            dry_run: if True, simulate execution (no SLURM jobs)
        """
        self.model_key = model
        self.model_id = MODELS.get(model, model)
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.cache_dir = cache_dir or str(PROJECT_DIR / "data" / "llm_cache" / "iterative")
        self.use_ml = use_ml
        self.use_shap = use_shap
        self.use_kb = use_kb
        self.use_feedback = use_feedback
        self.dry_run = dry_run

        # Load configs
        iter_config_path = config_path or PROJECT_DIR / "configs" / "iterative.yaml"
        with open(iter_config_path) as f:
            self.iter_config = yaml.safe_load(f)

        train_config_path = PROJECT_DIR / "configs" / "training.yaml"
        with open(train_config_path) as f:
            self.train_config = yaml.safe_load(f)

        # Defaults for optional components
        self.models = {}
        self.feature_cols = []
        self.explainer = None
        self.kb = []

        # Load ML models
        if self.use_ml:
            self._load_ml_models()

        # Load SHAP explainer
        if self.use_shap and self.use_ml:
            self._load_shap_explainer()

        # Load KB
        if self.use_kb:
            self._load_knowledge_base()

        # Initialize command builder and executor
        from src.llm.benchmark_command_builder import BenchmarkCommandBuilder
        self.builder = BenchmarkCommandBuilder(
            config_path=iter_config_path,
            scratch_dir=self.iter_config["slurm"]["scratch_dir"],
        )

        if not self.dry_run:
            from src.llm.iterative_executor import IterativeExecutor
            self.executor = IterativeExecutor(self.iter_config)
        else:
            self.executor = None

        # API key
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            # Try OpenAI key for direct API calls
            self.api_key = os.environ.get("OPENAI_API_KEY", "")

        # Cost tracking
        self.total_tokens_input = 0
        self.total_tokens_output = 0

        logger.info(
            "IterativeOptimizer: model=%s, max_iter=%d, ml=%s, shap=%s, kb=%s, feedback=%s, dry=%s",
            self.model_key, max_iterations, use_ml, use_shap, use_kb, use_feedback, dry_run,
        )

    def _load_ml_models(self):
        """Load Phase 2 biquality XGBoost models."""
        model_path = PROJECT_DIR / "models" / "phase2" / "xgboost_biquality_w100.pkl"
        with open(model_path, "rb") as f:
            self.models = pickle.load(f)

        # Get feature columns
        import pandas as pd
        prod_feat = pd.read_parquet(
            PROJECT_DIR / self.train_config["paths"]["production_features"]
        )
        exclude = set(self.train_config.get("exclude_features", []))
        for col in prod_feat.columns:
            if col.startswith("_") or col.startswith("drishti_"):
                exclude.add(col)
        self.feature_cols = [c for c in prod_feat.columns if c not in exclude]
        logger.info("  ML models loaded: %d dimensions, %d features", len(self.models), len(self.feature_cols))

    def _load_shap_explainer(self):
        """Load SHAP TreeExplainers for per-dimension attribution."""
        try:
            from src.ioprescriber.explainer import Explainer
            self.explainer = Explainer(
                models=self.models,
                feature_cols=self.feature_cols,
            )
            logger.info("  SHAP explainer loaded")
        except Exception as e:
            logger.warning("  SHAP explainer failed to load: %s", e)
            self.explainer = None
            self.use_shap = False

    def _load_knowledge_base(self):
        """Load the benchmark knowledge base."""
        kb_path = PROJECT_DIR / "data" / "knowledge_base" / "knowledge_base_full.json"
        if kb_path.exists():
            with open(kb_path) as f:
                self.kb = json.load(f)
            logger.info("  KB loaded: %d entries", len(self.kb))
        else:
            # Fallback: build from recommendation module
            from src.llm.recommendation import load_knowledge_base
            self.kb = load_knowledge_base()
            logger.info("  KB built: %d entries", len(self.kb))

    # =========================================================================
    # ML Detection + SHAP
    # =========================================================================

    def detect_bottlenecks(self, features_dict):
        """Run ML classifier on Darshan features.

        Returns:
            predictions: dict {dimension: confidence}
            detected: list of dimension names above threshold
        """
        if not self.use_ml:
            return {}, ["unknown"]

        X = np.array(
            [[features_dict.get(col, 0) for col in self.feature_cols]],
            dtype=np.float32,
        )

        predictions = {}
        for dim in DIMENSIONS:
            if dim in self.models:
                prob = float(self.models[dim].predict_proba(X)[0][1])
                predictions[dim] = round(prob, 4)

        threshold = self.iter_config.get("iteration", {}).get("convergence_threshold", 0.3)
        detected = [d for d in DIMENSIONS if predictions.get(d, 0) > threshold and d != "healthy"]
        if not detected:
            detected = ["healthy"]

        return predictions, detected

    def get_shap_features(self, features_dict, detected_dims):
        """Get SHAP top features per detected dimension.

        Returns:
            dict {dimension: [list of {feature, value, abs_importance, direction}]}
        """
        if not self.use_shap or not self.explainer:
            return {}

        try:
            X = np.array(
                [[features_dict.get(col, 0) for col in self.feature_cols]],
                dtype=np.float32,
            )
            shap_results = self.explainer.explain(X, detected_dims)
            return shap_results
        except Exception as e:
            logger.warning("  SHAP failed: %s", e)
            return {}

    # =========================================================================
    # KB Retrieval
    # =========================================================================

    def retrieve_kb_evidence(self, detected_dims, features_dict, top_k=3):
        """Retrieve matching KB entries for detected bottlenecks."""
        if not self.use_kb:
            return []

        from src.llm.recommendation import retrieve_relevant_entries
        signature = {k: features_dict.get(k, 0) for k in [
            "avg_write_size", "small_io_ratio", "seq_write_ratio",
            "metadata_time_ratio", "collective_ratio", "total_bw_mb_s",
            "nprocs", "POSIX_BYTES_WRITTEN", "POSIX_FSYNCS",
        ]}
        return retrieve_relevant_entries(self.kb, detected_dims, signature, top_k)

    # =========================================================================
    # LLM Prompt Building
    # =========================================================================

    def build_prompt(self, iteration, workload_config, detected_dims, predictions,
                     shap_features, kb_matches, darshan_before, darshan_after=None,
                     current_config=None, best_speedup=None, rollback=False):
        """Build structured prompt for current iteration.

        Key differences from Track B:
        - Asks LLM to output benchmark CONFIG CHANGES (not arbitrary code)
        - Includes iteration feedback (before/after Darshan)
        - Includes SHAP features for targeted guidance
        - Rollback hint when regression detected
        """
        system_prompt = """You are an HPC I/O performance optimization expert.
You optimize benchmark configurations to fix detected I/O bottlenecks.

RULES:
1. Output ONLY benchmark parameter changes in the specified JSON format.
2. Every recommendation MUST be grounded in the benchmark evidence provided.
3. Target the specific bottleneck dimensions detected by the ML classifier.
4. If a previous iteration made things worse, try a COMPLETELY DIFFERENT strategy.
5. Do NOT fabricate performance numbers.
6. CRITICAL: Maintain the SAME total data volume (block_size * segments * nprocs).
   If you increase transfer_size, keep block_size and segments so total bytes stay the same.
   For example, if original is -t 64 -b 1M -s 100, change to -t 1048576 -b 1M -s 100
   (only change transfer size, NOT block or segments). This ensures fair speedup comparison.
7. Valid IOR parameters: -a (POSIX/MPIIO), -t (transfer size in bytes), -b (block size),
   -s (segments), -F (file-per-proc), -c (collective MPI-IO), -e (fsync at end),
   -C (reorder tasks), -Y (fsync per write), -z (random offsets),
   -O useO_DIRECT=1 (bypass page cache, requires -t >= 4096).
8. The transfer_size (-t) controls I/O granularity. The block_size (-b) and segments (-s)
   control total data volume. Only change -t to fix access_granularity. Only change -a/-c
   to fix interface_choice. Only remove -Y to fix throughput. Only remove -z to fix access_pattern.
9. For HACC-IO benchmarks:
   - executable: 'posix_shared', 'mpiio_shared', or 'fpp' (I/O backend)
   - num_particles: integer [50, 10000000] (data volume = 38 bytes * num_particles per rank)
   - collective_buffering: 'enabled' or 'disabled' (ROMIO aggregation control)
   - To fix interface_choice: switch from posix_shared to mpiio_shared
   - To fix file_strategy: switch from fpp to mpiio_shared
   - Increase num_particles to increase data volume (amortize overhead)
10. For custom (load_imbalance) benchmarks:
   - imbalance_factor: float [1.0, 100.0] (rank 0 writes this many times more data)
   - base_size_mb: integer [1, 500] (base data size per non-zero rank in MB)
   - To fix parallelism_efficiency: reduce imbalance_factor toward 1.0
"""

        # ML detection section
        if self.use_ml and detected_dims != ["unknown"]:
            detection_str = "\n".join(
                f"  - {dim}: {predictions.get(dim, 0):.2f} confidence -- {DIM_DESCRIPTIONS.get(dim, '')}"
                for dim in detected_dims
            )
        else:
            detection_str = "  (ML detection disabled -- analyze raw Darshan counters)"

        # SHAP section
        shap_str = ""
        if shap_features:
            for dim, features in shap_features.items():
                if not features:
                    continue
                shap_str += f"\n  {dim}:\n"
                for feat in features[:5]:
                    shap_str += (
                        f"    - {feat['feature']} = {feat['value']:.4f} "
                        f"(importance={feat['abs_importance']:.4f}, {feat['direction']})\n"
                    )

        # KB section
        kb_str = ""
        if kb_matches:
            for i, match in enumerate(kb_matches[:3]):
                e = match["entry"]
                kb_str += f"\n  Evidence {i+1} ({e['entry_id']}):\n"
                for fix in e.get("fixes", [])[:1]:
                    kb_str += f"    Cause: {fix.get('cause', 'N/A')}\n"
                    kb_str += f"    Fix: {fix.get('fix', 'N/A')}\n"
                    if fix.get("code_before"):
                        kb_str += f"    Before: {fix['code_before']}\n"
                    if fix.get("code_after"):
                        kb_str += f"    After: {fix['code_after']}\n"

        # Current config section
        config_str = ""
        if current_config:
            config_str = json.dumps(current_config, indent=2)

        # Darshan metrics
        key_metrics = [
            "avg_write_size", "small_io_ratio", "seq_write_ratio",
            "total_bw_mb_s", "write_bw_mb_s", "metadata_time_ratio",
            "nprocs", "POSIX_BYTES_WRITTEN", "POSIX_WRITES", "POSIX_FSYNCS",
        ]
        before_str = "\n".join(
            f"  {k}: {darshan_before.get(k, 0):.4f}"
            for k in key_metrics if darshan_before.get(k, 0) != 0
        )

        user_prompt = f"""## Iteration {iteration + 1}/{self.max_iterations}

## Benchmark: {workload_config.get('benchmark', 'ior')} -- {workload_config.get('description', '')}

## Current Benchmark Config:
{config_str}

## Detected Bottlenecks (ML classifier, Micro-F1=0.923):
{detection_str}
"""

        if shap_str:
            user_prompt += f"""
## Key Contributing Features (SHAP attribution):
{shap_str}
"""

        if kb_str:
            user_prompt += f"""
## Benchmark Evidence (verified Knowledge Base):
{kb_str}
"""

        user_prompt += f"""
## Current Darshan Profile:
{before_str}
"""

        # Iteration feedback
        if iteration > 0 and darshan_after and self.use_feedback:
            after_str = "\n".join(
                f"  {k}: {darshan_before.get(k, 0):.4f} -> {darshan_after.get(k, 0):.4f}"
                for k in key_metrics
                if darshan_before.get(k, 0) != 0 or darshan_after.get(k, 0) != 0
            )
            user_prompt += f"""
## Previous Iteration Results (before -> after):
{after_str}

## Best Speedup So Far: {best_speedup or 'N/A'}x
"""
            if rollback:
                user_prompt += """
## WARNING: Previous iteration caused REGRESSION. Try a DIFFERENT strategy.
"""

        benchmark_type = workload_config.get("benchmark", "ior")
        if benchmark_type == "mdtest":
            user_prompt += """
## Task:
Propose mdtest parameter changes to fix the detected bottlenecks.
Respond in JSON:
{
  "strategy": "brief description of optimization strategy",
  "config_changes": {
    "items_per_rank": "new item count (optional)",
    "write_bytes": "new write size per file (optional)",
    "unique_dir": true/false,
    "files_only": true/false
  },
  "expected_improvement": "estimated speedup based on KB evidence",
  "changes_made": ["list of specific changes and WHY they help"],
  "kb_citations": ["list of KB entry IDs used"]
}
"""
        elif benchmark_type == "hacc_io":
            user_prompt += """
## Task:
Propose HACC-IO parameter changes to fix the detected bottlenecks.
Respond in JSON:
{
  "strategy": "brief description of optimization strategy",
  "config_changes": {
    "executable": "posix_shared or mpiio_shared or fpp (optional)",
    "num_particles": "particle count per rank (optional)",
    "collective_buffering": "enabled or disabled (optional)"
  },
  "expected_improvement": "estimated speedup based on KB evidence",
  "changes_made": ["list of specific changes and WHY they help"],
  "kb_citations": ["list of KB entry IDs used"]
}
"""
        elif benchmark_type == "custom":
            user_prompt += """
## Task:
Propose load_imbalance parameter changes to fix the detected bottlenecks.
Respond in JSON:
{
  "strategy": "brief description of optimization strategy",
  "config_changes": {
    "imbalance_factor": "new imbalance factor (optional, 1.0 = balanced)",
    "base_size_mb": "new base data size in MB (optional)"
  },
  "expected_improvement": "estimated speedup based on KB evidence",
  "changes_made": ["list of specific changes and WHY they help"],
  "kb_citations": ["list of KB entry IDs used"]
}
"""
        else:
            user_prompt += """
## Task:
Propose benchmark parameter changes to fix the detected bottlenecks.
Respond in JSON:
{
  "strategy": "brief description of optimization strategy",
  "config_changes": {
    "api": "POSIX or MPIIO (optional, only if changing)",
    "transfer_size": "new size in bytes (optional)",
    "block_size": "new size (optional)",
    "segments": "new count (optional)",
    "file_per_proc": true/false,
    "extra_flags": "full flag string e.g. '-e -C -w -r'",
    "collective": true/false
  },
  "expected_improvement": "estimated speedup based on KB evidence",
  "changes_made": ["list of specific changes and WHY they help"],
  "kb_citations": ["list of KB entry IDs used"]
}
"""
        return system_prompt, user_prompt

    # =========================================================================
    # LLM Calling (OpenRouter, with caching)
    # =========================================================================

    def call_llm(self, system_prompt, user_prompt):
        """Call LLM via OpenRouter with response caching."""
        import hashlib

        cache_key = hashlib.md5(
            (system_prompt + user_prompt + self.model_id).encode()
        ).hexdigest()
        cache_path = Path(self.cache_dir) / f"{cache_key}.json"

        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            logger.info("  Cache hit: %s", cache_path.name[:12])
            meta = cached.get("metadata", {})
            self.total_tokens_input += meta.get("tokens_input", 0)
            self.total_tokens_output += meta.get("tokens_output", 0)
            return cached["response"], meta

        t0 = time.perf_counter()

        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        response = client.chat.completions.create(
            model=self.model_id,
            max_tokens=2000,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        text = response.choices[0].message.content
        tokens_in = getattr(response.usage, "prompt_tokens", 0)
        tokens_out = getattr(response.usage, "completion_tokens", 0)
        latency_ms = (time.perf_counter() - t0) * 1000

        metadata = {
            "model": self.model_id,
            "latency_ms": round(latency_ms, 1),
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        self.total_tokens_input += tokens_in
        self.total_tokens_output += tokens_out

        # Cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({
                "response": text, "metadata": metadata,
                "system_prompt": system_prompt, "user_prompt": user_prompt,
            }, f, indent=2)

        return text, metadata

    def parse_llm_response(self, response_text):
        """Parse LLM JSON response, handle malformed output."""
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]

        try:
            return json.loads(text.strip()), None
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning("  JSON parse failed: %s", str(e)[:80])
            return None, str(e)

    # =========================================================================
    # Cost Computation
    # =========================================================================

    def compute_cost_usd(self):
        """Compute total API cost in USD."""
        costs = MODEL_COSTS.get(self.model_id, {"input": 1.0, "output": 3.0})
        cost = (
            self.total_tokens_input * costs["input"] / 1_000_000
            + self.total_tokens_output * costs["output"] / 1_000_000
        )
        return round(cost, 4)

    # =========================================================================
    # Core Optimization Loop
    # =========================================================================

    def run_optimization(self, workload_name, run_id=0):
        """Run the full iterative optimization loop for one workload.

        Args:
            workload_name: key from configs/iterative.yaml workloads
            run_id: repetition ID for multi-run experiments

        Returns:
            dict with full optimization history, metrics, and cost
        """
        workload_config = self.iter_config["workloads"][workload_name]
        bad_config = dict(workload_config["bad_config"])

        logger.info("=" * 70)
        logger.info("ITERATIVE OPTIMIZATION: %s (run %d)", workload_name, run_id)
        logger.info("  Bottleneck: %s", workload_config.get("bottleneck"))
        logger.info("  Description: %s", workload_config.get("description"))
        logger.info("=" * 70)

        # Reset cost tracking for this run
        self.total_tokens_input = 0
        self.total_tokens_output = 0

        history = {
            "workload": workload_name,
            "run_id": run_id,
            "model": self.model_key,
            "model_id": self.model_id,
            "max_iterations": self.max_iterations,
            "config": {
                "use_ml": self.use_ml, "use_shap": self.use_shap,
                "use_kb": self.use_kb, "use_feedback": self.use_feedback,
            },
            "iterations": [],
            "best_iteration": -1,
            "best_speedup": 1.0,
            "best_config": None,
            "final_status": "not_started",
            "total_cost_usd": 0,
            "total_tokens": 0,
            "total_llm_latency_ms": 0,
            "total_execution_time_s": 0,
        }

        # Determine benchmark type
        benchmark_type = workload_config.get("benchmark", "ior")

        # Step 1: Execute the "bad" baseline config to get initial Darshan
        logger.info("  Step 1: Running baseline (bad) config...")
        current_config = dict(bad_config)

        if not self.dry_run:
            job_base = f"iter_{workload_name}_r{run_id}_baseline"
            job_scratch = f"{self.iter_config['slurm']['scratch_dir']}/{job_base}"
            if benchmark_type == "mdtest":
                sanitized = dict(current_config)
                baseline_cmd = self.builder.build_mdtest_command(sanitized, output_dir=job_scratch)
                errs = []
            elif benchmark_type == "hacc_io":
                valid, sanitized, errs = self.builder.validate_hacc_params(current_config)
                baseline_cmd = self.builder.build_hacc_command(sanitized, output_dir=job_scratch)
            elif benchmark_type == "custom":
                valid, sanitized, errs = self.builder.validate_custom_params(current_config)
                baseline_cmd = self.builder.build_custom_command(sanitized, output_dir=job_scratch)
            else:
                valid, sanitized, errs = self.builder.validate_ior_params(current_config)
                baseline_cmd = self.builder.build_ior_command(sanitized, output_dir=job_scratch)

            exec_kwargs = {"job_name": job_base, "benchmark_type": benchmark_type}
            if benchmark_type == "hacc_io":
                exec_kwargs["hacc_config"] = sanitized
            baseline_result = self.executor.execute_benchmark(
                baseline_cmd,
                **exec_kwargs,
            )

            if not baseline_result["success"]:
                logger.error("  Baseline execution failed!")
                history["final_status"] = "baseline_failed"
                return history

            baseline_features = baseline_result["features"]
            baseline_metrics = baseline_result["metrics"]
            baseline_bw = baseline_metrics.get("write_bw_mb_s", 0) or baseline_metrics.get("total_bw_mb_s", 0.001)
            history["baseline_bw"] = baseline_bw
            history["baseline_metrics"] = baseline_metrics
            logger.info("  Baseline BW: %.2f MB/s", baseline_bw)
        else:
            # Dry run: use features from test data
            baseline_features = self._load_test_features(workload_name)
            baseline_metrics = {k: baseline_features.get(k, 0) for k in [
                "total_bw_mb_s", "write_bw_mb_s", "avg_write_size",
                "small_io_ratio", "seq_write_ratio", "metadata_time_ratio",
                "POSIX_BYTES_WRITTEN", "POSIX_WRITES", "POSIX_FSYNCS", "nprocs",
            ]}
            baseline_bw = baseline_metrics.get("write_bw_mb_s", 0.001) or 0.001
            history["baseline_bw"] = baseline_bw
            history["baseline_metrics"] = baseline_metrics
            logger.info("  [DRY RUN] Baseline features loaded")

        # Initial ML detection
        predictions, detected = self.detect_bottlenecks(baseline_features)
        logger.info("  Initial detection: %s", detected)

        if "healthy" in detected and len(detected) == 1:
            logger.info("  No bottlenecks detected -- already healthy")
            history["final_status"] = "already_healthy"
            return history

        # SHAP features
        shap_features = self.get_shap_features(baseline_features, detected)

        # KB evidence
        kb_matches = self.retrieve_kb_evidence(detected, baseline_features)
        logger.info("  KB matches: %d", len(kb_matches))

        # Iteration state
        best_features = baseline_features.copy()
        best_config = dict(current_config)
        best_speedup = 1.0
        best_bw = baseline_bw
        current_features = baseline_features.copy()
        rollback = False

        plateau_threshold = self.iter_config.get("iteration", {}).get("plateau_threshold", 0.05)
        convergence_threshold = self.iter_config.get("iteration", {}).get("convergence_threshold", 0.3)

        for iteration in range(self.max_iterations):
            logger.info("")
            logger.info("--- Iteration %d/%d ---", iteration + 1, self.max_iterations)

            # Build prompt
            sys_p, usr_p = self.build_prompt(
                iteration=iteration,
                workload_config=workload_config,
                detected_dims=detected,
                predictions=predictions,
                shap_features=shap_features,
                kb_matches=kb_matches,
                darshan_before=baseline_metrics,
                darshan_after=current_features if iteration > 0 else None,
                current_config=current_config,
                best_speedup=best_speedup if iteration > 0 else None,
                rollback=rollback,
            )

            # Call LLM (with retries for parse failures)
            max_retries = self.iter_config.get("iteration", {}).get("max_parse_retries", 3)
            parsed = None
            parse_error = None
            metadata = {}

            for retry in range(max_retries):
                response_text, metadata = self.call_llm(sys_p, usr_p)
                parsed, parse_error = self.parse_llm_response(response_text)
                if parsed:
                    break
                logger.warning("  Parse retry %d/%d", retry + 1, max_retries)
                # Clear cache for retry with slightly modified prompt
                if retry < max_retries - 1:
                    sys_p += f"\n(Retry {retry + 1}: please ensure valid JSON output)"

            iteration_record = {
                "iteration": iteration,
                "detected_dims": detected,
                "predictions": {d: predictions.get(d, 0) for d in detected},
                "strategy": parsed.get("strategy", "unknown") if parsed else "parse_error",
                "config_changes": parsed.get("config_changes", {}) if parsed else {},
                "changes_made": parsed.get("changes_made", []) if parsed else [],
                "kb_citations": parsed.get("kb_citations", []) if parsed else [],
                "metadata": metadata,
                "parse_error": parse_error is not None,
            }

            if not parsed:
                logger.error("  Failed to parse LLM response after %d retries", max_retries)
                iteration_record["executed"] = False
                history["iterations"].append(iteration_record)
                continue

            logger.info("  Strategy: %s", parsed.get("strategy", "?")[:80])
            logger.info("  Changes: %s", parsed.get("changes_made", [])[:3])

            # Apply config changes
            config_changes = self.builder.parse_llm_config_changes(parsed)
            if not config_changes:
                config_changes = parsed.get("config_changes", {})

            new_config = self.builder.apply_changes_to_config(current_config, config_changes)

            # Validate and build command
            if benchmark_type == "mdtest":
                sanitized = dict(new_config)
                errs = []
                iteration_record["validated_config"] = sanitized
                iteration_record["validation_errors"] = errs
            elif benchmark_type == "hacc_io":
                valid, sanitized, errs = self.builder.validate_hacc_params(new_config)
                if errs:
                    logger.warning("  Config validation warnings: %s", errs[:3])
                iteration_record["validated_config"] = sanitized
                iteration_record["validation_errors"] = errs
            elif benchmark_type == "custom":
                valid, sanitized, errs = self.builder.validate_custom_params(new_config)
                if errs:
                    logger.warning("  Config validation warnings: %s", errs[:3])
                iteration_record["validated_config"] = sanitized
                iteration_record["validation_errors"] = errs
            else:
                valid, sanitized, errs = self.builder.validate_ior_params(new_config)
                if errs:
                    logger.warning("  Config validation warnings: %s", errs[:3])
                iteration_record["validated_config"] = sanitized
                iteration_record["validation_errors"] = errs

            # Execute
            if not self.dry_run:
                iter_job = f"iter_{workload_name}_r{run_id}_i{iteration}"
                iter_scratch = f"{self.iter_config['slurm']['scratch_dir']}/{iter_job}"
                if benchmark_type == "mdtest":
                    cmd = self.builder.build_mdtest_command(sanitized, output_dir=iter_scratch)
                elif benchmark_type == "hacc_io":
                    cmd = self.builder.build_hacc_command(sanitized, output_dir=iter_scratch)
                elif benchmark_type == "custom":
                    cmd = self.builder.build_custom_command(sanitized, output_dir=iter_scratch)
                else:
                    cmd = self.builder.build_ior_command(sanitized, output_dir=iter_scratch)
                logger.info("  Executing: %s", cmd[:120])

                exec_kwargs = {"job_name": iter_job, "benchmark_type": benchmark_type}
                if benchmark_type == "hacc_io":
                    exec_kwargs["hacc_config"] = sanitized
                exec_result = self.executor.execute_benchmark(
                    cmd,
                    **exec_kwargs,
                )

                iteration_record["executed"] = exec_result["success"]
                iteration_record["execution_time_s"] = exec_result["elapsed_s"]
                history["total_execution_time_s"] += exec_result["elapsed_s"]

                if exec_result["success"]:
                    new_features = exec_result["features"]
                    new_metrics = exec_result["metrics"]
                    new_bw = new_metrics.get("write_bw_mb_s", 0) or new_metrics.get("total_bw_mb_s", 0.001)
                    speedup = round(new_bw / baseline_bw, 2)

                    logger.info("  Result: BW=%.2f MB/s, speedup=%.2fx", new_bw, speedup)

                    # Re-detect
                    new_predictions, new_detected = self.detect_bottlenecks(new_features)
                    iteration_record["new_predictions"] = new_predictions
                    iteration_record["new_detected"] = new_detected
                    iteration_record["speedup"] = speedup
                    iteration_record["new_bw"] = new_bw

                    # Check for regression
                    if speedup > best_speedup:
                        best_speedup = speedup
                        best_config = dict(sanitized)
                        best_features = new_features.copy()
                        best_bw = new_bw
                        history["best_iteration"] = iteration
                        rollback = False
                        logger.info("  NEW BEST: %.2fx at iteration %d", speedup, iteration)
                    elif new_bw < baseline_bw * 0.9:
                        # Regression: new is worse than baseline
                        logger.warning("  REGRESSION: %.2fx -- rolling back", speedup)
                        iteration_record["rollback"] = True
                        rollback = True
                        # Rollback: use best config as starting point for next iteration
                        current_config = dict(best_config)
                        current_features = best_features.copy()
                    else:
                        rollback = False

                    if not rollback:
                        current_config = dict(sanitized)
                        current_features = new_features.copy()
                        predictions = new_predictions
                        detected = new_detected

                    # SHAP on new features
                    if self.use_shap and not rollback:
                        shap_features = self.get_shap_features(new_features, new_detected)

                else:
                    logger.error("  Execution FAILED")
                    iteration_record["executed"] = False
            else:
                # Dry run
                iteration_record["executed"] = False
                iteration_record["simulated"] = True
                logger.info("  [DRY RUN] Proposed config: %s", sanitized)

            history["iterations"].append(iteration_record)
            history["total_llm_latency_ms"] += metadata.get("latency_ms", 0)

            # Convergence check
            remaining = [d for d in detected if d != "healthy" and predictions.get(d, 0) > convergence_threshold]
            if not remaining:
                logger.info("  CONVERGED: all bottleneck confidences below %.1f", convergence_threshold)
                history["final_status"] = "converged"
                break

            # Plateau check (only after iteration 1, only if execution succeeded)
            if iteration > 0 and not rollback and iteration_record.get("executed"):
                prev_speedup = history["iterations"][-2].get("speedup", 1.0) if len(history["iterations"]) > 1 else 1.0
                cur_speedup = iteration_record.get("speedup", 1.0)
                if abs(cur_speedup - prev_speedup) / max(prev_speedup, 0.001) < plateau_threshold:
                    logger.info("  PLATEAU: <%.0f%% improvement, stopping", plateau_threshold * 100)
                    history["final_status"] = "plateau"
                    break
        else:
            history["final_status"] = "max_iterations_reached"

        # Final summary
        history["best_speedup"] = best_speedup
        history["best_config"] = best_config
        history["total_iterations"] = len(history["iterations"])
        history["total_tokens"] = self.total_tokens_input + self.total_tokens_output
        history["total_cost_usd"] = self.compute_cost_usd()

        logger.info("")
        logger.info("=" * 70)
        logger.info(
            "RESULT: %s -- %d iterations, best=%.2fx, status=%s, cost=$%.4f",
            workload_name, history["total_iterations"],
            best_speedup, history["final_status"], history["total_cost_usd"],
        )
        logger.info("=" * 70)

        return history

    def _load_test_features(self, workload_name):
        """Load test features for dry-run mode (first sample with matching bottleneck)."""
        import pandas as pd

        test_feat = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "benchmark" / "test_features.parquet")
        test_labels = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "benchmark" / "test_labels.parquet")

        workload_config = self.iter_config["workloads"][workload_name]
        bottleneck = workload_config.get("bottleneck", "access_granularity")

        if bottleneck in test_labels.columns:
            mask = test_labels[bottleneck] == 1
            if mask.any():
                idx = test_feat.index[mask.values][0]
                return test_feat.iloc[idx].to_dict()

        # Fallback: first non-healthy sample
        return test_feat.iloc[0].to_dict()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Track C: ML-Guided Iterative LLM Code Optimization"
    )
    parser.add_argument("--workload", default=None, help="Workload name from iterative.yaml")
    parser.add_argument("--model", default="claude-sonnet",
                        choices=["claude-sonnet", "gpt-4o", "llama-70b"])
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=1, help="Repetitions per workload")
    parser.add_argument("--sweep", action="store_true", help="Run all workloads")
    parser.add_argument("--dry-run", action="store_true", help="Simulate (no SLURM)")
    parser.add_argument("--no-ml", action="store_true", help="Ablation: disable ML")
    parser.add_argument("--no-shap", action="store_true", help="Ablation: disable SHAP")
    parser.add_argument("--no-kb", action="store_true", help="Ablation: disable KB")
    parser.add_argument("--no-feedback", action="store_true", help="Ablation: no iteration feedback")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    optimizer = IterativeOptimizer(
        model=args.model,
        max_iterations=args.max_iterations,
        use_ml=not args.no_ml,
        use_shap=not args.no_shap,
        use_kb=not args.no_kb,
        use_feedback=not args.no_feedback,
        dry_run=args.dry_run,
    )

    # Determine workloads
    if args.sweep:
        workloads = list(optimizer.iter_config["workloads"].keys())
    elif args.workload:
        workloads = [args.workload]
    else:
        workloads = ["ior_small_posix"]  # Default test workload

    logger.info("Workloads: %s", workloads)
    logger.info("Model: %s, Runs: %d, Max iterations: %d",
                args.model, args.n_runs, args.max_iterations)

    all_results = []
    for workload in workloads:
        for run in range(args.n_runs):
            result = optimizer.run_optimization(workload, run_id=run)
            all_results.append(result)

    # Save results
    output_path = args.output or str(
        PROJECT_DIR / "results" / "iterative" / f"iterative_results_{args.model}_{int(time.time())}.json"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved: %s", output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("ITERATIVE OPTIMIZATION SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"  {r.get('workload','?'):30s} run={r.get('run_id',0)} "
              f"iters={r.get('total_iterations',0)} speedup={r.get('best_speedup',1.0):.2f}x "
              f"status={r.get('final_status','?')} cost=${r.get('total_cost_usd',0):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
