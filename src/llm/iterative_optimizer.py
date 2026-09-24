"""ML-guided benchmark parameter experiment with Darshan feedback.

Architecture (research-backed -- STELLAR/PerfCoder/POLO/Self-Refine):
  1. ML Classifier detects bottleneck type + SHAP features
  2. KB Retriever finds matching benchmark fix patterns
  3. LLM proposes a benchmark configuration
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
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

def condition_name(*, use_ml, use_shap, use_kb, use_feedback, max_iterations):
    """Return the declared experiment condition for one supported configuration."""
    disabled = []
    if not use_ml:
        disabled.append("no_ml")
    elif not use_shap:
        disabled.append("no_shap")
    if not use_kb:
        disabled.append("no_kb")
    if not use_feedback:
        disabled.append("no_feedback")
    if len(disabled) > 1:
        raise ValueError(f"combined ablation switches are unsupported: {disabled}")
    if disabled:
        if max_iterations == 1:
            raise ValueError("single-shot cannot be combined with a component ablation")
        return disabled[0]
    return "single_shot" if max_iterations == 1 else "full"

# OpenRouter model IDs (same as single-shot recommender)
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


def supported_workloads(config):
    """Return workload names whose constructed label is supported."""
    return [
        name for name, workload in config["workloads"].items()
        if workload.get("classifier_supported", True)
    ]


class IterativeOptimizer:
    """ML-guided benchmark parameter experiment with Darshan feedback.

    The core loop:
        detect(darshan) -> explain(shap) -> retrieve(kb) ->
        generate_config(llm) -> validate(builder) -> execute(slurm) ->
        re_detect(darshan) -> evaluate -> iterate_or_stop
    """

    def __init__(self, config_path=None, model_path=None, kb_path=None,
                 model="claude-sonnet",
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
        self.model_path = model_path
        self.kb_path = kb_path
        self.condition = condition_name(
            use_ml=use_ml,
            use_shap=use_shap,
            use_kb=use_kb,
            use_feedback=use_feedback,
            max_iterations=max_iterations,
        )

        # Load configs
        iter_config_path = config_path or PROJECT_DIR / "configs" / "iterative.yaml"
        with open(iter_config_path) as f:
            self.iter_config = yaml.safe_load(f)

        # Defaults for optional components
        self.models = {}
        self.feature_cols = []
        self.explainer = None
        self.kb = []
        self.detector = None
        self.retriever = None

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
        """Load the same final-evaluation bundle used by IOPrescriber."""
        if not self.model_path:
            raise ValueError("model_path is required when ML detection is enabled")
        from src.ioprescriber.detector import Detector
        self.detector = Detector(self.model_path)
        self.models = self.detector.models
        self.feature_cols = self.detector.feature_cols
        logger.info("  ML models loaded: %d dimensions, %d features", len(self.models), len(self.feature_cols))

    def _load_shap_explainer(self):
        """Load SHAP TreeExplainers for per-dimension attribution."""
        from src.ioprescriber.explainer import Explainer
        self.explainer = Explainer(models=self.models, feature_cols=self.feature_cols)
        logger.info("  SHAP explainer loaded")

    def _load_knowledge_base(self):
        """Load the benchmark knowledge base."""
        if not self.kb_path:
            raise ValueError("kb_path is required when KB retrieval is enabled")
        from src.ioprescriber.retriever import Retriever
        self.retriever = Retriever(self.kb_path)
        self.kb = self.retriever.kb
        logger.info("  KB loaded: %d entries", len(self.kb))

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

        return self.detector.detect_from_features(features_dict)

    def get_shap_features(self, features_dict, detected_dims):
        """Get SHAP top features per detected dimension.

        Returns:
            dict {dimension: [list of {feature, value, abs_importance, direction}]}
        """
        if not self.use_shap or not self.explainer:
            return {}

        X = self.detector.feature_vector(features_dict)
        return self.explainer.explain(X, detected_dims)

    # =========================================================================
    # KB Retrieval
    # =========================================================================

    def retrieve_kb_evidence(self, detected_dims, features_dict, top_k=3):
        """Retrieve matching KB entries for detected bottlenecks."""
        if not self.use_kb:
            return []

        return self.retriever.retrieve(detected_dims, features_dict, top_k=top_k)

    # =========================================================================
    # LLM Prompt Building
    # =========================================================================

    def build_prompt(self, iteration, workload_config, detected_dims, predictions,
                     shap_features, kb_matches, darshan_before, darshan_after=None,
                     current_config=None, best_speedup=None, rollback=False,
                     work_changed=False):
        """Build the benchmark parameter experiment prompt.

        Key differences from single-shot recommendation:
        - Asks LLM to output benchmark CONFIG CHANGES (not arbitrary code)
        - Includes iteration feedback (before/after Darshan)
        - Includes SHAP features for targeted guidance
        - Rollback hint when regression detected
        """
        system_prompt = """You are an HPC I/O measurement expert.
You propose benchmark parameter changes for a parameter-tuning experiment. This
experiment does not generate or validate application source-code fixes.

RULES:
1. Output ONLY benchmark parameter changes in the specified JSON format.
2. KB citations support only the detected bottleneck dimension. They do not
   validate a proposed benchmark parameter change.
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
8. The transfer_size (-t) controls I/O granularity and file-offset alignment. The block_size
   (-b) and segments (-s) control total data volume. Only change -t to fix
   access_granularity. Use a transfer size that divides the block size to fix
   request_alignment. Only change -a/-c to fix interface_choice. Only remove -Y to fix
   throughput_utilization. Only remove -z to fix access_pattern.
9. For h5bench (HDF5) benchmarks:
   - DIM_1: integer [64, 16777216] (elements per rank per timestep, 8 bytes each)
   - COLLECTIVE_DATA: 'YES' or 'NO' (HDF5 collective I/O)
   - COLLECTIVE_METADATA: 'YES' or 'NO' (HDF5 collective metadata)
   - TIMESTEPS: integer [1, 100] (number of write/read timesteps)
   - MEM_PATTERN: 'CONTIG' or 'INTERLEAVED' (memory layout)
   - FILE_PATTERN: 'CONTIG' or 'INTERLEAVED' (file layout)
   - To fix interface_choice: enable COLLECTIVE_DATA=YES
   - Keep DIM_1 and TIMESTEPS constant to maintain the same work
11. For DLIO (ML I/O) benchmarks:
   - record_length: integer [64, 16777216] (bytes per training sample)
   - num_files_train: integer [10, 10000] (number of training data files)
   - num_samples_per_file: integer [1, 1000] (samples per file)
   - batch_size: integer [1, 256] (training batch size)
   - read_threads: integer [1, 16] (data loading threads)
   - computation_time: float [0.0, 10.0] (simulated compute time per batch)
   - epochs: integer [1, 10] (training epochs)
   - format: 'npz', 'hdf5', 'csv', or 'tfrecord' (data format)
   - sample_shuffle: 'off', 'random', or 'seed' (sample ordering)
   - file_shuffle: 'off', 'random', or 'seed' (file ordering)
   - To fix access_pattern: set sample_shuffle=off, file_shuffle=off
   - To fix throughput_utilization: increase batch_size and read_threads
12. For HACC-IO benchmarks:
   - executable: 'posix_shared', 'mpiio_shared', or 'fpp' (I/O backend)
   - num_particles: integer [50, 10000000] (data volume = 38 bytes * num_particles per rank)
   - collective_buffering: 'enabled' or 'disabled' (ROMIO aggregation control)
   - To fix interface_choice: switch from posix_shared to mpiio_shared
   - To fix file_strategy: switch from fpp to mpiio_shared
   - Keep num_particles constant to maintain the same work
13. For custom (load_imbalance) benchmarks:
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
                kb_str += f"    Dimensions: {', '.join(e['bottleneck_labels'])}\n"
                source = e["source_code"]
                kb_str += (f"    Source: {source['repository']} at {source['revision']}\n"
                           f"    Path: {source['path']}\n")

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

## Detected Bottlenecks (ML classifier):
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
            if work_changed:
                user_prompt += """
## WARNING: Previous proposal CHANGED THE WORKLOAD (bytes or items moved differ from the
## baseline by more than the tolerance). It was rejected. Keep the amount of data, the
## number of items/files and the number of steps identical; change only how the I/O is done.
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
  "evidence_scope": "KB supports diagnosis only",
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
  "evidence_scope": "KB supports diagnosis only",
  "changes_made": ["list of specific changes and WHY they help"],
  "kb_citations": ["list of KB entry IDs used"]
}
"""
        elif benchmark_type == "h5bench":
            user_prompt += """
## Task:
Propose h5bench (HDF5) parameter changes to fix the detected bottlenecks.
Respond in JSON:
{
  "strategy": "brief description of optimization strategy",
  "config_changes": {
    "DIM_1": "elements per rank per timestep (optional)",
    "COLLECTIVE_DATA": "YES or NO (optional)",
    "COLLECTIVE_METADATA": "YES or NO (optional)",
    "TIMESTEPS": "number of timesteps (optional)",
    "MEM_PATTERN": "CONTIG or INTERLEAVED (optional)",
    "FILE_PATTERN": "CONTIG or INTERLEAVED (optional)"
  },
  "evidence_scope": "KB supports diagnosis only",
  "changes_made": ["list of specific changes and WHY they help"],
  "kb_citations": ["list of KB entry IDs used"]
}
"""
        elif benchmark_type == "dlio":
            user_prompt += """
## Task:
Propose DLIO (ML I/O) parameter changes to fix the detected bottlenecks.
Respond in JSON:
{
  "strategy": "brief description of optimization strategy",
  "config_changes": {
    "record_length": "bytes per sample (optional)",
    "num_files_train": "number of training files (optional)",
    "num_samples_per_file": "samples per file (optional)",
    "batch_size": "training batch size (optional)",
    "read_threads": "data loading threads (optional)",
    "computation_time": "simulated compute seconds per batch (optional)",
    "format": "npz or hdf5 or csv or tfrecord (optional)",
    "sample_shuffle": "off or random or seed (optional)",
    "file_shuffle": "off or random or seed (optional)"
  },
  "evidence_scope": "KB supports diagnosis only",
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
  "evidence_scope": "KB supports diagnosis only",
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
  "evidence_scope": "KB supports diagnosis only",
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
        cache_request = {
            "schema_version": 2,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "user_prompt_sha256": hashlib.sha256(user_prompt.encode()).hexdigest(),
            "model": self.model_id,
            "resolved_model": self.model_id,
            "temperature": self.temperature,
            "max_tokens": 2000,
            "endpoint": "https://openrouter.ai/api/v1",
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        cache_path = Path(self.cache_dir) / f"{cache_key}.json"

        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("request") != cache_request:
                raise ValueError(f"iterative cache contract mismatch: {cache_path}")
            self._validate_iterative_response(json.loads(cached["response"]))
            logger.info("  Cache hit: %s", cache_path.name[:12])
            source = cached.get("metadata", {})
            return cached["response"], {
                "model": source.get("model", self.model_id),
                "resolved_model": source.get("resolved_model"),
                "cache_hit": True,
                "api_latency_ms": 0.0,
                "tokens_input": 0,
                "tokens_output": 0,
                "request_id": None,
                "cache_source_request_id": source.get("request_id"),
            }

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
        resolved_model = getattr(response, "model", None)
        if resolved_model != self.model_id:
            raise ValueError(
                f"provider resolved {self.model_id} to {resolved_model}; use an exact model ID")

        metadata = {
            "model": self.model_id,
            "resolved_model": resolved_model,
            "cache_hit": False,
            "api_latency_ms": round(latency_ms, 1),
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "request_id": getattr(response, "id", None),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        self._validate_iterative_response(json.loads(text))

        self.total_tokens_input += tokens_in
        self.total_tokens_output += tokens_out

        # Cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({
                "cache_schema_version": 2,
                "request": cache_request,
                "response": text, "metadata": metadata,
            }, f, indent=2)

        return text, metadata

    @staticmethod
    def _validate_iterative_response(parsed):
        if not isinstance(parsed, dict):
            raise ValueError("iterative response must be an object")
        if not isinstance(parsed.get("strategy"), str) or not parsed["strategy"]:
            raise ValueError("iterative response needs a strategy")
        if not isinstance(parsed.get("config_changes"), dict):
            raise ValueError("iterative config_changes must be an object")
        for key in ("changes_made", "kb_citations"):
            value = parsed.get(key)
            if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
                raise ValueError(f"iterative {key} must be a string list")
        if parsed.get("evidence_scope") != "KB supports diagnosis only":
            raise ValueError("iterative response must limit KB evidence to diagnosis")
        return parsed

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
            return self._validate_iterative_response(json.loads(text.strip())), None
        except (json.JSONDecodeError, IndexError, ValueError) as e:
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

    def _execute_repeated(self, cmd, exec_kwargs, repeats):
        """Run one configuration ``repeats`` times; median wall time decides.

        Returns (first successful executor result, aggregated measurement,
        list of per-run measurements).  A single run keeps the old behaviour
        except that the objective is wall time (closed_loop_metrics).
        """
        from .closed_loop_metrics import aggregate_repeats
        first, measurements, allocation_elapsed = None, [], 0.0
        self._last_execution_allocation_s = 0.0
        for k in range(max(1, int(repeats))):
            # The job name must stay identical across repeats: the executor derives the
            # per-job scratch directory from it, and that directory has to match the output
            # path already baked into ``cmd``. Renaming a repeat made the job create (and
            # delete) a different directory while the benchmark still wrote into the first
            # job's, which failed with ENOENT. Repeats run sequentially, so one directory is
            # enough; SLURM job ids keep the logs and Darshan files apart.
            kw = dict(exec_kwargs)
            res = self.executor.execute_benchmark(cmd, **kw)
            allocation_elapsed += float(res.get("elapsed_s", 0.0))
            self._last_execution_allocation_s = allocation_elapsed
            if res.get("success") and res.get("measurement"):
                measurements.append(res["measurement"])
                if first is None:
                    first = res
            else:
                logger.warning("  repeat %d failed (job %s)", k, res.get("job_id"))
                return None, None, []
        confidence = float(self.iter_config.get("iteration", {}).get("confidence", 0.90))
        aggregate = aggregate_repeats(measurements, confidence)
        aggregate["allocation_elapsed_s"] = allocation_elapsed
        return first, aggregate, measurements

    def _execute_interleaved(self, control_cmd, control_kwargs, cand_cmd, cand_kwargs, repeats):
        """Run control and candidate alternately: control, candidate, control, candidate, ...

        The median wall time of an unchanged configuration drifts with cluster load over hours
        (measured on Delta: 13.6 s, 20.3 s and 17.0 s for the same IOR run at different times
        of one night), far more than the gain threshold. A baseline measured once at the start
        is therefore not a valid reference for a candidate measured later. Interleaving gives
        both configurations the same time window, so slow drift affects them equally, while
        the order-statistic interval of each median absorbs the short bursts.

        Returns (first successful candidate result, candidate aggregate, control aggregate).
        """
        from .closed_loop_metrics import aggregate_repeats
        confidence = float(self.iter_config.get("iteration", {}).get("confidence", 0.90))
        cand_first, cand_meas, ctrl_meas = None, [], []
        cand_elapsed = ctrl_elapsed = 0.0
        self._last_execution_allocation_s = 0.0
        for k in range(max(1, int(repeats))):
            for label, cmd, kw, sink in (("control", control_cmd, control_kwargs, ctrl_meas),
                                         ("candidate", cand_cmd, cand_kwargs, cand_meas)):
                res = self.executor.execute_benchmark(cmd, **dict(kw))
                if label == "candidate":
                    cand_elapsed += float(res.get("elapsed_s", 0.0))
                else:
                    ctrl_elapsed += float(res.get("elapsed_s", 0.0))
                self._last_execution_allocation_s = cand_elapsed + ctrl_elapsed
                if res.get("success") and res.get("measurement"):
                    sink.append(res["measurement"])
                    if label == "candidate" and cand_first is None:
                        cand_first = res
                else:
                    logger.warning("  %s run %d failed (job %s)", label, k, res.get("job_id"))
                    return None, None, None
        candidate = aggregate_repeats(cand_meas, confidence)
        control = aggregate_repeats(ctrl_meas, confidence)
        candidate["allocation_elapsed_s"] = cand_elapsed
        control["allocation_elapsed_s"] = ctrl_elapsed
        return cand_first, candidate, control

    def run_optimization(self, workload_name, run_id=0):
        """Run the full iterative optimization loop for one workload.

        Args:
            workload_name: key from configs/iterative.yaml workloads
            run_id: repetition ID for multi-run experiments

        Returns:
            dict with full optimization history, metrics, and cost
        """
        workload_config = self.iter_config["workloads"][workload_name]
        if not workload_config.get("classifier_supported", True):
            raise ValueError(
                f"workload {workload_name} is excluded by the classifier-label audit")
        bad_config = dict(workload_config["bad_config"])
        from .closed_loop_metrics import work_params_changed
        reference_config = workload_config.get("known_good_config")
        if reference_config:
            inconsistent = work_params_changed(
                workload_config.get("benchmark", "ior"), bad_config, reference_config)
            if inconsistent:
                raise ValueError(
                    f"workload {workload_name} reference changes work parameters {inconsistent}; "
                    "define an equal-work reference before running it")

        logger.info("=" * 70)
        logger.info("ITERATIVE PARAMETER EXPERIMENT: %s (run %d)", workload_name, run_id)
        logger.info("  Bottleneck: %s", workload_config.get("bottleneck"))
        logger.info("  Description: %s", workload_config.get("description"))
        logger.info("=" * 70)

        # Reset cost tracking for this run
        self.total_tokens_input = 0
        self.total_tokens_output = 0

        repeats = int(self.iter_config.get("iteration", {}).get("repeats", 8)) if not self.dry_run else 1
        history = {
            "schema_version": 2,
            "workload": workload_name,
            "run_id": run_id,
            "condition": self.condition,
            "result_id": (
                f"{workload_name}:{self.model_key}:{self.condition}:{run_id}"
            ),
            "model": self.model_key,
            "model_id": self.model_id,
            "max_iterations": self.max_iterations,
            "config": {
                "use_ml": self.use_ml, "use_shap": self.use_shap,
                "use_kb": self.use_kb, "use_feedback": self.use_feedback,
                "dry_run": self.dry_run,
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
        slurm_resources = {
            "nodes": int(self.iter_config["slurm"]["nodes"]),
            "ntasks": int(self.iter_config["slurm"]["ntasks"]),
            "cpus_per_task": int(self.iter_config["slurm"].get("cpus_per_task", 1)),
            "walltime": str(self.iter_config["slurm"]["walltime"]),
        }
        slurm_resources.update(workload_config.get("slurm_override", {}))
        history["slurm_resources"] = slurm_resources

        # Step 1: Execute the "bad" baseline config to get initial Darshan
        logger.info("  Step 1: Running baseline (bad) config...")
        current_config = dict(bad_config)

        if not self.dry_run:
            # Include model key in job name to avoid scratch dir collisions
            # when the same workload runs concurrently with different LLMs
            model_short = self.model_key.replace("claude-sonnet", "claude").replace("llama-70b", "llama")
            job_base = f"iter_{workload_name}_r{run_id}_{model_short}_baseline"
            job_scratch = f"{self.iter_config['slurm']['scratch_dir']}/{job_base}"
            if benchmark_type == "mdtest":
                valid, sanitized, errs = self.builder.validate_mdtest_params(current_config)
                baseline_cmd = self.builder.build_mdtest_command(sanitized, output_dir=job_scratch) if valid else None
            elif benchmark_type == "hacc_io":
                valid, sanitized, errs = self.builder.validate_hacc_params(current_config)
                baseline_cmd = self.builder.build_hacc_command(sanitized, output_dir=job_scratch)
            elif benchmark_type == "custom":
                valid, sanitized, errs = self.builder.validate_custom_params(current_config)
                baseline_cmd = self.builder.build_custom_command(sanitized, output_dir=job_scratch)
            elif benchmark_type == "h5bench":
                valid, sanitized, errs = self.builder.validate_h5bench_params(current_config)
                config_path = os.path.join(
                    self.executor.results_dir, f"{job_base}_config.json"
                )
                baseline_cmd = self.builder.build_h5bench_config(
                    sanitized, output_dir=job_scratch, config_path=config_path
                )
                # baseline_cmd is (write_cmd, read_cmd, config_path) for h5bench
                baseline_cmd = (baseline_cmd[0], baseline_cmd[1])
            elif benchmark_type == "dlio":
                valid, sanitized, errs = self.builder.validate_dlio_params(current_config)
                baseline_cmd = self.builder.build_dlio_command(
                    sanitized, data_dir=job_scratch
                )
                # baseline_cmd is (datagen_cmd, training_cmd) for DLIO
            else:
                valid, sanitized, errs = self.builder.validate_ior_params(current_config)
                baseline_cmd = self.builder.build_ior_command(sanitized, output_dir=job_scratch)

            if not valid:
                raise ValueError(f"baseline configuration is invalid: {errs}")

            exec_kwargs = {"job_name": job_base, "benchmark_type": benchmark_type,
                           "slurm_resources": slurm_resources}
            if benchmark_type == "hacc_io":
                exec_kwargs["hacc_config"] = sanitized
            elif benchmark_type == "h5bench":
                exec_kwargs["h5bench_config"] = sanitized
            elif benchmark_type == "dlio":
                exec_kwargs["dlio_config"] = sanitized
            exec_kwargs.update({"workload": workload_name, "work_config": sanitized})
            baseline_exec_kwargs = dict(exec_kwargs)
            baseline_result, baseline_meas, baseline_runs = self._execute_repeated(
                baseline_cmd, exec_kwargs, repeats)

            if baseline_result is None or baseline_meas is None:
                logger.error("  Baseline execution failed!")
                history["total_execution_time_s"] += getattr(
                    self, "_last_execution_allocation_s", 0.0)
                history["final_status"] = "baseline_failed"
                return history

            baseline_features = baseline_result["features"]
            baseline_metrics = baseline_result["metrics"]
            baseline_bw = baseline_meas.get("write_bw_mb_s") or baseline_metrics.get("total_bw_mb_s", 0.001)
            history["baseline_bw"] = baseline_bw
            history["baseline_metrics"] = baseline_metrics
            history["baseline_walltime_s"] = baseline_meas["walltime_s"]
            history["baseline_measurement"] = baseline_meas
            history["baseline_execution_time_s"] = baseline_meas.get("allocation_elapsed_s", 0.0)
            history["total_execution_time_s"] += history["baseline_execution_time_s"]
            history["baseline_darshan_paths"] = baseline_result.get("darshan_paths", [])
            logger.info("  Baseline wall time: %.2f s (median of %d, %.0f%% CI [%.2f, %.2f], relMAD %.1f%%, "
                        "range %.0f%%), BW %.2f MB/s",
                        baseline_meas["walltime_s"], baseline_meas["n_repeats"],
                        100 * baseline_meas["ci_coverage"], baseline_meas["ci_lower_s"],
                        baseline_meas["ci_upper_s"], 100 * baseline_meas["rel_mad"],
                        100 * baseline_meas["spread_rel"], baseline_bw)
            if not baseline_meas["ci_valid"]:
                logger.warning("  Only %d usable baseline runs: no valid confidence interval, so no "
                               "candidate can be accepted on timing", baseline_meas["n_repeats"])
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
            baseline_meas = {
                "walltime_s": float(baseline_features.get("runtime_seconds", 0) or 1.0),
                "write_bw_mb_s": baseline_bw,
                "bytes_total": float(baseline_features.get("POSIX_BYTES_WRITTEN", 0) or 0),
                "spread_rel": 0.0, "n_repeats": 1,
            }
            history["baseline_walltime_s"] = baseline_meas["walltime_s"]
            logger.info("  [DRY RUN] Baseline features loaded")

        # Initial ML detection
        predictions, detected = self.detect_bottlenecks(baseline_features)
        logger.info("  Initial detection: %s", detected)

        if "healthy" in detected and len(detected) == 1:
            logger.info("  No bottlenecks detected -- already healthy")
            history["final_status"] = "already_healthy"
            history["total_iterations"] = 0
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
        current_features = baseline_features.copy()
        rollback = False

        plateau_threshold = self.iter_config.get("iteration", {}).get("plateau_threshold", 0.05)
        it_cfg = self.iter_config.get("iteration", {})
        work_tolerance = it_cfg.get("work_tolerance", 0.25)
        regression_factor = it_cfg.get("regression_factor", 0.9)
        min_gain = it_cfg.get("min_gain", 0.05)
        history["objective"] = {"metric": "walltime_s (phase-summed Darshan run_time, median of repeats)",
                                "decision": "candidate accepted only if the order-statistic confidence "
                                            "interval of its median lies entirely below the baseline's",
                                "confidence": float(it_cfg.get("confidence", 0.90)),
                                "repeats": repeats, "work_tolerance": work_tolerance,
                                "regression_factor": regression_factor, "min_gain": min_gain}
        last_work_changed = False
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
                work_changed=last_work_changed,
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

            if self.use_kb:
                retrieved = {match["entry"]["entry_id"]: match["entry"] for match in kb_matches}
                citations = parsed["kb_citations"]
                if not citations or len(citations) != len(set(citations)):
                    iteration_record["executed"] = False
                    iteration_record["evidence_error"] = "citations must be a nonempty unique list"
                    history["iterations"].append(iteration_record)
                    history["total_llm_latency_ms"] += metadata.get("api_latency_ms", 0)
                    continue
                unknown = set(citations) - set(retrieved)
                unrelated = [entry_id for entry_id in citations
                             if not (set(retrieved[entry_id]["bottleneck_labels"]) & set(detected))]
                if unknown or unrelated:
                    iteration_record["executed"] = False
                    iteration_record["evidence_error"] = {
                        "unknown_citations": sorted(unknown),
                        "unrelated_citations": sorted(unrelated),
                    }
                    history["iterations"].append(iteration_record)
                    history["total_llm_latency_ms"] += metadata.get("api_latency_ms", 0)
                    continue

            # Apply config changes
            config_changes = self.builder.parse_llm_config_changes(parsed)
            if not config_changes:
                config_changes = parsed.get("config_changes", {})

            new_config = self.builder.apply_changes_to_config(current_config, config_changes)

            # Validate and build command
            if benchmark_type == "mdtest":
                valid, sanitized, errs = self.builder.validate_mdtest_params(new_config)
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
            elif benchmark_type == "h5bench":
                valid, sanitized, errs = self.builder.validate_h5bench_params(new_config)
                if errs:
                    logger.warning("  Config validation warnings: %s", errs[:3])
                iteration_record["validated_config"] = sanitized
                iteration_record["validation_errors"] = errs
            elif benchmark_type == "dlio":
                valid, sanitized, errs = self.builder.validate_dlio_params(new_config)
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

            if not valid:
                logger.warning("  REJECTED before execution: invalid proposal %s", errs[:3])
                iteration_record["executed"] = False
                iteration_record["rejected_invalid_config"] = True
                history["iterations"].append(iteration_record)
                history["total_llm_latency_ms"] += metadata.get("api_latency_ms", 0)
                continue

            # Pre-run work guard (A1): a proposal that changes a work-defining
            # parameter is rejected without spending a job on it.
            from .closed_loop_metrics import work_params_changed
            changed_work = work_params_changed(benchmark_type, workload_config.get("bad_config"), sanitized)
            if changed_work:
                logger.warning("  REJECTED before execution: proposal changes work parameters %s", changed_work)
                iteration_record["executed"] = False
                iteration_record["rejected_work_changed"] = True
                iteration_record["changed_work_params"] = changed_work
                iteration_record["rollback"] = True
                last_work_changed = True
                rollback = True
                current_config = dict(best_config)
                current_features = best_features.copy()
                history["iterations"].append(iteration_record)
                history["total_llm_latency_ms"] += metadata.get("api_latency_ms", 0)
                continue

            # Execute
            if not self.dry_run:
                iter_job = f"iter_{workload_name}_r{run_id}_{model_short}_i{iteration}"
                iter_scratch = f"{self.iter_config['slurm']['scratch_dir']}/{iter_job}"
                if benchmark_type == "mdtest":
                    cmd = self.builder.build_mdtest_command(sanitized, output_dir=iter_scratch)
                elif benchmark_type == "hacc_io":
                    cmd = self.builder.build_hacc_command(sanitized, output_dir=iter_scratch)
                elif benchmark_type == "custom":
                    cmd = self.builder.build_custom_command(sanitized, output_dir=iter_scratch)
                elif benchmark_type == "h5bench":
                    config_path = os.path.join(
                        self.executor.results_dir, f"{iter_job}_config.json"
                    )
                    write_cmd, read_cmd, _ = self.builder.build_h5bench_config(
                        sanitized, output_dir=iter_scratch, config_path=config_path
                    )
                    cmd = (write_cmd, read_cmd)
                elif benchmark_type == "dlio":
                    cmd = self.builder.build_dlio_command(
                        sanitized, data_dir=iter_scratch
                    )
                else:
                    cmd = self.builder.build_ior_command(sanitized, output_dir=iter_scratch)

                if isinstance(cmd, tuple):
                    logger.info("  Executing: %s (phase 1)", cmd[0][:100])
                else:
                    logger.info("  Executing: %s", cmd[:120])

                exec_kwargs = {"job_name": iter_job, "benchmark_type": benchmark_type,
                               "slurm_resources": slurm_resources,
                               "workload": workload_name, "work_config": sanitized}
                if benchmark_type == "hacc_io":
                    exec_kwargs["hacc_config"] = sanitized
                elif benchmark_type == "h5bench":
                    exec_kwargs["h5bench_config"] = sanitized
                elif benchmark_type == "dlio":
                    exec_kwargs["dlio_config"] = sanitized
                control_meas = None
                if it_cfg.get("contemporaneous_control", True) and not self.dry_run:
                    exec_first, new_meas, control_meas = self._execute_interleaved(
                        baseline_cmd, baseline_exec_kwargs, cmd, exec_kwargs, repeats)
                else:
                    exec_first, new_meas, _ = self._execute_repeated(cmd, exec_kwargs, repeats)
                exec_result = exec_first or {"success": False, "elapsed_s": 0, "job_id": None}
                exec_result["success"] = bool(exec_first) and new_meas is not None

                iteration_record["executed"] = exec_result["success"]
                if new_meas is None:
                    candidate_time = getattr(self, "_last_execution_allocation_s", 0.0)
                    control_time = 0.0
                else:
                    candidate_time = new_meas.get("allocation_elapsed_s", 0.0)
                    control_time = control_meas.get("allocation_elapsed_s", 0.0) if control_meas else 0.0
                iteration_record["candidate_execution_time_s"] = candidate_time
                iteration_record["control_execution_time_s"] = control_time
                iteration_record["execution_time_s"] = candidate_time + control_time
                history["total_execution_time_s"] += candidate_time + control_time

                if exec_result["success"]:
                    from .closed_loop_metrics import evaluate_candidate
                    new_features = exec_result["features"]
                    new_metrics = exec_result["metrics"]
                    new_bw = new_meas.get("write_bw_mb_s") or new_metrics.get("total_bw_mb_s", 0.001)
                    reference_meas = control_meas or baseline_meas
                    iteration_record["reference"] = "contemporaneous_control" if control_meas else "initial_baseline"
                    if control_meas:
                        iteration_record["control_walltime_s"] = control_meas["walltime_s"]
                        iteration_record["control_walltime_runs_s"] = control_meas["walltime_runs_s"]
                        iteration_record["control_walltime_ci_s"] = [control_meas["ci_lower_s"],
                                                                     control_meas["ci_upper_s"]]
                        logger.info("  Control (baseline re-run alongside): %.2f s (median of %d, CI [%.2f, %.2f])",
                                    control_meas["walltime_s"], control_meas["n_repeats"],
                                    control_meas["ci_lower_s"], control_meas["ci_upper_s"])
                    decision = evaluate_candidate(
                        reference_meas, new_meas, best_speedup,
                        work_tolerance=work_tolerance, regression_factor=regression_factor,
                        min_gain=min_gain)
                    speedup = decision["speedup"]
                    last_work_changed = decision["rejected_work_changed"]

                    logger.info("  Result: wall %.2f s (median of %d, CI [%.2f, %.2f]) -> speedup %.2fx "
                                "(CI %s), verdict %s; BW %.2f MB/s (%.2fx); work ratio %s (%s)%s",
                                new_meas["walltime_s"], new_meas["n_repeats"],
                                new_meas["ci_lower_s"], new_meas["ci_upper_s"],
                                speedup, decision["speedup_ci"], decision["verdict"],
                                new_bw, decision["bw_speedup"] or 0.0,
                                decision["work_ratio"], decision["work_source"],
                                "  REJECTED: workload changed" if last_work_changed else "")

                    # Re-detect
                    new_predictions, new_detected = self.detect_bottlenecks(new_features)
                    iteration_record["new_predictions"] = new_predictions
                    iteration_record["new_detected"] = new_detected
                    iteration_record["speedup"] = speedup
                    iteration_record["bw_speedup"] = decision["bw_speedup"]
                    iteration_record["new_bw"] = new_bw
                    iteration_record["walltime_s"] = new_meas["walltime_s"]
                    iteration_record["walltime_runs_s"] = new_meas["walltime_runs_s"]
                    iteration_record["n_phases"] = new_meas.get("n_phases")
                    iteration_record["work_ratio"] = decision["work_ratio"]
                    iteration_record["work_source"] = decision["work_source"]
                    iteration_record["noise_margin"] = decision["noise_margin"]
                    iteration_record["verdict"] = decision["verdict"]
                    iteration_record["accepted"] = decision["accepted"]
                    iteration_record["regression"] = decision["regression"]
                    iteration_record["decision_basis"] = decision["decision_basis"]
                    iteration_record["speedup_ci"] = decision["speedup_ci"]
                    iteration_record["walltime_ci_s"] = [new_meas["ci_lower_s"], new_meas["ci_upper_s"]]
                    iteration_record["rel_mad"] = new_meas["rel_mad"]
                    iteration_record["rejected_work_changed"] = last_work_changed
                    iteration_record["darshan_paths"] = exec_result.get("darshan_paths", [])
                    iteration_record["darshan_record_cap_hit"] = new_meas.get("darshan_record_cap_hit", False)

                    if decision["accepted"]:
                        best_speedup = speedup
                        best_config = dict(sanitized)
                        best_features = new_features.copy()
                        history["best_iteration"] = iteration
                        rollback = False
                        logger.info("  NEW BEST: %.2fx at iteration %d", speedup, iteration)
                    elif decision["regression"]:
                        logger.warning("  REGRESSION%s: %.2fx -- rolling back",
                                       " (workload changed)" if last_work_changed else "", speedup)
                        iteration_record["rollback"] = True
                        rollback = True
                        current_config = dict(best_config)
                        current_features = best_features.copy()
                    else:
                        iteration_record["rollback"] = True
                        rollback = True
                        current_config = dict(best_config)
                        current_features = best_features.copy()

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
                    iteration_record["execution_error"] = True
            else:
                # Dry run
                iteration_record["executed"] = False
                iteration_record["simulated"] = True
                logger.info("  [DRY RUN] Proposed config: %s", sanitized)

            history["iterations"].append(iteration_record)
            history["total_llm_latency_ms"] += metadata.get("api_latency_ms", 0)

            if iteration_record.get("execution_error"):
                history["final_status"] = "candidate_execution_failed"
                break

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
        from src.models.biquality import load_final_benchmark_test_frames

        _, test_feat, test_labels, _ = load_final_benchmark_test_frames(self.model_path)

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

def publish_results(results, output_path):
    """Validate and publish one immutable iterative result file."""
    from src.llm.iterative_result import validate_iterative_result

    if not results:
        raise ValueError("no iterative results to publish")
    for result in results:
        validate_iterative_result(result)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"iterative output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary iterative output already exists: {temporary}")
    with temporary.open("x") as handle:
        json.dump(results, handle, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="IOSage ML-guided benchmark parameter experiment"
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
    parser.add_argument("--config", default=str(PROJECT_DIR / "configs" / "iterative.yaml"))
    parser.add_argument("--model-bundle", help="Final-evaluation training bundle")
    parser.add_argument("--knowledge-base", help="Schema 2 measured-evidence KB")
    args = parser.parse_args()
    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive")
    if args.n_runs < 1:
        parser.error("--n-runs must be positive")
    switches = sum((args.no_ml, args.no_shap, args.no_kb, args.no_feedback))
    if switches > 1:
        parser.error("select at most one component ablation")
    if switches and args.max_iterations == 1:
        parser.error("single-shot cannot be combined with a component ablation")
    if args.dry_run and args.output:
        parser.error("--output cannot be used with --dry-run")
    if not args.no_ml and not args.model_bundle:
        parser.error("--model-bundle is required unless --no-ml is set")
    if not args.no_kb and not args.knowledge_base:
        parser.error("--knowledge-base is required unless --no-kb is set")

    optimizer = IterativeOptimizer(
        config_path=args.config,
        model_path=args.model_bundle,
        kb_path=args.knowledge_base,
        model=args.model,
        max_iterations=args.max_iterations,
        use_ml=not args.no_ml,
        use_shap=not args.no_shap and not args.no_ml,
        use_kb=not args.no_kb,
        use_feedback=not args.no_feedback,
        dry_run=args.dry_run,
    )

    # Determine workloads
    if args.sweep:
        workloads = supported_workloads(optimizer.iter_config)
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
    configured_root = Path(optimizer.iter_config["slurm"]["results_dir"])
    if not configured_root.is_absolute():
        configured_root = PROJECT_DIR / configured_root
    published_root = (PROJECT_DIR / "results" / "iterative").resolve()
    output_path = Path(args.output) if args.output else (
        configured_root / f"iterative_results_{args.model}_{int(time.time())}.json")
    resolved_output = output_path.resolve()
    if resolved_output == published_root or published_root in resolved_output.parents:
        raise ValueError("new iterative results cannot be written under results/iterative")
    if args.dry_run:
        logger.info("Dry run completed; simulated records were not published")
    else:
        publish_results(all_results, output_path)
        logger.info("Results saved: %s", output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("ITERATIVE PARAMETER EXPERIMENT SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"  {r.get('workload','?'):30s} run={r.get('run_id',0)} "
              f"iters={r.get('total_iterations',0)} speedup={r.get('best_speedup',1.0):.2f}x "
              f"status={r.get('final_status','?')} cost=${r.get('total_cost_usd',0):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
