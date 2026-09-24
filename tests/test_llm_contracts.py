"""Contracts that stop unsupported evidence and unsafe LLM output."""

import hashlib
import json
import tempfile
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_production_case_study import agreement_category
from scripts.run_fair_ablation import _healthy_ablation_result, _normalize_result
from scripts.run_ioprescriber_ablation import run_full_system
from scripts.run_llm_evaluation import compute_summary as compute_llm_summary
from src.ioprescriber.contracts import (
    KB_SCHEMA_VERSION,
    PIPELINE_RESULT_VERSION,
    RECOMMENDATION_SCHEMA_VERSION,
    score_grounding,
    validate_qualitative_response,
    validate_knowledge_base,
    validate_pipeline_result,
    validate_recommendation,
)
from src.ioprescriber.recommender import Recommender
from src.ioprescriber.retriever import Retriever
from src.llm.benchmark_command_builder import BenchmarkCommandBuilder
from src.llm.iterative_executor import IterativeExecutor
from src.llm.iterative_optimizer import IterativeOptimizer
from src.llm.evidence_kb import build_knowledge_base


def _entry():
    return {
        "entry_id": "kb2:ior/1/a.darshan",
        "sample_id": "ior/1/a.darshan",
        "job_group": "ior/1",
        "benchmark": "ior",
        "scenario": "small",
        "bottleneck_labels": ["access_granularity"],
        "darshan_signature": {"avg_write_size": 64.0},
        "source_code": {
            "repository": "https://example.invalid/repo",
            "revision": "abc123",
            "path": "src/io.c",
        },
        "fixes": [{
            "fix_id": "fix-buffer",
            "dimension": "access_granularity",
            "description": "buffer adjacent writes",
            "code_before": "write(fd, p, 64);",
            "code_after": "write(fd, p, count);",
            "api_change": "POSIX write size",
            "measurement": {
                "metric": "walltime_speedup",
                "speedup": 1.25,
                "accepted": True,
                "correctness_passed": True,
                "source_revision": "abc123",
                "before_job_ids": ["10"],
                "after_job_ids": ["11"],
                "before_commands": ["srun app --mode before"],
                "after_commands": ["srun app --mode after"],
                "before_walltime_s": [10.0],
                "after_walltime_s": [8.0],
                "result_paths": ["results/run.json"],
                "correctness_result_path": "results/correctness.json",
            },
        }],
        "shap_top_features": {},
    }


def _response():
    return {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "diagnosis": "The trace contains small writes.",
        "recommendations": [{
            "priority": 1,
            "bottleneck_dimension": "access_granularity",
            "explanation": "Buffer adjacent writes.",
            "code_before": "write(fd, p, 64);",
            "code_after": "write(fd, p, count);",
            "expected_speedup": 1.25,
            "kb_citation": "kb2:ior/1/a.darshan",
            "evidence_fix_id": "fix-buffer",
            "confidence": "high",
            "api_change": "POSIX write size",
        }],
    }


def test_kb_and_grounding_require_measured_exact_claims():
    document = {
        "schema_version": KB_SCHEMA_VERSION,
        "allowed_sample_ids": ["ior/1/a.darshan"],
        "allowed_job_groups": ["ior/1"],
        "entries": [_entry()],
    }
    assert validate_knowledge_base(document) == document["entries"]
    matches = [{"entry": _entry()}]
    score = score_grounding(_response(), matches)
    assert score["groundedness_score"] == 1.0
    unsupported = _response()
    unsupported["recommendations"][0]["expected_speedup"] = 999.0
    score = score_grounding(unsupported, matches)
    assert score["groundedness_score"] == 0.0
    assert score["component_scores"]["measurement"] == 0.0


def test_wrong_shaped_recommendations_are_rejected():
    bad = _response()
    bad["recommendations"] = "not-a-list"
    try:
        validate_recommendation(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("a string recommendations field was accepted")

    qualitative = {
        "schema_version": 1,
        "diagnosis": "Small writes are present.",
        "recommendations": [{
            "priority": 1, "bottleneck_dimension": "access_granularity",
            "explanation": "Buffer writes.", "code_before": "write small",
            "code_after": "write buffer", "api_change": "larger POSIX call",
            "confidence": "low", "expected_speedup": 2.0,
        }],
    }
    try:
        validate_qualitative_response(qualitative)
    except ValueError:
        pass
    else:
        raise AssertionError("a numeric claim without evidence was accepted")


def test_retriever_refuses_query_overlap():
    document = {
        "schema_version": KB_SCHEMA_VERSION,
        "allowed_sample_ids": ["ior/1/a.darshan"],
        "allowed_job_groups": ["ior/1"],
        "entries": [_entry()],
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "kb.json"
        path.write_text(json.dumps(document))
        retriever = Retriever(path)
        cases = [
            ({}, {"query_sample_id": "ior/1/a.darshan"}),
            ({}, {"query_job_group": "ior/1"}),
            ({"_benchmark": "ior", "_ground_truth_job_id": 1,
              "_source_path": "/logs/a.darshan"}, {}),
        ]
        for features, kwargs in cases:
            try:
                retriever.retrieve(["access_granularity"], features, **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError("evaluation overlap was accepted")


def test_cache_hit_reports_zero_api_work_and_validates_contract():
    with tempfile.TemporaryDirectory() as directory:
        recommender = Recommender(cache_dir=directory)
        system, user = "system", "user"
        request = recommender._cache_request(system, user, "trial-1")
        key = hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path = Path(directory) / f"{key}.json"
        path.write_text(json.dumps({
            "cache_schema_version": 2,
            "request": request,
            "response": json.dumps(_response()),
            "metadata": {"model": recommender.model_id, "request_id": "req-old",
                         "tokens_input": 50, "tokens_output": 20},
        }))
        _, metadata = recommender.call_llm(system, user, cache_namespace="trial-1")
        assert metadata["cache_hit"] is True
        assert metadata["tokens_input"] == metadata["tokens_output"] == 0
        assert metadata["api_latency_ms"] == 0.0
        assert metadata["cache_source_request_id"] == "req-old"


def test_prompt_includes_exact_source_revision_and_path():
    recommender = Recommender()
    _, prompt = recommender.build_prompt(
        {"access_granularity": 0.9}, ["access_granularity"], {},
        [{"entry": _entry()}], {"avg_write_size": 64.0})
    assert "https://example.invalid/repo at abc123" in prompt
    assert "Path: src/io.c" in prompt


def test_pipeline_result_contract_has_one_layout():
    result = {
        "schema_version": PIPELINE_RESULT_VERSION,
        "workload": "w",
        "pipeline_latency_ms": 1.0,
        "detection": {"predictions": {}, "detected": ["healthy"]},
        "attribution": {},
        "retrieval": {"n_entries": 0, "entries": []},
        "recommendation": {"parsed": None, "groundedness": None, "metadata": None},
    }
    validate_pipeline_result(result)
    result["step2_shap"] = {}
    try:
        validate_pipeline_result(result)
    except ValueError:
        pass
    else:
        raise AssertionError("retired pipeline fields were accepted")


def test_ablation_summarizers_consume_the_versioned_pipeline_layout():
    result = {
        "schema_version": PIPELINE_RESULT_VERSION,
        "workload": "w", "pipeline_latency_ms": 1.0,
        "detection": {"predictions": {}, "detected": ["healthy"]},
        "attribution": {}, "retrieval": {"n_entries": 0, "entries": []},
        "recommendation": {"parsed": None, "groundedness": None, "metadata": None},
    }
    fair = _normalize_result(result, "C0", "w", True, True, True)
    assert fair["detected"] == ["healthy"] and fair["recommendation"] is None

    class _Pipeline:
        @staticmethod
        def analyze(_features, workload_name):
            assert workload_name == "FULL_w"
            return result

    original = run_full_system(_Pipeline(), {}, "w")
    assert original["detected"] == ["healthy"] and original["recommendation"] is None


def test_llm_summary_accepts_a_healthy_no_call_and_counts_real_calls():
    healthy = {
        "detection": {"detected": ["healthy"]},
        "recommendation": {"parsed": None, "groundedness": None, "metadata": None},
    }
    bottleneck = {
        "detection": {"detected": ["access_granularity"]},
        "recommendation": {
            "parsed": {"recommendations": [{"priority": 1}]},
            "groundedness": {"groundedness_score": 1.0},
            "metadata": {"api_latency_ms": 10.0, "tokens_input": 4,
                         "tokens_output": 3},
        },
    }
    summary = compute_llm_summary({"model": [healthy, bottleneck]})["model"]
    assert summary["n_healthy_detections"] == 1
    assert summary["n_recommendation_calls"] == 1
    assert summary["tokens_mean"] == 7


def test_fair_ablation_healthy_result_has_no_api_artifacts():
    result = _healthy_ablation_result("C2_no_kb", "w", True, False)
    assert result["detected"] == ["healthy"]
    assert result["recommendation"] is None
    assert result["metadata"] == {}
    assert result["groundedness"] == {}


def test_mdtest_model_fields_cannot_inject_shell_commands():
    builder = BenchmarkCommandBuilder()
    valid, _, errors = builder.validate_mdtest_params({
        "items_per_rank": "1; touch /tmp/a", "write_bytes": 0,
        "read_bytes": 0, "files_only": True, "unique_dir": False})
    assert not valid and errors
    valid, values, errors = builder.validate_mdtest_params({
        "items_per_rank": 10, "write_bytes": 0, "read_bytes": 0,
        "files_only": True, "unique_dir": False})
    assert valid and not errors
    assert ";" not in builder.build_mdtest_command(values, "/tmp/s")


def test_slurm_override_is_rendered_and_h5bench_requires_both_phases():
    with tempfile.TemporaryDirectory() as directory:
        config = {
            "slurm": {
                "account": "a", "partition": "p", "nodes": 1, "ntasks": 2,
                "cpus_per_task": 1, "walltime": "00:10:00",
                "scratch_dir": f"{directory}/scratch",
                "darshan_log_dir": f"{directory}/logs",
                "results_dir": f"{directory}/results",
                "darshan_lib": "/lib/darshan.so",
                "darshan_config": "/tmp/darshan.conf",
            },
            "paths": {"h5bench_modules": "hdf5-module"},
        }
        executor = IterativeExecutor(config)
        script = executor.generate_slurm_script(
            "job", "ior -o /tmp/x", slurm_resources={
                "nodes": 4, "ntasks": 64, "cpus_per_task": 2,
                "walltime": "01:02:03"})
        text = Path(script).read_text()
        assert "#SBATCH --nodes=4" in text and "#SBATCH --ntasks=64" in text
        assert "#SBATCH --cpus-per-task=2" in text and "#SBATCH --time=01:02:03" in text
        assert "#SBATCH --export=NONE" in text
        assert "source /etc/profile ||" in text
        assert "profile.d/modules.sh" not in text
        h5 = executor.generate_slurm_script(
            "h5", ("/bin/true cfg", "/bin/true /tmp/output.h5"), "h5bench",
            h5bench_config={"COLLECTIVE_DATA": "NO"})
        h5_text = Path(h5).read_text()
        assert "read phase cannot run" in h5_text
        assert 'exit "$READ_RC"' in h5_text


def test_iterative_schema_and_disagreement_directions():
    parsed = {"strategy": "change API", "config_changes": {},
              "changes_made": [], "kb_citations": [],
              "evidence_scope": "KB supports diagnosis only"}
    assert IterativeOptimizer._validate_iterative_response(parsed) == parsed
    states = {
        (False, False): "agree_healthy",
        (True, True): "agree_bottleneck",
        (False, True): "heuristic_healthy_ml_bottleneck",
        (True, False): "heuristic_bottleneck_ml_healthy",
    }
    assert {state: agreement_category(*state) for state in states} == states


def test_evidence_builder_uses_development_ids_and_joins_shap_by_id():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        features = pd.DataFrame({
            "_benchmark": ["ior", "ior", "ior"],
            "_ground_truth_job_id": [1, 2, 3],
            "_source_path": ["/logs/a.darshan", "/logs/b.darshan", "/logs/c.darshan"],
            "_scenario": ["a", "b", "c"],
            "avg_write_size": [64.0, 128.0, 256.0],
        })
        labels = pd.DataFrame({
            "benchmark": ["ior", "ior", "ior"], "job_id": [1, 2, 3],
            "scenario": ["a", "b", "c"],
            "access_granularity": [1, 1, 1],
        })
        for dimension in (
                "metadata_intensity", "parallelism_efficiency", "access_pattern",
                "interface_choice", "file_strategy", "throughput_utilization"):
            labels[dimension] = 0
        features_path, labels_path = root / "features.parquet", root / "labels.parquet"
        features.to_parquet(features_path, index=False)
        labels.to_parquet(labels_path, index=False)
        ids = np.array(["ior/1/a.darshan", "ior/2/b.darshan", "ior/3/c.darshan"])
        groups = np.array(["ior/1", "ior/2", "ior/3"])
        splits_path = root / "splits.npz"
        np.savez(splits_path, bench_train=np.array([0]), bench_val=np.array([1]),
                 bench_test=np.array([2]), bench_ids=ids, bench_groups=groups)
        record = _entry()
        record.pop("entry_id")
        measurements_path = root / "measurements.json"
        (root / "results").mkdir()
        (root / "results" / "run.json").write_text("{}")
        (root / "results" / "correctness.json").write_text("{}")
        measurements_path.write_text(json.dumps({"schema_version": 1, "records": [record]}))
        shap_path = root / "shap.pkl"
        with open(shap_path, "wb") as handle:
            pickle.dump({
                "sample_ids": ids[::-1],
                "feature_names": ["avg_write_size"],
                "shap_dict": {"access_granularity": np.array([[3.0], [2.0], [1.0]])},
            }, handle)
        document = build_knowledge_base(
            features_path, labels_path, splits_path, measurements_path,
            ["avg_write_size"], shap_path)
        entry = document["entries"][0]
        assert entry["sample_id"] == ids[0]
        assert entry["shap_top_features"]["access_granularity"][0]["shap_value"] == 1.0

        record["sample_id"] = ids[2]
        measurements_path.write_text(json.dumps({"schema_version": 1, "records": [record]}))
        try:
            build_knowledge_base(
                features_path, labels_path, splits_path, measurements_path,
                ["avg_write_size"], shap_path)
        except ValueError as error:
            assert "final-evaluation" in str(error)
        else:
            raise AssertionError("final-evaluation evidence entered the KB")

        record["sample_id"] = ids[0]
        measurements_path.write_text(json.dumps({"schema_version": 1, "records": [record]}))
        np.savez(splits_path, bench_train=np.array([0, 0]), bench_val=np.array([1]),
                 bench_test=np.array([2]), bench_ids=ids, bench_groups=groups)
        try:
            build_knowledge_base(
                features_path, labels_path, splits_path, measurements_path,
                ["avg_write_size"], shap_path)
        except ValueError as error:
            assert "duplicate" in str(error)
        else:
            raise AssertionError("a duplicate training index was accepted")
