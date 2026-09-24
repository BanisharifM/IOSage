"""Regression checks for Batch 6 configuration and operations gates."""

import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
TRASH = PROJECT_DIR / ".codex-trash" / "batch6_test_artifacts"


def fresh_dir(name):
    path = TRASH / f"{name}_{time.time_ns()}"
    path.mkdir(parents=True)
    return path


def test_archive_inventory_detects_and_repairs_partial_extraction():
    root = fresh_dir("archive")
    archive = root / "logs.tar.gz"
    payloads = {"one.darshan": b"one", "nested/two.darshan": b"second"}
    with tarfile.open(archive, "w:gz") as handle:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    (root / "one.darshan").write_bytes(payloads["one.darshan"])
    inspect = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "scripts/archive_inventory.py"), str(archive),
         "--output-dir", str(root)], capture_output=True, text=True)
    assert inspect.returncode == 3, inspect.stderr
    repaired = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "scripts/archive_inventory.py"), str(archive),
         "--output-dir", str(root), "--staging-root", str(TRASH / "staging"), "--extract"],
        capture_output=True, text=True)
    assert repaired.returncode == 0, repaired.stderr
    assert (root / "nested/two.darshan").read_bytes() == payloads["nested/two.darshan"]
    inventory = json.loads((root / ".logs.tar.gz.inventory.json").read_text())
    assert inventory["member_count"] == 2


def test_collection_verifier_rejects_an_empty_expected_year():
    root = fresh_dir("empty_collection")
    (root / "2024").mkdir()
    result = subprocess.run(
        ["bash", str(PROJECT_DIR / "scripts/verify_darshan_logs.sh"),
         "--dir", str(root), "--years", "2024", "--python", sys.executable],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "no archives found" in result.stderr


def test_ground_truth_verifier_rejects_an_empty_sample_set():
    root = fresh_dir("empty_ground_truth")
    (root / "ior").mkdir()
    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "scripts/verify_all_ground_truth.py"),
         "--log-dir", str(root), "--bench-type", "ior"],
        cwd=PROJECT_DIR, capture_output=True, text=True)
    assert result.returncode != 0
    assert "verified sample set differs" in result.stderr


def test_download_rejects_month_without_year_before_globus_access():
    result = subprocess.run(
        ["bash", str(PROJECT_DIR / "scripts/download_darshan_logs.sh"), "--month", "6"],
        cwd=PROJECT_DIR, capture_output=True, text=True)
    assert result.returncode == 2
    assert "requires --year" in result.stderr


def test_download_maps_terminal_globus_states_to_distinct_statuses():
    root = fresh_dir("globus")
    binary = root / "globus"
    binary.write_text("""#!/usr/bin/env bash
if [[ "$1" == whoami ]]; then echo tester
elif [[ "$1" == transfer ]]; then echo '{"task_id":"test-task"}'
elif [[ "$1 $2" == 'task wait' ]]; then exit 1
elif [[ "$1 $2" == 'task show' ]]; then printf '{"status":"%s"}\\n' "$FAKE_STATUS"
else exit 9
fi
""")
    binary.chmod(0o755)
    statuses = {"SUCCEEDED": 0, "ACTIVE": 50, "FAILED": 1}
    for status, expected in statuses.items():
        env = os.environ.copy()
        env.update(PATH=f"{root}:{env['PATH']}", FAKE_STATUS=status)
        result = subprocess.run(
            ["bash", str(PROJECT_DIR / "scripts/download_darshan_logs.sh"),
             "--year", "2024", "--dest-base", str(root / "destination")],
            cwd=PROJECT_DIR, env=env, capture_output=True, text=True)
        assert result.returncode == expected, (status, result.stdout, result.stderr)


def test_workload_without_comparison_requires_an_explicit_fix():
    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "scripts/measurement_study/decision_validation.py"),
         "--workload", "ior_healthy_baseline", "--output", str(TRASH / "unused.json")],
        cwd=PROJECT_DIR, capture_output=True, text=True)
    assert result.returncode == 2
    assert "no comparison configuration" in result.stderr


def test_benchmark_config_is_consumed_and_matches_supported_benchmarks():
    config = yaml.safe_load((PROJECT_DIR / "configs/benchmarks.yaml").read_text())
    assert config["schema_version"] == 1
    assert set(config["benchmarks"]) == set(config["expected_counts"])
    source = (PROJECT_DIR / "scripts/build_label_manifest.py").read_text()
    assert "load_benchmark_config(args.config)" in source


def test_dependency_manifests_are_exact_and_extras_are_named():
    for name in ("requirements.txt", "requirements-wisio.txt",
                 "requirements-notebooks.txt", "requirements-test.txt"):
        active = [line.strip() for line in (PROJECT_DIR / name).read_text().splitlines()
                  if line.strip() and not line.startswith(("#", "-r"))]
        assert active
        assert all("==" in line and ">=" not in line for line in active)
    assert "openai==2.29.0" in (PROJECT_DIR / "requirements.txt").read_text()
    assert "wisio[darshan]==0.1.1" in (PROJECT_DIR / "requirements-wisio.txt").read_text()


def test_only_the_active_darshan_configuration_remains():
    assert (PROJECT_DIR / "configs/darshan_runtime.conf").is_file()
    assert not (PROJECT_DIR / "configs/darshan_apps.conf").exists()
    assert not (PROJECT_DIR / "configs/darshan_dlio.conf").exists()
    assert not (PROJECT_DIR / "configs/feature_extraction.yaml").exists()


def test_notebooks_have_no_stored_outputs_or_fixed_paper_assertions():
    for path in sorted((PROJECT_DIR / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs") == []
                compile("".join(cell["source"]), f"{path}:cell", "exec")
        text = path.read_text()
        assert "0.923" not in text
        assert "623 logs" not in text
        assert "../paper/" not in text
