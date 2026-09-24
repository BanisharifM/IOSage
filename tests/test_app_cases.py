"""Preregistration registry (configs/app_cases.yaml) and its loader."""
import copy
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.apps import app_cases  # noqa: E402

CASES = PROJECT_DIR / "configs" / "app_cases.yaml"


def _rejects(doc, fragment):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, "cases.yaml")
        path.write_text(yaml.safe_dump(doc))
        try:
            app_cases.load(path)
        except ValueError as exc:
            assert fragment in str(exc), str(exc)
        else:
            raise AssertionError(f"accepted a registry missing {fragment}")


def test_registered_nek5000_cases_load_and_expose_their_inputs():
    doc = app_cases.load(CASES)
    assert doc["protocol"]["repeats"] == 11 and doc["protocol"]["storage_tier"] == "/work/hdd"
    assert app_cases.case_args(doc, "nek5000", "N1") == ["200", "50", "1", "yes"]
    assert app_cases.case_args(doc, "nek5000", "N2") == ["200", "50", "32", "yes"]
    assert list(app_cases.pairs(doc, "nek5000")) == ["N2:N1"]
    rule = app_cases.pairs(doc, "nek5000")["N2:N1"]
    assert "final_step" in rule["exact_fields"] and "final_time" in rule["fields"]
    assert app_cases.expected_io(doc, "nek5000", "N2") == {"files_per_checkpoint": 32, "write_module": "STDIO",
                                                            "shared_file": False}
    assert app_cases.work_invariant(doc, "nek5000")["steps"] == 200
    assert (PROJECT_DIR / doc["applications"]["nek5000"]["script"]).is_file()
    assert doc["applications"]["nek5000"]["cases"]["N2"]["expected_labels"] == ["file_strategy"]
    assert doc["applications"]["nek5000"]["cases"]["N1"]["expected_labels"] == []


def test_loader_rejects_incomplete_registrations():
    base = yaml.safe_load(CASES.read_text())
    doc = copy.deepcopy(base); doc["applications"]["nek5000"]["cases"]["N2"].pop("fix_case")
    _rejects(doc, "fix_case")
    doc = copy.deepcopy(base); doc["applications"]["nek5000"]["cases"]["N1"]["args"] = ["200", "50", 1, True]
    _rejects(doc, "args")
    doc = copy.deepcopy(base); doc["applications"]["nek5000"]["cases"]["N2"].pop("io_structure")
    _rejects(doc, "io_structure")
    doc = copy.deepcopy(base); doc["applications"]["nek5000"]["equivalence"] = {"source": "x"}
    _rejects(doc, "exact_fields or fields")
    doc = copy.deepcopy(base); doc["protocol"]["repeats"] = 3
    _rejects(doc, "repeats")
    doc = copy.deepcopy(base); doc["applications"]["nek5000"]["case_order"] = ["N2"]
    _rejects(doc, "case_order")


if __name__ == "__main__":
    test_registered_nek5000_cases_load_and_expose_their_inputs()
    test_loader_rejects_incomplete_registrations()
    print("app_cases tests pass")
