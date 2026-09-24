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
    n1 = doc["applications"]["nek5000"]["cases"]["N1"]
    n2 = doc["applications"]["nek5000"]["cases"]["N2"]
    assert n1["classifier_supported"] is False and n1["expected_labels"] == []
    assert n2["classifier_supported"] is False and n2["expected_labels"] == []


def test_registered_wrf_cases_expose_their_inputs():
    doc = app_cases.load(CASES)
    assert app_cases.case_args(doc, "wrf", "W1") == ["2"] and app_cases.case_args(doc, "wrf", "W2") == ["11"]
    rule = app_cases.pairs(doc, "wrf")["W1:W2"]
    assert "history.wrfout_d01_2019-11-27_00_00_00.variables" in rule["exact_subtrees"]
    assert "final_time" in rule["exact_fields"] and not rule["fields"]
    assert doc["applications"]["wrf"]["secondary_metric"] == "app_metric"
    assert doc["applications"]["wrf"]["case_order"] == ["W1", "W2"]
    for case in ("W1", "W2"):
        spec = doc["applications"]["wrf"]["cases"][case]
        assert spec["classifier_supported"] is False and spec["expected_labels"] == []
    assert app_cases.expected_io(doc, "wrf", "W2")["collective"] is True
    assert app_cases.work_invariant(doc, "wrf")["expected_time_steps"] == 50
    assert (PROJECT_DIR / doc["applications"]["wrf"]["script"]).is_file()


def test_loader_rejects_incomplete_registrations():
    base = yaml.safe_load(CASES.read_text())
    doc = copy.deepcopy(base); doc["applications"]["nek5000"]["cases"]["N2"].pop("fix_case")
    _rejects(doc, "fix_case")
    doc = copy.deepcopy(base); doc["applications"]["nek5000"]["cases"]["N1"]["args"] = ["200", "50", 1, True]
    _rejects(doc, "args")
    doc = copy.deepcopy(base); doc["applications"]["nek5000"]["cases"]["N2"].pop("io_structure")
    _rejects(doc, "io_structure")
    doc = copy.deepcopy(base); doc["applications"]["nek5000"]["cases"]["N2"].pop("classifier_supported")
    _rejects(doc, "classifier_supported")
    doc = copy.deepcopy(base); doc["applications"]["nek5000"]["cases"]["N2"]["classifier_supported"] = "no"
    _rejects(doc, "classifier_supported must be Boolean")
    doc = copy.deepcopy(base); doc["applications"]["nek5000"]["cases"]["N2"]["expected_labels"] = ["file_strategy"]
    _rejects(doc, "cannot register classifier labels")
    doc = copy.deepcopy(base); doc["applications"]["nek5000"]["cases"]["N2"]["classifier_supported"] = True
    _rejects(doc, "needs an expected label")
    doc = copy.deepcopy(base); doc["applications"]["nek5000"]["equivalence"] = {"source": "x"}
    _rejects(doc, "equivalence needs")
    doc = copy.deepcopy(base); doc["protocol"]["repeats"] = 3
    _rejects(doc, "repeats")
    doc = copy.deepcopy(base); doc["applications"]["nek5000"]["case_order"] = ["N2"]
    _rejects(doc, "case_order")
    doc = copy.deepcopy(base); doc["applications"]["wrf"]["cases"]["W1"]["expected_labels"] = ["interface_choice"]
    _rejects(doc, "classifier_supported")
    doc = copy.deepcopy(base); doc["applications"]["wrf"]["secondary_metric"] = "wall_s"
    _rejects(doc, "secondary_metric")


if __name__ == "__main__":
    test_registered_nek5000_cases_load_and_expose_their_inputs()
    test_loader_rejects_incomplete_registrations()
    print("app_cases tests pass")
