"""Preregistered application cases (``configs/app_cases.yaml``).

The file fixes, before any scored run, what each case is, what the detector is expected to
say, which case is the documented fix, what work must stay identical, how the scientific
result of a problem run and its fix run are compared, and what file organization Darshan
must show. ``load`` validates the structure and returns the parsed document; the other
functions give the series driver and the audit their inputs.
"""
from pathlib import Path

import yaml

ROLES = {"problem", "fix", "abstain"}
KINDS = {"natural", "constructed"}
REQUIRED_CASE = ("role", "kind", "args", "knobs", "classifier_supported",
                 "expected_labels", "allowed_extra_labels", "source", "io_structure")
REQUIRED_APP = ("script", "resources", "work_invariant", "equivalence", "cases", "case_order")
REQUIRED_PROTOCOL = ("storage_tier", "repeats", "controls_per_case", "ordering", "confidence",
                     "decision_rule", "decision_source")
REQUIRED_IO = ("write_module", "shared_file")   # applications add their own structure keys


def load(path):
    """Parsed and validated document; raises ValueError listing every problem."""
    doc = yaml.safe_load(Path(path).read_text())
    problems = []
    if not isinstance(doc, dict):
        raise ValueError("app_cases document is not a mapping")
    protocol = doc.get("protocol")
    if not isinstance(protocol, dict):
        problems.append("protocol missing")
    else:
        for key in REQUIRED_PROTOCOL:
            if key not in protocol:
                problems.append(f"protocol.{key} missing")
        if isinstance(protocol.get("repeats"), bool) or not isinstance(protocol.get("repeats"), int) \
                or protocol.get("repeats", 0) < 5:
            problems.append("protocol.repeats must be an integer of at least 5")
        if protocol.get("controls_per_case") != 1:
            problems.append("protocol.controls_per_case must be 1")
    apps = doc.get("applications")
    if not isinstance(apps, dict) or not apps:
        problems.append("applications missing or empty")
        apps = {}
    for app, spec in apps.items():
        prefix = f"applications.{app}"
        if not isinstance(spec, dict):
            problems.append(f"{prefix} is not a mapping")
            continue
        for key in REQUIRED_APP:
            if key not in spec:
                problems.append(f"{prefix}.{key} missing")
        cases = spec.get("cases") or {}
        order = spec.get("case_order") or []
        if sorted(order) != sorted(cases):
            problems.append(f"{prefix}.case_order {order} does not list exactly the cases {sorted(cases)}")
        equivalence = spec.get("equivalence") or {}
        if not isinstance(equivalence.get("source"), str) or not equivalence.get("source"):
            problems.append(f"{prefix}.equivalence.source missing")
        if not any(equivalence.get(k) for k in ("exact_fields", "fields", "exact_subtrees", "checkpoints")):
            problems.append(f"{prefix}.equivalence needs exact_fields, exact_subtrees, fields or checkpoints")
        if not isinstance(equivalence.get("exact_subtrees", []), list):
            problems.append(f"{prefix}.equivalence.exact_subtrees must be a list of dotted paths")
        if spec.get("secondary_metric") not in (None, "app_metric"):
            problems.append(f"{prefix}.secondary_metric must be app_metric when given")
        for name, tolerance in (equivalence.get("fields") or {}).items():
            if not isinstance(tolerance, dict) or not {"absolute", "relative"} <= set(tolerance):
                problems.append(f"{prefix}.equivalence.fields.{name} needs absolute and relative")
        checkpoints = equivalence.get("checkpoints")
        if checkpoints is not None and not isinstance(checkpoints, list):
            problems.append(f"{prefix}.equivalence.checkpoints must be a list of blocks")
        for i, block in enumerate(checkpoints if isinstance(checkpoints, list) else []):
            if not isinstance(block, dict) or not {"steps", "components", "stats"} <= set(block):
                problems.append(f"{prefix}.equivalence.checkpoints[{i}] needs steps, components and stats")
                continue
            for stat, tolerance in block["stats"].items():
                if not isinstance(tolerance, dict) or not {"absolute", "relative"} <= set(tolerance):
                    problems.append(f"{prefix}.equivalence.checkpoints[{i}].stats.{stat} needs absolute and relative")
        for case, cspec in cases.items():
            cprefix = f"{prefix}.cases.{case}"
            if not isinstance(cspec, dict):
                problems.append(f"{cprefix} is not a mapping")
                continue
            for key in REQUIRED_CASE:
                if key not in cspec:
                    problems.append(f"{cprefix}.{key} missing")
            if cspec.get("role") not in ROLES:
                problems.append(f"{cprefix}.role must be one of {sorted(ROLES)}")
            if cspec.get("kind") not in KINDS:
                problems.append(f"{cprefix}.kind must be one of {sorted(KINDS)}")
            args = cspec.get("args")
            if not isinstance(args, list) or not args or not all(isinstance(a, str) and a for a in args):
                problems.append(f"{cprefix}.args must be a non-empty list of strings (quote yes/no in YAML)")
            for key in ("expected_labels", "allowed_extra_labels"):
                if not isinstance(cspec.get(key), list):
                    problems.append(f"{cprefix}.{key} must be a list")
            classifier_supported = cspec.get("classifier_supported")
            if not isinstance(classifier_supported, bool):
                problems.append(f"{cprefix}.classifier_supported must be Boolean")
            elif not classifier_supported and (
                    cspec.get("expected_labels") or cspec.get("allowed_extra_labels")):
                problems.append(
                    f"{cprefix} cannot register classifier labels when classifier_supported is false"
                )
            elif (classifier_supported and cspec.get("role") == "problem"
                  and not cspec.get("expected_labels")):
                problems.append(
                    f"{cprefix} needs an expected label when classifier_supported is true"
                )
            io = cspec.get("io_structure")
            if not isinstance(io, dict) or any(k not in io for k in REQUIRED_IO):
                problems.append(f"{cprefix}.io_structure needs {REQUIRED_IO}")
            if cspec.get("role") == "problem":
                fix = cspec.get("fix_case")
                if fix not in cases or cases.get(fix, {}).get("role") != "fix":
                    problems.append(f"{cprefix}.fix_case must name a case with role fix")
            elif cspec.get("role") == "fix" and not any(c.get("fix_case") == case for c in cases.values()
                                                       if isinstance(c, dict)):
                problems.append(f"{cprefix} is a fix that no problem case names")
    if problems:
        raise ValueError("app_cases.yaml: " + "; ".join(problems))
    return doc


def application(doc, app):
    if app not in doc["applications"]:
        raise KeyError(f"application {app!r} is not registered")
    return doc["applications"][app]


def case_args(doc, app, case):
    return list(application(doc, app)["cases"][case]["args"])


def expand_rule(equivalence):
    """The explicit audit rule (``exact_fields``, ``fields``) of one registered equivalence.

    Every ``checkpoints`` block expands per registered step and variable component into
    ``checkpoints.step_<n>.<field>`` exact entries and ``checkpoints.step_<n>.variables.<comp>.<stat>``
    tolerance entries, so the registration stays readable and the audit sees every name."""
    exact = list(equivalence.get("exact_fields") or [])
    fields = dict(equivalence.get("fields") or {})
    subtrees = list(equivalence.get("exact_subtrees") or [])
    for block in equivalence.get("checkpoints") or []:
        for step in block["steps"]:
            for name in block.get("exact", []):
                exact.append(f"checkpoints.step_{step}.{name}")
            for component in block["components"]:
                for stat, tolerance in block["stats"].items():
                    fields[f"checkpoints.step_{step}.variables.{component}.{stat}"] = dict(tolerance)
    rule = {"exact_fields": exact, "fields": fields}
    if subtrees:
        rule["exact_subtrees"] = subtrees
    return rule


def pairs(doc, app):
    """{"PROBLEM:FIX": explicit equivalence rule} for every problem case of the application."""
    spec = application(doc, app)
    rule = expand_rule(spec["equivalence"])
    return {f"{case}:{cspec['fix_case']}": rule for case, cspec in spec["cases"].items() if cspec["role"] == "problem"}


def expected_io(doc, app, case):
    return application(doc, app)["cases"][case]["io_structure"]


def work_invariant(doc, app):
    return application(doc, app)["work_invariant"]
