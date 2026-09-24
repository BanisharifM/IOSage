"""Verify benchmark labels against the shared observable rule set."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from src.data.label_rules import (
    BOTTLENECK_DIMENSIONS,
    DIMENSION_NAMES,
    parallelism_present,
    rule_details,
    rule_frame,
)

logger = logging.getLogger(__name__)

def rank_imbalance_present(features: Mapping[str, object]) -> bool:
    """Return the shared parallelism rule for one feature row."""
    return bool(parallelism_present(features))


def bottleneck_rules(
    features: Mapping[str, object],
) -> dict[str, tuple[bool, str]]:
    """Return the shared rule decisions and their observed values."""
    decisions = rule_frame(features).iloc[0]
    details = rule_details(features)
    return {
        dimension: (bool(decisions[dimension]), details[dimension])
        for dimension in BOTTLENECK_DIMENSIONS
    }


def cleaning_rule(
    features: Mapping[str, object],
    cleaning: Mapping[str, object],
) -> tuple[bool, str]:
    """Return whether a feature row meets an explicit cleaning configuration."""
    runtime = float(features['runtime_seconds'])
    total_bytes = float(features['io_bytes_all'])
    total_ops = float(features['io_ops_all'])
    checks = (
        (runtime >= float(cleaning['min_duration_seconds']),
         f"runtime={runtime:.1f}s < {float(cleaning['min_duration_seconds'])}s"),
        (total_bytes >= float(cleaning['min_total_bytes']),
         f"total_bytes={total_bytes:.0f} < {float(cleaning['min_total_bytes']):.0f}"),
        (total_ops >= float(cleaning['min_io_ops']),
         f"total_ops={total_ops:.0f} < {float(cleaning['min_io_ops']):.0f}"),
    )
    for passes, reason in checks:
        if not passes:
            return False, reason
    return True, ''


def verify_benchmark_log(
    features: Mapping[str, object],
    intended_labels: Mapping[str, int],
    cleaning: Mapping[str, object],
    intended_validity: Mapping[str, int] | None = None,
) -> tuple[bool, dict[str, object]]:
    """Check each controlled target against its intended binary label."""
    report: dict[str, object] = {'checks': {}, 'passed_checks': 0, 'total_checks': 0}
    unknown = set(intended_labels) - set(DIMENSION_NAMES)
    if unknown:
        raise ValueError(f"intended labels contain unknown dimensions {sorted(unknown)}")
    if intended_validity is None:
        if intended_labels.get('healthy', 0) == 1:
            valid_dimensions = list(BOTTLENECK_DIMENSIONS)
        else:
            valid_dimensions = [
                dimension for dimension in BOTTLENECK_DIMENSIONS
                if intended_labels.get(dimension, 0) == 1
            ]
    else:
        unknown_validity = set(intended_validity) - set(DIMENSION_NAMES)
        if unknown_validity:
            raise ValueError(
                f"intended validity contains unknown dimensions {sorted(unknown_validity)}")
        valid_dimensions = [
            dimension for dimension in BOTTLENECK_DIMENSIONS
            if intended_validity.get(dimension, 0) == 1
        ]
    if not valid_dimensions:
        raise ValueError("intended labels have no valid problem target")

    clean_passed, clean_reason = cleaning_rule(features, cleaning)
    report['cleaning_rule'] = clean_passed
    report['cleaning_reason'] = clean_reason
    rules = bottleneck_rules(features)

    def record(name: str, passed: bool, detail: str) -> None:
        report['total_checks'] += 1
        report['checks'][name] = {
            'status': 'pass' if passed else 'fail',
            'value': detail,
        }
        if passed:
            report['passed_checks'] += 1

    incomplete_modules = [
        module.upper() for module in ('posix', 'mpiio', 'stdio')
        if bool(features[f'partial_{module}'])
    ]
    record(
        'module_records_complete',
        not incomplete_modules,
        ('incomplete=' + ','.join(incomplete_modules)) if incomplete_modules else 'complete',
    )

    for dimension in valid_dimensions:
        present, detail = rules[dimension]
        expected = bool(intended_labels.get(dimension, 0))
        record(f'{dimension}/rule', present == expected,
               f"expected={int(expected)} observed={int(present)} {detail}")

    if report['total_checks'] == 0:
        raise AssertionError("no rule evaluated")
    passed = report['passed_checks'] == report['total_checks']
    if not passed:
        failed = {
            name: result['value'] for name, result in report['checks'].items()
            if result['status'] != 'pass'
        }
        logger.warning("Benchmark failed verification: %s", failed)
    return passed, report
