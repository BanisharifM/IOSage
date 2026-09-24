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

HEALTHY_MIN_BYTES_PER_S = 1024.0


def rank_imbalance_present(features: Mapping[str, object]) -> bool:
    """Return the shared parallelism rule for one feature row."""
    return bool(parallelism_present(features))


def bottleneck_rules(
    features: Mapping[str, object],
) -> dict[str, tuple[bool, str]]:
    """Return the seven shared rule decisions and their observed values."""
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
) -> tuple[bool, dict[str, object]]:
    """Check that one sample's features match its intended labels."""
    report: dict[str, object] = {'checks': {}, 'passed_checks': 0, 'total_checks': 0}
    unknown = set(intended_labels) - set(DIMENSION_NAMES)
    if unknown:
        raise ValueError(f"intended labels contain unknown dimensions {sorted(unknown)}")
    positives = [dimension for dimension in DIMENSION_NAMES
                 if intended_labels.get(dimension, 0) == 1]
    if not positives:
        raise ValueError("intended labels name no dimension")
    if 'healthy' in positives and len(positives) > 1:
        raise ValueError(f"healthy cannot be combined with {positives}")

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

    if 'healthy' in positives:
        rate = float(features['io_bytes_all']) / max(float(features['runtime_seconds']), 1e-9)
        record('healthy/data_rate', rate >= HEALTHY_MIN_BYTES_PER_S,
               f"bytes_per_s={rate:.0f}")
        for dimension, (present, detail) in rules.items():
            record(f'healthy/no_{dimension}', not present, detail)
    else:
        for dimension in positives:
            present, detail = rules[dimension]
            record(f'{dimension}/rule', present, detail)

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
