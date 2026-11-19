"""Registrations of the built-in ``calibration`` metrics."""

from __future__ import annotations

from calibrax.core.models import MetricDirection
from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import (
    MetricEntry,
    MetricFn,
    MetricProperties,
    MetricSignature,
    MetricTier,
)
from calibrax.metrics.functional.calibration import (
    adaptive_calibration_error,
    brier_score,
    classwise_ece,
    empirical_power,
    expected_calibration_error,
    false_discovery_proportion,
    maximum_calibration_error,
)


def register() -> None:
    """Register 7 calibration and selective-inference metrics at import time.

    Note: reliability_diagram_bins, brier_decomposition, and
    conformal_fdr_selection are NOT registered (they return dicts or masks,
    not scalar floats).
    """
    registry = MetricRegistry()
    # (name, fn, description, direction, is_proper, is_differentiable)
    builtins: list[tuple[str, MetricFn, str, MetricDirection, bool, bool]] = [
        (
            "brier_score",
            brier_score,
            "Brier score (probability MSE)",
            MetricDirection.LOWER,
            True,
            True,
        ),
        (
            "expected_calibration_error",
            expected_calibration_error,
            "Expected calibration error",
            MetricDirection.LOWER,
            False,
            True,
        ),
        (
            "maximum_calibration_error",
            maximum_calibration_error,
            "Maximum calibration error",
            MetricDirection.LOWER,
            False,
            True,
        ),
        (
            "adaptive_calibration_error",
            adaptive_calibration_error,
            "Adaptive calibration error",
            MetricDirection.LOWER,
            False,
            True,
        ),
        (
            "classwise_ece",
            classwise_ece,
            "Classwise expected calibration error",
            MetricDirection.LOWER,
            False,
            True,
        ),
        (
            "false_discovery_proportion",
            false_discovery_proportion,
            "Realized false discovery proportion V / max(R, 1)",
            MetricDirection.LOWER,
            False,
            False,
        ),
        (
            "empirical_power",
            empirical_power,
            "Realized discovery power among true alternatives",
            MetricDirection.HIGHER,
            False,
            False,
        ),
    ]
    for name, fn, desc, direction, is_proper_flag, is_diff_flag in builtins:
        entry = MetricEntry(
            name=name,
            fn=fn,
            tier=MetricTier.PURE_FUNCTION,
            domain="calibration",
            direction=direction,
            description=desc,
            signature=MetricSignature.PREDICTIONS_TARGETS,
            properties=MetricProperties(
                is_proper=is_proper_flag,
                is_differentiable=is_diff_flag,
            ),
        )
        registry.register(name, entry)
