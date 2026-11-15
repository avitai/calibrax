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
    expected_calibration_error,
    maximum_calibration_error,
)


def register() -> None:
    """Register 5 calibration metrics at import time.

    Note: reliability_diagram_bins and brier_decomposition are NOT
    registered (they return dicts, not floats).
    """
    registry = MetricRegistry()
    # (name, fn, description, direction, is_proper)
    builtins: list[tuple[str, MetricFn, str, MetricDirection, bool]] = [
        ("brier_score", brier_score, "Brier score (probability MSE)", MetricDirection.LOWER, True),
        (
            "expected_calibration_error",
            expected_calibration_error,
            "Expected calibration error",
            MetricDirection.LOWER,
            False,
        ),
        (
            "maximum_calibration_error",
            maximum_calibration_error,
            "Maximum calibration error",
            MetricDirection.LOWER,
            False,
        ),
        (
            "adaptive_calibration_error",
            adaptive_calibration_error,
            "Adaptive calibration error",
            MetricDirection.LOWER,
            False,
        ),
        (
            "classwise_ece",
            classwise_ece,
            "Classwise expected calibration error",
            MetricDirection.LOWER,
            False,
        ),
    ]
    for name, fn, desc, direction, is_proper_flag in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="calibration",
                direction=direction,
                description=desc,
                signature=MetricSignature.PREDICTIONS_TARGETS,
                properties=MetricProperties(is_proper=is_proper_flag),
            )
            registry.register(name, entry)
