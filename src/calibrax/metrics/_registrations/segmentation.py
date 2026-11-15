"""Registrations of the built-in ``segmentation`` metrics."""

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
from calibrax.metrics.functional.segmentation import dice_coefficient, iou, pixel_accuracy


def register() -> None:
    """Register 3 segmentation metrics at import time."""
    registry = MetricRegistry()
    # (name, fn, description)
    builtins: list[tuple[str, MetricFn, str]] = [
        ("iou", iou, "Intersection over Union (Jaccard index)"),
        ("dice_coefficient", dice_coefficient, "Dice coefficient (F1 for segmentation)"),
        ("pixel_accuracy", pixel_accuracy, "Fraction of correctly classified pixels"),
    ]
    for name, fn, desc in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="segmentation",
                direction=MetricDirection.HIGHER,
                description=desc,
                signature=MetricSignature.PREDICTIONS_TARGETS,
                # Integer masks: no gradient reaches the predictions.
                properties=MetricProperties(is_symmetric=True, is_differentiable=False),
            )
            registry.register(name, entry)
