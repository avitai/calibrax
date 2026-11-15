"""Registrations of the built-in ``fairness`` metrics."""

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
from calibrax.metrics.functional.fairness import (
    demographic_parity_ratio,
    disparate_impact_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
)


def register() -> None:
    """Register 4 fairness metrics at import time."""
    registry = MetricRegistry()
    # (name, fn, description, direction)
    builtins: list[tuple[str, MetricFn, str, MetricDirection]] = [
        (
            "demographic_parity_ratio",
            demographic_parity_ratio,
            "Ratio of positive prediction rates across demographic groups",
            MetricDirection.HIGHER,
        ),
        (
            "equalized_odds_difference",
            equalized_odds_difference,
            "Max absolute difference in TPR or FPR across groups",
            MetricDirection.LOWER,
        ),
        (
            "equal_opportunity_difference",
            equal_opportunity_difference,
            "Absolute difference in TPR across demographic groups",
            MetricDirection.LOWER,
        ),
        (
            "disparate_impact_ratio",
            disparate_impact_ratio,
            "Disparate impact ratio (80% rule)",
            MetricDirection.HIGHER,
        ),
    ]
    for name, fn, desc, direction in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="fairness",
                direction=direction,
                description=desc,
                signature=MetricSignature.CUSTOM,
                # Predictions are thresholded: no gradient reaches them.
                properties=MetricProperties(is_differentiable=False),
            )
            registry.register(name, entry)
