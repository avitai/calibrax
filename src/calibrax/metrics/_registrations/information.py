"""Registrations of the built-in ``information`` metrics."""

from __future__ import annotations

from calibrax.core.models import MetricDirection
from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import MetricEntry, MetricFn, MetricSignature, MetricTier
from calibrax.metrics.functional.information import (
    conditional_entropy,
    cross_entropy,
    entropy,
    mutual_information,
    normalized_mutual_information,
)


def register() -> None:
    """Register 5 information-theoretic metrics at import time.

    Note: fisher_information_matrix is NOT registered (returns matrix, not float).
    """
    registry = MetricRegistry()
    # (name, fn, description, direction, signature)
    builtins: list[tuple[str, MetricFn, str, MetricDirection, MetricSignature]] = [
        (
            "entropy",
            entropy,
            "Shannon entropy",
            MetricDirection.INFO,
            MetricSignature.SINGLE_INPUT,
        ),
        (
            "cross_entropy",
            cross_entropy,
            "Cross-entropy",
            MetricDirection.LOWER,
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        (
            "mutual_information",
            mutual_information,
            "Mutual information",
            MetricDirection.HIGHER,
            MetricSignature.SINGLE_INPUT,
        ),
        (
            "conditional_entropy",
            conditional_entropy,
            "Conditional entropy H(Y|X)",
            MetricDirection.LOWER,
            MetricSignature.SINGLE_INPUT,
        ),
        (
            "normalized_mutual_information",
            normalized_mutual_information,
            "Normalized mutual information",
            MetricDirection.HIGHER,
            MetricSignature.SINGLE_INPUT,
        ),
    ]
    for name, fn, desc, direction, sig in builtins:
        entry = MetricEntry(
            name=name,
            fn=fn,
            tier=MetricTier.PURE_FUNCTION,
            domain="information",
            direction=direction,
            description=desc,
            signature=sig,
        )
        registry.register(name, entry)
