"""The built-in metric spec and the loop that registers a list of them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from calibrax.core.models import MetricDirection
from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import (
    MetricEntry,
    MetricFn,
    MetricProperties,
    MetricSignature,
    MetricTier,
)


@dataclass(frozen=True)
class _BuiltinMetricSpec:
    """Named built-in metric registration metadata."""

    name: str
    fn: MetricFn
    description: str
    direction: MetricDirection
    properties: MetricProperties
    signature: MetricSignature = MetricSignature.PREDICTIONS_TARGETS
    domain: str = "general"


def _register_specs(registry: MetricRegistry, specs: Sequence[_BuiltinMetricSpec]) -> None:
    """Register every spec that is not already in the registry."""
    for spec in specs:
        if not registry.has(spec.name):
            registry.register(
                spec.name,
                MetricEntry(
                    name=spec.name,
                    fn=spec.fn,
                    tier=MetricTier.PURE_FUNCTION,
                    domain=spec.domain,
                    direction=spec.direction,
                    description=spec.description,
                    signature=spec.signature,
                    properties=spec.properties,
                ),
            )
