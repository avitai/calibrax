"""MetricRegistry: singleton registry for metric implementations."""

from __future__ import annotations

from collections.abc import Callable

from calibrax.core.models import MetricDirection
from calibrax.core.registry import SingletonRegistry
from calibrax.metrics._types import MetricEntry, MetricProperties, MetricSignature, MetricTier


class MetricRegistry(SingletonRegistry[MetricEntry]):
    """Singleton registry for metric implementations.

    Extends SingletonRegistry[MetricEntry] with metric-specific queries
    (by domain, tier) and auto-registration of built-in metrics.

    Usage:
        registry = MetricRegistry()
        entry = registry.get("mse")
        print(entry.tier, entry.domain)
    """

    def get_function(self, name: str) -> Callable[..., float]:
        """Retrieve the callable for a Tier 0 metric.

        Args:
            name: Metric name.

        Returns:
            The metric's pure function.

        Raises:
            KeyError: If metric not found.
            TypeError: If the metric has no callable (Tier 1-3).
        """
        entry = self.get(name)
        if entry.fn is None:
            msg = f"Metric '{name}' has no callable function (tier={entry.tier})"
            raise TypeError(msg)
        return entry.fn

    def list_by_domain(self, domain: str) -> list[MetricEntry]:
        """Return all metrics in a given domain.

        Args:
            domain: Domain name (e.g., "general", "image").

        Returns:
            List of matching MetricEntry objects.
        """
        return [self.get(name) for name in self.list_names() if self.get(name).domain == domain]

    def list_by_tier(self, tier: MetricTier) -> list[MetricEntry]:
        """Return all metrics of a given computational tier.

        Args:
            tier: The MetricTier to filter by.

        Returns:
            List of matching MetricEntry objects.
        """
        return [self.get(name) for name in self.list_names() if self.get(name).tier == tier]

    def list_jit_compatible(self) -> list[MetricEntry]:
        """Return all JIT-compatible metrics.

        Returns:
            List of MetricEntry objects where is_jit_compatible is True.
        """
        return [
            self.get(name)
            for name in self.list_names()
            if self.get(name).properties.is_jit_compatible
        ]

    def list_true_metrics(self) -> list[MetricEntry]:
        """Return metrics satisfying metric space axioms.

        Returns:
            List of MetricEntry objects where is_true_metric is True.
        """
        return [
            self.get(name) for name in self.list_names() if self.get(name).properties.is_true_metric
        ]

    def list_proper_scoring_rules(self) -> list[MetricEntry]:
        """Return proper scoring rules.

        Returns:
            List of MetricEntry objects where is_proper is True.
        """
        return [self.get(name) for name in self.list_names() if self.get(name).properties.is_proper]

    def list_by_invariance(self, invariance: str) -> list[MetricEntry]:
        """Return metrics invariant under a given transformation group.

        Args:
            invariance: Transformation name (e.g., "translation", "rotation").

        Returns:
            List of MetricEntry objects with the specified invariance.
        """
        return [
            self.get(name)
            for name in self.list_names()
            if invariance in self.get(name).properties.invariances
        ]


def register_metric(
    name: str,
    *,
    tier: MetricTier = MetricTier.PURE_FUNCTION,
    domain: str = "general",
    direction: MetricDirection = MetricDirection.LOWER,
    description: str = "",
    required_extra: str = "",
    signature: MetricSignature = MetricSignature.PREDICTIONS_TARGETS,
    properties: MetricProperties | None = None,
) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """Decorator that registers a function in the MetricRegistry.

    Args:
        name: Unique metric name.
        tier: Computational tier.
        domain: Metric domain.
        direction: Whether lower or higher is better.
        description: Human-readable description.
        required_extra: PyPI extra needed for this metric.
        signature: Input signature type.
        properties: Mathematical and capability properties of the metric.

    Returns:
        Decorator that registers the function and returns it unchanged.
    """

    def decorator(fn: Callable[..., float]) -> Callable[..., float]:
        entry = MetricEntry(
            name=name,
            fn=fn,
            tier=tier,
            domain=domain,
            direction=direction,
            description=description,
            required_extra=required_extra,
            signature=signature,
            properties=properties if properties is not None else MetricProperties(),
        )
        MetricRegistry().register(name, entry)
        return fn

    return decorator
