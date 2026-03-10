"""Metric composition: collections, weighted combinations, suites, thresholds.

Provides higher-level abstractions for grouping and combining metrics:

- ``MetricCollection``: Group multiple metrics, compute all in one call.
- ``WeightedMetric``: Weighted combination of metric values into a single score.
- ``MetricSuite``: Named groups of metrics with domain awareness.
- ``ThresholdMetric``: Wrap a metric with a pass/fail threshold for CI gates.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import MetricTier


class MetricCollection:
    """Group multiple metrics, compute all in one call.

    Supports Tier 0 pure functions via callable references.

    Usage:
        collection = MetricCollection({
            "mse": mse,
            "mae": mae,
        })
        results = collection.compute_functional(predictions, targets)
        # {"mse": 0.01, "mae": 0.05}

    Attributes:
        metrics: Dictionary mapping metric names to callables.
    """

    def __init__(
        self,
        metrics: dict[str, Callable[..., float]],
    ) -> None:
        """Initialize with a dictionary of named metric functions.

        Args:
            metrics: Mapping of metric names to callable functions.
        """
        self._metrics: dict[str, Callable[..., float]] = dict(metrics)

    def compute_functional(
        self,
        predictions: Any,
        targets: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute all functional metrics.

        Calls each callable metric with (predictions, targets, **kwargs).

        Args:
            predictions: Predicted values.
            targets: Ground truth values.
            **kwargs: Additional keyword arguments passed to each function.

        Returns:
            Dictionary mapping metric names to computed float values.
        """
        results: dict[str, float] = {}
        for name, fn in self._metrics.items():
            results[name] = float(fn(predictions, targets, **kwargs))
        return results

    def add(self, name: str, metric: Callable[..., float]) -> None:
        """Add a metric to the collection.

        Args:
            name: Name for the metric.
            metric: Callable metric function.
        """
        self._metrics[name] = metric

    def remove(self, name: str) -> None:
        """Remove a metric by name.

        Args:
            name: Name of the metric to remove.

        Raises:
            KeyError: If metric name not found.
        """
        if name not in self._metrics:
            msg = f"Metric '{name}' not found in collection"
            raise KeyError(msg)
        del self._metrics[name]

    @property
    def names(self) -> list[str]:
        """Return all metric names in the collection."""
        return list(self._metrics.keys())

    @classmethod
    def from_registry(
        cls,
        *,
        domain: str | None = None,
        tier: MetricTier = MetricTier.PURE_FUNCTION,
    ) -> MetricCollection:
        """Create a collection from all registered metrics matching filters.

        Args:
            domain: Filter by domain (None = all domains).
            tier: Filter by tier (default: PURE_FUNCTION).

        Returns:
            MetricCollection with matching metrics.
        """
        registry = MetricRegistry()
        entries = registry.list_by_tier(tier)
        if domain is not None:
            entries = [e for e in entries if e.domain == domain]
        metrics = {e.name: e.fn for e in entries if e.fn is not None}
        return cls(metrics)


class WeightedMetric:
    """Weighted combination of metric values into a single score.

    Usage:
        weighted = WeightedMetric({"mse": 0.7, "mae": 0.3})
        score = weighted.compute({"mse": 0.01, "mae": 0.05})
        # 0.7 * 0.01 + 0.3 * 0.05 = 0.022

    Attributes:
        weights: Dictionary mapping metric names to float weights.
    """

    def __init__(self, weights: dict[str, float]) -> None:
        """Initialize with metric weights.

        Args:
            weights: Metric name to weight mapping. Weights need not sum to 1.

        Raises:
            ValueError: If weights dict is empty.
        """
        if not weights:
            msg = "Weights dictionary must not be empty"
            raise ValueError(msg)
        self._weights = dict(weights)

    def compute(self, metric_values: dict[str, float]) -> float:
        """Compute weighted sum of metric values.

        Args:
            metric_values: Dictionary of metric name to value.

        Returns:
            Weighted sum as a Python float.

        Raises:
            KeyError: If a required metric is missing from metric_values.
        """
        total = 0.0
        for name, weight in self._weights.items():
            if name not in metric_values:
                msg = f"Required metric '{name}' not found in metric_values"
                raise KeyError(msg)
            total += weight * metric_values[name]
        return total

    @property
    def weights(self) -> dict[str, float]:
        """Return the weights dictionary."""
        return dict(self._weights)

    @property
    def normalized_weights(self) -> dict[str, float]:
        """Return weights normalized to sum to 1.0."""
        total = sum(self._weights.values())
        if total == 0.0:
            return dict(self._weights)
        return {name: w / total for name, w in self._weights.items()}


class MetricSuite:
    """Named groups of metrics with tier/domain awareness.

    Organizes metrics into named groups for structured evaluation.
    Can auto-populate from the MetricRegistry.

    Usage:
        suite = MetricSuite()
        suite.add_group("regression", ["mse", "mae", "rmse"])
        suite.add_group("classification", ["accuracy", "f1_score"])
        results = suite.compute_all(predictions, targets)
        # {"regression": {"mse": ..., "mae": ..., "rmse": ...},
        #  "classification": {"accuracy": ..., "f1_score": ...}}

    Attributes:
        groups: Dictionary mapping group names to metric name lists.
    """

    def __init__(self) -> None:
        """Initialize an empty metric suite."""
        self._groups: dict[str, list[str]] = {}

    def add_group(self, group_name: str, metric_names: list[str]) -> None:
        """Add a named group of metrics.

        Args:
            group_name: Name for the group.
            metric_names: List of metric names (must be registered in MetricRegistry).

        Raises:
            KeyError: If any metric name is not in the registry.
        """
        registry = MetricRegistry()
        for name in metric_names:
            if not registry.has(name):
                msg = f"Metric '{name}' not found in registry"
                raise KeyError(msg)
        self._groups[group_name] = list(metric_names)

    def compute_all(
        self,
        predictions: Any,
        targets: Any,
    ) -> dict[str, dict[str, float]]:
        """Compute all metrics in all groups.

        Args:
            predictions: Predicted values.
            targets: Ground truth values.

        Returns:
            Nested dict: {group_name: {metric_name: value}}.
        """
        registry = MetricRegistry()
        results: dict[str, dict[str, float]] = {}
        for group_name, metric_names in self._groups.items():
            group_results: dict[str, float] = {}
            for name in metric_names:
                entry = registry.get(name)
                if entry.fn is not None:
                    group_results[name] = float(entry.fn(predictions, targets))
            results[group_name] = group_results
        return results

    def list_groups(self) -> list[str]:
        """Return all group names."""
        return list(self._groups.keys())

    @classmethod
    def from_registry_domains(cls) -> MetricSuite:
        """Create a suite grouped by domain from the registry.

        Returns:
            MetricSuite with one group per domain containing all
            Tier 0 metrics in that domain.
        """
        registry = MetricRegistry()
        suite = cls()
        tier0 = registry.list_by_tier(MetricTier.PURE_FUNCTION)

        domains: dict[str, list[str]] = {}
        for entry in tier0:
            if entry.fn is not None:
                domains.setdefault(entry.domain, []).append(entry.name)

        for domain, names in sorted(domains.items()):
            suite._groups[domain] = names  # noqa: SLF001
        return suite


class ThresholdMetric:
    """Wrap a metric with a pass/fail threshold.

    Usage:
        threshold = ThresholdMetric("mse", max_value=0.01)
        result = threshold.evaluate(predictions, targets)
        # {"value": 0.005, "passed": True, "threshold": 0.01, "metric_name": "mse"}

    Attributes:
        metric_name: Name of the metric to evaluate.
        min_value: Minimum acceptable value (for HIGHER metrics).
        max_value: Maximum acceptable value (for LOWER metrics).
    """

    def __init__(
        self,
        metric_name: str,
        *,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        """Initialize threshold metric.

        Args:
            metric_name: Registered metric name.
            min_value: Minimum acceptable value (metric must be >= this).
            max_value: Maximum acceptable value (metric must be <= this).

        Raises:
            ValueError: If neither min_value nor max_value is provided.
            KeyError: If metric_name is not in the registry.
        """
        if min_value is None and max_value is None:
            msg = "At least one of min_value or max_value must be provided"
            raise ValueError(msg)

        registry = MetricRegistry()
        if not registry.has(metric_name):
            msg = f"Metric '{metric_name}' not found in registry"
            raise KeyError(msg)

        self._metric_name = metric_name
        self._min_value = min_value
        self._max_value = max_value
        self._entry = registry.get(metric_name)

    @property
    def metric_name(self) -> str:
        """Get the metric name."""
        return self._metric_name

    @property
    def min_value(self) -> float | None:
        """Get the minimum threshold value."""
        return self._min_value

    @property
    def max_value(self) -> float | None:
        """Get the maximum threshold value."""
        return self._max_value

    def evaluate(self, predictions: Any, targets: Any) -> dict[str, Any]:
        """Compute the metric and check against threshold.

        Args:
            predictions: Predicted values.
            targets: Ground truth values.

        Returns:
            Dict with "value" (float), "passed" (bool), "threshold" (float),
            "metric_name" (str).
        """
        if self._entry.fn is None:
            msg = f"Metric '{self._metric_name}' has no callable function"
            raise ValueError(msg)

        value = float(self._entry.fn(predictions, targets))
        passed = True
        threshold = self._max_value if self._max_value is not None else self._min_value

        if self._min_value is not None and value < self._min_value:
            passed = False
        if self._max_value is not None and value > self._max_value:
            passed = False

        return {
            "value": value,
            "passed": passed,
            "threshold": threshold,
            "metric_name": self._metric_name,
        }
