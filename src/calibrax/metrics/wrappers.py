"""Metric wrappers: enhance any metric with confidence intervals, per-class breakdown, tracking.

Wrappers follow the decorator pattern -- they wrap any existing metric function
and add additional behavior without modifying the original.

- ``BootstrapMetric``: Bootstrap confidence interval estimation.
- ``ClasswiseWrapper``: Per-class metric breakdown.
- ``MetricTracker``: Historical tracking with best-value detection.
- ``MinMaxTracker``: Running min/max/current tracking.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


class BootstrapMetric:
    """Wrap any metric function with bootstrap confidence interval estimation.

    A measurement without uncertainty is incomplete. This wrapper
    provides bootstrap-based confidence intervals for any metric.

    Usage:
        bootstrap = BootstrapMetric(mse, num_bootstraps=1000, confidence=0.95)
        result = bootstrap.compute(predictions, targets)
        # {"value": 0.01, "lower": 0.008, "upper": 0.012, "samples": (...)}

    Attributes:
        metric_fn: The wrapped metric function.
        num_bootstraps: Number of bootstrap resamples.
        confidence: Confidence level for interval.
    """

    def __init__(
        self,
        metric_fn: Callable[..., float],
        *,
        num_bootstraps: int = 1000,
        confidence: float = 0.95,
        seed: int = 0,
    ) -> None:
        """Initialize bootstrap wrapper.

        Args:
            metric_fn: Pure function with signature (predictions, targets) -> float.
            num_bootstraps: Number of bootstrap resamples.
            confidence: Confidence level (0 < confidence < 1).
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If confidence is not in (0, 1).
        """
        if not 0 < confidence < 1:
            msg = f"confidence must be in (0, 1), got {confidence}"
            raise ValueError(msg)
        self._metric_fn = metric_fn
        self._num_bootstraps = num_bootstraps
        self._confidence = confidence
        self._seed = seed

    @property
    def metric_fn(self) -> Callable[..., float]:
        """Get the wrapped metric function."""
        return self._metric_fn

    @property
    def num_bootstraps(self) -> int:
        """Get the number of bootstrap resamples."""
        return self._num_bootstraps

    @property
    def confidence(self) -> float:
        """Get the confidence level."""
        return self._confidence

    def compute(
        self,
        predictions: Any,
        targets: Any,
    ) -> dict[str, Any]:
        """Compute metric with bootstrap confidence interval.

        Args:
            predictions: Predicted values.
            targets: Ground truth values.

        Returns:
            Dict with "value" (point estimate), "lower" (CI lower bound),
            "upper" (CI upper bound), "samples" (all bootstrap values).
        """
        predictions = jnp.asarray(predictions)
        targets = jnp.asarray(targets)
        n = predictions.shape[0]

        # Point estimate on full data
        value = float(self._metric_fn(predictions, targets))

        # Bootstrap resamples
        key = jax.random.PRNGKey(self._seed)
        samples = []
        for i in range(self._num_bootstraps):
            key, subkey = jax.random.split(key)
            indices = jax.random.randint(subkey, shape=(n,), minval=0, maxval=n)
            boot_pred = predictions[indices]
            boot_tgt = targets[indices]
            samples.append(float(self._metric_fn(boot_pred, boot_tgt)))

        sorted_samples = sorted(samples)
        alpha = 1.0 - self._confidence
        lower_idx = int(alpha / 2 * self._num_bootstraps)
        upper_idx = int((1 - alpha / 2) * self._num_bootstraps) - 1
        lower_idx = max(0, lower_idx)
        upper_idx = min(len(sorted_samples) - 1, upper_idx)

        return {
            "value": value,
            "lower": sorted_samples[lower_idx],
            "upper": sorted_samples[upper_idx],
            "samples": tuple(samples),
        }


class ClasswiseWrapper:
    """Wrap any metric to compute it separately for each class.

    Provides per-class breakdown of any (predictions, targets) -> float
    metric. Useful for identifying which classes a model performs poorly on.

    Usage:
        classwise = ClasswiseWrapper(mse, class_names=["cat", "dog", "bird"])
        result = classwise.compute(predictions, targets, labels)
        # {"cat": 0.01, "dog": 0.03, "bird": 0.02, "mean": 0.02}
    """

    def __init__(
        self,
        metric_fn: Callable[..., float],
        *,
        class_names: list[str] | None = None,
    ) -> None:
        """Initialize classwise wrapper.

        Args:
            metric_fn: Pure function with signature (predictions, targets) -> float.
            class_names: Optional human-readable class names. If None, uses
                integer indices as keys.
        """
        self._metric_fn = metric_fn
        self._class_names = class_names

    def compute(
        self,
        predictions: Any,
        targets: Any,
        labels: Any,
    ) -> dict[str, float]:
        """Compute metric per class.

        Args:
            predictions: Predicted values.
            targets: Ground truth values.
            labels: Class labels for grouping (integer array).

        Returns:
            Dict mapping class names to metric values, plus "mean" key.
        """
        predictions = jnp.asarray(predictions)
        targets = jnp.asarray(targets)
        labels = jnp.asarray(labels)

        unique_labels = jnp.unique(labels, size=int(jnp.max(labels)) + 1)
        results: dict[str, float] = {}
        values = []

        for i, label in enumerate(unique_labels):
            mask = labels == label
            if int(jnp.sum(mask)) < 1:
                continue
            class_pred = predictions[mask]
            class_tgt = targets[mask]
            val = float(self._metric_fn(class_pred, class_tgt))

            if self._class_names is not None and i < len(self._class_names):
                key = self._class_names[i]
            else:
                key = str(int(label))
            results[key] = val
            values.append(val)

        if values:
            results["mean"] = sum(values) / len(values)
        else:
            results["mean"] = 0.0

        return results


class MetricTracker:
    """Track a metric's history across multiple evaluation epochs.

    Maintains a history of metric values with automatic best-value
    detection based on direction (higher/lower is better).

    Usage:
        tracker = MetricTracker(mse, direction="lower")
        tracker.increment(predictions_1, targets_1)
        tracker.increment(predictions_2, targets_2)
        print(tracker.best())        # Lowest MSE seen
        print(tracker.history)       # (0.05, 0.03)
        print(tracker.best_epoch)    # 1 (0-indexed)
    """

    def __init__(
        self,
        metric_fn: Callable[..., float],
        *,
        direction: str = "lower",
    ) -> None:
        """Initialize metric tracker.

        Args:
            metric_fn: Pure function with signature (predictions, targets) -> float.
            direction: "lower" or "higher" -- determines what "best" means.

        Raises:
            ValueError: If direction is not "lower" or "higher".
        """
        if direction not in ("lower", "higher"):
            msg = f"direction must be 'lower' or 'higher', got '{direction}'"
            raise ValueError(msg)
        self._metric_fn = metric_fn
        self._direction = direction
        self._history: list[float] = []

    def increment(self, predictions: Any, targets: Any) -> float:
        """Compute metric and add to history.

        Args:
            predictions: Predicted values.
            targets: Ground truth values.

        Returns:
            The computed metric value.
        """
        value = float(self._metric_fn(predictions, targets))
        self._history.append(value)
        return value

    def best(self) -> float:
        """Return the best metric value seen so far.

        Returns:
            Best value (min for "lower", max for "higher").

        Raises:
            ValueError: If no values have been tracked.
        """
        if not self._history:
            msg = "No values have been tracked yet"
            raise ValueError(msg)
        if self._direction == "lower":
            return min(self._history)
        return max(self._history)

    @property
    def best_epoch(self) -> int:
        """Return the epoch index of the best metric value.

        Raises:
            ValueError: If no values have been tracked.
        """
        if not self._history:
            msg = "No values have been tracked yet"
            raise ValueError(msg)
        if self._direction == "lower":
            return self._history.index(min(self._history))
        return self._history.index(max(self._history))

    @property
    def history(self) -> tuple[float, ...]:
        """Return all tracked values as an immutable tuple."""
        return tuple(self._history)

    def reset(self) -> None:
        """Clear all tracked history."""
        self._history.clear()


class MinMaxTracker:
    """Track running min, max, and current value for any metric.

    Useful for monitoring metric ranges during training without
    storing full history.

    Usage:
        tracker = MinMaxTracker(mse)
        tracker.update(predictions, targets)
        print(tracker.current)  # Latest value
        print(tracker.min)      # Lowest seen
        print(tracker.max)      # Highest seen
    """

    def __init__(self, metric_fn: Callable[..., float]) -> None:
        """Initialize min/max tracker.

        Args:
            metric_fn: Pure function with signature (predictions, targets) -> float.
        """
        self._metric_fn = metric_fn
        self._current: float | None = None
        self._min: float | None = None
        self._max: float | None = None

    def update(self, predictions: Any, targets: Any) -> float:
        """Compute metric and update min/max tracking.

        Args:
            predictions: Predicted values.
            targets: Ground truth values.

        Returns:
            The computed metric value.
        """
        value = float(self._metric_fn(predictions, targets))
        self._current = value
        if self._min is None or value < self._min:
            self._min = value
        if self._max is None or value > self._max:
            self._max = value
        return value

    @property
    def current(self) -> float | None:
        """Return the most recently computed value."""
        return self._current

    @property
    def min(self) -> float | None:
        """Return the minimum value seen."""
        return self._min

    @property
    def max(self) -> float | None:
        """Return the maximum value seen."""
        return self._max

    def reset(self) -> None:
        """Reset all tracking state."""
        self._current = None
        self._min = None
        self._max = None
