"""Base classes for stateful metrics (Tier 1-2).

Tier 1 metrics use frozen pretrained backbones to extract features
and accumulate statistics across batches. Tier 2 metrics add trainable
calibration layers on top of backbone features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import flax.nnx as nnx

from calibrax.metrics.plotting import MetricPlotMixin


class FrozenBackboneMetric(MetricPlotMixin, ABC):
    """Base class for Tier 1 metrics with frozen pretrained backbones.

    Implements the StatefulMetricProtocol lifecycle:
    - update(**kwargs): Extract features from backbone, accumulate statistics
    - compute(): Produce final metric from accumulated statistics
    - reset(): Clear accumulated state

    Subclasses must implement:
    - _extract_features(**kwargs): Run backbone on input batch
    - _accumulate(features): Update running statistics with new features
    - _compute_from_accumulated(): Produce final result from statistics

    The backbone model is frozen (no gradient updates). Subclasses load
    pretrained weights in __init__.

    Args:
        name: Unique metric identifier.

    Examples:
        >>> class MyMetric(FrozenBackboneMetric):
        ...     def __init__(self):
        ...         super().__init__(name="my_metric")
        ...         self._values = []
        ...     def reset(self):
        ...         self._values = []
        ...     def _extract_features(self, **kwargs):
        ...         return kwargs["data"]
        ...     def _accumulate(self, features):
        ...         self._values.append(float(features.mean()))
        ...     def _compute_from_accumulated(self):
        ...         return {"my_metric": sum(self._values) / len(self._values)}
    """

    def __init__(self, name: str) -> None:
        """Initialize the frozen backbone metric.

        Args:
            name: Unique metric name.
        """
        self._name = name

    @property
    def name(self) -> str:
        """Get the metric name."""
        return self._name

    def update(self, **kwargs: Any) -> None:
        """Extract features and accumulate statistics.

        Args:
            **kwargs: Batch data (images, text, etc.).
        """
        features = self._extract_features(**kwargs)
        self._accumulate(features)

    def compute(self) -> dict[str, float]:
        """Compute final metric from accumulated statistics.

        Returns:
            Dictionary mapping metric names to values.
        """
        return self._compute_from_accumulated()

    @abstractmethod
    def reset(self) -> None:
        """Reset accumulated state for a new evaluation."""
        ...

    @abstractmethod
    def _extract_features(self, **kwargs: Any) -> Any:
        """Extract features from input using the frozen backbone.

        Args:
            **kwargs: Input batch data.

        Returns:
            Extracted features (backbone-dependent format).
        """
        ...

    @abstractmethod
    def _accumulate(self, features: Any) -> None:
        """Accumulate statistics from extracted features.

        Args:
            features: Features returned by _extract_features.
        """
        ...

    @abstractmethod
    def _compute_from_accumulated(self) -> dict[str, float]:
        """Compute the final metric value from accumulated statistics.

        Returns:
            Dictionary of metric values.
        """
        ...


class LearnedMetric(MetricPlotMixin, nnx.Module):
    """Base class for Tier 2 metrics with trainable calibration layers.

    Extends nnx.Module for JAX transform compatibility (jit, grad, vmap).
    Inherits train()/eval() mode switching from nnx.Module.

    Subclasses should implement update/compute/reset following the
    StatefulMetricProtocol pattern, but with trainable parameters
    that can be optimized.

    Args:
        name: Metric name identifier.
        rngs: RNG streams for parameter initialization.

    Examples:
        >>> class MyLearnedMetric(LearnedMetric):
        ...     def __init__(self, *, rngs):
        ...         super().__init__(name="my_metric", rngs=rngs)
        ...         self._linear = nnx.Linear(4, 1, rngs=rngs)
    """

    def __init__(self, name: str, *, rngs: nnx.Rngs) -> None:
        """Initialize learned metric.

        Args:
            name: Metric name.
            rngs: RNG streams for parameter initialization.
        """
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        """Get the metric name."""
        return self._name
