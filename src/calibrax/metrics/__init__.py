"""Metrics: JAX-native evaluation metrics for the calibrax ecosystem.

Provides a 4-tier metric system:
- Tier 0: Pure functions via ``calibrax.metrics.functional``
- Tier 1: Frozen backbone metrics via ``calibrax.metrics.stateful``
- Tier 2: Learned calibration metrics via ``calibrax.metrics.stateful``
- Tier 3: Metric learning losses via ``calibrax.metrics.learning``

MetricRegistry for metric discovery, and calculate_all for batch computation.

Individual metric functions live in their domain modules
(e.g., ``calibrax.metrics.functional.regression``). This top-level package
exports registry infrastructure, types, composition, and wrapper classes.
"""

from calibrax.metrics._builtin_registrations import _register_all_builtins, calculate_all
from calibrax.metrics._registry import MetricRegistry, register_metric
from calibrax.metrics._types import MetricEntry, MetricProperties, MetricSignature, MetricTier
from calibrax.metrics.composition import (
    MetricCollection,
    MetricSuite,
    ThresholdMetric,
    WeightedMetric,
)
from calibrax.metrics.stateful import FrozenBackboneMetric, LearnedMetric
from calibrax.metrics.wrappers import (
    BootstrapMetric,
    ClasswiseWrapper,
    MetricTracker,
    MinMaxTracker,
)


# Register built-in metrics on first import
_register_all_builtins()

__all__ = [
    # Composition
    "MetricCollection",
    "MetricSuite",
    "ThresholdMetric",
    "WeightedMetric",
    # Registry & types
    "MetricEntry",
    "MetricProperties",
    "MetricRegistry",
    "MetricSignature",
    "MetricTier",
    "calculate_all",
    "register_metric",
    # Stateful base classes
    "FrozenBackboneMetric",
    "LearnedMetric",
    # Wrappers
    "BootstrapMetric",
    "ClasswiseWrapper",
    "MetricTracker",
    "MinMaxTracker",
]
