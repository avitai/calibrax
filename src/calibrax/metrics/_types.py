"""Metric type definitions: MetricTier enum, MetricProperties, and MetricEntry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import jax

from calibrax.core.models import MetricDirection


MetricFn = Callable[..., jax.Array | float]
"""A metric function: a JAX array out, or a Python float from a host-side metric (text, loops)."""

MetricValues = dict[str, jax.Array | float]
"""Metric results by name, as ``calculate_all`` returns them."""


class MetricTier(StrEnum):
    """Computational tier of a metric implementation."""

    PURE_FUNCTION = "pure_function"
    FROZEN_BACKBONE = "frozen_backbone"
    LEARNED = "learned"
    METRIC_LEARNING = "metric_learning"


class MetricSignature(StrEnum):
    """Input signature type for a metric function.

    Documents what inputs the metric expects, enabling correct dispatch
    and clear documentation.
    """

    PREDICTIONS_TARGETS = "predictions_targets"
    ENSEMBLE_PREDICTIONS_TARGETS = "ensemble_predictions_targets"
    DISTRIBUTIONS = "distributions"
    SAMPLES = "samples"
    FEATURES_LABELS = "features_labels"
    SINGLE_INPUT = "single_input"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricProperties:
    """Mathematical and capability properties of a metric.

    Groups boolean axiom flags and invariance metadata that describe
    a metric's mathematical guarantees and JAX compatibility.

    Attributes:
        is_true_metric: Satisfies identity + symmetry + triangle inequality
            (metric space axioms).
        is_symmetric: d(x, y) = d(y, x). True for all proper metrics, false
            for KL divergence, etc.
        is_proper: Is a proper scoring rule (minimized by true distribution).
            Relevant for calibration/probabilistic metrics (Brier, log loss, CRPS).
        is_differentiable: ``jax.grad`` with respect to the first argument is finite and not
            identically zero on generic inputs: False where that argument is labels, ranks,
            a thresholded value or text, whose gradient is zero or does not exist.
        is_jit_compatible: ``jax.jit`` of the call reproduces the eager value, with keyword
            arguments that set shapes (counts, cutoffs, bins) passed statically. False for
            string-based metrics (BLEU, ROUGE).

    ``tests/metrics/test_registry_conformance.py`` checks both flags, and each entry's
    signature, against every registered metric.
        invariances: Transformation groups under which the metric is invariant,
            following the Erlangen Program. Documents what symmetries the metric
            preserves. Common values: "translation" (Lp norms), "rotation"
            (Euclidean), "scale" (cosine, MAPE), "permutation" (Hausdorff,
            Chamfer), "reparametrization" (Fisher), "affine" (Mahalanobis).
            Empty tuple means invariances are not documented or the metric has
            no standard invariance classification.
    """

    is_true_metric: bool = False
    is_symmetric: bool = False
    is_proper: bool = False
    is_differentiable: bool = True
    is_jit_compatible: bool = True
    invariances: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricEntry:
    """Registry metadata for a metric implementation.

    Describes HOW to compute a metric (callable, tier, dependencies).
    Distinct from MetricDef in core/models.py which describes HOW TO
    INTERPRET a metric result (direction, unit, priority, group).

    Attributes:
        name: Unique metric identifier (e.g., "mse", "fid", "contrastive_loss").
        fn: Pure function reference for Tier 0, or None for Tier 1-3.
        tier: Computational tier determining the execution pattern.
        domain: Metric domain (e.g., "general", "image", "text", "audio").
        direction: Whether lower or higher values are better.
        description: Human-readable description of the metric.
        signature: Input signature type documenting what arguments the metric expects.
        properties: Mathematical and capability properties of the metric.
    """

    name: str
    fn: MetricFn | None
    tier: MetricTier
    domain: str = "general"
    direction: MetricDirection = MetricDirection.LOWER
    description: str = ""
    signature: MetricSignature = MetricSignature.PREDICTIONS_TARGETS
    properties: MetricProperties = MetricProperties()
