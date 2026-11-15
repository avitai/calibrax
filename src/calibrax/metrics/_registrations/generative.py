"""Registrations of the built-in ``generative`` metrics."""

from __future__ import annotations

from calibrax.core.models import MetricDirection
from calibrax.metrics._registrations._spec import _BuiltinMetricSpec, _register_specs
from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import MetricProperties, MetricSignature
from calibrax.metrics.functional.generative import (
    density_weighted_precision,
    density_weighted_recall,
    distance_to_closest_record,
    frechet_feature_distance,
    inception_score,
    manifold_precision,
    manifold_recall,
    memorization_rate,
)
from calibrax.metrics.functional.statistical import correlation_preservation


def register() -> None:
    """Register the sample-based generative-model metrics."""
    registry = MetricRegistry()
    counting = MetricProperties(is_differentiable=False, is_jit_compatible=True)
    smooth = MetricProperties(is_differentiable=True, is_jit_compatible=True)
    builtins = [
        _BuiltinMetricSpec(
            "frechet_feature_distance",
            frechet_feature_distance,
            "Frechet distance between the Gaussians fitted to real and generated features (FID core)",
            MetricDirection.LOWER,
            MetricProperties(is_symmetric=True, is_differentiable=True, is_jit_compatible=True),
            signature=MetricSignature.SAMPLES,
            domain="generative",
        ),
        _BuiltinMetricSpec(
            "inception_score",
            inception_score,
            "exp of the mean KL between conditional and marginal class probabilities",
            MetricDirection.HIGHER,
            smooth,
            signature=MetricSignature.SINGLE_INPUT,
            domain="generative",
        ),
        _BuiltinMetricSpec(
            "correlation_preservation",
            correlation_preservation,
            "One minus the mean absolute difference of real and generated feature correlations",
            MetricDirection.HIGHER,
            smooth,
            signature=MetricSignature.SAMPLES,
            domain="generative",
        ),
        _BuiltinMetricSpec(
            "manifold_precision",
            manifold_precision,
            "Fraction of generated samples inside the real k-NN manifold",
            MetricDirection.HIGHER,
            counting,
            signature=MetricSignature.SAMPLES,
            domain="generative",
        ),
        _BuiltinMetricSpec(
            "manifold_recall",
            manifold_recall,
            "Fraction of real samples inside the generated k-NN manifold",
            MetricDirection.HIGHER,
            counting,
            signature=MetricSignature.SAMPLES,
            domain="generative",
        ),
        _BuiltinMetricSpec(
            "density_weighted_precision",
            density_weighted_precision,
            "Manifold precision weighted by real-manifold density",
            MetricDirection.HIGHER,
            counting,
            signature=MetricSignature.SAMPLES,
            domain="generative",
        ),
        _BuiltinMetricSpec(
            "density_weighted_recall",
            density_weighted_recall,
            "Manifold recall weighted by generated-manifold density",
            MetricDirection.HIGHER,
            counting,
            signature=MetricSignature.SAMPLES,
            domain="generative",
        ),
        _BuiltinMetricSpec(
            "distance_to_closest_record",
            distance_to_closest_record,
            "Mean normalised distance from generated to closest real record",
            MetricDirection.HIGHER,
            MetricProperties(is_differentiable=True, is_jit_compatible=True),
            signature=MetricSignature.SAMPLES,
            domain="generative",
        ),
        _BuiltinMetricSpec(
            "memorization_rate",
            memorization_rate,
            "Fraction of generated records identical to a real record",
            MetricDirection.LOWER,
            counting,
            signature=MetricSignature.SAMPLES,
            domain="generative",
        ),
    ]
    _register_specs(registry, builtins)
