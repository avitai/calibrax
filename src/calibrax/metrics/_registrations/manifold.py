"""Registrations of the built-in ``manifold`` metrics."""

from __future__ import annotations

from calibrax.core.models import MetricDirection
from calibrax.metrics._registrations._spec import _BuiltinMetricSpec, _register_specs
from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import (
    MetricEntry,
    MetricFn,
    MetricProperties,
    MetricSignature,
    MetricTier,
)
from calibrax.metrics.functional.manifold import (
    grassmann_distance,
    spd_affine_invariant_distance,
    spd_log_euclidean_distance,
    stiefel_distance,
    ultrahyperbolic_distance,
)


def register() -> None:
    """Register 5 manifold distance metrics at import time."""
    registry = MetricRegistry()

    # (name, fn, description, is_true_metric, invariances)
    builtins: list[tuple[str, MetricFn, str, bool, tuple[str, ...]]] = [
        (
            "spd_affine_invariant_distance",
            spd_affine_invariant_distance,
            "Affine-invariant Riemannian distance on SPD manifold",
            True,
            ("affine", "congruence"),
        ),
        (
            "spd_log_euclidean_distance",
            spd_log_euclidean_distance,
            "Log-Euclidean distance between SPD matrices",
            True,
            ("orthogonal",),
        ),
        (
            "grassmann_distance",
            grassmann_distance,
            "Geodesic distance on the Grassmann manifold",
            True,
            ("orthogonal",),
        ),
        (
            "stiefel_distance",
            stiefel_distance,
            "Frobenius distance on the Stiefel manifold",
            True,
            (),
        ),
    ]
    for name, fn, desc, is_true, invs in builtins:
        if not registry.has(name):
            registry.register(
                name,
                MetricEntry(
                    name=name,
                    fn=fn,
                    tier=MetricTier.PURE_FUNCTION,
                    domain="manifold",
                    direction=MetricDirection.LOWER,
                    description=desc,
                    signature=MetricSignature.SAMPLES,
                    properties=MetricProperties(
                        is_true_metric=is_true,
                        is_symmetric=True,
                        invariances=invs,
                    ),
                ),
            )
    # The signature (p, q) is a required keyword: fn(a, b) cannot call it.
    _register_specs(
        registry,
        [
            _BuiltinMetricSpec(
                "ultrahyperbolic_distance",
                ultrahyperbolic_distance,
                "Geodesic distance on pseudo-hyperboloid with mixed signature",
                MetricDirection.LOWER,
                MetricProperties(is_symmetric=True),
                signature=MetricSignature.CUSTOM,
                domain="manifold",
            )
        ],
    )
