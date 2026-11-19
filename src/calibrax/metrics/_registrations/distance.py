"""Registrations of the built-in ``distance`` metrics."""

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
from calibrax.metrics.functional.distance import (
    chebyshev_distance,
    cosine_distance,
    euclidean_distance,
    hamming_distance,
    jaccard_distance,
    lorentz_distance,
    mahalanobis_distance,
    manhattan_distance,
    minkowski_distance,
    poincare_distance,
    randers_distance,
)


def register() -> None:
    """Register 11 distance metrics at import time."""
    registry = MetricRegistry()
    # (name, fn, description, is_true_metric, is_symmetric, is_differentiable, invariances)
    builtins: list[tuple[str, MetricFn, str, bool, bool, bool, tuple[str, ...]]] = [
        (
            "cosine_distance",
            cosine_distance,
            "Cosine distance (1 - similarity)",
            False,
            True,
            True,
            ("scale",),
        ),
        (
            "euclidean_distance",
            euclidean_distance,
            "Euclidean (L2) distance",
            True,
            True,
            True,
            ("rotation", "translation"),
        ),
        (
            "manhattan_distance",
            manhattan_distance,
            "Manhattan (L1) distance",
            True,
            True,
            True,
            ("translation",),
        ),
        (
            "chebyshev_distance",
            chebyshev_distance,
            "Chebyshev (L-infinity) distance",
            True,
            True,
            True,
            ("translation",),
        ),
        (
            "mahalanobis_distance",
            mahalanobis_distance,
            "Mahalanobis distance",
            True,
            True,
            True,
            (),
        ),
        (
            "hamming_distance",
            hamming_distance,
            "Hamming distance (fraction differing)",
            True,
            True,
            False,
            (),
        ),
        (
            "minkowski_distance",
            minkowski_distance,
            "Minkowski (Lp) distance",
            True,
            True,
            True,
            ("translation",),
        ),
        (
            "jaccard_distance",
            jaccard_distance,
            "Jaccard distance (set-based)",
            True,
            True,
            True,
            (),
        ),
        (
            "poincare_distance",
            poincare_distance,
            "Poincare ball geodesic distance",
            True,
            True,
            True,
            (),
        ),
        (
            "lorentz_distance",
            lorentz_distance,
            "Lorentz hyperboloid geodesic distance",
            True,
            True,
            True,
            ("lorentz",),
        ),
    ]
    for name, fn, desc, is_true, is_sym, is_diff, invs in builtins:
        entry = MetricEntry(
            name=name,
            fn=fn,
            tier=MetricTier.PURE_FUNCTION,
            domain="distance",
            direction=MetricDirection.LOWER,
            description=desc,
            signature=MetricSignature.PREDICTIONS_TARGETS,
            properties=MetricProperties(
                is_true_metric=is_true,
                is_symmetric=is_sym,
                is_differentiable=is_diff,
                invariances=invs,
            ),
        )
        registry.register(name, entry)
    # A direction and a magnitude are required keywords: fn(predictions, targets) cannot call it.
    _register_specs(
        registry,
        [
            _BuiltinMetricSpec(
                "randers_distance",
                randers_distance,
                "Randers asymmetric Finsler distance",
                MetricDirection.LOWER,
                MetricProperties(is_differentiable=True),
                signature=MetricSignature.CUSTOM,
                domain="distance",
            )
        ],
    )
