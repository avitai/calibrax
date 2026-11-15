"""Registrations of the built-in ``graph`` metrics."""

from __future__ import annotations

from calibrax.core.models import MetricDirection
from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import (
    MetricEntry,
    MetricFn,
    MetricProperties,
    MetricSignature,
    MetricTier,
)
from calibrax.metrics.functional.graph import (
    graph_edit_distance_approx,
    resistance_distance,
    shortest_path_distance,
    spectral_distance,
)


def register() -> None:
    """Register 4 graph distance metrics at import time."""
    registry = MetricRegistry()

    # Between-graph metrics (two adjacency matrices)
    between_graph: list[tuple[str, MetricFn, str, bool, bool, bool]] = [
        (
            "spectral_distance",
            spectral_distance,
            "Laplacian eigenvalue spectrum distance between graphs",
            True,
            True,
            True,
        ),
        (
            "graph_edit_distance_approx",
            graph_edit_distance_approx,
            "Approximate graph edit distance via spectral relaxation",
            False,
            True,
            True,
        ),
    ]
    for name, fn, desc, is_true, is_sym, is_diff in between_graph:
        if not registry.has(name):
            registry.register(
                name,
                MetricEntry(
                    name=name,
                    fn=fn,
                    tier=MetricTier.PURE_FUNCTION,
                    domain="graph",
                    direction=MetricDirection.LOWER,
                    description=desc,
                    signature=MetricSignature.SAMPLES,
                    properties=MetricProperties(
                        is_true_metric=is_true,
                        is_symmetric=is_sym,
                        is_differentiable=is_diff,
                        invariances=("permutation",),
                    ),
                ),
            )

    # Within-graph metrics (single adjacency matrix)
    within_graph: list[tuple[str, MetricFn, str, bool]] = [
        (
            "resistance_distance",
            resistance_distance,
            "Effective electrical resistance distance matrix",
            True,
        ),
        (
            "shortest_path_distance",
            shortest_path_distance,
            "Floyd-Warshall all-pairs shortest paths",
            False,
        ),
    ]
    for name, fn, desc, is_diff in within_graph:
        if not registry.has(name):
            registry.register(
                name,
                MetricEntry(
                    name=name,
                    fn=fn,
                    tier=MetricTier.PURE_FUNCTION,
                    domain="graph",
                    direction=MetricDirection.LOWER,
                    description=desc,
                    signature=MetricSignature.SINGLE_INPUT,
                    properties=MetricProperties(
                        is_true_metric=True,
                        is_symmetric=True,
                        is_differentiable=is_diff,
                    ),
                ),
            )
