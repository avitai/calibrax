"""Registrations of the built-in ``geometric`` metrics."""

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
from calibrax.metrics.functional.geometric import (
    chamfer_distance,
    directed_hausdorff,
    earth_movers_distance_1d,
    hausdorff_distance,
    rmsd,
)


def register() -> None:
    """Register 4 geometric metrics at import time."""
    registry = MetricRegistry()
    # (name, fn, description, is_true_metric, is_symmetric)
    builtins: list[tuple[str, MetricFn, str, bool, bool]] = [
        ("chamfer_distance", chamfer_distance, "Chamfer distance between point sets", False, True),
        ("rmsd", rmsd, "RMSD between centred conformations over their shared atoms", True, True),
        (
            "earth_movers_distance_1d",
            earth_movers_distance_1d,
            "1D Earth Mover's Distance (Wasserstein-1)",
            True,
            True,
        ),
        (
            "directed_hausdorff",
            directed_hausdorff,
            "Directed Hausdorff distance (asymmetric)",
            False,
            False,
        ),
        (
            "hausdorff_distance",
            hausdorff_distance,
            "Hausdorff distance (symmetric)",
            True,
            True,
        ),
    ]
    for name, fn, desc, is_true, is_sym in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="geometric",
                direction=MetricDirection.LOWER,
                description=desc,
                signature=MetricSignature.SAMPLES,
                properties=MetricProperties(
                    is_true_metric=is_true,
                    is_symmetric=is_sym,
                ),
            )
            registry.register(name, entry)
