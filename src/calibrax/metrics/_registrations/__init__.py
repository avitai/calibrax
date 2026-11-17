"""The built-in metric registrations, one module per domain as ``functional`` has.

``_register_all_builtins`` registers every domain's metrics, in this order; ``calculate_all``
evaluates the requested ones in one call.
"""

from __future__ import annotations

from calibrax.metrics._registrations import (
    audio,
    calibration,
    classification,
    clustering,
    distance,
    divergence,
    fairness,
    forecasting,
    generative,
    geometric,
    graph,
    image,
    information,
    manifold,
    ranking,
    regression,
    segmentation,
    seismic,
    statistical,
    survival,
    text,
    uncertainty,
)
from calibrax.metrics._registrations.calculate import calculate_all


__all__ = ["calculate_all"]


def _register_all_builtins() -> None:
    """Register all built-in metrics across all domains."""
    regression.register()
    classification.register()
    calibration.register()
    segmentation.register()
    distance.register()
    divergence.register()
    information.register()
    ranking.register()
    statistical.register()
    text.register()
    audio.register()
    geometric.register()
    manifold.register()
    graph.register()
    image.register()
    fairness.register()
    clustering.register()
    forecasting.register()
    uncertainty.register()
    generative.register()
    seismic.register()
    survival.register()
