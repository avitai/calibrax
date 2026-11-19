"""Registrations of the built-in ``audio`` metrics."""

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
from calibrax.metrics.functional.audio import (
    mel_cepstral_distortion,
    signal_to_noise_ratio,
    spectral_convergence,
)


def register() -> None:
    """Register 3 audio metrics at import time."""
    registry = MetricRegistry()
    builtins: list[tuple[str, MetricFn, str, MetricDirection]] = [
        (
            "spectral_convergence",
            spectral_convergence,
            "Spectral convergence ratio",
            MetricDirection.LOWER,
        ),
        (
            "mel_cepstral_distortion",
            mel_cepstral_distortion,
            "Mel cepstral distortion (dB)",
            MetricDirection.LOWER,
        ),
        (
            "signal_to_noise_ratio",
            signal_to_noise_ratio,
            "Signal-to-noise ratio (dB)",
            MetricDirection.HIGHER,
        ),
    ]
    for name, fn, desc, direction in builtins:
        entry = MetricEntry(
            name=name,
            fn=fn,
            tier=MetricTier.PURE_FUNCTION,
            domain="audio",
            direction=direction,
            description=desc,
            signature=MetricSignature.PREDICTIONS_TARGETS,
            properties=MetricProperties(is_symmetric=True),
        )
        registry.register(name, entry)
