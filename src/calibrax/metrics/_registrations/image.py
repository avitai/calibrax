"""Registrations of the built-in ``image`` metrics."""

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
from calibrax.metrics.functional.image import ms_ssim, psnr, ssim, vendi_score


def register() -> None:
    """Register 4 image quality metrics at import time."""
    registry = MetricRegistry()
    builtins: list[tuple[str, MetricFn, str]] = [
        ("psnr", psnr, "Peak Signal-to-Noise Ratio (dB)"),
        ("ssim", ssim, "Structural Similarity Index Measure"),
        ("ms_ssim", ms_ssim, "Multi-Scale Structural Similarity"),
    ]
    for name, fn, desc in builtins:
        entry = MetricEntry(
            name=name,
            fn=fn,
            tier=MetricTier.PURE_FUNCTION,
            domain="image",
            direction=MetricDirection.HIGHER,
            description=desc,
            signature=MetricSignature.PREDICTIONS_TARGETS,
            properties=MetricProperties(is_symmetric=True),
        )
        registry.register(name, entry)
    # One similarity matrix in, a diversity out.
    _register_specs(
        registry,
        [
            _BuiltinMetricSpec(
                "vendi_score",
                vendi_score,
                "Vendi Score (diversity via eigenvalue entropy)",
                MetricDirection.HIGHER,
                MetricProperties(),
                signature=MetricSignature.SINGLE_INPUT,
                domain="image",
            )
        ],
    )
