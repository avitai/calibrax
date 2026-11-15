"""Registrations of the built-in ``divergence`` metrics."""

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
from calibrax.metrics.functional.divergence import (
    bregman_divergence,
    chi_squared_divergence,
    f_divergence,
    hellinger_distance,
    js_divergence,
    kl_divergence,
    kolmogorov_smirnov_distance,
    mmd,
    mmd_squared_unbiased,
    registry_sliced_wasserstein,
    renyi_divergence,
    reverse_kl_divergence,
    sinkhorn_divergence,
    total_variation,
    wasserstein_1d,
)


def register() -> None:
    """Register 13 divergence metrics at import time."""
    registry = MetricRegistry()
    # (name, fn, description, is_true_metric, is_symmetric, is_differentiable, signature)
    builtins: list[tuple[str, MetricFn, str, bool, bool, bool, MetricSignature]] = [
        (
            "kl_divergence",
            kl_divergence,
            "Kullback-Leibler divergence",
            False,
            False,
            True,
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        (
            "reverse_kl_divergence",
            reverse_kl_divergence,
            "Reverse KL divergence (mode-seeking)",
            False,
            False,
            True,
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        (
            "js_divergence",
            js_divergence,
            "Jensen-Shannon divergence",
            True,
            True,
            True,
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        (
            "total_variation",
            total_variation,
            "Total variation distance",
            True,
            True,
            True,
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        (
            "hellinger_distance",
            hellinger_distance,
            "Hellinger distance",
            True,
            True,
            True,
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        (
            "chi_squared_divergence",
            chi_squared_divergence,
            "Pearson chi-squared divergence",
            False,
            False,
            True,
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        (
            "renyi_divergence",
            renyi_divergence,
            "Renyi alpha-divergence",
            False,
            False,
            True,
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        (
            "f_divergence",
            f_divergence,
            "Unified f-divergence",
            False,
            False,
            True,
            MetricSignature.CUSTOM,
        ),
        (
            "wasserstein_1d",
            wasserstein_1d,
            "1D Wasserstein-1 distance",
            True,
            True,
            True,
            MetricSignature.SAMPLES,
        ),
        (
            "kolmogorov_smirnov_distance",
            kolmogorov_smirnov_distance,
            "Kolmogorov-Smirnov distance between two samples",
            True,
            True,
            False,
            MetricSignature.SAMPLES,
        ),
        (
            "mmd",
            mmd,
            "Maximum Mean Discrepancy",
            True,
            True,
            True,
            MetricSignature.SAMPLES,
        ),
        (
            "mmd_squared_unbiased",
            mmd_squared_unbiased,
            "Unbiased estimate of squared Maximum Mean Discrepancy (two-sample test statistic)",
            False,
            True,
            True,
            MetricSignature.SAMPLES,
        ),
        (
            "sinkhorn_divergence",
            sinkhorn_divergence,
            "Debiased Sinkhorn divergence",
            False,
            True,
            True,
            MetricSignature.SAMPLES,
        ),
        (
            "sliced_wasserstein",
            registry_sliced_wasserstein,
            "Sliced Wasserstein distance over a fixed projection set",
            False,
            True,
            True,
            MetricSignature.SAMPLES,
        ),
        (
            "bregman_divergence",
            bregman_divergence,
            "Bregman divergence with convex generator",
            False,
            False,
            True,
            MetricSignature.CUSTOM,
        ),
    ]
    for name, fn, desc, is_true, is_sym, is_diff, sig in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="divergence",
                direction=MetricDirection.LOWER,
                description=desc,
                signature=sig,
                properties=MetricProperties(
                    is_true_metric=is_true,
                    is_symmetric=is_sym,
                    is_differentiable=is_diff,
                ),
            )
            registry.register(name, entry)
