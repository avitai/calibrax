"""Registrations of the built-in ``ranking`` metrics."""

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
from calibrax.metrics.functional.ranking import (
    bedroc,
    coverage,
    enrichment_factor,
    hit_rate,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    sequential_discovery_nauc,
    sequential_shortfall,
)


def register() -> None:
    """Register the ranking/retrieval metrics at import time."""
    registry = MetricRegistry()
    # Ranks carry no gradient to the scores; the cutoff k is a required keyword.
    ranked = MetricProperties(is_differentiable=False)
    at_k: list[tuple[str, MetricFn, str, MetricSignature]] = [
        (
            "ndcg",
            ndcg,
            "Normalized Discounted Cumulative Gain",
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        ("ndcg_at_k", ndcg_at_k, "NDCG truncated to top-k", MetricSignature.CUSTOM),
        (
            "mean_average_precision",
            mean_average_precision,
            "Mean Average Precision",
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        ("precision_at_k", precision_at_k, "Precision at k", MetricSignature.CUSTOM),
        ("recall_at_k", recall_at_k, "Recall at k", MetricSignature.CUSTOM),
        (
            "mean_reciprocal_rank",
            mean_reciprocal_rank,
            "Mean Reciprocal Rank",
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        ("hit_rate", hit_rate, "Hit rate at k", MetricSignature.CUSTOM),
        (
            "bedroc",
            bedroc,
            "Boltzmann-Enhanced Discrimination of ROC",
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        (
            "enrichment_factor",
            enrichment_factor,
            "Early enrichment factor at screening fraction",
            MetricSignature.PREDICTIONS_TARGETS,
        ),
    ]
    _register_specs(
        registry,
        [
            _BuiltinMetricSpec(
                name, fn, desc, MetricDirection.HIGHER, ranked, signature=sig, domain="ranking"
            )
            for name, fn, desc, sig in at_k
        ]
        + [
            _BuiltinMetricSpec(
                "sequential_discovery_nauc",
                sequential_discovery_nauc,
                "Oracle-normalized area under sequential discovery curve",
                MetricDirection.HIGHER,
                MetricProperties(is_differentiable=True, is_jit_compatible=True),
                signature=MetricSignature.CUSTOM,
                domain="ranking",
            ),
            _BuiltinMetricSpec(
                "sequential_shortfall",
                sequential_shortfall,
                "Relative sequential discovery shortfall against oracle curve",
                MetricDirection.LOWER,
                MetricProperties(is_differentiable=True, is_jit_compatible=True),
                signature=MetricSignature.CUSTOM,
                domain="ranking",
            ),
        ],
    )
    # One input and a required catalog size: a suite cannot call it as fn(predictions, targets).
    registry.register(
        "coverage",
        MetricEntry(
            name="coverage",
            fn=coverage,
            tier=MetricTier.PURE_FUNCTION,
            domain="ranking",
            direction=MetricDirection.HIGHER,
            description="Catalog coverage",
            signature=MetricSignature.CUSTOM,
            properties=MetricProperties(is_differentiable=False),
        ),
    )
