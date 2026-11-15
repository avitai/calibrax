"""Registrations of the built-in ``statistical`` metrics."""

from __future__ import annotations

from calibrax.core.models import MetricDirection
from calibrax.metrics._registrations._spec import _BuiltinMetricSpec, _register_specs
from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import MetricProperties, MetricSignature
from calibrax.metrics.functional.statistical import (
    concordance_correlation,
    kendall_tau,
    pearson_correlation,
    skewness,
    spearman_rank_correlation,
)


def register() -> None:
    """Register 5 statistical correlation metrics at import time."""
    registry = MetricRegistry()
    symmetric = MetricProperties(is_symmetric=True)
    # Rank correlations carry no gradient to the values they rank.
    ranked = MetricProperties(is_symmetric=True, is_differentiable=False)
    _register_specs(
        registry,
        [
            _BuiltinMetricSpec(
                "pearson_correlation",
                pearson_correlation,
                "Pearson correlation coefficient",
                MetricDirection.HIGHER,
                symmetric,
                domain="statistical",
            ),
            _BuiltinMetricSpec(
                "spearman_rank_correlation",
                spearman_rank_correlation,
                "Spearman rank correlation",
                MetricDirection.HIGHER,
                ranked,
                domain="statistical",
            ),
            _BuiltinMetricSpec(
                "kendall_tau",
                kendall_tau,
                "Kendall rank correlation coefficient",
                MetricDirection.HIGHER,
                ranked,
                domain="statistical",
            ),
            _BuiltinMetricSpec(
                "concordance_correlation",
                concordance_correlation,
                "Lin's concordance correlation",
                MetricDirection.HIGHER,
                symmetric,
                domain="statistical",
            ),
            _BuiltinMetricSpec(
                "skewness",
                skewness,
                "Third standardised moment of a sample",
                MetricDirection.INFO,
                MetricProperties(is_differentiable=True, is_jit_compatible=True),
                signature=MetricSignature.SINGLE_INPUT,
                domain="statistical",
            ),
        ],
    )
