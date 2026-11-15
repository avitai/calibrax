"""Registrations of the built-in ``forecasting`` metrics."""

from __future__ import annotations

from calibrax.core.models import MetricDirection
from calibrax.metrics._registrations._spec import _BuiltinMetricSpec, _register_specs
from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import MetricProperties, MetricSignature
from calibrax.metrics.functional.forecasting import (
    energy_score,
    ensemble_ranked_probability_score,
    event_reliability,
    fair_crps,
    ranked_probability_score,
    spread_skill_ratio,
)


def register() -> None:
    """Register the scalar forecast verification scores.

    Note: rank_histogram, pit_histogram and ranked_probability_skill_score are
    NOT registered (they return arrays, not scalars).
    """
    registry = MetricRegistry()
    proper = MetricProperties(is_proper=True, is_differentiable=True, is_jit_compatible=True)
    builtins = [
        _BuiltinMetricSpec(
            "fair_crps",
            fair_crps,
            "Fair (finite-ensemble bias-corrected) CRPS",
            MetricDirection.LOWER,
            proper,
            signature=MetricSignature.ENSEMBLE_PREDICTIONS_TARGETS,
            domain="forecasting",
        ),
        _BuiltinMetricSpec(
            "energy_score",
            energy_score,
            "Energy score for multivariate ensemble forecasts",
            MetricDirection.LOWER,
            proper,
            signature=MetricSignature.ENSEMBLE_PREDICTIONS_TARGETS,
            domain="forecasting",
        ),
        _BuiltinMetricSpec(
            "spread_skill_ratio",
            spread_skill_ratio,
            "Unbiased ensemble spread over ensemble-mean RMSE",
            MetricDirection.INFO,
            MetricProperties(is_differentiable=True, is_jit_compatible=True),
            signature=MetricSignature.ENSEMBLE_PREDICTIONS_TARGETS,
            domain="forecasting",
        ),
        _BuiltinMetricSpec(
            "ranked_probability_score",
            ranked_probability_score,
            "Ranked probability score for ordered categories",
            MetricDirection.LOWER,
            proper,
            signature=MetricSignature.PREDICTIONS_TARGETS,
            domain="forecasting",
        ),
        _BuiltinMetricSpec(
            "event_reliability",
            event_reliability,
            "Reliability component of the Brier decomposition",
            MetricDirection.LOWER,
            MetricProperties(is_differentiable=True, is_jit_compatible=True),
            signature=MetricSignature.PREDICTIONS_TARGETS,
            domain="forecasting",
        ),
        _BuiltinMetricSpec(
            "ensemble_ranked_probability_score",
            ensemble_ranked_probability_score,
            "Fair ranked probability score of a continuous ensemble at thresholds",
            MetricDirection.LOWER,
            MetricProperties(is_proper=True, is_differentiable=False, is_jit_compatible=True),
            signature=MetricSignature.CUSTOM,
            domain="forecasting",
        ),
    ]
    _register_specs(registry, builtins)
