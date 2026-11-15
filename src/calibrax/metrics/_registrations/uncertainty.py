"""Registrations of the built-in ``uncertainty`` metrics."""

from __future__ import annotations

from calibrax.core.models import MetricDirection
from calibrax.metrics._registrations._spec import _BuiltinMetricSpec, _register_specs
from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import MetricProperties, MetricSignature
from calibrax.metrics.functional.uncertainty import (
    anees,
    gaussian_nll,
    interval_score,
    mpiw,
    non_credibility_index,
    picp,
    regression_calibration_error,
)


def register() -> None:
    """Register the scalar uncertainty-quantification metrics.

    Note: predictive_entropy and ensemble_mutual_information are per-sample,
    winkler_score is interval_score under its older name, and
    chi2_confidence_interval returns a pair; none is registered.
    """
    registry = MetricRegistry()
    differentiable = MetricProperties(is_differentiable=True, is_jit_compatible=True)
    counting = MetricProperties(is_differentiable=False, is_jit_compatible=True)
    builtins = [
        _BuiltinMetricSpec(
            "picp",
            picp,
            "Prediction interval coverage probability",
            MetricDirection.INFO,
            counting,
            signature=MetricSignature.CUSTOM,
            domain="uncertainty",
        ),
        _BuiltinMetricSpec(
            "mpiw",
            mpiw,
            "Mean prediction interval width",
            MetricDirection.LOWER,
            differentiable,
            signature=MetricSignature.CUSTOM,
            domain="uncertainty",
        ),
        _BuiltinMetricSpec(
            "interval_score",
            interval_score,
            "Interval (Winkler) score of central prediction intervals",
            MetricDirection.LOWER,
            MetricProperties(is_proper=True, is_differentiable=True, is_jit_compatible=True),
            signature=MetricSignature.CUSTOM,
            domain="uncertainty",
        ),
        _BuiltinMetricSpec(
            "gaussian_nll",
            gaussian_nll,
            "Mean Gaussian negative log-likelihood",
            MetricDirection.LOWER,
            MetricProperties(is_proper=True, is_differentiable=True, is_jit_compatible=True),
            signature=MetricSignature.CUSTOM,
            domain="uncertainty",
        ),
        _BuiltinMetricSpec(
            "regression_calibration_error",
            regression_calibration_error,
            "Quantile calibration error of a Gaussian predictive",
            MetricDirection.LOWER,
            counting,
            signature=MetricSignature.CUSTOM,
            domain="uncertainty",
        ),
        _BuiltinMetricSpec(
            "anees",
            anees,
            "Average normalised estimation error squared",
            MetricDirection.INFO,
            differentiable,
            signature=MetricSignature.CUSTOM,
            domain="uncertainty",
        ),
        _BuiltinMetricSpec(
            "non_credibility_index",
            non_credibility_index,
            "Non-credibility index in decibels",
            MetricDirection.INFO,
            differentiable,
            signature=MetricSignature.CUSTOM,
            domain="uncertainty",
        ),
    ]
    _register_specs(registry, builtins)
