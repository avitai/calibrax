"""Registrations of the built-in ``regression`` metrics."""

from __future__ import annotations

from calibrax.core.models import MetricDirection
from calibrax.metrics._registrations._spec import _BuiltinMetricSpec
from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import MetricEntry, MetricProperties, MetricSignature, MetricTier
from calibrax.metrics.functional.regression import (
    charbonnier_loss,
    crps,
    explained_variance,
    huber_loss,
    log_cosh_loss,
    mae,
    mape,
    max_error,
    mean_error,
    mse,
    quantile_loss,
    r_squared,
    r_squared_adjusted,
    relative_error,
    relative_l2_error,
    rmse,
    smape,
)


def register() -> None:
    """Register all 14 regression metrics at import time."""
    registry = MetricRegistry()
    builtins = [
        _BuiltinMetricSpec(
            "mse",
            mse,
            "Mean squared error",
            MetricDirection.LOWER,
            MetricProperties(is_symmetric=True, is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "mae",
            mae,
            "Mean absolute error",
            MetricDirection.LOWER,
            MetricProperties(is_symmetric=True, is_true_metric=True, is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "rmse",
            rmse,
            "Root mean squared error",
            MetricDirection.LOWER,
            MetricProperties(is_symmetric=True, is_true_metric=True, is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "mean_error",
            mean_error,
            "Mean signed prediction error (bias)",
            MetricDirection.INFO,
            MetricProperties(is_differentiable=True, is_jit_compatible=True),
        ),
        _BuiltinMetricSpec(
            "r_squared",
            r_squared,
            "Coefficient of determination",
            MetricDirection.HIGHER,
            MetricProperties(is_differentiable=True),
        ),
        # Beside r_squared because it is the same quantity with a degrees-of-freedom
        # penalty, and both read `coefficient_of_determination`. Its CUSTOM signature is
        # load-bearing: `num_predictors` is a required keyword, so this cannot be called as
        # `fn(predictions, targets)`, and the CUSTOM declaration is what keeps it out of
        # `calculate_all`'s default set, which dispatches exactly that way.
        _BuiltinMetricSpec(
            "r_squared_adjusted",
            r_squared_adjusted,
            "Adjusted R-squared",
            MetricDirection.HIGHER,
            MetricProperties(),
            signature=MetricSignature.CUSTOM,
        ),
        _BuiltinMetricSpec(
            "mape",
            mape,
            "Mean absolute percentage error",
            MetricDirection.LOWER,
            MetricProperties(is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "relative_error",
            relative_error,
            "Mean relative error (L2 norm ratio)",
            MetricDirection.LOWER,
            MetricProperties(is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "relative_l2_error",
            relative_l2_error,
            "Mean per-sample relative L2 error",
            MetricDirection.LOWER,
            MetricProperties(is_differentiable=True, is_jit_compatible=True),
        ),
        _BuiltinMetricSpec(
            "explained_variance",
            explained_variance,
            "Explained variance score",
            MetricDirection.HIGHER,
            MetricProperties(is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "max_error",
            max_error,
            "Maximum absolute error",
            MetricDirection.LOWER,
            MetricProperties(is_symmetric=True, is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "huber_loss",
            huber_loss,
            "Huber loss (robust regression)",
            MetricDirection.LOWER,
            MetricProperties(is_symmetric=True, is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "charbonnier_loss",
            charbonnier_loss,
            "Charbonnier loss (differentiable L1)",
            MetricDirection.LOWER,
            MetricProperties(is_symmetric=True, is_differentiable=True, is_jit_compatible=True),
        ),
        _BuiltinMetricSpec(
            "quantile_loss",
            quantile_loss,
            "Quantile (pinball) loss",
            MetricDirection.LOWER,
            MetricProperties(is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "log_cosh_loss",
            log_cosh_loss,
            "Log-cosh loss",
            MetricDirection.LOWER,
            MetricProperties(is_symmetric=True, is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "smape",
            smape,
            "Symmetric mean absolute percentage error",
            MetricDirection.LOWER,
            MetricProperties(is_symmetric=True, is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "crps",
            crps,
            "Continuous ranked probability score for ensemble forecasts",
            MetricDirection.LOWER,
            MetricProperties(is_proper=True, is_differentiable=True, is_jit_compatible=True),
            signature=MetricSignature.ENSEMBLE_PREDICTIONS_TARGETS,
        ),
    ]
    for spec in builtins:
        if not registry.has(spec.name):
            entry = MetricEntry(
                name=spec.name,
                fn=spec.fn,
                tier=MetricTier.PURE_FUNCTION,
                domain=spec.domain,
                direction=spec.direction,
                description=spec.description,
                signature=spec.signature,
                properties=spec.properties,
            )
            registry.register(spec.name, entry)
