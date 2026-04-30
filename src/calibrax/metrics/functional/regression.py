"""Regression metrics for benchmark result comparison.

Pure functions for computing standard regression metrics between predictions
and targets. All functions accept JAX arrays and return scalar values.

Includes 13 metrics: MSE, MAE, RMSE, R-squared, MAPE, relative error,
explained variance, max error, Huber loss, quantile loss, log-cosh loss,
SMAPE, and CRPS.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON, _prepare_arrays, safe_divide


def _prepare_ensemble_forecast_arrays(predictions: Any, targets: Any) -> tuple[Any, Any]:
    """Prepare ensemble forecast arrays for probabilistic regression metrics."""
    pred = jnp.asarray(predictions, dtype=jnp.float32)
    target = jnp.asarray(targets, dtype=jnp.float32)

    if pred.ndim != 2:
        msg = f"predictions must be 2-dimensional, got shape {pred.shape}"
        raise ValueError(msg)
    if pred.shape[1] < 2:
        msg = f"predictions must contain at least two ensemble members, got {pred.shape[1]}"
        raise ValueError(msg)
    if target.ndim == 0:
        target = target[None]
    if target.ndim != 1:
        msg = f"targets must be scalar or 1-dimensional, got shape {target.shape}"
        raise ValueError(msg)
    if pred.shape[0] != target.shape[0]:
        msg = (
            "predictions and targets must have matching sample count: "
            f"{pred.shape[0]} != {target.shape[0]}"
        )
        raise ValueError(msg)

    return pred, target


def mse(predictions: Any, targets: Any) -> Any:
    """Mean squared error.

    Computes the average of squared differences between predictions and
    targets: ``mean((predictions - targets)^2)``.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        Not a true metric -- violates the triangle inequality.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Mean squared error as a scalar value.

    Raises:
        ValueError: If shapes do not match.
    """
    p, t = _prepare_arrays(predictions, targets)
    return jnp.mean((p - t) ** 2)


def mae(predictions: Any, targets: Any) -> Any:
    """Mean absolute error.

    Computes the average of absolute differences between predictions and
    targets: ``mean(|predictions - targets|)``.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        True metric -- satisfies identity, symmetry, and triangle inequality.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Mean absolute error as a scalar value.

    Raises:
        ValueError: If shapes do not match.
    """
    p, t = _prepare_arrays(predictions, targets)
    return jnp.mean(jnp.abs(p - t))


def rmse(predictions: Any, targets: Any) -> Any:
    """Root mean squared error.

    Computes ``sqrt(mean((predictions - targets)^2))``.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        True metric -- satisfies identity, symmetry, and triangle inequality.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Root mean squared error as a scalar value.

    Raises:
        ValueError: If shapes do not match.
    """
    return jnp.sqrt(mse(predictions, targets))


def r_squared(predictions: Any, targets: Any) -> Any:
    """Coefficient of determination (R-squared).

    Computes ``1 - SS_res / SS_tot`` where SS_res is the residual sum of
    squares and SS_tot is the total sum of squares.

    Note:
        Direction: HIGHER (1.0 = perfect fit, 0.0 = mean predictor).
        Range: (-inf, 1].
        Not a true metric -- not a distance function.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        R-squared value as a scalar value. Values near 1.0 indicate good fit.

    Raises:
        ValueError: If shapes do not match.
    """
    p, t = _prepare_arrays(predictions, targets)
    ss_res = jnp.sum((t - p) ** 2)
    ss_tot = jnp.sum((t - jnp.mean(t)) ** 2)
    return 1.0 - ss_res / (ss_tot + _EPSILON)


def mape(predictions: Any, targets: Any) -> Any:
    """Mean absolute percentage error.

    Computes ``mean(|targets - predictions| / |targets|)``.

    Note:
        Direction: LOWER (0.0 = perfect, 1.0 = 100% error).
        Range: [0, inf).
        Not a true metric -- not symmetric in general and division by
        target magnitude breaks metric space axioms.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        MAPE as a scalar value (0.0 = perfect, 1.0 = 100% error).

    Raises:
        ValueError: If shapes do not match.
    """
    p, t = _prepare_arrays(predictions, targets)
    return jnp.mean(jnp.abs((t - p) / (jnp.abs(t) + _EPSILON)))


def relative_error(predictions: Any, targets: Any) -> Any:
    """Mean relative error (L2 norm ratio).

    Computes ``||predictions - targets||_2 / ||targets||_2``.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        Not a true metric -- normalization by target norm breaks
        the triangle inequality.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Relative error as a scalar value.

    Raises:
        ValueError: If shapes do not match.
    """
    p, t = _prepare_arrays(predictions, targets)
    return jnp.sqrt(jnp.sum((p - t) ** 2)) / (jnp.sqrt(jnp.sum(t**2)) + _EPSILON)


def explained_variance(predictions: Any, targets: Any) -> Any:
    """Explained variance score.

    Computes ``1 - Var(targets - predictions) / Var(targets)``. Similar to
    R-squared but uses variance of residuals instead of sum of squares.

    Note:
        Direction: HIGHER (1.0 = perfect).
        Range: (-inf, 1].
        Not a true metric -- not a distance function.
        Unlike R-squared, explained variance is invariant to constant bias
        in predictions.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Explained variance as a scalar value.

    Raises:
        ValueError: If shapes do not match.
    """
    p, t = _prepare_arrays(predictions, targets)
    residual_var = jnp.var(t - p)
    target_var = jnp.var(t)
    return 1.0 - residual_var / (target_var + _EPSILON)


def max_error(predictions: Any, targets: Any) -> Any:
    """Maximum absolute error.

    Computes ``max(|prediction_i - target_i|)`` -- the worst-case error.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        Not a true metric on distributions (operates on individual errors,
        not on distribution distance).

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Maximum absolute error as a scalar value.

    Raises:
        ValueError: If shapes do not match.
    """
    p, t = _prepare_arrays(predictions, targets)
    return jnp.max(jnp.abs(p - t))


def huber_loss(
    predictions: Any,
    targets: Any,
    *,
    delta: float = 1.0,
) -> Any:
    """Huber loss (robust regression loss).

    Quadratic for small errors (``|e| <= delta``), linear for large errors.
    ``L = 0.5 * e^2`` if ``|e| <= delta``, else ``delta * (|e| - 0.5 * delta)``.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        Not a true metric.
        Degrades to 0.5 * MSE for small errors, to delta * MAE for large
        errors. More robust to outliers than MSE.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        delta: Threshold where loss transitions from quadratic to linear.

    Returns:
        Mean Huber loss as a scalar value.

    Raises:
        ValueError: If shapes do not match.
    """
    p, t = _prepare_arrays(predictions, targets)
    err = p - t
    abs_err = jnp.abs(err)
    quadratic = 0.5 * err**2
    linear = delta * (abs_err - 0.5 * delta)
    return jnp.mean(jnp.where(abs_err <= delta, quadratic, linear))


def quantile_loss(
    predictions: Any,
    targets: Any,
    *,
    quantile: float = 0.5,
) -> Any:
    """Quantile (pinball) loss for quantile regression.

    Asymmetric loss: ``L = q * max(0, t - p) + (1 - q) * max(0, p - t)``.
    At ``quantile=0.5``, equivalent to ``0.5 * MAE``.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        Not a true metric -- asymmetric by design.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        quantile: Target quantile in (0, 1). 0.5 = median.

    Returns:
        Mean quantile loss as a scalar value.

    Raises:
        ValueError: If shapes do not match.
    """
    p, t = _prepare_arrays(predictions, targets)
    diff = t - p
    return jnp.mean(jnp.where(diff >= 0, quantile * diff, (quantile - 1.0) * diff))


def log_cosh_loss(predictions: Any, targets: Any) -> Any:
    """Log-cosh loss.

    Computes ``mean(log(cosh(predictions - targets)))``. Smooth approximation
    to MAE: approximately quadratic for small errors, linear for large errors.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        Not a true metric.
        Twice differentiable everywhere (unlike MAE), making it suitable
        for second-order optimization methods.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Mean log-cosh loss as a scalar value.

    Raises:
        ValueError: If shapes do not match.
    """
    p, t = _prepare_arrays(predictions, targets)
    err = p - t
    # Numerically stable: log(cosh(x)) = logaddexp(x, -x) - log(2)
    return jnp.mean(jnp.logaddexp(err, -err) - jnp.log(2.0))


def smape(predictions: Any, targets: Any) -> Any:
    """Symmetric mean absolute percentage error.

    Computes ``mean(|p - t| / ((|p| + |t|) / 2))``. Unlike MAPE, SMAPE is
    symmetric: ``SMAPE(p, t) = SMAPE(t, p)``.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, 2].
        Not a true metric -- normalization breaks metric space axioms.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        SMAPE as a scalar value.

    Raises:
        ValueError: If shapes do not match.
    """
    p, t = _prepare_arrays(predictions, targets)
    numerator = jnp.abs(p - t)
    denominator = (jnp.abs(p) + jnp.abs(t)) / 2.0
    return jnp.mean(safe_divide(numerator, denominator))


def crps(predictions: Any, targets: Any) -> Any:
    """Continuous ranked probability score for ensemble forecasts.

    Computes the empirical ensemble CRPS:
    ``mean(|X - y|) - 0.5 * mean(|X_i - X_j|)`` averaged over samples.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        Proper scoring rule for probabilistic forecasts.

    Args:
        predictions: Forecast ensemble with shape ``(n_samples, n_members)``.
        targets: Observed targets with shape ``(n_samples,)``. A scalar target
            is accepted for a single forecast sample.

    Returns:
        Mean empirical CRPS as a scalar JAX array.

    Raises:
        ValueError: If inputs do not have compatible ensemble forecast shapes.
    """
    pred, target = _prepare_ensemble_forecast_arrays(predictions, targets)
    forecast_error = jnp.mean(jnp.abs(pred - target[:, None]), axis=1)
    pairwise = jnp.abs(pred[:, :, None] - pred[:, None, :])
    ensemble_spread = 0.5 * jnp.mean(pairwise, axis=(1, 2))
    return jnp.mean(forecast_error - ensemble_spread)
