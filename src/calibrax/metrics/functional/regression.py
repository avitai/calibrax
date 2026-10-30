"""Regression metrics for benchmark result comparison.

Pure functions for computing standard regression metrics between predictions
and targets. All functions accept JAX arrays and return scalar values.

Includes 15 metrics: MSE, MAE, RMSE, R-squared, adjusted R-squared, MAPE,
relative error, explained variance, max error, Huber loss, Charbonnier loss,
quantile loss, log-cosh loss, SMAPE, and CRPS.

R-squared and its adjusted form sit together because they are one quantity: both read
:func:`calibrax.metrics._utils.coefficient_of_determination`, and the adjusted form only
applies a degrees-of-freedom penalty on top.

The losses (MSE, MAE, Huber, Charbonnier, relative L2) take keyword-only
``mask``, ``weights``, ``reduction`` and ``axis``, reduced by
``calibrax.metrics._utils.reduce_values``, so a masked or weighted evaluation
means the same thing in every one of them.
"""

from __future__ import annotations

from typing import Final

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from calibrax.metrics._utils import (
    _EPSILON,
    _prepare_arrays,
    _prepare_ensemble_arrays,
    coefficient_of_determination,
    reduce_values,
    safe_divide,
    safe_norm,
    safe_root,
)


#: Default smoothing constant for :func:`charbonnier_loss`, and the loss at zero error.
#: Named rather than inlined because the fused regression suite in
#: ``calibrax.metrics._builtin_registrations`` recomputes this metric at its defaults, and a
#: second literal there would be a default that could drift away from this one unnoticed.
CHARBONNIER_EPSILON: Final = 1e-3

#: Default exponent for :func:`charbonnier_loss`. One gives a smoothed L1.
CHARBONNIER_ALPHA: Final = 1.0

#: Default error magnitude at which :func:`huber_loss` turns from quadratic to linear.
HUBER_DELTA: Final = 1.0

#: Default quantile for :func:`quantile_loss`. One half makes it half the mean absolute
#: error, which is why the fused suite's old inline copy could use a single coefficient for
#: both branches and still agree: at this level, and only at this level, ``q == 1 - q``.
QUANTILE_LEVEL: Final = 0.5


def _huber_elementwise(
    squared_error: jax.Array, absolute_error: jax.Array, delta: float
) -> jax.Array:
    """Per-element Huber loss, from residual pieces the caller already has.

    Quadratic below ``delta`` and linear above it, joined so the value and the first
    derivative are continuous at the knee: ``0.5 * e^2`` against
    ``delta * (|e| - 0.5 * delta)``.

    Exists so that :func:`huber_loss` and the fused regression suite in
    ``calibrax.metrics._builtin_registrations`` share one expression. Both already compute
    the squared and absolute residual for other metrics, so taking them as arguments costs
    the fused path nothing.

    **The linear branch must keep its ``delta`` factor.** Written ``|e| - 0.5`` it is
    correct at ``delta == 1.0`` and wrong everywhere else, which is what the fused suite
    had: a collapsed form that agreed with this function only at the default. A test drives
    this kernel at several deltas, because agreement at the default alone cannot tell a
    shared formula from a coincidence.

    Args:
        squared_error: ``e ** 2``, elementwise.
        absolute_error: ``abs(e)``, elementwise.
        delta: Error magnitude at which the loss turns from quadratic to linear.

    Returns:
        The per-element loss, unreduced.
    """
    return jnp.where(
        absolute_error <= delta,
        0.5 * squared_error,
        delta * (absolute_error - 0.5 * delta),
    )


def _quantile_elementwise(prediction_error: jax.Array, quantile: float) -> jax.Array:
    """Per-element quantile (pinball) loss, in terms of ``predictions - targets``.

    Penalises the two directions of error by ``quantile`` and ``1 - quantile``, so the
    minimiser is the requested quantile of the target distribution rather than its mean.

    Takes ``predictions - targets`` rather than the residual ``targets - predictions``
    because every other metric in the fused suite is written on that difference, and the
    suite binds it once. The two forms are the same loss: with ``r = -pe``,
    ``q * r`` is ``-q * pe`` and ``(q - 1) * r`` is ``(1 - q) * pe``.

    **The two branches must carry different coefficients.** A single ``0.5`` for both is
    correct at ``quantile == 0.5`` and wrong elsewhere, since that is the one level where
    ``q == 1 - q``. A test drives this kernel away from one half for that reason.

    Args:
        prediction_error: ``predictions - targets``, elementwise.
        quantile: Quantile level in ``(0, 1)``.

    Returns:
        The per-element loss, unreduced.
    """
    return jnp.where(
        prediction_error <= 0.0,
        -quantile * prediction_error,
        (1.0 - quantile) * prediction_error,
    )


def mse(  # noqa: DOC502  # raised by _prepare_arrays
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    reduction: str = "mean",
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
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
        mask: Elements to keep (boolean, broadcastable), or ``None`` for all.
        weights: Element weights (broadcastable), or ``None`` for unit weights.
        reduction: ``"none"``, ``"mean"``, ``"sum"`` or ``"batch_sum"``.
        axis: Axis or axes for ``"mean"`` and ``"sum"``.

    Returns:
        Mean squared error as a scalar value, or the reduced values.

    Raises:
        ValueError: If shapes do not match.
    """
    p, t = _prepare_arrays(predictions, targets)
    return reduce_values((p - t) ** 2, mask=mask, weights=weights, reduction=reduction, axis=axis)


def mae(  # noqa: DOC502  # raised by _prepare_arrays
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    reduction: str = "mean",
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
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
        mask: Elements to keep (boolean, broadcastable), or ``None`` for all.
        weights: Element weights (broadcastable), or ``None`` for unit weights.
        reduction: ``"none"``, ``"mean"``, ``"sum"`` or ``"batch_sum"``.
        axis: Axis or axes for ``"mean"`` and ``"sum"``.

    Returns:
        Mean absolute error as a scalar value, or the reduced values.

    Raises:
        ValueError: If shapes do not match.
    """
    p, t = _prepare_arrays(predictions, targets)
    return reduce_values(jnp.abs(p - t), mask=mask, weights=weights, reduction=reduction, axis=axis)


def rmse(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    reduction: str = "mean",
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """Root mean squared error.

    The root of the mean squared error over ``axis`` (every element when ``None``), after the
    mask and weights, as in ``mse``: ``sqrt(sum(w * (p - t)^2) / sum(w))``. ``reduction`` then
    combines the roots left over the remaining axes: ``"none"`` returns them, ``"mean"``
    averages them, ``"sum"`` adds them. Without an ``axis`` there is one root, which every
    reduction returns. The derivative at a perfect prediction is 0, not NaN.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        True metric -- satisfies identity, symmetry, and triangle inequality.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        mask: Elements to keep (boolean, broadcastable), or ``None`` for all.
        weights: Element weights (broadcastable), or ``None`` for unit weights.
        reduction: ``"none"``, ``"mean"`` or ``"sum"`` over the roots.
        axis: Axis or axes the mean under the root runs over.

    Returns:
        Root mean squared error as a scalar value, or the roots reduced.

    Raises:
        ValueError: If shapes do not match, or ``reduction`` is ``"batch_sum"``, which has no
            meaning for a root of a mean.
    """
    if reduction == "batch_sum":
        msg = "rmse takes reduction 'none', 'mean' or 'sum'; 'batch_sum' has no root of a mean"
        raise ValueError(msg)
    roots = safe_root(mse(predictions, targets, mask=mask, weights=weights, axis=axis))
    return reduce_values(roots, reduction=reduction)


def r_squared(  # noqa: DOC502  # raised by _prepare_arrays
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> jax.Array:
    """Coefficient of determination (R-squared).

    Computes ``1 - SS_res / SS_tot`` where SS_res is the residual sum of
    squares and SS_tot is the total sum of squares.

    With ``weights``, both sums are weighted **and the mean inside SS_tot is the weighted
    mean of the targets**. That second part is the one a weighted implementation usually
    gets wrong: weighting the sums while centring on the unweighted mean gives a different
    number, and the result disagrees with every reference implementation. Checked against
    ``sklearn.metrics.r2_score(sample_weight=...)``.

    No ``reduction`` or ``axis``, unlike the error metrics in this module: R-squared is a
    ratio of two reductions over the whole sample, so there is no per-element value for
    ``reduction="none"`` to return and no axis to reduce along that leaves the statistic
    meaning what it means.

    Note:
        Direction: HIGHER (1.0 = perfect fit, 0.0 = mean predictor).
        Range: (-inf, 1].
        Not a true metric -- not a distance function.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        mask: Elements to keep (boolean, broadcastable), or ``None`` for all.
        weights: Element weights (broadcastable), or ``None`` for unit weights.

    Returns:
        R-squared value as a scalar value. Values near 1.0 indicate good fit.

    Raises:
        ValueError: If shapes do not match.
    """
    p, t = _prepare_arrays(predictions, targets)
    return coefficient_of_determination(p, t, mask=mask, weights=weights)


def r_squared_adjusted(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    num_predictors: int,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> jax.Array:
    """Adjusted R-squared.

    ``1 - (1-R^2)(n-1)/(n-p-1)`` where p is number of predictors.
    Penalizes adding predictors that don't improve fit.

    The R-squared term comes from
    :func:`calibrax.metrics._utils.coefficient_of_determination`, the same function
    :func:`calibrax.metrics.functional.regression.r_squared` uses, so the two cannot
    drift apart.

    **A mask changes ``n``; weights do not.** A masked element is not an observation, so
    it leaves the sample and the degrees-of-freedom correction shrinks with it. A weight
    scales an observation that is still there, which is how ``statsmodels`` treats it:
    ``WLS.nobs`` is ``float(self.wexog.shape[0])``, the observation count -- measured on
    statsmodels 0.15.0, a 40-point WLS fit whose weights summed to 25.21 reports
    ``nobs == 40.0``. Taking ``sum(weights)`` as an
    effective sample size instead would make the penalty depend on the scale of the
    weights, so that doubling every weight changed the answer.

    Note:
        Direction: HIGHER (1.0 = perfect fit).
        Range: (-inf, 1].

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        num_predictors: Number of predictors in the model.
        mask: Elements to keep (boolean, broadcastable), or ``None`` for all. Reduces
            ``n``, because an excluded element is not an observation.
        weights: Element weights (broadcastable), or ``None`` for unit weights. Do not
            change ``n``.

    Returns:
        Adjusted R-squared as a scalar value.
    """
    p_arr = jnp.asarray(predictions).ravel()
    t_arr = jnp.asarray(targets).ravel()
    # A Python int when there is no mask, so the usual path costs no reduction; a traced
    # count when there is one, which keeps the function traceable either way.
    n = (
        t_arr.shape[0]
        if mask is None
        else jnp.sum(jnp.broadcast_to(jnp.asarray(mask), t_arr.shape))
    )
    r2 = coefficient_of_determination(p_arr, t_arr, mask=mask, weights=weights)
    return 1.0 - (1.0 - r2) * (n - 1) / (n - num_predictors - 1 + _EPSILON)


def mape(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:  # noqa: DOC502  # raised by _prepare_arrays
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


def relative_error(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:  # noqa: DOC502  # raised by _prepare_arrays
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
    return safe_norm(p - t) / (safe_norm(t) + _EPSILON)


def explained_variance(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:  # noqa: DOC502  # raised by _prepare_arrays
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


def max_error(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:  # noqa: DOC502  # raised by _prepare_arrays
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


def huber_loss(  # noqa: DOC502  # raised by _prepare_arrays
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    delta: float = HUBER_DELTA,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    reduction: str = "mean",
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
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
        mask: Elements to keep (boolean, broadcastable), or ``None`` for all.
        weights: Element weights (broadcastable), or ``None`` for unit weights.
        reduction: ``"none"``, ``"mean"``, ``"sum"`` or ``"batch_sum"``.
        axis: Axis or axes for ``"mean"`` and ``"sum"``.

    Returns:
        Mean Huber loss as a scalar value, or the reduced values.

    Raises:
        ValueError: If shapes do not match.
    """
    p, t = _prepare_arrays(predictions, targets)
    err = p - t
    return reduce_values(
        _huber_elementwise(err**2, jnp.abs(err), delta),
        mask=mask,
        weights=weights,
        reduction=reduction,
        axis=axis,
    )


def charbonnier_loss(  # noqa: DOC502  # raised by _prepare_arrays
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    epsilon: float = CHARBONNIER_EPSILON,
    alpha: float = CHARBONNIER_ALPHA,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    reduction: str = "mean",
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """Charbonnier loss, a differentiable L1: ``(e^2 + epsilon^2)^(alpha / 2)``.

    Note:
        Direction: LOWER (``epsilon^alpha`` = perfect).
        Range: [epsilon^alpha, inf).
        Not a true metric. Smooth at zero error, unlike MAE, so its gradient is
        finite everywhere; the usual choice for image restoration losses.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        epsilon: Smoothing constant; the loss at zero error.
        alpha: Exponent applied to the smoothed absolute error.
        mask: Elements to keep (boolean, broadcastable), or ``None`` for all.
        weights: Element weights (broadcastable), or ``None`` for unit weights.
        reduction: ``"none"``, ``"mean"``, ``"sum"`` or ``"batch_sum"``.
        axis: Axis or axes for ``"mean"`` and ``"sum"``.

    Returns:
        Mean Charbonnier loss as a scalar value, or the reduced values.

    Raises:
        ValueError: If shapes do not match.
    """
    p, t = _prepare_arrays(predictions, targets)
    smoothed = jnp.sqrt((p - t) ** 2 + epsilon**2) ** alpha
    return reduce_values(smoothed, mask=mask, weights=weights, reduction=reduction, axis=axis)


def quantile_loss(  # noqa: DOC502  # raised by _prepare_arrays
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    quantile: float = QUANTILE_LEVEL,
) -> jax.Array:
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
    return jnp.mean(_quantile_elementwise(p - t, quantile))


def log_cosh_loss(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:  # noqa: DOC502  # raised by _prepare_arrays
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


def smape(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:  # noqa: DOC502  # raised by _prepare_arrays
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


def crps(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:  # noqa: DOC502  # raised by _prepare_ensemble_arrays
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
    pred, target = _prepare_ensemble_arrays(predictions, targets)
    forecast_error = jnp.mean(jnp.abs(pred - target[:, None]), axis=1)
    pairwise = jnp.abs(pred[:, :, None] - pred[:, None, :])
    ensemble_spread = 0.5 * jnp.mean(pairwise, axis=(1, 2))
    return jnp.mean(forecast_error - ensemble_spread)


def per_sample_relative_l2(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:  # noqa: DOC502  # raised by _prepare_arrays
    """Relative L2 error of each sample, ``||pred_i - target_i|| / ||target_i||``.

    Each leading-axis sample is flattened before the norms are taken (the PDEBench
    convention for operator learning); a zero-norm target is guarded by the shared
    epsilon.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        Returns one value per sample, so it is not a registered metric; see
        ``relative_l2_error`` for the mean.

    Args:
        predictions: Predicted fields with shape ``(n_samples, *field_dims)``.
        targets: Ground truth fields, same shape.

    Returns:
        Per-sample relative L2 errors with shape ``(n_samples,)``.

    Raises:
        ValueError: If shapes do not match.
    """
    pred, target = _prepare_arrays(predictions, targets)
    batch = pred.shape[0]
    numerator = safe_norm((pred - target).reshape(batch, -1), axis=1)
    denominator = safe_norm(target.reshape(batch, -1), axis=1)
    return safe_divide(numerator, denominator)


def relative_l2_error(  # noqa: DOC502  # raised by reduce_values
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    reduction: str = "mean",
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """Mean over samples of the per-sample relative L2 error.

    Differs from ``relative_error``, which takes one global norm ratio over the
    whole array: this is the batch mean of ``||pred_i - target_i|| / ||target_i||``.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).

    Args:
        predictions: Predicted fields with shape ``(n_samples, *field_dims)``.
        targets: Ground truth fields, same shape.
        mask: Samples to keep (boolean, shape ``(n_samples,)``), or ``None`` for all.
        weights: Per-sample weights, or ``None`` for unit weights.
        reduction: ``"none"``, ``"mean"``, ``"sum"`` or ``"batch_sum"``.
        axis: Axis for ``"mean"`` and ``"sum"``.

    Returns:
        Scalar mean relative L2 error as a JAX array, or the reduced values.

    Raises:
        ValueError: If shapes do not match.
    """
    return reduce_values(
        per_sample_relative_l2(predictions, targets),
        mask=mask,
        weights=weights,
        reduction=reduction,
        axis=axis,
    )
