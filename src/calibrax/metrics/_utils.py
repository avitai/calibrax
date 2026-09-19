"""Shared utilities for metric functions.

Provides numerical stability guards and validation helpers used across
all metric modules. Private module — not part of the public API.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


_EPSILON = 1e-8
_EPSILON_CLIP = 1e-7


def _validate_shapes(predictions: ArrayLike, targets: ArrayLike) -> None:
    """Validate that predictions and targets have matching shapes.

    Args:
        predictions: Predicted values array.
        targets: Ground truth values array.

    Raises:
        ValueError: If shapes do not match.
    """
    p = jnp.asarray(predictions)
    t = jnp.asarray(targets)
    if p.shape != t.shape:
        msg = f"Shape mismatch: predictions {p.shape} vs targets {t.shape}"
        raise ValueError(msg)


def _prepare_arrays(predictions: ArrayLike, targets: ArrayLike) -> tuple[jax.Array, jax.Array]:  # noqa: DOC502  # raised by _validate_shapes
    """Validate shapes and convert predictions/targets to JAX arrays.

    Combines shape validation with array conversion — the standard
    preamble for regression, calibration, and similar metrics.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Tuple of (predictions_array, targets_array) as JAX arrays.

    Raises:
        ValueError: If shapes do not match.
    """
    _validate_shapes(predictions, targets)
    return jnp.asarray(predictions), jnp.asarray(targets)


def _prepare_class_arrays(  # noqa: DOC502  # raised by _validate_shapes
    predictions: ArrayLike, targets: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """Validate shapes and convert to int32 class index arrays.

    Standard preamble for classification, segmentation, and clustering
    metrics that operate on discrete class labels.

    Args:
        predictions: Predicted class labels.
        targets: Ground truth class labels.

    Returns:
        Tuple of (predictions_array, targets_array) as int32 JAX arrays.

    Raises:
        ValueError: If shapes do not match.
    """
    _validate_shapes(predictions, targets)
    return (
        jnp.asarray(predictions).astype(jnp.int32),
        jnp.asarray(targets).astype(jnp.int32),
    )


def safe_divide(
    numerator: ArrayLike,
    denominator: ArrayLike,
    *,
    eps: float = _EPSILON,
) -> jax.Array:
    """Division guarded against zero/near-zero denominators.

    Replaces scattered ``x / (y + _EPSILON)`` patterns with a centralized,
    consistent numerical stability guard. All metric modules should use
    this instead of manual epsilon addition.

    Args:
        numerator: Dividend array or scalar.
        denominator: Divisor array or scalar.
        eps: Small constant added to denominator to prevent division by zero.

    Returns:
        Result of numerator / (denominator + eps).
    """
    return jnp.asarray(numerator) / (jnp.asarray(denominator) + eps)


def safe_root(x: ArrayLike, *, order: float = 2.0) -> jax.Array:
    """The ``order``-th root of non-negative ``x``, with derivative 0 where ``x`` is 0.

    The plain root's derivative is infinite at 0, so ``jax.grad`` of a distance built on it
    is NaN at a perfect match. Following the JAX FAQ ("Gradients contain NaN where using
    where"), the root is taken of a value guarded by an inner ``where`` and the zero case is
    selected by an outer one; the outer ``where`` alone keeps the NaN. The derivative at 0 is
    0, the convention of ``optax.safe_norm``.

    Args:
        x: Non-negative values.
        order: The root's order; 2 is the square root.

    Returns:
        ``x ** (1 / order)`` element-wise.
    """
    x = jnp.asarray(x)
    positive = x > 0.0
    safe = jnp.where(positive, x, jnp.ones_like(x))
    return jnp.where(positive, safe ** (1.0 / order), jnp.zeros_like(x))


def safe_norm(x: ArrayLike, *, axis: int | tuple[int, ...] | None = None) -> jax.Array:
    """The Euclidean norm of ``x`` over ``axis`` (all elements when ``None``), zero-safe.

    ``jnp.linalg.norm`` has a NaN gradient at the zero vector, which a distance reaches at a
    perfect match; this is ``safe_root`` of the sum of squares, with gradient 0 there.

    Args:
        x: Values.
        axis: Axis or axes to reduce.

    Returns:
        The norm, reduced over ``axis``.
    """
    return safe_root(jnp.sum(jnp.square(x), axis=axis))


def safe_log(x: ArrayLike, *, eps: float = _EPSILON) -> jax.Array:
    """Logarithm guarded against zero/negative inputs.

    Clamps input to ``[eps, inf)`` before taking log. Essential for
    divergence, information-theoretic, and calibration metrics that
    compute log of probabilities or ratios.

    Args:
        x: Input array or scalar.
        eps: Minimum value to clamp to before log.

    Returns:
        ``jnp.log(max(x, eps))``.
    """
    return jnp.log(jnp.maximum(x, eps))


# Ensemble forecasts are (samples, members) with at least two members.
_ENSEMBLE_NDIM = 2
# A matrix of records or features: (n_samples, n_features).
_FEATURE_MATRIX_NDIM = 2
# A batch of series: (batch, sequence, features).
_SERIES_NDIM = 3
# Feature correlations need at least two features.
_MIN_CORRELATED_FEATURES = 2
_MIN_ENSEMBLE_MEMBERS = 2
_MULTIVARIATE_ENSEMBLE_NDIM = 3


def _prepare_ensemble_arrays(
    predictions: ArrayLike, targets: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """Validate and convert an ensemble forecast and its targets.

    Args:
        predictions: Forecast ensemble with shape ``(n_samples, n_members)``.
        targets: Observed targets with shape ``(n_samples,)``; a scalar is accepted
            for a single sample.

    Returns:
        Tuple of float32 JAX arrays ``(predictions, targets)``.

    Raises:
        ValueError: If the ensemble is not two-dimensional with at least two
            members, or the targets do not match its sample count.
    """
    pred = jnp.asarray(predictions, dtype=jnp.float32)
    target = jnp.asarray(targets, dtype=jnp.float32)
    if pred.ndim != _ENSEMBLE_NDIM:
        msg = f"predictions must be 2-dimensional, got shape {pred.shape}"
        raise ValueError(msg)
    if pred.shape[1] < _MIN_ENSEMBLE_MEMBERS:
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


def _prepare_multivariate_ensemble_arrays(
    predictions: ArrayLike, targets: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """Validate and convert a multivariate ensemble forecast and its targets.

    Args:
        predictions: Ensemble with shape ``(n_samples, n_members, n_outputs)``.
        targets: Observed targets with shape ``(n_samples, n_outputs)``.

    Returns:
        Tuple of float32 JAX arrays ``(predictions, targets)``.

    Raises:
        ValueError: If the shapes are not ``(n, m, d)`` and ``(n, d)`` with at
            least two members.
    """
    pred = jnp.asarray(predictions, dtype=jnp.float32)
    target = jnp.asarray(targets, dtype=jnp.float32)
    if pred.ndim != _MULTIVARIATE_ENSEMBLE_NDIM:
        msg = f"predictions must be 3-dimensional (samples, members, outputs), got {pred.shape}"
        raise ValueError(msg)
    if pred.shape[1] < _MIN_ENSEMBLE_MEMBERS:
        msg = f"predictions must contain at least two ensemble members, got {pred.shape[1]}"
        raise ValueError(msg)
    if target.shape != (pred.shape[0], pred.shape[2]):
        msg = (
            f"targets must have shape (n_samples, n_outputs) = {(pred.shape[0], pred.shape[2])}, "
            f"got {target.shape}"
        )
        raise ValueError(msg)
    return pred, target


def matrix_sqrtm(matrix: ArrayLike, *, eps: float = _EPSILON) -> jax.Array:
    """Square root of a symmetric positive semi-definite matrix by eigendecomposition.

    Eigenvalues are floored at ``eps`` before the square root, which keeps the
    result real and finite for covariance matrices that are only semi-definite in
    floating point. Stays on backends where ``jax.scipy.linalg.sqrtm`` has no kernel.

    Args:
        matrix: Symmetric positive semi-definite matrix with shape ``(d, d)``.
        eps: Floor applied to the eigenvalues.

    Returns:
        The matrix square root with shape ``(d, d)``.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(jnp.asarray(matrix))
    sqrt_eigenvalues = jnp.sqrt(jnp.maximum(eigenvalues, eps))
    return eigenvectors @ jnp.diag(sqrt_eigenvalues) @ eigenvectors.T


_REDUCTIONS = ("none", "mean", "sum", "batch_sum")


def reduce_values(
    values: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    reduction: str = "mean",
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """Reduce element-wise loss values under an optional mask and weights.

    Every loss in the functional tier reduces through this one function, so a mask,
    weights, a reduction and an axis mean the same thing everywhere:

    - ``mask`` (boolean, broadcastable to ``values``) excludes elements: they contribute
      nothing to a sum and are not counted in a mean;
    - ``weights`` (broadcastable to ``values``) scale elements, and a mean is the weighted
      mean ``sum(w * x) / sum(w)`` over the unmasked elements;
    - ``reduction`` is ``"none"`` (element-wise, masked elements are 0), ``"mean"``,
      ``"sum"`` or ``"batch_sum"`` (sum over the non-batch axes, mean over the leading
      batch axis; ``axis`` is ignored);
    - ``axis`` restricts ``"mean"`` and ``"sum"`` to the given axes.

    A mean over no unmasked element (an all-false mask, or weights that sum to zero) is
    ``0.0``: one documented, finite result, so a caller that treats an empty selection as
    an error can check the mask at the host boundary instead of finding a NaN later.

    Args:
        values: Element-wise loss values.
        mask: Elements to keep, or ``None`` for all of them.
        weights: Element weights, or ``None`` for unit weights.
        reduction: One of ``"none"``, ``"mean"``, ``"sum"``, ``"batch_sum"``.
        axis: Axis or axes for ``"mean"`` and ``"sum"``.

    Returns:
        The reduced value(s).

    Raises:
        ValueError: If ``reduction`` is not one of the supported modes.
    """
    if reduction not in _REDUCTIONS:
        msg = f"Unknown reduction: {reduction!r}. Use one of {_REDUCTIONS}."
        raise ValueError(msg)
    values = jnp.asarray(values)
    if mask is None and weights is None:
        # The plain reductions: no scale array, no guarded division, so a loss without a
        # mask or weights costs what the bare formula costs.
        return _plain_reduction(values, reduction, axis)
    scale = jnp.ones_like(values) if weights is None else jnp.broadcast_to(weights, values.shape)
    if mask is not None:
        scale = jnp.where(jnp.asarray(mask), scale, jnp.zeros_like(scale))
    return _scaled_reduction(values * scale, scale, reduction, axis)


def _plain_reduction(
    values: jax.Array, reduction: str, axis: int | tuple[int, ...] | None
) -> jax.Array:
    """Reduce unscaled values."""
    if reduction == "none":
        return values
    if reduction == "sum":
        return jnp.sum(values, axis=axis)
    if reduction == "batch_sum":
        return _batch_sum(values)
    return jnp.mean(values, axis=axis)


def _scaled_reduction(
    scaled: jax.Array, scale: jax.Array, reduction: str, axis: int | tuple[int, ...] | None
) -> jax.Array:
    """Reduce masked or weighted values; a mean divides by the scale that survived the mask."""
    if reduction == "none":
        return scaled
    if reduction == "sum":
        return jnp.sum(scaled, axis=axis)
    if reduction == "batch_sum":
        return _batch_sum(scaled) if scaled.ndim > 1 else _mean_of_scaled(scaled, scale, axis=None)
    return _mean_of_scaled(scaled, scale, axis=axis)


def _batch_sum(values: jax.Array) -> jax.Array:
    """Sum over the non-batch axes, mean over the leading batch axis."""
    if values.ndim <= 1:
        return jnp.mean(values)
    batch = values.shape[0]
    return jnp.mean(jnp.sum(values.reshape(batch, -1), axis=-1))


def _mean_of_scaled(
    scaled: jax.Array, scale: jax.Array, *, axis: int | tuple[int, ...] | None
) -> jax.Array:
    """``sum(scaled) / sum(scale)`` with ``0.0`` where the scale sums to zero."""
    numerator = jnp.sum(scaled, axis=axis)
    denominator = jnp.sum(scale, axis=axis)
    empty = denominator == 0
    # The guarded operand goes inside the select too, so the gradient of the empty
    # branch is finite rather than the derivative of a division by zero.
    safe_denominator = jnp.where(empty, jnp.ones_like(denominator), denominator)
    return jnp.where(empty, jnp.zeros_like(numerator), numerator / safe_denominator)
