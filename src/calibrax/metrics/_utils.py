"""Shared utilities for metric functions.

Provides numerical stability guards and validation helpers used across
all metric modules. Private module — not part of the public API.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


_EPSILON = 1e-8
_EPSILON_CLIP = 1e-7


def _validate_shapes(predictions: Any, targets: Any) -> None:
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


def _prepare_arrays(predictions: Any, targets: Any) -> tuple[jax.Array, jax.Array]:  # noqa: DOC502  # raised by _validate_shapes
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


def _prepare_class_arrays(predictions: Any, targets: Any) -> tuple[Any, Any]:  # noqa: DOC502  # raised by _validate_shapes
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
    numerator: Any,
    denominator: Any,
    *,
    eps: float = _EPSILON,
) -> Any:
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
    return numerator / (denominator + eps)


def safe_log(x: Any, *, eps: float = _EPSILON) -> Any:
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
_MIN_ENSEMBLE_MEMBERS = 2
_MULTIVARIATE_ENSEMBLE_NDIM = 3


def _prepare_ensemble_arrays(predictions: Any, targets: Any) -> tuple[jax.Array, jax.Array]:
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
    predictions: Any, targets: Any
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


def matrix_sqrtm(matrix: Any, *, eps: float = _EPSILON) -> jax.Array:
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
