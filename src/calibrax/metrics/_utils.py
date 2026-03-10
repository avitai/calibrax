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


def _prepare_arrays(predictions: Any, targets: Any) -> tuple[jax.Array, jax.Array]:
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


def _prepare_class_arrays(predictions: Any, targets: Any) -> tuple[Any, Any]:
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
