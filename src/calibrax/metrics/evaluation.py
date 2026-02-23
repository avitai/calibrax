"""JAX-native evaluation metrics for benchmark result comparison.

Pure functions for computing standard regression metrics between predictions
and targets. All functions accept JAX arrays and return Python floats
(wrapped for JAX scalar safety).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax.numpy as jnp


_EPSILON = 1e-8


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


def mse(predictions: Any, targets: Any) -> float:
    """Mean squared error.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Mean squared error as a Python float.

    Raises:
        ValueError: If shapes do not match.
    """
    _validate_shapes(predictions, targets)
    p = jnp.asarray(predictions)
    t = jnp.asarray(targets)
    return float(jnp.mean((p - t) ** 2))


def mae(predictions: Any, targets: Any) -> float:
    """Mean absolute error.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Mean absolute error as a Python float.

    Raises:
        ValueError: If shapes do not match.
    """
    _validate_shapes(predictions, targets)
    p = jnp.asarray(predictions)
    t = jnp.asarray(targets)
    return float(jnp.mean(jnp.abs(p - t)))


def rmse(predictions: Any, targets: Any) -> float:
    """Root mean squared error.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Root mean squared error as a Python float.

    Raises:
        ValueError: If shapes do not match.
    """
    return float(jnp.sqrt(mse(predictions, targets)))


def r_squared(predictions: Any, targets: Any) -> float:
    """Coefficient of determination (R-squared).

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        R-squared value as a Python float. Values near 1.0 indicate good fit.

    Raises:
        ValueError: If shapes do not match.
    """
    _validate_shapes(predictions, targets)
    p = jnp.asarray(predictions)
    t = jnp.asarray(targets)
    ss_res = jnp.sum((t - p) ** 2)
    ss_tot = jnp.sum((t - jnp.mean(t)) ** 2)
    return float(1.0 - ss_res / (ss_tot + _EPSILON))


def mape(predictions: Any, targets: Any) -> float:
    """Mean absolute percentage error.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        MAPE as a Python float (0.0 = perfect, 1.0 = 100% error).

    Raises:
        ValueError: If shapes do not match.
    """
    _validate_shapes(predictions, targets)
    p = jnp.asarray(predictions)
    t = jnp.asarray(targets)
    return float(jnp.mean(jnp.abs((t - p) / (jnp.abs(t) + _EPSILON))))


def relative_error(predictions: Any, targets: Any) -> float:
    """Mean relative error (L2 norm ratio).

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Relative error as a Python float.

    Raises:
        ValueError: If shapes do not match.
    """
    _validate_shapes(predictions, targets)
    p = jnp.asarray(predictions)
    t = jnp.asarray(targets)
    return float(jnp.sqrt(jnp.sum((p - t) ** 2)) / (jnp.sqrt(jnp.sum(t**2)) + _EPSILON))


METRIC_FUNCTIONS: dict[str, Callable[..., float]] = {
    "mse": mse,
    "mae": mae,
    "rmse": rmse,
    "r_squared": r_squared,
    "mape": mape,
    "relative_error": relative_error,
}


def calculate_all(
    predictions: Any,
    targets: Any,
    *,
    metrics: Sequence[str] | None = None,
) -> dict[str, float]:
    """Calculate multiple evaluation metrics at once.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        metrics: Subset of metric names to compute. Defaults to all available.

    Returns:
        Dictionary mapping metric names to computed values.

    Raises:
        ValueError: If shapes do not match or an unknown metric is requested.
    """
    names = list(metrics) if metrics is not None else list(METRIC_FUNCTIONS)
    results: dict[str, float] = {}
    for name in names:
        fn = METRIC_FUNCTIONS.get(name)
        if fn is None:
            msg = f"Unknown metric: {name!r}. Available: {sorted(METRIC_FUNCTIONS)}"
            raise ValueError(msg)
        results[name] = fn(predictions, targets)
    return results
