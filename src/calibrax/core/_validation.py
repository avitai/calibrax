"""Shared eager scalar extraction helpers for parameter validation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


def eager_scalar_float(value: ArrayLike) -> float | None:
    """Extract a Python ``float`` from a concrete numeric scalar or 0-D NumPy array.

    Returns ``None`` for :class:`jax.Array` instances (including traced arrays
    under ``jax.jit``, ``jax.vmap``, or ``jax.grad``), booleans, and non-scalar
    arrays so callers can validate eager Python and NumPy scalars without
    breaking JAX transformations.

    Args:
        value: Candidate scalar or array-like parameter.

    Returns:
        Extracted Python ``float`` for eager numeric scalars, or ``None``.
    """
    if isinstance(value, jax.Array | bool | np.bool_):
        return None
    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)
    if (
        isinstance(value, np.ndarray)
        and value.ndim == 0
        and (np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.floating))
    ):
        return float(value.item())
    return None


def validate_scalar_numeric(value: ArrayLike, *, name: str) -> tuple[float | None, jax.Array]:
    """Validate that ``value`` is a 0-D non-boolean numeric scalar or JAX array.

    Args:
        value: Candidate scalar hyperparameter.
        name: Parameter name used in ``ValueError`` diagnostics.

    Returns:
        Tuple ``(eager_float, raw_array)`` where ``eager_float`` is ``None`` for
        :class:`jax.Array` instances (including tracers) and ``raw_array`` is the
        0-D :class:`jax.Array` representation.

    Raises:
        TypeError: If ``value`` has boolean dtype.
        ValueError: If ``value`` has ``ndim != 0``.
    """
    raw = jnp.asarray(value)
    if raw.dtype == jnp.bool_:
        raise TypeError(f"{name} must be a numeric scalar, got boolean")
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a scalar (0-D), got shape {raw.shape}")
    return eager_scalar_float(value), raw
