"""Shared eager scalar extraction helpers for parameter validation."""

from __future__ import annotations

import jax
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
