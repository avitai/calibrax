"""Shared eager scalar extraction helpers for parameter validation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


def _is_numeric_dtype(dtype: jax.typing.DTypeLike) -> bool:
    """Return whether ``dtype`` is an integer or floating scalar dtype (excluding bool)."""
    return bool(jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(dtype, jnp.floating))


def _extract_jax_scalar_float(value: jax.Array) -> float | None:
    """Extract a Python ``float`` from a concrete 0-D numeric :class:`jax.Array`."""
    if value.ndim != 0 or not _is_numeric_dtype(value.dtype):
        return None
    try:
        return float(value)
    except jax.errors.ConcretizationTypeError:
        return None


def eager_scalar_float(value: ArrayLike) -> float | None:
    """Extract a Python ``float`` from a concrete numeric scalar or 0-D array.

    Inspects Python scalars, NumPy/``ml_dtypes`` ``np.generic`` scalars, 0-D NumPy
    arrays, and concrete 0-D :class:`jax.Array` values, returning ``None`` for JAX
    tracers (inside ``jax.jit``, ``jax.vmap``, or ``jax.grad``), booleans, and
    non-scalar arrays.

    Args:
        value: Candidate scalar or array-like parameter.

    Returns:
        Extracted Python ``float`` for concrete numeric scalars, or ``None``.
    """
    if isinstance(value, bool | np.bool_):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, np.generic) and _is_numeric_dtype(value.dtype):
        return float(value)
    if isinstance(value, np.ndarray) and value.ndim == 0 and _is_numeric_dtype(value.dtype):
        return float(value.item())
    if isinstance(value, jax.Array):
        return _extract_jax_scalar_float(value)
    return None


def validate_scalar_numeric(
    value: ArrayLike,
    *,
    name: str,
    allow_traced: bool = True,
) -> tuple[float | None, jax.Array]:
    """Validate that ``value`` is a 0-D non-boolean numeric scalar or JAX array.

    Args:
        value: Candidate scalar hyperparameter.
        name: Parameter name used in ``TypeError`` and ``ValueError`` diagnostics.
        allow_traced: Whether traced 0-D :class:`jax.Array` values are permitted.
            When ``False``, passing a JAX tracer raises ``TypeError`` instructing
            the caller to mark ``name`` in ``static_argnames``.

    Returns:
        Tuple ``(eager_float, raw_array)`` where ``eager_float`` is extracted for
        all concrete numeric scalars (including concrete 0-D :class:`jax.Array`
        instances) and ``None`` only for traced arrays when ``allow_traced=True``.

    Raises:
        TypeError: If ``value`` has boolean dtype, is non-numeric, or is traced
            when ``allow_traced=False``.
        ValueError: If ``value`` has ``ndim != 0``.
    """
    if isinstance(value, str | bytes):
        raise TypeError(f"{name} must be a numeric scalar, got {type(value).__name__}")
    raw = jnp.asarray(value)
    if raw.dtype == jnp.bool_ or not _is_numeric_dtype(raw.dtype):
        raise TypeError(f"{name} must be a numeric scalar, got {raw.dtype}")
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a scalar (0-D), got shape {raw.shape}")
    eager = eager_scalar_float(value)
    if eager is None and not allow_traced:
        raise TypeError(
            f"{name} must be a concrete static scalar; when wrapping with jax.jit, "
            f"pass static_argnames=({name!r},)."
        )
    return eager, raw
