"""Exact host-side decimal rank, screening cutoff, and Farey BH threshold helpers.

Order-statistic ranks in conformal prediction, early-enrichment screening cutoffs,
and discrete Benjamini-Hochberg count thresholds map caller-specified probabilities
(such as ``alpha = 0.1``, ``q = 0.3``, or ``fraction = 0.07``) to integer indices.
Evaluating ``ceil((n + 1) * (1 - alpha))`` or ``ceil(N * fraction)`` in IEEE-754
binary floating point turns exact decimal boundaries into off-by-one integer jumps
whenever a decimal lacks an exact binary mantissa (for example, ``float(0.3) > 3/10``
while ``float32(0.9) < 9/10`` and ``100 * 0.07 > 7``).

This module converts concrete static probability hyperparameters to their shortest
round-trip decimal :class:`fractions.Fraction` on the host and evaluates all integer
ranks and count thresholds via exact integer arithmetic.
"""

from __future__ import annotations

import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from calibrax.core._validation import validate_scalar_numeric


_U32_LIMIT = 1 << 32
_ACCEPTED_FLOAT_SCALARS: dict[np.dtype[np.floating], type[np.floating]] = {
    np.dtype(np.float64): np.float64,
    np.dtype(np.float32): np.float32,
    np.dtype(np.float16): np.float16,
}


def _validate_positive_int(value: int, *, name: str) -> int:
    """Validate that ``value`` is a positive integer ``>= 1``."""
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    out = int(value)
    if out < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return out


def decimal_fraction(value: ArrayLike, *, name: str = "value") -> Fraction:
    """Convert a concrete numeric scalar to its shortest round-trip decimal ``Fraction``.

    Formats accepted IEEE-754 binary floating-point scalars (Python ``float``,
    ``float64``, ``float32``, and ``float16``) with one Dragon4 shortest-unique rule,
    ``np.format_float_positional(x, unique=True, trim="-")``, so ``0.1`` and ``0.3``
    in any accepted dtype map to ``Fraction(1, 10)`` and ``Fraction(3, 10)`` while
    sub-boundary floats such as ``np.nextafter(np.float32(0.05), np.float32(0.0))``
    stay strictly below ``1/20``. Low-mantissa non-IEEE formats such as ``bfloat16``
    are refused with ``TypeError`` asking for a Python ``float``.

    Args:
        value: Concrete static numeric scalar (``int`` or ``float16``/``float32``/``float64``).
        name: Parameter name used in ``TypeError`` and ``ValueError`` diagnostics.

    Returns:
        Exact decimal :class:`fractions.Fraction` representation of ``value``.

    Raises:
        TypeError: If ``value`` is boolean, non-numeric, a JAX tracer, or uses an
            unsupported floating-point dtype such as ``bfloat16``.
        ValueError: If ``value`` is non-finite or not a 0-D scalar.
    """
    eager, _ = validate_scalar_numeric(value, name=name, allow_traced=False)
    if eager is None or not math.isfinite(eager):
        raise ValueError(f"{name} must be finite, got {value}")
    raw_dtype = getattr(value, "dtype", None)
    if isinstance(value, int | np.integer) or (
        raw_dtype is not None and jnp.issubdtype(raw_dtype, jnp.integer)
    ):
        return Fraction(int(eager), 1)
    if isinstance(value, float):
        scalar = np.float64(eager)
    else:
        scalar_cls = (
            _ACCEPTED_FLOAT_SCALARS.get(np.dtype(raw_dtype)) if raw_dtype is not None else None
        )
        if scalar_cls is None:
            raise TypeError(
                f"{name} has unsupported floating-point dtype {raw_dtype}; pass a Python float."
            )
        scalar = scalar_cls(eager)
    return Fraction(np.format_float_positional(scalar, unique=True, trim="-"))


def _open_unit_fraction(value: ArrayLike, *, name: str) -> Fraction:
    """Convert ``value`` via :func:`decimal_fraction` and enforce ``0 < value < 1``."""
    frac = decimal_fraction(value, name=name)
    if not (Fraction(0, 1) < frac < Fraction(1, 1)):
        raise ValueError(f"{name} must be in (0, 1), got {value}")
    return frac


def conformal_quantile_rank(n: int, alpha: ArrayLike) -> int:  # noqa: DOC502
    """Exact 1-based split-conformal order-statistic rank ``ceil((n + 1) * (1 - alpha))``.

    Evaluates ``(n + 1) - floor((n + 1) * decimal_fraction(alpha))`` in exact integer
    arithmetic (Lei et al., *J. Am. Stat. Assoc.* 2018, Thm. 2.1), matching R's
    ``stats::quantile`` relative-fuzz floor convention without IEEE-754 binary drift.

    Args:
        n: Calibration sample count (must be ``>= 1``).
        alpha: Static miscoverage rate in ``(0, 1)``.

    Returns:
        1-based order-statistic rank in ``1 .. n + 1`` (where ``n + 1`` indicates that
        the finite-sample conformal quantile exceeds the empirical maximum and is ``+inf``).

    Raises:
        TypeError: If ``n`` is not an integer or ``alpha`` is boolean, non-numeric, or traced.
        ValueError: If ``n < 1`` or ``alpha`` lies outside ``(0, 1)``.
    """
    n_int = _validate_positive_int(n, name="n")
    q_alpha = _open_unit_fraction(alpha, name="alpha")
    total = n_int + 1
    return total - ((total * q_alpha.numerator) // q_alpha.denominator)


def minimum_conformal_calibration_size(alpha: ArrayLike) -> int:  # noqa: DOC502
    """Smallest calibration size ``n >= 1`` yielding a finite split-conformal quantile.

    Computes ``ceil(1 / decimal_fraction(alpha)) - 1``, the smallest integer ``n >= 1``
    for which ``conformal_quantile_rank(n, alpha) <= n`` (Lei et al., *JASA* 2018).

    Args:
        alpha: Static miscoverage rate in ``(0, 1)``.

    Returns:
        Minimum calibration sample size ``n >= 1``.

    Raises:
        TypeError: If ``alpha`` is boolean, non-numeric, or traced.
        ValueError: If ``alpha`` lies outside ``(0, 1)``.
    """
    q_alpha = _open_unit_fraction(alpha, name="alpha")
    ceil_inv = (q_alpha.denominator + q_alpha.numerator - 1) // q_alpha.numerator
    return max(1, ceil_inv - 1)


def conformal_plus_ranks(n: int, alpha: ArrayLike) -> tuple[int, int]:  # noqa: DOC502
    """Exact 1-based lower and upper order-statistic ranks for Jackknife+ and CV+.

    Returns ``(lo_rank, hi_rank) = (floor((n + 1) * alpha), ceil((n + 1) * (1 - alpha)))``
    for symmetric two-sided predictive intervals (Barber et al., *Ann. Stat.* 2021, Eq. 5),
    where ``lo_rank == 0`` indicates ``-inf`` and ``hi_rank == n + 1`` indicates ``+inf``.

    Args:
        n: Calibration or cross-validation sample size (must be ``>= 1``).
        alpha: Static total miscoverage rate in ``(0, 1)``.

    Returns:
        Tuple ``(lo_rank, hi_rank)`` of 1-based order-statistic indices.

    Raises:
        TypeError: If ``n`` is not an integer or ``alpha`` is boolean, non-numeric, or traced.
        ValueError: If ``n < 1`` or ``alpha`` lies outside ``(0, 1)``.
    """
    n_int = _validate_positive_int(n, name="n")
    q_alpha = _open_unit_fraction(alpha, name="alpha")
    total = n_int + 1
    lo_rank = (total * q_alpha.numerator) // q_alpha.denominator
    return lo_rank, total - lo_rank


def screening_cutoff_count(n_items: int, fraction: ArrayLike) -> int:  # noqa: DOC503
    """Exact top-``k`` screening cutoff ``min(N, max(1, ceil(N * decimal_fraction(fraction))))``.

    Evaluates ``ceil(N * fraction)`` in exact decimal rational arithmetic so exact
    percentage cutoffs such as ``(100, 0.07) -> 7`` and ``(10_000_001, 0.01) -> 100_001``
    avoid IEEE-754 binary mantissa overshoot or ``float32`` precision loss.

    Args:
        n_items: Total number of ranked items ``N >= 1``.
        fraction: Static screening fraction in ``(0, 1]``.

    Returns:
        Integer cutoff count ``k`` in ``1 .. n_items``.

    Raises:
        TypeError: If ``n_items`` is not an integer or ``fraction`` is boolean or traced.
        ValueError: If ``n_items < 1`` or ``fraction`` lies outside ``(0, 1]``.
    """
    n_int = _validate_positive_int(n_items, name="n_items")
    q_frac = decimal_fraction(fraction, name="fraction")
    if not (Fraction(0, 1) < q_frac <= Fraction(1, 1)):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    ceil_prod = (n_int * q_frac.numerator + q_frac.denominator - 1) // q_frac.denominator
    return min(n_int, max(1, ceil_prod))


def farey_floor_rational(alpha: Fraction, max_denominator: int) -> Fraction:  # noqa: DOC503
    """Largest rational ``beta <= alpha`` with denominator ``<= max_denominator`` in ``O(log D)``.

    Computes the lower Farey neighbor ``beta = max_{1 <= d <= D} floor(d * alpha) / d``
    via the continued-fraction expansion of ``alpha`` (Hardy and Wright, *An Introduction
    to the Theory of Numbers*, Ch. 3 and Ch. 10):

    1. Because ``beta <= alpha``, ``floor(d * beta) <= floor(d * alpha)`` for all ``d >= 1``.
    2. By definition of the maximum, ``beta >= floor(d * alpha) / d``, so
       ``d * beta >= floor(d * alpha)`` and hence ``floor(d * beta) >= floor(d * alpha)``
       for every ``1 <= d <= max_denominator``.

    Args:
        alpha: Non-negative exact rational target.
        max_denominator: Positive upper bound ``D >= 1`` on the returned denominator.

    Returns:
        Exact :class:`fractions.Fraction` ``beta <= alpha`` with
        ``beta.denominator <= max_denominator`` and identical integer floors on ``1 .. D``.

    Raises:
        TypeError: If ``max_denominator`` is not an integer.
        ValueError: If ``alpha < 0`` or ``max_denominator < 1``.
    """
    max_den = _validate_positive_int(max_denominator, name="max_denominator")
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    if alpha.denominator <= max_den:
        return alpha

    num, den = alpha.numerator, alpha.denominator
    p_prev2, p_prev1 = 0, 1
    q_prev2, q_prev1 = 1, 0
    step = 0
    while den != 0:
        a_step, rem = divmod(num, den)
        q_curr = a_step * q_prev1 + q_prev2
        if q_curr > max_den:
            if step % 2 == 0:
                k_semi = (max_den - q_prev2) // q_prev1
                return Fraction(k_semi * p_prev1 + p_prev2, k_semi * q_prev1 + q_prev2)
            return Fraction(p_prev1, q_prev1)
        p_curr = a_step * p_prev1 + p_prev2
        p_prev2, p_prev1 = p_prev1, p_curr
        q_prev2, q_prev1 = q_prev1, q_curr
        num, den = den, rem
        step += 1
    return Fraction(p_prev1, q_prev1)


def _u32_mul_floor_div(k_u32: jax.Array, b_num: int, b_den: int, *, max_k: int) -> jax.Array:
    """Overflow-free ``floor((k_u32 * b_num) / b_den)`` in ``uint32`` for ``0 <= k <= max_k``."""
    if b_num == 0:
        return jnp.zeros_like(k_u32, dtype=jnp.int32)
    den_u = jnp.uint32(b_den)
    if max_k * b_num < _U32_LIMIT:
        return ((k_u32 * jnp.uint32(b_num)) // den_u).astype(jnp.int32)

    q_k = k_u32 // den_u
    r_k = k_u32 % den_u
    quot = jnp.zeros_like(k_u32)
    rem = jnp.zeros_like(k_u32)
    for bit_char in bin(b_num)[2:]:
        carry_dbl = rem >= (den_u - rem)
        quot = (quot << jnp.uint32(1)) + carry_dbl.astype(jnp.uint32)
        rem = jnp.where(carry_dbl, rem - (den_u - rem), rem << jnp.uint32(1))
        if bit_char == "1":
            carry_add = rem >= (den_u - r_k)
            quot = quot + q_k + carry_add.astype(jnp.uint32)
            rem = jnp.where(carry_add, rem - (den_u - r_k), rem + r_k)
    return quot.astype(jnp.int32)


def _try_concrete_int(denom: int | jax.Array) -> int | None:
    """Extract a concrete integer ``denom`` on the host, or ``None`` if traced."""
    if isinstance(denom, bool | np.bool_):
        raise TypeError("denom must be an integer, got boolean")
    if isinstance(denom, int | np.integer):
        return int(denom)
    raw = jnp.asarray(denom)
    if raw.ndim != 0 or not jnp.issubdtype(raw.dtype, jnp.integer):
        raise ValueError(f"denom must be a 0-D integer scalar, got shape {raw.shape}, {raw.dtype}")
    try:
        return int(raw)
    except jax.errors.ConcretizationTypeError:
        return None


def discrete_bh_count_thresholds(  # noqa: DOC503
    m: int,
    denom: int | jax.Array,
    *,
    q: ArrayLike,
    max_denominator: int | None = None,
) -> jax.Array:
    """Exact Benjamini-Hochberg integer count thresholds for ``r = 1 .. m``.

    Computes ``floor(r * denom * decimal_fraction(q) / m)`` for conformal p-values
    ``(1 + c_(r)) / denom``:

    - When ``denom`` is concrete (including ``1 + n_cal`` inside ``jax.jit``), evaluates
      all ``m`` integer floors directly in exact host bigints.
    - When ``denom`` is a traced integer bounded by ``1 <= denom <= max_denominator``
      (such as ``1 + n_cal_null`` under ``jax.vmap``), reduces ``decimal_fraction(q) / m``
      via :func:`farey_floor_rational` with bound ``m * max_denominator`` and evaluates
      the exact integer quotient in ``uint32`` without ``int64`` or float rounding.

    Args:
        m: Number of hypotheses ``m >= 1``.
        denom: Positive conformal p-value denominator (concrete ``int`` or traced 0-D ``int32``).
        q: Static target false discovery rate in ``(0, 1)``.
        max_denominator: Upper bound ``D >= 1`` on ``denom`` required when ``denom`` is traced.

    Returns:
        1-D ``int32`` JAX array of shape ``(m,)`` holding maximum admissible ``1 + c_(r)``
        counts for sorted ranks ``r = 1 .. m``.

    Raises:
        TypeError: If ``m`` or ``denom`` is non-integer, or ``q`` is boolean or traced.
        ValueError: If ``m < 1``, ``q`` lies outside ``(0, 1)``, or ``denom`` is traced
            without a positive ``max_denominator``.
    """
    m_int = _validate_positive_int(m, name="m")
    q_frac = _open_unit_fraction(q, name="q")
    d_concrete = _try_concrete_int(denom)
    if d_concrete is not None:
        if d_concrete <= 0:
            return jnp.zeros(m_int, dtype=jnp.int32)
        num = q_frac.numerator
        den = m_int * q_frac.denominator
        return jnp.asarray(
            [(r_idx * d_concrete * num) // den for r_idx in range(1, m_int + 1)],
            dtype=jnp.int32,
        )

    if max_denominator is None:
        raise ValueError("max_denominator is required when denom is a traced JAX array")
    max_den = _validate_positive_int(max_denominator, name="max_denominator")
    max_k = m_int * max_den
    beta = farey_floor_rational(Fraction(q_frac.numerator, m_int * q_frac.denominator), max_k)
    denom_arr = jnp.asarray(denom, dtype=jnp.int32)
    safe_denom = jnp.maximum(denom_arr, 0).astype(jnp.uint32)
    k_u32 = jnp.arange(1, m_int + 1, dtype=jnp.uint32) * safe_denom
    out = _u32_mul_floor_div(k_u32, beta.numerator, beta.denominator, max_k=max_k)
    return jnp.where(denom_arr > 0, out, 0)
