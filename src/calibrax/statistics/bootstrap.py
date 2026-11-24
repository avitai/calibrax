"""The percentile bootstrap confidence interval of a statistic.

Resampling with replacement, the statistic on each resample, and the interval between the
``(1 - confidence) / 2`` and ``(1 + confidence) / 2`` quantiles of those values (the percentile
interval, Efron and Tibshirani 1993; ``scipy.stats.bootstrap(method="percentile")``). Several
arrays are resampled together, with the same indices along their first axis, as scipy's
``paired=True``. By default every resample is evaluated in one ``jax.vmap``, so the statistic
runs once as a batched computation instead of once per resample, and the whole function traces
under ``jax.jit``.

**The default's memory is linear in ``num_resamples``; with ``batch_size`` it does not depend on
it.** Each resample draws its own indices from its own key, ``jax.random.split`` and then an
int32 ``jax.random.randint`` inside the mapped function, as jaxgsa's ``resample_keys`` and
``draw_rows`` do, so no ``num_resamples`` by length index array ever exists. With
``batch_size=None`` one ``vmap`` holds every resample's indices, gathers and workspace at once:
XLA's compiled memory analysis of a Spearman correlation over 2,778,200 rows gives 16.1 GB of
temporaries at 100 resamples and 161 GB at 1,000. With ``batch_size`` only one group is live:
1.64 GB at ``batch_size=10`` for the same statistic and rows, at 100, 1,000 or 10,000
resamples. This is the trade scipy's ``bootstrap(batch=...)`` documents as ``O(batch * n)``.
Larger groups run faster: on CPU, at 100 resamples over those rows, ``batch_size=50`` took about
26 s, 10 about 32 s, and 1 about 137 s.

**Grouping is not bit-identical to the default, and the difference is a few ULP.** Each
resample's indices are the same at every batch size, and its *mathematical* value is
independent of the grouping, but XLA's reduction layout is not: a statistic containing
floating-point reductions over the data axis accumulates in an order that depends on the batch
shape. Measured with float32 data at 32 resamples over 400 to 50,000 rows, on CPU and on an RTX
4090, a mean moved at most 1.6 ULP and a mean-centred dot product at most 3 ULP. So a caller
who has pinned figures to the last bit must re-measure rather than assume. A ``batch_size``
above ``num_resamples`` was exact on both backends, because it leaves a single group and no
scan, which also means it saves no memory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx
from jax.scipy.special import ndtr, ndtri
from jax.typing import ArrayLike
from substrax.rng import key_from


DEFAULT_RESAMPLES = 1000
_MIN_BCA_LENGTH = 2
# numpy's interpolation measures from the nearer neighbour: below this fraction from the lower.
_NEARER_LOWER = 0.5


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True, kw_only=True)
class BootstrapInterval:
    """A statistic's value on the data and its percentile bootstrap interval.

    Attributes:
        value: The statistic on the full data.
        lower: The interval's lower bound.
        upper: The interval's upper bound.
        samples: The statistic on each resample, shape ``(num_resamples,)``.
    """

    value: jax.Array
    lower: jax.Array
    upper: jax.Array
    samples: jax.Array


def check_confidence(confidence: float) -> None:
    """Refuse a confidence level outside ``(0, 1)``.

    Args:
        confidence: The interval's confidence level.

    Raises:
        ValueError: If ``confidence`` is outside ``(0, 1)``.
    """
    if not 0.0 < confidence < 1.0:
        msg = f"confidence must lie in (0, 1), got {confidence}"
        raise ValueError(msg)


def _percentile(ordered: jax.Array, q: float) -> jax.Array:
    """The ``q`` quantile of sorted values by linear interpolation, as ``numpy.percentile``.

    numpy's interpolation (``_lerp``) is exact when both neighbours are equal and monotone in
    ``q``; ``jnp.quantile`` weights them as ``low * (1 - w) + high * w``, which gives a constant
    sample's quantile one float32 ulp away from the constant.
    """
    position = q * (ordered.shape[0] - 1)
    below = int(position // 1)
    above = min(below + 1, ordered.shape[0] - 1)
    fraction = position - below
    low, high = ordered[below], ordered[above]
    step = high - low
    return low + step * fraction if fraction < _NEARER_LOWER else high - step * (1.0 - fraction)


def _percentile_traced(ordered: jax.Array, q: jax.Array) -> jax.Array:
    """JAX-traced linear quantile interpolation matching :func:`_percentile`'s ``_lerp`` rule."""
    last = ordered.shape[0] - 1
    safe_q = jnp.clip(q, jnp.asarray(0.0, dtype=q.dtype), jnp.asarray(1.0, dtype=q.dtype))
    position = safe_q * jnp.asarray(float(last), dtype=q.dtype)
    below_f = jnp.floor(position)
    below = below_f.astype(jnp.int32)
    above = jnp.minimum(below + 1, last)
    fraction = position - below_f
    low, high = ordered[below], ordered[above]
    step = high - low
    one = jnp.asarray(1.0, dtype=fraction.dtype)
    return jnp.where(
        fraction < _NEARER_LOWER, low + step * fraction, high - step * (one - fraction)
    )


def _bca_bounds(
    value: jax.Array,
    ordered: jax.Array,
    samples: jax.Array,
    arrays: list[jax.Array],
    statistic: Callable[..., ArrayLike],
    *,
    confidence: float,
    batch_size: int | None,
) -> tuple[jax.Array, jax.Array]:
    """Bias-corrected and accelerated (BCa) bounds (Efron 1987; SciPy ``_bca_interval``)."""
    work_dtype = (
        ordered.dtype
        if jnp.issubdtype(ordered.dtype, jnp.floating)
        else jnp.result_type(ordered.dtype, jnp.float32)
    )
    ordered_f = ordered.astype(work_dtype)
    samples_f = samples.astype(work_dtype)
    val_f = value.astype(work_dtype)

    b_count = samples_f.shape[0]
    nonzeros = jnp.count_nonzero(samples_f < val_f) + jnp.count_nonzero(samples_f <= val_f)
    pct = nonzeros.astype(work_dtype) / jnp.asarray(2.0 * b_count, dtype=work_dtype)
    valid_pct = (pct > 0.0) & (pct < 1.0)
    z0_hat = ndtri(jnp.where(valid_pct, pct, jnp.asarray(0.5, dtype=work_dtype)))

    length = arrays[0].shape[0]
    base = jnp.arange(length - 1, dtype=jnp.int32)

    def one_jackknife(drop_idx: jax.Array) -> jax.Array:
        keep_idx = jnp.where(base < drop_idx, base, base + 1)
        return jnp.asarray(statistic(*(a[keep_idx] for a in arrays)))

    drop_indices = jnp.arange(length, dtype=jnp.int32)
    theta_jack = (
        jax.vmap(one_jackknife)(drop_indices)
        if batch_size is None
        else jax.lax.map(one_jackknife, drop_indices, batch_size=batch_size)
    ).astype(work_dtype)

    u = jnp.asarray(float(length - 1), dtype=work_dtype) * (jnp.mean(theta_jack) - theta_jack)
    u_sq_sum = jnp.sum(u * u)
    valid_jack = u_sq_sum > 0.0
    one = jnp.asarray(1.0, dtype=work_dtype)
    safe_u_sq = jnp.where(valid_jack, u_sq_sum, one)
    a_hat = jnp.sum(u * u * u) / (
        jnp.asarray(6.0, dtype=work_dtype) * safe_u_sq * jnp.sqrt(safe_u_sq)
    )

    tail = (1.0 - confidence) / 2.0
    z_alpha = ndtri(jnp.asarray(tail, dtype=work_dtype))
    num1 = z0_hat + z_alpha
    num2 = z0_hat - z_alpha
    denom1 = one - a_hat * num1
    denom2 = one - a_hat * num2
    valid = (
        valid_pct
        & valid_jack
        & (denom1 > 0.0)
        & (denom2 > 0.0)
        & (~jnp.isnan(val_f))
        & (~jnp.any(jnp.isnan(ordered_f)))
    )
    alpha_1 = ndtr(z0_hat + num1 / jnp.where(denom1 > 0.0, denom1, one))
    alpha_2 = ndtr(z0_hat + num2 / jnp.where(denom2 > 0.0, denom2, one))
    nan_val = jnp.asarray(jnp.nan, dtype=work_dtype)
    return (
        jnp.where(valid, _percentile_traced(ordered_f, alpha_1), nan_val),
        jnp.where(valid, _percentile_traced(ordered_f, alpha_2), nan_val),
    )


def bootstrap_interval(
    statistic: Callable[..., ArrayLike],
    *data: ArrayLike,
    key: jax.Array | nnx.Rngs,
    num_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    batch_size: int | None = None,
    method: Literal["percentile", "bca"] = "percentile",
) -> BootstrapInterval:
    """The percentile or BCa bootstrap confidence interval of ``statistic`` over ``data``.

    Args:
        statistic: A function of the arrays in ``data`` returning a scalar; it must trace under
            ``jax.vmap``.
        *data: Arrays resampled together along their first axis.
        key: The key the resampling indices are drawn from, or an ``nnx.Rngs`` whose ``sample``
            or ``default`` stream supplies it. It is split into one key per resample.
        num_resamples: Number of bootstrap resamples.
        confidence: Confidence level of the interval, in ``(0, 1)``.
        batch_size: Resamples evaluated together. ``None`` evaluates all of them in one
            ``jax.vmap``, holding every resample's indices and intermediates at once, so its
            memory grows with ``num_resamples`` (and as ``O(n^2)`` during the ``n`` leave-one-out
            jackknife evaluations when ``method="bca"``). An integer groups both bootstrap
            resamples and BCa leave-one-out jackknife passes through ``jax.lax.map``: memory then
            scales as ``O(batch_size * n)`` regardless of ``num_resamples`` or ``n``. Give one
            when ``num_resamples`` times the data's length (or ``n^2`` under ``method="bca"``) is
            large. Every batch size draws the same resamples; the grouping can move a reduction's
            last bits by a few ULP (the module docstring carries the measurements), so re-measure
            before adopting it under figures pinned exactly.
        method: Interval construction method: ``"percentile"`` (default) or ``"bca"``
            (Efron's bias-corrected and accelerated interval using leave-one-out jackknife
            acceleration, matching ``scipy.stats.bootstrap(..., method="BCa")``).

    Returns:
        The statistic's value, the interval and the resampled values.

    Raises:
        ValueError: If no array is given, the arrays differ in length, ``num_resamples`` is below
            one, ``batch_size`` is given and below one, ``confidence`` is outside ``(0, 1)``,
            ``method`` is unsupported, or ``method="bca"`` is requested with fewer than two
            observations.
    """
    resample_key = key_from(key, streams=("sample", "default"), context="bootstrap_interval")
    if not data:
        msg = "bootstrap_interval needs at least one array to resample"
        raise ValueError(msg)
    if num_resamples < 1:
        msg = f"num_resamples must be at least 1, got {num_resamples}"
        raise ValueError(msg)
    if batch_size is not None and batch_size < 1:
        msg = f"batch_size must be at least 1 when given, got {batch_size}"
        raise ValueError(msg)
    if method not in ("percentile", "bca"):
        msg = f"method must be 'percentile' or 'bca', got {method!r}"
        raise ValueError(msg)
    check_confidence(confidence)
    arrays = [jnp.asarray(array) for array in data]
    length = arrays[0].shape[0]
    if any(array.shape[0] != length for array in arrays):
        lengths = [array.shape[0] for array in arrays]
        msg = f"arrays resampled together must have the same length, got {lengths}"
        raise ValueError(msg)
    if method == "bca" and length < _MIN_BCA_LENGTH:
        msg = f"BCa bootstrap requires at least 2 observations, got {length}"
        raise ValueError(msg)

    value = jnp.asarray(statistic(*arrays))
    # One key per resample, and each resample draws its own indices inside the mapped function,
    # as jaxgsa's `resample_keys` and `draw_rows` do: no `(num_resamples, length)` index array
    # exists, so a group's draw is all that is live, and a resample's indices depend only on its
    # key, never on the grouping. int32, so x64 neither doubles the draw nor changes its stream.
    resample_keys = jax.random.split(resample_key, num_resamples)

    def one_resample(draw_key: jax.Array) -> jax.Array:
        index = jax.random.randint(draw_key, (length,), 0, length, dtype=jnp.int32)
        return jnp.asarray(statistic(*(a[index] for a in arrays)))

    # `batch_size=None` evaluates every resample in one `jax.vmap`. With a batch size,
    # `jax.lax.map` vmaps one group at a time, so a reduction inside the statistic accumulates
    # over a group rather than over every resample at once: that is the few-ULP difference.
    if batch_size is None:
        samples = jax.vmap(one_resample)(resample_keys)
    else:
        samples = jax.lax.map(one_resample, resample_keys, batch_size=batch_size)
    ordered = jnp.sort(samples)
    if method == "bca":
        lower, upper = _bca_bounds(
            value,
            ordered,
            samples,
            arrays,
            statistic,
            confidence=confidence,
            batch_size=batch_size,
        )
    else:
        tail = (1.0 - confidence) / 2.0
        lower, upper = _percentile(ordered, tail), _percentile(ordered, 1.0 - tail)
    return BootstrapInterval(value=value, lower=lower, upper=upper, samples=samples)
