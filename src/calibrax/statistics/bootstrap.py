"""The percentile bootstrap confidence interval of a statistic.

Resampling with replacement, the statistic on each resample, and the interval between the
``(1 - confidence) / 2`` and ``(1 + confidence) / 2`` quantiles of those values (the percentile
interval, Efron and Tibshirani 1993; ``scipy.stats.bootstrap(method="percentile")``). Several
arrays are resampled together, with the same indices along their first axis, as scipy's
``paired=True``. By default every resample is evaluated in one ``jax.vmap``, so the statistic
runs once as a batched computation instead of once per resample, and the whole function traces
under ``jax.jit``.

**That default costs memory linear in ``num_resamples``, which ``batch_size`` bounds.** A
``vmap`` over the resample axis keeps every resample's gather and workspace live at once: the
index array is ``num_resamples`` by the data's length, and each resampled array matches it, so
at 100 resamples over 2,778,200 rows that is 1.11 GB of indices and 1.11 GB per array before
the statistic's own workspace. ``batch_size`` evaluates the resamples in groups through
``jax.lax.map`` instead, bounding the live set to one group.

**It is not bit-identical to the default, and the difference is a few ULP.** Each resample's
*mathematical* value is independent of the grouping, but XLA's reduction layout is not: a
statistic containing floating-point reductions over the data axis accumulates in an order that
depends on the batch shape. Measured on CPU with float32 samples at 32 resamples over 400
rows, a mean-centred dot product moved **2.9e-06 (about 3 ULP)** at batch sizes 1, 3 and 7, and
Spearman's rank correlation moved **8.9e-08 (about 1.5 ULP)**; the same Spearman at 100
resamples over 50,000 and 500,000 rows moved nothing at all. So whether a difference appears
depends on the statistic, the lengths and the grouping, and a caller who has pinned figures to
the last bit must re-measure rather than assume. A ``batch_size`` above ``num_resamples`` is
the one case measured exact for both statistics, because it leaves a single group and no scan
-- which also means it saves no memory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike
from substrax.rng import key_from


DEFAULT_RESAMPLES = 1000
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


def bootstrap_interval(
    statistic: Callable[..., ArrayLike],
    *data: ArrayLike,
    key: jax.Array | nnx.Rngs,
    num_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    batch_size: int | None = None,
) -> BootstrapInterval:
    """The percentile bootstrap interval of ``statistic`` over ``data``.

    Args:
        statistic: A function of the arrays in ``data`` returning a scalar; it must trace under
            ``jax.vmap``.
        *data: Arrays resampled together along their first axis.
        key: The key the resampling indices are drawn from, or an ``nnx.Rngs`` whose ``sample``
            or ``default`` stream supplies it.
        num_resamples: Number of bootstrap resamples.
        confidence: Confidence level of the interval, in ``(0, 1)``.
        batch_size: Resamples evaluated together. ``None`` evaluates all of them in one
            ``jax.vmap``, which is fastest and holds every resample's intermediates at once;
            an integer groups them through ``jax.lax.map``, bounding the live set to one group.
            Give one when ``num_resamples`` times the data's length is large. The grouping can
            move a reduction's last bits by a few ULP -- the module docstring carries the
            measurements and the arithmetic -- so re-measure before adopting it under figures
            pinned exactly.

    Returns:
        The statistic's value, the interval and the resampled values.

    Raises:
        ValueError: If no array is given, the arrays differ in length, ``num_resamples`` is below
            one, ``batch_size`` is given and below one, or ``confidence`` is outside ``(0, 1)``.
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
    check_confidence(confidence)
    arrays = [jnp.asarray(array) for array in data]
    length = arrays[0].shape[0]
    if any(array.shape[0] != length for array in arrays):
        lengths = [array.shape[0] for array in arrays]
        msg = f"arrays resampled together must have the same length, got {lengths}"
        raise ValueError(msg)

    value = jnp.asarray(statistic(*arrays))
    indices = jax.random.randint(resample_key, (num_resamples, length), 0, length)

    def one_resample(index: jax.Array) -> jax.Array:
        return jnp.asarray(statistic(*(a[index] for a in arrays)))

    # The default path is left exactly as it was, so no existing caller's speed or numbers
    # change; `batch_size` is strictly opt-in. `jax.lax.map` with a batch size vmaps each group
    # in turn, which is where the few-ULP difference comes from: the reduction inside the
    # statistic accumulates over a group rather than over every resample at once.
    if batch_size is None:
        samples = jax.vmap(one_resample)(indices)
    else:
        samples = jax.lax.map(one_resample, indices, batch_size=batch_size)
    tail = (1.0 - confidence) / 2.0
    ordered = jnp.sort(samples)
    lower, upper = _percentile(ordered, tail), _percentile(ordered, 1.0 - tail)
    return BootstrapInterval(value=value, lower=lower, upper=upper, samples=samples)
