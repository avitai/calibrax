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
            or ``default`` stream supplies it. It is split into one key per resample.
        num_resamples: Number of bootstrap resamples.
        confidence: Confidence level of the interval, in ``(0, 1)``.
        batch_size: Resamples evaluated together. ``None`` evaluates all of them in one
            ``jax.vmap``, holding every resample's indices and intermediates at once, so its
            memory grows with ``num_resamples``. An integer groups them through
            ``jax.lax.map``: memory then scales with ``batch_size`` times the data's length,
            whatever ``num_resamples`` is. Give one when ``num_resamples`` times the data's
            length is large. Every batch size draws the same resamples; the grouping can move a
            reduction's last bits by a few ULP (the module docstring carries the measurements),
            so re-measure before adopting it under figures pinned exactly.

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
    tail = (1.0 - confidence) / 2.0
    ordered = jnp.sort(samples)
    lower, upper = _percentile(ordered, tail), _percentile(ordered, 1.0 - tail)
    return BootstrapInterval(value=value, lower=lower, upper=upper, samples=samples)
