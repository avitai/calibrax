"""The percentile bootstrap confidence interval of a statistic.

Resampling with replacement, the statistic on each resample, and the interval between the
``(1 - confidence) / 2`` and ``(1 + confidence) / 2`` quantiles of those values (the percentile
interval, Efron and Tibshirani 1993; ``scipy.stats.bootstrap(method="percentile")``). Several
arrays are resampled together, with the same indices along their first axis, as scipy's
``paired=True``. Every resample is evaluated in one ``jax.vmap``, so the statistic runs once as
a batched computation instead of once per resample, and the whole function traces under
``jax.jit``.
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

    Returns:
        The statistic's value, the interval and the resampled values.

    Raises:
        ValueError: If no array is given, the arrays differ in length, ``num_resamples`` is below
            one or ``confidence`` is outside ``(0, 1)``.
    """
    resample_key = key_from(key, streams=("sample", "default"), context="bootstrap_interval")
    if not data:
        msg = "bootstrap_interval needs at least one array to resample"
        raise ValueError(msg)
    if num_resamples < 1:
        msg = f"num_resamples must be at least 1, got {num_resamples}"
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
    samples = jax.vmap(lambda index: jnp.asarray(statistic(*(a[index] for a in arrays))))(indices)
    tail = (1.0 - confidence) / 2.0
    ordered = jnp.sort(samples)
    lower, upper = _percentile(ordered, tail), _percentile(ordered, 1.0 - tail)
    return BootstrapInterval(value=value, lower=lower, upper=upper, samples=samples)
