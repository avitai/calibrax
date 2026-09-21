"""Summary statistics of a sample, and the observations that stand apart from it.

:func:`summarize` describes a sample — its centre, its spread and whether that spread is small
enough to call the measurement stable — and :func:`outlier_mask` marks the observations a
median absolute deviation puts far from the centre (the modified Z-score of Iglewicz and
Hoaglin 1993). Both are ordinary JAX functions over arrays: they trace under ``jax.jit``,
map under ``jax.vmap`` and differentiate under ``jax.grad``, and :class:`SampleSummary` is a
pytree, so a summary crosses a transform boundary like any other value.

Neither draws a random number. An interval around these statistics comes from
:func:`calibrax.statistics.bootstrap_interval`, which takes the key it resamples with.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


type Samples = ArrayLike | Sequence[float]
"""Measurements, as an array or as the sequence a timer collects them into.

``jax.typing.ArrayLike`` deliberately excludes arbitrary sequences, and a benchmark's samples
are usually a list of Python floats, so both are named here.
"""


# Coefficient of variation threshold for measurement stability.
# CV < this value means "stable" measurement (low noise).
STABILITY_CV_THRESHOLD: float = 0.10

# Modified Z-score threshold for outlier detection (Iglewicz & Hoaglin).
OUTLIER_Z_THRESHOLD: float = 3.5

# MAD consistency constant: 1 / inverse_normal_cdf(3/4) ~ 0.6745.
# Scales MAD to be a consistent estimator of sigma for normal distributions.
_MAD_CONSISTENCY_CONSTANT: float = 0.6745

# A median absolute deviation from fewer than three samples says nothing about outliers.
_MIN_SAMPLES_FOR_MAD = 3


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True, kw_only=True)
class SampleSummary:
    """What a sample of measurements looks like.

    Attributes:
        mean: Arithmetic mean.
        median: Median value.
        std: Sample standard deviation (ddof=1), zero for a single observation.
        minimum: Smallest value.
        maximum: Largest value.
        cv: Coefficient of variation, the standard deviation over the mean; zero when the
            mean is zero, where the ratio says nothing.
        is_stable: Whether ``cv`` is below :data:`STABILITY_CV_THRESHOLD`.
    """

    mean: jax.Array
    median: jax.Array
    std: jax.Array
    minimum: jax.Array
    maximum: jax.Array
    cv: jax.Array
    is_stable: jax.Array


def summarize(samples: Samples) -> SampleSummary:
    """Describe a sample: its centre, its spread, and whether the spread is small.

    Args:
        samples: The measurements, along the array's only axis.

    Returns:
        The summary.
    """
    values = jnp.asarray(samples)
    mean = jnp.mean(values)
    # ddof=1 needs a second observation; with one, the sample has no spread to estimate.
    std = jnp.std(values, ddof=1) if values.shape[-1] > 1 else jnp.zeros_like(mean)
    # The ratio is undefined at a zero mean. Both branches are evaluated under a transform,
    # so the divisor is made safe before the division rather than after: dividing by zero
    # first would leave a NaN in the gradient that selecting the other branch cannot remove.
    is_centred_at_zero = mean == 0
    cv = jnp.where(is_centred_at_zero, 0.0, std / jnp.where(is_centred_at_zero, 1.0, mean))
    return SampleSummary(
        mean=mean,
        median=jnp.median(values),
        std=std,
        minimum=jnp.min(values),
        maximum=jnp.max(values),
        cv=cv,
        is_stable=cv < STABILITY_CV_THRESHOLD,
    )


def outlier_mask(samples: Samples, *, threshold: float = OUTLIER_Z_THRESHOLD) -> jax.Array:
    """Mark the observations whose modified Z-score exceeds ``threshold``.

    The score measures each observation against the median absolute deviation rather than the
    standard deviation, so the outliers themselves do not inflate the scale they are judged by.

    Args:
        samples: The measurements, along the array's only axis.
        threshold: The modified Z-score an outlier exceeds.

    Returns:
        A boolean array the shape of ``samples``, true where an observation stands apart.
        Nothing stands apart in fewer than three observations, which fix no scale, nor in a
        sample whose deviations are all zero.
    """
    values = jnp.asarray(samples)
    if values.shape[-1] < _MIN_SAMPLES_FOR_MAD:
        return jnp.zeros(values.shape, dtype=bool)
    median = jnp.median(values)
    deviation = jnp.median(jnp.abs(values - median))
    # As in `summarize`, the divisor is made safe before the division, not after.
    has_no_scale = deviation == 0
    score = _MAD_CONSISTENCY_CONSTANT * (values - median) / jnp.where(has_no_scale, 1.0, deviation)
    return jnp.where(has_no_scale, False, jnp.abs(score) > threshold)
