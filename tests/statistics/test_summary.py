"""Summary statistics and outlier detection, checked against numpy and under the transforms."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing import TraceCounter

from calibrax.statistics import (
    outlier_mask,
    OUTLIER_Z_THRESHOLD,
    SampleSummary,
    STABILITY_CV_THRESHOLD,
    summarize,
)


_SAMPLES = [1.0, 2.0, 3.0, 4.0, 5.0]


def test_the_summary_matches_numpy() -> None:
    data = np.random.default_rng(0).normal(3.0, 2.0, size=200)

    summary = summarize(data)

    assert float(summary.mean) == pytest.approx(float(np.mean(data)), rel=1e-6)
    assert float(summary.median) == pytest.approx(float(np.median(data)), rel=1e-6)
    assert float(summary.std) == pytest.approx(float(np.std(data, ddof=1)), rel=1e-6)
    assert float(summary.minimum) == pytest.approx(float(np.min(data)), rel=1e-6)
    assert float(summary.maximum) == pytest.approx(float(np.max(data)), rel=1e-6)


def test_the_coefficient_of_variation_is_the_std_over_the_mean() -> None:
    summary = summarize(_SAMPLES)

    expected = float(np.std(_SAMPLES, ddof=1)) / float(np.mean(_SAMPLES))
    assert float(summary.cv) == pytest.approx(expected, rel=1e-6)


def test_a_single_observation_has_no_spread() -> None:
    summary = summarize([42.0])

    assert float(summary.mean) == float(summary.median) == 42.0
    assert float(summary.std) == 0.0
    assert float(summary.cv) == 0.0
    assert bool(summary.is_stable)


def test_a_zero_mean_gives_a_zero_coefficient_of_variation() -> None:
    summary = summarize([-1.0, 0.0, 1.0])

    assert float(summary.mean) == 0.0
    assert float(summary.cv) == 0.0


def test_identical_observations_are_stable() -> None:
    summary = summarize([2.0, 2.0, 2.0, 2.0])

    assert float(summary.cv) == 0.0
    assert bool(summary.is_stable)


def test_stability_follows_the_threshold() -> None:
    steady = summarize([100.0, 101.0, 99.0, 100.5, 99.5])
    noisy = summarize([100.0, 150.0, 50.0, 120.0, 80.0])

    assert float(steady.cv) < STABILITY_CV_THRESHOLD
    assert bool(steady.is_stable)
    assert float(noisy.cv) >= STABILITY_CV_THRESHOLD
    assert not bool(noisy.is_stable)


def test_the_summary_is_a_pytree_of_arrays() -> None:
    summary = summarize(_SAMPLES)

    leaves = jax.tree.leaves(summary)
    assert len(leaves) == 7
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)

    doubled = jax.tree.map(lambda leaf: leaf * 2, summarize([1.0, 2.0, 3.0]))
    assert isinstance(doubled, SampleSummary)
    assert float(doubled.mean) == 4.0


def test_jit_traces_once_for_data_of_the_same_shape() -> None:
    counter = TraceCounter()
    compiled = jax.jit(counter.wrap(summarize))

    with counter.expect(new_traces=1):
        compiled(jnp.asarray(_SAMPLES))
    with counter.expect(new_traces=0):
        compiled(jnp.asarray([5.0, 4.0, 3.0, 2.0, 1.0]))


def test_vmap_summarizes_each_row() -> None:
    rows = jnp.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

    summary = jax.vmap(summarize)(rows)

    assert summary.mean.shape == (2,)
    assert [float(value) for value in summary.mean] == [2.0, 20.0]


def test_nnx_jit_returns_the_same_summary() -> None:
    eager = summarize(_SAMPLES)

    compiled = nnx.jit(summarize)(jnp.asarray(_SAMPLES))

    assert float(compiled.mean) == pytest.approx(float(eager.mean))
    assert float(compiled.cv) == pytest.approx(float(eager.cv))


def test_the_gradient_of_the_mean_is_finite_at_a_zero_mean() -> None:
    def mean_of(data: jax.Array) -> jax.Array:
        return summarize(data).mean

    gradient = jax.grad(mean_of)(jnp.asarray([-1.0, 0.0, 1.0]))

    assert jnp.all(jnp.isfinite(gradient))


def test_the_gradient_of_the_coefficient_of_variation_is_finite_at_a_zero_mean() -> None:
    def cv_of(data: jax.Array) -> jax.Array:
        return summarize(data).cv

    gradient = jax.grad(cv_of)(jnp.asarray([-1.0, 0.0, 1.0]))

    assert jnp.all(jnp.isfinite(gradient))


def test_outliers_are_the_planted_ones() -> None:
    mask = outlier_mask([1.0, 1.1, 0.9, 1.05, 100.0, 0.95, 1.0])

    assert [int(index) for index in jnp.flatnonzero(mask)] == [4]


def test_uniform_observations_hold_no_outliers() -> None:
    mask = outlier_mask([1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98])

    assert not bool(jnp.any(mask))


def test_identical_observations_hold_no_outliers() -> None:
    mask = outlier_mask([5.0, 5.0, 5.0, 5.0, 5.0])

    assert not bool(jnp.any(mask))


def test_too_few_observations_for_a_deviation_hold_no_outliers() -> None:
    assert not bool(jnp.any(outlier_mask([])))
    assert not bool(jnp.any(outlier_mask([1.0, 100.0])))


def test_a_lower_threshold_flags_more() -> None:
    data = [1.0, 1.1, 0.9, 1.05, 2.0, 0.95, 1.0]

    default = outlier_mask(data)
    lenient = outlier_mask(data, threshold=OUTLIER_Z_THRESHOLD)
    strict = outlier_mask(data, threshold=1.0)

    assert int(jnp.sum(strict)) > int(jnp.sum(default))
    assert int(jnp.sum(lenient)) == int(jnp.sum(default))


def test_the_mask_keeps_the_shape_of_the_observations() -> None:
    mask = outlier_mask(_SAMPLES)

    assert mask.shape == (len(_SAMPLES),)
    assert mask.dtype == jnp.bool_


def test_outliers_under_jit_and_vmap() -> None:
    rows = jnp.asarray([[1.0, 1.1, 0.9, 100.0], [1.0, 1.1, 0.9, 1.05]])

    masks = jax.jit(jax.vmap(outlier_mask))(rows)

    assert masks.shape == (2, 4)
    assert [int(value) for value in masks[0]] == [0, 0, 0, 1]
    assert not bool(jnp.any(masks[1]))
