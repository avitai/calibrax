"""The percentile bootstrap interval, checked against scipy.stats.bootstrap as the reference."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from scipy import stats

from calibrax.statistics import bootstrap_interval, BootstrapInterval


def _mean(x: jax.Array) -> jax.Array:
    return jnp.mean(x)


def test_the_interval_matches_scipys_percentile_bootstrap() -> None:
    data = np.random.default_rng(0).normal(3.0, 2.0, size=200)

    ours = bootstrap_interval(_mean, data, key=jax.random.key(0), num_resamples=9999)
    reference = stats.bootstrap(
        (data,), np.mean, n_resamples=9999, method="percentile", rng=np.random.default_rng(1)
    ).confidence_interval

    # Two Monte Carlo estimates of the same interval: the endpoints agree to a fraction of the
    # interval's width (about 0.55 here).
    width = reference.high - reference.low
    assert float(ours.lower) == pytest.approx(reference.low, abs=0.05 * width)
    assert float(ours.upper) == pytest.approx(reference.high, abs=0.05 * width)
    assert float(ours.value) == pytest.approx(float(np.mean(data)), rel=1e-6)


def test_paired_arrays_are_resampled_with_the_same_indices() -> None:
    x = jnp.arange(50.0)

    result = bootstrap_interval(
        lambda a, b: jnp.max(jnp.abs(a - b)), x, x, key=jax.random.key(0), num_resamples=64
    )

    assert float(jnp.max(result.samples)) == 0.0


def test_the_samples_hold_one_value_per_resample() -> None:
    result = bootstrap_interval(_mean, jnp.arange(10.0), key=jax.random.key(0), num_resamples=17)

    assert isinstance(result, BootstrapInterval)
    assert result.samples.shape == (17,)
    assert float(result.lower) <= float(result.value) <= float(result.upper)


def test_the_same_key_reproduces_and_another_key_differs() -> None:
    data = jnp.linspace(0.0, 1.0, 40)

    first = bootstrap_interval(_mean, data, key=jax.random.key(3), num_resamples=100)
    again = bootstrap_interval(_mean, data, key=jax.random.key(3), num_resamples=100)
    other = bootstrap_interval(_mean, data, key=jax.random.key(4), num_resamples=100)

    np.testing.assert_array_equal(np.asarray(first.samples), np.asarray(again.samples))
    assert not np.array_equal(np.asarray(first.samples), np.asarray(other.samples))


def test_an_rngs_stream_supplies_the_key() -> None:
    data = jnp.linspace(0.0, 1.0, 40)

    from_rngs = bootstrap_interval(_mean, data, key=nnx.Rngs(sample=5), num_resamples=50)
    from_key = bootstrap_interval(_mean, data, key=nnx.Rngs(sample=5).sample(), num_resamples=50)

    np.testing.assert_array_equal(np.asarray(from_rngs.samples), np.asarray(from_key.samples))


def test_a_key_is_required() -> None:
    with pytest.raises(TypeError):
        bootstrap_interval(_mean, jnp.ones(3), key=None, num_resamples=10)  # type: ignore[arg-type]


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5])
def test_a_confidence_outside_zero_to_one_is_refused(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_interval(_mean, jnp.ones(3), key=jax.random.key(0), confidence=confidence)


def test_arrays_of_different_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="same length"):
        bootstrap_interval(
            lambda a, b: jnp.mean(a - b), jnp.ones(3), jnp.ones(4), key=jax.random.key(0)
        )


def test_no_data_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one array"):
        bootstrap_interval(_mean, key=jax.random.key(0))


def test_a_single_observation_gives_a_point_interval() -> None:
    result = bootstrap_interval(_mean, jnp.array([7.0]), key=jax.random.key(0), num_resamples=20)

    assert float(result.lower) == float(result.upper) == float(result.value) == 7.0


def test_jit_traces_once_for_new_data_and_keys() -> None:
    from substrax.testing import TraceCounter

    counter = TraceCounter()
    counted_mean = counter.wrap(_mean)
    compiled = jax.jit(
        lambda data, key: bootstrap_interval(counted_mean, data, key=key, num_resamples=32)
    )
    with counter.expect(new_traces=2):  # the point estimate and the vmapped resample
        compiled(jnp.arange(8.0), jax.random.key(0))
    with counter.expect(new_traces=0):
        compiled(jnp.arange(8.0) * 2.0, jax.random.key(1))
