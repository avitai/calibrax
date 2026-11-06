"""The percentile bootstrap interval, checked against scipy.stats.bootstrap as the reference."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from scipy import stats
from substrax.testing import TraceCounter

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
    counter = TraceCounter()
    counted_mean = counter.wrap(_mean)
    compiled = jax.jit(
        lambda data, key: bootstrap_interval(counted_mean, data, key=key, num_resamples=32)
    )
    with counter.expect(new_traces=2):  # the point estimate and the vmapped resample
        compiled(jnp.arange(8.0), jax.random.key(0))
    with counter.expect(new_traces=0):
        compiled(jnp.arange(8.0) * 2.0, jax.random.key(1))


def _spread(a: jax.Array, b: jax.Array) -> jax.Array:
    """A paired statistic with a reduction in it, so batching could plausibly shift a bit."""
    centred = a - jnp.mean(a)
    return jnp.sum(centred * (b - jnp.mean(b))) / jnp.sqrt(jnp.sum(centred**2) + 1e-12)


#: How far a grouped resample may sit from the ungrouped one, in ULP at the samples' scale.
#: Derived from measurement rather than fitted to a failure: on CPU at 32 resamples over 400
#: rows the mean-centred dot product below moved 3 ULP and Spearman 1.5, so this sits a little
#: over twice the worst observed. It is a *rounding* bound -- the grouping changes the order a
#: reduction accumulates in, nothing else -- and a batch that reordered or dropped a resample
#: would exceed it by orders of magnitude.
_GROUPING_ULP_BOUND = 8


@pytest.mark.parametrize("batch_size", [1, 3, 7, 32])
def test_batching_the_resamples_stays_within_rounding(batch_size: int) -> None:
    """`batch_size` bounds memory, and what it costs is a few ULP -- measured, not assumed.

    Each resample's mathematical value is independent of the grouping, but XLA's reduction
    layout is not: a statistic that reduces over the data axis accumulates in an order that
    depends on the batch shape. So this asserts a rounding bound rather than equality: Spearman
    is exact at 100 resamples over 50,000 rows, but the statistic below is not exact at 400, so
    exactness is a per-case measurement.
    """
    data = np.random.default_rng(3).normal(size=400).astype(np.float32)
    other = (data * 0.5 + np.random.default_rng(4).normal(size=400)).astype(np.float32)
    shared = {"key": jax.random.key(11), "num_resamples": 32}

    reference = bootstrap_interval(_spread, data, other, **shared)
    batched = bootstrap_interval(_spread, data, other, **shared, batch_size=batch_size)

    scale = np.float32(np.max(np.abs(np.asarray(reference.samples))))
    allowed = _GROUPING_ULP_BOUND * float(np.spacing(scale))
    worst = float(np.max(np.abs(np.asarray(reference.samples) - np.asarray(batched.samples))))

    assert worst <= allowed, f"grouped resamples moved {worst}, above {allowed}"
    assert float(batched.value) == float(reference.value), "the point estimate is not resampled"


def test_the_rounding_bound_could_have_failed() -> None:
    """A positive control: a different key moves the samples far beyond the bound above."""
    data = np.random.default_rng(3).normal(size=400).astype(np.float32)
    shared = {"num_resamples": 32}

    one = bootstrap_interval(_mean, data, key=jax.random.key(11), **shared)
    other = bootstrap_interval(_mean, data, key=jax.random.key(12), **shared, batch_size=4)

    scale = np.float32(np.max(np.abs(np.asarray(one.samples))))
    worst = float(np.max(np.abs(np.asarray(one.samples) - np.asarray(other.samples))))

    assert worst > _GROUPING_ULP_BOUND * float(np.spacing(scale))


def test_a_batch_size_above_the_resample_count_is_exact() -> None:
    """The one grouping measured exact for every statistic tried, and why.

    Above `num_resamples` there is a single group and no scan, so the computation is the
    default's shape reached by another route. It therefore saves no memory either, which is
    worth knowing before reaching for it as a safe default.
    """
    data = np.random.default_rng(5).normal(size=50).astype(np.float32)
    other = (data * 0.5).astype(np.float32)
    shared = {"key": jax.random.key(1), "num_resamples": 8}

    for statistic in (_mean, _spread):
        arrays = (data,) if statistic is _mean else (data, other)
        assert jnp.array_equal(
            bootstrap_interval(statistic, *arrays, **shared, batch_size=1000).samples,
            bootstrap_interval(statistic, *arrays, **shared).samples,
        )


@pytest.mark.parametrize("batch_size", [0, -1])
def test_a_batch_size_below_one_is_refused(batch_size: int) -> None:
    """Zero would divide the resamples into no groups and evaluate nothing."""
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        bootstrap_interval(_mean, jnp.arange(8.0), key=jax.random.key(0), batch_size=batch_size)
