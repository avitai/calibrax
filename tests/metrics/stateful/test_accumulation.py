"""Device accumulation of batch means through the public metric lifecycle."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles

from calibrax.metrics.stateful import MeanAccumulator


def test_short_batch_is_weighted_by_its_actual_records() -> None:
    accumulator = MeanAccumulator(("loss", "accuracy"))
    accumulator.update({"loss": jnp.float32(1), "accuracy": jnp.float32(0.5)}, jnp.int32(2))
    accumulator.update({"loss": jnp.float32(3), "accuracy": jnp.float32(1)}, jnp.int32(1))

    actual = accumulator.compute()
    np.testing.assert_allclose(actual["loss"], 5 / 3, rtol=1e-6)
    np.testing.assert_allclose(actual["accuracy"], 2 / 3, rtol=1e-6)


@pytest.mark.parametrize("empty_value", [float("nan"), float("inf"), -float("inf")])
def test_empty_batch_cannot_contaminate_a_later_observation(empty_value: float) -> None:
    accumulator = MeanAccumulator(("loss",))
    accumulator.update({"loss": jnp.float32(empty_value)}, jnp.int32(0))
    assert np.isnan(float(accumulator.compute()["loss"]))

    accumulator.update({"loss": jnp.float32(7)}, jnp.int32(2))
    np.testing.assert_array_equal(accumulator.compute()["loss"], 7)


@pytest.mark.parametrize(
    ("metrics", "records", "error"),
    [
        ({"accuracy": jnp.float32(1)}, jnp.int32(1), KeyError),
        ({"loss": jnp.ones(2)}, jnp.int32(1), ValueError),
        ({"loss": jnp.float32(1)}, jnp.ones(2, jnp.int32), ValueError),
        ({"loss": jnp.float32(1)}, jnp.float32(1.5), TypeError),
        ({"loss": jnp.float32(1)}, jnp.bool_(True), TypeError),
    ],
)
def test_invalid_update_is_refused_without_advancing_the_window(
    metrics: dict[str, jax.Array], records: jax.Array, error: type[Exception]
) -> None:
    accumulator = MeanAccumulator(("loss",))
    accumulator.update({"loss": jnp.float32(2)}, jnp.int32(2))
    with pytest.raises(error):
        accumulator.update(metrics, records)
    accumulator.update({"loss": jnp.float32(8)}, jnp.int32(1))
    np.testing.assert_array_equal(accumulator.compute()["loss"], 4)


def test_integer_and_boolean_means_keep_record_weights() -> None:
    accumulator = MeanAccumulator(("loss", "accuracy"))
    accumulator.update({"loss": jnp.int32(3), "accuracy": jnp.bool_(True)}, jnp.int32(2))
    accumulator.update({"loss": jnp.int32(6), "accuracy": jnp.bool_(False)}, jnp.int32(1))
    np.testing.assert_array_equal(accumulator.compute()["loss"], 4)
    np.testing.assert_allclose(accumulator.compute()["accuracy"], 2 / 3, rtol=1e-6)


def test_reset_and_missing_metrics_have_independent_windows() -> None:
    accumulator = MeanAccumulator(("loss", "accuracy"))
    assert all(np.isnan(float(value)) for value in accumulator.compute().values())
    accumulator.update({"loss": jnp.float32(2)}, jnp.int32(8))
    accumulator.update({"accuracy": jnp.float32(0.75)}, jnp.int32(1))
    np.testing.assert_array_equal(accumulator.compute()["loss"], 2)
    np.testing.assert_array_equal(accumulator.compute()["accuracy"], 0.75)

    accumulator.reset()
    assert all(np.isnan(float(value)) for value in accumulator.compute().values())
    accumulator.update({"loss": jnp.float32(9)}, jnp.int32(1))
    np.testing.assert_array_equal(accumulator.compute()["loss"], 9)
    assert np.isnan(float(accumulator.compute()["accuracy"]))


@pytest.mark.parametrize("graph", [True, False])
def test_native_donated_binding_keeps_metrics_and_reset_on_device(graph: bool) -> None:
    accumulator = MeanAccumulator(("loss",))

    def record(
        current: MeanAccumulator, mean: jax.Array, records: jax.Array
    ) -> dict[str, jax.Array]:
        current.update({"loss": mean}, records)
        return current.compute()

    bound = nnx.jit_partial(record, accumulator, graph=graph, graph_updates=False, donate_argnums=0)
    means = jnp.asarray([1, 3, jnp.nan], jnp.float32)
    counts = jnp.asarray([2, 1, 0], jnp.int32)
    inputs = list(zip(list(means), list(counts), strict=True))
    with jax.transfer_guard_device_to_host("disallow"):
        first = bound(*inputs[0])
        with expect_compiles(0):
            bound(*inputs[1])
            actual = bound(*inputs[2])
        nnx.jit(lambda current: current.reset(), graph=graph)(accumulator)
        empty = accumulator.compute()

    np.testing.assert_array_equal(first["loss"], 1)
    np.testing.assert_allclose(actual["loss"], 5 / 3, rtol=1e-6)
    assert np.isnan(float(empty["loss"]))


def test_scan_retains_the_record_weighted_window() -> None:
    accumulator = MeanAccumulator(("loss",))

    @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=nnx.Carry)
    def record(current: MeanAccumulator, mean: jax.Array, records: jax.Array) -> MeanAccumulator:
        current.update({"loss": mean}, records)
        return current

    record(accumulator, jnp.array([1.0, 3.0, jnp.nan]), jnp.array([2, 1, 0], jnp.int32))
    np.testing.assert_allclose(accumulator.compute()["loss"], 5 / 3, rtol=1e-6)


def _scaled_mean(scale: jax.Array) -> jax.Array:
    accumulator = MeanAccumulator(("loss",))
    accumulator.update({"loss": scale}, jnp.int32(2))
    accumulator.update({"loss": 3 * scale}, jnp.int32(1))
    return accumulator.compute()["loss"]


def test_vmap_and_differentiation_preserve_native_metric_math() -> None:
    scales = jnp.array([1.0, 2.0])
    mapped = nnx.vmap(_scaled_mean, in_axes=0, out_axes=0)(scales)
    np.testing.assert_allclose(mapped, [5 / 3, 10 / 3], rtol=1e-6)
    actual, derivative = jax.jit(jax.value_and_grad(_scaled_mean))(jnp.float32(2))
    np.testing.assert_allclose(actual, 10 / 3, rtol=1e-6)
    np.testing.assert_allclose(derivative, 5 / 3, rtol=1e-6)


def test_an_empty_observation_has_zero_derivative_without_hiding_counted_nan() -> None:
    def mean(values: jax.Array) -> jax.Array:
        accumulator = MeanAccumulator(("loss",))
        accumulator.update({"loss": values[0]}, jnp.int32(0))
        accumulator.update({"loss": values[1]}, jnp.int32(2))
        return accumulator.compute()["loss"]

    values = jnp.array([jnp.nan, 7.0])
    result, reverse = jax.jit(jax.value_and_grad(mean))(values)
    forward = jax.jit(jax.jacfwd(mean))(values)
    np.testing.assert_array_equal(result, 7)
    np.testing.assert_array_equal(reverse, [0, 1])
    np.testing.assert_array_equal(forward, [0, 1])
    assert np.isnan(float(jax.jit(mean)(jnp.array([1.0, jnp.nan]))))
