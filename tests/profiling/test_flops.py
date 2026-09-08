"""Tests for FlopsCounter and FlopsResult.

The counter reports XLA's HLO cost analysis of the function, so the assertions are
the arithmetic identities ``HloCostAnalysis`` implements: matmul (M,K)@(K,N) ->
2*M*K*N, elementwise ops -> one per output element, a reduction over n elements ->
n - 1, transcendentals reported separately, a conditional at its most expensive
branch, a loop body counted once.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from hypothesis import given, settings, strategies as st

from calibrax.profiling.flops import FlopsCounter, FlopsResult, FlopsUnavailableError


def _matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    return a @ b


class TestFlopsResult:
    """Tests for FlopsResult frozen dataclass."""

    def test_construction(self) -> None:
        result = FlopsResult(total_flops=100, transcendentals=4, function_name="f")
        assert result.total_flops == 100
        assert result.transcendentals == 4
        assert result.function_name == "f"

    def test_frozen_immutability(self) -> None:
        result = FlopsResult(total_flops=100, transcendentals=0, function_name="f")
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.total_flops = 1  # type: ignore[misc]


class TestIdentities:
    """Counts follow the arithmetic identities of XLA's cost analysis."""

    def test_simple_add(self) -> None:
        result = FlopsCounter().count(lambda x: x + 1.0, jnp.ones((4, 4)))
        assert result.total_flops == 16

    def test_matmul_flops(self) -> None:
        m, k, n = 8, 4, 6
        result = FlopsCounter().count(_matmul, jnp.ones((m, k)), jnp.ones((k, n)))
        assert result.total_flops == 2 * m * k * n

    @given(
        m=st.integers(min_value=1, max_value=16),
        k=st.integers(min_value=1, max_value=16),
        n=st.integers(min_value=1, max_value=16),
    )
    @settings(max_examples=20, deadline=None)
    def test_matmul_identity_holds_for_any_shape(self, m: int, k: int, n: int) -> None:
        result = FlopsCounter().count(_matmul, jnp.ones((m, k)), jnp.ones((k, n)))
        assert result.total_flops == 2 * m * k * n

    def test_elementwise_chain(self) -> None:
        result = FlopsCounter().count(lambda x: (x + 1.0) * 2.0, jnp.ones((3, 5)))
        assert result.total_flops == 30  # add 15 + mul 15

    def test_reduction_counts_n_minus_one_adds(self) -> None:
        result = FlopsCounter().count(jnp.sum, jnp.ones((4, 8)))
        assert result.total_flops == 31

    def test_transcendentals_are_reported_separately(self) -> None:
        result = FlopsCounter().count(jnp.sin, jnp.ones((4, 8)))
        assert result.total_flops == 0
        assert result.transcendentals == 32

    def test_valid_convolution_identity(self) -> None:
        def conv(x: jax.Array) -> jax.Array:
            return jax.lax.conv_general_dilated(
                x,
                jnp.ones((3, 3, 1, 4)),
                (1, 1),
                "VALID",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )

        result = FlopsCounter().count(conv, jnp.ones((1, 8, 8, 1)))
        assert result.total_flops == 2 * 36 * 9 * 4  # 6x6 outputs, 3x3 taps, 4 channels

    def test_identity_function_has_zero_cost(self) -> None:
        result = FlopsCounter().count(lambda x: x, jnp.ones((4,)))
        assert result.total_flops == 0
        assert result.transcendentals == 0

    def test_static_argnums_handled(self) -> None:
        def fn_with_static(x: jax.Array, n: int) -> jax.Array:
            return x * n

        result = FlopsCounter().count(fn_with_static, jnp.ones((5,)), 3, static_argnums=(1,))
        assert result.total_flops == 5

    def test_function_name_captured(self) -> None:
        def my_special_fn(x: jax.Array) -> jax.Array:
            return x * 2.0

        result = FlopsCounter().count(my_special_fn, jnp.ones((2,)))
        assert result.function_name == "my_special_fn"

    def test_callable_without_a_name_is_named_by_type(self) -> None:
        scaled = functools.partial(jnp.multiply, 2.0)
        result = FlopsCounter().count(scaled, jnp.ones((2,)))
        assert result.function_name == "partial"
        assert result.total_flops == 2


class TestNesting:
    """Nothing hides inside a nested computation."""

    def test_nested_jit_counts_the_same_as_flat(self) -> None:
        a, b = jnp.ones((64, 128)), jnp.ones((128, 40))
        flat = FlopsCounter().count(_matmul, a, b)
        nested = FlopsCounter().count(jax.jit(_matmul), a, b)

        def outer_fn(x: jax.Array, y: jax.Array) -> jax.Array:
            return jax.jit(_matmul)(x, y)

        outer = FlopsCounter().count(outer_fn, a, b)
        assert flat.total_flops == 2 * 64 * 128 * 40
        assert nested.total_flops == flat.total_flops
        assert outer.total_flops == flat.total_flops

    def test_custom_jvp_is_counted(self) -> None:
        result = FlopsCounter().count(jax.nn.relu, jnp.ones((4, 8)))
        assert result.total_flops == 32

    def test_checkpoint_is_counted(self) -> None:
        fn = jax.checkpoint(lambda x: jnp.sin(x) @ jnp.ones((8, 3)))
        result = FlopsCounter().count(fn, jnp.ones((4, 8)))
        assert result.total_flops == 2 * 4 * 8 * 3
        assert result.transcendentals == 32

    def test_conditional_counts_its_most_expensive_branch(self) -> None:
        def switch(x: jax.Array) -> jax.Array:
            return jax.lax.switch(1, [lambda v: v, lambda v: v * 2, lambda v: v * 3], x)

        result = FlopsCounter().count(switch, jnp.ones((4, 8)))
        assert result.total_flops == 32  # the sum of the branches would be 64

    def test_loop_body_is_counted_once(self) -> None:
        def scan(x: jax.Array) -> jax.Array:
            return jax.lax.scan(lambda c, e: (c + e, e), jnp.zeros(8), x)[0]

        result = FlopsCounter().count(scan, jnp.ones((4, 8)))
        assert 8 <= result.total_flops < 4 * 8  # body plus loop bookkeeping, not 4 bodies

    def test_mlp_layer(self) -> None:
        def mlp(x: jax.Array, w: jax.Array) -> jax.Array:
            return jax.nn.relu(x @ w)

        batch, features_in, features_out = 8, 16, 32
        x = jnp.ones((batch, features_in))
        w = jnp.ones((features_in, features_out))
        result = FlopsCounter().count(mlp, x, w)
        assert result.total_flops == 2 * batch * features_in * features_out + batch * features_out

    def test_nnx_inference_model(self) -> None:
        """A function closing over an NNX module; the zero bias may fold away at lowering."""
        model = nnx.Linear(16, 32, rngs=nnx.Rngs(0))

        def forward(x: jax.Array) -> jax.Array:
            return model(x)

        result = FlopsCounter().count(forward, jnp.ones((8, 16)))
        assert result.total_flops >= 2 * 8 * 16 * 32

    def test_nnx_state_and_prng_key_arguments(self) -> None:
        """The training-step shape: merged state, a typed key, a random draw."""
        graphdef, state = nnx.split(nnx.Linear(16, 32, rngs=nnx.Rngs(0)))

        def step(module_state: nnx.State, x: jax.Array, key: jax.Array) -> jax.Array:
            model = nnx.merge(graphdef, module_state)
            return (model(x) * jax.random.normal(key, (8, 32))).sum()

        result = FlopsCounter().count(step, state, jnp.ones((8, 16)), jax.random.key(0))
        assert result.total_flops > 2 * 8 * 16 * 32
        assert result.transcendentals > 0


class TestUnavailable:
    def test_custom_call_without_a_cost_raises(self) -> None:
        def callback(x: jax.Array) -> jax.Array:
            return jax.pure_callback(lambda v: v * 2, jax.ShapeDtypeStruct((4,), jnp.float32), x)

        with pytest.raises(FlopsUnavailableError, match="callback"):
            FlopsCounter().count(callback, jnp.ones((4,)))

    def test_the_error_is_a_value_error(self) -> None:
        assert issubclass(FlopsUnavailableError, ValueError)


class TestWithoutCpuBackend:
    """A process started without a CPU backend still gets an analysis.

    The counter then lowers for the default device and, when that lowering carries
    no analysis (PJRT plugins), reads the compiled executable's. On an accelerator
    that is the optimised executable, so fused or library kernels can change or
    withhold the count; the matmul identity below survives it.
    """

    def test_falls_back_to_the_default_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_devices = jax.devices

        def devices_without_cpu(backend: str | None = None) -> list[jax.Device]:
            if backend == "cpu":
                raise RuntimeError("Unknown backend cpu. Available backends are ['cuda']")
            return real_devices(backend)

        monkeypatch.setattr(jax, "devices", devices_without_cpu)
        result = FlopsCounter().count(_matmul, jnp.ones((64, 128)), jnp.ones((128, 40)))
        assert result.total_flops == 2 * 64 * 128 * 40
