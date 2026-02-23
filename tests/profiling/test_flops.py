"""Tests for FlopsCounter and FlopsResult.

Uses known mathematical identities to verify FLOP counting:
matmul (M,K)@(K,N) -> 2*M*K*N, elementwise -> product of output shape.
"""

import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from calibrax.profiling.flops import FlopsCounter, FlopsResult


class TestFlopsResult:
    """Tests for FlopsResult frozen dataclass."""

    def test_construction(self) -> None:
        result = FlopsResult(
            total_flops=1000,
            flops_by_operation={"dot_general": 800, "add": 200},
            num_operations=5,
            function_name="matmul",
        )
        assert result.total_flops == 1000
        assert result.flops_by_operation["dot_general"] == 800

    def test_frozen_immutability(self) -> None:
        result = FlopsResult(
            total_flops=1000,
            flops_by_operation={},
            num_operations=1,
            function_name="test",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.total_flops = 0  # type: ignore[misc]


class TestFlopsCounter:
    """Tests for FlopsCounter FLOP counting."""

    def test_simple_add(self) -> None:
        def add_fn(x: jax.Array, y: jax.Array) -> jax.Array:
            return x + y

        counter = FlopsCounter()
        x = jnp.ones((4, 4))
        y = jnp.ones((4, 4))
        result = counter.count(add_fn, x, y)

        assert isinstance(result, FlopsResult)
        assert result.total_flops == 16  # 4*4 = 16 elements

    def test_matmul_flops(self) -> None:
        """Matmul (M,K)@(K,N) should give exactly 2*M*K*N FLOPs."""
        m, k, n = 8, 4, 6

        def matmul_fn(a: jax.Array, b: jax.Array) -> jax.Array:
            return a @ b

        counter = FlopsCounter()
        a = jnp.ones((m, k))
        b = jnp.ones((k, n))
        result = counter.count(matmul_fn, a, b)

        expected = 2 * m * k * n
        assert result.total_flops == expected

    def test_elementwise_chain(self) -> None:
        def chain(x: jax.Array) -> jax.Array:
            return (x + 1.0) * 2.0

        counter = FlopsCounter()
        x = jnp.ones((3, 5))
        result = counter.count(chain, x)

        # add: 15 flops, mul: 15 flops = 30
        assert result.total_flops == 30

    def test_flops_by_operation_breakdown(self) -> None:
        def fn(a: jax.Array, b: jax.Array) -> jax.Array:
            return a @ b + jnp.ones((4, 4))

        counter = FlopsCounter()
        a = jnp.ones((4, 4))
        b = jnp.ones((4, 4))
        result = counter.count(fn, a, b)

        assert "dot_general" in result.flops_by_operation
        assert "add" in result.flops_by_operation

    def test_function_name_captured(self) -> None:
        def my_special_fn(x: jax.Array) -> jax.Array:
            return x * 2.0

        counter = FlopsCounter()
        result = counter.count(my_special_fn, jnp.ones((2,)))

        assert result.function_name == "my_special_fn"

    def test_static_argnums_handled(self) -> None:
        def fn_with_static(
            x: jax.Array,
            n: int,
        ) -> jax.Array:
            return x * n

        counter = FlopsCounter()
        x = jnp.ones((5,))
        result = counter.count(fn_with_static, x, 3, static_argnums=(1,))

        assert isinstance(result, FlopsResult)
        assert result.total_flops == 5

    def test_mlp_layer(self) -> None:
        """Realistic test: linear + relu."""

        def mlp(x: jax.Array, w: jax.Array) -> jax.Array:
            return jax.nn.relu(x @ w)

        counter = FlopsCounter()
        batch, features_in, features_out = 8, 16, 32
        x = jnp.ones((batch, features_in))
        w = jnp.ones((features_in, features_out))
        result = counter.count(mlp, x, w)

        # matmul: 2 * 8 * 16 * 32 = 8192
        # relu: 8 * 32 = 256 (elementwise comparison)
        assert result.total_flops > 0
        assert "dot_general" in result.flops_by_operation

    def test_nnx_inference_model(self) -> None:
        """FlopsCounter works with inference-only NNX models."""
        model = nnx.Linear(16, 32, rngs=nnx.Rngs(0))

        def forward(x: jax.Array) -> jax.Array:
            return model(x)

        counter = FlopsCounter()
        x = jnp.ones((8, 16))
        result = counter.count(forward, x)

        assert result.total_flops > 0
        assert "dot_general" in result.flops_by_operation

    def test_classify_reduction_uses_input_size(self) -> None:
        counter = FlopsCounter()
        fake_eqn = SimpleNamespace()
        with patch.object(counter, "_input_size", return_value=123) as mock_input_size:
            flops = counter._classify_primitive_flops("reduce_sum", fake_eqn, {})
        assert flops == 123
        mock_input_size.assert_called_once_with(fake_eqn)

    def test_classify_conv_and_nested_dispatch(self) -> None:
        counter = FlopsCounter()
        fake_eqn = SimpleNamespace()
        with (
            patch.object(counter, "_count_conv", return_value=10) as mock_conv,
            patch.object(counter, "_count_nested_jaxpr", return_value=20) as mock_nested,
        ):
            conv_flops = counter._classify_primitive_flops("conv_general_dilated", fake_eqn, {})
            nested_flops = counter._classify_primitive_flops("scan", fake_eqn, {})

        assert conv_flops == 10
        assert nested_flops == 20
        mock_conv.assert_called_once_with(fake_eqn)
        mock_nested.assert_called_once_with(fake_eqn, {})

    def test_classify_unknown_primitive_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        counter = FlopsCounter()
        fake_eqn = SimpleNamespace()
        with caplog.at_level("WARNING"):
            flops = counter._classify_primitive_flops("unknown_primitive", fake_eqn, {})
        assert flops == 0
        assert any("Unknown primitive" in msg for msg in caplog.messages)

    def test_count_dot_general_missing_output_shape_returns_zero(self) -> None:
        counter = FlopsCounter()
        fake_eqn = SimpleNamespace(
            outvars=[SimpleNamespace(aval=object())],
            invars=[SimpleNamespace(aval=SimpleNamespace(shape=(4, 8)))],
            params={"dimension_numbers": (((1,), (0,)), ((), ()))},
        )
        assert counter._count_dot_general(fake_eqn) == 0

    def test_count_dot_general_missing_dimension_numbers_returns_zero(self) -> None:
        counter = FlopsCounter()
        fake_eqn = SimpleNamespace(
            outvars=[SimpleNamespace(aval=SimpleNamespace(shape=(4, 16)))],
            invars=[SimpleNamespace(aval=SimpleNamespace(shape=(4, 8)))],
            params={},
        )
        assert counter._count_dot_general(fake_eqn) == 0

    def test_count_dot_general_missing_lhs_shape_returns_zero(self) -> None:
        counter = FlopsCounter()
        fake_eqn = SimpleNamespace(
            outvars=[SimpleNamespace(aval=SimpleNamespace(shape=(4, 16)))],
            invars=[SimpleNamespace(aval=object())],
            params={"dimension_numbers": (((1,), (0,)), ((), ()))},
        )
        assert counter._count_dot_general(fake_eqn) == 0

    def test_count_conv_missing_output_shape_returns_zero(self) -> None:
        counter = FlopsCounter()
        fake_eqn = SimpleNamespace(
            outvars=[SimpleNamespace(aval=object())],
            invars=[None, SimpleNamespace(aval=SimpleNamespace(shape=(3, 3, 16, 32)))],
        )
        assert counter._count_conv(fake_eqn) == 0

    def test_count_conv_missing_kernel_shape_returns_zero(self) -> None:
        counter = FlopsCounter()
        fake_eqn = SimpleNamespace(
            outvars=[SimpleNamespace(aval=SimpleNamespace(shape=(8, 8, 32)))],
            invars=[None, SimpleNamespace(aval=object())],
        )
        assert counter._count_conv(fake_eqn) == 0

    def test_count_nested_jaxpr_counts_jaxpr_and_closed_jaxpr(self) -> None:
        counter = FlopsCounter()
        closed_jaxpr = jax.make_jaxpr(lambda x: x + 1)(jnp.ones((1,)))
        eqn = SimpleNamespace(
            params={
                "jaxpr": closed_jaxpr.jaxpr,
                "closed_jaxpr": closed_jaxpr,
                "irrelevant": 123,
            }
        )
        with patch.object(counter, "_count_jaxpr", side_effect=[5, 7]) as mock_count_jaxpr:
            total = counter._count_nested_jaxpr(eqn, {})

        assert total == 12
        assert mock_count_jaxpr.call_count == 2

    def test_output_size_handles_empty_outvars_and_missing_shape(self) -> None:
        counter = FlopsCounter()
        eqn_no_outvars = SimpleNamespace(outvars=[])
        eqn_missing_shape = SimpleNamespace(outvars=[SimpleNamespace(aval=object())])

        assert counter._output_size(eqn_no_outvars) == 0
        assert counter._output_size(eqn_missing_shape) == 0

    def test_input_size_handles_empty_invars_and_missing_shape(self) -> None:
        counter = FlopsCounter()
        eqn_no_invars = SimpleNamespace(invars=[])
        eqn_missing_shape = SimpleNamespace(invars=[SimpleNamespace(aval=object())])

        assert counter._input_size(eqn_no_invars) == 0
        assert counter._input_size(eqn_missing_shape) == 0
