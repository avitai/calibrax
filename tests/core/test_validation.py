"""Tests for shared scalar and boolean validation in calibrax.core."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.core import eager_bool, eager_scalar_float, validate_scalar_numeric


class TestEagerScalarAndBoolValidation:
    """Tests for eager_scalar_float, eager_bool, and validate_scalar_numeric."""

    def test_extracts_python_numpy_and_concrete_jax_0d_scalars(self) -> None:
        assert eager_scalar_float(3) == 3.0
        assert eager_scalar_float(-0.25) == -0.25
        assert eager_scalar_float(np.int64(7)) == 7.0
        assert eager_scalar_float(np.float32(0.5)) == pytest.approx(0.5)
        assert eager_scalar_float(np.float64(1.25)) == 1.25
        assert eager_scalar_float(np.array(2.5, dtype=np.float32)) == pytest.approx(2.5)
        assert eager_scalar_float(np.array(-4, dtype=np.int32)) == -4.0
        assert eager_scalar_float(jnp.array(1.5, dtype=jnp.float32)) == pytest.approx(1.5)
        assert eager_scalar_float(jnp.array(-3, dtype=jnp.int32)) == -3.0

    def test_returns_none_for_booleans_tracers_and_nd_arrays(self) -> None:
        assert eager_scalar_float(True) is None
        assert eager_scalar_float(False) is None
        assert eager_scalar_float(np.bool_(True)) is None
        assert eager_scalar_float(np.array(True, dtype=bool)) is None
        assert eager_scalar_float(jnp.array(True)) is None
        assert eager_scalar_float(np.array([1.0])) is None
        assert eager_scalar_float(jnp.array([1.0, 2.0])) is None
        assert eager_scalar_float("not_a_number") is None  # type: ignore[arg-type]

        seen: list[float | None] = []

        @jax.jit
        def _probe(x: jax.Array) -> jax.Array:
            seen.append(eager_scalar_float(x))
            return x + 1.0

        _probe(jnp.array(0.5))
        assert seen == [None]

    def test_eager_bool_extracts_concrete_predicates_and_skips_tracers(self) -> None:
        assert eager_bool(True) is True
        assert eager_bool(False) is False
        assert eager_bool(np.bool_(True)) is True
        assert eager_bool(np.array(False, dtype=bool)) is False
        assert eager_bool(jnp.all(jnp.array([1.0, 2.0]) > 0.0)) is True
        assert eager_bool(jnp.any(jnp.array([1.0, -2.0]) < 0.0)) is True

        seen_bool: list[bool | None] = []

        @jax.jit
        def _probe_pred(x: jax.Array) -> jax.Array:
            seen_bool.append(eager_bool(jnp.all(x > 0.0)))
            return x * 2.0

        _probe_pred(jnp.array([1.0, 2.0]))
        assert seen_bool == [None]

        with pytest.raises(TypeError, match="boolean"):
            eager_bool(jnp.array(1.0))
        with pytest.raises(ValueError, match="0-D"):
            eager_bool(jnp.array([True, False]))

    def test_validate_scalar_numeric_handles_traced_and_static_modes(self) -> None:
        for bad_bool in (
            True,
            False,
            np.bool_(True),
            np.array(False, dtype=bool),
            jnp.array(True),
        ):
            with pytest.raises(TypeError, match="param"):
                validate_scalar_numeric(bad_bool, name="param")
        for bad_nd in ([0.5], (0.5, 0.6), np.array([0.5]), jnp.array([0.5, 0.6])):
            with pytest.raises(ValueError, match="param"):
                validate_scalar_numeric(bad_nd, name="param")  # type: ignore[arg-type]

        eager_f, arr_f = validate_scalar_numeric(jnp.asarray(1.5), name="param")
        assert eager_f == pytest.approx(1.5)
        assert arr_f.ndim == 0

        vmapped = jax.jit(jax.vmap(lambda x: validate_scalar_numeric(x, name="param")[1]))(
            jnp.array([0.1, 0.2], dtype=jnp.float32)
        )
        np.testing.assert_allclose(np.asarray(vmapped), np.array([0.1, 0.2]), atol=1e-7)

        with pytest.raises(TypeError, match="static_argnames"):
            jax.jit(lambda x: validate_scalar_numeric(x, name="alpha", allow_traced=False)[1])(
                jnp.array(0.1, dtype=jnp.float32)
            )
