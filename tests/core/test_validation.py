"""Tests for shared eager scalar extraction in calibrax.core._validation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.core._validation import eager_scalar_float, validate_scalar_numeric


class TestEagerScalarFloat:
    """Tests for eager_scalar_float and validate_scalar_numeric."""

    def test_extracts_python_and_numpy_numeric_scalars_and_0d_arrays(self) -> None:
        assert eager_scalar_float(3) == 3.0
        assert eager_scalar_float(-0.25) == -0.25
        assert eager_scalar_float(np.int64(7)) == 7.0
        assert eager_scalar_float(np.float32(0.5)) == pytest.approx(0.5)
        assert eager_scalar_float(np.float64(1.25)) == 1.25
        assert eager_scalar_float(np.array(2.5, dtype=np.float32)) == pytest.approx(2.5)
        assert eager_scalar_float(np.array(-4, dtype=np.int32)) == -4.0

    def test_returns_none_for_booleans_jax_arrays_tracers_and_nd_arrays(self) -> None:
        assert eager_scalar_float(True) is None
        assert eager_scalar_float(False) is None
        assert eager_scalar_float(np.bool_(True)) is None
        assert eager_scalar_float(np.array(True, dtype=bool)) is None
        assert eager_scalar_float(np.array([1.0])) is None
        assert eager_scalar_float(jnp.array(0.5)) is None
        assert eager_scalar_float("not_a_number") is None  # type: ignore[arg-type]

        seen: list[float | None] = []

        @jax.jit
        def _probe(x: jax.Array) -> jax.Array:
            seen.append(eager_scalar_float(x))
            return x + 1.0

        _probe(jnp.array(0.5))
        assert seen == [None]

    def test_validate_scalar_numeric_rejects_booleans_and_non_scalars(self) -> None:
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

        eager_f, arr_f = validate_scalar_numeric(np.float32(0.25), name="param")
        assert eager_f == pytest.approx(0.25)
        assert arr_f.ndim == 0

        vmapped = jax.jit(jax.vmap(lambda x: validate_scalar_numeric(x, name="param")[1]))(
            jnp.array([0.1, 0.2], dtype=jnp.float32)
        )
        np.testing.assert_allclose(np.asarray(vmapped), np.array([0.1, 0.2]), atol=1e-7)
