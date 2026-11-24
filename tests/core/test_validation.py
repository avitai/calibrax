"""Tests for shared eager scalar extraction in calibrax.core._validation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.core._validation import eager_scalar_float


class TestEagerScalarFloat:
    """Tests for eager_scalar_float."""

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

        seen: list[float | None] = []

        @jax.jit
        def _probe(x: jax.Array) -> jax.Array:
            seen.append(eager_scalar_float(x))
            return x + 1.0

        _probe(jnp.array(0.5))
        assert seen == [None]
