"""The rescaled norm family: ``safe_norm``, ``safe_hypot``, ``floor_at`` and ``safe_asinh``.

Ported from hyperbolix (MIT, timoklein/hyperbolix at 96aba05, ``tests/test_math_utils.py``), in the
float32 arms calibrax computes in: a norm or hypotenuse that neither overflows nor underflows where
its squares would, is exactly 0 with an exactly-zero gradient at the origin, and propagates NaN and
inf rather than hiding them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.metrics._utils import floor_at, safe_asinh, safe_hypot, safe_norm


def test_safe_norm_does_not_overflow_or_underflow_in_float32() -> None:
    """``1e30`` and ``3e-34`` are float32 values whose squares are not (1e60 overflows, 1e-68 flushes)."""
    big = jnp.asarray([1e30, 0.0, 0.0], dtype=jnp.float32)
    assert float(safe_norm(big)) == pytest.approx(1e30, rel=1e-6)
    assert not np.isfinite(float(jnp.sqrt(jnp.sum(big**2))))  # the rescale is load-bearing

    tiny = jnp.asarray([3e-34, 4e-34, 0.0], dtype=jnp.float32)
    assert float(safe_norm(tiny)) == pytest.approx(5e-34, rel=1e-5)
    assert float(jnp.sqrt(jnp.sum(tiny**2))) == 0.0  # the plain sum of squares flushes to zero


def test_safe_norm_is_exactly_zero_with_an_exactly_zero_gradient_at_the_origin() -> None:
    zero = jnp.zeros(4, dtype=jnp.float32)
    assert float(safe_norm(zero)) == 0.0
    gradient = np.asarray(jax.grad(safe_norm)(zero))
    assert np.all(gradient == 0.0)


@pytest.mark.parametrize("vector", [[3.0, -4.0, 12.0], [1e-30, 2e-30, -2e-30], [1e30, -1e30, 0.0]])
def test_safe_norm_gradient_is_the_unit_vector(vector: list[float]) -> None:
    """The rescale is a stop-gradient power of two, so the gradient is ``v / ||v||`` at any scale."""
    v = jnp.asarray(vector, dtype=jnp.float32)
    expected = np.asarray(vector, dtype=np.float64) / np.linalg.norm(
        np.asarray(vector, dtype=np.float64)
    )
    np.testing.assert_allclose(np.asarray(jax.grad(safe_norm)(v)), expected, rtol=1e-6)


def test_safe_norm_reduces_over_the_axis_it_is_given() -> None:
    values = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    expected = np.linalg.norm(np.asarray(values, dtype=np.float64), axis=1)
    np.testing.assert_allclose(np.asarray(safe_norm(values, axis=1)), expected, rtol=1e-6)
    assert float(safe_norm(values)) == pytest.approx(
        float(np.linalg.norm(np.asarray(values))), rel=1e-6
    )


def test_safe_norm_passes_infinity_and_nan_through() -> None:
    assert np.isinf(float(safe_norm(jnp.asarray([jnp.inf, 1.0, 0.0]))))
    assert np.isnan(float(safe_norm(jnp.asarray([jnp.nan, 1.0, 0.0]))))


@pytest.mark.parametrize(
    ("p", "q"), [(3.0, 4.0), (1e-20, 1e-25), (0.0, 2.5), (-7.0, 0.0), (1e30, 1.0)]
)
def test_safe_hypot_matches_numpy_and_survives_extreme_legs(p: float, q: float) -> None:
    out = float(safe_hypot(jnp.asarray(p, dtype=jnp.float32), jnp.asarray(q, dtype=jnp.float32)))
    assert out == pytest.approx(float(np.hypot(p, q)), rel=1e-6)


def test_safe_hypot_is_exactly_zero_with_zero_gradients_at_the_origin() -> None:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    assert float(safe_hypot(zero, zero)) == 0.0
    gp, gq = jax.grad(safe_hypot, argnums=(0, 1))(zero, zero)
    assert float(gp) == 0.0
    assert float(gq) == 0.0


def test_floor_at_is_maximum_and_keeps_a_nan() -> None:
    values = jnp.asarray([-1.0, 0.5, 2.0, jnp.nan], dtype=jnp.float32)
    floored = np.asarray(floor_at(values, 1.0))
    np.testing.assert_array_equal(floored[:3], [1.0, 1.0, 2.0])
    assert np.isnan(floored[3])


@pytest.mark.parametrize("x", [1e20, -3e25, 1e30])
def test_safe_asinh_gradient_survives_the_float32_squaring_overflow(x: float) -> None:
    """jax's ``asinh`` derivative forms ``x * x``, which overflows past 1.8e19 and returns 0."""
    value = jnp.asarray(x, dtype=jnp.float32)
    assert float(jax.grad(jnp.arcsinh)(value)) == 0.0  # the upstream rule the wrapper corrects
    assert float(jax.grad(safe_asinh)(value)) == pytest.approx(1.0 / abs(x), rel=1e-6)
    assert float(safe_asinh(value)) == float(jnp.arcsinh(value))
