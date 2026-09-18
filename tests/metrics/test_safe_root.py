"""``safe_root``: the ``order``-th root with a finite derivative at zero."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.metrics._utils import safe_root


@pytest.mark.parametrize("order", [1.0, 2.0, 3.0])
def test_values_match_the_plain_root(order: float) -> None:
    values = jnp.array([0.0, 0.25, 4.0, 27.0])
    np.testing.assert_allclose(
        np.asarray(safe_root(values, order=order)), np.asarray(values) ** (1.0 / order), rtol=1e-6
    )


@pytest.mark.parametrize("order", [2.0, 3.0])
def test_the_derivative_at_zero_is_zero_and_elsewhere_the_root_s(order: float) -> None:
    """The plain root's derivative is infinite at 0, which ``jax.grad`` turns into NaN."""
    plain = jax.grad(lambda x: x ** (1.0 / order))(0.0)
    assert not bool(jnp.isfinite(plain))

    assert float(jax.grad(lambda x: safe_root(x, order=order))(0.0)) == 0.0
    at_four = float(jax.grad(lambda x: safe_root(x, order=order))(4.0))
    assert at_four == pytest.approx((1.0 / order) * 4.0 ** (1.0 / order - 1.0))


def test_jit_and_vmap() -> None:
    values = jnp.array([[0.0, 1.0], [4.0, 9.0]])
    jitted = jax.jit(safe_root)(values)
    mapped = jax.vmap(safe_root)(values)
    np.testing.assert_allclose(np.asarray(jitted), [[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(np.asarray(mapped), [[0.0, 1.0], [2.0, 3.0]])
