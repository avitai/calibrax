"""``safe_root``: the ``order``-th root with a finite derivative at zero."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.metrics._utils import safe_norm, safe_root


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


def test_a_nan_propagates_rather_than_becoming_zero() -> None:
    """The predicate selects the degenerate branch, so a ``NaN`` reaches the root.

    Written the other way round -- selecting the root branch on ``x > 0.0`` -- a ``NaN``
    fails that comparison too and leaves as ``0.0``. Measured at ``9b0a951``, that turned
    missing data into a perfect score in four metrics: ``rmse``, ``rmsd`` and
    ``pairwise_rmsd`` returned ``0.0``, and ``relative_l2_error`` a plausible ``0.681925``.

    This is the convention the docstring cites rather than a new one: measured on optax
    0.2.8, ``safe_norm([nan, 1.0], 0.0)`` is ``nan`` while ``safe_norm([0.0, 0.0], 0.0)``
    is ``0.0`` with a zero gradient, because ``norm <= min_norm`` is ``False`` for ``NaN``.
    """
    assert bool(jnp.isnan(safe_root(jnp.asarray(jnp.nan))))
    assert bool(jnp.isnan(safe_root(jnp.asarray(jnp.nan), order=3.0)))
    mixed = safe_root(jnp.array([0.0, 4.0, jnp.nan, 9.0]))
    np.testing.assert_array_equal(np.asarray(jnp.isnan(mixed)), [False, False, True, False])
    np.testing.assert_allclose(np.asarray(mixed)[[0, 1, 3]], [0.0, 2.0, 3.0], rtol=1e-6)


def test_a_negative_is_still_taken_to_zero() -> None:
    """Deliberately unchanged, and not an oversight.

    ``mmd`` and ``hellinger_distance`` clamp with ``jnp.maximum(..., 0.0)`` before calling,
    because cancellation puts their sums slightly below zero, and ``mahalanobis_distance``
    forms ``diff @ prec @ diff`` from a caller-supplied matrix that need not be positive
    semi-definite. Whether an out-of-contract negative should raise or propagate is a
    separate question from whether a ``NaN`` should survive.
    """
    np.testing.assert_allclose(
        np.asarray(safe_root(jnp.array([-4.0, -1e-7, 0.0]))), [0.0, 0.0, 0.0]
    )
    assert float(jax.grad(safe_root)(jnp.asarray(-4.0))) == 0.0


def test_the_zero_gradient_survives_every_transformation() -> None:
    """The reason the function exists: ``jax.grad`` of a distance at a perfect match."""
    zero = jnp.asarray(0.0)
    assert float(jax.grad(safe_root)(zero)) == 0.0
    assert float(jax.jit(jax.grad(safe_root))(zero)) == 0.0
    assert float(jax.jacfwd(safe_root)(zero)) == 0.0
    batched = jax.vmap(jax.grad(safe_root))(jnp.array([0.0, 4.0]))
    np.testing.assert_allclose(np.asarray(batched), [0.0, 0.25], rtol=1e-6)
    # Second order still works where the root is differentiable twice.
    assert float(jax.grad(jax.grad(safe_root))(jnp.asarray(4.0))) == pytest.approx(-0.03125)


def test_safe_norm_matches_the_convention_it_cites() -> None:
    """Three properties, the same three ``optax.safe_norm`` has (measured on 0.2.8)."""
    assert float(safe_norm(jnp.zeros(3))) == 0.0
    np.testing.assert_allclose(np.asarray(jax.grad(safe_norm)(jnp.zeros(3))), [0.0, 0.0, 0.0])
    assert float(safe_norm(jnp.array([3.0, 4.0]))) == pytest.approx(5.0)
    assert bool(jnp.isnan(safe_norm(jnp.array([jnp.nan, 1.0]))))
