"""``safe_root``: the ``order``-th root with a finite derivative at zero."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.metrics._utils import safe_norm, safe_root
from calibrax.metrics.functional.distance import mahalanobis_distance, minkowski_distance
from calibrax.metrics.functional.divergence import hellinger_distance
from calibrax.metrics.functional.geometric import rmsd
from calibrax.metrics.functional.regression import rmse


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


def test_a_negative_is_a_domain_error_and_yields_nan() -> None:
    """A negative radicand is out of contract, and now reads as one.

    This assertion is the reverse of what it was, because the contract changed and not
    because an implementation was accommodated. The evidence for the change: measured,
    ``numpy.sqrt(-4.0)`` is ``nan`` with "invalid value encountered in sqrt",
    ``jax.numpy.sqrt(-4.0)`` is ``nan``, ``scipy.spatial.distance.mahalanobis`` with an
    indefinite inverse-covariance is ``nan`` (scipy 1.17.0), and
    ``sklearn.metrics.pairwise_distances(metric="mahalanobis")`` with that matrix is ``nan``
    (1.8.0). None of the three raises and none clamps. Clamping was calibrax's own
    invention, shared with nothing.

    Raising is not available to a traceable primitive: a Python branch on the value is
    ``TracerBoolConversionError`` under ``jax.jit`` and ``jax.vmap``, and ``checkify`` would
    change the return to ``(err, value)`` across all 24 call sites.
    """
    assert bool(jnp.all(jnp.isnan(safe_root(jnp.array([-4.0, -1e-7, -1e-30])))))
    assert bool(jnp.isnan(jax.grad(safe_root)(jnp.asarray(-4.0))))
    assert bool(jnp.isnan(safe_root(jnp.asarray(-27.0), order=3.0)))


def test_zero_is_the_only_guarded_point() -> None:
    """The guard is the singularity and nothing else -- including negative zero.

    The root's derivative is infinite at zero, which is the one case needing a branch. IEEE
    makes ``-0.0 == 0.0`` true, so negative zero is the same guarded point rather than a
    negative.
    """
    np.testing.assert_array_equal(np.asarray(safe_root(jnp.array([0.0, -0.0]))), [0.0, 0.0])
    assert float(jax.grad(safe_root)(jnp.asarray(-0.0))) == 0.0
    mixed = safe_root(jnp.array([0.0, 4.0, -4.0, 9.0]))
    np.testing.assert_array_equal(np.asarray(jnp.isnan(mixed)), [False, False, True, False])
    np.testing.assert_allclose(np.asarray(mixed)[[0, 1, 3]], [0.0, 2.0, 3.0], rtol=1e-6)


def test_only_mahalanobis_can_reach_the_domain_error() -> None:
    """The blast radius, as a standing check rather than a one-off measurement.

    Of the nine direct ``safe_root`` call sites, only ``mahalanobis_distance`` forms a
    radicand that can go negative, and only when the caller's precision matrix is not
    positive semi-definite. A PSD matrix never does: identical points give exactly zero and
    near-identical points give a small *positive* value.
    """
    a = jnp.array([1.0, 0.0, 0.0])
    b = jnp.array([0.0, 1.0, 0.0])
    psd = jnp.eye(3) * 2.0
    indefinite = jnp.diag(jnp.array([1.0, -3.0, 1.0]))

    assert float(mahalanobis_distance(a, b, precision_matrix=psd)) == pytest.approx(2.0)
    assert float(mahalanobis_distance(a, a, precision_matrix=psd)) == 0.0
    assert bool(jnp.isnan(mahalanobis_distance(a, b, precision_matrix=indefinite)))

    # Every other direct caller passes a sum of squares, an absolute value raised to a
    # power, or a value it has already clamped -- so each stays finite on ordinary input.
    assert jnp.isfinite(rmse(jnp.array([1.0, 2.0]), jnp.array([1.5, 2.5])))
    assert jnp.isfinite(rmsd(jnp.zeros((3, 3)), jnp.ones((3, 3))))
    assert jnp.isfinite(hellinger_distance(jnp.array([0.5, 0.5]), jnp.array([0.4, 0.6])))
    assert jnp.isfinite(minkowski_distance(a, b, p=3.0))


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


#: A spread of magnitudes rather than U(0, 1): whether eager ``pow`` and ``sqrt`` land on the
#: same float32 is decided by the mantissa, so one narrow band under-samples the disagreement.
def _magnitude_grid() -> jnp.ndarray:
    """20,000 positive float32 values across twelve orders of magnitude."""
    rng = np.random.default_rng(0)
    spread = np.concatenate([rng.random(8_000), rng.random(6_000) * 1e6, rng.random(6_000) * 1e-6])
    return jnp.asarray(spread, dtype=jnp.float32)


def _bits(values: jnp.ndarray) -> np.ndarray:
    """The float32 bit patterns, so a one-ULP difference cannot hide in a tolerance."""
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def test_order_two_is_bit_identical_to_jnp_sqrt() -> None:
    """At order 2 the root is ``jnp.sqrt``, to the bit.

    ``x ** 0.5`` is not. Measured over this grid it lands one ULP away on 12 of 20,000
    values, and over a million it is 691 -- every one of them exactly one ULP, counted as
    the distance between the bit patterns rather than against a float64 spacing.
    """
    grid = _magnitude_grid()
    np.testing.assert_array_equal(_bits(safe_root(grid)), _bits(jnp.sqrt(grid)))


def test_order_two_agrees_with_its_own_jitted_value() -> None:
    """The defect a caller meets: the same call returning two different floats.

    Jitted ``pow(x, 0.5)`` and jitted ``sqrt(x)`` are the same float32 on every one of
    1,000,000 inputs tried, so the compiled path always produced the square root while the
    eager path produced a power. Eager and jitted therefore disagreed on 12 of these 20,000
    values, which is what breaks an ``abs=0.0`` comparison between a jitted metric and an
    eager reference. The agreement happens below HLO: a ``sqrt`` module and a
    ``pow(x, 0.5)`` module are two different graphs that compute the same float32.
    """
    grid = _magnitude_grid()
    np.testing.assert_array_equal(_bits(safe_root(grid)), _bits(jax.jit(safe_root)(grid)))


def test_a_negative_infinity_is_a_domain_error_like_any_other_negative() -> None:
    """``-inf`` is negative, so the contract makes it ``NaN``.

    ``pow`` disagrees: IEEE 754 defines ``pow(-inf, 0.5)`` as ``+inf``, so a negative
    radicand of infinite magnitude read as *infinitely far apart* rather than as the domain
    error every smaller negative already produced. ``jnp.sqrt(-inf)`` is ``NaN``, as is
    ``numpy.sqrt(-inf)``.

    Reachable from one place, measured: ``mahalanobis_distance`` with a precision matrix
    that is not positive semi-definite and differences large enough to overflow forms a
    radicand of ``-inf``. Every other call site forms ``+inf`` or ``NaN`` there instead, and
    ``sqrt`` and ``pow`` agree on both.
    """
    assert bool(jnp.isnan(safe_root(jnp.asarray(-jnp.inf))))
    assert float(safe_root(jnp.asarray(jnp.inf))) == float(jnp.inf)

    big = jnp.asarray(1e38, dtype=jnp.float32)
    a = jnp.array([big, 0.0, 0.0])
    b = jnp.array([-big, 0.0, 0.0])
    indefinite = jnp.diag(jnp.array([-1.0, 1.0, 1.0]))
    assert float(jnp.dot(a - b, jnp.dot(indefinite, a - b))) == float(-jnp.inf)
    assert bool(jnp.isnan(mahalanobis_distance(a, b, precision_matrix=indefinite)))


def test_negative_zero_is_the_one_input_where_the_guard_overrides_the_root() -> None:
    """``jnp.sqrt(-0.0)`` keeps the sign; the guarded root does not, by construction.

    IEEE makes ``-0.0 == 0.0`` true, so negative zero is the guarded singularity and leaves
    as ``+0.0``. This is the only input at which the order-2 root is not ``jnp.sqrt`` bit for
    bit, and it is the guard doing its job rather than an accident of the root used.
    """
    assert _bits(jnp.sqrt(jnp.asarray(-0.0)))[()] == 0x8000_0000
    assert _bits(safe_root(jnp.asarray(-0.0)))[()] == 0x0000_0000
    assert float(jax.grad(safe_root)(jnp.asarray(-0.0))) == 0.0


def test_an_integer_order_of_two_takes_the_square_root_path() -> None:
    """``order=2`` and ``order=2.0`` are the same order, so they must agree to the bit."""
    grid = _magnitude_grid()
    np.testing.assert_array_equal(
        _bits(safe_root(grid, order=2)), _bits(safe_root(grid, order=2.0))
    )


@pytest.mark.parametrize("order", [1.0, 3.0, 2.5])
def test_orders_other_than_two_keep_the_general_path(order: float) -> None:
    """Only order 2 changes: every other order is still the power it always was, to the bit."""
    grid = _magnitude_grid()
    np.testing.assert_array_equal(_bits(safe_root(grid, order=order)), _bits(grid ** (1.0 / order)))


def test_a_traced_order_is_not_branched_on() -> None:
    """``order`` may arrive as a tracer, and a Python branch on one is an error.

    ``minkowski_distance`` and ``sliced_wasserstein`` both forward a caller's ``p`` here, so
    ``jit`` over that argument makes ``order`` a tracer. The order-2 shortcut is therefore
    chosen from the static Python value only; a traced order takes the general power path,
    which is what it took before the shortcut existed.
    """
    traced = jax.jit(lambda x, order: safe_root(x, order=order))
    np.testing.assert_allclose(np.asarray(traced(jnp.asarray(4.0), jnp.asarray(2.0))), 2.0)
    np.testing.assert_allclose(
        np.asarray(traced(jnp.asarray(27.0), jnp.asarray(3.0))), 3.0, rtol=1e-6
    )
    assert float(jax.jit(jax.grad(lambda x: traced(x, jnp.asarray(2.0))))(jnp.asarray(0.0))) == 0.0
