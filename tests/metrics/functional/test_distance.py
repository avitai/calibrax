"""Tests for distance metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from substrax.testing import TraceCounter

from calibrax.core.models import MetricDirection
from calibrax.metrics import MetricRegistry
from calibrax.metrics.functional.distance import (
    chebyshev_distance,
    cosine_distance,
    euclidean_distance,
    hamming_distance,
    jaccard_distance,
    lorentz_distance,
    mahalanobis_distance,
    manhattan_distance,
    minkowski_distance,
    poincare_distance,
    randers_distance,
)


class TestCosineDistance:
    """Tests for cosine_distance."""

    def test_identical_vectors(self) -> None:
        a = jnp.array([1.0, 2.0, 3.0])
        assert cosine_distance(a, a) == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_vectors(self) -> None:
        a = jnp.array([1.0, 0.0])
        b = jnp.array([0.0, 1.0])
        assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-5)

    def test_opposite_vectors(self) -> None:
        a = jnp.array([1.0, 0.0])
        b = jnp.array([-1.0, 0.0])
        assert cosine_distance(a, b) == pytest.approx(2.0, abs=1e-5)

    def test_batch_input(self) -> None:
        a = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        b = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        assert cosine_distance(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        a = jnp.array([1.0, 0.0])
        result = cosine_distance(a, a)
        assert isinstance(result, jax.Array)


class TestEuclideanDistance:
    """Tests for euclidean_distance."""

    def test_identical_vectors(self) -> None:
        a = jnp.array([1.0, 2.0, 3.0])
        assert euclidean_distance(a, a) == pytest.approx(0.0, abs=1e-5)

    def test_known_value(self) -> None:
        a = jnp.array([1.0, 0.0])
        b = jnp.array([0.0, 1.0])
        assert euclidean_distance(a, b) == pytest.approx(jnp.sqrt(2.0), abs=1e-5)

    def test_triangle_inequality(self) -> None:
        a = jnp.array([0.0, 0.0])
        b = jnp.array([1.0, 0.0])
        c = jnp.array([0.0, 1.0])
        d_ac = euclidean_distance(a, c)
        d_ab = euclidean_distance(a, b)
        d_bc = euclidean_distance(b, c)
        assert d_ac <= d_ab + d_bc + 1e-6

    def test_returns_jax_scalar(self) -> None:
        a = jnp.array([1.0, 0.0])
        result = euclidean_distance(a, a)
        assert isinstance(result, jax.Array)


class TestManhattanDistance:
    """Tests for manhattan_distance."""

    def test_identical_vectors(self) -> None:
        a = jnp.array([1.0, 2.0])
        assert manhattan_distance(a, a) == pytest.approx(0.0, abs=1e-5)

    def test_known_value(self) -> None:
        a = jnp.array([1.0, 0.0])
        b = jnp.array([0.0, 1.0])
        assert manhattan_distance(a, b) == pytest.approx(2.0, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        a = jnp.array([1.0, 0.0])
        result = manhattan_distance(a, a)
        assert isinstance(result, jax.Array)


class TestChebyshevDistance:
    """Tests for chebyshev_distance."""

    def test_identical_vectors(self) -> None:
        a = jnp.array([1.0, 2.0, 3.0])
        assert chebyshev_distance(a, a) == pytest.approx(0.0, abs=1e-5)

    def test_known_value(self) -> None:
        a = jnp.array([1.0, 5.0, 3.0])
        b = jnp.array([2.0, 1.0, 3.0])
        # max(|1-2|, |5-1|, |3-3|) = 4
        assert chebyshev_distance(a, b) == pytest.approx(4.0, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        a = jnp.array([1.0, 0.0])
        result = chebyshev_distance(a, a)
        assert isinstance(result, jax.Array)


class TestMahalanobisDistance:
    """Tests for mahalanobis_distance."""

    def test_identity_equals_euclidean(self) -> None:
        a = jnp.array([1.0, 0.0])
        b = jnp.array([0.0, 1.0])
        mahal = mahalanobis_distance(a, b)
        euclid = euclidean_distance(a, b)
        assert mahal == pytest.approx(euclid, abs=1e-5)

    def test_known_value(self) -> None:
        a = jnp.array([1.0, 0.0])
        b = jnp.array([0.0, 1.0])
        # Precision matrix scales dimension 0 by 4
        prec = jnp.array([[4.0, 0.0], [0.0, 1.0]])
        # sqrt((1)^2*4 + (-1)^2*1) = sqrt(5)
        result = mahalanobis_distance(a, b, precision_matrix=prec)
        assert result == pytest.approx(float(jnp.sqrt(5.0)), abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        a = jnp.array([1.0, 0.0])
        result = mahalanobis_distance(a, a)
        assert isinstance(result, jax.Array)


class TestHammingDistance:
    """Tests for hamming_distance."""

    def test_identical(self) -> None:
        a = jnp.array([1, 0, 1, 1])
        assert hamming_distance(a, a) == pytest.approx(0.0, abs=1e-5)

    def test_completely_different(self) -> None:
        a = jnp.array([1, 1, 1])
        b = jnp.array([0, 0, 0])
        assert hamming_distance(a, b) == pytest.approx(1.0, abs=1e-5)

    def test_known_value(self) -> None:
        a = jnp.array([1, 0, 1])
        b = jnp.array([1, 1, 1])
        # 1 of 3 differ
        assert hamming_distance(a, b) == pytest.approx(1.0 / 3.0, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        a = jnp.array([1, 0])
        result = hamming_distance(a, a)
        assert isinstance(result, jax.Array)


class TestMinkowskiDistance:
    """Tests for minkowski_distance."""

    def test_p1_equals_manhattan(self) -> None:
        a = jnp.array([1.0, 0.0, 3.0])
        b = jnp.array([0.0, 2.0, 1.0])
        mink = minkowski_distance(a, b, p=1.0)
        manh = manhattan_distance(a, b)
        assert mink == pytest.approx(manh, abs=1e-5)

    def test_p2_equals_euclidean(self) -> None:
        a = jnp.array([1.0, 0.0, 3.0])
        b = jnp.array([0.0, 2.0, 1.0])
        mink = minkowski_distance(a, b, p=2.0)
        euclid = euclidean_distance(a, b)
        assert mink == pytest.approx(euclid, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        a = jnp.array([1.0, 0.0])
        result = minkowski_distance(a, a, p=3.0)
        assert isinstance(result, jax.Array)


class TestJaccardDistance:
    """Tests for jaccard_distance."""

    def test_identical_sets(self) -> None:
        a = jnp.array([1, 1, 0, 1])
        assert jaccard_distance(a, a) == pytest.approx(0.0, abs=1e-5)

    def test_disjoint_sets(self) -> None:
        a = jnp.array([1, 1, 0, 0])
        b = jnp.array([0, 0, 1, 1])
        assert jaccard_distance(a, b) == pytest.approx(1.0, abs=1e-5)

    def test_known_value(self) -> None:
        # intersection=1, union=3 → J=1/3 → distance=2/3
        a = jnp.array([1, 1, 0])
        b = jnp.array([0, 1, 1])
        assert jaccard_distance(a, b) == pytest.approx(2.0 / 3.0, abs=1e-5)

    def test_symmetric(self) -> None:
        a = jnp.array([1, 1, 0, 1])
        b = jnp.array([0, 1, 1, 0])
        assert jaccard_distance(a, b) == pytest.approx(jaccard_distance(b, a), abs=1e-6)

    def test_empty_sets(self) -> None:
        a = jnp.array([0, 0, 0])
        b = jnp.array([0, 0, 0])
        # Both empty → safe_divide handles 0/0
        result = jaccard_distance(a, b)
        assert isinstance(result, jax.Array)

    def test_returns_jax_scalar(self) -> None:
        a = jnp.array([1, 0])
        result = jaccard_distance(a, a)
        assert isinstance(result, jax.Array)


class TestPoincareDistance:
    """Tests for poincare_distance."""

    @pytest.mark.parametrize("point", [[0.0, 0.0], [0.3, 0.2], [0.6, -0.7]])
    def test_identical_points_are_exactly_zero_with_a_zero_gradient(
        self, point: list[float]
    ) -> None:
        """hyperbolix's contract (MIT, 96aba05): ``d(x, x)`` is 0 with the direction-free gradient 0.

        The distance is a cone at ``x = y``, so no derivative exists there; ``safe_root``'s
        convention supplies 0 where ``arccosh``'s infinite derivative at 1 gave NaN.
        """
        x = jnp.asarray(point, dtype=jnp.float32)
        assert float(poincare_distance(x, x)) == 0.0
        for gradient in (
            jax.jit(jax.grad(lambda p: poincare_distance(p, p)))(x),
            jax.jit(jax.grad(lambda p: poincare_distance(p, x)))(x),
            jax.jit(jax.grad(lambda q: poincare_distance(x, q)))(x),
        ):
            assert np.all(np.asarray(gradient) == 0.0)

    @pytest.mark.parametrize("separation", [1e-3, 1e-5, 1e-7])
    def test_a_close_pair_keeps_its_float32_digits(self, separation: float) -> None:
        """``arccosh(1 + 2t)`` rounds the ``2t`` of a close pair into the leading 1 and read 0.

        The oracle is the distance of the stored float32 coordinates in float64. The ``asinh``
        form rounds a handful of times (the difference, its squared norm and root, each
        ``1 - c||x||^2``, their product and root, a quotient, ``asinh``), each within ``eps``
        relative, so ``8 eps`` bounds its relative error.
        """
        x = jnp.asarray([0.3, 0.1], dtype=jnp.float32)
        y = x + jnp.asarray([separation, 0.0], dtype=jnp.float32)
        x64, y64 = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        t = np.sum((x64 - y64) ** 2) / ((1.0 - x64 @ x64) * (1.0 - y64 @ y64))
        expected = 2.0 * np.arcsinh(np.sqrt(t))
        eps = float(jnp.finfo(jnp.float32).eps)
        assert float(poincare_distance(x, y)) == pytest.approx(expected, rel=8 * eps)

    @pytest.mark.parametrize("c", [0.3, 1.0])
    @pytest.mark.parametrize("a", [7.2, 9.0])
    def test_a_far_pair_is_float32_accurate(self, c: float, a: float) -> None:
        """Ported from hyperbolix (MIT, 96aba05, ``tests/test_manifolds.py``), as it stands.

        ``x = r e`` and ``y = -x``, each at scaled radius ``a``, so ``sqrt(c) d = 2a`` (14.4 and
        18, inside the float32 chart). The oracle is the stored coordinates' exact distance
        ``4 atanh(sqrt(c) ||x||) / sqrt(c)`` and its gradient ``lambda_x x_hat`` in float64; the
        tolerances, ``1e-4`` relative on the value and ``1e-3`` on the gradient, are hyperbolix's.
        """
        e = np.ones(8) / np.sqrt(8.0)
        x = jnp.asarray((np.tanh(a / 2.0) / np.sqrt(c) * e).astype(np.float32))
        y = -x
        x64 = np.asarray(x, dtype=np.float64)
        r = float(np.linalg.norm(x64))
        expected = 4.0 * np.arctanh(np.sqrt(c) * r) / np.sqrt(c)
        expected_gradient = 2.0 / (1.0 - c * r * r) * x64 / r

        assert abs(float(poincare_distance(x, y, curvature=c)) - expected) <= 1e-4 * expected
        gradient = np.asarray(
            jax.grad(lambda p: poincare_distance(p, y, curvature=c))(x), dtype=np.float64
        )
        error = np.linalg.norm(gradient - expected_gradient)
        assert error <= 1e-3 * np.linalg.norm(expected_gradient)

    def test_origin_to_point(self) -> None:
        origin = jnp.array([0.0, 0.0])
        point = jnp.array([0.5, 0.0])
        result = poincare_distance(origin, point)
        # Known: d(0, r) = arccosh(1 + 2r^2/(1-r^2))
        # r=0.5: arccosh(1 + 2*0.25/0.75) = arccosh(1 + 2/3) = arccosh(5/3)
        expected = float(jnp.arccosh(5.0 / 3.0))
        assert result == pytest.approx(expected, abs=1e-4)

    def test_symmetric(self) -> None:
        a = jnp.array([0.1, 0.2])
        b = jnp.array([0.3, -0.1])
        assert poincare_distance(a, b) == pytest.approx(poincare_distance(b, a), abs=1e-5)

    def test_triangle_inequality(self) -> None:
        a = jnp.array([0.1, 0.0])
        b = jnp.array([0.0, 0.2])
        c = jnp.array([-0.1, 0.1])
        d_ac = poincare_distance(a, c)
        d_ab = poincare_distance(a, b)
        d_bc = poincare_distance(b, c)
        assert d_ac <= d_ab + d_bc + 1e-5

    def test_diverges_near_boundary(self) -> None:
        origin = jnp.array([0.0, 0.0])
        near = jnp.array([0.5, 0.0])
        far = jnp.array([0.99, 0.0])
        d_near = poincare_distance(origin, near)
        d_far = poincare_distance(origin, far)
        assert d_far > d_near

    def test_curvature_parameter(self) -> None:
        a = jnp.array([0.0, 0.0])
        b = jnp.array([0.3, 0.0])
        d1 = poincare_distance(a, b, curvature=1.0)
        d2 = poincare_distance(a, b, curvature=2.0)
        # Different curvatures produce different distances
        assert d1 != pytest.approx(d2, abs=1e-3)

    def test_returns_jax_scalar(self) -> None:
        a = jnp.array([0.1, 0.2])
        result = poincare_distance(a, a)
        assert isinstance(result, jax.Array)


class TestLorentzDistance:
    """Tests for lorentz_distance."""

    @pytest.mark.parametrize("spatial", [[0.0, 0.0], [1.0, 0.0], [0.3, -0.7], [3e4, 1e4]])
    def test_identical_points_are_exactly_zero_with_a_zero_gradient(
        self, spatial: list[float]
    ) -> None:
        """The docstring's "0.0 = identical points", with no floor and a finite gradient.

        hyperbolix's contract: one point given as both arguments has distance exactly 0 and an
        exactly-zero gradient. Against a separately computed copy, the two spatial directions can
        round an ulp apart inside ``jax.grad``, and at the cone point that picks a unit
        subgradient, so there the gradient is finite rather than zero.
        """
        s = jnp.asarray(spatial, dtype=jnp.float32)
        a = jnp.concatenate([jnp.sqrt(1.0 + jnp.sum(s**2))[None], s])
        assert float(lorentz_distance(a, a)) == 0.0
        to_itself = np.asarray(jax.grad(lambda x: lorentz_distance(x, x))(a))
        assert np.all(to_itself == 0.0)
        one_sided = np.asarray(jax.grad(lambda x: lorentz_distance(x, a))(a))
        assert np.all(np.isfinite(one_sided))

    @pytest.mark.parametrize("radius", [9.0, 20.0, 30.0, 43.0])
    def test_distance_along_a_radial_geodesic_is_the_radius_difference(self, radius: float) -> None:
        """Ported from hyperbolix (MIT, 96aba05, ``tests/test_manifolds.py``), its float32 arm.

        ``y``'s spatial part is ``x``'s scaled by an exact 2, so both lie on one geodesic ray
        through the origin and their distance is the difference of their radii, which the
        ``arccosh(-<x, y>)`` form loses to cancellation past radius ~10 in float32. The tolerance,
        ``4e-3`` absolute and relative, is hyperbolix's for this test in float32.
        """
        direction = np.array([1.0, 2.0, -2.0], dtype=np.float64) / 3.0
        x_s = (np.sinh(radius) * direction).astype(np.float32)
        y_s = (2.0 * x_s).astype(np.float32)
        x_time = np.sqrt(1.0 + x_s.astype(np.float64) @ x_s.astype(np.float64))
        y_time = np.sqrt(1.0 + y_s.astype(np.float64) @ y_s.astype(np.float64))
        x = jnp.asarray(np.concatenate([[x_time], x_s]).astype(np.float32))
        y = jnp.asarray(np.concatenate([[y_time], y_s]).astype(np.float32))
        expected = np.arcsinh(np.linalg.norm(y_s.astype(np.float64))) - np.arcsinh(
            np.linalg.norm(x_s.astype(np.float64))
        )
        assert float(lorentz_distance(x, y)) == pytest.approx(expected, abs=4e-3, rel=4e-3)

    def test_symmetry_is_exact(self) -> None:
        a = jnp.array([jnp.sqrt(1.0 + 0.34), 0.5, 0.3])
        b = jnp.array([jnp.sqrt(1.0 + 1.01), -0.1, 1.0])
        assert float(lorentz_distance(a, b)) == float(lorentz_distance(b, a))

    def test_known_value(self) -> None:
        # Two points on hyperboloid
        a = jnp.array([1.0, 0.0, 0.0])  # origin
        # b at spatial coord (1,0): x_0 = sqrt(1+1) = sqrt(2)
        b = jnp.array([jnp.sqrt(2.0), 1.0, 0.0])
        # -inner = -(-1*sqrt(2) + 0) = sqrt(2)
        # arccosh(sqrt(2)) ≈ 0.8814
        result = lorentz_distance(a, b)
        expected = float(jnp.arccosh(jnp.sqrt(2.0)))
        assert result == pytest.approx(expected, abs=1e-3)

    def test_symmetric(self) -> None:
        a = jnp.array([jnp.sqrt(2.0), 1.0, 0.0])
        b = jnp.array([jnp.sqrt(1.25), 0.5, 0.0])
        assert lorentz_distance(a, b) == pytest.approx(lorentz_distance(b, a), abs=1e-5)

    def test_triangle_inequality(self) -> None:
        a = jnp.array([1.0, 0.0, 0.0])
        b = jnp.array([jnp.sqrt(2.0), 1.0, 0.0])
        c = jnp.array([jnp.sqrt(2.0), 0.0, 1.0])
        d_ac = lorentz_distance(a, c)
        d_ab = lorentz_distance(a, b)
        d_bc = lorentz_distance(b, c)
        assert d_ac <= d_ab + d_bc + 1e-5

    def test_returns_jax_scalar(self) -> None:
        a = jnp.array([1.0, 0.0])
        result = lorentz_distance(a, a)
        assert isinstance(result, jax.Array)


class TestRandersDistance:
    """``d(a, b) = ||b - a|| + magnitude * <u, b - a>`` with ``u`` the unit ``direction``.

    The drift is ``magnitude * u``; ``magnitude < 1`` keeps the distance positive (Randers
    1941), and it is a Python float checked on the host, as in Finsler MDS (Dages et al. 2025).
    """

    def test_zero_magnitude_is_euclidean(self) -> None:
        a = jnp.array([0.0, 0.0])
        b = jnp.array([3.0, 4.0])
        result = randers_distance(a, b, direction=jnp.array([1.0, 0.0]), magnitude=0.0)
        assert float(result) == pytest.approx(5.0)

    def test_the_value_along_and_against_the_drift(self) -> None:
        a = jnp.array([0.0, 0.0])
        b = jnp.array([1.0, 0.0])
        direction = jnp.array([2.0, 0.0])  # normalised to (1, 0)
        assert float(randers_distance(a, b, direction=direction, magnitude=0.5)) == pytest.approx(
            1.5
        )
        assert float(randers_distance(b, a, direction=direction, magnitude=0.5)) == pytest.approx(
            0.5
        )

    def test_only_the_direction_of_the_direction_matters(self) -> None:
        a = jnp.array([0.0, 1.0, 2.0])
        b = jnp.array([1.0, -1.0, 0.5])
        direction = jnp.array([0.3, -0.4, 1.2])
        once = randers_distance(a, b, direction=direction, magnitude=0.4)
        scaled = randers_distance(a, b, direction=7.0 * direction, magnitude=0.4)
        assert float(once) == pytest.approx(float(scaled), rel=1e-6)

    def test_a_zero_direction_gives_the_euclidean_distance(self) -> None:
        a = jnp.array([0.0, 0.0])
        b = jnp.array([3.0, 4.0])
        result = randers_distance(a, b, direction=jnp.zeros(2), magnitude=0.5)
        assert float(result) == pytest.approx(5.0)

    @pytest.mark.parametrize("magnitude", [1.0, 1.5, -0.1])
    def test_a_magnitude_outside_zero_to_one_is_refused(self, magnitude: float) -> None:
        with pytest.raises(ValueError, match="magnitude"):
            randers_distance(jnp.zeros(2), jnp.ones(2), direction=jnp.ones(2), magnitude=magnitude)

    def test_an_array_magnitude_is_refused(self) -> None:
        with pytest.raises(TypeError, match="Python float"):
            randers_distance(
                jnp.zeros(2),
                jnp.ones(2),
                direction=jnp.ones(2),
                # The wrong type on purpose: the runtime check is what is tested.
                magnitude=jnp.array(0.5),  # pyright: ignore[reportArgumentType]
            )

    def test_positive_for_any_pair_below_the_bound(self) -> None:
        points = jax.random.normal(jax.random.key(0), (64, 3))
        others = jax.random.normal(jax.random.key(1), (64, 3))
        per_pair = jax.vmap(
            lambda x, y: randers_distance(
                x, y, direction=jnp.array([1.0, -2.0, 0.5]), magnitude=0.99
            )
        )(points, others)
        assert bool(jnp.all(per_pair > 0.0))

    def test_a_batch_is_the_mean_of_its_pairs(self) -> None:
        a = jax.random.normal(jax.random.key(2), (5, 3))
        b = jax.random.normal(jax.random.key(3), (5, 3))
        direction = jnp.array([0.0, 1.0, 0.0])
        batch = randers_distance(a, b, direction=direction, magnitude=0.3)
        pairs = [
            randers_distance(x, y, direction=direction, magnitude=0.3)
            for x, y in zip(a, b, strict=True)
        ]
        assert float(batch) == pytest.approx(float(jnp.mean(jnp.stack(pairs))), rel=1e-6)

    def test_jit_traces_once_with_a_traced_direction(self) -> None:
        counter = TraceCounter()
        compiled = jax.jit(counter.wrap(randers_distance), static_argnames=("magnitude",))
        a = jnp.zeros(3)
        b = jnp.array([1.0, 2.0, 0.5])
        with counter.expect(new_traces=1):
            first = compiled(a, b, direction=jnp.array([1.0, 0.0, 0.0]), magnitude=0.5)
        with counter.expect(new_traces=0):
            compiled(a, b * 2.0, direction=jnp.array([0.0, 1.0, 0.0]), magnitude=0.5)
        eager = randers_distance(a, b, direction=jnp.array([1.0, 0.0, 0.0]), magnitude=0.5)
        assert float(first) == pytest.approx(float(eager), rel=1e-6)

    def test_gradient_is_finite_at_identity_and_with_respect_to_the_direction(self) -> None:
        a = jnp.array([0.3, -1.2, 2.0])
        direction = jnp.array([0.1, 0.0, -0.2])
        at_identity = jax.grad(
            lambda x: randers_distance(a, x, direction=direction, magnitude=0.5)
        )(a)
        wrt_direction = jax.grad(
            lambda d: randers_distance(a, a + 1.0, direction=d, magnitude=0.5)
        )(direction)
        assert bool(jnp.all(jnp.isfinite(at_identity)))
        assert bool(jnp.all(jnp.isfinite(wrt_direction)))

    def test_returns_jax_scalar(self) -> None:
        result = randers_distance(
            jnp.zeros(2), jnp.array([1.0, 0.0]), direction=jnp.ones(2), magnitude=0.0
        )
        assert isinstance(result, jax.Array)
        assert result.shape == ()


class TestDistanceMetricRegistration:
    """Tests for distance metric registration in MetricRegistry."""

    def test_all_distance_metrics_registered(self) -> None:
        registry = MetricRegistry()
        expected = [
            "cosine_distance",
            "euclidean_distance",
            "manhattan_distance",
            "chebyshev_distance",
            "mahalanobis_distance",
            "hamming_distance",
            "minkowski_distance",
            "jaccard_distance",
            "poincare_distance",
            "lorentz_distance",
            "randers_distance",
        ]
        for name in expected:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_distance_domain(self) -> None:
        registry = MetricRegistry()
        distance_metrics = registry.list_by_domain("distance")
        assert len(distance_metrics) == 11

    def test_all_direction_lower(self) -> None:
        registry = MetricRegistry()
        distance_metrics = registry.list_by_domain("distance")
        for m in distance_metrics:
            assert m.direction == MetricDirection.LOWER

    def test_true_metric_flags(self) -> None:
        registry = MetricRegistry()
        # True metrics
        for name in ["euclidean_distance", "manhattan_distance", "poincare_distance"]:
            assert registry.get(name).properties.is_true_metric is True
        # Not true metrics (cosine violates triangle, randers is asymmetric)
        for name in ["cosine_distance", "randers_distance"]:
            assert registry.get(name).properties.is_true_metric is False

    def test_randers_not_symmetric(self) -> None:
        registry = MetricRegistry()
        assert registry.get("randers_distance").properties.is_symmetric is False

    def test_invariance_queries(self) -> None:
        registry = MetricRegistry()
        rotation_invariant = registry.list_by_invariance("rotation")
        names = {m.name for m in rotation_invariant}
        assert "euclidean_distance" in names
        assert "manhattan_distance" not in names
