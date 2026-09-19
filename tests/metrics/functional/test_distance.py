"""Tests for distance metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
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

    def test_identical_points(self) -> None:
        a = jnp.array([0.3, 0.2])
        assert poincare_distance(a, a) == pytest.approx(0.0, abs=1e-4)

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

    def test_identical_points(self) -> None:
        # Point on hyperboloid: x_0 = sqrt(1 + ||x_rest||^2)
        a = jnp.array([jnp.sqrt(2.0), 1.0, 0.0])
        assert lorentz_distance(a, a) == pytest.approx(0.0, abs=1e-3)

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
