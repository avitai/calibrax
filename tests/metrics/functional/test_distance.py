"""Tests for distance metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

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
    """Tests for randers_distance."""

    def test_zero_drift_equals_euclidean(self) -> None:
        a = jnp.array([1.0, 0.0])
        b = jnp.array([0.0, 1.0])
        drift = jnp.array([0.0, 0.0])
        randers = randers_distance(a, b, drift=drift)
        euclid = euclidean_distance(a, b)
        assert randers == pytest.approx(euclid, abs=1e-5)

    def test_asymmetric(self) -> None:
        a = jnp.array([0.0, 0.0])
        b = jnp.array([1.0, 0.0])
        drift = jnp.array([0.5, 0.0])
        d_ab = randers_distance(a, b, drift=drift)
        d_ba = randers_distance(b, a, drift=drift)
        assert d_ab != pytest.approx(d_ba, abs=1e-3)

    def test_with_drift_positive(self) -> None:
        a = jnp.array([0.0, 0.0])
        b = jnp.array([1.0, 0.0])
        drift = jnp.array([0.5, 0.0])
        # Traveling with drift costs more (drift adds to distance)
        d_with = randers_distance(a, b, drift=drift)
        d_euclid = euclidean_distance(a, b)
        # d_R = ||b-a|| + <drift, b-a> = 1 + 0.5 = 1.5
        assert d_with == pytest.approx(1.5, abs=1e-5)
        assert d_with > d_euclid

    def test_against_drift_costs_less(self) -> None:
        a = jnp.array([1.0, 0.0])
        b = jnp.array([0.0, 0.0])
        drift = jnp.array([0.5, 0.0])
        # b - a = (-1, 0), <drift, b-a> = -0.5
        # d_R = 1 + (-0.5) = 0.5
        d_against = randers_distance(a, b, drift=drift)
        assert d_against == pytest.approx(0.5, abs=1e-5)

    def test_subsonic_validation(self) -> None:
        a = jnp.array([0.0, 0.0])
        b = jnp.array([1.0, 0.0])
        drift = jnp.array([1.0, 0.0])  # ||drift|| = 1.0 → invalid
        with pytest.raises(ValueError, match="Sub-sonic condition"):
            randers_distance(a, b, drift=drift)

    def test_always_positive(self) -> None:
        a = jnp.array([0.0, 0.0])
        b = jnp.array([1.0, 0.5])
        drift = jnp.array([0.3, -0.2])
        result = randers_distance(a, b, drift=drift)
        assert result > 0.0

    def test_returns_jax_scalar(self) -> None:
        a = jnp.array([0.0, 0.0])
        b = jnp.array([1.0, 0.0])
        drift = jnp.array([0.0, 0.0])
        result = randers_distance(a, b, drift=drift)
        assert isinstance(result, jax.Array)


class TestDistanceMetricRegistration:
    """Tests for distance metric registration in MetricRegistry."""

    def test_all_distance_metrics_registered(self) -> None:
        from calibrax.metrics import MetricRegistry

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
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        distance_metrics = registry.list_by_domain("distance")
        assert len(distance_metrics) == 11

    def test_all_direction_lower(self) -> None:
        from calibrax.core.models import MetricDirection
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        distance_metrics = registry.list_by_domain("distance")
        for m in distance_metrics:
            assert m.direction == MetricDirection.LOWER

    def test_true_metric_flags(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        # True metrics
        for name in ["euclidean_distance", "manhattan_distance", "poincare_distance"]:
            assert registry.get(name).properties.is_true_metric is True
        # Not true metrics (cosine violates triangle, randers is asymmetric)
        for name in ["cosine_distance", "randers_distance"]:
            assert registry.get(name).properties.is_true_metric is False

    def test_randers_not_symmetric(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        assert registry.get("randers_distance").properties.is_symmetric is False

    def test_invariance_queries(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        rotation_invariant = registry.list_by_invariance("rotation")
        names = {m.name for m in rotation_invariant}
        assert "euclidean_distance" in names
        assert "manhattan_distance" not in names
