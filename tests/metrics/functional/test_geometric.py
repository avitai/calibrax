"""Tests for geometric and point cloud metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.geometric import (
    chamfer_distance,
    directed_hausdorff,
    earth_movers_distance_1d,
    hausdorff_distance,
)


class TestChamferDistance:
    """Tests for chamfer_distance."""

    def test_identical_sets(self) -> None:
        pts = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = chamfer_distance(pts, pts)
        assert result == pytest.approx(0.0, abs=1e-3)

    def test_known_value(self) -> None:
        a = jnp.array([[0.0, 0.0]])
        b = jnp.array([[1.0, 0.0]])
        result = chamfer_distance(a, b)
        assert result == pytest.approx(1.0, abs=1e-3)

    def test_symmetry(self) -> None:
        a = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        b = jnp.array([[0.5, 0.5], [1.5, 0.5]])
        assert chamfer_distance(a, b) == pytest.approx(chamfer_distance(b, a), abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        pts = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        result = chamfer_distance(pts, pts)
        assert isinstance(result, jax.Array)


class TestEarthMoversDistance1D:
    """Tests for earth_movers_distance_1d."""

    def test_identical(self) -> None:
        a = jnp.array([1.0, 2.0, 3.0])
        result = earth_movers_distance_1d(a, a)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_known_shift(self) -> None:
        a = jnp.array([0.0, 1.0, 2.0])
        b = jnp.array([1.0, 2.0, 3.0])
        result = earth_movers_distance_1d(a, b)
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_delegates_to_wasserstein(self) -> None:
        from calibrax.metrics.functional.divergence import wasserstein_1d

        a = jnp.array([1.0, 3.0, 5.0])
        b = jnp.array([2.0, 4.0, 6.0])
        assert earth_movers_distance_1d(a, b) == pytest.approx(wasserstein_1d(a, b), abs=1e-6)


class TestDirectedHausdorff:
    """Tests for directed_hausdorff."""

    def test_identical_sets(self) -> None:
        pts = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        result = directed_hausdorff(pts, pts)
        assert result == pytest.approx(0.0, abs=1e-3)

    def test_known_value(self) -> None:
        a = jnp.array([[0.0, 0.0], [3.0, 0.0]])
        b = jnp.array([[0.0, 0.0]])
        # Point [3,0] is 3.0 from [0,0] → directed Hausdorff = 3.0
        result = directed_hausdorff(a, b)
        assert result == pytest.approx(3.0, abs=1e-3)

    def test_asymmetric(self) -> None:
        a = jnp.array([[0.0, 0.0], [5.0, 0.0]])
        b = jnp.array([[0.0, 0.0]])
        # DH(A→B) = 5.0 (farthest point in A from nearest in B)
        # DH(B→A) = 0.0 (all points in B have nearest in A at 0)
        dh_ab = directed_hausdorff(a, b)
        dh_ba = directed_hausdorff(b, a)
        assert dh_ab != pytest.approx(dh_ba, abs=0.1)

    def test_returns_jax_scalar(self) -> None:
        pts = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        result = directed_hausdorff(pts, pts)
        assert isinstance(result, jax.Array)


class TestHausdorffDistance:
    """Tests for hausdorff_distance."""

    def test_identical_sets(self) -> None:
        pts = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        result = hausdorff_distance(pts, pts)
        assert result == pytest.approx(0.0, abs=1e-3)

    def test_known_value(self) -> None:
        a = jnp.array([[0.0, 0.0], [3.0, 0.0]])
        b = jnp.array([[0.0, 0.0]])
        result = hausdorff_distance(a, b)
        assert result == pytest.approx(3.0, abs=1e-3)

    def test_symmetry(self) -> None:
        a = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        b = jnp.array([[0.5, 0.5]])
        assert hausdorff_distance(a, b) == pytest.approx(hausdorff_distance(b, a), abs=1e-5)

    def test_greater_than_chamfer(self) -> None:
        a = jnp.array([[0.0, 0.0], [5.0, 0.0]])
        b = jnp.array([[0.0, 0.0], [2.0, 0.0]])
        hd = hausdorff_distance(a, b)
        cd = chamfer_distance(a, b)
        assert hd >= cd - 1e-5

    def test_delegates_to_directed(self) -> None:
        a = jnp.array([[0.0, 0.0], [3.0, 0.0]])
        b = jnp.array([[1.0, 0.0]])
        expected = max(directed_hausdorff(a, b), directed_hausdorff(b, a))
        assert hausdorff_distance(a, b) == pytest.approx(expected, abs=1e-6)


class TestGeometricMetricRegistration:
    """Tests for geometric metric registration."""

    def test_all_registered(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        expected = [
            "chamfer_distance",
            "earth_movers_distance_1d",
            "directed_hausdorff",
            "hausdorff_distance",
        ]
        for name in expected:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_geometric_domain(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        geo_metrics = registry.list_by_domain("geometric")
        assert len(geo_metrics) == 4

    def test_hausdorff_is_true_metric(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        hd = registry.get("hausdorff_distance")
        assert hd.properties.is_true_metric is True
        assert hd.properties.is_symmetric is True

    def test_directed_hausdorff_asymmetric(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        dh = registry.get("directed_hausdorff")
        assert dh.properties.is_symmetric is False
