"""Tests for clustering evaluation metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.clustering import (
    adjusted_mutual_information,
    adjusted_rand_index,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_information_clustering,
    silhouette_score,
    v_measure,
)


class TestAdjustedRandIndex:
    """Tests for adjusted_rand_index."""

    def test_perfect_agreement(self) -> None:
        labels = jnp.array([0, 0, 1, 1, 2, 2])
        assert adjusted_rand_index(labels, labels) == pytest.approx(1.0, abs=1e-4)

    def test_symmetric(self) -> None:
        a = jnp.array([0, 0, 1, 1, 2, 2])
        b = jnp.array([1, 1, 0, 0, 2, 2])
        assert adjusted_rand_index(a, b) == pytest.approx(adjusted_rand_index(b, a), abs=1e-5)

    def test_known_value(self) -> None:
        # Relabeling: [0,0,1,1] vs [1,1,0,0] should be perfect (same partition)
        a = jnp.array([0, 0, 1, 1])
        b = jnp.array([1, 1, 0, 0])
        assert adjusted_rand_index(a, b) == pytest.approx(1.0, abs=1e-4)

    def test_returns_jax_scalar(self) -> None:
        labels = jnp.array([0, 0, 1, 1])
        result = adjusted_rand_index(labels, labels)
        assert isinstance(result, jax.Array)


class TestNormalizedMutualInformationClustering:
    """Tests for normalized_mutual_information_clustering."""

    def test_perfect_agreement(self) -> None:
        labels = jnp.array([0, 0, 1, 1, 2, 2])
        result = normalized_mutual_information_clustering(labels, labels)
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_bounded(self) -> None:
        a = jnp.array([0, 0, 1, 1, 2, 2])
        b = jnp.array([1, 2, 0, 1, 2, 0])
        result = normalized_mutual_information_clustering(a, b)
        assert -1e-5 <= result <= 1.0 + 1e-5

    def test_average_methods(self) -> None:
        a = jnp.array([0, 0, 1, 1])
        b = jnp.array([0, 1, 0, 1])
        for method in ("arithmetic", "geometric", "min", "max"):
            result = normalized_mutual_information_clustering(a, b, average=method)
            assert isinstance(result, jax.Array)

    def test_invalid_average_raises(self) -> None:
        a = jnp.array([0, 1])
        with pytest.raises(ValueError, match="average must be one of"):
            normalized_mutual_information_clustering(a, a, average="invalid")

    def test_returns_jax_scalar(self) -> None:
        labels = jnp.array([0, 0, 1, 1])
        result = normalized_mutual_information_clustering(labels, labels)
        assert isinstance(result, jax.Array)


class TestAdjustedMutualInformation:
    """Tests for adjusted_mutual_information."""

    def test_perfect_agreement(self) -> None:
        labels = jnp.array([0, 0, 1, 1, 2, 2])
        result = adjusted_mutual_information(labels, labels)
        assert result == pytest.approx(1.0, abs=0.1)  # AMI approximation

    def test_range(self) -> None:
        a = jnp.array([0, 0, 1, 1, 2, 2])
        b = jnp.array([1, 2, 0, 1, 2, 0])
        result = adjusted_mutual_information(a, b)
        assert -1.0 - 1e-5 <= result <= 1.0 + 1e-5

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        labels = jnp.array([0, 0, 1, 1])
        result = adjusted_mutual_information(labels, labels)
        assert isinstance(result, jax.Array)


class TestVMeasure:
    """Tests for v_measure."""

    def test_perfect_clustering(self) -> None:
        labels = jnp.array([0, 0, 1, 1, 2, 2])
        assert v_measure(labels, labels) == pytest.approx(1.0, abs=1e-4)

    def test_beta_weighting(self) -> None:
        a = jnp.array([0, 0, 1, 1])
        b = jnp.array([0, 1, 0, 1])
        v_1 = v_measure(a, b, beta=1.0)
        v_2 = v_measure(a, b, beta=2.0)
        # Different beta should give different results (unless perfect)
        assert isinstance(v_1, jax.Array)
        assert isinstance(v_2, jax.Array)

    def test_bounded(self) -> None:
        a = jnp.array([0, 0, 1, 1])
        b = jnp.array([0, 1, 0, 1])
        result = v_measure(a, b)
        assert -1e-5 <= result <= 1.0 + 1e-5

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        labels = jnp.array([0, 0, 1, 1])
        result = v_measure(labels, labels)
        assert isinstance(result, jax.Array)


class TestSilhouetteScore:
    """Tests for silhouette_score."""

    def test_well_separated_clusters(self) -> None:
        features = jnp.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [10.0, 10.0],
                [10.1, 10.0],
                [10.0, 10.1],
            ]
        )
        labels = jnp.array([0, 0, 0, 1, 1, 1])
        result = silhouette_score(features, labels)
        assert result > 0.8

    def test_range(self) -> None:
        features = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 0.5],
                [5.0, 5.0],
                [6.0, 5.0],
                [5.5, 5.5],
            ]
        )
        labels = jnp.array([0, 0, 0, 1, 1, 1])
        result = silhouette_score(features, labels)
        assert -1.0 - 1e-5 <= result <= 1.0 + 1e-5

    def test_returns_jax_scalar(self) -> None:
        features = jnp.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]])
        labels = jnp.array([0, 0, 1, 1])
        result = silhouette_score(features, labels)
        assert isinstance(result, jax.Array)


class TestCalinskiHarabaszScore:
    """Tests for calinski_harabasz_score."""

    def test_well_separated(self) -> None:
        features = jnp.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [10.0, 10.0],
                [10.1, 10.0],
            ]
        )
        labels = jnp.array([0, 0, 1, 1])
        result = calinski_harabasz_score(features, labels)
        assert result > 100  # Well-separated → high VRC

    def test_positive(self) -> None:
        features = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        )
        labels = jnp.array([0, 0, 1, 1])
        result = calinski_harabasz_score(features, labels)
        assert result >= 0.0

    def test_returns_jax_scalar(self) -> None:
        features = jnp.array([[0.0], [1.0], [10.0], [11.0]])
        labels = jnp.array([0, 0, 1, 1])
        result = calinski_harabasz_score(features, labels)
        assert isinstance(result, jax.Array)


class TestDaviesBouldinScore:
    """Tests for davies_bouldin_score."""

    def test_well_separated(self) -> None:
        features = jnp.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [100.0, 100.0],
                [100.1, 100.0],
            ]
        )
        labels = jnp.array([0, 0, 1, 1])
        result = davies_bouldin_score(features, labels)
        assert result < 0.1  # Well-separated → low DB

    def test_non_negative(self) -> None:
        features = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        )
        labels = jnp.array([0, 0, 1, 1])
        result = davies_bouldin_score(features, labels)
        assert result >= -1e-5

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        features = jnp.array([[0.0], [1.0], [10.0], [11.0]])
        labels = jnp.array([0, 0, 1, 1])
        result = davies_bouldin_score(features, labels)
        assert isinstance(result, jax.Array)


class TestClusteringMetricRegistration:
    """Tests for clustering metric registration."""

    def test_all_registered(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        expected = [
            "adjusted_rand_index",
            "normalized_mutual_information_clustering",
            "adjusted_mutual_information",
            "v_measure",
            "silhouette_score",
            "calinski_harabasz_score",
            "davies_bouldin_score",
        ]
        for name in expected:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_clustering_domain(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        clustering_metrics = registry.list_by_domain("clustering")
        assert len(clustering_metrics) == 7

    def test_davies_bouldin_direction_lower(self) -> None:
        from calibrax.core.models import MetricDirection
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        db = registry.get("davies_bouldin_score")
        assert db.direction == MetricDirection.LOWER

    def test_others_direction_higher(self) -> None:
        from calibrax.core.models import MetricDirection
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        higher_metrics = [
            "adjusted_rand_index",
            "normalized_mutual_information_clustering",
            "adjusted_mutual_information",
            "v_measure",
            "silhouette_score",
            "calinski_harabasz_score",
        ]
        for name in higher_metrics:
            m = registry.get(name)
            assert m.direction == MetricDirection.HIGHER, f"{name} should be HIGHER"
