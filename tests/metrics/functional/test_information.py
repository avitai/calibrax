"""Tests for information-theoretic metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.information import (
    conditional_entropy,
    cross_entropy,
    entropy,
    fisher_information_matrix,
    mutual_information,
    normalized_mutual_information,
)


class TestEntropy:
    """Tests for entropy."""

    def test_uniform_distribution(self) -> None:
        # Maximum entropy = log(n)
        p = jnp.array([0.25, 0.25, 0.25, 0.25])
        assert entropy(p) == pytest.approx(float(jnp.log(4.0)), abs=1e-5)

    def test_deterministic_distribution(self) -> None:
        p = jnp.array([1.0, 0.0, 0.0])
        assert entropy(p) == pytest.approx(0.0, abs=1e-5)

    def test_non_negative(self) -> None:
        p = jnp.array([0.3, 0.7])
        assert entropy(p) >= -1e-6

    def test_binary_entropy(self) -> None:
        p = jnp.array([0.5, 0.5])
        assert entropy(p) == pytest.approx(float(jnp.log(2.0)), abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        p = jnp.array([0.5, 0.5])
        result = entropy(p)
        assert isinstance(result, jax.Array)


class TestCrossEntropy:
    """Tests for cross_entropy."""

    def test_same_distribution(self) -> None:
        p = jnp.array([0.3, 0.7])
        ce = cross_entropy(p, p)
        h = entropy(p)
        assert ce == pytest.approx(h, abs=1e-5)

    def test_always_gte_entropy(self) -> None:
        p = jnp.array([0.3, 0.7])
        q = jnp.array([0.6, 0.4])
        ce = cross_entropy(p, q)
        h = entropy(p)
        assert ce >= h - 1e-5

    def test_known_value(self) -> None:
        p = jnp.array([1.0, 0.0])
        q = jnp.array([0.5, 0.5])
        # -1.0*log(0.5) - 0.0*log(0.5) = log(2)
        assert cross_entropy(p, q) == pytest.approx(float(jnp.log(2.0)), abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        p = jnp.array([0.5, 0.5])
        result = cross_entropy(p, p)
        assert isinstance(result, jax.Array)


class TestMutualInformation:
    """Tests for mutual_information."""

    def test_independent_variables(self) -> None:
        # p(x,y) = p(x)*p(y) → MI = 0
        joint = jnp.array([[0.25, 0.25], [0.25, 0.25]])
        assert mutual_information(joint) == pytest.approx(0.0, abs=1e-5)

    def test_perfect_dependence(self) -> None:
        # Knowing X determines Y → MI = H(X) = H(Y)
        joint = jnp.array([[0.5, 0.0], [0.0, 0.5]])
        mi = mutual_information(joint)
        h_x = entropy(jnp.array([0.5, 0.5]))
        assert mi == pytest.approx(h_x, abs=1e-5)

    def test_non_negative(self) -> None:
        joint = jnp.array([[0.3, 0.1], [0.2, 0.4]])
        assert mutual_information(joint) >= -1e-6

    def test_returns_jax_scalar(self) -> None:
        joint = jnp.array([[0.25, 0.25], [0.25, 0.25]])
        result = mutual_information(joint)
        assert isinstance(result, jax.Array)


class TestConditionalEntropy:
    """Tests for conditional_entropy."""

    def test_independent_variables(self) -> None:
        # H(Y|X) = H(Y) when independent
        joint = jnp.array([[0.25, 0.25], [0.25, 0.25]])
        h_cond = conditional_entropy(joint)
        h_y = entropy(jnp.array([0.5, 0.5]))
        assert h_cond == pytest.approx(h_y, abs=1e-5)

    def test_perfect_dependence(self) -> None:
        # H(Y|X) = 0 when X determines Y
        joint = jnp.array([[0.5, 0.0], [0.0, 0.5]])
        assert conditional_entropy(joint) == pytest.approx(0.0, abs=1e-5)

    def test_chain_rule(self) -> None:
        # H(X,Y) = H(X) + H(Y|X)
        joint = jnp.array([[0.3, 0.1], [0.2, 0.4]])
        h_joint = entropy(joint.ravel())
        p_x = jnp.sum(joint, axis=1)
        h_x = entropy(p_x)
        h_y_given_x = conditional_entropy(joint)
        assert h_joint == pytest.approx(h_x + h_y_given_x, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        joint = jnp.array([[0.5, 0.0], [0.0, 0.5]])
        result = conditional_entropy(joint)
        assert isinstance(result, jax.Array)


class TestNormalizedMutualInformation:
    """Tests for normalized_mutual_information."""

    def test_independent_variables(self) -> None:
        joint = jnp.array([[0.25, 0.25], [0.25, 0.25]])
        assert normalized_mutual_information(joint) == pytest.approx(0.0, abs=1e-5)

    def test_perfect_dependence(self) -> None:
        joint = jnp.array([[0.5, 0.0], [0.0, 0.5]])
        assert normalized_mutual_information(joint) == pytest.approx(1.0, abs=1e-5)

    def test_bounded(self) -> None:
        joint = jnp.array([[0.3, 0.1], [0.2, 0.4]])
        result = normalized_mutual_information(joint)
        assert 0.0 <= result <= 1.0 + 1e-5

    def test_returns_jax_scalar(self) -> None:
        joint = jnp.array([[0.5, 0.0], [0.0, 0.5]])
        result = normalized_mutual_information(joint)
        assert isinstance(result, jax.Array)


class TestFisherInformationMatrix:
    """Tests for fisher_information_matrix."""

    def test_normal_distribution(self) -> None:
        # For N(mu, 1), Fisher info for mu is I(mu) = 1
        def log_prob(theta: jnp.ndarray) -> jnp.ndarray:
            return -0.5 * jnp.sum(theta**2)

        fim = fisher_information_matrix(log_prob, jnp.array([0.0]))
        assert float(fim[0, 0]) == pytest.approx(1.0, abs=1e-4)

    def test_returns_matrix(self) -> None:
        def log_prob(theta: jnp.ndarray) -> jnp.ndarray:
            return -0.5 * jnp.sum(theta**2)

        fim = fisher_information_matrix(log_prob, jnp.array([1.0, 2.0]))
        assert fim.shape == (2, 2)

    def test_positive_semidefinite(self) -> None:
        def log_prob(theta: jnp.ndarray) -> jnp.ndarray:
            return -0.5 * jnp.sum(theta**2)

        fim = fisher_information_matrix(log_prob, jnp.array([1.0, 2.0]))
        eigenvalues = jnp.linalg.eigvalsh(fim)
        assert jnp.all(eigenvalues >= -1e-6)

    def test_symmetric(self) -> None:
        def log_prob(theta: jnp.ndarray) -> jnp.ndarray:
            return -0.5 * (theta[0] ** 2 + 2 * theta[1] ** 2)

        fim = fisher_information_matrix(log_prob, jnp.array([1.0, 1.0]))
        assert jnp.allclose(fim, fim.T, atol=1e-6)


class TestInformationMetricRegistration:
    """Tests for information metric registration in MetricRegistry."""

    def test_information_metrics_registered(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        expected = [
            "entropy",
            "cross_entropy",
            "mutual_information",
            "conditional_entropy",
            "normalized_mutual_information",
        ]
        for name in expected:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_information_domain(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        info_metrics = registry.list_by_domain("information")
        assert len(info_metrics) == 5

    def test_entropy_direction_info(self) -> None:
        from calibrax.core.models import MetricDirection
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        assert registry.get("entropy").direction == MetricDirection.INFO

    def test_fisher_not_registered(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        assert not registry.has("fisher_information_matrix")
