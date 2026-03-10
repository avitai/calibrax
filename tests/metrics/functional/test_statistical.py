"""Tests for statistical correlation metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.statistical import (
    concordance_correlation,
    kendall_tau,
    pearson_correlation,
    r_squared_adjusted,
    spearman_rank_correlation,
)


class TestPearsonCorrelation:
    """Tests for pearson_correlation."""

    def test_perfect_positive(self) -> None:
        a = jnp.array([1.0, 2.0, 3.0, 4.0])
        assert pearson_correlation(a, a) == pytest.approx(1.0, abs=1e-5)

    def test_perfect_negative(self) -> None:
        a = jnp.array([1.0, 2.0, 3.0, 4.0])
        b = jnp.array([4.0, 3.0, 2.0, 1.0])
        assert pearson_correlation(a, b) == pytest.approx(-1.0, abs=1e-5)

    def test_uncorrelated(self) -> None:
        a = jnp.array([1.0, -1.0, 1.0, -1.0])
        b = jnp.array([1.0, 1.0, -1.0, -1.0])
        assert pearson_correlation(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        a = jnp.array([1.0, 2.0, 3.0])
        result = pearson_correlation(a, a)
        assert isinstance(result, jax.Array)


class TestSpearmanRankCorrelation:
    """Tests for spearman_rank_correlation."""

    def test_perfect_monotonic(self) -> None:
        a = jnp.array([1.0, 2.0, 3.0, 4.0])
        b = jnp.array([10.0, 20.0, 30.0, 40.0])
        assert spearman_rank_correlation(a, b) == pytest.approx(1.0, abs=1e-5)

    def test_perfect_inverse(self) -> None:
        a = jnp.array([1.0, 2.0, 3.0, 4.0])
        b = jnp.array([40.0, 30.0, 20.0, 10.0])
        assert spearman_rank_correlation(a, b) == pytest.approx(-1.0, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        a = jnp.array([1.0, 2.0, 3.0])
        result = spearman_rank_correlation(a, a)
        assert isinstance(result, jax.Array)


class TestKendallTau:
    """Tests for kendall_tau."""

    def test_perfect_agreement(self) -> None:
        a = jnp.array([1.0, 2.0, 3.0, 4.0])
        assert kendall_tau(a, a) == pytest.approx(1.0, abs=1e-5)

    def test_perfect_disagreement(self) -> None:
        a = jnp.array([1.0, 2.0, 3.0, 4.0])
        b = jnp.array([4.0, 3.0, 2.0, 1.0])
        assert kendall_tau(a, b) == pytest.approx(-1.0, abs=1e-5)

    def test_known_value(self) -> None:
        # [1,2,3] vs [1,3,2]: concordant=(1,2),(1,3)=2, discordant=(2,3)=1
        # tau = (2-1)/3 = 1/3
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([1.0, 3.0, 2.0])
        assert kendall_tau(a, b) == pytest.approx(1.0 / 3.0, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        a = jnp.array([1.0, 2.0])
        result = kendall_tau(a, a)
        assert isinstance(result, jax.Array)


class TestConcordanceCorrelation:
    """Tests for concordance_correlation."""

    def test_perfect_agreement(self) -> None:
        a = jnp.array([1.0, 2.0, 3.0, 4.0])
        assert concordance_correlation(a, a) == pytest.approx(1.0, abs=1e-5)

    def test_high_correlation_low_agreement(self) -> None:
        # Perfect correlation but shifted → CCC < 1
        a = jnp.array([1.0, 2.0, 3.0, 4.0])
        b = jnp.array([11.0, 12.0, 13.0, 14.0])  # shifted by 10
        ccc = concordance_correlation(a, b)
        r = pearson_correlation(a, b)
        assert r == pytest.approx(1.0, abs=1e-5)
        assert ccc < r

    def test_range(self) -> None:
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([3.0, 1.0, 2.0])
        result = concordance_correlation(a, b)
        assert -1.0 - 1e-5 <= result <= 1.0 + 1e-5

    def test_returns_jax_scalar(self) -> None:
        a = jnp.array([1.0, 2.0])
        result = concordance_correlation(a, a)
        assert isinstance(result, jax.Array)


class TestRSquaredAdjusted:
    """Tests for r_squared_adjusted."""

    def test_known_value(self) -> None:
        predictions = jnp.array([1.1, 2.1, 2.9, 4.0])
        targets = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = r_squared_adjusted(predictions, targets, num_predictors=1)
        assert result > 0.9

    def test_penalty_for_predictors(self) -> None:
        predictions = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r2_1 = r_squared_adjusted(predictions, targets, num_predictors=1)
        r2_3 = r_squared_adjusted(predictions, targets, num_predictors=3)
        # More predictors → lower adjusted R² (penalizes complexity)
        assert r2_1 >= r2_3 - 1e-5

    def test_returns_jax_scalar(self) -> None:
        predictions = jnp.array([1.0, 2.0])
        targets = jnp.array([1.0, 2.0])
        result = r_squared_adjusted(predictions, targets, num_predictors=1)
        assert isinstance(result, jax.Array)


class TestStatisticalMetricRegistration:
    """Tests for statistical metric registration."""

    def test_all_registered(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        expected = [
            "pearson_correlation",
            "spearman_rank_correlation",
            "kendall_tau",
            "concordance_correlation",
            "r_squared_adjusted",
        ]
        for name in expected:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_statistical_domain(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        stat_metrics = registry.list_by_domain("statistical")
        assert len(stat_metrics) == 5

    def test_all_direction_higher(self) -> None:
        from calibrax.core.models import MetricDirection
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        for m in registry.list_by_domain("statistical"):
            assert m.direction == MetricDirection.HIGHER
