"""Tests for statistical correlation metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.statistical import (
    autocorrelation,
    concordance_correlation,
    correlation_preservation,
    kendall_tau,
    pearson_correlation,
    r_squared_adjusted,
    skewness,
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
            "skewness",
        ]
        for name in expected:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_statistical_domain(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        stat_metrics = registry.list_by_domain("statistical")
        assert len(stat_metrics) == 6

    def test_correlations_are_higher_and_skewness_is_informational(self) -> None:
        from calibrax.core.models import MetricDirection
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        for m in registry.list_by_domain("statistical"):
            expected = MetricDirection.INFO if m.name == "skewness" else MetricDirection.HIGHER
            assert m.direction == expected, m.name


class TestCorrelationPreservation:
    """Tests for correlation_preservation."""

    def test_identical_matrices_score_one(self) -> None:
        real = jnp.array([[1.0, 2.0, 3.0], [2.0, 4.1, 5.9], [3.0, 5.9, 9.2], [4.0, 8.2, 12.0]])
        assert correlation_preservation(real, real) == pytest.approx(1.0, abs=1e-5)

    def test_flipped_correlation_scores_low(self) -> None:
        real = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        generated = jnp.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
        assert correlation_preservation(real, generated) == pytest.approx(0.0, abs=1e-5)

    def test_constant_feature_yields_a_finite_score(self) -> None:
        real = jnp.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        generated = jnp.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
        assert bool(jnp.isfinite(correlation_preservation(real, generated)))

    def test_mismatched_features_raise(self) -> None:
        with pytest.raises(ValueError, match="feature dimension"):
            correlation_preservation(jnp.ones((4, 2)), jnp.ones((4, 3)))


class TestAutocorrelation:
    """Tests for autocorrelation."""

    def test_lag_zero_is_one_and_length_is_max_lag(self) -> None:
        series = jnp.sin(jnp.linspace(0.0, 12.0, 64)).reshape(1, 64, 1)
        result = autocorrelation(series, max_lag=8)
        assert result.shape == (8,)
        assert result[0] == pytest.approx(1.0, abs=1e-5)

    def test_periodic_series_recovers_its_period(self) -> None:
        period = 8
        series = jnp.tile(jnp.sin(jnp.linspace(0.0, 2 * jnp.pi, period, endpoint=False)), 16)
        result = autocorrelation(series.reshape(1, -1, 1), max_lag=period + 1)
        assert result[period] == pytest.approx(1.0, abs=1e-2)

    def test_constant_series_is_finite(self) -> None:
        result = autocorrelation(jnp.ones((2, 16, 3)), max_lag=4)
        assert bool(jnp.all(jnp.isfinite(result)))


class TestSkewness:
    """Tests for skewness."""

    def test_symmetric_data_has_zero_skew(self) -> None:
        assert skewness(jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])) == pytest.approx(0.0, abs=1e-6)

    def test_right_tail_is_positive(self) -> None:
        assert float(skewness(jnp.array([0.0, 0.0, 0.0, 1.0, 10.0]))) > 0.0

    def test_constant_data_is_zero(self) -> None:
        assert skewness(jnp.full((6,), 3.0)) == pytest.approx(0.0, abs=1e-6)
