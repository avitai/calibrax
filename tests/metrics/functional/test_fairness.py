"""Tests for fairness evaluation metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.fairness import (
    demographic_parity_ratio,
    disparate_impact_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    group_metric_breakdown,
)


class TestDemographicParityRatio:
    """Tests for demographic_parity_ratio."""

    def test_perfect_parity(self) -> None:
        preds = jnp.array([1, 1, 0, 1, 1, 0])
        groups = jnp.array([0, 0, 0, 1, 1, 1])
        # Group 0: 2/3 positive, Group 1: 2/3 positive → ratio = 1.0
        assert demographic_parity_ratio(preds, groups) == pytest.approx(1.0, abs=1e-4)

    def test_complete_disparity(self) -> None:
        preds = jnp.array([1, 1, 1, 0, 0, 0])
        groups = jnp.array([0, 0, 0, 1, 1, 1])
        # Group 0: 3/3 positive, Group 1: 0/3 positive → ratio = 0.0
        assert demographic_parity_ratio(preds, groups) == pytest.approx(0.0, abs=1e-4)

    def test_known_value(self) -> None:
        preds = jnp.array([1, 1, 0, 0, 1, 0, 0, 0])
        groups = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Group 0: 2/4 = 0.5, Group 1: 1/4 = 0.25 → min(0.5, 0.25) ratio = 0.5
        result = demographic_parity_ratio(preds, groups)
        assert result == pytest.approx(0.5, abs=1e-4)

    def test_multi_group(self) -> None:
        preds = jnp.array([1, 0, 1, 0, 1, 1])
        groups = jnp.array([0, 0, 1, 1, 2, 2])
        result = demographic_parity_ratio(preds, groups)
        assert 0.0 <= result <= 1.0

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        preds = jnp.array([1, 0, 1, 0])
        groups = jnp.array([0, 0, 1, 1])
        result = demographic_parity_ratio(preds, groups)
        assert isinstance(result, jax.Array)


class TestEqualizedOddsDifference:
    """Tests for equalized_odds_difference."""

    def test_perfect_equalized_odds(self) -> None:
        preds = jnp.array([1, 1, 0, 1, 1, 0])
        targets = jnp.array([1, 1, 0, 1, 1, 0])
        groups = jnp.array([0, 0, 0, 1, 1, 1])
        assert equalized_odds_difference(preds, targets, groups) == pytest.approx(0.0, abs=1e-4)

    def test_tpr_disparity(self) -> None:
        # Group 0: TPR=1.0 (both positives predicted correctly)
        # Group 1: TPR=0.5 (one positive missed)
        preds = jnp.array([1, 1, 0, 1, 0, 0])
        targets = jnp.array([1, 1, 0, 1, 1, 0])
        groups = jnp.array([0, 0, 0, 1, 1, 1])
        result = equalized_odds_difference(preds, targets, groups)
        assert result > 0.0

    def test_known_value(self) -> None:
        # Group 0: TP=2, FP=0, FN=0, TN=1 → TPR=1.0, FPR=0.0
        # Group 1: TP=1, FP=0, FN=1, TN=1 → TPR=0.5, FPR=0.0
        preds = jnp.array([1, 1, 0, 1, 0, 0])
        targets = jnp.array([1, 1, 0, 1, 1, 0])
        groups = jnp.array([0, 0, 0, 1, 1, 1])
        result = equalized_odds_difference(preds, targets, groups)
        assert result == pytest.approx(0.5, abs=1e-4)

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        preds = jnp.array([1, 0, 1, 0])
        targets = jnp.array([1, 0, 1, 0])
        groups = jnp.array([0, 0, 1, 1])
        result = equalized_odds_difference(preds, targets, groups)
        assert isinstance(result, jax.Array)


class TestEqualOpportunityDifference:
    """Tests for equal_opportunity_difference."""

    def test_perfect_equal_opportunity(self) -> None:
        preds = jnp.array([1, 1, 0, 1, 1, 0])
        targets = jnp.array([1, 1, 0, 1, 1, 0])
        groups = jnp.array([0, 0, 0, 1, 1, 1])
        assert equal_opportunity_difference(preds, targets, groups) == pytest.approx(0.0, abs=1e-4)

    def test_known_value(self) -> None:
        # Same as equalized_odds test but only TPR matters
        preds = jnp.array([1, 1, 0, 1, 0, 0])
        targets = jnp.array([1, 1, 0, 1, 1, 0])
        groups = jnp.array([0, 0, 0, 1, 1, 1])
        result = equal_opportunity_difference(preds, targets, groups)
        assert result == pytest.approx(0.5, abs=1e-4)

    def test_bounded(self) -> None:
        preds = jnp.array([1, 0, 1, 0])
        targets = jnp.array([1, 1, 0, 0])
        groups = jnp.array([0, 0, 1, 1])
        result = equal_opportunity_difference(preds, targets, groups)
        assert -1e-5 <= result <= 1.0 + 1e-5


class TestDisparateImpactRatio:
    """Tests for disparate_impact_ratio."""

    def test_delegates_to_dpr(self) -> None:
        preds = jnp.array([1, 1, 0, 0, 1, 0])
        groups = jnp.array([0, 0, 0, 1, 1, 1])
        assert disparate_impact_ratio(preds, groups) == pytest.approx(
            demographic_parity_ratio(preds, groups), abs=1e-6
        )

    def test_eighty_percent_rule(self) -> None:
        # Group 0: 4/5 positive, Group 1: 2/5 positive
        # ratio = (2/5) / (4/5) = 0.5 < 0.8 → disparate impact
        preds = jnp.array([1, 1, 1, 1, 0, 1, 1, 0, 0, 0])
        groups = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        result = disparate_impact_ratio(preds, groups)
        assert result < 0.8


class TestGroupMetricBreakdown:
    """Tests for group_metric_breakdown."""

    def test_per_group_mse(self) -> None:
        from calibrax.metrics.functional.regression import mse

        preds = jnp.array([1.0, 2.0, 3.0, 4.0])
        targets = jnp.array([1.0, 2.0, 3.0, 4.0])
        groups = jnp.array([0, 0, 1, 1])
        result = group_metric_breakdown(mse, preds, targets, groups)
        assert result["0"] == pytest.approx(0.0, abs=1e-5)
        assert result["1"] == pytest.approx(0.0, abs=1e-5)

    def test_returns_dict(self) -> None:
        from calibrax.metrics.functional.regression import mse

        preds = jnp.array([1.0, 2.0, 3.0, 4.0])
        targets = jnp.array([1.0, 2.0, 3.0, 4.0])
        groups = jnp.array([0, 0, 1, 1])
        result = group_metric_breakdown(mse, preds, targets, groups)
        assert isinstance(result, dict)

    def test_all_groups_represented(self) -> None:
        from calibrax.metrics.functional.regression import mae

        preds = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        groups = jnp.array([0, 0, 1, 1, 2, 2])
        result = group_metric_breakdown(mae, preds, targets, groups)
        assert len(result) == 3
        assert "0" in result
        assert "1" in result
        assert "2" in result


class TestFairnessMetricRegistration:
    """Tests for fairness metric registration."""

    def test_all_registered(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        expected = [
            "demographic_parity_ratio",
            "equalized_odds_difference",
            "equal_opportunity_difference",
            "disparate_impact_ratio",
        ]
        for name in expected:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_fairness_domain(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        fairness_metrics = registry.list_by_domain("fairness")
        assert len(fairness_metrics) == 4

    def test_direction_assignments(self) -> None:
        from calibrax.core.models import MetricDirection
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        higher = ["demographic_parity_ratio", "disparate_impact_ratio"]
        lower = ["equalized_odds_difference", "equal_opportunity_difference"]
        for name in higher:
            assert registry.get(name).direction == MetricDirection.HIGHER
        for name in lower:
            assert registry.get(name).direction == MetricDirection.LOWER
