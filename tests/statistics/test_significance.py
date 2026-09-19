"""Tests for calibrax.statistics.significance module."""

from __future__ import annotations

import pytest

from calibrax.statistics.significance import (
    effect_size,
    mann_whitney_u,
    paired_significance_test,
    welch_t_test,
)


class TestWelchTTest:
    """Tests for welch_t_test."""

    def test_same_distribution_high_p(self) -> None:
        """Same distribution should give high p-value."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.1, 2.1, 3.1, 3.9, 4.9]
        _, p = welch_t_test(a, b)
        assert p > 0.5

    def test_different_distributions_low_p(self) -> None:
        """Very different distributions should give low p-value."""
        a = [1.0, 1.1, 1.0, 1.1, 1.0, 1.1, 1.0, 1.1]
        b = [100.0, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0, 100.1]
        _, p = welch_t_test(a, b)
        assert p < 0.01

    def test_returns_statistic_and_p(self) -> None:
        """Should return a (statistic, p_value) tuple."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        stat, p = welch_t_test(a, b)
        assert isinstance(stat, float)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0


class TestMannWhitneyU:
    """Tests for mann_whitney_u."""

    def test_same_data_high_p(self) -> None:
        """Same data should give high p-value."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        _, p = mann_whitney_u(data, data)
        assert p > 0.5

    def test_separated_distributions_low_p(self) -> None:
        """Completely separated distributions should give low p-value."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [100.0, 200.0, 300.0, 400.0, 500.0]
        _, p = mann_whitney_u(a, b)
        assert p < 0.05

    def test_returns_statistic_and_p(self) -> None:
        """Should return a (statistic, p_value) tuple."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        stat, p = mann_whitney_u(a, b)
        assert isinstance(stat, float)
        assert isinstance(p, float)


class TestPairedSignificanceTest:
    """Tests for paired_significance_test."""

    def test_identical_data_not_significant(self) -> None:
        """Identical paired data should not be significant."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = paired_significance_test(data, data)
        assert result.significant is False
        assert result.method == "wilcoxon"

    def test_shifted_data_significant(self) -> None:
        """Clearly shifted data should be significant."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        b = [x + 10.0 for x in a]
        result = paired_significance_test(a, b)
        assert result.significant is True

    def test_empty_raises_value_error(self) -> None:
        """Empty samples should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            paired_significance_test([], [1.0])

    def test_mismatched_lengths_raises_value_error(self) -> None:
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="equal lengths"):
            paired_significance_test([1.0, 2.0], [1.0])

    def test_has_effect_size(self) -> None:
        """Result should include effect size."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0]
        result = paired_significance_test(a, b)
        assert isinstance(result.effect_size, float)
        assert result.effect_size >= 0.0

    def test_p_value_range(self) -> None:
        """p-value should be between 0 and 1."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.5, 2.5, 3.5, 4.5, 5.5]
        result = paired_significance_test(a, b)
        assert 0.0 <= result.p_value <= 1.0

    def test_custom_alpha(self) -> None:
        """Custom alpha should affect significance flag."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        b = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        strict = paired_significance_test(a, b, alpha=0.001)
        lenient = paired_significance_test(a, b, alpha=0.99)
        # If strict is significant, lenient must be too
        if strict.significant:
            assert lenient.significant


class TestEffectSize:
    """Tests for effect_size."""

    def test_identical_samples_zero(self) -> None:
        """Identical samples should have effect size 0."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert effect_size(data, data) == pytest.approx(0.0)

    def test_known_cohens_d(self) -> None:
        """Known shifted samples should produce expected Cohen's d."""
        # Use varied samples to avoid pooled_std=0 when var=0
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [4.0, 5.0, 6.0, 7.0, 8.0]
        d = effect_size(a, b)
        assert d > 0.0
        # Both have same variance, shift of 3, std ~ 1.58
        # Cohen's d ~ 3/1.58 ~ 1.9
        assert d == pytest.approx(1.897, abs=0.1)

    def test_zero_variance(self) -> None:
        """Samples with zero variance should return 0."""
        a = [5.0, 5.0, 5.0]
        b = [5.0, 5.0, 5.0]
        assert effect_size(a, b) == pytest.approx(0.0)

    def test_non_negative(self) -> None:
        """Effect size should always be non-negative (absolute value)."""
        a = [10.0, 20.0, 30.0]
        b = [1.0, 2.0, 3.0]
        assert effect_size(a, b) >= 0.0
        assert effect_size(b, a) >= 0.0
