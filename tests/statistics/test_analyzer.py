"""Tests for calibrax.statistics.analyzer module."""

from __future__ import annotations

import pytest

from calibrax.statistics.analyzer import (
    STABILITY_CV_THRESHOLD,
    StatisticalAnalyzer,
    StatisticalResult,
)


@pytest.fixture
def analyzer() -> StatisticalAnalyzer:
    """Create a StatisticalAnalyzer with fixed seed."""
    return StatisticalAnalyzer(bootstrap_resamples=500, seed=42)


class TestStatisticalResult:
    """Tests for StatisticalResult dataclass."""

    def test_frozen(self) -> None:
        """StatisticalResult should be immutable."""
        result = StatisticalResult(
            mean=1.0,
            median=1.0,
            std=0.1,
            min=0.9,
            max=1.1,
            cv=0.1,
            ci_lower=0.95,
            ci_upper=1.05,
            n=10,
            is_stable=True,
        )
        with pytest.raises(AttributeError):
            result.mean = 2.0  # type: ignore[misc]

    def test_to_dict_from_dict_round_trip(self) -> None:
        """to_dict/from_dict should preserve all fields."""
        original = StatisticalResult(
            mean=5.0,
            median=4.5,
            std=1.2,
            min=3.0,
            max=8.0,
            cv=0.24,
            ci_lower=4.0,
            ci_upper=6.0,
            n=20,
            is_stable=False,
        )
        reconstructed = StatisticalResult.from_dict(original.to_dict())
        assert reconstructed.mean == original.mean
        assert reconstructed.median == original.median
        assert reconstructed.std == original.std
        assert reconstructed.min == original.min
        assert reconstructed.max == original.max
        assert reconstructed.cv == original.cv
        assert reconstructed.ci_lower == original.ci_lower
        assert reconstructed.ci_upper == original.ci_upper
        assert reconstructed.n == original.n
        assert reconstructed.is_stable == original.is_stable


class TestSummarize:
    """Tests for StatisticalAnalyzer.summarize."""

    def test_known_data(self, analyzer: StatisticalAnalyzer) -> None:
        """summarize() should compute correct statistics for known data."""
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = analyzer.summarize(samples)
        assert result.mean == pytest.approx(3.0)
        assert result.median == pytest.approx(3.0)
        assert result.min == pytest.approx(1.0)
        assert result.max == pytest.approx(5.0)
        assert result.n == 5

    def test_std_calculation(self, analyzer: StatisticalAnalyzer) -> None:
        """std should use ddof=1 (sample std)."""
        samples = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = analyzer.summarize(samples)
        # numpy std(ddof=1) for this data is ~2.138
        assert result.std == pytest.approx(2.138, abs=0.01)

    def test_cv_calculation(self, analyzer: StatisticalAnalyzer) -> None:
        """CV should be std/mean."""
        samples = [10.0, 10.1, 9.9, 10.0, 10.0]
        result = analyzer.summarize(samples)
        expected_cv = result.std / result.mean
        assert result.cv == pytest.approx(expected_cv)

    def test_single_sample(self, analyzer: StatisticalAnalyzer) -> None:
        """Single sample should have std=0, CI=(value, value)."""
        result = analyzer.summarize([42.0])
        assert result.mean == pytest.approx(42.0)
        assert result.std == pytest.approx(0.0)
        assert result.ci_lower == pytest.approx(42.0)
        assert result.ci_upper == pytest.approx(42.0)
        assert result.n == 1

    def test_stability_flag_stable(self, analyzer: StatisticalAnalyzer) -> None:
        """CV < threshold should set is_stable=True."""
        samples = [100.0, 100.1, 99.9, 100.0, 100.05]
        result = analyzer.summarize(samples)
        assert result.cv < STABILITY_CV_THRESHOLD
        assert result.is_stable is True

    def test_stability_flag_unstable(self, analyzer: StatisticalAnalyzer) -> None:
        """CV >= threshold should set is_stable=False."""
        samples = [1.0, 10.0, 1.0, 10.0, 1.0]
        result = analyzer.summarize(samples)
        assert result.cv >= STABILITY_CV_THRESHOLD
        assert result.is_stable is False

    def test_zero_mean(self, analyzer: StatisticalAnalyzer) -> None:
        """CV should be 0 when mean is 0."""
        samples = [-1.0, 1.0, -1.0, 1.0]
        result = analyzer.summarize(samples)
        assert result.cv == pytest.approx(0.0)

    def test_identical_samples(self, analyzer: StatisticalAnalyzer) -> None:
        """Identical samples should have std=0, is_stable=True."""
        samples = [5.0, 5.0, 5.0, 5.0, 5.0]
        result = analyzer.summarize(samples)
        assert result.std == pytest.approx(0.0)
        assert result.cv == pytest.approx(0.0)
        assert result.is_stable is True


class TestBootstrapCI:
    """Tests for StatisticalAnalyzer.bootstrap_ci."""

    def test_normal_data_bounds(self, analyzer: StatisticalAnalyzer) -> None:
        """CI should contain the true mean for normal-like data."""
        samples = [float(x) for x in range(1, 101)]
        lo, hi = analyzer.bootstrap_ci(samples)
        true_mean = 50.5
        assert lo < true_mean < hi

    def test_single_sample(self, analyzer: StatisticalAnalyzer) -> None:
        """Single sample CI should be (value, value)."""
        lo, hi = analyzer.bootstrap_ci([7.0])
        assert lo == pytest.approx(7.0)
        assert hi == pytest.approx(7.0)

    def test_reproducible_with_same_seed(self) -> None:
        """Same seed should produce identical CIs."""
        a = StatisticalAnalyzer(seed=123)
        b = StatisticalAnalyzer(seed=123)
        samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        assert a.bootstrap_ci(samples) == b.bootstrap_ci(samples)

    def test_ci_width_scales_with_variability(self) -> None:
        """Higher variability should produce wider CI."""
        analyzer = StatisticalAnalyzer(seed=42)
        tight = [10.0, 10.01, 9.99, 10.0, 10.0, 10.01, 9.99, 10.0]
        wide = [1.0, 20.0, 3.0, 18.0, 5.0, 16.0, 7.0, 14.0]
        lo_t, hi_t = analyzer.bootstrap_ci(tight)
        analyzer2 = StatisticalAnalyzer(seed=42)
        lo_w, hi_w = analyzer2.bootstrap_ci(wide)
        assert (hi_t - lo_t) < (hi_w - lo_w)


class TestDetectOutliers:
    """Tests for StatisticalAnalyzer.detect_outliers."""

    def test_finds_planted_outliers(self, analyzer: StatisticalAnalyzer) -> None:
        """Should detect clearly planted outliers."""
        samples = [10.0, 10.1, 9.9, 10.0, 10.05, 10.0, 100.0]
        outliers = analyzer.detect_outliers(samples)
        assert 6 in outliers

    def test_no_outliers_in_uniform_data(self, analyzer: StatisticalAnalyzer) -> None:
        """Uniform data should have no outliers."""
        samples = [5.0, 5.1, 4.9, 5.0, 5.05, 4.95, 5.0, 5.02, 4.98]
        outliers = analyzer.detect_outliers(samples)
        assert outliers == []

    def test_empty_list(self, analyzer: StatisticalAnalyzer) -> None:
        """Empty list should return empty."""
        assert analyzer.detect_outliers([]) == []

    def test_short_list(self, analyzer: StatisticalAnalyzer) -> None:
        """Lists with fewer than 3 elements should return empty."""
        assert analyzer.detect_outliers([1.0]) == []
        assert analyzer.detect_outliers([1.0, 2.0]) == []

    def test_all_same_values(self, analyzer: StatisticalAnalyzer) -> None:
        """All identical values should have MAD=0, return empty."""
        assert analyzer.detect_outliers([3.0, 3.0, 3.0, 3.0, 3.0]) == []

    def test_multiple_outliers(self, analyzer: StatisticalAnalyzer) -> None:
        """Should detect multiple outliers."""
        # Use varied inliers so MAD != 0
        samples = [9.0, 10.0, 11.0, 9.5, 10.5, 10.0, 9.8, -100.0, 120.0]
        outliers = analyzer.detect_outliers(samples)
        assert 7 in outliers
        assert 8 in outliers

    def test_custom_threshold(self, analyzer: StatisticalAnalyzer) -> None:
        """Lower threshold should detect more outliers."""
        samples = [10.0, 10.1, 9.9, 10.0, 10.05, 10.0, 15.0]
        high_threshold = analyzer.detect_outliers(samples, threshold=5.0)
        low_threshold = analyzer.detect_outliers(samples, threshold=2.0)
        assert len(low_threshold) >= len(high_threshold)
