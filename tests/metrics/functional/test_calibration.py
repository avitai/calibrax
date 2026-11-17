"""Tests for calibration metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.core.models import MetricDirection
from calibrax.metrics import MetricRegistry
from calibrax.metrics.functional.calibration import (
    adaptive_calibration_error,
    brier_decomposition,
    brier_score,
    classwise_ece,
    conformal_fdr_selection,
    empirical_power,
    expected_calibration_error,
    false_discovery_proportion,
    maximum_calibration_error,
    reliability_diagram_bins,
)


class TestBrierScore:
    """Tests for brier_score."""

    def test_perfect_predictions(self) -> None:
        predictions = jnp.array([1.0, 0.0, 1.0, 0.0])
        targets = jnp.array([1, 0, 1, 0])
        assert brier_score(predictions, targets) == pytest.approx(0.0, abs=1e-6)

    def test_worst_predictions(self) -> None:
        predictions = jnp.array([0.0, 1.0, 0.0, 1.0])
        targets = jnp.array([1, 0, 1, 0])
        assert brier_score(predictions, targets) == pytest.approx(1.0, abs=1e-6)

    def test_uniform_predictions(self) -> None:
        predictions = jnp.array([0.5, 0.5, 0.5, 0.5])
        targets = jnp.array([1, 0, 1, 0])
        assert brier_score(predictions, targets) == pytest.approx(0.25, abs=1e-6)

    def test_asymmetric_predictions(self) -> None:
        predictions = jnp.array([0.9, 0.1, 0.8, 0.2])
        targets = jnp.array([1, 0, 1, 0])
        result = brier_score(predictions, targets)
        assert 0.0 < result < 0.25

    def test_returns_jax_scalar(self) -> None:
        predictions = jnp.array([0.5, 0.5])
        targets = jnp.array([1, 0])
        result = brier_score(predictions, targets)
        assert isinstance(result, jax.Array)


class TestExpectedCalibrationError:
    """Tests for expected_calibration_error."""

    def test_perfectly_calibrated(self) -> None:
        # Predictions matching actual frequencies within bins
        predictions = jnp.array([0.1, 0.1, 0.9, 0.9])
        targets = jnp.array([0, 0, 1, 1])
        result = expected_calibration_error(predictions, targets, num_bins=5)
        assert result == pytest.approx(0.0, abs=0.15)

    def test_completely_miscalibrated(self) -> None:
        # All predict 0.9 but none are positive
        predictions = jnp.array([0.9, 0.9, 0.9, 0.9])
        targets = jnp.array([0, 0, 0, 0])
        result = expected_calibration_error(predictions, targets, num_bins=10)
        assert result > 0.5

    def test_range_zero_to_one(self) -> None:
        predictions = jnp.array([0.3, 0.7, 0.2, 0.8, 0.5])
        targets = jnp.array([0, 1, 0, 1, 1])
        result = expected_calibration_error(predictions, targets)
        assert 0.0 <= result <= 1.0

    def test_returns_jax_scalar(self) -> None:
        predictions = jnp.array([0.5, 0.5])
        targets = jnp.array([1, 0])
        result = expected_calibration_error(predictions, targets)
        assert isinstance(result, jax.Array)

    def test_custom_num_bins(self) -> None:
        predictions = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        targets = jnp.array([0, 0, 1, 1, 1])
        result_5 = expected_calibration_error(predictions, targets, num_bins=5)
        result_20 = expected_calibration_error(predictions, targets, num_bins=20)
        # Both should be valid floats in [0, 1]
        assert 0.0 <= result_5 <= 1.0
        assert 0.0 <= result_20 <= 1.0


class TestMaximumCalibrationError:
    """Tests for maximum_calibration_error."""

    def test_perfectly_calibrated(self) -> None:
        predictions = jnp.array([0.1, 0.1, 0.9, 0.9])
        targets = jnp.array([0, 0, 1, 1])
        result = maximum_calibration_error(predictions, targets, num_bins=5)
        assert result == pytest.approx(0.0, abs=0.15)

    def test_worst_case_bin(self) -> None:
        # All predictions in one bin, all wrong
        predictions = jnp.array([0.9, 0.9, 0.9, 0.9])
        targets = jnp.array([0, 0, 0, 0])
        result = maximum_calibration_error(predictions, targets, num_bins=10)
        assert result > 0.5

    def test_mce_geq_ece(self) -> None:
        predictions = jnp.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.7])
        targets = jnp.array([0, 1, 0, 1, 1, 0])
        ece = expected_calibration_error(predictions, targets, num_bins=5)
        mce = maximum_calibration_error(predictions, targets, num_bins=5)
        assert mce >= ece - 1e-6

    def test_returns_jax_scalar(self) -> None:
        predictions = jnp.array([0.5, 0.5])
        targets = jnp.array([1, 0])
        result = maximum_calibration_error(predictions, targets)
        assert isinstance(result, jax.Array)


class TestReliabilityDiagramBins:
    """Tests for reliability_diagram_bins."""

    def test_returns_dict_with_expected_keys(self) -> None:
        predictions = jnp.array([0.1, 0.5, 0.9])
        targets = jnp.array([0, 1, 1])
        result = reliability_diagram_bins(predictions, targets, num_bins=5)
        assert isinstance(result, dict)
        assert "bin_edges" in result
        assert "bin_accuracies" in result
        assert "bin_confidences" in result
        assert "bin_counts" in result

    def test_bin_edges_shape(self) -> None:
        predictions = jnp.array([0.1, 0.5, 0.9])
        targets = jnp.array([0, 1, 1])
        result = reliability_diagram_bins(predictions, targets, num_bins=5)
        assert result["bin_edges"].shape == (6,)  # num_bins + 1

    def test_bin_counts_sum_to_n(self) -> None:
        predictions = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        targets = jnp.array([0, 0, 1, 1, 1])
        result = reliability_diagram_bins(predictions, targets, num_bins=5)
        assert int(jnp.sum(result["bin_counts"])) == 5

    def test_arrays_have_correct_shape(self) -> None:
        num_bins = 10
        predictions = jnp.array([0.2, 0.4, 0.6, 0.8])
        targets = jnp.array([0, 0, 1, 1])
        result = reliability_diagram_bins(predictions, targets, num_bins=num_bins)
        assert result["bin_accuracies"].shape == (num_bins,)
        assert result["bin_confidences"].shape == (num_bins,)
        assert result["bin_counts"].shape == (num_bins,)


class TestBrierDecomposition:
    """Tests for brier_decomposition."""

    def test_returns_dict_with_expected_keys(self) -> None:
        predictions = jnp.array([0.1, 0.5, 0.9])
        targets = jnp.array([0, 1, 1])
        result = brier_decomposition(predictions, targets)
        assert isinstance(result, dict)
        assert "calibration" in result
        assert "resolution" in result
        assert "uncertainty" in result
        assert "brier" in result

    def test_decomposition_relationship(self) -> None:
        # brier ≈ calibration - resolution + uncertainty
        predictions = jnp.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.7])
        targets = jnp.array([0, 0, 1, 1, 0, 1])
        result = brier_decomposition(predictions, targets, num_bins=5)
        expected_brier = result["calibration"] - result["resolution"] + result["uncertainty"]
        assert result["brier"] == pytest.approx(expected_brier, abs=0.05)

    def test_uncertainty_is_base_rate_variance(self) -> None:
        targets = jnp.array([0, 0, 1, 1, 1])
        predictions = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
        result = brier_decomposition(predictions, targets)
        base_rate = 3.0 / 5.0
        expected_uncertainty = base_rate * (1.0 - base_rate)
        assert result["uncertainty"] == pytest.approx(expected_uncertainty, abs=1e-6)

    def test_calibration_non_negative(self) -> None:
        predictions = jnp.array([0.2, 0.8, 0.3, 0.7])
        targets = jnp.array([0, 1, 1, 0])
        result = brier_decomposition(predictions, targets)
        assert result["calibration"] >= -1e-6

    def test_resolution_non_negative(self) -> None:
        predictions = jnp.array([0.2, 0.8, 0.3, 0.7])
        targets = jnp.array([0, 1, 1, 0])
        result = brier_decomposition(predictions, targets)
        assert result["resolution"] >= -1e-6


class TestAdaptiveCalibrationError:
    """Tests for adaptive_calibration_error."""

    def test_perfectly_calibrated(self) -> None:
        # Simple case: predictions match targets
        predictions = jnp.array([0.0, 0.0, 1.0, 1.0])
        targets = jnp.array([0, 0, 1, 1])
        result = adaptive_calibration_error(predictions, targets, num_bins=2)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_completely_miscalibrated(self) -> None:
        predictions = jnp.array([0.9, 0.9, 0.9, 0.9])
        targets = jnp.array([0, 0, 0, 0])
        result = adaptive_calibration_error(predictions, targets, num_bins=2)
        assert result > 0.5

    def test_range_zero_to_one(self) -> None:
        predictions = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        targets = jnp.array([0, 0, 1, 1, 1])
        result = adaptive_calibration_error(predictions, targets)
        assert 0.0 <= result <= 1.0

    def test_returns_a_jax_scalar(self) -> None:
        result = adaptive_calibration_error(jnp.array([0.5, 0.5]), jnp.array([1, 0]))

        assert isinstance(result, jax.Array)
        assert result.shape == ()

    def test_the_bins_weigh_their_gap_by_their_size(self) -> None:
        # Bins {0.1, 0.3} (accuracy 0.5, confidence 0.2) and {0.7, 0.9} (1.0, 0.8):
        # (2 * 0.3 + 2 * 0.2) / 4.
        result = adaptive_calibration_error(
            jnp.array([0.9, 0.1, 0.7, 0.3]), jnp.array([1, 0, 1, 1]), num_bins=2
        )

        assert result == pytest.approx(0.25, abs=1e-6)

    def test_the_remainder_goes_to_the_first_bins(self) -> None:
        # Five samples in two bins: {0.1, 0.2, 0.3} then {0.8, 0.9}.
        # |1/3 - 0.2| * 3 + |1.0 - 0.85| * 2, over 5.
        result = adaptive_calibration_error(
            jnp.array([0.1, 0.2, 0.3, 0.8, 0.9]), jnp.array([0, 0, 1, 1, 1]), num_bins=2
        )

        assert result == pytest.approx((abs(1 / 3 - 0.2) * 3 + 0.15 * 2) / 5, abs=1e-6)

    def test_traces_and_differentiates(self) -> None:
        predictions = jax.random.uniform(jax.random.key(0), (50,))
        targets = (jax.random.uniform(jax.random.key(1), (50,)) > 0.5).astype(jnp.int32)

        traced = jax.jit(adaptive_calibration_error, static_argnames=("num_bins",))(
            predictions, targets, num_bins=7
        )
        gradient = jax.grad(adaptive_calibration_error)(predictions, targets)

        assert traced == pytest.approx(
            float(adaptive_calibration_error(predictions, targets, num_bins=7)), abs=1e-6
        )
        assert bool(jnp.all(jnp.isfinite(gradient)))

    def test_equal_mass_binning(self) -> None:
        # With 4 samples and 2 bins, each bin should have 2 samples
        predictions = jnp.array([0.1, 0.3, 0.7, 0.9])
        targets = jnp.array([0, 0, 1, 1])
        result = adaptive_calibration_error(predictions, targets, num_bins=2)
        # Well-calibrated case — low error
        assert result < 0.3


class TestClasswiseECE:
    """Tests for classwise_ece."""

    def test_perfectly_calibrated_multiclass(self) -> None:
        # 3 classes, predictions match targets
        predictions = jnp.array(
            [
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9],
            ]
        )
        targets = jnp.array([0, 1, 2])
        result = classwise_ece(predictions, targets, num_bins=5)
        assert result == pytest.approx(0.0, abs=0.15)

    def test_requires_2d_predictions(self) -> None:
        predictions = jnp.array([0.5, 0.5, 0.5])
        targets = jnp.array([0, 1, 0])
        with pytest.raises(ValueError, match="2D predictions"):
            classwise_ece(predictions, targets)

    def test_range_zero_to_one(self) -> None:
        predictions = jnp.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.3, 0.5],
                [0.6, 0.3, 0.1],
            ]
        )
        targets = jnp.array([0, 1, 2, 0])
        result = classwise_ece(predictions, targets, num_bins=5)
        assert 0.0 <= result <= 1.0

    def test_returns_jax_scalar(self) -> None:
        predictions = jnp.array([[0.8, 0.2], [0.3, 0.7]])
        targets = jnp.array([0, 1])
        result = classwise_ece(predictions, targets)
        assert isinstance(result, jax.Array)

    def test_infers_num_classes(self) -> None:
        predictions = jnp.array(
            [
                [0.5, 0.3, 0.2],
                [0.1, 0.6, 0.3],
            ]
        )
        targets = jnp.array([0, 1])
        # Should infer 3 classes from predictions shape
        result = classwise_ece(predictions, targets, num_bins=3)
        assert isinstance(result, jax.Array)


class TestCalibrationMetricRegistration:
    """Tests for calibration metric registration in MetricRegistry."""

    def test_calibration_metrics_registered(self) -> None:
        registry = MetricRegistry()
        expected = [
            "brier_score",
            "expected_calibration_error",
            "maximum_calibration_error",
            "adaptive_calibration_error",
            "classwise_ece",
            "false_discovery_proportion",
            "empirical_power",
        ]
        for name in expected:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_calibration_domain(self) -> None:
        registry = MetricRegistry()
        calibration_metrics = registry.list_by_domain("calibration")
        assert len(calibration_metrics) == 7
        names = {m.name for m in calibration_metrics}
        assert "brier_score" in names
        assert "expected_calibration_error" in names
        assert "false_discovery_proportion" in names
        assert "empirical_power" in names

    def test_brier_score_is_proper(self) -> None:
        registry = MetricRegistry()
        entry = registry.get("brier_score")
        assert entry.properties.is_proper is True

    def test_calibration_metrics_directions(self) -> None:
        registry = MetricRegistry()
        for m in registry.list_by_domain("calibration"):
            expected = (
                MetricDirection.HIGHER if m.name == "empirical_power" else MetricDirection.LOWER
            )
            assert m.direction == expected


def _stratcp_get_sel_single_reference(
    cal_scores: np.ndarray,
    cal_null_mask: np.ndarray,
    test_scores: np.ndarray,
    q: float,
) -> np.ndarray:
    """Exact Jin & Candes (Biometrika 2023) / StratCP get_sel_single selection mask."""
    n = cal_scores.size
    m = test_scores.size
    mu = np.concatenate([cal_scores, test_scores])
    is_cal_null = np.concatenate([cal_null_mask.astype(np.float64), np.zeros(m)])
    is_test = np.concatenate([np.zeros(n), np.ones(m)])
    # Break ties between calibration and test conservatively (cal nulls before test)
    order = np.lexsort((is_test, -mu))
    rr = (
        (1.0 + np.cumsum(is_cal_null[order]))
        / np.maximum(1.0, np.cumsum(is_test[order]))
        * (m / (1.0 + n))
    )
    ok = np.flatnonzero((rr <= q) & (is_test[order] == 1.0))
    if ok.size == 0:
        return np.zeros(m, dtype=bool)
    tau_hat = float(np.min(mu[order][ok]))
    return test_scores >= tau_hat


class TestSelectiveConformalFDR:
    """Selective-conformal FDR selection (Jin & Candes 2023; StratCP) and FDP/power metrics."""

    def test_matches_stratcp_reference_and_controls_fdr(self) -> None:
        rng = np.random.default_rng(123)
        for _ in range(25):
            n_cal, n_test = 80, 40
            cal_labels = rng.binomial(1, 0.35, size=n_cal)
            cal_scores = (rng.normal(loc=2.2 * cal_labels, scale=1.0, size=n_cal)).astype(
                np.float32
            )
            test_labels = rng.binomial(1, 0.35, size=n_test)
            test_scores = (rng.normal(loc=2.2 * test_labels, scale=1.0, size=n_test)).astype(
                np.float32
            )
            null_mask = cal_labels == 0

            expected = _stratcp_get_sel_single_reference(cal_scores, null_mask, test_scores, q=0.15)
            actual = np.asarray(
                conformal_fdr_selection(cal_scores, test_scores, q=0.15, cal_null_mask=null_mask)
            )
            np.testing.assert_array_equal(actual, expected)

            # Passing null-only calibration scores matches Bates et al. (2023) null-pool selection.
            null_only = cal_scores[null_mask]
            expected_null_pool = _stratcp_get_sel_single_reference(
                null_only, np.ones(null_only.size, dtype=bool), test_scores, q=0.15
            )
            actual_null_pool = np.asarray(conformal_fdr_selection(null_only, test_scores, q=0.15))
            np.testing.assert_array_equal(actual_null_pool, expected_null_pool)

    def test_false_discovery_proportion_and_empirical_power_identities(self) -> None:
        selected = jnp.array([True, True, False, True, False])
        is_null = jnp.array([False, True, True, False, False])
        is_alt = ~is_null

        # 3 selected, 1 is null -> FDP = 1/3; 3 true alternatives, 2 selected -> power = 2/3.
        assert float(false_discovery_proportion(selected, is_null)) == pytest.approx(
            1.0 / 3.0, abs=1e-7
        )
        assert float(empirical_power(selected, is_alt)) == pytest.approx(2.0 / 3.0, abs=1e-7)

        # Zero selections on non-empty test set -> FDP == 0.0 (BH 1995 definition), power == 0.0.
        none_sel = jnp.zeros(5, dtype=bool)
        assert float(false_discovery_proportion(none_sel, is_null)) == 0.0
        assert float(empirical_power(none_sel, is_alt)) == 0.0

        # Zero true alternatives -> empirical_power is undefined (NaN).
        assert np.isnan(float(empirical_power(selected, jnp.zeros(5, dtype=bool))))

        # Empty test set -> both return NaN.
        empty = jnp.zeros(0, dtype=bool)
        assert np.isnan(float(false_discovery_proportion(empty, empty)))
        assert np.isnan(float(empirical_power(empty, empty)))

    def test_validation_and_jit_vmap_compatibility(self) -> None:
        with pytest.raises(ValueError, match="q must be in"):
            conformal_fdr_selection(jnp.ones(4), jnp.ones(3), q=0.0)
        with pytest.raises(ValueError, match="Shape mismatch"):
            conformal_fdr_selection(jnp.ones(4), jnp.ones(3), cal_null_mask=jnp.ones(2, dtype=bool))
        with pytest.raises(ValueError, match="Shape mismatch"):
            false_discovery_proportion(jnp.ones(2, dtype=bool), jnp.ones(3, dtype=bool))
        with pytest.raises(ValueError, match="Shape mismatch"):
            empirical_power(jnp.ones(2, dtype=bool), jnp.ones(3, dtype=bool))

        cal = jnp.linspace(-2.0, 1.0, 32, dtype=jnp.float32)
        test_batch = jnp.array([[2.0, 2.5, -1.0, 0.0], [-1.5, -0.5, 0.2, 3.0]], dtype=jnp.float32)
        vmapped = jax.jit(jax.vmap(lambda ts: conformal_fdr_selection(cal, ts, q=0.2)))(test_batch)
        assert vmapped.shape == (2, 4)
        assert vmapped.dtype == jnp.bool_
