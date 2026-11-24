"""Tests for calibration metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from substrax.testing import TraceCounter

from calibrax.core.models import MetricDirection
from calibrax.metrics import MetricRegistry
from calibrax.metrics.functional.calibration import (
    _carry_limbs,
    _dyadic_q_limbs,
    _shift_prod_limbs,
    _u31_pair_limbs,
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

    def test_rank3_batch_and_jit_vmap_parity(self) -> None:
        rng = np.random.default_rng(77)
        logits = rng.normal(size=(7, 5, 3)).astype(np.float32)
        probs = jax.nn.softmax(jnp.asarray(logits), axis=-1)
        targets = jnp.asarray(rng.integers(0, 3, size=(7, 5)), dtype=jnp.int32)

        direct = classwise_ece(probs, targets, num_bins=5)
        pooled = classwise_ece(probs.reshape(35, 3), targets.reshape(35), num_bins=5)
        assert float(direct) == pytest.approx(float(pooled), abs=1e-7)

        counter = TraceCounter()
        compiled = jax.jit(counter.wrap(lambda p, t: classwise_ece(p, t, num_bins=5)))
        with counter.expect(new_traces=1):
            compiled(probs, targets)
        with counter.expect(new_traces=0):
            compiled(probs, (targets + 1) % 3)

        vmapped = jax.jit(jax.vmap(lambda p, t: classwise_ece(p, t, num_bins=5)))(probs, targets)
        assert vmapped.shape == (7,)
        for row in range(7):
            expected_row = float(classwise_ece(probs[row], targets[row], num_bins=5))
            assert float(vmapped[row]) == pytest.approx(expected_row, abs=1e-7)

        with pytest.raises(ValueError, match="Shape mismatch"):
            classwise_ece(probs, jnp.zeros((7, 4), dtype=jnp.int32))
        assert float(classwise_ece(probs, targets, num_bins=5, num_classes=3)) == pytest.approx(
            float(direct), abs=1e-7
        )
        with pytest.raises(ValueError, match="num_classes"):
            classwise_ece(probs, targets, num_bins=5, num_classes=5)


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
    """Selective-conformal FDR selection (Bates et al. 2023; Jin & Candes 2023) and FDP/power."""

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

            # Default marginal_null regime matches Bates et al. (2023) null-pool selection.
            null_only = cal_scores[null_mask]
            expected_marginal = _stratcp_get_sel_single_reference(
                null_only, np.ones(null_only.size, dtype=bool), test_scores, q=0.15
            )
            actual_marginal = np.asarray(
                conformal_fdr_selection(cal_scores, test_scores, q=0.15, cal_null_mask=null_mask)
            )
            actual_null_pool = np.asarray(conformal_fdr_selection(null_only, test_scores, q=0.15))
            np.testing.assert_array_equal(actual_marginal, expected_marginal)
            np.testing.assert_array_equal(actual_null_pool, expected_marginal)

            # Opt-in joint_iid regime matches Jin & Candes (2023) / StratCP get_sel_single.
            expected_joint = _stratcp_get_sel_single_reference(
                cal_scores, null_mask, test_scores, q=0.15
            )
            actual_joint = np.asarray(
                conformal_fdr_selection(
                    cal_scores,
                    test_scores,
                    q=0.15,
                    cal_null_mask=null_mask,
                    null_proportion_regime="joint_iid",
                )
            )
            np.testing.assert_array_equal(actual_joint, expected_joint)

    def test_marginal_null_rejects_decoys_under_null_proportion_shift(self) -> None:
        cal_s = jnp.linspace(0.0, 1.0, 100, dtype=jnp.float32)
        null_m = jnp.arange(100) < 10
        sel_decoy = conformal_fdr_selection(
            cal_s, jnp.array([-10.0], dtype=jnp.float32), q=0.2, cal_null_mask=null_m
        )
        assert sel_decoy.tolist() == [False]
        assert jax.jit(lambda ts: conformal_fdr_selection(cal_s, ts, q=0.2, cal_null_mask=null_m))(
            jnp.array([-10.0], dtype=jnp.float32)
        ).tolist() == [False]

        # Zero calibration nulls cannot form conformal p-values < 1 under marginal_null.
        no_nulls = jnp.zeros(20, dtype=bool)
        assert not bool(
            jnp.any(
                conformal_fdr_selection(
                    jnp.linspace(0.0, 1.0, 20),
                    jnp.array([10.0, 20.0]),
                    q=0.5,
                    cal_null_mask=no_nulls,
                )
            )
        )

        # NaN in test or calibration scores yields no selection.
        assert not bool(jnp.any(conformal_fdr_selection(cal_s, jnp.array([jnp.nan, 5.0]), q=0.2)))
        assert not bool(
            jnp.any(conformal_fdr_selection(cal_s.at[0].set(jnp.nan), jnp.array([5.0]), q=0.2))
        )

        # Severe label shift: 15% null calibration set vs 95% null test set controls empirical FDR.
        rng = np.random.default_rng(909)
        fdps = []
        for _ in range(80):
            cal_is_null = rng.random(120) < 0.15
            cal_is_null[:2] = True
            cal_draw = np.where(
                cal_is_null, rng.normal(0.0, 1.0, 120), rng.normal(3.0, 1.0, 120)
            ).astype(np.float32)
            test_is_null = rng.random(200) < 0.95
            test_draw = np.where(
                test_is_null, rng.normal(0.0, 1.0, 200), rng.normal(3.0, 1.0, 200)
            ).astype(np.float32)
            sel = conformal_fdr_selection(cal_draw, test_draw, q=0.2, cal_null_mask=cal_is_null)
            fdps.append(float(false_discovery_proportion(sel, test_is_null)))
        assert float(np.mean(fdps)) <= 0.22

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
        for bad_q in (
            0.0,
            1.0,
            -0.2,
            1.5,
            float("nan"),
            np.float32(1.0),
            np.array(1.5, dtype=np.float32),
            np.array(-0.1, dtype=np.float64),
            np.array(np.nan, dtype=np.float32),
            np.array([0.05, 0.2], dtype=np.float32),
            jnp.array([0.05, 0.2], dtype=jnp.float32),
        ):
            with pytest.raises(ValueError, match="q"):
                conformal_fdr_selection(jnp.ones(4), jnp.ones(3), q=bad_q)
        for bool_q in (False, True):
            with pytest.raises(TypeError, match="q"):
                conformal_fdr_selection(jnp.ones(4), jnp.ones(3), q=bool_q)
        with pytest.raises(ValueError, match="null_proportion_regime"):
            conformal_fdr_selection(
                jnp.ones(4),
                jnp.ones(3),
                null_proportion_regime="unknown",  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="Shape mismatch"):
            conformal_fdr_selection(jnp.ones(4), jnp.ones(3), cal_null_mask=jnp.ones(2, dtype=bool))
        with pytest.raises(ValueError, match="Shape mismatch"):
            false_discovery_proportion(jnp.ones(2, dtype=bool), jnp.ones(3, dtype=bool))
        with pytest.raises(ValueError, match="Shape mismatch"):
            empirical_power(jnp.ones(2, dtype=bool), jnp.ones(3, dtype=bool))

        sel_seq = conformal_fdr_selection([0.0, 0.1, 0.2, 0.3], (5.0, -1.0), q=0.5)  # type: ignore[arg-type]
        sel_tup = conformal_fdr_selection((0.0, 0.1, 0.2, 0.3), [5.0, -1.0], q=0.5)  # type: ignore[arg-type]
        assert sel_seq.tolist() == [True, False]
        assert sel_tup.tolist() == [True, False]

        cal = jnp.linspace(-2.0, 1.0, 32, dtype=jnp.float32)
        test_batch = jnp.array([[2.0, 2.5, -1.0, 0.0], [-1.5, -0.5, 0.2, 3.0]], dtype=jnp.float32)
        vmapped = jax.jit(jax.vmap(lambda ts: conformal_fdr_selection(cal, ts, q=0.2)))(test_batch)
        assert vmapped.shape == (2, 4)
        assert vmapped.dtype == jnp.bool_

        q_grid = jnp.array([0.05, 0.30], dtype=jnp.float32)
        traced_q = jax.jit(
            jax.vmap(lambda q_val: conformal_fdr_selection(cal, test_batch[0], q=q_val))
        )(q_grid)
        assert traced_q.shape == (2, 4)
        assert traced_q.dtype == jnp.bool_

    def test_exact_step_up_threshold_rejects_sub_boundary_q(self) -> None:
        # 59 null calibration scores of 0.0 and 3 test scores [1.0, -1.0, -1.0]:
        # conformal p-values are [1/60, 60/60, 60/60] -> exact BH q-values [0.05, 1.0, 1.0].
        cal60 = jnp.zeros(59, dtype=jnp.float32)
        test = jnp.array([1.0, -1.0, -1.0], dtype=jnp.float32)

        assert conformal_fdr_selection(cal60, test, q=0.05).tolist() == [True, False, False]
        assert conformal_fdr_selection(cal60, test, q=np.float32(0.05)).tolist() == [
            True,
            False,
            False,
        ]
        assert jax.jit(lambda c, t: conformal_fdr_selection(c, t, q=0.05))(
            cal60, test
        ).tolist() == [
            True,
            False,
            False,
        ]

        q_below_f32 = np.nextafter(np.float32(0.05), np.float32(0.0))
        assert conformal_fdr_selection(cal60, test, q=float(q_below_f32)).tolist() == [
            False,
            False,
            False,
        ]
        assert conformal_fdr_selection(cal60, test, q=q_below_f32).tolist() == [False, False, False]
        assert jax.jit(lambda qv: conformal_fdr_selection(cal60, test, q=qv))(
            jnp.asarray(q_below_f32)
        ).tolist() == [False, False, False]

        # 99 null calibration scores -> exact BH q-value is 3/100. Both float32(0.03) and binary64
        # 0.03 lie strictly below 3/100, so exact rational comparison rejects 3/100 and matches
        # traced float32(0.03), whereas nextafter(0.03, 1.0) > 3/100 selects the first item.
        cal100 = jnp.zeros(99, dtype=jnp.float32)
        assert conformal_fdr_selection(cal100, test, q=np.float32(0.03)).tolist() == [
            False,
            False,
            False,
        ]
        assert jax.jit(lambda qv: conformal_fdr_selection(cal100, test, q=qv))(
            jnp.float32(0.03)
        ).tolist() == [False, False, False]
        assert conformal_fdr_selection(cal100, test, q=0.03).tolist() == [False, False, False]
        assert conformal_fdr_selection(cal100, test, q=float(np.nextafter(0.03, 1.0))).tolist() == [
            True,
            False,
            False,
        ]

        # Arbitrarily small q values (whose binary denominators exceed 2^100) must never overflow
        # limb representations into spurious selections.
        for tiny_q in (1e-30, 1e-300, float(np.finfo(np.float64).tiny)):
            assert conformal_fdr_selection(cal60, test, q=tiny_q).tolist() == [
                False,
                False,
                False,
            ]
            assert jax.jit(lambda c, t, qv=tiny_q: conformal_fdr_selection(c, t, q=qv))(
                cal60, test
            ).tolist() == [False, False, False]

        # Exact dyadic q = 0.25 = 2^-2: 11 null calibration scores -> denom=12,
        # BH q-values [3/12=0.25, 1.0, 1.0].
        cal12 = jnp.zeros(11, dtype=jnp.float32)
        assert conformal_fdr_selection(cal12, test, q=0.25).tolist() == [True, False, False]
        assert conformal_fdr_selection(
            cal12, test, q=float(np.nextafter(np.float32(0.25), np.float32(0.0)))
        ).tolist() == [False, False, False]

        # Large 60-bit rank*denom product (2^60 * 0.125 = 2^57) where shifted[4] occupies high bits
        # of the 5-limb 70-bit window: LHS = 2^58 > RHS = 2^57 must compare strictly greater.
        lhs_hi = _u31_pair_limbs(jnp.array([1 << 29], dtype=jnp.int32), jnp.int32(1 << 29))
        rd_hi = _u31_pair_limbs(jnp.array([1 << 30], dtype=jnp.int32), jnp.int32(1 << 30))
        q_limbs_hi, shift_hi = _dyadic_q_limbs(jnp.asarray(0.125), 0.125)
        prod_acc_hi = [jnp.zeros_like(lhs_hi[0]) for _ in range(9)]
        for idx_i, rdi in enumerate(rd_hi):
            for idx_k, qk in enumerate(q_limbs_hi):
                prod_acc_hi[idx_i + idx_k] = prod_acc_hi[idx_i + idx_k] + rdi * qk
        rhs_shifted_hi = _shift_prod_limbs(_carry_limbs(prod_acc_hi), shift_hi)
        lhs_val = sum(int(lhs_hi[k][0]) << (14 * k) for k in range(5))
        rhs_val = sum(int(rhs_shifted_hi[k][0]) << (14 * k) for k in range(5))
        assert lhs_val == (1 << 58)
        assert rhs_val == (1 << 57)

        # Concrete float64 q near 1.0 (which rounds to 1.0 in float32) remains valid in (0, 1).
        assert conformal_fdr_selection(cal12, test, q=float(np.nextafter(1.0, 0.0))).tolist() == [
            True,
            False,
            False,
        ]

        with jax.enable_x64(True):
            cal64 = jnp.zeros(59, dtype=jnp.float64)
            test64 = jnp.array([1.0, -1.0, -1.0], dtype=jnp.float64)
            q_below_f64 = float(np.nextafter(0.05, 0.0))
            assert conformal_fdr_selection(cal64, test64, q=0.05).tolist() == [True, False, False]
            assert conformal_fdr_selection(cal64, test64, q=q_below_f64).tolist() == [
                False,
                False,
                False,
            ]
            assert jax.jit(lambda qv: conformal_fdr_selection(cal64, test64, q=qv))(
                jnp.float64(q_below_f64)
            ).tolist() == [False, False, False]

    def test_rank3_nested_vmap_parity(self) -> None:
        rng = np.random.default_rng(2026)
        cal_np = rng.normal(size=(7, 5, 16)).astype(np.float32)
        null_mask_np = rng.random(size=(7, 5, 16)) > 0.25
        null_mask_np[:, :, 0] = True
        is_null_np = rng.random(size=(7, 5, 3)) > 0.45
        # Ensure one cell has all true nulls (zero alternatives -> NaN empirical_power).
        is_null_np[0, 0, :] = True
        is_alt_np = ~is_null_np
        test_np = (rng.normal(size=(7, 5, 3)) + 2.5 * is_alt_np.astype(np.float32)).astype(
            np.float32
        )

        cal_j = jnp.asarray(cal_np)
        null_mask_j = jnp.asarray(null_mask_np)
        test_j = jnp.asarray(test_np)
        is_null_j = jnp.asarray(is_null_np)
        is_alt_j = jnp.asarray(is_alt_np)

        nested_sel = jax.jit(
            jax.vmap(
                jax.vmap(
                    lambda cs, nm, ts: conformal_fdr_selection(cs, ts, q=0.2, cal_null_mask=nm)
                )
            )
        )(cal_j, null_mask_j, test_j)
        nested_joint = jax.jit(
            jax.vmap(
                jax.vmap(
                    lambda cs, nm, ts: conformal_fdr_selection(
                        cs,
                        ts,
                        q=0.2,
                        cal_null_mask=nm,
                        null_proportion_regime="joint_iid",
                    )
                )
            )
        )(cal_j, null_mask_j, test_j)
        nested_fdp = jax.jit(jax.vmap(jax.vmap(false_discovery_proportion)))(nested_sel, is_null_j)
        nested_pow = jax.jit(jax.vmap(jax.vmap(empirical_power)))(nested_sel, is_alt_j)

        assert nested_sel.shape == (7, 5, 3)
        assert nested_sel.dtype == jnp.bool_
        assert nested_fdp.shape == (7, 5)
        assert nested_pow.shape == (7, 5)

        for i in range(7):
            for j in range(5):
                null_sub = cal_np[i, j][null_mask_np[i, j]]
                ref_marginal = _stratcp_get_sel_single_reference(
                    null_sub, np.ones(null_sub.size, dtype=bool), test_np[i, j], q=0.2
                )
                np.testing.assert_array_equal(np.asarray(nested_sel[i, j]), ref_marginal)
                ref_joint = _stratcp_get_sel_single_reference(
                    cal_np[i, j], null_mask_np[i, j], test_np[i, j], q=0.2
                )
                np.testing.assert_array_equal(np.asarray(nested_joint[i, j]), ref_joint)
                eager_fdp = float(false_discovery_proportion(nested_sel[i, j], is_null_j[i, j]))
                assert float(nested_fdp[i, j]) == pytest.approx(eager_fdp, abs=1e-7)
                eager_pow = float(empirical_power(nested_sel[i, j], is_alt_j[i, j]))
                if np.isnan(eager_pow):
                    assert np.isnan(float(nested_pow[i, j]))
                else:
                    assert float(nested_pow[i, j]) == pytest.approx(eager_pow, abs=1e-7)
