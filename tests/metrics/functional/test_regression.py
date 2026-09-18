"""Tests for calibrax.metrics.functional.regression module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.regression import (
    charbonnier_loss,
    crps,
    explained_variance,
    huber_loss,
    log_cosh_loss,
    mae,
    mape,
    max_error,
    mse,
    quantile_loss,
    r_squared,
    relative_error,
    relative_l2_error,
    rmse,
    smape,
)


class TestMSE:
    """Tests for mean squared error."""

    def test_perfect_predictions(self) -> None:
        """MSE should be 0 for identical arrays."""
        targets = jnp.array([1.0, 2.0, 3.0])
        assert mse(targets, targets) == pytest.approx(0.0, abs=1e-7)

    def test_known_value(self) -> None:
        """MSE should match hand-calculated value."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 3.0, 5.0])
        # (0 + 1 + 4) / 3 = 5/3
        assert mse(predictions, targets) == pytest.approx(5.0 / 3.0, rel=1e-5)

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array, not a Python float."""
        result = mse(jnp.ones(3), jnp.zeros(3))
        assert isinstance(result, jax.Array)


class TestMAE:
    """Tests for mean absolute error."""

    def test_perfect_predictions(self) -> None:
        """MAE should be 0 for identical arrays."""
        targets = jnp.array([1.0, 2.0, 3.0])
        assert mae(targets, targets) == pytest.approx(0.0, abs=1e-7)

    def test_known_value(self) -> None:
        """MAE should match hand-calculated value."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([2.0, 4.0, 6.0])
        # (1 + 2 + 3) / 3 = 2.0
        assert mae(predictions, targets) == pytest.approx(2.0, rel=1e-5)


class TestRMSE:
    """Tests for root mean squared error."""

    def test_perfect_predictions(self) -> None:
        """RMSE should be 0 for identical arrays."""
        targets = jnp.array([1.0, 2.0, 3.0])
        assert rmse(targets, targets) == pytest.approx(0.0, abs=1e-7)

    def test_known_value(self) -> None:
        """RMSE should be sqrt of MSE."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 3.0, 5.0])
        expected = (5.0 / 3.0) ** 0.5
        assert rmse(predictions, targets) == pytest.approx(expected, rel=1e-5)

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array, not a Python float."""
        result = rmse(jnp.ones(3), jnp.zeros(3))
        assert isinstance(result, jax.Array)

    def test_axis_gives_one_root_per_remaining_index(self) -> None:
        predictions = jnp.array([[0.0, 0.0], [3.0, 4.0]])
        targets = jnp.zeros((2, 2))
        # Row roots: sqrt(0) = 0 and sqrt((9 + 16) / 2).
        per_row = rmse(predictions, targets, axis=-1, reduction="none")
        assert per_row.tolist() == pytest.approx([0.0, (12.5) ** 0.5])
        assert float(rmse(predictions, targets, axis=-1)) == pytest.approx((12.5**0.5) / 2)
        assert float(rmse(predictions, targets, axis=-1, reduction="sum")) == pytest.approx(
            12.5**0.5
        )

    def test_without_an_axis_every_reduction_returns_the_one_root(self) -> None:
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 3.0, 5.0])
        for reduction in ("none", "mean", "sum"):
            assert float(rmse(predictions, targets, reduction=reduction)) == pytest.approx(
                (5.0 / 3.0) ** 0.5, rel=1e-6
            )

    def test_mask_excludes_elements_before_the_root(self) -> None:
        predictions = jnp.array([2.0, 100.0, 4.0])
        targets = jnp.array([0.0, 0.0, 0.0])
        mask = jnp.array([True, False, True])
        # sqrt((4 + 16) / 2)
        assert float(rmse(predictions, targets, mask=mask)) == pytest.approx(10.0**0.5)

    def test_weights_give_the_weighted_mean_under_the_root(self) -> None:
        predictions = jnp.array([1.0, 3.0])
        targets = jnp.zeros(2)
        weights = jnp.array([3.0, 1.0])
        # sqrt((3 * 1 + 1 * 9) / 4) = sqrt(3)
        assert float(rmse(predictions, targets, weights=weights)) == pytest.approx(3.0**0.5)

    def test_batch_sum_is_refused(self) -> None:
        with pytest.raises(ValueError, match="batch_sum"):
            rmse(jnp.ones((2, 3)), jnp.zeros((2, 3)), reduction="batch_sum")

    def test_gradient_is_finite_at_a_perfect_prediction(self) -> None:
        targets = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        for axis in (None, -1):
            grad = jax.grad(lambda p, a=axis: rmse(p, targets, axis=a))(targets)
            assert bool(jnp.all(jnp.isfinite(grad)))
        away = jax.grad(lambda p: rmse(p, targets))(targets + 1.0)
        assert bool(jnp.all(jnp.isfinite(away)))
        assert float(jnp.abs(away).sum()) > 0.0

    def test_jit_traces_once_and_vmap_matches_the_axis_form(self) -> None:
        from substrax.testing import TraceCounter

        counter = TraceCounter()
        compiled = jax.jit(counter.wrap(rmse), static_argnames=("reduction", "axis"))
        predictions = jnp.arange(6.0).reshape(2, 3)
        targets = jnp.ones((2, 3))
        with counter.expect(new_traces=1):
            compiled(predictions, targets, axis=-1, reduction="none")
        with counter.expect(new_traces=0):
            compiled(predictions * 2.0, targets, axis=-1, reduction="none")
        batched = jax.vmap(rmse)(predictions, targets)
        assert batched.tolist() == pytest.approx(
            rmse(predictions, targets, axis=-1, reduction="none").tolist()
        )


class TestRSquared:
    """Tests for coefficient of determination."""

    def test_perfect_fit(self) -> None:
        """R-squared should be ~1.0 for perfect predictions."""
        targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert r_squared(targets, targets) == pytest.approx(1.0, abs=1e-5)

    def test_poor_fit(self) -> None:
        """R-squared should be low for poor predictions."""
        predictions = jnp.array([5.0, 4.0, 3.0, 2.0, 1.0])
        targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = r_squared(predictions, targets)
        assert result < 0.0  # Negative R^2 means worse than mean

    def test_mean_predictor(self) -> None:
        """Predicting the mean should give R-squared ~0."""
        targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = jnp.full_like(targets, jnp.mean(targets))
        assert r_squared(predictions, targets) == pytest.approx(0.0, abs=1e-5)


class TestMAPE:
    """Tests for mean absolute percentage error."""

    def test_perfect_predictions(self) -> None:
        """MAPE should be ~0 for identical arrays."""
        targets = jnp.array([1.0, 2.0, 3.0])
        assert mape(targets, targets) == pytest.approx(0.0, abs=1e-5)

    def test_known_value(self) -> None:
        """MAPE should match hand-calculated value."""
        predictions = jnp.array([1.1, 2.2, 3.3])
        targets = jnp.array([1.0, 2.0, 3.0])
        # |0.1/1| + |0.2/2| + |0.3/3| = 0.1 + 0.1 + 0.1 => mean = 0.1
        assert mape(predictions, targets) == pytest.approx(0.1, rel=1e-3)

    def test_handles_near_zero_targets(self) -> None:
        """MAPE should not produce inf for near-zero targets."""
        predictions = jnp.array([0.1])
        targets = jnp.array([0.0])
        result = mape(predictions, targets)
        assert jnp.isfinite(result)


class TestRelativeError:
    """Tests for mean relative error."""

    def test_perfect_predictions(self) -> None:
        """Relative error should be ~0 for identical arrays."""
        targets = jnp.array([1.0, 2.0, 3.0])
        assert relative_error(targets, targets) == pytest.approx(0.0, abs=1e-5)

    def test_known_value(self) -> None:
        """Relative error should be L2 norm ratio."""
        predictions = jnp.array([2.0, 4.0])
        targets = jnp.array([1.0, 2.0])
        # diff = [1, 2], norm = sqrt(5)
        # target norm = sqrt(5)
        # relative = sqrt(5) / sqrt(5) = 1.0
        assert relative_error(predictions, targets) == pytest.approx(1.0, rel=1e-5)

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array, not a Python float."""
        result = relative_error(jnp.ones(3), jnp.ones(3) * 2)
        assert isinstance(result, jax.Array)


class TestExplainedVariance:
    """Tests for explained variance score."""

    def test_perfect_predictions(self) -> None:
        """Explained variance should be ~1.0 for perfect predictions."""
        targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert explained_variance(targets, targets) == pytest.approx(1.0, abs=1e-5)

    def test_mean_predictor(self) -> None:
        """Predicting the mean should give explained variance ~0."""
        targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = jnp.full_like(targets, jnp.mean(targets))
        assert explained_variance(predictions, targets) == pytest.approx(0.0, abs=1e-5)

    def test_constant_bias_invariance(self) -> None:
        """Explained variance should be invariant to constant bias."""
        targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Predictions = targets + constant offset
        predictions = targets + 10.0
        # Residuals have zero variance -> explained variance = 1.0
        assert explained_variance(predictions, targets) == pytest.approx(1.0, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array, not a Python float."""
        result = explained_variance(jnp.ones(3), jnp.arange(3.0))
        assert isinstance(result, jax.Array)


class TestMaxError:
    """Tests for maximum absolute error."""

    def test_perfect_predictions(self) -> None:
        """Max error should be 0 for identical arrays."""
        targets = jnp.array([1.0, 2.0, 3.0])
        assert max_error(targets, targets) == pytest.approx(0.0, abs=1e-7)

    def test_known_value(self) -> None:
        """Max error should return the largest absolute difference."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 3.0, 6.0])
        # Errors: |0|, |1|, |3| -> max = 3.0
        assert max_error(predictions, targets) == pytest.approx(3.0, rel=1e-5)

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array, not a Python float."""
        result = max_error(jnp.ones(3), jnp.zeros(3))
        assert isinstance(result, jax.Array)


class TestHuberLoss:
    """Tests for Huber loss."""

    def test_perfect_predictions(self) -> None:
        """Huber loss should be 0 for identical arrays."""
        targets = jnp.array([1.0, 2.0, 3.0])
        assert huber_loss(targets, targets) == pytest.approx(0.0, abs=1e-7)

    def test_small_errors_quadratic(self) -> None:
        """For small errors, Huber loss should equal 0.5 * MSE."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.1, 2.2, 3.3])
        # Errors within delta=1.0 -> quadratic regime
        expected_mse = mse(predictions, targets)
        assert huber_loss(predictions, targets) == pytest.approx(0.5 * expected_mse, rel=1e-5)

    def test_large_errors_linear(self) -> None:
        """For large errors, Huber loss should be linear in |error|."""
        predictions = jnp.array([0.0])
        targets = jnp.array([10.0])
        # |error| = 10 >> delta=1.0 -> linear: delta * (|e| - 0.5*delta) = 1*(10-0.5) = 9.5
        assert huber_loss(predictions, targets, delta=1.0) == pytest.approx(9.5, rel=1e-5)

    def test_custom_delta(self) -> None:
        """Custom delta should change the transition threshold."""
        predictions = jnp.array([0.0])
        targets = jnp.array([2.0])
        # delta=5.0: |error|=2 <= 5 -> quadratic: 0.5 * 4 = 2.0
        assert huber_loss(predictions, targets, delta=5.0) == pytest.approx(2.0, rel=1e-5)

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array, not a Python float."""
        result = huber_loss(jnp.ones(3), jnp.zeros(3))
        assert isinstance(result, jax.Array)


class TestQuantileLoss:
    """Tests for quantile (pinball) loss."""

    def test_perfect_predictions(self) -> None:
        """Quantile loss should be 0 for identical arrays."""
        targets = jnp.array([1.0, 2.0, 3.0])
        assert quantile_loss(targets, targets) == pytest.approx(0.0, abs=1e-7)

    def test_median_equals_half_mae(self) -> None:
        """At quantile=0.5, loss should equal 0.5 * MAE."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([2.0, 4.0, 6.0])
        expected_mae = mae(predictions, targets)
        assert quantile_loss(predictions, targets, quantile=0.5) == pytest.approx(
            0.5 * expected_mae, rel=1e-5
        )

    def test_asymmetric_penalty(self) -> None:
        """High quantile should penalize under-prediction more."""
        predictions = jnp.array([0.0])
        targets = jnp.array([1.0])
        # Under-prediction: diff = 1.0, q=0.9 -> 0.9 * 1.0 = 0.9
        assert quantile_loss(predictions, targets, quantile=0.9) == pytest.approx(0.9, rel=1e-5)
        # Over-prediction: diff = -1.0, q=0.9 -> (0.9 - 1.0) * (-1.0) = 0.1
        assert quantile_loss(targets, predictions, quantile=0.9) == pytest.approx(0.1, rel=1e-5)

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array, not a Python float."""
        result = quantile_loss(jnp.ones(3), jnp.zeros(3))
        assert isinstance(result, jax.Array)


class TestLogCoshLoss:
    """Tests for log-cosh loss."""

    def test_perfect_predictions(self) -> None:
        """Log-cosh loss should be 0 for identical arrays."""
        targets = jnp.array([1.0, 2.0, 3.0])
        assert log_cosh_loss(targets, targets) == pytest.approx(0.0, abs=1e-7)

    def test_small_errors_approx_half_mse(self) -> None:
        """For small errors, log-cosh ≈ 0.5 * error^2 (MSE-like)."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.01, 2.01, 3.01])
        result = log_cosh_loss(predictions, targets)
        expected_half_mse = 0.5 * mse(predictions, targets)
        assert result == pytest.approx(expected_half_mse, rel=1e-2)

    def test_always_non_negative(self) -> None:
        """Log-cosh loss should always be >= 0."""
        predictions = jnp.array([-5.0, 0.0, 5.0])
        targets = jnp.array([5.0, 0.0, -5.0])
        assert log_cosh_loss(predictions, targets) >= 0.0

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array, not a Python float."""
        result = log_cosh_loss(jnp.ones(3), jnp.zeros(3))
        assert isinstance(result, jax.Array)


class TestSMAPE:
    """Tests for symmetric mean absolute percentage error."""

    def test_perfect_predictions(self) -> None:
        """SMAPE should be ~0 for identical arrays."""
        targets = jnp.array([1.0, 2.0, 3.0])
        assert smape(targets, targets) == pytest.approx(0.0, abs=1e-5)

    def test_symmetry(self) -> None:
        """SMAPE(p, t) should equal SMAPE(t, p)."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([2.0, 4.0, 6.0])
        assert smape(predictions, targets) == pytest.approx(smape(targets, predictions), rel=1e-5)

    def test_known_value(self) -> None:
        """SMAPE should match hand-calculated value."""
        predictions = jnp.array([2.0])
        targets = jnp.array([1.0])
        # |2-1| / ((|2|+|1|)/2) = 1 / 1.5 = 2/3
        assert smape(predictions, targets) == pytest.approx(2.0 / 3.0, rel=1e-4)

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array, not a Python float."""
        result = smape(jnp.ones(3), jnp.ones(3) * 2)
        assert isinstance(result, jax.Array)


class TestCRPS:
    """Tests for empirical continuous ranked probability score."""

    def test_known_ensemble_value(self) -> None:
        """CRPS should match a hand-calculated ensemble value."""
        predictions = jnp.array([[0.0, 1.0, 2.0]])
        targets = jnp.array([1.0])
        assert crps(predictions, targets) == pytest.approx(2.0 / 9.0, rel=1e-6)

    def test_exact_ensemble_is_lower_than_spread_ensemble(self) -> None:
        """Exact ensembles should score better than spread ensembles."""
        exact = crps(jnp.array([[1.0, 1.0, 1.0]]), jnp.array([1.0]))
        spread = crps(jnp.array([[0.0, 1.0, 2.0]]), jnp.array([1.0]))
        assert exact < spread

    def test_one_dimensional_predictions_raise(self) -> None:
        """CRPS requires an explicit ensemble-member dimension."""
        with pytest.raises(ValueError, match="2-dimensional"):
            crps(jnp.array([0.0, 1.0, 2.0]), jnp.array([1.0, 1.0, 1.0]))

    def test_single_member_ensemble_raises(self) -> None:
        """CRPS requires at least two ensemble members."""
        with pytest.raises(ValueError, match="at least two ensemble members"):
            crps(jnp.array([[1.0], [2.0]]), jnp.array([1.0, 2.0]))

    def test_sample_count_mismatch_raises(self) -> None:
        """Prediction and target sample counts must match."""
        with pytest.raises(ValueError, match="matching sample count"):
            crps(jnp.ones((2, 3)), jnp.ones(3))

    def test_scalar_target_for_single_sample(self) -> None:
        """A scalar target is accepted for a single ensemble forecast."""
        predictions = jnp.array([[0.0, 1.0, 2.0]])
        assert crps(predictions, jnp.array(1.0)) == pytest.approx(2.0 / 9.0, rel=1e-6)

    def test_returns_jax_scalar_under_jit(self) -> None:
        """JIT-compiled CRPS should return a JAX scalar array."""
        result = jax.jit(crps)(jnp.array([[0.0, 1.0, 2.0]]), jnp.array([1.0]))
        assert isinstance(result, jax.Array)
        assert result == pytest.approx(2.0 / 9.0, rel=1e-6)


class TestShapeValidation:
    """Tests for input shape validation."""

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched shapes should raise ValueError."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            mse(jnp.ones(3), jnp.ones(4))

    def test_2d_arrays_work(self) -> None:
        """Functions should work with multi-dimensional arrays."""
        predictions = jnp.ones((2, 3))
        targets = jnp.zeros((2, 3))
        result = mse(predictions, targets)
        assert result == pytest.approx(1.0)

    def test_scalar_inputs_work(self) -> None:
        """Functions should work with scalar inputs."""
        result = mse(jnp.array(1.0), jnp.array(2.0))
        assert result == pytest.approx(1.0)


class TestRelativeL2Error:
    """The per-sample relative L2 of operator learning (PDEBench convention)."""

    def test_per_sample_ratios(self) -> None:
        from calibrax.metrics.functional.regression import per_sample_relative_l2, relative_l2_error

        target = jnp.array([[3.0, 4.0], [6.0, 8.0]])  # norms 5 and 10
        per_sample = per_sample_relative_l2(jnp.zeros((2, 2)), target)
        assert per_sample.shape == (2,)
        assert bool(jnp.allclose(per_sample, 1.0, atol=1e-4))
        assert relative_l2_error(jnp.zeros((2, 2)), target) == pytest.approx(1.0, rel=1e-4)

    def test_identical_fields_have_zero_error(self) -> None:
        from calibrax.metrics.functional.regression import relative_l2_error

        field = jnp.arange(12.0).reshape(3, 4)
        assert relative_l2_error(field, field) == pytest.approx(0.0, abs=1e-6)

    def test_flattens_trailing_axes_per_sample(self) -> None:
        from calibrax.metrics.functional.regression import relative_l2_error

        target = jnp.ones((4, 8, 8, 1))
        assert relative_l2_error(target * 1.1, target) == pytest.approx(0.1, rel=1e-3)

    def test_zero_target_is_guarded(self) -> None:
        from calibrax.metrics.functional.regression import relative_l2_error

        assert bool(jnp.isfinite(relative_l2_error(jnp.ones((1, 4)), jnp.zeros((1, 4)))))

    def test_jit_grad_vmap(self) -> None:
        from calibrax.metrics.functional.regression import per_sample_relative_l2, relative_l2_error

        pred, target = jnp.full((6, 5), 0.5), jnp.ones((6, 5))
        assert bool(jnp.isfinite(jax.jit(relative_l2_error)(pred, target)))
        assert bool(jnp.all(jnp.isfinite(jax.grad(relative_l2_error)(pred, target))))
        assert jax.vmap(per_sample_relative_l2)(pred[None], target[None]).shape == (1, 6)


class TestMaskedWeightedReduction:
    """The losses take a mask, weights, a reduction and an axis, and agree with hand computations."""

    predictions = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, 1.0, 1.0]])
    targets = jnp.array([[1.0, 1.0, 1.0], [4.0, 4.0, 4.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    row_mask = jnp.array([True, True, False, True])

    def test_masked_mean_equals_the_mean_over_valid_rows(self) -> None:
        masked = mse(self.predictions, self.targets, mask=self.row_mask[:, None])
        valid = mse(self.predictions[self.row_mask], self.targets[self.row_mask])
        assert jnp.allclose(masked, valid)

    def test_weights_match_a_hand_computation(self) -> None:
        weights = jnp.array([1.0, 2.0, 0.5, 4.0])[:, None]
        squared = (self.predictions - self.targets) ** 2
        expected = jnp.sum(weights * squared) / jnp.sum(jnp.broadcast_to(weights, squared.shape))
        assert jnp.allclose(mse(self.predictions, self.targets, weights=weights), expected)

    def test_an_all_false_mask_gives_zero(self) -> None:
        """One documented result, finite, so a caller can check it at the host boundary."""
        nothing = jnp.zeros_like(self.predictions, dtype=bool)
        for loss in (mse, mae, huber_loss):
            value = loss(self.predictions, self.targets, mask=nothing)
            assert jnp.isfinite(value)
            assert value == 0.0

    def test_reductions_and_axis(self) -> None:
        squared = (self.predictions - self.targets) ** 2
        assert jnp.allclose(mse(self.predictions, self.targets, reduction="none"), squared)
        assert jnp.allclose(mse(self.predictions, self.targets, reduction="sum"), jnp.sum(squared))
        assert jnp.allclose(mse(self.predictions, self.targets, axis=0), jnp.mean(squared, axis=0))
        masked_none = mse(
            self.predictions, self.targets, mask=self.row_mask[:, None], reduction="none"
        )
        assert jnp.all(masked_none[2] == 0.0)

    def test_rejects_an_unknown_reduction(self) -> None:
        with pytest.raises(ValueError, match="reduction"):
            mse(self.predictions, self.targets, reduction="median")

    def test_jit_and_grad_work_through_the_mask(self) -> None:
        mask = self.row_mask[:, None]

        def loss(p: jax.Array) -> jax.Array:
            return mae(p, self.targets, mask=mask)

        eager = loss(self.predictions)
        assert jnp.allclose(jax.jit(loss)(self.predictions), eager)
        gradient = jax.grad(loss)(self.predictions)
        assert jnp.all(jnp.isfinite(gradient))
        assert jnp.all(gradient[2] == 0.0)

    def test_huber_and_relative_l2_take_the_same_keywords(self) -> None:
        huber_masked = huber_loss(self.predictions, self.targets, mask=self.row_mask[:, None])
        huber_valid = huber_loss(self.predictions[self.row_mask], self.targets[self.row_mask])
        assert jnp.allclose(huber_masked, huber_valid)
        per_sample_weights = jnp.array([1.0, 1.0, 0.0, 1.0])
        weighted = relative_l2_error(self.predictions, self.targets, weights=per_sample_weights)
        unweighted = relative_l2_error(self.predictions[self.row_mask], self.targets[self.row_mask])
        assert jnp.allclose(weighted, unweighted)


class TestCharbonnierLoss:
    """Charbonnier loss, ``(e^2 + eps^2)^(alpha/2)``, a differentiable L1."""

    def test_perfect_predictions_give_epsilon(self) -> None:
        x = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(charbonnier_loss(x, x, epsilon=1e-3), 1e-3)

    def test_known_value(self) -> None:
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([0.0, 0.0, 0.0])
        expected = jnp.mean(jnp.sqrt(predictions**2 + 1e-3**2))
        assert jnp.allclose(charbonnier_loss(predictions, targets), expected)

    def test_alpha_and_reduction(self) -> None:
        predictions = jnp.array([[3.0, 0.0], [0.0, 4.0]])
        targets = jnp.zeros_like(predictions)
        elementwise = charbonnier_loss(
            predictions, targets, epsilon=0.0, alpha=2.0, reduction="none"
        )
        assert jnp.allclose(elementwise, predictions**2)
        assert jnp.allclose(
            charbonnier_loss(predictions, targets, epsilon=0.0, reduction="sum"), 7.0
        )

    def test_is_differentiable_at_zero_error(self) -> None:
        x = jnp.array([0.0, 0.0])
        gradient = jax.grad(lambda p: charbonnier_loss(p, x))(x)
        assert jnp.all(jnp.isfinite(gradient))
