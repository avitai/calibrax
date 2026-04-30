"""Tests for calibrax.metrics.functional.regression module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.regression import (
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
