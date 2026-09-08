"""Tests for the uncertainty domain.

Closed-form references from Gneiting & Raftery 2007 (interval score), Kuleshov et
al. 2018 (regression calibration error), Gal & Ghahramani 2016 (entropy
decomposition), Bar-Shalom et al. 2002 (ANEES) and Li & Zhao 2006 (NCI).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from calibrax.metrics.functional.uncertainty import (
    anees,
    chi2_confidence_interval,
    ensemble_mutual_information,
    gaussian_nll,
    interval_score,
    mpiw,
    non_credibility_index,
    picp,
    predictive_entropy,
    regression_calibration_error,
    winkler_score,
)


class TestGaussianNll:
    def test_closed_form(self) -> None:
        mean = np.array([0.0, 1.0, -2.0])
        variance = np.array([1.0, 4.0, 0.25])
        target = np.array([0.5, 0.0, -1.5])
        expected = float(
            np.mean(0.5 * (np.log(2 * np.pi * variance) + (target - mean) ** 2 / variance))
        )
        assert gaussian_nll(
            jnp.asarray(mean), jnp.asarray(variance), jnp.asarray(target)
        ) == pytest.approx(expected, rel=1e-6, abs=1e-7)

    def test_jit_and_grad(self) -> None:
        mean, variance, target = jnp.zeros(4), jnp.ones(4), jnp.array([0.0, 1.0, -1.0, 0.5])
        assert jax.jit(gaussian_nll)(mean, variance, target) == pytest.approx(
            float(gaussian_nll(mean, variance, target)), abs=1e-6
        )
        grad = jax.grad(lambda m: gaussian_nll(m, variance, target))(mean)
        assert grad.shape == (4,)
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            gaussian_nll(jnp.zeros(3), jnp.ones(3), jnp.zeros(4))


class TestIntervals:
    def test_picp_counts_the_covered_fraction(self) -> None:
        lower = jnp.array([0.0, -1.0, 1.0, 0.5, 5.0])
        upper = jnp.array([2.0, 1.0, 3.0, 2.5, 6.0])
        target = jnp.array([1.0, 0.5, 1.5, 1.0, 0.0])
        assert picp(lower, upper, target) == pytest.approx(0.8, abs=1e-7)

    def test_mpiw_is_the_mean_width(self) -> None:
        assert mpiw(jnp.array([0.0, -2.0, 1.0]), jnp.array([1.0, 1.0, 4.0])) == pytest.approx(
            7.0 / 3.0, abs=1e-6
        )

    def test_interval_score_closed_form(self) -> None:
        lower, upper = jnp.zeros(3), jnp.ones(3)
        targets = jnp.array([0.5, -0.5, 1.5])  # inside, below, above
        expected = (1.0 + (1.0 + 20 * 0.5) + (1.0 + 20 * 0.5)) / 3
        assert interval_score(lower, upper, targets, alpha=0.1) == pytest.approx(expected, abs=1e-5)

    def test_winkler_is_the_interval_score(self) -> None:
        args = (jnp.array([0.0]), jnp.array([1.0]), jnp.array([2.0]))
        assert winkler_score(*args, alpha=0.1) == pytest.approx(
            float(interval_score(*args, alpha=0.1))
        )

    @given(
        width=st.floats(min_value=0.0, max_value=5.0),
        offset=st.floats(min_value=-5.0, max_value=5.0),
        alpha=st.floats(min_value=0.01, max_value=0.5),
    )
    @settings(max_examples=40, deadline=None)
    def test_interval_score_is_at_least_the_width_and_picp_is_a_fraction(
        self, width: float, offset: float, alpha: float
    ) -> None:
        lower = jnp.array([0.0])
        upper = jnp.array([width])
        target = jnp.array([offset])
        assert float(interval_score(lower, upper, target, alpha=alpha)) >= width - 1e-6
        assert 0.0 <= float(picp(lower, upper, target)) <= 1.0

    def test_vmap_over_trials(self) -> None:
        rng = np.random.default_rng(0)
        mean = jnp.asarray(rng.normal(size=(3, 16)))
        lower, upper = mean - 1.0, mean + 1.0
        target = jnp.asarray(rng.normal(size=(3, 16)))
        per_trial = jax.vmap(picp)(lower, upper, target)
        assert per_trial.shape == (3,)


class TestRegressionCalibrationError:
    def test_near_zero_for_a_calibrated_predictive(self) -> None:
        rng = np.random.default_rng(42)
        n = 2048
        error = regression_calibration_error(
            jnp.zeros(n),
            jnp.ones(n),
            jnp.asarray(rng.standard_normal(n)),
            quantile_levels=jnp.linspace(0.05, 0.95, 10),
        )
        assert float(error) < 0.05

    def test_detects_a_biased_predictive(self) -> None:
        rng = np.random.default_rng(0)
        n = 2048
        error = regression_calibration_error(
            jnp.full((n,), 5.0),
            jnp.ones(n),
            jnp.asarray(rng.standard_normal(n)),
            quantile_levels=jnp.linspace(0.05, 0.95, 10),
        )
        assert float(error) > 0.4


class TestEnsembleEntropies:
    def test_predictive_entropy_is_the_entropy_of_the_mean(self) -> None:
        probs = jnp.array(
            [[[0.9, 0.1], [0.3, 0.7]], [[0.7, 0.3], [0.4, 0.6]], [[0.8, 0.2], [0.5, 0.5]]]
        )
        mean = jnp.mean(probs, axis=0)
        expected = -jnp.sum(mean * jnp.log(mean), axis=-1)
        assert bool(jnp.allclose(predictive_entropy(probs), expected, atol=1e-6))

    def test_consensus_has_zero_entropy_and_zero_mutual_information(self) -> None:
        probs = jnp.tile(jnp.array([[1.0, 0.0]]), (5, 3, 1))
        assert bool(jnp.allclose(predictive_entropy(probs), 0.0, atol=1e-6))
        assert bool(jnp.allclose(ensemble_mutual_information(probs), 0.0, atol=1e-6))

    def test_mutual_information_decomposition(self) -> None:
        probs = jnp.array([[[0.9, 0.1], [0.5, 0.5]], [[0.1, 0.9], [0.5, 0.5]]])
        mean = jnp.mean(probs, axis=0)
        h_mean = -jnp.sum(mean * jnp.log(mean + 1e-8), axis=-1)
        h_members = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
        expected = h_mean - jnp.mean(h_members, axis=0)
        assert bool(jnp.allclose(ensemble_mutual_information(probs), expected, atol=1e-5))

    @given(seed=st.integers(min_value=0, max_value=500))
    @settings(max_examples=25, deadline=None)
    def test_entropy_bounds_mutual_information(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        raw = rng.uniform(0.05, 1.0, size=(4, 6, 3))
        probs = jnp.asarray(raw / raw.sum(axis=-1, keepdims=True))
        entropy = predictive_entropy(probs)
        information = ensemble_mutual_information(probs)
        assert bool(jnp.all(information >= -1e-6))
        assert bool(jnp.all(information <= entropy + 1e-6))
        assert bool(jnp.all(entropy <= jnp.log(3.0) + 1e-6))


class TestCredibility:
    def test_anees_is_one_when_calibrated(self) -> None:
        references = jax.random.normal(jax.random.key(0), (4000, 2))
        covariances = jnp.broadcast_to(jnp.eye(2), (4000, 2, 2))
        assert anees(jnp.zeros((4000, 2)), covariances, references) == pytest.approx(1.0, abs=0.1)

    @pytest.mark.parametrize(("scale", "bound"), [(0.25, "above"), (4.0, "below")])
    def test_anees_reads_over_and_under_confidence(self, scale: float, bound: str) -> None:
        references = jax.random.normal(jax.random.key(1), (2000, 2))
        covariances = jnp.broadcast_to(scale * jnp.eye(2), (2000, 2, 2))
        value = float(anees(jnp.zeros((2000, 2)), covariances, references))
        assert (value > 2.0) if bound == "above" else (value < 0.5)

    def test_nci_is_zero_when_the_covariances_agree(self) -> None:
        references = jax.random.normal(jax.random.key(3), (4000, 2))
        covariances = jnp.broadcast_to(jnp.eye(2), (4000, 2, 2))
        value = non_credibility_index(jnp.zeros((4000, 2)), covariances, references, covariances)
        assert abs(float(value)) < 0.5

    def test_nci_is_positive_when_the_prediction_is_over_confident(self) -> None:
        references = jax.random.normal(jax.random.key(4), (2000, 2))
        predicted = jnp.broadcast_to(0.1 * jnp.eye(2), (2000, 2, 2))
        reference = jnp.broadcast_to(jnp.eye(2), (2000, 2, 2))
        assert (
            float(non_credibility_index(jnp.zeros((2000, 2)), predicted, references, reference))
            > 0.0
        )

    def test_anees_jit_compatible(self) -> None:
        references = jax.random.normal(jax.random.key(5), (100, 2))
        covariances = jnp.broadcast_to(jnp.eye(2), (100, 2, 2))
        assert bool(jnp.isfinite(jax.jit(anees)(jnp.zeros((100, 2)), covariances, references)))

    def test_chi2_interval_matches_scipy(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        lower, upper = chi2_confidence_interval(3, percentile=0.99)
        assert float(lower) == pytest.approx(stats.chi2(df=3).ppf(0.005), rel=1e-6)
        assert float(upper) == pytest.approx(stats.chi2(df=3).ppf(0.995), rel=1e-6)

    def test_chi2_interval_rejects_bad_percentile(self) -> None:
        with pytest.raises(ValueError, match="percentile must be in"):
            chi2_confidence_interval(2, percentile=1.5)
