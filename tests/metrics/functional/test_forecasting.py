"""Tests for the forecasting domain.

The closed-form cases come from the source references (Gneiting & Raftery 2007,
Ferro 2014, Hamill 2001, Fortin et al. 2014, Epstein 1969, Murphy 1971 and 1973,
Ferro, Richardson & Weigel 2008); the property tests pin the relations between the
estimators.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from calibrax.metrics.functional.forecasting import (
    energy_score,
    ensemble_ranked_probability_score,
    event_reliability,
    fair_crps,
    pit_histogram,
    rank_histogram,
    ranked_probability_score,
    ranked_probability_skill_score,
    spread_skill_ratio,
)
from calibrax.metrics.functional.regression import crps


class TestFairCrps:
    def test_two_member_closed_form(self) -> None:
        # error term (|1-2| + |3-2|) / 2 = 1; fair spread |1-3| * 2 / (2 * 1) = 2 -> 1 - 0.5 * 2
        assert fair_crps(jnp.array([[1.0, 3.0]]), jnp.array([2.0])) == pytest.approx(0.0, abs=1e-6)

    def test_is_below_the_empirical_crps_for_finite_ensembles(self) -> None:
        rng = np.random.default_rng(0)
        predictions = jnp.asarray(rng.standard_normal((16, 5)))
        targets = jnp.asarray(rng.standard_normal(16))
        assert float(fair_crps(predictions, targets)) < float(crps(predictions, targets))

    @given(
        n_members=st.integers(min_value=2, max_value=12),
        seed=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=25, deadline=None)
    def test_equals_empirical_crps_with_the_pairwise_term_rescaled(
        self, n_members: int, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        predictions = jnp.asarray(rng.standard_normal((6, n_members)))
        targets = jnp.asarray(rng.standard_normal(6))
        empirical = float(crps(predictions, targets))
        error = float(jnp.mean(jnp.abs(predictions - targets[:, None])))
        spread_biased = 2.0 * (error - empirical)
        expected = error - 0.5 * spread_biased * n_members / (n_members - 1)
        assert float(fair_crps(predictions, targets)) == pytest.approx(expected, abs=1e-5)

    def test_rejects_a_single_member(self) -> None:
        with pytest.raises(ValueError, match="at least two ensemble members"):
            fair_crps(jnp.ones((4, 1)), jnp.ones(4))

    def test_jit_compatible(self) -> None:
        predictions = jnp.asarray(np.random.default_rng(0).standard_normal((8, 4)))
        targets = jnp.zeros(8)
        assert jax.jit(fair_crps)(predictions, targets) == pytest.approx(
            float(fair_crps(predictions, targets)), abs=1e-6
        )


class TestEnergyScore:
    def test_two_member_closed_form(self) -> None:
        ensemble = jnp.array([[[1.0, 0.0], [3.0, 0.0]]])
        targets = jnp.array([[2.0, 0.0]])
        # mean ||X_i - y|| = 1; mean pairwise ||X_i - X_j|| = (0 + 2 + 2 + 0) / 4 = 1
        assert energy_score(ensemble, targets) == pytest.approx(0.5, abs=1e-6)

    def test_reduces_to_crps_for_one_output(self) -> None:
        rng = np.random.default_rng(1)
        predictions = jnp.asarray(rng.standard_normal((10, 6)))
        targets = jnp.asarray(rng.standard_normal(10))
        assert energy_score(predictions[:, :, None], targets[:, None]) == pytest.approx(
            float(crps(predictions, targets)), abs=1e-5
        )

    def test_rejects_wrong_target_shape(self) -> None:
        with pytest.raises(ValueError, match="targets must have shape"):
            energy_score(jnp.ones((3, 4, 2)), jnp.ones((3, 3)))


class TestRankHistogram:
    def test_counts_the_target_position(self) -> None:
        ensemble = jnp.array([[1.0, 2.0, 3.0]] * 3)
        counts = rank_histogram(ensemble, jnp.array([0.0, 2.5, 4.0]))
        assert counts.shape == (4,)
        assert counts.tolist() == [1, 0, 1, 1]

    def test_counts_sum_to_the_sample_count(self) -> None:
        rng = np.random.default_rng(2)
        ensemble = jnp.asarray(rng.standard_normal((50, 7)))
        counts = rank_histogram(ensemble, jnp.asarray(rng.standard_normal(50)))
        assert int(jnp.sum(counts)) == 50


class TestSpreadSkillRatio:
    def test_is_one_for_a_calibrated_ensemble(self) -> None:
        rng = np.random.default_rng(1234)
        ensemble = jnp.asarray(rng.standard_normal((4096, 20)))
        targets = jnp.asarray(rng.standard_normal(4096))
        assert spread_skill_ratio(ensemble, targets) == pytest.approx(1.0, abs=0.05)

    def test_small_ensembles_are_not_biased_low(self) -> None:
        rng = np.random.default_rng(5678)
        ensemble = jnp.asarray(rng.standard_normal((8192, 5)))
        targets = jnp.asarray(rng.standard_normal(8192))
        assert spread_skill_ratio(ensemble, targets) == pytest.approx(1.0, abs=0.1)

    def test_below_one_when_under_dispersed(self) -> None:
        rng = np.random.default_rng(0)
        ensemble = jnp.asarray(0.1 * rng.standard_normal((2048, 30)))
        targets = jnp.asarray(rng.standard_normal(2048))
        assert float(spread_skill_ratio(ensemble, targets)) < 0.5


class TestPitHistogram:
    def test_is_flat_for_a_calibrated_gaussian(self) -> None:
        rng = np.random.default_rng(0)
        n = 4096
        counts = pit_histogram(jnp.zeros(n), jnp.ones(n), jnp.asarray(rng.standard_normal(n)))
        assert counts.shape == (10,)
        assert bool(jnp.all(jnp.abs(counts - n / 10) < 0.25 * n / 10))

    def test_bin_count_is_configurable(self) -> None:
        counts = pit_histogram(jnp.zeros(8), jnp.ones(8), jnp.zeros(8), num_bins=4)
        assert counts.shape == (4,)
        assert int(jnp.sum(counts)) == 8


class TestRankedProbabilityScore:
    def test_three_class_closed_form(self) -> None:
        # cumulative probs (0.5, 0.8, 1.0) against cumulative observation (0, 1, 1)
        expected = 0.5**2 + 0.2**2 + 0.0**2
        assert ranked_probability_score(
            jnp.array([[0.5, 0.3, 0.2]]), jnp.array([1])
        ) == pytest.approx(expected, abs=1e-6)

    def test_is_zero_for_a_perfect_forecast(self) -> None:
        assert ranked_probability_score(
            jnp.array([[0.0, 1.0, 0.0]]), jnp.array([1])
        ) == pytest.approx(0.0, abs=1e-6)


class TestEventReliability:
    def test_zero_for_a_calibrated_forecast(self) -> None:
        probs = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        assert event_reliability(probs, probs, num_bins=5) == pytest.approx(0.0, abs=1e-6)

    def test_known_squared_gap(self) -> None:
        # one bin with forecasts 0.1 and events (0, 1): gap (0.1 - 0.5)^2 = 0.16
        assert event_reliability(
            jnp.array([0.1, 0.1]), jnp.array([0.0, 1.0]), num_bins=5
        ) == pytest.approx(0.16, abs=1e-6)

    def test_is_non_negative(self) -> None:
        probs = jnp.linspace(0.05, 0.95, 10)
        events = jnp.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1], dtype=jnp.float32)
        assert float(event_reliability(probs, events, num_bins=5)) >= 0.0


class TestEnsembleRankedProbabilityScore:
    def test_biased_single_threshold_gap(self) -> None:
        # members {0, 1}, target 0, threshold 0.5: predicted CDF 0.5 vs observed 1
        out = ensemble_ranked_probability_score(
            jnp.array([[0.0, 1.0]]), jnp.array([0.0]), thresholds=jnp.array([0.5]), fair=False
        )
        assert out == pytest.approx(0.25, abs=1e-6)

    def test_fair_form_subtracts_the_finite_sample_bias(self) -> None:
        out = ensemble_ranked_probability_score(
            jnp.array([[0.0, 1.0]]), jnp.array([0.0]), thresholds=jnp.array([0.5]), fair=True
        )
        assert out == pytest.approx(0.0, abs=1e-6)

    def test_fair_expectation_does_not_depend_on_ensemble_size(self) -> None:
        rng = np.random.default_rng(0)
        thresholds = jnp.linspace(-2.0, 2.0, 9)
        targets = jnp.asarray(rng.standard_normal(2048))

        def mean_score(n_members: int) -> float:
            samples = jnp.asarray(rng.standard_normal((2048, n_members)))
            return float(ensemble_ranked_probability_score(samples, targets, thresholds=thresholds))

        assert abs(mean_score(8) - mean_score(64)) < 0.05

    def test_jit_compatible(self) -> None:
        thresholds = jnp.array([0.5])
        out = jax.jit(lambda s, t: ensemble_ranked_probability_score(s, t, thresholds=thresholds))(
            jnp.array([[0.0, 1.0]]), jnp.array([0.0])
        )
        assert out.shape == ()


class TestRankedProbabilitySkillScore:
    @pytest.mark.parametrize(
        ("rps", "reference", "expected"),
        [(0.0, 0.5, 1.0), (0.4, 0.4, 0.0), (1.0, 0.5, -1.0)],
    )
    def test_closed_form(self, rps: float, reference: float, expected: float) -> None:
        assert ranked_probability_skill_score(rps, reference) == pytest.approx(expected, abs=1e-6)

    def test_broadcasts(self) -> None:
        skills = ranked_probability_skill_score(
            jnp.array([0.0, 0.4, 1.0]), jnp.array([0.5, 0.4, 0.5])
        )
        assert skills.shape == (3,)
