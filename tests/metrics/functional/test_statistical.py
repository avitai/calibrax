"""Tests for statistical correlation metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from calibrax.core.models import MetricDirection
from calibrax.metrics import MetricRegistry
from calibrax.metrics.functional.statistical import (
    autocorrelation,
    concordance_correlation,
    correlation_preservation,
    kendall_tau,
    pearson_correlation,
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


class TestRankCorrelationsWithTies:
    """Tied values rank by their average rank, as scipy's do."""

    TIED_A = jnp.array([1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0])
    TIED_B = jnp.array([2.0, 1.0, 3.0, 3.0, 5.0, 4.0, 6.0])

    def test_spearman_matches_scipy_with_ties(self) -> None:
        reference = stats.spearmanr(np.asarray(self.TIED_A), np.asarray(self.TIED_B)).statistic

        assert spearman_rank_correlation(self.TIED_A, self.TIED_B) == pytest.approx(
            reference, abs=1e-5
        )

    def test_spearman_does_not_depend_on_the_order_of_the_pairs(self) -> None:
        order = jnp.array([4, 0, 6, 2, 5, 1, 3])

        assert spearman_rank_correlation(self.TIED_A, self.TIED_B) == pytest.approx(
            float(spearman_rank_correlation(self.TIED_A[order], self.TIED_B[order])), abs=1e-6
        )

    def test_kendall_tau_is_tau_b_with_ties(self) -> None:
        reference = stats.kendalltau(np.asarray(self.TIED_A), np.asarray(self.TIED_B)).statistic

        assert kendall_tau(self.TIED_A, self.TIED_B) == pytest.approx(reference, abs=1e-5)

    def test_a_tied_variable_agrees_perfectly_with_itself(self) -> None:
        assert kendall_tau(self.TIED_A, self.TIED_A) == pytest.approx(1.0, abs=1e-5)
        assert spearman_rank_correlation(self.TIED_A, self.TIED_A) == pytest.approx(1.0, abs=1e-5)

    def test_both_trace_under_jit(self) -> None:
        for metric in (spearman_rank_correlation, kendall_tau):
            eager = float(metric(self.TIED_A, self.TIED_B))
            assert float(jax.jit(metric)(self.TIED_A, self.TIED_B)) == pytest.approx(
                eager, abs=1e-6
            )


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


class TestStatisticalMetricRegistration:
    """Tests for statistical metric registration."""

    def test_all_registered(self) -> None:
        registry = MetricRegistry()
        expected = [
            "pearson_correlation",
            "spearman_rank_correlation",
            "kendall_tau",
            "concordance_correlation",
            "skewness",
        ]
        for name in expected:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_statistical_domain(self) -> None:
        registry = MetricRegistry()
        stat_metrics = registry.list_by_domain("statistical")
        # Five, not six: r_squared_adjusted is a general-domain metric, beside r_squared.
        assert len(stat_metrics) == 5

    def test_correlations_are_higher_and_skewness_is_informational(self) -> None:
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


#: ``(a, b, keep)`` triples whose kept subset is what the coefficient must describe. The
#: reference is ``scipy.stats.spearmanr`` on the physically subsetted arrays, so the
#: assertion is against an outside implementation rather than against a rearrangement of
#: this one.
_MASKED_SPEARMAN_CASES = [
    ("kept_positive_infinity", [1.0, 2.0, np.inf, 4.0, 5.0, 6.0], [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]),
    ("kept_negative_infinity", [1.0, 2.0, -np.inf, 4.0, 5.0, 6.0], [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]),
    (
        "two_kept_infinities_tie",
        [1.0, np.inf, np.inf, 4.0, 5.0, 6.0],
        [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
    ),
    ("both_infinities_kept", [-np.inf, 2.0, np.inf, 4.0, 5.0, 6.0], [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]),
    ("excluded_infinity", [1.0, 2.0, 3.0, 4.0, np.inf, 6.0], [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]),
    ("excluded_nan", [1.0, 2.0, 3.0, 4.0, np.nan, 6.0], [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]),
    ("ties_among_kept", [1.0, 2.0, 2.0, 4.0, 5.0, 6.0], [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]),
    ("finite_only", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]),
]

#: The last two elements are excluded in every case above.
_MASKED_SPEARMAN_KEEP = [True, True, True, True, False, False]


class TestMaskedSpearmanRanksTheKeptSubset:
    """A mask selects a subsample, so the coefficient is the one that subsample has.

    Sending the excluded elements to ``+inf`` so they sort last makes them tie with a kept
    ``+inf``, and ``rankdata`` gives tied values their average rank, so such an element
    carries a rank averaged across positions that are not in the sample unless the tie is
    corrected for. On the first case below the uncorrected value is ``0.8315`` where the
    subsample's coefficient, and scipy's, is ``0.8000``.

    The correction is exact. A kept element below ``+inf`` has the excluded neither under it
    nor tied with it, so its rank is already the subset's; a kept ``+inf`` is tied with every
    excluded element, and averaging over that tie adds ``excluded / 2``. The reference for
    every case here is ``scipy.stats.spearmanr`` on the physically subsetted arrays, so these
    assert against an outside implementation rather than against a rearrangement of this one.
    """

    @pytest.mark.parametrize(
        ("name", "a", "b"), _MASKED_SPEARMAN_CASES, ids=[c[0] for c in _MASKED_SPEARMAN_CASES]
    )
    def test_it_equals_scipy_on_the_physical_subset(
        self, name: str, a: list[float], b: list[float]
    ) -> None:
        keep = np.asarray(_MASKED_SPEARMAN_KEEP)
        reference = stats.spearmanr(np.asarray(a)[keep], np.asarray(b)[keep]).statistic
        measured = spearman_rank_correlation(jnp.asarray(a), jnp.asarray(b), mask=jnp.asarray(keep))
        assert float(measured) == pytest.approx(float(reference), abs=1e-6), name

    def test_a_kept_nan_still_reads_nan(self) -> None:
        """A ``NaN`` inside the sample is missing data, not a value to rank around."""
        a = jnp.asarray([1.0, 2.0, jnp.nan, 4.0, 5.0, 6.0])
        b = jnp.asarray([2.0, 1.0, 4.0, 3.0, 6.0, 5.0])
        keep = jnp.asarray(_MASKED_SPEARMAN_KEEP)
        assert bool(jnp.isnan(spearman_rank_correlation(a, b, mask=keep)))

    def test_a_mask_shaped_like_a_two_dimensional_input(self) -> None:
        """The mask folds at the input's own shape, so a 2-D mask over 2-D data works."""
        rng = np.random.default_rng(0)
        a2 = rng.normal(size=(4, 3))
        b2 = a2 + rng.normal(scale=0.5, size=(4, 3))
        keep2 = rng.random((4, 3)) > 0.3
        reference = stats.spearmanr(a2.ravel()[keep2.ravel()], b2.ravel()[keep2.ravel()]).statistic
        measured = spearman_rank_correlation(
            jnp.asarray(a2), jnp.asarray(b2), mask=jnp.asarray(keep2)
        )
        assert float(measured) == pytest.approx(float(reference), abs=1e-5)

    def test_it_traces_under_jit_and_vmap_over_masks(self) -> None:
        """One mask per bin, mapped rather than looped, which is how a curve is built."""
        rng = np.random.default_rng(1)
        a = jnp.asarray(rng.normal(size=8))
        b = jnp.asarray(a + rng.normal(scale=0.5, size=8))
        masks = jnp.stack([jnp.arange(8) < kept for kept in (4, 6, 8)])
        curve = jax.jit(jax.vmap(lambda m: spearman_rank_correlation(a, b, mask=m)))(masks)
        assert curve.shape == (3,)
        assert bool(jnp.all(jnp.isfinite(curve)))
        for row, kept in zip(range(3), (4, 6, 8), strict=True):
            keep = np.arange(8) < kept
            reference = stats.spearmanr(np.asarray(a)[keep], np.asarray(b)[keep]).statistic
            assert float(curve[row]) == pytest.approx(float(reference), abs=1e-5)

    def test_an_all_true_mask_agrees_with_passing_no_mask(self) -> None:
        """The masked path must land where the unmasked path already is."""
        a = jnp.asarray([1.0, 2.0, jnp.inf, 4.0, 5.0, 6.0])
        b = jnp.asarray([2.0, 1.0, 4.0, 3.0, 6.0, 5.0])
        full = jnp.ones(6, dtype=bool)
        assert float(spearman_rank_correlation(a, b, mask=full)) == pytest.approx(
            float(spearman_rank_correlation(a, b)), abs=1e-6
        )
