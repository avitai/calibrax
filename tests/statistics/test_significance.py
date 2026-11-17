"""Tests for calibrax.statistics.significance module."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import false_discovery_control, nct, t as student_t, wilcoxon
from substrax.testing import TraceCounter

from calibrax.statistics import (
    alpha_investing,
    AlphaInvestingResult,
    benjamini_hochberg_adjust,
    minimum_detectable_effect,
    statistical_power,
)
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

    def test_no_nonzero_difference_is_the_exact_null_without_a_warning(self) -> None:
        """Every difference zero leaves no pair to rank once zeros are dropped (``"wilcox"``).

        The exact null distribution of an empty sample puts all its mass on a statistic of 0, so
        the p-value is 1, which is ``scipy.stats.wilcoxon(..., method="exact")``'s answer. Its
        ``method="auto"`` divides 0 by 0 on the way and warns.
        """
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = paired_significance_test(data, data)
        reference = wilcoxon(data, data, zero_method="wilcox", method="exact")
        assert result.p_value == float(reference.pvalue) == 1.0
        assert result.statistic == float(reference.statistic) == 0.0

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


class TestBenjaminiHochbergAdjust:
    """Step-up Benjamini-Hochberg q-values (JRSS-B 1995)."""

    def test_matches_scipy_false_discovery_control_with_and_without_ties(self) -> None:
        rng = np.random.default_rng(11)
        for p_np in [
            np.array([0.005, 0.011, 0.04, 0.04, 0.20, 0.85], dtype=np.float32),
            rng.uniform(0.0, 1.0, size=64).astype(np.float32),
            np.array([1.0, 1.0, 0.5], dtype=np.float32),
        ]:
            expected = false_discovery_control(p_np.astype(np.float64), method="bh")
            actual = np.asarray(benjamini_hochberg_adjust(jnp.asarray(p_np)))
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    def test_empty_out_of_bounds_and_jit_vmap(self) -> None:
        empty = benjamini_hochberg_adjust(jnp.zeros(0, dtype=jnp.float32))
        assert empty.shape == (0,)

        bad = benjamini_hochberg_adjust(jnp.array([0.1, -0.2, 0.5], dtype=jnp.float32))
        assert bool(jnp.all(jnp.isnan(bad)))

        counter = TraceCounter()
        compiled = jax.jit(counter.wrap(benjamini_hochberg_adjust))
        batch = jnp.array([[0.01, 0.04, 0.2], [0.5, 0.02, 0.9]], dtype=jnp.float32)
        with counter.expect(new_traces=1):
            compiled(batch[0])
        with counter.expect(new_traces=0):
            compiled(batch[1])
        vmapped = jax.vmap(benjamini_hochberg_adjust)(batch)
        for row_idx in range(2):
            expected_row = false_discovery_control(np.asarray(batch[row_idx], dtype=np.float64))
            np.testing.assert_allclose(np.asarray(vmapped[row_idx]), expected_row, atol=1e-6)


class TestAlphaInvesting:
    """Foster and Stine (JRSS-B 2008) sequential mFDR control."""

    def test_exact_wealth_recursion_and_non_negativity(self) -> None:
        # w_0 = 0.05, gamma = 0.5, payout = 0.05:
        # Step 0: alpha_0 = 0.025 / 1.025 = 1/41 ~ 0.02439024. p_0 = 0.01 <= alpha_0 -> REJECT!
        #         w_1 = 0.05 + 0.05 = 0.10.
        # Step 1: alpha_1 = 0.05 / 1.05 = 1/21 ~ 0.04761905. p_1 = 0.20 > alpha_1 -> ACCEPT!
        #         w_2 = 0.10 - 0.05 = 0.05.
        # Step 2: alpha_2 = 0.025 / 1.025 = 1/41. p_2 = 0.50 > alpha_2 -> ACCEPT!
        #         w_3 = 0.025.
        p_vals = jnp.array([0.01, 0.20, 0.50], dtype=jnp.float32)
        res = alpha_investing(p_vals, wealth=0.05, pay_out=0.05, invest_fraction=0.5)
        assert isinstance(res, AlphaInvestingResult)
        assert res.rejected.tolist() == [True, False, False]
        np.testing.assert_allclose(
            np.asarray(res.thresholds),
            np.array([1.0 / 41.0, 1.0 / 21.0, 1.0 / 41.0]),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            np.asarray(res.wealth),
            np.array([0.10, 0.05, 0.025]),
            atol=1e-7,
        )
        assert bool(jnp.all(res.wealth >= 0.0))

    def test_validation_empty_and_jit_pytree_registration(self) -> None:
        empty_res = alpha_investing(jnp.zeros(0, dtype=jnp.float32))
        assert empty_res.rejected.shape == (0,)
        assert empty_res.thresholds.shape == (0,)
        assert empty_res.wealth.shape == (0,)

        with pytest.raises(ValueError, match="wealth"):
            alpha_investing(jnp.array([0.01]), wealth=0.0)
        with pytest.raises(ValueError, match="pay_out"):
            alpha_investing(jnp.array([0.01]), pay_out=-0.01)
        with pytest.raises(ValueError, match="invest_fraction"):
            alpha_investing(jnp.array([0.01]), invest_fraction=1.0)

        counter = TraceCounter()
        compiled = jax.jit(counter.wrap(alpha_investing))
        with counter.expect(new_traces=1):
            out = compiled(jnp.array([0.001, 0.5, 0.002, 0.8], dtype=jnp.float32))
        with counter.expect(new_traces=0):
            compiled(jnp.array([0.4, 0.3, 0.2, 0.1], dtype=jnp.float32))
        assert out.rejected.tolist() == [True, False, True, False]


class TestPowerAndMDE:
    """Exact non-central Student-t statistical power and minimum detectable effect."""

    def test_statistical_power_matches_scipy_nct_two_sample_and_paired(self) -> None:
        # Independent two-sample: n=30 per group -> df=58, ncp = d * sqrt(15).
        d = 0.65
        n = 30
        alpha = 0.05
        df_ind = 2 * n - 2
        ncp_ind = d * np.sqrt(n / 2.0)
        t_crit_ind = float(student_t.ppf(1.0 - alpha / 2.0, df_ind))
        expected_ind = float(
            nct.sf(t_crit_ind, df_ind, ncp_ind) + nct.cdf(-t_crit_ind, df_ind, ncp_ind)
        )
        assert statistical_power(d, n, alpha=alpha, paired=False) == pytest.approx(
            expected_ind, abs=1e-12
        )

        # Paired: n=25 pairs -> df=24, ncp = d * sqrt(25).
        n_pair = 25
        df_pair = n_pair - 1
        ncp_pair = d * np.sqrt(n_pair)
        t_crit_pair = float(student_t.ppf(1.0 - alpha / 2.0, df_pair))
        expected_pair = float(
            nct.sf(t_crit_pair, df_pair, ncp_pair) + nct.cdf(-t_crit_pair, df_pair, ncp_pair)
        )
        assert statistical_power(d, n_pair, alpha=alpha, paired=True) == pytest.approx(
            expected_pair, abs=1e-12
        )

    @pytest.mark.parametrize("paired", [False, True])
    @pytest.mark.parametrize("alternative", ["two-sided", "greater"])
    def test_minimum_detectable_effect_inverts_statistical_power(
        self, paired: bool, alternative: str
    ) -> None:
        mde = minimum_detectable_effect(
            40,
            alpha=0.05,
            power=0.80,
            paired=paired,
            alternative=alternative,  # type: ignore[arg-type]
        )
        recovered = statistical_power(
            mde,
            40,
            alpha=0.05,
            paired=paired,
            alternative=alternative,  # type: ignore[arg-type]
        )
        assert recovered == pytest.approx(0.80, abs=1e-10)
        assert minimum_detectable_effect(
            40,
            alpha=0.05,
            power=0.80,
            paired=paired,
            alternative=alternative,  # type: ignore[arg-type]
            std=2.5,
        ) == pytest.approx(2.5 * mde, abs=1e-11)

    def test_invalid_power_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="n must be at least 2"):
            statistical_power(0.5, 1)
        with pytest.raises(ValueError, match="alpha"):
            statistical_power(0.5, 20, alpha=1.0)
        with pytest.raises(ValueError, match="power"):
            minimum_detectable_effect(20, alpha=0.05, power=0.04)
        with pytest.raises(ValueError, match="std"):
            minimum_detectable_effect(20, std=0.0)
