"""Tests for calibrax.statistics.significance module."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import (
    f as f_dist,
    false_discovery_control,
    nct,
    studentized_range,
    t as student_t,
    ttest_rel,
    tukey_hsd,
    wilcoxon,
)
from substrax.testing import TraceCounter

from calibrax.statistics import (
    alpha_investing,
    AlphaInvestingResult,
    benjamini_hochberg_adjust,
    minimum_detectable_effect,
    nadeau_bengio_t_test,
    NadeauBengioResult,
    RepeatedMeasuresTukeyResult,
    rm_tukey_hsd,
    statistical_power,
    TukeyPairwiseComparison,
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

        seq_list = benjamini_hochberg_adjust([0.01, 0.04, 0.2])  # type: ignore[arg-type]
        seq_tuple = benjamini_hochberg_adjust((0.01, 0.04, 0.2))  # type: ignore[arg-type]
        expected_seq = false_discovery_control(np.array([0.01, 0.04, 0.2], dtype=np.float64))
        np.testing.assert_allclose(np.asarray(seq_list), expected_seq, atol=1e-6)
        np.testing.assert_allclose(np.asarray(seq_tuple), expected_seq, atol=1e-6)

        with jax.enable_x64(True):
            assert benjamini_hochberg_adjust(jnp.array([0.01, 0.2], dtype=jnp.float32)).dtype == (
                jnp.float32
            )
            assert benjamini_hochberg_adjust(jnp.array([0.01, 0.2], dtype=jnp.float64)).dtype == (
                jnp.float64
            )


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

        for bad_w in (0.0, -0.1, float("nan"), np.array(-1.0, dtype=np.float32), np.array(np.nan)):
            with pytest.raises(ValueError, match="wealth"):
                alpha_investing(jnp.array([0.01]), wealth=bad_w)
        for bad_p in (0.0, -0.01, float("nan"), np.array(-0.5, dtype=np.float32), np.array(np.nan)):
            with pytest.raises(ValueError, match="pay_out"):
                alpha_investing(jnp.array([0.01]), pay_out=bad_p)
        for bad_g in (
            0.0,
            1.0,
            1.5,
            float("nan"),
            np.array(1.5, dtype=np.float32),
            np.array(np.nan),
        ):
            with pytest.raises(ValueError, match="invest_fraction"):
                alpha_investing(jnp.array([0.01]), invest_fraction=bad_g)

        seq_ai = alpha_investing([0.01, 0.20, 0.50])  # type: ignore[arg-type]
        tup_ai = alpha_investing((0.01, 0.20, 0.50))  # type: ignore[arg-type]
        assert seq_ai.rejected.tolist() == [True, False, False]
        assert tup_ai.rejected.tolist() == [True, False, False]

        traced_ai = jax.jit(
            jax.vmap(
                lambda w, po, g: alpha_investing(
                    jnp.array([0.01, 0.20], dtype=jnp.float32),
                    wealth=w,
                    pay_out=po,
                    invest_fraction=g,
                )
            )
        )(
            jnp.array([0.05, 0.10], dtype=jnp.float32),
            jnp.array([0.05, 0.05], dtype=jnp.float32),
            jnp.array([0.5, 0.5], dtype=jnp.float32),
        )
        assert traced_ai.rejected.shape == (2, 2)

        with jax.enable_x64(True):
            ai32 = alpha_investing(jnp.array([0.01, 0.2], dtype=jnp.float32))
            ai64 = alpha_investing(jnp.array([0.01, 0.2], dtype=jnp.float64))
            assert ai32.wealth.dtype == jnp.float32
            assert ai64.wealth.dtype == jnp.float64

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


class TestNadeauBengioTTest:
    """Nadeau & Bengio (2003) variance-corrected paired t-test and Student-t ROPE."""

    def test_matches_nadeau_bengio_eq14_and_baycomp_rope_posteriors(self) -> None:
        rng = np.random.default_rng(2025)
        scores_b = rng.normal(loc=0.78, scale=0.03, size=25)
        scores_a = scores_b + rng.normal(loc=0.025, scale=0.02, size=25)

        # 5x5 repeated CV: n_folds=5 <-> test_fraction=0.2 <-> n_train=80, n_test=20 (rho = 0.25).
        res_folds = nadeau_bengio_t_test(scores_a, scores_b, n_folds=5, alpha=0.05, rope=0.01)
        res_frac = nadeau_bengio_t_test(
            scores_a, scores_b, test_fraction=0.2, alpha=0.05, rope=0.01
        )
        res_counts = nadeau_bengio_t_test(
            scores_a, scores_b, n_train=80, n_test=20, alpha=0.05, rope=0.01
        )
        assert isinstance(res_folds, NadeauBengioResult)
        assert res_frac.statistic == pytest.approx(res_folds.statistic, abs=1e-15)
        assert res_counts.statistic == pytest.approx(res_folds.statistic, abs=1e-15)
        assert res_counts.p_value == pytest.approx(res_folds.p_value, abs=1e-15)

        # Exact relation to naive paired t-test: SE_nb = SE_naive * sqrt(1 + J * rho).
        naive = ttest_rel(scores_a, scores_b)
        inflation = np.sqrt(1.0 + 25.0 * 0.25)
        assert res_folds.df == 24
        assert res_folds.statistic == pytest.approx(float(naive.statistic) / inflation, abs=1e-14)
        expected_p = float(2.0 * student_t.sf(abs(res_folds.statistic), 24))
        assert res_folds.p_value == pytest.approx(expected_p, abs=1e-15)

        # Verify CI and ROPE probabilities against quadrature of Student-t posterior.
        p_left, p_rope, p_right = res_folds.rope_probabilities
        assert p_left + p_rope + p_right == pytest.approx(1.0, abs=1e-15)
        quad_rope, _ = quad(
            lambda x: student_t.pdf(x, df=24, loc=res_folds.mean_diff, scale=res_folds.se),
            -0.01,
            0.01,
        )
        assert p_rope == pytest.approx(quad_rope, rel=1e-11, abs=1e-14)

        # Round-trip serialization.
        reloaded = NadeauBengioResult.from_dict(res_folds.to_dict())
        assert reloaded.statistic == pytest.approx(res_folds.statistic, abs=1e-15)
        assert reloaded.rope_probabilities == pytest.approx(res_folds.rope_probabilities)

    def test_one_sided_alternatives_zero_variance_and_extreme_tail_rope_precision(self) -> None:
        a = np.array([0.80, 0.82, 0.79, 0.85, 0.84])
        b = np.array([0.75, 0.76, 0.74, 0.78, 0.77])
        two = nadeau_bengio_t_test(a, b, n_folds=5, alternative="two-sided")
        greater = nadeau_bengio_t_test(a, b, n_folds=5, alternative="greater")
        less = nadeau_bengio_t_test(a, b, n_folds=5, alternative="less")
        assert greater.p_value == pytest.approx(two.p_value / 2.0, abs=1e-15)
        assert less.p_value == pytest.approx(1.0 - greater.p_value, abs=1e-15)
        assert np.isinf(greater.ci_upper)
        assert greater.ci_upper > 0.0
        assert np.isinf(less.ci_lower)
        assert less.ci_lower < 0.0

        # Extreme-tail ROPE: mean_diff = 12.0, se ~ 0.05, rope = 0.5 -> naive 1 - p_left - p_right
        # suffers 1.0 - 1.0 catastrophic cancellation in float64, whereas direct tail subtraction
        # matches quadrature accurately.
        far_a = b + 12.0 + np.array([0.01, -0.02, 0.0, 0.02, -0.01])
        far_res = nadeau_bengio_t_test(far_a, b, n_folds=5, rope=0.5)
        _, p_rope_far, _ = far_res.rope_probabilities
        quad_far, _ = quad(
            lambda x: student_t.pdf(x, df=4, loc=far_res.mean_diff, scale=far_res.se),
            -0.5,
            0.5,
        )
        assert p_rope_far > 0.0
        assert p_rope_far == pytest.approx(quad_far, rel=1e-9)

        # Zero variance (constant difference): exact step-function behavior without warnings.
        same = nadeau_bengio_t_test(a, a, n_folds=5, rope=0.0)
        assert same.statistic == 0.0
        assert same.p_value == 1.0
        assert same.significant is False
        assert same.rope_probabilities == (0.5, 0.0, 0.5)

        same_rope = nadeau_bengio_t_test(a, a, n_folds=5, rope=0.02)
        assert same_rope.rope_probabilities == (0.0, 1.0, 0.0)

        shifted = nadeau_bengio_t_test(a + 0.1, a, n_folds=5, rope=0.02)
        assert np.isinf(shifted.statistic)
        assert shifted.statistic > 0.0
        assert shifted.p_value == 0.0
        assert shifted.significant is True
        assert shifted.rope_probabilities == (0.0, 0.0, 1.0)

    def test_scale_invariance_at_tiny_and_large_magnitudes(self) -> None:
        rng = np.random.default_rng(42)
        sa = rng.normal(0.82, 0.03, size=10)
        sb = rng.normal(0.79, 0.03, size=10)
        base = nadeau_bengio_t_test(sa, sb, n_folds=2, rope=0.01)
        for scale in (1e-16, 1e-8, 1e8):
            scaled = nadeau_bengio_t_test(sa * scale, sb * scale, n_folds=2, rope=0.01 * scale)
            assert scaled.se > 0.0
            assert scaled.statistic == pytest.approx(base.statistic, rel=1e-12, abs=1e-12)
            assert scaled.p_value == pytest.approx(base.p_value, abs=1e-12)
            assert scaled.significant == base.significant
            assert scaled.rope_probabilities == pytest.approx(base.rope_probabilities, abs=1e-12)

        zeros = np.zeros(5, dtype=np.float64)
        zero_res = nadeau_bengio_t_test(zeros, zeros, n_folds=5)
        assert zero_res.statistic == 0.0
        assert zero_res.p_value == 1.0

        # Sub-ULP arithmetic round-trip noise must not trigger a false-positive t = inf.
        probe = np.array([1.0, 2.0, 3.0, 7.0, 11.0], dtype=np.float64)
        roundtrip = nadeau_bengio_t_test((probe / 3.0) * 3.0, probe, n_folds=5)
        assert roundtrip.statistic == 0.0
        assert roundtrip.p_value == 1.0

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            nadeau_bengio_t_test([0.5], [0.4], n_folds=5)
        with pytest.raises(ValueError, match="shape"):
            nadeau_bengio_t_test([0.5, 0.6], [0.4, 0.5, 0.6], n_folds=5)
        with pytest.raises(ValueError, match="exactly one"):
            nadeau_bengio_t_test([0.5, 0.6], [0.4, 0.5])
        with pytest.raises(ValueError, match="exactly one"):
            nadeau_bengio_t_test([0.5, 0.6], [0.4, 0.5], n_folds=5, test_fraction=0.2)
        with pytest.raises(ValueError, match="Both n_train and n_test"):
            nadeau_bengio_t_test([0.5, 0.6], [0.4, 0.5], n_train=80)
        with pytest.raises(ValueError, match="rope"):
            nadeau_bengio_t_test([0.5, 0.6], [0.4, 0.5], n_folds=5, rope=-0.01)


class TestRepeatedMeasuresTukeyHSD:
    """Two-way repeated-measures ANOVA + Tukey HSD (Ash et al., JCIM 2025)."""

    def test_matches_ash_2025_rm_anova_tukey_and_is_block_shift_invariant(self) -> None:
        rng = np.random.default_rng(19)
        n_blocks, n_models = 12, 4
        block_effects = rng.normal(loc=0.0, scale=0.35, size=(n_blocks, 1))
        model_effects = np.array([[0.85, 0.84, 0.72, 0.65]])
        noise = rng.normal(loc=0.0, scale=0.025, size=(n_blocks, n_models))
        scores = block_effects + model_effects + noise
        names = ("gnn", "rf", "ridge", "dummy")

        res = rm_tukey_hsd(scores, model_names=names, alpha=0.05, higher_is_better=True)
        assert isinstance(res, RepeatedMeasuresTukeyResult)
        assert res.df_resid == (n_blocks - 1) * (n_models - 1)
        assert len(res.comparisons) == n_models * (n_models - 1) // 2
        assert res.rank_order[0] == "gnn"
        assert "gnn" in res.top_tier
        assert "rf" in res.top_tier
        assert "dummy" not in res.top_tier

        # Direct primary-source formula check (Ash et al. 2025 / studentized_range):
        resid = (
            scores
            - scores.mean(axis=1, keepdims=True)
            - scores.mean(axis=0, keepdims=True)
            + scores.mean()
        )
        expected_mse = float(np.sum(resid**2) / ((n_blocks - 1) * (n_models - 1)))
        assert res.mse == pytest.approx(expected_mse, abs=1e-15)
        ms_model = float(n_blocks * np.sum((scores.mean(axis=0) - scores.mean()) ** 2) / 3)
        expected_f = ms_model / expected_mse
        assert res.f_statistic == pytest.approx(expected_f, rel=1e-13)
        assert res.anova_p_value == pytest.approx(
            float(f_dist.sf(expected_f, 3, res.df_resid)), abs=1e-14
        )

        # Adding arbitrary per-block difficulty offsets leaves MSE, F, q, p_adj, and top_tier invariant.
        shifted_scores = scores + np.linspace(-10.0, 25.0, n_blocks)[:, None]
        res_shifted = rm_tukey_hsd(shifted_scores, model_names=names, alpha=0.05)
        assert res_shifted.mse == pytest.approx(res.mse, rel=1e-12, abs=1e-14)
        assert res_shifted.f_statistic == pytest.approx(res.f_statistic, rel=1e-12)
        assert res_shifted.top_tier == res.top_tier
        for c1, c2 in zip(res.comparisons, res_shifted.comparisons, strict=True):
            assert isinstance(c1, TukeyPairwiseComparison)
            assert c2.p_value == pytest.approx(c1.p_value, abs=1e-12)

        # Verify exact studentized_range parameterization agreement against scipy.stats.tukey_hsd
        # on block-centered data (accounting for df_rm = (B-1)(M-1) vs df_1way = M(B-1)).
        bc = scores - scores.mean(axis=1, keepdims=True)
        scipy_1w = tukey_hsd(*[bc[:, m] for m in range(n_models)])
        scale_df = np.sqrt((n_models - 1) / n_models)
        comp_01 = res.comparisons[0]
        q_1w = abs(comp_01.q_statistic) / scale_df
        assert float(studentized_range.sf(q_1w, n_models, n_models * (n_blocks - 1))) == (
            pytest.approx(float(scipy_1w.pvalue[0, 1]), abs=1e-14)
        )

        # Serialization round-trip.
        reloaded = RepeatedMeasuresTukeyResult.from_dict(res.to_dict())
        assert reloaded.top_tier == res.top_tier
        assert len(reloaded.comparisons) == len(res.comparisons)

    def test_lower_is_better_rope_and_zero_residual_mse(self) -> None:
        # Deterministic additive errors: RMSE = block_base + [0.10, 0.105, 0.30] -> MSE == 0.
        block_base = np.array([[1.0], [1.5], [2.0], [2.5]])
        rmse_matrix = block_base + np.array([[0.10, 0.105, 0.30]])
        res_strict = rm_tukey_hsd(
            rmse_matrix,
            model_names=("m_best", "m_close", "m_bad"),
            higher_is_better=False,
            rope=0.0,
        )
        assert res_strict.mse == 0.0
        assert res_strict.rank_order == ("m_best", "m_close", "m_bad")
        assert res_strict.top_tier == ("m_best",)

        # With practical equivalence rope=0.01, m_close (diff 0.005 <= 0.01) joins top_tier.
        res_rope = rm_tukey_hsd(
            rmse_matrix,
            model_names=("m_best", "m_close", "m_bad"),
            higher_is_better=False,
            rope=0.01,
        )
        assert res_rope.top_tier == ("m_best", "m_close")

    def test_scale_invariance_at_tiny_and_large_magnitudes(self) -> None:
        rng = np.random.default_rng(42)
        scores_2 = rng.normal(loc=[0.75, 0.70], scale=0.08, size=(10, 2))
        base = rm_tukey_hsd(scores_2, model_names=("A", "B"), alpha=0.05, rope=0.02)
        t_rel = ttest_rel(scores_2[:, 0], scores_2[:, 1])

        for scale in (1e-16, 1e-8, 1e8):
            scaled = rm_tukey_hsd(
                scores_2 * scale, model_names=("A", "B"), alpha=0.05, rope=0.02 * scale
            )
            assert scaled.mse > 0.0
            assert scaled.f_statistic == pytest.approx(base.f_statistic, rel=1e-12)
            assert scaled.anova_p_value == pytest.approx(base.anova_p_value, abs=1e-12)
            assert scaled.comparisons[0].p_value == pytest.approx(float(t_rel.pvalue), abs=1e-12)
            assert scaled.rank_order == base.rank_order
            assert scaled.top_tier == base.top_tier

        zero_scores = np.zeros((4, 3), dtype=np.float64)
        zero_tukey = rm_tukey_hsd(zero_scores, model_names=("A", "B", "C"))
        assert zero_tukey.mse == 0.0
        assert zero_tukey.f_statistic == 0.0
        assert zero_tukey.anova_p_value == 1.0
        assert zero_tukey.top_tier == ("A", "B", "C")

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="2-dimensional"):
            rm_tukey_hsd(np.ones(5))
        with pytest.raises(ValueError, match="at least 2"):
            rm_tukey_hsd(np.ones((1, 3)))
        with pytest.raises(ValueError, match="model_names"):
            rm_tukey_hsd(np.ones((4, 3)), model_names=("a", "b"))
        with pytest.raises(ValueError, match="unique"):
            rm_tukey_hsd(np.ones((4, 2)), model_names=("dup", "dup"))
