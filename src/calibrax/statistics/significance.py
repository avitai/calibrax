"""Statistical significance, multiple testing, and power analysis for benchmark comparisons.

Provides host-side classical and correlated-resample significance tests (Welch's t-test,
Mann-Whitney U, paired Wilcoxon signed-rank, Nadeau-Bengio variance-corrected paired t-test
with Student-t ROPE posteriors, two-way repeated-measures ANOVA + Tukey HSD, Cohen's d, and
exact non-central Student-t power and minimum detectable effect) alongside JAX-traceable
multiple-hypothesis control (:func:`benjamini_hochberg_adjust` and Foster-Stine sequential
:func:`alpha_investing`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.typing import ArrayLike
from numpy.typing import ArrayLike as HostArrayLike
from scipy.optimize import brentq
from scipy.stats import (
    f as f_dist,
    mannwhitneyu,
    nct,
    norm,
    studentized_range,
    t as student_t,
    ttest_ind,
    wilcoxon,
)
from substrax.records import read_record
from substrax.typing import JsonValue

from calibrax.core._validation import validate_scalar_numeric
from calibrax.core.models import SignificanceResult


_MIN_SAMPLE_SIZE = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True, kw_only=True)
class AlphaInvestingResult:
    """Sequential Foster-Stine alpha-investing decisions and wealth trajectory.

    Attributes:
        rejected: Boolean array of shape ``(m,)`` marking rejected null hypotheses.
        thresholds: Per-step test levels ``alpha_t`` of shape ``(m,)``.
        wealth: Post-step alpha-wealth ``W_t`` of shape ``(m,)``.
    """

    rejected: jax.Array
    thresholds: jax.Array
    wealth: jax.Array


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Welch's t-test for unequal variances.

    Args:
        a: First sample measurements.
        b: Second sample measurements.

    Returns:
        Tuple of (t_statistic, p_value).
    """
    statistic, p_value = ttest_ind(a, b, equal_var=False)
    return (float(statistic), float(p_value))


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Mann-Whitney U test for non-parametric distribution comparison.

    Args:
        a: First sample measurements.
        b: Second sample measurements.

    Returns:
        Tuple of (u_statistic, p_value).
    """
    statistic, p_value = mannwhitneyu(a, b, alternative="two-sided")
    return (float(statistic), float(p_value))


def paired_significance_test(
    a: list[float],
    b: list[float],
    *,
    alpha: float = 0.05,
) -> SignificanceResult:
    """Wilcoxon signed-rank test for paired samples.

    Tests whether two related samples have the same distribution, with
    ``scipy.stats.wilcoxon``.

    Args:
        a: First sample (e.g., baseline measurements).
        b: Second sample (e.g., current measurements). Must be same length as a.
        alpha: Significance threshold (default 0.05).

    Returns:
        SignificanceResult with p_value, statistic, effect_size (Cohen's d),
        significant flag, and method name.

    Raises:
        ValueError: If samples are empty or have different lengths.
    """
    if not a or not b:
        raise ValueError("Cannot test significance on empty samples")
    if len(a) != len(b):
        raise ValueError(f"Paired test requires equal lengths: len(a)={len(a)}, len(b)={len(b)}")

    es = effect_size(a, b)

    if all(x == y for x, y in zip(a, b, strict=True)):
        # "wilcox" drops every zero difference, so none is left to rank: the exact null puts all
        # its mass on a statistic of 0 and the p-value is 1, scipy's method="exact" answer, which
        # its method="auto" reaches only through a 0/0 that warns.
        return SignificanceResult(
            p_value=1.0, statistic=0.0, effect_size=es, significant=False, method="wilcoxon"
        )
    statistic, p_value = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return SignificanceResult(
        p_value=float(p_value),
        statistic=float(statistic),
        effect_size=es,
        significant=float(p_value) < alpha,
        method="wilcoxon",
    )


def effect_size(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d effect size for two independent samples.

    Args:
        a: First sample.
        b: Second sample.

    Returns:
        Absolute Cohen's d value. Returns 0.0 if pooled std is zero.
    """
    n_a = len(a)
    n_b = len(b)
    mean_a = sum(a) / max(n_a, 1)
    mean_b = sum(b) / max(n_b, 1)
    var_a = sum((x - mean_a) ** 2 for x in a) / max(n_a - 1, 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / max(n_b - 1, 1)
    pooled_std = math.sqrt((var_a + var_b) / 2) if (var_a + var_b) > 0 else 0.0
    return abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0


def benjamini_hochberg_adjust(p_values: ArrayLike) -> jax.Array:
    """Step-up Benjamini-Hochberg FDR-adjusted p-values (q-values).

    Computes ``q_(k) = min(1, min_{r >= k} (m / r) * p_(r))`` on sorted p-values
    (Benjamini and Hochberg, JRSS-B 1995), matching
    ``scipy.stats.false_discovery_control(..., method="bh")`` while remaining
    compatible with ``jax.jit`` and ``jax.vmap``. Out-of-range entries outside
    ``[0, 1]`` or ``NaN`` propagate ``NaN`` across the returned vector.

    Args:
        p_values: Unadjusted p-values in ``[0, 1]``.

    Returns:
        Adjusted q-values with the same shape as ``p_values``.
    """
    p_raw = jnp.asarray(p_values)
    out_dtype = jnp.result_type(p_raw.dtype, jnp.float32)
    p = p_raw.astype(out_dtype)
    flat = p.ravel()
    m = flat.size
    if m == 0:
        return jnp.zeros(p.shape, dtype=out_dtype)

    one = jnp.asarray(1.0, dtype=out_dtype)
    zero = jnp.asarray(0.0, dtype=out_dtype)
    invalid = jnp.any(jnp.isnan(flat) | (flat < zero) | (flat > one))
    safe_flat = jnp.where(jnp.isnan(flat), one, jnp.clip(flat, zero, one))
    order = jnp.argsort(safe_flat)
    sorted_p = safe_flat[order]
    ranks = jnp.arange(1, m + 1, dtype=out_dtype)
    scaled = sorted_p * (jnp.asarray(float(m), dtype=out_dtype) / ranks)
    monotone = jnp.minimum(lax.cummin(scaled, axis=0, reverse=True), one)
    adjusted = jnp.zeros_like(safe_flat).at[order].set(monotone)
    return jnp.where(invalid, jnp.full_like(adjusted, jnp.nan), adjusted).reshape(p.shape)


def alpha_investing(
    p_values: ArrayLike,
    *,
    wealth: ArrayLike = 0.05,
    pay_out: ArrayLike = 0.05,
    invest_fraction: ArrayLike = 0.5,
) -> AlphaInvestingResult:
    """Sequential online mFDR control via Foster-Stine alpha-investing (JRSS-B 2008).

    Tests a sequential stream of hypotheses ``p_1, ..., p_m`` without knowing future
    p-values in advance. At step ``t``, invests fraction ``gamma = invest_fraction`` of
    current wealth ``W_{t-1}`` by setting ``alpha_t = (gamma * W_{t-1}) / (1 + gamma * W_{t-1})``.
    Rejections earn ``pay_out``, while non-rejections spend ``alpha_t / (1 - alpha_t) =
    gamma * W_{t-1}``, keeping wealth strictly non-negative.

    Args:
        p_values: Sequential p-values in ``[0, 1]``.
        wealth: Initial alpha-wealth ``W_0 > 0`` (defaults to ``0.05``).
        pay_out: Wealth earned upon each rejection ``omega > 0`` (defaults to ``0.05``).
        invest_fraction: Fraction of remaining wealth risked per step in ``(0, 1)``
            (defaults to ``0.5``).

    Returns:
        :class:`AlphaInvestingResult` containing ``rejected``, ``thresholds``, and ``wealth``.

    Raises:
        ValueError: If eager ``wealth <= 0``, ``pay_out <= 0``, or ``invest_fraction``
            lies outside ``(0, 1)``.
    """
    eager_wealth, w_raw = validate_scalar_numeric(wealth, name="wealth")
    if eager_wealth is not None and not (eager_wealth > 0.0):
        raise ValueError(f"wealth must be positive, got {wealth}")
    eager_pay_out, po_raw = validate_scalar_numeric(pay_out, name="pay_out")
    if eager_pay_out is not None and not (eager_pay_out > 0.0):
        raise ValueError(f"pay_out must be positive, got {pay_out}")
    eager_invest, gamma_raw = validate_scalar_numeric(invest_fraction, name="invest_fraction")
    if eager_invest is not None and not (0.0 < eager_invest < 1.0):
        raise ValueError(f"invest_fraction must be in (0, 1), got {invest_fraction}")

    p_raw = jnp.asarray(p_values).ravel()
    out_dtype = jnp.result_type(p_raw.dtype, w_raw, po_raw, gamma_raw, jnp.float32)
    p = p_raw.astype(out_dtype)
    if p.size == 0:
        empty_f = jnp.zeros((0,), dtype=out_dtype)
        return AlphaInvestingResult(
            rejected=jnp.zeros((0,), dtype=bool),
            thresholds=empty_f,
            wealth=empty_f,
        )

    w0 = w_raw.astype(out_dtype)
    reward = po_raw.astype(out_dtype)
    gamma = gamma_raw.astype(out_dtype)
    one = jnp.asarray(1.0, dtype=out_dtype)

    def _step(
        cur_wealth: jax.Array, p_t: jax.Array
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        bet = gamma * cur_wealth
        alpha_t = bet / (one + bet)
        rej = p_t <= alpha_t
        next_wealth = jnp.where(rej, cur_wealth + reward, cur_wealth - bet)
        return next_wealth, (rej, alpha_t, next_wealth)

    _, (rejected, thresholds, wealth_hist) = lax.scan(_step, w0, p)
    return AlphaInvestingResult(rejected=rejected, thresholds=thresholds, wealth=wealth_hist)


def _design_df_and_eff_n(n: int, *, paired: bool, n_b: int | None) -> tuple[int, float]:
    """Degrees of freedom and effective sample size for paired or two-sample t-tests."""
    if n < _MIN_SAMPLE_SIZE:
        raise ValueError(f"n must be at least 2, got {n}")
    if paired:
        return n - 1, float(n)
    nb = n if n_b is None else n_b
    if nb < _MIN_SAMPLE_SIZE:
        raise ValueError(f"n_b must be at least 2, got {nb}")
    return n + nb - 2, (float(n) * float(nb)) / float(n + nb)


def statistical_power(
    effect_size: float,
    n: int,
    *,
    alpha: float = 0.05,
    paired: bool = False,
    n_b: int | None = None,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> float:
    """Exact Student-t statistical power via the non-central t distribution.

    Args:
        effect_size: Standardized effect size (Cohen's ``d``).
        n: Sample size (number of pairs when ``paired=True``, or first-group size).
        alpha: Type-I error rate in ``(0, 1)``.
        paired: If ``True``, use paired/one-sample ``t`` with ``df = n - 1``; otherwise
            two-sample ``t`` with ``df = n + n_b - 2``.
        n_b: Optional second-group size when ``paired=False`` (defaults to ``n``).
        alternative: Alternative hypothesis (``"two-sided"``, ``"greater"``, or ``"less"``).

    Returns:
        Exact statistical power in ``[0, 1]``.

    Raises:
        ValueError: If ``n < 2``, ``n_b < 2``, ``alpha`` is outside ``(0, 1)``, or
            ``alternative`` is unrecognized.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    df, n_eff = _design_df_and_eff_n(n, paired=paired, n_b=n_b)
    root_n = math.sqrt(n_eff)

    if alternative == "two-sided":
        ncp = abs(effect_size) * root_n
        t_crit = float(student_t.ppf(1.0 - alpha / 2.0, df))
        return float(nct.sf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp))
    if alternative == "greater":
        ncp = effect_size * root_n
        t_crit = float(student_t.ppf(1.0 - alpha, df))
        return float(nct.sf(t_crit, df, ncp))
    if alternative == "less":
        ncp = effect_size * root_n
        t_crit = float(student_t.ppf(alpha, df))
        return float(nct.cdf(t_crit, df, ncp))
    raise ValueError(f"Unsupported alternative: {alternative!r}")


def minimum_detectable_effect(
    n: int,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    paired: bool = False,
    n_b: int | None = None,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    std: float = 1.0,
) -> float:
    """Minimum detectable effect size (MDE) inverting :func:`statistical_power`.

    Solves for the non-negative effect magnitude whose exact non-central ``t`` power
    equals ``power``, scaled by ``std`` (returning Cohen's ``d`` when ``std=1.0``).

    Args:
        n: Sample size (number of pairs when ``paired=True``, or first-group size).
        alpha: Type-I error rate in ``(0, 1)``.
        power: Target statistical power in ``(alpha, 1)``.
        paired: Whether the design is paired/one-sample.
        n_b: Optional second-group size when ``paired=False``.
        alternative: Alternative hypothesis (``"two-sided"``, ``"greater"``, or ``"less"``).
        std: Outcome standard deviation scaling standardized MDE into raw units.

    Returns:
        Positive minimum detectable effect ``d * std``.

    Raises:
        ValueError: If sample sizes, ``alpha``, ``power``, ``std``, or ``alternative``
            are out of range.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (alpha < power < 1.0):
        raise ValueError(f"power must be strictly in (alpha, 1), got {power} for alpha={alpha}")
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"Unsupported alternative: {alternative!r}")

    _, n_eff = _design_df_and_eff_n(n, paired=paired, n_b=n_b)
    # By symmetry of Student-t, magnitude for "less" equals magnitude for "greater".
    solve_alt: Literal["two-sided", "greater"] = (
        "two-sided" if alternative == "two-sided" else "greater"
    )
    alpha_tail = alpha / 2.0 if solve_alt == "two-sided" else alpha
    z_guess = float(norm.ppf(1.0 - alpha_tail) + norm.ppf(power)) / math.sqrt(n_eff)
    hi = max(z_guess * 1.5, 1e-3)
    while (
        statistical_power(hi, n, alpha=alpha, paired=paired, n_b=n_b, alternative=solve_alt) < power
    ):
        hi *= 2.0

    d_star = float(
        brentq(
            lambda d: (
                statistical_power(d, n, alpha=alpha, paired=paired, n_b=n_b, alternative=solve_alt)
                - power
            ),
            0.0,
            hi,
            xtol=1e-12,
        )
    )
    return d_star * std


@dataclass(frozen=True, slots=True, kw_only=True)
class NadeauBengioResult:
    """Result of a Nadeau-Bengio variance-corrected paired t-test with ROPE posteriors.

    Attributes:
        statistic: Variance-corrected Student-t statistic ``mean_diff / se``.
        p_value: Frequentist Student-t p-value for ``alternative``.
        mean_diff: Mean paired difference ``mean(scores_a - scores_b)``.
        se: Nadeau-Bengio corrected standard error ``sqrt((1 / J + n_test / n_train) * s_d^2)``.
        df: Degrees of freedom ``J - 1``.
        ci_lower: Lower ``(1 - alpha)`` confidence bound on ``mean_diff``.
        ci_upper: Upper ``(1 - alpha)`` confidence bound on ``mean_diff``.
        significant: Whether ``p_value < alpha``.
        rope_probabilities: Closed-form Student-t posterior probabilities
            ``(p_left, p_rope, p_right)`` for ``mu < -rope``, ``|mu| <= rope``, and
            ``mu > +rope`` (Benavoli et al., JMLR 2017).
    """

    statistic: float
    p_value: float
    mean_diff: float
    se: float
    df: int
    ci_lower: float
    ci_upper: float
    significant: bool
    rope_probabilities: tuple[float, float, float]

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "mean_diff": float(self.mean_diff),
            "se": float(self.se),
            "df": int(self.df),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "significant": self.significant,
            "rope_probabilities": [float(p) for p in self.rope_probabilities],
        }

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> NadeauBengioResult:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds an invalid value.
        """
        return read_record(cls, data)


@dataclass(frozen=True, slots=True, kw_only=True)
class TukeyPairwiseComparison:
    """One pairwise model comparison from two-way repeated-measures Tukey HSD.

    Attributes:
        model_a: First model identifier.
        model_b: Second model identifier.
        mean_diff: Mean difference ``mean(model_a) - mean(model_b)`` across shared splits.
        q_statistic: Studentized range statistic ``|mean_diff| / sqrt(mse / n_blocks)``.
        p_value: Simultaneous family-wise Tukey-adjusted p-value.
        ci_lower: Lower simultaneous ``(1 - alpha)`` confidence bound on ``mean_diff``.
        ci_upper: Upper simultaneous ``(1 - alpha)`` confidence bound on ``mean_diff``.
        significant: Whether ``p_value < alpha`` and ``|mean_diff| > rope``.
    """

    model_a: str
    model_b: str
    mean_diff: float
    q_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    significant: bool

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "mean_diff": float(self.mean_diff),
            "q_statistic": float(self.q_statistic),
            "p_value": float(self.p_value),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "significant": self.significant,
        }

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> TukeyPairwiseComparison:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds an invalid value.
        """
        return read_record(cls, data)


@dataclass(frozen=True, slots=True, kw_only=True)
class RepeatedMeasuresTukeyResult:
    """Two-way repeated-measures ANOVA and simultaneous Tukey HSD comparison across models.

    Attributes:
        model_names: Model identifiers in input column order.
        model_means: Per-model mean score across shared benchmark splits/blocks.
        mse: Two-way additive ANOVA residual mean square.
        df_resid: Residual degrees of freedom ``(n_blocks - 1) * (n_models - 1)``.
        q_critical: Studentized range critical value at level ``1 - alpha``.
        f_statistic: Repeated-measures omnibus F-statistic for model differences.
        anova_p_value: Omnibus p-value from ``F(n_models - 1, df_resid)``.
        comparisons: All ``n_models * (n_models - 1) // 2`` pairwise Tukey comparisons.
        rank_order: Models ordered from best to worst mean score.
        top_tier: Models in ``rank_order`` statistically or practically indistinguishable
            from ``rank_order[0]`` (``p_adj >= alpha`` or ``|mean_best - mean_m| <= rope``).
    """

    model_names: tuple[str, ...]
    model_means: dict[str, float]
    mse: float
    df_resid: int
    q_critical: float
    f_statistic: float
    anova_p_value: float
    comparisons: tuple[TukeyPairwiseComparison, ...]
    rank_order: tuple[str, ...]
    top_tier: tuple[str, ...]

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "model_names": list(self.model_names),
            "model_means": {k: float(v) for k, v in self.model_means.items()},
            "mse": float(self.mse),
            "df_resid": int(self.df_resid),
            "q_critical": float(self.q_critical),
            "f_statistic": float(self.f_statistic),
            "anova_p_value": float(self.anova_p_value),
            "comparisons": [c.to_dict() for c in self.comparisons],
            "rank_order": list(self.rank_order),
            "top_tier": list(self.top_tier),
        }

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> RepeatedMeasuresTukeyResult:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds an invalid value.
        """
        return read_record(cls, data)


def _resolve_split_ratio(
    *,
    test_fraction: float | None,
    n_folds: int | None,
    n_train: int | None,
    n_test: int | None,
) -> float:
    """Resolve ``n_test / n_train`` from exactly one split-ratio specification."""
    if (n_train is None) != (n_test is None):
        raise ValueError("Both n_train and n_test must be provided together")
    specs = (test_fraction is not None) + (n_folds is not None) + (n_train is not None)
    if specs != 1:
        raise ValueError("Specify exactly one of test_fraction, n_folds, or (n_train, n_test)")
    if test_fraction is not None:
        if not (0.0 < test_fraction < 1.0):
            raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
        return float(test_fraction) / float(1.0 - test_fraction)
    if n_folds is not None:
        if n_folds < _MIN_SAMPLE_SIZE:
            raise ValueError(f"n_folds must be at least 2, got {n_folds}")
        return 1.0 / float(n_folds - 1)
    if n_train is None or n_test is None or n_train < 1 or n_test < 1:
        raise ValueError(f"n_train and n_test must be positive, got ({n_train}, {n_test})")
    return float(n_test) / float(n_train)


def _student_t_p_and_ci(
    mean_diff: float,
    se: float,
    df: int,
    *,
    alpha: float,
    alternative: Literal["two-sided", "greater", "less"],
) -> tuple[float, float, float, float]:
    """Compute ``(statistic, p_value, ci_lower, ci_upper)`` with exact ``se == 0`` handling."""
    if se == 0.0:
        t_stat = 0.0 if mean_diff == 0.0 else math.copysign(math.inf, mean_diff)
        if alternative == "two-sided":
            return t_stat, (1.0 if mean_diff == 0.0 else 0.0), mean_diff, mean_diff
        if alternative == "greater":
            return t_stat, (0.0 if mean_diff > 0.0 else 1.0), mean_diff, math.inf
        return t_stat, (0.0 if mean_diff < 0.0 else 1.0), -math.inf, mean_diff

    t_stat = mean_diff / se
    if alternative == "two-sided":
        p_val = float(2.0 * student_t.sf(abs(t_stat), df))
        half = float(student_t.ppf(1.0 - alpha / 2.0, df)) * se
        return t_stat, p_val, mean_diff - half, mean_diff + half
    t_crit = float(student_t.ppf(1.0 - alpha, df)) * se
    if alternative == "greater":
        return t_stat, float(student_t.sf(t_stat, df)), mean_diff - t_crit, math.inf
    return t_stat, float(student_t.cdf(t_stat, df)), -math.inf, mean_diff + t_crit


def _student_t_rope_probabilities(
    mean_diff: float, se: float, df: int, rope: float
) -> tuple[float, float, float]:
    """Compute ``(p_left, p_rope, p_right)`` without ``1 - ~1`` tail cancellation."""
    if se == 0.0:
        if rope == 0.0:
            p_right = float(mean_diff > 0.0) + 0.5 * float(mean_diff == 0.0)
            return 1.0 - p_right, 0.0, p_right
        p_left = float(mean_diff < -rope)
        p_right = float(mean_diff > rope)
        return p_left, 1.0 - p_left - p_right, p_right

    if rope == 0.0:
        z0 = -mean_diff / se
        if mean_diff >= 0.0:
            p_left = float(student_t.cdf(z0, df))
            return p_left, 0.0, 1.0 - p_left
        p_right = float(student_t.sf(z0, df))
        return 1.0 - p_right, 0.0, p_right

    z_lo = (-rope - mean_diff) / se
    z_hi = (rope - mean_diff) / se
    p_left = float(student_t.cdf(z_lo, df))
    p_right = float(student_t.sf(z_hi, df))
    if mean_diff >= rope:
        p_rope = max(0.0, float(student_t.cdf(z_hi, df) - student_t.cdf(z_lo, df)))
    elif mean_diff <= -rope:
        p_rope = max(0.0, float(student_t.sf(z_lo, df) - student_t.sf(z_hi, df)))
    else:
        p_rope = max(0.0, 1.0 - p_left - p_right)
    return p_left, p_rope, p_right


def _validate_nadeau_bengio_inputs(
    scores_a: HostArrayLike | ArrayLike,
    scores_b: HostArrayLike | ArrayLike,
    *,
    alpha: float,
    alternative: str,
    rope: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate paired split score arrays and test hyperparameters."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError(
            f"scores_a and scores_b must be 1-D arrays of identical shape, got {a.shape}, {b.shape}"
        )
    if a.size < _MIN_SAMPLE_SIZE:
        raise ValueError(f"nadeau_bengio_t_test requires at least 2 paired scores, got {a.size}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("scores_a and scores_b must contain finite values")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"Unsupported alternative: {alternative!r}")
    if not math.isfinite(rope) or rope < 0.0:
        raise ValueError(f"rope must be finite and non-negative, got {rope}")
    return a, b


def nadeau_bengio_t_test(  # noqa: PLR0913, DOC502
    scores_a: HostArrayLike | ArrayLike,
    scores_b: HostArrayLike | ArrayLike,
    *,
    test_fraction: float | None = None,
    n_folds: int | None = None,
    n_train: int | None = None,
    n_test: int | None = None,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    rope: float = 0.0,
) -> NadeauBengioResult:
    """Nadeau-Bengio variance-corrected paired t-test and Bayesian Student-t ROPE test.

    Corrects the underestimated standard error of repeated cross-validation or random
    subsampling paired differences via ``SE = sqrt((1 / J + n_test / n_train) * s_d^2)``
    with ``df = J - 1`` (Nadeau and Bengio, Mach. Learn. 2003, Eq. 14) and computes
    closed-form Student-t posterior ROPE probabilities ``(p_left, p_rope, p_right)``
    (Benavoli et al., JMLR 2017).

    Specify the test-to-train overlap correction via **exactly one** of ``test_fraction``
    (``rho = f / (1 - f)``), ``n_folds`` (``rho = 1 / (n_folds - 1)`` for repeated K-fold
    CV), or ``(n_train, n_test)`` (``rho = n_test / n_train``).

    Args:
        scores_a: Per-split evaluation scores for model A (length ``J >= 2``).
        scores_b: Per-split evaluation scores for model B on the same splits.
        test_fraction: Fraction of total data in each test split in ``(0, 1)``.
        n_folds: Number of folds ``K >= 2`` in repeated K-fold cross-validation.
        n_train: Training split sample count (must be paired with ``n_test``).
        n_test: Test split sample count (must be paired with ``n_train``).
        alpha: Significance threshold in ``(0, 1)`` (defaults to ``0.05``).
        alternative: Alternative hypothesis (``"two-sided"``, ``"greater"``, or ``"less"``).
        rope: Half-width of the region of practical equivalence ``[-rope, +rope] >= 0``.

    Returns:
        :class:`NadeauBengioResult` containing the corrected t-statistic, p-value,
        confidence interval, and ROPE posterior probabilities.

    Raises:
        ValueError: If inputs are not finite 1-D arrays of equal length ``>= 2``,
            split-ratio arguments are missing or ambiguous, or ``alpha``/``rope``/
            ``alternative`` are out of range.
    """
    a, b = _validate_nadeau_bengio_inputs(
        scores_a, scores_b, alpha=alpha, alternative=alternative, rope=rope
    )
    rho = _resolve_split_ratio(
        test_fraction=test_fraction, n_folds=n_folds, n_train=n_train, n_test=n_test
    )
    diffs = a - b
    n_splits = int(diffs.size)
    df = n_splits - 1
    mean_diff = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1))
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    tol = 8.0 * float(np.finfo(np.float64).eps) * scale
    if scale == 0.0 or var_d <= tol * tol:
        var_d = 0.0
        if abs(mean_diff) <= tol:
            mean_diff = 0.0
    se = math.sqrt((1.0 / float(n_splits) + rho) * var_d)

    t_stat, p_val, ci_lo, ci_hi = _student_t_p_and_ci(
        mean_diff, se, df, alpha=alpha, alternative=alternative
    )
    rope_probs = _student_t_rope_probabilities(mean_diff, se, df, rope)
    return NadeauBengioResult(
        statistic=t_stat,
        p_value=p_val,
        mean_diff=mean_diff,
        se=se,
        df=df,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        significant=p_val < alpha,
        rope_probabilities=rope_probs,
    )


def _validate_tukey_inputs(
    scores: HostArrayLike | ArrayLike,
    *,
    model_names: Sequence[str] | None,
    alpha: float,
    rope: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Validate repeated-measures score matrix and model names."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 2:  # noqa: PLR2004
        raise ValueError(
            f"scores must be 2-dimensional (n_blocks, n_models), got shape {arr.shape}"
        )
    n_blocks, n_models = arr.shape
    if n_blocks < _MIN_SAMPLE_SIZE or n_models < _MIN_SAMPLE_SIZE:
        raise ValueError(
            f"scores requires at least 2 blocks and 2 models, got ({n_blocks}, {n_models})"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("scores must contain finite values")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not math.isfinite(rope) or rope < 0.0:
        raise ValueError(f"rope must be finite and non-negative, got {rope}")

    names = (
        tuple(f"model_{k}" for k in range(n_models)) if model_names is None else tuple(model_names)
    )
    if len(names) != n_models:
        raise ValueError(f"model_names length {len(names)} must match n_models={n_models}")
    if len(set(names)) != n_models:
        raise ValueError("model_names must be unique")
    return arr, names


def _build_tukey_comparisons(
    means_vec: np.ndarray,
    names: tuple[str, ...],
    *,
    se_mean: float,
    q_crit: float,
    df_resid: int,
    alpha: float,
    rope: float,
    tol: float,
) -> tuple[tuple[TukeyPairwiseComparison, ...], np.ndarray]:
    """Vectorized simultaneous Tukey HSD pairwise table and symmetric ``p_adj`` matrix."""
    n_models = len(names)
    idx_i, idx_j = np.triu_indices(n_models, k=1)
    diffs = means_vec[idx_i] - means_vec[idx_j]
    abs_diffs = np.abs(diffs)

    if se_mean == 0.0:
        q_stats = np.where(abs_diffs <= tol, 0.0, np.inf)
        p_adjs = np.where(abs_diffs <= tol, 1.0, 0.0)
        ci_half = 0.0
    else:
        q_stats = abs_diffs / se_mean
        p_adjs = np.asarray(studentized_range.sf(q_stats, n_models, df_resid), dtype=np.float64)
        ci_half = q_crit * se_mean

    p_matrix = np.ones((n_models, n_models), dtype=np.float64)
    p_matrix[idx_i, idx_j] = p_adjs
    p_matrix[idx_j, idx_i] = p_adjs

    comparisons = tuple(
        TukeyPairwiseComparison(
            model_a=names[int(i)],
            model_b=names[int(j)],
            mean_diff=float(d),
            q_statistic=float(q_val),
            p_value=float(p_val),
            ci_lower=float(d - ci_half),
            ci_upper=float(d + ci_half),
            significant=bool((p_val < alpha) and (abs(float(d)) > rope)),
        )
        for i, j, d, q_val, p_val in zip(idx_i, idx_j, diffs, q_stats, p_adjs, strict=True)
    )
    return comparisons, p_matrix


def rm_tukey_hsd(  # noqa: DOC502
    scores: HostArrayLike | ArrayLike,
    *,
    model_names: Sequence[str] | None = None,
    alpha: float = 0.05,
    higher_is_better: bool = True,
    rope: float = 0.0,
) -> RepeatedMeasuresTukeyResult:
    """Two-way repeated-measures ANOVA and simultaneous all-pairs Tukey HSD comparison.

    Evaluates ``M >= 2`` models across ``B >= 2`` shared benchmark splits or tasks
    (rows = splits/blocks, columns = models), removing shared split-to-split difficulty
    variance via two-way additive ANOVA residuals before applying Tukey's studentized
    range test (Ash et al., J. Chem. Inf. Model. 2025, DOI: 10.1021/acs.jcim.5c01609).

    Args:
        scores: Score matrix of shape ``(n_blocks, n_models)``.
        model_names: Optional sequence of ``n_models`` unique model names.
        alpha: Family-wise error rate in ``(0, 1)`` (defaults to ``0.05``).
        higher_is_better: If ``True``, larger scores rank better; if ``False``, smaller
            scores rank better.
        rope: Practical equivalence margin ``>= 0.0``; models within ``rope`` of the
            top-ranked model's mean are retained in ``top_tier``.

    Returns:
        :class:`RepeatedMeasuresTukeyResult` with omnibus ANOVA statistics, simultaneous
        pairwise comparisons, ranked model names, and ``top_tier``.

    Raises:
        ValueError: If ``scores`` is not a finite 2-D matrix with ``n_blocks >= 2`` and
            ``n_models >= 2``, ``model_names`` are invalid, or ``alpha``/``rope`` are
            out of range.
    """
    arr, names = _validate_tukey_inputs(scores, model_names=model_names, alpha=alpha, rope=rope)
    n_blocks, n_models = arr.shape
    block_means = np.mean(arr, axis=1, keepdims=True)
    means_vec = np.mean(arr, axis=0)
    grand_mean = float(np.mean(arr))

    residuals = arr - block_means - means_vec[None, :] + grand_mean
    df_resid = (n_blocks - 1) * (n_models - 1)
    scale = float(np.max(np.abs(arr)))
    tol = 8.0 * float(np.finfo(np.float64).eps) * scale
    ss_res = float(np.sum(residuals * residuals))
    mse = (
        0.0
        if scale == 0.0 or ss_res <= float(n_blocks * n_models) * (tol * tol)
        else ss_res / float(df_resid)
    )

    ss_model = float(n_blocks * np.sum((means_vec - grand_mean) ** 2))
    ms_model = ss_model / float(n_models - 1)
    if mse == 0.0:
        no_effect = ss_model <= float(n_blocks * n_models) * (tol * tol)
        f_stat = 0.0 if no_effect else math.inf
        anova_p = 1.0 if no_effect else 0.0
    else:
        f_stat = ms_model / mse
        anova_p = float(f_dist.sf(f_stat, n_models - 1, df_resid))

    q_crit = float(studentized_range.ppf(1.0 - alpha, n_models, df_resid))
    se_mean = math.sqrt(mse / float(n_blocks))
    comparisons, p_matrix = _build_tukey_comparisons(
        means_vec,
        names,
        se_mean=se_mean,
        q_crit=q_crit,
        df_resid=df_resid,
        alpha=alpha,
        rope=rope,
        tol=tol,
    )

    sort_key = -means_vec if higher_is_better else means_vec
    order = np.argsort(sort_key, kind="stable")
    rank_order = tuple(names[int(idx)] for idx in order)
    best_idx = int(order[0])
    best_mean = float(means_vec[best_idx])
    top_tier = tuple(
        names[int(idx)]
        for idx in order
        if int(idx) == best_idx
        or p_matrix[best_idx, int(idx)] >= alpha
        or abs(best_mean - float(means_vec[int(idx)])) <= rope
    )
    return RepeatedMeasuresTukeyResult(
        model_names=names,
        model_means={name: float(means_vec[i]) for i, name in enumerate(names)},
        mse=mse,
        df_resid=df_resid,
        q_critical=q_crit,
        f_statistic=f_stat,
        anova_p_value=anova_p,
        comparisons=comparisons,
        rank_order=rank_order,
        top_tier=top_tier,
    )
