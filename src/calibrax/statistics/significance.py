"""Statistical significance, multiple testing, and power analysis for benchmark comparisons.

Provides classical two-sample and paired tests (Welch's t-test, Mann-Whitney U, paired
Wilcoxon signed-rank, Cohen's d, and exact non-central Student-t power and minimum
detectable effect) alongside JAX-traceable multiple-hypothesis control
(:func:`benjamini_hochberg_adjust` and Foster-Stine sequential :func:`alpha_investing`),
and re-exports :mod:`calibrax.statistics.nadeau_bengio` and
:mod:`calibrax.statistics.tukey`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike
from scipy.optimize import brentq
from scipy.stats import (
    mannwhitneyu,
    nct,
    norm,
    t as student_t,
    ttest_ind,
    wilcoxon,
)

from calibrax.core._validation import validate_scalar_numeric
from calibrax.core.models import SignificanceResult
from calibrax.statistics.nadeau_bengio import nadeau_bengio_t_test, NadeauBengioResult
from calibrax.statistics.tukey import (
    RepeatedMeasuresTukeyResult,
    rm_tukey_hsd,
    TukeyPairwiseComparison,
)


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


def alpha_investing(  # noqa: DOC503
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
        TypeError: If ``wealth``, ``pay_out``, or ``invest_fraction`` is boolean or non-numeric.
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


__all__ = [
    "AlphaInvestingResult",
    "NadeauBengioResult",
    "RepeatedMeasuresTukeyResult",
    "TukeyPairwiseComparison",
    "alpha_investing",
    "benjamini_hochberg_adjust",
    "effect_size",
    "mann_whitney_u",
    "minimum_detectable_effect",
    "nadeau_bengio_t_test",
    "paired_significance_test",
    "rm_tukey_hsd",
    "statistical_power",
    "welch_t_test",
]
