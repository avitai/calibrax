"""Statistical significance tests for benchmark comparisons.

Provides Welch's t-test, Mann-Whitney U, paired Wilcoxon signed-rank test
(with pure-Python sign test fallback), and Cohen's d effect size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from calibrax.core.models import SignificanceResult


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Welch's t-test for unequal variances.

    Requires scipy. Raises ImportError with clear message if unavailable.

    Args:
        a: First sample measurements.
        b: Second sample measurements.

    Returns:
        Tuple of (t_statistic, p_value).

    Raises:
        ImportError: If scipy is not installed.
    """
    try:
        from scipy.stats import ttest_ind
    except ImportError as exc:
        raise ImportError(
            "scipy is required for welch_t_test. Install with: uv pip install scipy"
        ) from exc

    result = ttest_ind(a, b, equal_var=False)
    stat = float(result.statistic)  # pyright: ignore[reportAttributeAccessIssue]
    pval = float(result.pvalue)  # pyright: ignore[reportAttributeAccessIssue]
    return (stat, pval)


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Mann-Whitney U test for non-parametric distribution comparison.

    Requires scipy. Raises ImportError with clear message if unavailable.

    Args:
        a: First sample measurements.
        b: Second sample measurements.

    Returns:
        Tuple of (u_statistic, p_value).

    Raises:
        ImportError: If scipy is not installed.
    """
    try:
        from scipy.stats import mannwhitneyu
    except ImportError as exc:
        raise ImportError(
            "scipy is required for mann_whitney_u. Install with: uv pip install scipy"
        ) from exc

    result = mannwhitneyu(a, b, alternative="two-sided")
    stat = float(result.statistic)  # pyright: ignore[reportAttributeAccessIssue]
    pval = float(result.pvalue)  # pyright: ignore[reportAttributeAccessIssue]
    return (stat, pval)


def paired_significance_test(
    a: list[float],
    b: list[float],
    *,
    alpha: float = 0.05,
) -> SignificanceResult:
    """Wilcoxon signed-rank test for paired samples.

    Tests whether two related samples have the same distribution.
    Uses scipy.stats.wilcoxon when available, falls back to a pure-Python
    sign test approximation for small samples.

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

    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        stat = float(result.statistic)  # pyright: ignore[reportAttributeAccessIssue]
        pval = float(result.pvalue)  # pyright: ignore[reportAttributeAccessIssue]
        return SignificanceResult(
            p_value=pval,
            statistic=stat,
            effect_size=es,
            significant=pval < alpha,
            method="wilcoxon",
        )
    except ImportError:
        return _sign_test_fallback(a, b, es, alpha)


def _sign_test_fallback(
    a: list[float], b: list[float], es: float, alpha: float
) -> SignificanceResult:
    """Pure-Python sign test fallback when scipy is unavailable."""
    diffs = [ai - bi for ai, bi in zip(a, b)]
    non_zero = [d for d in diffs if d != 0.0]
    if not non_zero:
        return SignificanceResult(
            p_value=1.0,
            statistic=0.0,
            effect_size=es,
            significant=False,
            method="wilcoxon",
        )
    positives = sum(1 for d in non_zero if d > 0)
    nn = len(non_zero)
    k = min(positives, nn - positives)
    p_value = 0.0
    for i in range(k + 1):
        p_value += _binom_coeff(nn, i) * (0.5**nn)
    p_value = min(p_value * 2.0, 1.0)
    return SignificanceResult(
        p_value=p_value,
        statistic=float(positives),
        effect_size=es,
        significant=p_value < alpha,
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


def _binom_coeff(n: int, k: int) -> int:
    """Binomial coefficient C(n, k). Pure Python.

    Args:
        n: Total items.
        k: Items to choose.

    Returns:
        Number of ways to choose k items from n.
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result
