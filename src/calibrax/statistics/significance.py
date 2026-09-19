"""Statistical significance tests for benchmark comparisons.

Provides Welch's t-test, Mann-Whitney U, the paired Wilcoxon signed-rank test, and Cohen's d
effect size. The tests are SciPy's, which calibrax has through JAX.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon

from calibrax.core.models import SignificanceResult


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
