"""Nadeau-Bengio variance-corrected paired t-test and Bayesian Student-t ROPE posteriors.

Implements the variance-corrected resampled paired t-test of Nadeau and Bengio
(*Mach. Learn.* 2003, Eq. 14) together with cancellation-free closed-form Student-t
ROPE posterior probabilities (Benavoli et al., *JMLR* 2017).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from jax.typing import ArrayLike
from numpy.typing import ArrayLike as HostArrayLike
from scipy.stats import t as student_t
from substrax.records import read_record
from substrax.typing import JsonValue


_MIN_SAMPLE_SIZE = 2


def _higham_gamma(k: int, dtype: np.dtype[np.floating]) -> float:
    """Higham (2nd ed., Eq. 3.4) rounding constant ``k * u / (1 - k * u)`` for ``u = eps / 2``."""
    u = 0.5 * float(np.finfo(dtype).eps)
    ku = min(float(k) * u, 0.5)
    return ku / (1.0 - ku)


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
            ``mu > +rope`` (Benavoli et al., *JMLR* 2017).
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


def _ratio_from_counts(n_train: int | None, n_test: int | None) -> float:
    """Validate paired split counts and return ``n_test / n_train``."""
    if n_train is None or n_test is None or min(n_train, n_test) < 1:
        raise ValueError(f"n_train and n_test must be positive, got ({n_train}, {n_test})")
    return float(n_test) / float(n_train)


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
    return _ratio_from_counts(n_train, n_test)


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


def _check_alpha_and_rope(alpha: float, rope: float, *, alternative: str) -> None:
    """Validate ``alpha``, ``rope``, and ``alternative`` arguments."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"Unsupported alternative: {alternative!r}")
    if not math.isfinite(rope) or rope < 0.0:
        raise ValueError(f"rope must be finite and non-negative, got {rope}")


def _validate_nadeau_bengio_inputs(
    scores_a: HostArrayLike | ArrayLike,
    scores_b: HostArrayLike | ArrayLike,
    *,
    alpha: float,
    alternative: str,
    rope: float,
) -> tuple[np.ndarray, np.ndarray, np.dtype[np.floating]]:
    """Validate paired split score arrays, test hyperparameters, and input precision."""
    raw_a = np.asarray(scores_a)
    raw_b = np.asarray(scores_b)
    float_dtypes = [x.dtype for x in (raw_a, raw_b) if np.issubdtype(x.dtype, np.floating)]
    work_dtype = np.dtype(np.result_type(*float_dtypes)) if float_dtypes else np.dtype(np.float64)
    a = raw_a.astype(np.float64)
    b = raw_b.astype(np.float64)
    if (a.ndim, b.ndim) != (1, 1) or a.shape != b.shape:
        raise ValueError(
            f"scores_a and scores_b must be 1-D arrays of identical shape, got {a.shape}, {b.shape}"
        )
    if a.size < _MIN_SAMPLE_SIZE:
        raise ValueError(f"nadeau_bengio_t_test requires at least 2 paired scores, got {a.size}")
    if not np.all(np.isfinite(a) & np.isfinite(b)):
        raise ValueError("scores_a and scores_b must contain finite values")
    _check_alpha_and_rope(alpha, rope, alternative=alternative)
    return a, b, work_dtype


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
    with ``df = J - 1`` (Nadeau and Bengio, *Mach. Learn.* 2003, Eq. 14) and computes
    closed-form Student-t posterior ROPE probabilities ``(p_left, p_rope, p_right)``
    (Benavoli et al., *JMLR* 2017). Zero-variance cutoffs are bounded in the input
    working floating-point precision via Higham (*Accuracy and Stability of Numerical
    Algorithms*, 2nd ed., Eq. 3.4) ``|fl(sum x_i) - sum x_i| <= gamma_{n-1} sum |x_i|``.

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
        confidence interval, ``significant = (p_value < alpha)``, and ROPE posterior
        probabilities.

    Raises:
        ValueError: If inputs are not finite 1-D arrays of equal length ``>= 2``,
            split-ratio arguments are missing or ambiguous, or ``alpha``/``rope``/
            ``alternative`` are out of range.
    """
    a, b, work_dtype = _validate_nadeau_bengio_inputs(
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
    scale = float(np.max(np.abs(a) + np.abs(b)))
    tol = _higham_gamma(n_splits + 2, work_dtype) * scale
    if scale == 0.0 or var_d <= (float(n_splits) / float(df)) * (tol * tol):
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
