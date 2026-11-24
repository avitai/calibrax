"""Two-way repeated-measures ANOVA and simultaneous Tukey HSD model comparisons.

Implements the repeated-measures ANOVA + Studentized-range Tukey HSD multi-model
benchmark comparison of Ash et al. (*J. Chem. Inf. Model.* 2025,
DOI: 10.1021/acs.jcim.5c01609), separating frequentist rejection
(``significant = p_value < alpha``, matching ``statsmodels`` ``multicomp.py:1310``)
from simultaneous-interval equivalence testing
(``is_practically_equivalent = ci_lower >= -rope and ci_upper <= rope``, the
simultaneous-interval form of Schuirmann 1987 TOST).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from jax.typing import ArrayLike
from numpy.typing import ArrayLike as HostArrayLike
from scipy.stats import f as f_dist, studentized_range
from substrax.records import read_record
from substrax.typing import JsonValue


_MIN_SAMPLE_SIZE = 2


def _higham_gamma(k: int, dtype: np.dtype[np.floating]) -> float:
    """Higham (2nd ed., Eq. 3.4) rounding constant ``k * u / (1 - k * u)`` for ``u = eps / 2``."""
    u = 0.5 * float(np.finfo(dtype).eps)
    ku = min(float(k) * u, 0.5)
    return ku / (1.0 - ku)


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
        significant: Whether ``p_value < alpha`` (frequentist null rejection, matching
            ``statsmodels.sandbox.stats.multicomp``).
        is_practically_equivalent: Whether the simultaneous ``(1 - alpha)`` confidence
            interval ``[ci_lower, ci_upper]`` lies inside ``[-rope, +rope]`` (equivalence
            testing in interval TOST form, Schuirmann 1987; ``statsmodels``
            ``ttost_paired`` and simultaneous Tukey ``confint``; conservative relative
            to unadjusted per-pair TOST).
    """

    model_a: str
    model_b: str
    mean_diff: float
    q_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    significant: bool
    is_practically_equivalent: bool

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
            "is_practically_equivalent": self.is_practically_equivalent,
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
        top_tier: Models in ``rank_order`` either not significantly worse than
            ``rank_order[0]`` (``p_adj >= alpha``) or practically equivalent to
            ``rank_order[0]`` (``is_practically_equivalent``, simultaneous CI inside
            ``[-rope, +rope]``).
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


def _resolve_model_names(model_names: Sequence[str] | None, n_models: int) -> tuple[str, ...]:
    """Resolve and validate ``n_models`` unique model identifiers."""
    names = (
        tuple(f"model_{k}" for k in range(n_models)) if model_names is None else tuple(model_names)
    )
    if len(names) != n_models:
        raise ValueError(f"model_names length {len(names)} must match n_models={n_models}")
    if len(set(names)) != n_models:
        raise ValueError("model_names must be unique")
    return names


def _validate_tukey_inputs(
    scores: HostArrayLike | ArrayLike,
    *,
    model_names: Sequence[str] | None,
    alpha: float,
    rope: float,
) -> tuple[np.ndarray, tuple[str, ...], np.dtype[np.floating]]:
    """Validate repeated-measures score matrix, model names, and input precision."""
    raw = np.asarray(scores)
    work_dtype = (
        np.dtype(raw.dtype) if np.issubdtype(raw.dtype, np.floating) else np.dtype(np.float64)
    )
    arr = raw.astype(np.float64)
    if arr.ndim != 2:  # noqa: PLR2004
        raise ValueError(
            f"scores must be 2-dimensional (n_blocks, n_models), got shape {arr.shape}"
        )
    n_blocks, n_models = arr.shape
    if min(n_blocks, n_models) < _MIN_SAMPLE_SIZE:
        raise ValueError(
            f"scores requires at least 2 blocks and 2 models, got ({n_blocks}, {n_models})"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("scores must contain finite values")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not math.isfinite(rope) or rope < 0.0:
        raise ValueError(f"rope must be finite and non-negative, got {rope}")
    return arr, _resolve_model_names(model_names, n_models), work_dtype


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
) -> tuple[tuple[TukeyPairwiseComparison, ...], np.ndarray, np.ndarray]:
    """Build simultaneous Tukey HSD pairwise table, ``p_adj`` matrix, and equivalence matrix."""
    n_models = len(names)
    idx_i, idx_j = np.triu_indices(n_models, k=1)
    diffs = np.where(
        np.abs(means_vec[idx_i] - means_vec[idx_j]) <= tol, 0.0, means_vec[idx_i] - means_vec[idx_j]
    )
    abs_diffs = np.abs(diffs)

    if se_mean == 0.0:
        q_stats = np.where(abs_diffs == 0.0, 0.0, np.inf)
        p_adjs = np.where(abs_diffs == 0.0, 1.0, 0.0)
        ci_half = 0.0
    else:
        q_stats = abs_diffs / se_mean
        p_adjs = np.asarray(studentized_range.sf(q_stats, n_models, df_resid), dtype=np.float64)
        ci_half = q_crit * se_mean

    ci_lows = diffs - ci_half
    ci_highs = diffs + ci_half
    equiv_flags = (ci_lows >= -rope) & (ci_highs <= rope)

    p_matrix = np.ones((n_models, n_models), dtype=np.float64)
    p_matrix[idx_i, idx_j] = p_adjs
    p_matrix[idx_j, idx_i] = p_adjs

    equiv_matrix = np.eye(n_models, dtype=bool)
    equiv_matrix[idx_i, idx_j] = equiv_flags
    equiv_matrix[idx_j, idx_i] = equiv_flags

    comparisons = tuple(
        TukeyPairwiseComparison(
            model_a=names[int(i)],
            model_b=names[int(j)],
            mean_diff=float(d),
            q_statistic=float(q_val),
            p_value=float(p_val),
            ci_lower=float(lo),
            ci_upper=float(hi),
            significant=bool(p_val < alpha),
            is_practically_equivalent=bool(eq),
        )
        for i, j, d, q_val, p_val, lo, hi, eq in zip(
            idx_i, idx_j, diffs, q_stats, p_adjs, ci_lows, ci_highs, equiv_flags, strict=True
        )
    )
    return comparisons, p_matrix, equiv_matrix


def _omnibus_f_test(
    means_vec: np.ndarray,
    grand_mean: float,
    *,
    mse: float,
    n_blocks: int,
    df_resid: int,
    diff_tol: float,
) -> tuple[float, float]:
    """Compute omnibus repeated-measures ``(f_statistic, anova_p_value)``."""
    n_models = int(means_vec.size)
    ss_model = float(n_blocks * np.sum((means_vec - grand_mean) ** 2))
    if mse == 0.0:
        no_effect = ss_model <= float(n_blocks * n_models) * (diff_tol * diff_tol)
        return (0.0, 1.0) if no_effect else (math.inf, 0.0)
    f_stat = (ss_model / float(n_models - 1)) / mse
    return f_stat, float(f_dist.sf(f_stat, n_models - 1, df_resid))


def _select_top_tier(
    names: tuple[str, ...],
    order: np.ndarray,
    p_matrix: np.ndarray,
    equiv_matrix: np.ndarray,
    *,
    alpha: float,
) -> tuple[str, ...]:
    """Return ranked models retained in ``top_tier`` relative to ``order[0]``."""
    best_idx = int(order[0])
    keep: list[str] = []
    for raw_idx in order:
        idx = int(raw_idx)
        if idx == best_idx or p_matrix[best_idx, idx] >= alpha or bool(equiv_matrix[best_idx, idx]):
            keep.append(names[idx])
    return tuple(keep)


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
    range test (Ash et al., *J. Chem. Inf. Model.* 2025, DOI: 10.1021/acs.jcim.5c01609).

    ``TukeyPairwiseComparison.significant`` reports frequentist rejection ``p_value < alpha``
    (matching ``statsmodels`` ``sandbox/stats/multicomp.py:1310``), while
    ``TukeyPairwiseComparison.is_practically_equivalent`` tests whether the comparison's
    simultaneous ``(1 - alpha)`` confidence interval ``[ci_lower, ci_upper]`` lies inside
    ``[-rope, +rope]`` (the interval form of Schuirmann 1987 TOST, conservative relative to
    unadjusted per-pair TOST). Zero-residual cutoffs use the working dtype's unit roundoff
    via Higham (2nd ed., Eq. 3.4).

    Args:
        scores: Score matrix of shape ``(n_blocks, n_models)``.
        model_names: Optional sequence of ``n_models`` unique model names.
        alpha: Family-wise error rate in ``(0, 1)`` (defaults to ``0.05``).
        higher_is_better: If ``True``, larger scores rank better; if ``False``, smaller
            scores rank better.
        rope: Practical equivalence half-width ``>= 0.0``; models whose simultaneous
            ``(1 - alpha)`` interval with the top-ranked model lies inside
            ``[-rope, +rope]`` are marked ``is_practically_equivalent`` and retained in
            ``top_tier``.

    Returns:
        :class:`RepeatedMeasuresTukeyResult` with omnibus ANOVA statistics, simultaneous
        pairwise comparisons, ranked model names, and ``top_tier``.

    Raises:
        ValueError: If ``scores`` is not a finite 2-D matrix with ``n_blocks >= 2`` and
            ``n_models >= 2``, ``model_names`` are invalid, or ``alpha``/``rope`` are
            out of range.
    """
    arr, names, work_dtype = _validate_tukey_inputs(
        scores, model_names=model_names, alpha=alpha, rope=rope
    )
    n_blocks, n_models = arr.shape
    block_means = np.mean(arr, axis=1, keepdims=True)
    means_vec = np.mean(arr, axis=0)
    grand_mean = float(np.mean(means_vec))

    residuals = arr - block_means - means_vec[None, :] + grand_mean
    df_resid = (n_blocks - 1) * (n_models - 1)
    scale = float(np.max(np.abs(arr)))
    res_tol = _higham_gamma(2 * (n_blocks + n_models) + 3, work_dtype) * scale
    diff_tol = _higham_gamma(2 * n_blocks + 2, work_dtype) * scale
    ss_res = float(np.sum(residuals * residuals))
    mse = (
        0.0
        if scale == 0.0 or ss_res <= float(n_blocks * n_models) * (res_tol * res_tol)
        else ss_res / float(df_resid)
    )
    f_stat, anova_p = _omnibus_f_test(
        means_vec, grand_mean, mse=mse, n_blocks=n_blocks, df_resid=df_resid, diff_tol=diff_tol
    )

    q_crit = float(studentized_range.ppf(1.0 - alpha, n_models, df_resid))
    se_mean = math.sqrt(mse / float(n_blocks))
    comparisons, p_matrix, equiv_matrix = _build_tukey_comparisons(
        means_vec,
        names,
        se_mean=se_mean,
        q_crit=q_crit,
        df_resid=df_resid,
        alpha=alpha,
        rope=rope,
        tol=diff_tol if mse == 0.0 else 0.0,
    )

    sort_key = -means_vec if higher_is_better else means_vec
    order = np.argsort(sort_key, kind="stable")
    rank_order = tuple(names[int(idx)] for idx in order)
    top_tier = _select_top_tier(names, order, p_matrix, equiv_matrix, alpha=alpha)
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
