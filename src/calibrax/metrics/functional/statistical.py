"""Statistical correlation and agreement metrics.

Pure functions for measuring statistical relationships between
variables. Covers linear correlation, rank correlation, and
agreement measures.

Includes 7 functions: pearson_correlation, spearman_rank_correlation,
kendall_tau, concordance_correlation, correlation_preservation, autocorrelation,
skewness.

``r_squared_adjusted`` lived here and does not any more: it is a goodness-of-fit measure
with a degrees-of-freedom penalty, not a correlation or an agreement measure, and it shares
its implementation with ``r_squared``. Both now live in
:mod:`calibrax.metrics.functional.regression`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats import rankdata
from jax.typing import ArrayLike

from calibrax.metrics._utils import (
    _EPSILON,
    _FEATURE_MATRIX_NDIM,
    _MIN_CORRELATED_FEATURES,
    _SERIES_NDIM,
    sample_weights,
    weighted_mean,
)


def pearson_correlation(
    a: ArrayLike,
    b: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> jax.Array:
    """Pearson correlation coefficient.

    Linear correlation: ``cov(a,b) / (std(a) * std(b))``.

    With ``weights`` the means, the covariance and both standard deviations are all
    weighted, which is the standard weighted correlation. Checked against
    ``numpy.cov(aweights=...)``.

    No ``reduction`` or ``axis``, unlike the error metrics in this package: a correlation
    is a ratio of reductions over the whole sample, so there is no per-element value for
    ``reduction="none"`` to return.

    Note:
        Direction: HIGHER (1.0 = perfect positive correlation).
        Range: [-1, 1].
        Measures linear association only.

    Args:
        a: First variable.
        b: Second variable.
        mask: Elements to keep (boolean, broadcastable), or ``None`` for all.
        weights: Element weights (broadcastable), or ``None`` for unit weights.

    Returns:
        Pearson r as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> pearson_correlation(jnp.array([1.0, 2.0, 3.0]),
        ...                     jnp.array([1.0, 2.0, 3.0]))
        1.0
    """
    a_arr = jnp.asarray(a).ravel()
    b_arr = jnp.asarray(b).ravel()
    w = sample_weights(mask, weights, a_arr.shape)
    a_centered = a_arr - weighted_mean(a_arr, w)
    b_centered = b_arr - weighted_mean(b_arr, w)
    cov = jnp.sum(w * a_centered * b_centered)
    std_a = jnp.sqrt(jnp.sum(w * a_centered**2))
    std_b = jnp.sqrt(jnp.sum(w * b_centered**2))
    return cov / (std_a * std_b + _EPSILON)


def spearman_rank_correlation(
    a: ArrayLike,
    b: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> jax.Array:
    """Spearman's rank correlation coefficient.

    Pearson correlation computed on ranks, which tied values share as their average
    (``jax.scipy.stats.rankdata``), so the coefficient does not depend on the order the pairs
    are given in.

    Takes ``mask`` and **deliberately not** ``weights``. Masking is unambiguous: it ranks
    the retained subset. Weighting a rank correlation is not. There is no commonly accepted
    weighted Spearman coefficient; Stata does not estimate one and SAS does not document
    its method; R's ``wCorr`` averages over permutations of tied weights while ``expss``
    rounds weights to integers to match SPSS; and the zero expectation under random
    rankings that unweighted Spearman enjoys does not generally survive weighting
    (arXiv:2504.08428). Choosing one of those silently would produce a number no other tool
    reproduces, so a caller passing ``weights`` gets a ``TypeError`` instead.

    Note:
        Direction: HIGHER (1.0 = perfect monotonic relationship).
        Range: [-1, 1].

    Args:
        a: First variable.
        b: Second variable.
        mask: Elements to keep (boolean, broadcastable), or ``None`` for all. Ranks are
            computed within the retained subset.

    Returns:
        Spearman rho as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> spearman_rank_correlation(jnp.array([1.0, 2.0, 3.0]),
        ...                          jnp.array([1.0, 2.0, 3.0]))
        1.0
    """
    a_arr = jnp.asarray(a).ravel()
    b_arr = jnp.asarray(b).ravel()
    if mask is None:
        return pearson_correlation(rankdata(a_arr), rankdata(b_arr))
    keep = jnp.broadcast_to(jnp.asarray(mask), a_arr.shape)
    # Rank within the retained subset, not within everything. Excluded elements go to
    # +inf so they sort last and take the highest ranks; the retained ones then hold ranks
    # 1..k among themselves, and the mask keeps the excluded out of the correlation.
    # Static shapes throughout, so this traces.
    ranked_a = rankdata(jnp.where(keep, a_arr, jnp.inf))
    ranked_b = rankdata(jnp.where(keep, b_arr, jnp.inf))
    return pearson_correlation(ranked_a, ranked_b, mask=keep)


def kendall_tau(
    a: ArrayLike,
    b: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> jax.Array:
    """Kendall rank correlation coefficient (tau-b).

    ``(concordant - discordant) / sqrt((n0 - n1) * (n0 - n2))``, where ``n0`` is the number of
    pairs and ``n1``, ``n2`` are the pairs tied in ``a`` and in ``b``. Without ties this is
    ``(concordant - discordant) / n0``, tau-a.

    **Weights are frequency weights and enter per pair.** Kendall's tau is a statistic over
    pairs, so a per-element weight lifts to the pair weight ``w_i * w_j``. That is why this
    takes ``weights`` where :func:`spearman_rank_correlation` does not: Spearman's ranks give
    a per-element weight nowhere to go, and no commonly accepted weighted Spearman exists,
    whereas here the lift is forced by the statistic's own shape. Measured over 60 fixtures,
    20 of them carrying ties, integer weights reproduce physically replicating each element
    ``w_i`` times to within 2.2e-16 of ``scipy.stats.kendalltau``. That equivalence is what
    makes these frequency weights rather than a convention; R's ``mariposa`` ships the same
    quantity as an exact frequency-weighted tau-b.

    The tie correction needs no separate term, which is not obvious. Replicating element
    ``i`` creates ``w_i * (w_i - 1) / 2`` pairs tied in **both** variables, and those add
    equally to the pair total and to both tie totals, so they cancel out of ``(n0 - n1)``
    and ``(n0 - n2)``. A test pins the cancellation, because adding the within-element pairs
    back also reproduces replication and would make the omission look like a bug.

    A mask excludes an element by giving it zero weight, which removes every pair that
    element belongs to, so masking equals computing the statistic on the subset.

    The unweighted branch is kept for **compiled size, not for compatibility**: the two
    branches agree on value at every size measured, but routing the default through a
    pair-weight matrix of ones costs 38 fusions against 18, roughly double, on the most
    expensive function in this module. Two earlier justifications for the branch were
    wrong and are recorded so they are not re-derived: it is not that float32 loses
    exactness summing a matrix of ones (measured exact at n=8192, past the 2**24 integer
    ceiling), and it is not that the values would otherwise drift.

    Note:
        Direction: HIGHER (1.0 = perfect agreement).
        Range: [-1, 1].
        Cost: O(n^2) in time and memory, weighted or not, because it materialises the
        pairwise difference matrices.

    Args:
        a: First variable.
        b: Second variable.
        mask: Elements to keep (boolean, broadcastable), or ``None`` for all.
        weights: Element frequency weights (broadcastable), or ``None`` for unit weights.

    Returns:
        Kendall tau as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> kendall_tau(jnp.array([1.0, 2.0, 3.0]),
        ...             jnp.array([1.0, 2.0, 3.0]))
        1.0
    """
    a_arr = jnp.asarray(a).ravel()
    b_arr = jnp.asarray(b).ravel()
    n = a_arr.shape[0]

    # Pairwise comparisons using broadcasting
    a_diff = a_arr[:, None] - a_arr[None, :]  # (n, n)
    b_diff = b_arr[:, None] - b_arr[None, :]  # (n, n)

    # Upper triangle only (avoid double counting and diagonal)
    upper = jnp.triu(jnp.ones((n, n), dtype=jnp.bool_), k=1)
    agreement = jnp.sign(a_diff) * jnp.sign(b_diff)

    if mask is None and weights is None:
        concordant_minus_discordant = jnp.sum(jnp.where(upper, agreement, 0.0))
        total_pairs = n * (n - 1) / 2
        tied_in_a = jnp.sum(upper & (a_diff == 0))
        tied_in_b = jnp.sum(upper & (b_diff == 0))
    else:
        # One weight per pair, zero wherever either element is excluded, so a masked
        # element takes every pair it belongs to out of all three totals at once.
        w = sample_weights(mask, weights, a_arr.shape)
        pair_weight = jnp.where(upper, w[:, None] * w[None, :], 0.0)
        concordant_minus_discordant = jnp.sum(pair_weight * agreement)
        total_pairs = jnp.sum(pair_weight)
        tied_in_a = jnp.sum(jnp.where(a_diff == 0, pair_weight, 0.0))
        tied_in_b = jnp.sum(jnp.where(b_diff == 0, pair_weight, 0.0))

    untied_pairs = jnp.sqrt((total_pairs - tied_in_a) * (total_pairs - tied_in_b))
    return concordant_minus_discordant / (untied_pairs + _EPSILON)


def concordance_correlation(
    a: ArrayLike,
    b: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> jax.Array:
    """Lin's concordance correlation coefficient.

    Measures agreement (not just correlation). Penalizes deviations
    from the identity line, unlike Pearson which only measures
    linear association.

    Note:
        Direction: HIGHER (1.0 = perfect agreement).
        Range: [-1, 1].
        CCC <= |Pearson r|. Equal only when means and variances match.

    With ``weights`` all four moments are weighted -- both means, both variances and the
    covariance -- which is Lin's coefficient computed on the weighted sample. Equal
    weights reproduce the unweighted value exactly, which a test pins.

    No ``reduction`` or ``axis``: the coefficient is a ratio of reductions over the whole
    sample, so there is no per-element value to return.

    Args:
        a: First variable.
        b: Second variable.
        mask: Elements to keep (boolean, broadcastable), or ``None`` for all.
        weights: Element weights (broadcastable), or ``None`` for unit weights.

    Returns:
        Concordance correlation as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> concordance_correlation(jnp.array([1.0, 2.0, 3.0]),
        ...                         jnp.array([1.0, 2.0, 3.0]))
        1.0
    """
    a_arr = jnp.asarray(a).ravel()
    b_arr = jnp.asarray(b).ravel()
    w = sample_weights(mask, weights, a_arr.shape)
    total = jnp.sum(w) + _EPSILON
    mean_a = weighted_mean(a_arr, w)
    mean_b = weighted_mean(b_arr, w)
    var_a = jnp.sum(w * (a_arr - mean_a) ** 2) / total
    var_b = jnp.sum(w * (b_arr - mean_b) ** 2) / total
    cov = jnp.sum(w * (a_arr - mean_a) * (b_arr - mean_b)) / total
    denom = var_a + var_b + (mean_a - mean_b) ** 2
    return 2.0 * cov / (denom + _EPSILON)


def correlation_preservation(real: ArrayLike, generated: ArrayLike) -> jax.Array:
    """How closely generated data reproduces the feature correlations of real data.

    One minus the mean absolute difference of the two Pearson correlation
    matrices over their off-diagonal entries, clipped to ``[0, 1]``. A constant
    feature has an undefined correlation, which counts as zero.

    Note:
        Direction: HIGHER (1.0 = identical correlation structure).
        Range: [0, 1].

    Args:
        real: Real records with shape ``(n_real, n_features)``.
        generated: Generated records with shape ``(n_generated, n_features)``.

    Returns:
        Scalar preservation score as a JAX array; ``1.0`` below two features.

    Raises:
        ValueError: If either input is not two-dimensional or the feature
            dimensions differ.
    """
    real_matrix = jnp.asarray(real, dtype=jnp.float32)
    generated_matrix = jnp.asarray(generated, dtype=jnp.float32)
    if real_matrix.ndim != _FEATURE_MATRIX_NDIM or generated_matrix.ndim != _FEATURE_MATRIX_NDIM:
        msg = f"records must be 2-dimensional, got {real_matrix.shape} and {generated_matrix.shape}"
        raise ValueError(msg)
    if real_matrix.shape[1] != generated_matrix.shape[1]:
        msg = (
            "real and generated records must share their feature dimension: "
            f"{real_matrix.shape[1]} != {generated_matrix.shape[1]}"
        )
        raise ValueError(msg)
    n_features = real_matrix.shape[1]
    if n_features < _MIN_CORRELATED_FEATURES:
        return jnp.asarray(1.0, dtype=jnp.float32)
    real_corr = jnp.nan_to_num(jnp.corrcoef(real_matrix, rowvar=False))
    generated_corr = jnp.nan_to_num(jnp.corrcoef(generated_matrix, rowvar=False))
    off_diagonal = 1.0 - jnp.eye(n_features)
    mean_abs_diff = jnp.sum(jnp.abs(real_corr - generated_corr) * off_diagonal) / jnp.sum(
        off_diagonal
    )
    return 1.0 - jnp.clip(mean_abs_diff, 0.0, 1.0)


def autocorrelation(series: ArrayLike, *, max_lag: int) -> jax.Array:
    """Autocorrelation function of a batch of sequences, averaged over batch and features.

    Sequences are centred per sequence; the lag-``k`` value is the mean product of
    the series with itself shifted by ``k``, normalised by the lag-0 value so the
    function starts at 1.

    Args:
        series: Sequences with shape ``(batch, sequence, features)``.
        max_lag: Number of lags to return, lag 0 included.

    Returns:
        Autocorrelation values with shape ``(max_lag,)``.

    Raises:
        ValueError: If ``max_lag`` exceeds the sequence length or is not positive.
    """
    data = jnp.asarray(series, dtype=jnp.float32)
    if data.ndim != _SERIES_NDIM:
        msg = f"series must have shape (batch, sequence, features), got {data.shape}"
        raise ValueError(msg)
    sequence_length = data.shape[1]
    if max_lag < 1 or max_lag > sequence_length:
        msg = (
            f"max_lag must be between 1 and the sequence length ({sequence_length}), got {max_lag}"
        )
        raise ValueError(msg)
    centred = data - jnp.mean(data, axis=1, keepdims=True)
    values = [jnp.mean(centred**2)]
    values.extend(jnp.mean(centred[:, :-lag, :] * centred[:, lag:, :]) for lag in range(1, max_lag))
    function = jnp.stack(values)
    return jnp.where(function[0] > 0.0, function / (function[0] + _EPSILON), function)


def skewness(data: ArrayLike) -> jax.Array:
    """Skewness of a sample: the third standardised moment.

    Note:
        Direction: INFO (0 for a symmetric sample; positive for a right tail).
        Range: (-inf, inf).

    Args:
        data: Sample values of any shape; all elements are pooled.

    Returns:
        Scalar skewness as a JAX array; ``0.0`` for a constant sample.
    """
    values = jnp.asarray(data, dtype=jnp.float32).ravel()
    mean = jnp.mean(values)
    std = jnp.std(values)
    standardised = (values - mean) / jnp.where(std > 0.0, std, 1.0)
    return jnp.where(std > 0.0, jnp.mean(standardised**3), 0.0)
