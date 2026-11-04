"""Statistical correlation and agreement metrics.

Pure functions for measuring statistical relationships between
variables. Covers linear correlation, rank correlation, and
agreement measures.

Includes 7 functions: pearson_correlation, spearman_rank_correlation,
kendall_tau, concordance_correlation, correlation_preservation, autocorrelation,
skewness.

``r_squared_adjusted`` is not here. It is a goodness-of-fit measure with a
degrees-of-freedom penalty rather than a correlation or an agreement measure, and it shares
its implementation with ``r_squared``, so both live in
:mod:`calibrax.metrics.functional.regression`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.stats import rankdata
from jax.typing import ArrayLike

from calibrax.metrics._utils import (
    _EPSILON,
    _FEATURE_MATRIX_NDIM,
    _MIN_CORRELATED_FEATURES,
    _SERIES_NDIM,
    nan_when_empty,
    safe_root,
    sample_weights,
    substitute_excluded,
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
        Pearson r as a scalar value; ``NaN`` for a selection with no members.

    Examples:
        >>> import jax.numpy as jnp
        >>> pearson_correlation(jnp.array([1.0, 2.0, 3.0]),
        ...                     jnp.array([1.0, 2.0, 3.0]))
        1.0
    """
    a_raw = jnp.asarray(a)
    a_arr = a_raw.ravel()
    b_arr = jnp.asarray(b).ravel()
    if mask is None and weights is None:
        # The unweighted path expresses "no weights" by not having any, rather than by
        # multiplying through a vector of ones. Both the cost and the answer turn on that:
        # routing it through `sample_weights` and `weighted_mean` moved the result by up to
        # 1 float32 ULP against the pre-weights implementation, and took one eager call from
        # 462 Python-level calls to 1014 -- on the branch every existing caller takes. The
        # suite's one benchmark covers `mse`, so nothing caught it.
        a_centered = a_arr - jnp.mean(a_arr)
        b_centered = b_arr - jnp.mean(b_arr)
        cov = jnp.sum(a_centered * b_centered)
        return cov / (
            jnp.sqrt(jnp.sum(a_centered**2)) * jnp.sqrt(jnp.sum(b_centered**2)) + _EPSILON
        )

    # Fold the mask and weights at the input's own shape, then flatten alongside the data.
    # Broadcasting against the raveled shape instead would refuse a mask shaped like a
    # multi-dimensional input, which is the shape a caller naturally has.
    w = sample_weights(mask, weights, a_raw.shape).ravel()
    flat_mask = (
        None
        if mask is None
        else jnp.broadcast_to(jnp.asarray(mask, dtype=bool), a_raw.shape).ravel()
    )
    # Before any arithmetic, so an excluded NaN reaches neither the moments nor the
    # gradient. The unweighted path returned above, so this always has work to do.
    given = None if weights is None else w
    a_arr = substitute_excluded(a_arr, flat_mask, given)
    b_arr = substitute_excluded(b_arr, flat_mask, given)
    return _weighted_pearson(a_arr, b_arr, w)


def _weighted_pearson(a: jax.Array, b: jax.Array, w: jax.Array) -> jax.Array:
    """The weighted Pearson r of two flat series whose excluded elements are already replaced.

    Args:
        a: First series, flat.
        b: Second series, flat.
        w: Folded weights, flat.

    Returns:
        Pearson r; ``NaN`` for a selection with no weight.
    """
    a_centered = a - weighted_mean(a, w)
    b_centered = b - weighted_mean(b, w)
    cov = jnp.sum(w * a_centered * b_centered)
    # The guarded root, because an empty selection or a constant input drives these sums
    # to zero, where the plain root's derivative is infinite and `d/dw` comes back non-finite
    # in every entry -- including the elements that were not the problem. The value is the
    # same number at every input; only the derivative at zero differs.
    std_a = safe_root(jnp.sum(w * a_centered**2))
    std_b = safe_root(jnp.sum(w * b_centered**2))
    return nan_when_empty(cov / (std_a * std_b + _EPSILON), w)


@jax.jit
def _frequency_ranks(values: jax.Array, weights: jax.Array) -> jax.Array:
    """Weighted midranks: ``jax.scipy.stats.rankdata``'s average ranks, weights for counts.

    ``rankdata`` sorts, marks where each run of equal values starts, finds the runs'
    boundaries with ``nonzero`` and averages each run's first and last position,
    ``.5 * (count[dense] + count[dense - 1] + 1)``, where ``count`` holds positions -- the
    number of elements before each boundary. With frequency weights the number of elements
    becomes the weight of elements, so ``count`` holds cumulative weights and the average is
    R Hmisc's ``wtd.rank``, ``cumsum(freqs) - .5*(freqs-1)`` per distinct value. For integer
    weights a rank is the midrank of the sample with each element repeated that many times,
    and with unit weights it is ``rankdata``'s. Each step below is ``rankdata``'s own, and so
    is the ``jax.jit`` around them; its private ``invert_permutation`` helper is the one
    ``.at[].set`` expression written inline.

    The ``jit`` is what makes the ``lax.cond`` cheap: an eager ``cond`` traces its branches on
    every call. Measured at 10,000 elements, without it a masked eager call took 49.4 ms on an
    RTX 4090 and 35.4 ms on CPU; with it, 1.5 ms and 4.1 ms. It also made the compiled
    weighted path at 1,000,000 elements 1.8x faster on CPU, 209.7 against 374.9 ms, with the
    same fusion count either way.

    Args:
        values: Values to rank, flat.
        weights: Their non-negative weights, flat.

    Returns:
        Each element's weighted midrank, or ``NaN`` everywhere if a value is ``NaN``, as
        ``rankdata`` returns.
    """
    size = values.size
    ordered, sorter = lax.sort_key_val(values, jnp.arange(size))
    inverse = jnp.empty_like(sorter).at[sorter].set(jnp.arange(size, dtype=sorter.dtype))
    obs = jnp.concatenate([jnp.array([True]), ordered[1:] != ordered[:-1]])
    dense = obs.cumsum()[inverse]
    boundaries = jnp.nonzero(obs, size=size + 1, fill_value=size)[0]
    cumulative = jnp.concatenate([jnp.zeros(1, dtype=weights.dtype), jnp.cumsum(weights[sorter])])
    count = cumulative[boundaries]
    ranks = 0.5 * (count[dense] + count[dense - 1] + 1.0)
    return lax.cond(
        jnp.any(jnp.isnan(values)), lambda r: jnp.full_like(r, jnp.nan), lambda r: r, ranks
    )


def spearman_rank_correlation(
    a: ArrayLike,
    b: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> jax.Array:
    """Spearman's rank correlation coefficient.

    Pearson correlation computed on ranks, which tied values share as their average
    (``jax.scipy.stats.rankdata``), so the coefficient does not depend on the order the pairs
    are given in.

    **Weights are frequency weights.** A weight ``w_i`` counts element ``i`` as observed
    ``w_i`` times, so for integer weights the coefficient is Spearman's of the sample with
    each element repeated that many times -- what SPSS computes for weighted cases (R's
    ``expss`` reproduces it by repeating rows) and what SAS's ``FREQ`` statement means.
    Integer weights equal that replication to within float32 of
    ``scipy.stats.spearmanr``. The ranks are R Hmisc's ``wtd.rank``, the weight below a value
    plus the midpoint of the weight tied with it, and the coefficient is their weighted
    Pearson r, which is how Spearman's coefficient of a contingency table is computed from
    its counts. ``wtd.rank`` takes any non-negative weights, so a weight need not be an
    integer: the ranks are the same function of the weights.

    Sampling or probability weights are a different question, with no accepted answer: R's
    ``wCorr`` uses its own convention, which does not reproduce replication, and Stata and
    SAS's ``WEIGHT`` statement offer none. Those are not what these weights mean.

    A mask is a 0/1 weight: masked, the coefficient is the retained subset's.

    Note:
        Direction: HIGHER (1.0 = perfect monotonic relationship).
        Range: [-1, 1].

    Args:
        a: First variable.
        b: Second variable.
        mask: Elements to keep (boolean, broadcastable), or ``None`` for all. Ranks are
            computed within the retained subset.
        weights: Frequency weights (non-negative, broadcastable), or ``None`` for one each.

    Returns:
        Spearman rho as a scalar value; ``NaN`` for a selection with no members.

    Examples:
        >>> import jax.numpy as jnp
        >>> spearman_rank_correlation(jnp.array([1.0, 2.0, 3.0]),
        ...                          jnp.array([1.0, 2.0, 3.0]))
        1.0
    """
    a_raw = jnp.asarray(a)
    a_arr = a_raw.ravel()
    b_arr = jnp.asarray(b).ravel()
    if mask is None and weights is None:
        return pearson_correlation(rankdata(a_arr), rankdata(b_arr))
    # Fold at the input's shape and flatten alongside the data, as the other statistics do.
    w = sample_weights(mask, weights, a_raw.shape).ravel()
    flat_mask = (
        None
        if mask is None
        else jnp.broadcast_to(jnp.asarray(mask, dtype=bool), a_raw.shape).ravel()
    )
    # An excluded element is replaced before ranking, so a NaN there reaches no rank. Its
    # weight is zero, so it adds nothing to any group and no rank moves.
    given = None if weights is None else w
    a_arr = substitute_excluded(a_arr, flat_mask, given)
    b_arr = substitute_excluded(b_arr, flat_mask, given)
    return _weighted_pearson(_frequency_ranks(a_arr, w), _frequency_ranks(b_arr, w), w)


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
    pairs, so a per-element weight lifts to the pair weight ``w_i * w_j``, forced by the
    statistic's own shape. Measured over 60 fixtures,
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
    expensive function in this module. Two plausible justifications for the branch are
    wrong, and are recorded so they are not re-derived: it is not that float32 loses
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
        Kendall tau as a scalar value; ``NaN`` for a selection with no members.

    Examples:
        >>> import jax.numpy as jnp
        >>> kendall_tau(jnp.array([1.0, 2.0, 3.0]),
        ...             jnp.array([1.0, 2.0, 3.0]))
        1.0
    """
    a_raw = jnp.asarray(a)
    a_arr = a_raw.ravel()
    b_arr = jnp.asarray(b).ravel()
    n = a_arr.shape[0]

    # Fold the mask and weights first, where there are any, and neutralise the excluded
    # elements before the differences are formed. Doing it afterwards would mean every
    # pair an excluded element belongs to carries its NaN, and the gradient would keep it
    # even once the value was repaired.
    pair_scale = (
        None
        if mask is None and weights is None
        else sample_weights(mask, weights, a_raw.shape).ravel()
    )
    if pair_scale is not None:
        flat_mask = (
            None
            if mask is None
            else jnp.broadcast_to(jnp.asarray(mask, dtype=bool), a_raw.shape).ravel()
        )
        given = None if weights is None else pair_scale
        a_arr = substitute_excluded(a_arr, flat_mask, given)
        b_arr = substitute_excluded(b_arr, flat_mask, given)

    # Pairwise comparisons using broadcasting
    a_diff = a_arr[:, None] - a_arr[None, :]  # (n, n)
    b_diff = b_arr[:, None] - b_arr[None, :]  # (n, n)

    # Upper triangle only (avoid double counting and diagonal)
    upper = jnp.triu(jnp.ones((n, n), dtype=jnp.bool_), k=1)
    agreement = jnp.sign(a_diff) * jnp.sign(b_diff)

    if pair_scale is None:
        concordant_minus_discordant = jnp.sum(jnp.where(upper, agreement, 0.0))
        total_pairs = n * (n - 1) / 2
        tied_in_a = jnp.sum(upper & (a_diff == 0))
        tied_in_b = jnp.sum(upper & (b_diff == 0))
        # Spelled exactly as it was before the weighted path existed, and kept in this
        # branch rather than shared below, so the unweighted compiled graph does not move.
        # Nothing here can reach the root's singularity: `total_pairs` is a positive
        # constant and the tie counts are integers below it unless every pair is tied, which
        # needs a constant input, and a constant input has no unweighted caller to serve.
        untied_pairs = jnp.sqrt((total_pairs - tied_in_a) * (total_pairs - tied_in_b))
    else:
        # One weight per pair, zero wherever either element is excluded, so a masked element
        # leaves every pair it belongs to out of all three totals at once. Folded at the
        # input's own shape then flattened, so the pair weights line up with the flattened
        # elements however the caller shaped them.
        pair_weight = jnp.where(upper, pair_scale[:, None] * pair_scale[None, :], 0.0)
        concordant_minus_discordant = jnp.sum(pair_weight * agreement)
        total_pairs = jnp.sum(pair_weight)
        tied_in_a = jnp.sum(jnp.where(a_diff == 0, pair_weight, 0.0))
        tied_in_b = jnp.sum(jnp.where(b_diff == 0, pair_weight, 0.0))
        # The guarded root here, where all three totals are weighted sums that an empty
        # selection drives to zero together. A constant input reaches it exactly as well:
        # every pair is tied, so `total_pairs - tied_in_b` is zero rather than merely small.
        untied_pairs = safe_root((total_pairs - tied_in_a) * (total_pairs - tied_in_b))
        return nan_when_empty(concordant_minus_discordant / (untied_pairs + _EPSILON), pair_scale)

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
    covariance -- which is Lin's coefficient computed on the weighted sample. Equal weights
    reproduce the unweighted value to within four float32 ULP, measured at 2.384e-07 over
    2,160 cases, rather than exactly: the unweighted path is a ``jnp.var``/``jnp.mean``
    expression, which keeps the default's published value stable and is not bitwise the same
    expression as a weighted sum over its own total.
    ``r_squared`` and ``pearson_correlation`` are bit-exact on the same grid; this one
    cannot be, and a test records the bound.

    No ``reduction`` or ``axis``: the coefficient is a ratio of reductions over the whole
    sample, so there is no per-element value to return.

    Args:
        a: First variable.
        b: Second variable.
        mask: Elements to keep (boolean, broadcastable), or ``None`` for all.
        weights: Element weights (broadcastable), or ``None`` for unit weights.

    Returns:
        Concordance correlation as a scalar value; ``NaN`` for a selection with no
        members.

    Examples:
        >>> import jax.numpy as jnp
        >>> concordance_correlation(jnp.array([1.0, 2.0, 3.0]),
        ...                         jnp.array([1.0, 2.0, 3.0]))
        1.0
    """
    a_raw = jnp.asarray(a)
    a_arr = a_raw.ravel()
    b_arr = jnp.asarray(b).ravel()
    if mask is None and weights is None:
        # As in `pearson_correlation`: the ones vector is neither free nor bit-neutral.
        # Measured against the pre-weights implementation, routing this path through it
        # moved the result by up to 2 float32 ULP and took an eager call from 424
        # Python-level calls to 1239.
        mean_a = jnp.mean(a_arr)
        mean_b = jnp.mean(b_arr)
        cov = jnp.mean((a_arr - mean_a) * (b_arr - mean_b))
        denom = jnp.var(a_arr) + jnp.var(b_arr) + (mean_a - mean_b) ** 2
        return 2.0 * cov / (denom + _EPSILON)

    w = sample_weights(mask, weights, a_raw.shape).ravel()
    flat_mask = (
        None
        if mask is None
        else jnp.broadcast_to(jnp.asarray(mask, dtype=bool), a_raw.shape).ravel()
    )
    # Before any arithmetic, so an excluded NaN reaches neither the moments nor the
    # gradient. The unweighted path returned above, so this always has work to do.
    given = None if weights is None else w
    a_arr = substitute_excluded(a_arr, flat_mask, given)
    b_arr = substitute_excluded(b_arr, flat_mask, given)
    total = jnp.sum(w) + _EPSILON
    mean_a = weighted_mean(a_arr, w)
    mean_b = weighted_mean(b_arr, w)
    var_a = jnp.sum(w * (a_arr - mean_a) ** 2) / total
    var_b = jnp.sum(w * (b_arr - mean_b) ** 2) / total
    cov = jnp.sum(w * (a_arr - mean_a) * (b_arr - mean_b)) / total
    denom = var_a + var_b + (mean_a - mean_b) ** 2
    return nan_when_empty(2.0 * cov / (denom + _EPSILON), w)


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
