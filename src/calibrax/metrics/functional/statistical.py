"""Statistical correlation and agreement metrics.

Pure functions for measuring statistical relationships between
variables. Covers linear correlation, rank correlation, and
agreement measures.

Includes 8 functions: pearson_correlation, spearman_rank_correlation,
kendall_tau, concordance_correlation, r_squared_adjusted, correlation_preservation,
autocorrelation, skewness.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON


def pearson_correlation(a: Any, b: Any) -> Any:
    """Pearson correlation coefficient.

    Linear correlation: ``cov(a,b) / (std(a) * std(b))``.

    Note:
        Direction: HIGHER (1.0 = perfect positive correlation).
        Range: [-1, 1].
        Measures linear association only.

    Args:
        a: First variable.
        b: Second variable.

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
    a_centered = a_arr - jnp.mean(a_arr)
    b_centered = b_arr - jnp.mean(b_arr)
    cov = jnp.sum(a_centered * b_centered)
    std_a = jnp.sqrt(jnp.sum(a_centered**2))
    std_b = jnp.sqrt(jnp.sum(b_centered**2))
    return cov / (std_a * std_b + _EPSILON)


def spearman_rank_correlation(a: Any, b: Any) -> Any:
    """Spearman's rank correlation coefficient.

    Pearson correlation computed on ranks. Measures monotonic association.

    Note:
        Direction: HIGHER (1.0 = perfect monotonic relationship).
        Range: [-1, 1].

    Args:
        a: First variable.
        b: Second variable.

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
    # Convert to ranks (0-indexed)
    rank_a = jnp.argsort(jnp.argsort(a_arr)).astype(jnp.float32)
    rank_b = jnp.argsort(jnp.argsort(b_arr)).astype(jnp.float32)
    return pearson_correlation(rank_a, rank_b)


def kendall_tau(a: Any, b: Any) -> Any:
    """Kendall rank correlation coefficient (tau-b).

    ``(concordant - discordant) / (n*(n-1)/2)``.

    Note:
        Direction: HIGHER (1.0 = perfect agreement).
        Range: [-1, 1].

    Args:
        a: First variable.
        b: Second variable.

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
    n = len(a_arr)

    # Pairwise comparisons using broadcasting
    a_diff = a_arr[:, None] - a_arr[None, :]  # (n, n)
    b_diff = b_arr[:, None] - b_arr[None, :]  # (n, n)

    # Upper triangle only (avoid double counting and diagonal)
    mask = jnp.triu(jnp.ones((n, n), dtype=jnp.bool_), k=1)
    concordant = jnp.sum(mask & (jnp.sign(a_diff) == jnp.sign(b_diff)) & (a_diff != 0))
    discordant = jnp.sum(
        mask & (jnp.sign(a_diff) != jnp.sign(b_diff)) & (a_diff != 0) & (b_diff != 0)
    )
    total_pairs = n * (n - 1) / 2

    return (concordant - discordant) / (total_pairs + _EPSILON)


def concordance_correlation(a: Any, b: Any) -> Any:
    """Lin's concordance correlation coefficient.

    Measures agreement (not just correlation). Penalizes deviations
    from the identity line, unlike Pearson which only measures
    linear association.

    Note:
        Direction: HIGHER (1.0 = perfect agreement).
        Range: [-1, 1].
        CCC <= |Pearson r|. Equal only when means and variances match.

    Args:
        a: First variable.
        b: Second variable.

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
    mean_a = jnp.mean(a_arr)
    mean_b = jnp.mean(b_arr)
    var_a = jnp.var(a_arr)
    var_b = jnp.var(b_arr)
    cov = jnp.mean((a_arr - mean_a) * (b_arr - mean_b))
    denom = var_a + var_b + (mean_a - mean_b) ** 2
    return 2.0 * cov / (denom + _EPSILON)


def r_squared_adjusted(
    predictions: Any,
    targets: Any,
    *,
    num_predictors: int,
) -> Any:
    """Adjusted R-squared.

    ``1 - (1-R^2)(n-1)/(n-p-1)`` where p is number of predictors.
    Penalizes adding predictors that don't improve fit.

    Note:
        Direction: HIGHER (1.0 = perfect fit).
        Range: (-inf, 1].

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        num_predictors: Number of predictors in the model.

    Returns:
        Adjusted R-squared as a scalar value.
    """
    p_arr = jnp.asarray(predictions).ravel()
    t_arr = jnp.asarray(targets).ravel()
    n = len(t_arr)

    ss_res = jnp.sum((t_arr - p_arr) ** 2)
    ss_tot = jnp.sum((t_arr - jnp.mean(t_arr)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + _EPSILON)

    return 1.0 - (1.0 - r2) * (n - 1) / (n - num_predictors - 1 + _EPSILON)


def correlation_preservation(real: Any, generated: Any) -> Any:
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
    if real_matrix.ndim != 2 or generated_matrix.ndim != 2:  # noqa: PLR2004
        msg = f"records must be 2-dimensional, got {real_matrix.shape} and {generated_matrix.shape}"
        raise ValueError(msg)
    if real_matrix.shape[1] != generated_matrix.shape[1]:
        msg = (
            "real and generated records must share their feature dimension: "
            f"{real_matrix.shape[1]} != {generated_matrix.shape[1]}"
        )
        raise ValueError(msg)
    n_features = real_matrix.shape[1]
    if n_features < 2:  # noqa: PLR2004
        return jnp.asarray(1.0, dtype=jnp.float32)
    real_corr = jnp.nan_to_num(jnp.corrcoef(real_matrix, rowvar=False))
    generated_corr = jnp.nan_to_num(jnp.corrcoef(generated_matrix, rowvar=False))
    off_diagonal = 1.0 - jnp.eye(n_features)
    mean_abs_diff = jnp.sum(jnp.abs(real_corr - generated_corr) * off_diagonal) / jnp.sum(
        off_diagonal
    )
    return 1.0 - jnp.clip(mean_abs_diff, 0.0, 1.0)


def autocorrelation(series: Any, *, max_lag: int) -> Any:
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
    if data.ndim != 3:  # noqa: PLR2004
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


def skewness(data: Any) -> Any:
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
