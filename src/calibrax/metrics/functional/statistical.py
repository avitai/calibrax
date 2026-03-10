"""Statistical correlation and agreement metrics.

Pure functions for measuring statistical relationships between
variables. Covers linear correlation, rank correlation, and
agreement measures.

Includes 5 functions: pearson_correlation, spearman_rank_correlation,
kendall_tau, concordance_correlation, r_squared_adjusted.
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

    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - num_predictors - 1 + _EPSILON)
    return adj
