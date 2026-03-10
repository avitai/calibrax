"""Information-theoretic functions.

Pure functions for measuring information content, entropy, and
dependence between random variables. Based on Shannon's information
theory and Fisher's information geometry.

Includes 6 functions: entropy, cross_entropy, mutual_information,
conditional_entropy, normalized_mutual_information,
fisher_information_matrix.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON, safe_log


def entropy(p: Any) -> Any:
    """Shannon entropy: ``-sum(p * log(p))``.

    Measures uncertainty or disorder of a probability distribution.
    Maximum for uniform distribution, zero for deterministic.

    Note:
        Direction: INFO (neither higher nor lower is inherently better).
        Range: [0, log(n)] where n is the number of outcomes.
        Uses natural logarithm.

    Args:
        p: Probability vector (must sum to ~1).

    Returns:
        Shannon entropy as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> entropy(jnp.array([0.5, 0.5]))  # ln(2) ≈ 0.693
        0.693...
    """
    p_arr = jnp.asarray(p).ravel()
    p_safe = jnp.maximum(p_arr, _EPSILON)
    return -jnp.sum(p_arr * safe_log(p_safe))


def cross_entropy(p: Any, q: Any) -> Any:
    """Cross-entropy: ``-sum(p * log(q))``.

    Measures the average number of nats needed to encode data from p
    using distribution q. Always >= entropy(p).

    Note:
        Direction: LOWER (lower = better approximation of p by q).
        Range: [0, inf).
        ``cross_entropy(p, q) = entropy(p) + kl_divergence(p, q)``.

    Args:
        p: True distribution (probability vector).
        q: Coding distribution (probability vector).

    Returns:
        Cross-entropy as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> p = jnp.array([0.5, 0.5])
        >>> cross_entropy(p, p)  # equals entropy(p) ≈ 0.693
        0.693...
    """
    p_arr = jnp.asarray(p).ravel()
    q_arr = jnp.asarray(q).ravel()
    return -jnp.sum(p_arr * safe_log(q_arr))


def mutual_information(joint: Any) -> Any:
    """Mutual information from a joint probability table.

    ``MI(X;Y) = sum_{i,j} p(i,j) * log(p(i,j) / (p(i)*p(j)))``.
    Measures statistical dependence between X and Y.

    Note:
        Direction: HIGHER (more information = stronger dependence).
        Range: [0, min(H(X), H(Y))].
        Symmetric: MI(X;Y) = MI(Y;X).

    Args:
        joint: 2D joint probability table of shape (n_x, n_y).
            Must sum to ~1.

    Returns:
        Mutual information as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> # Independent: p(x,y) = p(x)*p(y)
        >>> joint = jnp.array([[0.25, 0.25], [0.25, 0.25]])
        >>> mutual_information(joint)
        0.0
    """
    joint_arr = jnp.asarray(joint)
    # Marginals
    p_x = jnp.sum(joint_arr, axis=1)
    p_y = jnp.sum(joint_arr, axis=0)
    # Outer product of marginals
    expected = p_x[:, None] * p_y[None, :]
    # MI = sum p(x,y) * log(p(x,y) / p(x)p(y))
    joint_safe = jnp.maximum(joint_arr, _EPSILON)
    expected_safe = jnp.maximum(expected, _EPSILON)
    ratio = joint_safe / expected_safe
    terms = joint_arr * safe_log(ratio)
    # Zero out where joint is essentially zero
    terms = jnp.asarray(jnp.where(joint_arr > _EPSILON, terms, 0.0))
    return jnp.sum(terms)


def conditional_entropy(joint: Any) -> Any:
    """Conditional entropy H(Y|X) from a joint probability table.

    ``H(Y|X) = H(X,Y) - H(X)``. Measures remaining uncertainty
    about Y given knowledge of X.

    Note:
        Direction: LOWER (less uncertainty = better prediction).
        Range: [0, H(Y)].

    Args:
        joint: 2D joint probability table of shape (n_x, n_y).
            Must sum to ~1.

    Returns:
        Conditional entropy as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> # Perfect dependence: knowing X determines Y
        >>> joint = jnp.array([[0.5, 0.0], [0.0, 0.5]])
        >>> conditional_entropy(joint)
        0.0
    """
    joint_arr = jnp.asarray(joint)
    # H(X,Y)
    h_joint = entropy(joint_arr.ravel())
    # H(X)
    p_x = jnp.sum(joint_arr, axis=1)
    h_x = entropy(p_x)
    return h_joint - h_x


def normalized_mutual_information(joint: Any) -> Any:
    """Normalized mutual information: ``MI / sqrt(H(X) * H(Y))``.

    Bounded version of MI for comparing across different scales.

    Note:
        Direction: HIGHER (1.0 = perfect dependence).
        Range: [0, 1].
        Symmetric.

    Args:
        joint: 2D joint probability table of shape (n_x, n_y).
            Must sum to ~1.

    Returns:
        Normalized MI as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> joint = jnp.array([[0.5, 0.0], [0.0, 0.5]])
        >>> normalized_mutual_information(joint)
        1.0
    """
    joint_arr = jnp.asarray(joint)
    mi = mutual_information(joint_arr)
    p_x = jnp.sum(joint_arr, axis=1)
    p_y = jnp.sum(joint_arr, axis=0)
    h_x = entropy(p_x)
    h_y = entropy(p_y)
    h_combined = jnp.sqrt(h_x * h_y)
    nmi_value = mi / (h_combined + _EPSILON)
    return jnp.where(h_combined < _EPSILON, 1.0, nmi_value)


def fisher_information_matrix(
    log_prob_fn: Callable[..., Any],
    params: Any,
) -> Any:
    """Fisher information matrix at given parameter values.

    ``I(theta) = -E[nabla^2 log p(x|theta)]``. The unique Riemannian
    metric invariant under sufficient statistics (Chentsov's theorem).

    Note:
        NOT registered as a scalar metric — returns a matrix.
        Useful for natural gradient methods and information geometry.

    Args:
        log_prob_fn: Log-probability function taking params as input.
        params: Parameter values at which to compute the Fisher matrix.

    Returns:
        Fisher information matrix as a JAX array of shape
        (num_params, num_params).

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> def log_prob(theta):
        ...     return -0.5 * jnp.sum(theta ** 2)
        >>> fim = fisher_information_matrix(log_prob, jnp.array([1.0, 2.0]))
        >>> fim.shape
        (2, 2)
    """
    params_arr = jnp.asarray(params)
    hessian = jax.hessian(log_prob_fn)(params_arr)
    return -hessian
