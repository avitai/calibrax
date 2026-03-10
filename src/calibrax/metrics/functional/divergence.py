"""Statistical divergence functions between probability distributions.

Pure functions for measuring dissimilarity between distributions. Divergences
are generally asymmetric (Finslerian in nature), distinguishing them from
true distance metrics. This module covers f-divergences (KL, JS, Hellinger,
chi-squared, Renyi, TV), optimal transport metrics (Wasserstein, Sinkhorn,
sliced Wasserstein), kernel-based metrics (MMD), and Bregman divergences.

Includes 13 functions: kl_divergence, js_divergence, wasserstein_1d, mmd,
total_variation, reverse_kl_divergence, hellinger_distance,
chi_squared_divergence, renyi_divergence, f_divergence,
sinkhorn_divergence, sliced_wasserstein, bregman_divergence.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON, safe_divide, safe_log


def kl_divergence(p: Any, q: Any) -> Any:
    """Kullback-Leibler divergence: ``sum(p * log(p / q))``.

    Measures information lost when q is used to approximate p.
    NOT symmetric: ``KL(p||q) != KL(q||p)``.

    Note:
        Direction: LOWER (0.0 = identical distributions).
        Range: [0, inf).
        Not symmetric, not a true metric.
        Requires p and q to be probability vectors (sum to ~1).

    Args:
        p: True distribution (probability vector).
        q: Approximate distribution (probability vector).

    Returns:
        KL divergence as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> p = jnp.array([0.5, 0.5])
        >>> kl_divergence(p, p)
        0.0
    """
    p_arr = jnp.asarray(p).ravel()
    q_arr = jnp.asarray(q).ravel()
    # Only sum where p > 0 (0 * log(0/q) = 0 by convention)
    ratio = safe_divide(p_arr, q_arr)
    terms = p_arr * safe_log(ratio)
    # Zero out terms where p is essentially zero
    terms = jnp.asarray(jnp.where(p_arr > _EPSILON, terms, 0.0))
    return jnp.sum(terms)


def reverse_kl_divergence(p: Any, q: Any) -> Any:
    """Reverse KL divergence: ``KL(q || p)``.

    Mode-seeking variant, useful for variational inference.
    Penalizes q for placing mass where p has none.

    Note:
        Direction: LOWER (0.0 = identical distributions).
        Range: [0, inf).
        Not symmetric. ``reverse_kl(p,q) = kl(q,p)``.

    Args:
        p: First distribution.
        q: Second distribution.

    Returns:
        Reverse KL divergence as a scalar value.
    """
    return kl_divergence(q, p)


def js_divergence(p: Any, q: Any) -> Any:
    """Jensen-Shannon divergence.

    Symmetric, bounded version of KL divergence:
    ``JS(p,q) = 0.5 * KL(p||m) + 0.5 * KL(q||m)`` where ``m = 0.5*(p+q)``.

    Note:
        Direction: LOWER (0.0 = identical distributions).
        Range: [0, ln(2)] with natural log.
        Symmetric. Square root of JS is a true metric.

    Args:
        p: First probability vector.
        q: Second probability vector.

    Returns:
        JS divergence as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> p = jnp.array([0.5, 0.5])
        >>> js_divergence(p, p)
        0.0
    """
    p_arr = jnp.asarray(p).ravel()
    q_arr = jnp.asarray(q).ravel()
    m = 0.5 * (p_arr + q_arr)
    return 0.5 * kl_divergence(p_arr, m) + 0.5 * kl_divergence(q_arr, m)


def total_variation(p: Any, q: Any) -> Any:
    """Total variation distance: ``0.5 * sum(|p - q|)``.

    Both an f-divergence and an integral probability metric.
    Bounded [0, 1] for probability vectors.

    Note:
        Direction: LOWER (0.0 = identical).
        Range: [0, 1].
        True metric. Symmetric.

    Args:
        p: First probability vector.
        q: Second probability vector.

    Returns:
        Total variation distance as a scalar value.
    """
    p_arr = jnp.asarray(p).ravel()
    q_arr = jnp.asarray(q).ravel()
    return 0.5 * jnp.sum(jnp.abs(p_arr - q_arr))


def hellinger_distance(p: Any, q: Any) -> Any:
    """Hellinger distance between probability distributions.

    ``H(p,q) = sqrt(0.5 * sum((sqrt(p) - sqrt(q))^2))``.
    Related to TV by Pinsker's inequality.

    Note:
        Direction: LOWER (0.0 = identical).
        Range: [0, 1].
        True metric. Symmetric.

    Args:
        p: First probability vector.
        q: Second probability vector.

    Returns:
        Hellinger distance as a scalar value.
    """
    p_arr = jnp.asarray(p).ravel()
    q_arr = jnp.asarray(q).ravel()
    diff = jnp.sqrt(jnp.maximum(p_arr, 0.0)) - jnp.sqrt(jnp.maximum(q_arr, 0.0))
    return jnp.sqrt(0.5 * jnp.sum(diff**2))


def chi_squared_divergence(p: Any, q: Any) -> Any:
    """Pearson chi-squared divergence: ``sum((p - q)^2 / q)``.

    NOT symmetric. Sensitive to q near zero.

    Note:
        Direction: LOWER (0.0 = identical).
        Range: [0, inf).
        Not symmetric. Uses safe_divide for numerical stability.

    Args:
        p: Observed distribution.
        q: Expected distribution.

    Returns:
        Chi-squared divergence as a scalar value.
    """
    p_arr = jnp.asarray(p).ravel()
    q_arr = jnp.asarray(q).ravel()
    return jnp.sum(safe_divide((p_arr - q_arr) ** 2, q_arr))


def renyi_divergence(p: Any, q: Any, *, alpha: float = 0.5) -> Any:
    """Renyi alpha-divergence.

    ``D_alpha(p||q) = 1/(alpha-1) * log(sum(p^alpha * q^(1-alpha)))``.
    Generalizes KL (alpha -> 1).

    Note:
        Direction: LOWER (0.0 = identical).
        Range: [0, inf).
        Not symmetric.

    Args:
        p: First probability vector.
        q: Second probability vector.
        alpha: Order parameter. Must not equal 1.0 (use KL instead).

    Returns:
        Renyi divergence as a scalar value.

    Raises:
        ValueError: If alpha equals 1.0.
    """
    if abs(alpha - 1.0) < 1e-10:
        msg = "alpha=1.0 is undefined for Renyi divergence; use kl_divergence instead"
        raise ValueError(msg)
    p_arr = jnp.asarray(p).ravel()
    q_arr = jnp.asarray(q).ravel()
    # Avoid 0^alpha issues
    p_safe = jnp.maximum(p_arr, _EPSILON)
    q_safe = jnp.maximum(q_arr, _EPSILON)
    integrand = jnp.sum(p_safe**alpha * q_safe ** (1.0 - alpha))
    return (1.0 / (alpha - 1.0)) * safe_log(integrand)


def f_divergence(
    p: Any,
    q: Any,
    *,
    generator: Callable[[Any], Any],
) -> Any:
    """Unified f-divergence with arbitrary convex generator.

    ``D_f(p||q) = sum(q * f(p / q))`` where f is convex with f(1) = 0.

    Note:
        Direction: LOWER (0.0 = identical if f(1)=0).
        Range: [0, inf).
        Recovers KL (f(u)=u*log(u)), TV (f(u)=0.5*|u-1|),
        Hellinger, chi-squared as special cases.

    Args:
        p: First probability vector.
        q: Second probability vector.
        generator: Convex function f with f(1) = 0.

    Returns:
        f-divergence as a scalar value.
    """
    p_arr = jnp.asarray(p).ravel()
    q_arr = jnp.asarray(q).ravel()
    ratio = safe_divide(p_arr, q_arr)
    return jnp.sum(q_arr * generator(ratio))


def wasserstein_1d(p: Any, q: Any) -> Any:
    """1D Wasserstein-1 (Earth Mover's) distance between samples.

    For 1D data: sort both, take mean absolute difference.
    Operates on sample arrays, not probability vectors.

    Note:
        Direction: LOWER (0.0 = identical distributions).
        Range: [0, inf).
        True metric. Symmetric.

    Args:
        p: First sample array.
        q: Second sample array.

    Returns:
        Wasserstein-1 distance as a scalar value.
    """
    p_sorted = jnp.sort(jnp.asarray(p).ravel())
    q_sorted = jnp.sort(jnp.asarray(q).ravel())
    return jnp.mean(jnp.abs(p_sorted - q_sorted))


def _rbf_kernel(x: Any, y: Any, bandwidth: float) -> Any:
    """RBF (Gaussian) kernel matrix."""
    sq_dist = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return jnp.exp(-sq_dist / (2.0 * bandwidth**2))


def _laplace_kernel(x: Any, y: Any, bandwidth: float) -> Any:
    """Laplace kernel matrix."""
    dist = jnp.sum(jnp.abs(x[:, None, :] - y[None, :, :]), axis=-1)
    return jnp.exp(-dist / bandwidth)


def mmd(
    x: Any,
    y: Any,
    *,
    kernel: str = "rbf",
    bandwidth: float = 1.0,
) -> Any:
    """Maximum Mean Discrepancy between sample distributions.

    Measures distance using kernel mean embeddings. O(n^{-1/2})
    estimation rate regardless of dimension.

    Note:
        Direction: LOWER (0.0 = identical distributions).
        Range: [0, inf).
        True metric. Symmetric.

    Args:
        x: First sample matrix (n_samples, n_features).
        y: Second sample matrix (n_samples, n_features).
        kernel: Kernel type: ``"rbf"`` or ``"laplace"``.
        bandwidth: Kernel bandwidth parameter.

    Returns:
        MMD as a scalar value.
    """
    x_arr = jnp.asarray(x)
    y_arr = jnp.asarray(y)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    if y_arr.ndim == 1:
        y_arr = y_arr[:, None]

    kernel_fn = _rbf_kernel if kernel == "rbf" else _laplace_kernel

    kxx = kernel_fn(x_arr, x_arr, bandwidth)
    kyy = kernel_fn(y_arr, y_arr, bandwidth)
    kxy = kernel_fn(x_arr, y_arr, bandwidth)

    # Unbiased estimator: exclude diagonal
    n = x_arr.shape[0]
    m = y_arr.shape[0]
    kxx_sum = (jnp.sum(kxx) - jnp.trace(kxx)) / (n * (n - 1) + _EPSILON)
    kyy_sum = (jnp.sum(kyy) - jnp.trace(kyy)) / (m * (m - 1) + _EPSILON)
    kxy_sum = jnp.sum(kxy) / (n * m)

    mmd_sq = kxx_sum + kyy_sum - 2.0 * kxy_sum
    return jnp.sqrt(jnp.maximum(mmd_sq, 0.0))


def sinkhorn_divergence(
    x: Any,
    y: Any,
    *,
    regularization: float = 0.1,
    max_iter: int = 100,
    threshold: float = 1e-5,
) -> Any:
    """Debiased Sinkhorn divergence (entropic optimal transport).

    ``S(x,y) = OT_reg(x,y) - 0.5*(OT_reg(x,x) + OT_reg(y,y))``.
    Differentiable and JIT-compatible.

    Note:
        Direction: LOWER (0.0 = identical distributions).
        Range: [0, inf).
        Symmetric. Differentiable.
        Debiased: ``S(x,x) = 0``.

    Args:
        x: First sample matrix (n_samples, n_features).
        y: Second sample matrix (n_samples, n_features).
        regularization: Entropic regularization strength.
        max_iter: Maximum Sinkhorn iterations.
        threshold: Convergence threshold.

    Returns:
        Sinkhorn divergence as a scalar value.
    """
    x_arr = jnp.asarray(x)
    y_arr = jnp.asarray(y)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    if y_arr.ndim == 1:
        y_arr = y_arr[:, None]

    def _sinkhorn_cost(a: Any, b: Any) -> Any:
        # Cost matrix (squared Euclidean)
        cost = jnp.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        n_a = a.shape[0]
        n_b = b.shape[0]
        # Uniform marginals
        mu = jnp.ones(n_a) / n_a
        nu = jnp.ones(n_b) / n_b
        # Gibbs kernel
        k = jnp.exp(-cost / regularization)

        # Sinkhorn iterations via lax.while_loop for JIT compatibility
        def _not_converged(state: tuple[Any, Any, Any, Any]) -> Any:
            _, _, converged, step = state
            return (~converged) & (step < max_iter)

        def _step(state: tuple[Any, Any, Any, Any]) -> tuple[Any, Any, Any, Any]:
            u, v, _, step = state
            u_new = mu / (k @ v + _EPSILON)
            v_new = nu / (k.T @ u_new + _EPSILON)
            converged = jnp.max(jnp.abs(u_new - u)) < threshold
            return u_new, v_new, converged, step + 1

        init_state = (jnp.ones(n_a), jnp.ones(n_b), jnp.bool_(False), jnp.int32(0))
        u, v, _, _ = jax.lax.while_loop(_not_converged, _step, init_state)
        # Transport cost: sum_{ij} u_i K_{ij} v_j C_{ij}
        transport = jnp.sum(u[:, None] * k * v[None, :] * cost)
        return transport

    ot_xy = _sinkhorn_cost(x_arr, y_arr)
    ot_xx = _sinkhorn_cost(x_arr, x_arr)
    ot_yy = _sinkhorn_cost(y_arr, y_arr)
    # Floor at 0 — small negative values from numerical precision
    return jnp.maximum(ot_xy - 0.5 * (ot_xx + ot_yy), 0.0)


def sliced_wasserstein(
    x: Any,
    y: Any,
    *,
    num_projections: int = 50,
    p: float = 2.0,
    key: Any | None = None,
) -> Any:
    """Sliced Wasserstein distance.

    Project onto random 1D directions, compute exact 1D Wasserstein,
    average. Practical for high-dimensional distribution comparison.

    Note:
        Direction: LOWER (0.0 = identical distributions).
        Range: [0, inf).
        True metric. Symmetric.

    Args:
        x: First sample matrix (n_samples, n_features).
        y: Second sample matrix (n_samples, n_features).
        num_projections: Number of random 1D projections.
        p: Order of Wasserstein distance.
        key: JAX PRNG key for reproducibility. Uses fixed seed if None.

    Returns:
        Sliced Wasserstein distance as a scalar value.
    """
    x_arr = jnp.asarray(x)
    y_arr = jnp.asarray(y)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    if y_arr.ndim == 1:
        y_arr = y_arr[:, None]

    d = x_arr.shape[1]
    if key is None:
        key = jax.random.PRNGKey(42)

    # Random projections on unit sphere
    directions = jax.random.normal(key, (num_projections, d))
    directions = directions / (jnp.linalg.norm(directions, axis=1, keepdims=True) + _EPSILON)

    # Batched projection: (n_samples, d) @ (d, num_projections) -> (n_samples, num_projections)
    proj_x_all = x_arr @ directions.T
    proj_y_all = y_arr @ directions.T

    # Sort each projection independently
    proj_x_sorted = jnp.sort(proj_x_all, axis=0)
    proj_y_sorted = jnp.sort(proj_y_all, axis=0)

    # Wasserstein per projection
    per_proj = jnp.mean(jnp.abs(proj_x_sorted - proj_y_sorted) ** p, axis=0) ** (1.0 / p)
    return jnp.mean(per_proj)


def bregman_divergence(
    x: Any,
    y: Any,
    *,
    generator: Callable[[Any], Any],
    generator_grad: Callable[[Any], Any] | None = None,
) -> Any:
    """Bregman divergence with arbitrary convex generator.

    ``D_psi(x, y) = psi(x) - psi(y) - <grad_psi(y), x - y>``.
    Unifies squared Euclidean, KL, Itakura-Saito, and Mahalanobis.

    Note:
        Direction: LOWER (0.0 = identical points).
        Range: [0, inf).
        Not symmetric in general.

    Args:
        x: First point or batch.
        y: Second point or batch.
        generator: Strictly convex differentiable function psi.
        generator_grad: Gradient of generator. If None, computed
            via ``jax.grad(generator)``.

    Returns:
        Bregman divergence as a scalar value.
    """
    x_arr = jnp.asarray(x).ravel()
    y_arr = jnp.asarray(y).ravel()

    if generator_grad is None:
        generator_grad = jax.grad(generator)

    psi_x = generator(x_arr)
    psi_y = generator(y_arr)
    grad_y = generator_grad(y_arr)
    diff = x_arr - y_arr

    result = psi_x - psi_y - jnp.dot(grad_y, diff)
    return result
