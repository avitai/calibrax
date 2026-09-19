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

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike
from substrax.rng import key_from

from calibrax.metrics._utils import _EPSILON, safe_divide, safe_log, safe_root


# Renyi divergence is undefined at alpha = 1; treat values this close as 1.
_ALPHA_ONE_TOLERANCE = 1e-10


def kl_divergence(p: ArrayLike, q: ArrayLike) -> jax.Array:
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


def reverse_kl_divergence(p: ArrayLike, q: ArrayLike) -> jax.Array:
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


def js_divergence(p: ArrayLike, q: ArrayLike) -> jax.Array:
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


def total_variation(p: ArrayLike, q: ArrayLike) -> jax.Array:
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


def hellinger_distance(p: ArrayLike, q: ArrayLike) -> jax.Array:
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
    diff = safe_root(jnp.maximum(p_arr, 0.0)) - safe_root(jnp.maximum(q_arr, 0.0))
    return safe_root(0.5 * jnp.sum(diff**2))


def chi_squared_divergence(p: ArrayLike, q: ArrayLike) -> jax.Array:
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


def renyi_divergence(p: ArrayLike, q: ArrayLike, *, alpha: float = 0.5) -> jax.Array:
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
    if abs(alpha - 1.0) < _ALPHA_ONE_TOLERANCE:
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
    p: ArrayLike,
    q: ArrayLike,
    *,
    generator: Callable[[jax.Array], jax.Array],
) -> jax.Array:
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


def wasserstein_1d(p: ArrayLike, q: ArrayLike) -> jax.Array:
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


def _rbf_kernel(x: jax.Array, y: jax.Array, bandwidth: float) -> jax.Array:
    """RBF (Gaussian) kernel matrix."""
    sq_dist = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return jnp.exp(-sq_dist / (2.0 * bandwidth**2))


def _laplace_kernel(x: jax.Array, y: jax.Array, bandwidth: float) -> jax.Array:
    """Laplace kernel matrix."""
    dist = jnp.sum(jnp.abs(x[:, None, :] - y[None, :, :]), axis=-1)
    return jnp.exp(-dist / bandwidth)


def mmd(
    x: ArrayLike,
    y: ArrayLike,
    *,
    kernel: str = "rbf",
    bandwidth: float = 1.0,
) -> jax.Array:
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
    return safe_root(jnp.maximum(mmd_sq, 0.0))


def _entropic_ot(
    a: jax.Array, b: jax.Array, *, epsilon: float, max_iter: int, threshold: float
) -> jax.Array:
    """Entropic optimal transport ``OT_eps`` between uniform point clouds, squared Euclidean cost.

    Log-domain Sinkhorn (the potentials ``f``, ``g`` updated through ``logsumexp``, which the
    kernel ``exp(-C / eps)`` underflows without) until the row marginal's L1 error is below
    ``threshold`` or ``max_iter`` updates have run. The potentials are solved without
    gradient and the dual objective is evaluated at them: its gradient with respect to the
    cost is the transport plan (the envelope theorem), so ``jax.grad`` needs no pass through
    the iterations (Feydy et al. 2019, the GeomLoss construction).
    """
    cost = jnp.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    log_mu = jnp.full(a.shape[0], -jnp.log(a.shape[0]))
    log_nu = jnp.full(b.shape[0], -jnp.log(b.shape[0]))
    frozen = jax.lax.stop_gradient(cost)

    def update(potentials: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        f, g = potentials
        f = -epsilon * jax.nn.logsumexp(log_nu[None, :] + (g[None, :] - frozen) / epsilon, axis=1)
        g = -epsilon * jax.nn.logsumexp(log_mu[:, None] + (f[:, None] - frozen) / epsilon, axis=0)
        return f, g

    def marginal_error(f: jax.Array, g: jax.Array) -> jax.Array:
        log_plan = log_mu[:, None] + log_nu[None, :] + (f[:, None] + g[None, :] - frozen) / epsilon
        return jnp.sum(jnp.abs(jnp.exp(jax.nn.logsumexp(log_plan, axis=1)) - jnp.exp(log_mu)))

    def keep_going(state: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        f, g, step = state
        return (step < max_iter) & (marginal_error(f, g) > threshold)

    def step(
        state: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        f, g, count = state
        f, g = update((f, g))
        return f, g, count + 1

    zeros_a, zeros_b = jnp.zeros(a.shape[0], cost.dtype), jnp.zeros(b.shape[0], cost.dtype)
    f, g, _ = jax.lax.while_loop(keep_going, step, (*update((zeros_a, zeros_b)), jnp.int32(1)))
    f, g = jax.lax.stop_gradient(f), jax.lax.stop_gradient(g)
    log_plan = log_mu[:, None] + log_nu[None, :] + (f[:, None] + g[None, :] - cost) / epsilon
    # The dual objective: <f, mu> + <g, nu> - eps * (plan mass - 1).
    return (
        jnp.dot(f, jnp.exp(log_mu))
        + jnp.dot(g, jnp.exp(log_nu))
        - epsilon * (jnp.sum(jnp.exp(log_plan)) - 1.0)
    )


def sinkhorn_divergence(
    x: ArrayLike,
    y: ArrayLike,
    *,
    regularization: float = 0.1,
    max_iter: int = 1000,
    threshold: float = 1e-4,
) -> jax.Array:
    """Debiased Sinkhorn divergence between two point clouds (Feydy et al. 2019).

    ``S_eps(x, y) = OT_eps(x, y) - (OT_eps(x, x) + OT_eps(y, y)) / 2`` with ``OT_eps`` the
    entropic optimal transport objective under squared Euclidean cost and uniform weights
    (Genevay et al. 2018; Feydy et al. 2019, whose Theorem 1 makes it positive, zero only for
    equal clouds, and convex). Sinkhorn runs in the log domain until the marginal error is
    below ``threshold``; the gradient comes from the dual potentials, so ``jax.grad`` and
    ``jax.jit`` both hold and the memory does not grow with the iterations.

    Note:
        Direction: LOWER (0.0 = identical distributions).
        Range: [0, inf).
        Symmetric. Differentiable.
        Debiased: ``S(x,x) = 0``.

    Args:
        x: First sample matrix (n_samples, n_features).
        y: Second sample matrix (n_samples, n_features).
        regularization: The entropic regularization ``eps``, in the cost's units.
        max_iter: Most Sinkhorn updates per transport problem.
        threshold: L1 error of the row marginal at which Sinkhorn stops.

    Returns:
        Sinkhorn divergence as a scalar value.
    """
    x_arr = jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float32))
    y_arr = jnp.atleast_1d(jnp.asarray(y, dtype=jnp.float32))
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    if y_arr.ndim == 1:
        y_arr = y_arr[:, None]
    solve = functools.partial(
        _entropic_ot, epsilon=regularization, max_iter=max_iter, threshold=threshold
    )
    return solve(x_arr, y_arr) - 0.5 * (solve(x_arr, x_arr) + solve(y_arr, y_arr))


# The number of directions: the Monte Carlo error of the average over directions falls as
# L^(-1/2), about 3 % relative at 256 on a 10-dimensional Gaussian pair.
SLICED_WASSERSTEIN_PROJECTIONS = 256


def sliced_wasserstein(
    x: ArrayLike,
    y: ArrayLike,
    *,
    key: jax.Array | nnx.Rngs,
    num_projections: int = SLICED_WASSERSTEIN_PROJECTIONS,
    p: float = 2.0,
) -> jax.Array:
    """Sliced Wasserstein distance ``SW_p = (E_theta[W_p^p(theta_# x, theta_# y)])^(1/p)``.

    Both sample sets are projected onto ``num_projections`` directions drawn uniformly on the
    unit sphere; along each direction the exact one-dimensional ``W_p^p`` is the mean of the
    ``p``-th powers of the sorted differences, and the average over directions is taken before
    the ``1/p`` root (Bonneel et al. 2015; Nadjahi et al. 2020, eq. 5; POT's
    ``sliced_wasserstein_distance``).

    Note:
        Direction: LOWER (0.0 = identical distributions).
        Range: [0, inf).
        Symmetric. With a fixed set of directions the value is a pseudometric: two
        distributions that agree along every drawn direction are at distance 0.

    Args:
        x: First sample matrix (n_samples, n_features), or a vector of scalar samples.
        y: Second sample matrix with the same number of samples and features.
        key: The key the directions are drawn from, or an ``nnx.Rngs`` whose ``sample`` or
            ``default`` stream supplies it. There is no default: a fixed direction set would
            bias every estimate the same way.
        num_projections: Number of random directions.
        p: Order of the Wasserstein distance.

    Returns:
        The sliced Wasserstein distance as a scalar.

    Raises:
        ValueError: If ``x`` and ``y`` hold different numbers of samples.
    """
    directions_key = key_from(key, streams=("sample", "default"), context="sliced_wasserstein")
    x_arr = jnp.asarray(x)
    y_arr = jnp.asarray(y)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    if y_arr.ndim == 1:
        y_arr = y_arr[:, None]
    if x_arr.shape[0] != y_arr.shape[0]:
        msg = (
            "sliced_wasserstein needs the same number of samples in x and y, "
            f"got {x_arr.shape[0]} and {y_arr.shape[0]}"
        )
        raise ValueError(msg)

    directions = jax.random.normal(directions_key, (num_projections, x_arr.shape[1]))
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)

    # (n_samples, d) @ (d, num_projections): every projection at once, each column sorted.
    proj_x = jnp.sort(x_arr @ directions.T, axis=0)
    proj_y = jnp.sort(y_arr @ directions.T, axis=0)
    return safe_root(jnp.mean(jnp.abs(proj_x - proj_y) ** p), order=p)


# The registry's fixed projection set: the registry calls metrics as ``fn(predictions,
# targets)``, and a fixed set keeps values from different suite runs comparable.
SLICED_WASSERSTEIN_REGISTRY_SEED = 0


def registry_sliced_wasserstein(x: ArrayLike, y: ArrayLike) -> jax.Array:
    """``sliced_wasserstein`` over the registry's fixed projection set.

    The directions come from ``SLICED_WASSERSTEIN_REGISTRY_SEED``, the same in every call, so a
    suite compares like with like; over a fixed set the value is a pseudometric.

    Args:
        x: First sample matrix (n_samples, n_features).
        y: Second sample matrix with the same shape.

    Returns:
        The sliced Wasserstein distance as a scalar.
    """
    return sliced_wasserstein(x, y, key=jax.random.key(SLICED_WASSERSTEIN_REGISTRY_SEED))


def bregman_divergence(
    x: ArrayLike,
    y: ArrayLike,
    *,
    generator: Callable[[jax.Array], jax.Array],
    generator_grad: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
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

    return psi_x - psi_y - jnp.dot(grad_y, diff)


def kolmogorov_smirnov_distance(a: ArrayLike, b: ArrayLike) -> jax.Array:
    """Kolmogorov-Smirnov distance between two samples.

    The largest absolute gap between the two empirical CDFs, evaluated at every
    observed value. Both inputs are flattened; duplicates do not change the
    maximum gap, which keeps the function free of ``jnp.unique`` and traceable.

    Note:
        Direction: LOWER (0.0 = identical empirical distributions).
        Range: [0, 1].

    Args:
        a: Samples from the first distribution, any shape.
        b: Samples from the second distribution, any shape.

    Returns:
        Scalar KS distance as a JAX array.
    """
    first = jnp.sort(jnp.asarray(a).reshape(-1))
    second = jnp.sort(jnp.asarray(b).reshape(-1))
    grid = jnp.sort(jnp.concatenate([first, second]))
    first_cdf = jnp.searchsorted(first, grid, side="right") / first.shape[0]
    second_cdf = jnp.searchsorted(second, grid, side="right") / second.shape[0]
    return jnp.max(jnp.abs(first_cdf - second_cdf))
