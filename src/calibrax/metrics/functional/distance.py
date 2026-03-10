"""Distance functions across geometric spaces.

Pure functions for computing distances between vectors or distributions.
Covers the three constant-curvature geometries: Euclidean (zero curvature),
cosine/spherical (positive curvature), and Poincare/Lorentz/hyperbolic
(negative curvature). Also includes discrete set distances (Hamming, Jaccard)
and the Randers asymmetric Finsler distance for directed relationships.

Includes 11 functions: cosine_distance, euclidean_distance, manhattan_distance,
chebyshev_distance, mahalanobis_distance, hamming_distance, minkowski_distance,
jaccard_distance, poincare_distance, lorentz_distance, randers_distance.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON, safe_divide


def _batch_or_single(
    a: Any,
    b: Any,
    fn: Any,
    **kwargs: Any,
) -> Any:
    """Apply a distance function to 1D or 2D inputs.

    For 1D: compute single distance.
    For 2D: compute per-row distance, return mean.

    Args:
        a: First array (1D or 2D).
        b: Second array (1D or 2D).
        fn: Distance function accepting two 1D arrays.
        **kwargs: Additional keyword arguments for fn.

    Returns:
        Distance as a scalar value.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    if a.ndim == 1:
        return fn(a, b, **kwargs)
    # 2D: vmap over axis 0, then mean
    batched_fn = jax.vmap(lambda x, y: fn(x, y, **kwargs))
    return jnp.mean(batched_fn(a, b))


def cosine_distance(a: Any, b: Any) -> Any:
    """Cosine distance: ``1 - cosine_similarity(a, b)``.

    Measures angular separation between vectors. Lives on the
    unit sphere (positive curvature geometry).

    Note:
        Direction: LOWER (0.0 = identical direction).
        Range: [0, 2]. 0 = same direction, 1 = orthogonal, 2 = opposite.
        Not a true metric (violates triangle inequality), but
        ``arccos(1 - d)`` is a true metric on the sphere.
        Invariant under scaling.

    Args:
        a: First vector or batch of vectors.
        b: Second vector or batch of vectors.

    Returns:
        Cosine distance as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> cosine_distance(jnp.array([1.0, 0.0]), jnp.array([1.0, 0.0]))
        0.0
    """

    def _single(x: Any, y: Any) -> Any:
        similarity = jnp.dot(x, y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + _EPSILON)
        return 1.0 - similarity

    return _batch_or_single(a, b, _single)


def euclidean_distance(a: Any, b: Any) -> Any:
    """Euclidean (L2) distance.

    Standard distance in flat (zero curvature) Euclidean space.

    Note:
        Direction: LOWER (0.0 = identical).
        Range: [0, inf).
        True metric: satisfies identity, symmetry, triangle inequality.
        Invariant under rotation and translation.

    Args:
        a: First vector or batch of vectors.
        b: Second vector or batch of vectors.

    Returns:
        Euclidean distance as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> euclidean_distance(jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
        1.4142...
    """

    def _single(x: Any, y: Any) -> Any:
        return jnp.linalg.norm(x - y)

    return _batch_or_single(a, b, _single)


def manhattan_distance(a: Any, b: Any) -> Any:
    """Manhattan (L1) distance.

    Sum of absolute differences. Also called taxicab or city-block distance.

    Note:
        Direction: LOWER (0.0 = identical).
        Range: [0, inf).
        True metric. Invariant under translation.

    Args:
        a: First vector or batch of vectors.
        b: Second vector or batch of vectors.

    Returns:
        Manhattan distance as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> manhattan_distance(jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
        2.0
    """

    def _single(x: Any, y: Any) -> Any:
        return jnp.sum(jnp.abs(x - y))

    return _batch_or_single(a, b, _single)


def chebyshev_distance(a: Any, b: Any) -> Any:
    """Chebyshev (L-infinity) distance.

    Maximum absolute difference across dimensions.

    Note:
        Direction: LOWER (0.0 = identical).
        Range: [0, inf).
        True metric. Invariant under translation.

    Args:
        a: First vector or batch of vectors.
        b: Second vector or batch of vectors.

    Returns:
        Chebyshev distance as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> chebyshev_distance(jnp.array([1.0, 0.0]), jnp.array([0.0, 3.0]))
        3.0
    """

    def _single(x: Any, y: Any) -> Any:
        return jnp.max(jnp.abs(x - y))

    return _batch_or_single(a, b, _single)


def mahalanobis_distance(
    a: Any,
    b: Any,
    *,
    precision_matrix: Any | None = None,
) -> Any:
    """Mahalanobis distance.

    Generalized Euclidean distance weighted by an inverse covariance
    (precision) matrix: ``sqrt((a-b)^T S^-1 (a-b))``.

    Note:
        Direction: LOWER (0.0 = identical).
        Range: [0, inf).
        True metric when precision_matrix is positive definite.
        Reduces to Euclidean when precision_matrix is identity.

    Args:
        a: First vector or batch of vectors.
        b: Second vector or batch of vectors.
        precision_matrix: Inverse covariance matrix. If None, uses identity
            (equivalent to Euclidean distance).

    Returns:
        Mahalanobis distance as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> a = jnp.array([1.0, 0.0])
        >>> b = jnp.array([0.0, 1.0])
        >>> mahalanobis_distance(a, b)  # identity → Euclidean
        1.4142...
    """

    def _single(x: Any, y: Any, *, prec: Any | None = None) -> Any:
        diff = x - y
        if prec is None:
            return jnp.linalg.norm(diff)
        return jnp.sqrt(jnp.dot(diff, jnp.dot(prec, diff)))

    a_arr = jnp.asarray(a)
    b_arr = jnp.asarray(b)
    if a_arr.ndim == 1:
        return _single(a_arr, b_arr, prec=precision_matrix)
    batched = jax.vmap(lambda x, y: _single(x, y, prec=precision_matrix))
    return jnp.mean(batched(a_arr, b_arr))


def hamming_distance(a: Any, b: Any) -> Any:
    """Hamming distance: fraction of differing positions.

    Discrete metric on integer or boolean arrays.

    Note:
        Direction: LOWER (0.0 = identical).
        Range: [0, 1].
        True metric on discrete space.

    Args:
        a: First integer/boolean array.
        b: Second integer/boolean array.

    Returns:
        Hamming distance as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> hamming_distance(jnp.array([1, 0, 1]), jnp.array([1, 1, 1]))
        0.333...
    """
    a_arr = jnp.asarray(a).ravel()
    b_arr = jnp.asarray(b).ravel()
    return jnp.mean(a_arr != b_arr)


def minkowski_distance(a: Any, b: Any, *, p: float = 2.0) -> Any:
    """Minkowski (Lp) distance.

    Generalized distance: ``(sum|a_i - b_i|^p)^(1/p)``.
    p=1 is Manhattan, p=2 is Euclidean.

    Note:
        Direction: LOWER (0.0 = identical).
        Range: [0, inf).
        True metric for p >= 1.

    Args:
        a: First vector or batch of vectors.
        b: Second vector or batch of vectors.
        p: Order of the norm. Must be >= 1.

    Returns:
        Minkowski distance as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> minkowski_distance(jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]), p=1.0)
        2.0
    """

    def _single(x: Any, y: Any, *, p: float = 2.0) -> Any:
        return jnp.sum(jnp.abs(x - y) ** p) ** (1.0 / p)

    return _batch_or_single(a, b, _single, p=p)


def jaccard_distance(a: Any, b: Any) -> Any:
    """Jaccard distance: ``1 - |A intersection B| / |A union B|``.

    Set-based distance for binary vectors. Complement of the
    Jaccard similarity index.

    Note:
        Direction: LOWER (0.0 = identical sets).
        Range: [0, 1].
        True metric on the space of finite sets.
        Symmetric: ``J(A,B) = J(B,A)``.

    Args:
        a: First binary/non-negative vector.
        b: Second binary/non-negative vector.

    Returns:
        Jaccard distance as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> jaccard_distance(jnp.array([1, 1, 0]), jnp.array([1, 0, 1]))
        0.666...
    """
    a_arr = jnp.asarray(a).ravel().astype(jnp.float32)
    b_arr = jnp.asarray(b).ravel().astype(jnp.float32)
    intersection = jnp.sum(jnp.minimum(a_arr, b_arr))
    union = jnp.sum(jnp.maximum(a_arr, b_arr))
    return 1.0 - safe_divide(intersection, union)


def poincare_distance(
    a: Any,
    b: Any,
    *,
    curvature: float = 1.0,
) -> Any:
    """Geodesic distance in the Poincare ball model of hyperbolic space.

    Points must lie inside the ball: ``curvature * ||x||^2 < 1``.
    Hyperbolic space has exponential volume growth matching tree/hierarchy
    structures.

    Note:
        Direction: LOWER (0.0 = identical points).
        Range: [0, inf).
        True metric. Symmetric. Differentiable.
        Distance grows without bound as points approach the ball boundary.
        Complements cosine_distance (spherical/positive curvature)
        and euclidean_distance (flat/zero curvature).

    Args:
        a: First point or batch inside the Poincare ball.
        b: Second point or batch inside the Poincare ball.
        curvature: Curvature parameter (default 1.0). Higher values
            shrink the ball radius.

    Returns:
        Poincare distance as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> poincare_distance(jnp.array([0.0, 0.0]), jnp.array([0.5, 0.0]))
        1.0986...
    """

    def _single(x: Any, y: Any, *, curvature: float = 1.0) -> Any:
        c = curvature
        diff_sq = jnp.sum((x - y) ** 2)
        norm_x_sq = jnp.sum(x**2)
        norm_y_sq = jnp.sum(y**2)
        num = 2.0 * c * diff_sq
        denom = (1.0 - c * norm_x_sq) * (1.0 - c * norm_y_sq)
        arg = 1.0 + safe_divide(num, denom)
        # Clamp for numerical stability
        arg = jnp.maximum(arg, 1.0 + _EPSILON)
        return (1.0 / jnp.sqrt(c)) * jnp.arccosh(arg)

    return _batch_or_single(a, b, _single, curvature=curvature)


def lorentz_distance(a: Any, b: Any) -> Any:
    """Geodesic distance on the Lorentz (hyperboloid) model of hyperbolic space.

    Points live on the upper hyperboloid ``<x,x>_L = -1, x_0 > 0`` in R^(n+1).
    Uses the Minkowski inner product: ``<a,b>_L = -a_0*b_0 + sum(a_i*b_i)``.
    Isometric to the Poincare ball but numerically more stable near
    the boundary.

    Note:
        Direction: LOWER (0.0 = identical points).
        Range: [0, inf).
        True metric. Symmetric. Differentiable.
        Invariant under Lorentz transformations.
        More numerically stable than Poincare for optimization.

    Args:
        a: First point on the hyperboloid (first component is time-like).
        b: Second point on the hyperboloid.

    Returns:
        Lorentz distance as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> # Point on hyperboloid: x_0 = sqrt(1 + ||x||^2)
        >>> a = jnp.array([1.0, 0.0, 0.0])  # origin on hyperboloid
        >>> lorentz_distance(a, a)
        0.0
    """

    def _single(x: Any, y: Any) -> Any:
        # Minkowski inner product: -x_0*y_0 + sum(x_i*y_i)
        inner = -x[0] * y[0] + jnp.sum(x[1:] * y[1:])
        # Clamp for numerical stability (arccosh domain is [1, inf))
        clamped = jnp.maximum(-inner, 1.0 + _EPSILON)
        # Numerically stable arccosh: log(x + sqrt(x^2 - 1))
        return jnp.log(clamped + jnp.sqrt(jnp.maximum(clamped**2 - 1.0, _EPSILON)))

    return _batch_or_single(a, b, _single)


def randers_distance(a: Any, b: Any, *, drift: Any) -> Any:
    """Randers distance: asymmetric Finsler metric with directional bias.

    Adds a "wind" vector to Euclidean distance:
    ``d_R(a,b) = ||b-a||_2 + <drift, b-a>``.
    The drift vector must satisfy ``||drift|| < 1`` (sub-sonic condition).

    Note:
        Direction: LOWER (0.0 = identical points).
        Range: [0, inf).
        NOT a true metric (asymmetric: ``d(a,b) != d(b,a)``).
        The asymmetry is a geometric feature, not a deficiency.
        Applications: directed graph embeddings, latent space geometry.

    Args:
        a: First point or batch of points.
        b: Second point or batch of points.
        drift: Wind/bias vector. Must satisfy ``||drift|| < 1``.

    Returns:
        Randers distance as a scalar value.

    Raises:
        ValueError: If ``||drift|| >= 1`` (sub-sonic condition violated).

    Examples:
        >>> import jax.numpy as jnp
        >>> a = jnp.array([0.0, 0.0])
        >>> b = jnp.array([1.0, 0.0])
        >>> randers_distance(a, b, drift=jnp.array([0.0, 0.0]))
        1.0
    """
    drift_arr = jnp.asarray(drift)
    drift_norm = float(jnp.linalg.norm(drift_arr))
    if drift_norm >= 1.0:
        msg = f"Sub-sonic condition violated: ||drift|| = {drift_norm:.4f} >= 1.0"
        raise ValueError(msg)

    def _single(x: Any, y: Any, *, drift: Any) -> Any:
        diff = y - x
        return jnp.linalg.norm(diff) + jnp.dot(drift, diff)

    return _batch_or_single(a, b, _single, drift=drift_arr)
