"""Manifold distance metrics -- Riemannian and pseudo-Riemannian geometry.

Geometrically correct distances for structured mathematical objects that
live on curved manifolds rather than flat Euclidean space. Using naive
Euclidean distance on these objects produces geometrically meaningless
results.

**Manifold hierarchy** (each level generalizes the previous):
- Euclidean: flat space, standard L2 norm
- Riemannian: curved space with positive-definite metric tensor
- Pseudo-Riemannian: metric tensor may be indefinite (mixed signature)

**Manifolds covered:**
- **SPD(n)**: Symmetric Positive Definite matrices (covariance matrices,
  diffusion tensors). Two distances: affine-invariant (geometrically exact)
  and log-Euclidean (faster approximation).
- **Gr(p,n)**: Grassmann manifold of p-dimensional subspaces (PCA subspace
  comparison, feature space analysis). Basis-independent.
- **St(p,n)**: Stiefel manifold of orthonormal p-frames (orientation
  comparison, orthogonal constraints). Basis-dependent.
- **Pseudo-hyperboloid**: Mixed-curvature space for knowledge graph
  embeddings. Generalizes Lorentz/hyperbolic to arbitrary signature.

All implemented using JAX linear algebra. No external manifold libraries.
Registered with ``domain="manifold"``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON


def spd_affine_invariant_distance(a: Any, b: Any) -> Any:
    """Affine-invariant Riemannian distance between SPD matrices.

    Geodesic distance on the SPD manifold:
    ``d(A, B) = ||log(A^{-1/2} B A^{-1/2})||_F``

    Equivalent to ``sqrt(sum(log(lambda_i)^2))`` where lambda_i are the
    generalized eigenvalues of (B, A).

    Invariant under congruence: ``d(A, B) = d(MAM^T, MBM^T)`` for any
    invertible M. This is the natural distance for covariance matrices.

    Complexity: O(n^3) for generalized eigenvalue decomposition.

    Args:
        a: First SPD matrix, shape (n, n).
        b: Second SPD matrix, shape (n, n).

    Returns:
        Affine-invariant distance. Lower is better. 0.0 for identical matrices.

    Examples:
        >>> import jax.numpy as jnp
        >>> a = jnp.eye(3) * 2.0
        >>> spd_affine_invariant_distance(a, a)
        0.0
    """
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)

    # Compute A^{-1/2} via eigendecomposition: A = Q Lambda Q^T
    eig_vals, eig_vecs = jnp.linalg.eigh(a)
    eig_vals = jnp.maximum(eig_vals, _EPSILON)
    a_inv_sqrt = eig_vecs @ jnp.diag(1.0 / jnp.sqrt(eig_vals)) @ eig_vecs.T

    # M = A^{-1/2} B A^{-1/2} -- symmetric, has same eigenvalues as
    # the generalized problem B v = lambda A v
    m = a_inv_sqrt @ b @ a_inv_sqrt

    # Eigenvalues of M are the generalized eigenvalues
    eigenvalues = jnp.linalg.eigvalsh(m)
    eigenvalues = jnp.maximum(eigenvalues, _EPSILON)

    return jnp.sqrt(jnp.sum(jnp.log(eigenvalues) ** 2))


def spd_log_euclidean_distance(a: Any, b: Any) -> Any:
    """Log-Euclidean distance between SPD matrices.

    Maps SPD matrices to Euclidean space via matrix logarithm:
    ``d(A, B) = ||log(A) - log(B)||_F``

    Faster than affine-invariant but less geometrically faithful.
    Invariant under orthogonal transformations but NOT affine-invariant.

    Complexity: O(n^3) for eigendecomposition (no matrix inverse).

    Args:
        a: First SPD matrix, shape (n, n).
        b: Second SPD matrix, shape (n, n).

    Returns:
        Log-Euclidean distance. Lower is better. 0.0 for identical matrices.

    Examples:
        >>> import jax.numpy as jnp
        >>> a = jnp.eye(3) * 2.0
        >>> spd_log_euclidean_distance(a, a)
        0.0
    """
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)

    # Matrix log via eigendecomposition: A = Q Lambda Q^T -> log(A) = Q log(Lambda) Q^T
    log_a = _matrix_log_spd(a)
    log_b = _matrix_log_spd(b)

    return jnp.linalg.norm(log_a - log_b)


def _matrix_log_spd(m: jnp.ndarray) -> jnp.ndarray:
    """Matrix logarithm for SPD matrices via eigendecomposition.

    Args:
        m: SPD matrix, shape (n, n).

    Returns:
        Matrix logarithm, shape (n, n).
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(m)
    log_eigenvalues = jnp.log(jnp.maximum(eigenvalues, _EPSILON))
    return eigenvectors @ jnp.diag(log_eigenvalues) @ eigenvectors.T


def grassmann_distance(u: Any, v: Any) -> Any:
    """Geodesic distance on the Grassmann manifold Gr(p, n).

    Distance between two p-dimensional subspaces of R^n, based on
    principal angles. The principal angles theta_i = arccos(sigma_i)
    where sigma_i are singular values of U^T V.

    Distance = ``sqrt(sum(theta_i^2))``.

    Basis-independent: same subspace with different orthonormal basis
    gives distance 0.

    Complexity: O(n*p^2) for SVD of the p x p matrix U^T V.

    Args:
        u: First orthonormal matrix, shape (n, p).
        v: Second orthonormal matrix, shape (n, p).

    Returns:
        Grassmann distance. Lower is better. 0.0 for identical subspaces.
        Maximum pi/2 * sqrt(p) for fully orthogonal subspaces.

    Examples:
        >>> import jax.numpy as jnp
        >>> u = jnp.eye(3, 2)
        >>> grassmann_distance(u, u)
        0.0
    """
    u = jnp.asarray(u, dtype=jnp.float32)
    v = jnp.asarray(v, dtype=jnp.float32)

    # Singular values of U^T V
    sigma = jnp.linalg.svd(u.T @ v, compute_uv=False)

    # Clamp to valid range for arccos
    sigma = jnp.clip(sigma, -1.0, 1.0)

    # Principal angles
    theta = jnp.arccos(sigma)

    return jnp.sqrt(jnp.sum(theta**2))


def stiefel_distance(u: Any, v: Any) -> Any:
    """Extrinsic distance on the Stiefel manifold St(p, n).

    Frobenius distance between orthonormal p-frames: ``||U - V||_F``.

    Unlike Grassmann distance, this is basis-DEPENDENT: the same subspace
    with a different orthonormal basis gives a nonzero distance.

    Complexity: O(n*p) for element-wise difference and norm.

    Args:
        u: First orthonormal matrix, shape (n, p).
        v: Second orthonormal matrix, shape (n, p).

    Returns:
        Stiefel distance. Lower is better. 0.0 for identical frames.

    Examples:
        >>> import jax.numpy as jnp
        >>> u = jnp.eye(3, 2)
        >>> stiefel_distance(u, u)
        0.0
    """
    u = jnp.asarray(u, dtype=jnp.float32)
    v = jnp.asarray(v, dtype=jnp.float32)

    return jnp.linalg.norm(u - v)


def ultrahyperbolic_distance(
    a: Any,
    b: Any,
    *,
    signature: tuple[int, int],
) -> Any:
    """Geodesic distance on the pseudo-hyperboloid with given signature.

    Pseudo-Riemannian inner product:
    ``<a,b>_{p,q} = -sum(a[:p]*b[:p]) + sum(a[p:]*b[p:])``

    Points live on ``<x,x>_{p,q} = -1``. Distance = ``arccosh(-<a,b>_{p,q})``
    for timelike-separated points.

    Generalizes Lorentz distance (signature=(1,n)) to arbitrary
    mixed-curvature spaces (UltraE pattern, Xiong et al. KDD 2022).

    Special cases:
    - ``signature=(1, n)``: Lorentz/hyperboloid (pure hyperbolic)
    - ``signature=(0, n)``: Spherical (all spacelike)
    - ``signature=(p, q)`` with p,q > 0: Mixed curvature

    Complexity: O(n) for inner product and arccosh.

    Args:
        a: First point on the pseudo-hyperboloid.
        b: Second point on the pseudo-hyperboloid.
        signature: Tuple (p, q) specifying p timelike and q spacelike dims.

    Returns:
        Pseudo-Riemannian distance. Lower is better. 0.0 for identical points.

    Examples:
        >>> import jax.numpy as jnp
        >>> a = jnp.array([jnp.sqrt(2.0), 1.0, 0.0])
        >>> ultrahyperbolic_distance(a, a, signature=(1, 2))
        0.0
    """
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)

    p, _q = signature

    # Pseudo-Riemannian inner product
    inner = -jnp.sum(a[:p] * b[:p]) + jnp.sum(a[p:] * b[p:])

    # Clamp for numerical stability (arccosh domain is [1, inf))
    clamped = jnp.maximum(-inner, 1.0 + _EPSILON)

    # Numerically stable arccosh: log(x + sqrt(x^2 - 1))
    return jnp.log(clamped + jnp.sqrt(jnp.maximum(clamped**2 - 1.0, _EPSILON)))
