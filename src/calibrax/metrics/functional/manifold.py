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

from typing import Literal

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from calibrax.metrics._utils import _EPSILON, _prepare_arrays, safe_asinh, safe_norm, safe_root
from calibrax.metrics.functional._hyperboloid import hyperboloid_haversine_distance


# Knyazev and Argentati's switch between the sine and the cosine of a principal angle: pi/4.
_HALF_SQRT2 = 0.5**0.5

#: What ``ultrahyperbolic_distance`` returns for a pair no geodesic joins.
Disconnected = Literal["broken_geodesic", "nan"]
DISCONNECTED_CHOICES: tuple[str, ...] = ("broken_geodesic", "nan")


def spd_affine_invariant_distance(a: ArrayLike, b: ArrayLike) -> jax.Array:
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


def spd_log_euclidean_distance(a: ArrayLike, b: ArrayLike) -> jax.Array:
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

    return safe_norm(log_a - log_b)


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


def grassmann_distance(u: ArrayLike, v: ArrayLike) -> jax.Array:
    """Geodesic distance on the Grassmann manifold Gr(p, n): the norm of the principal angles.

    The principal angles between the two p-dimensional subspaces of R^n come from the
    singular values of both ``U^T V`` (their cosines) and ``(I - U U^T) V`` (their sines),
    as Knyazev and Argentati (2002, SIAM J. Sci. Comput. 23(6), Algorithm 3.1) compute them
    and ``scipy.linalg.subspace_angles`` does: an angle whose sine is below 1/sqrt(2) is
    taken from the sine, the rest from the cosine, since ``arccos`` near 1 cannot resolve a
    small angle in float32 (it read 3.45e-4 between a subspace and itself). Only singular
    values are differentiated, whose derivative stays finite where they repeat, and the norm
    is zero-safe, so ``jax.grad`` is finite at identical subspaces.

    Basis-independent: same subspace with different orthonormal basis gives distance 0.

    Complexity: O(n*p^2) for the two SVDs.

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

    overlap = u.T @ v
    # Cosines in descending order pair with sines in ascending order, angle by angle.
    cosines = jnp.clip(jnp.linalg.svd(overlap, compute_uv=False), 0.0, 1.0)
    sines = jnp.clip(jnp.sort(jnp.linalg.svd(v - u @ overlap, compute_uv=False)), 0.0, 1.0)
    small = sines < _HALF_SQRT2
    # Each branch evaluated only on values where its derivative is finite (double where).
    from_sine = jnp.arcsin(jnp.where(small, sines, 0.0))
    from_cosine = jnp.arccos(jnp.where(small, 0.0, cosines))
    return safe_norm(jnp.where(small, from_sine, from_cosine))


def stiefel_distance(u: ArrayLike, v: ArrayLike) -> jax.Array:
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

    return safe_norm(u - v)


def _minkowski_square(v: jax.Array, timelike: int) -> jax.Array:
    """``<v, v>_{p,q} = -sum(v[:p]**2) + sum(v[p:]**2)``, the indefinite quadratic form.

    Args:
        v: A vector of the ambient space.
        timelike: ``p``, the number of timelike (negative) coordinates.

    Returns:
        The form's value.
    """
    return -jnp.sum(v[:timelike] ** 2) + jnp.sum(v[timelike:] ** 2)


def _check_ultrahyperbolic_arguments(
    size: int, signature: tuple[int, int], disconnected: str
) -> None:
    """Refuse a signature or a choice the distance cannot honour, at trace time.

    Args:
        size: The points' number of coordinates.
        signature: ``(p, q)``.
        disconnected: The choice for a pair no geodesic joins.

    Raises:
        ValueError: If ``p`` is 0, ``p + q`` is not ``size``, or ``disconnected`` is not one of
            :data:`DISCONNECTED_CHOICES`.
    """
    p, q = signature
    if p < 1:
        msg = (
            f"signature {signature} has no points: <x, x> = -1 needs at least one timelike "
            "coordinate (the sphere is signature (n, 0))"
        )
        raise ValueError(msg)
    if p + q != size:
        msg = f"signature {signature} describes {p + q} coordinates; the points have {size} coordinates"
        raise ValueError(msg)
    if disconnected not in DISCONNECTED_CHOICES:
        msg = f"disconnected must be one of {DISCONNECTED_CHOICES}, got {disconnected!r}"
        raise ValueError(msg)


def ultrahyperbolic_distance(  # noqa: DOC502  # raised by _prepare_arrays and _check_ultrahyperbolic_arguments
    a: ArrayLike,
    b: ArrayLike,
    *,
    signature: tuple[int, int],
    disconnected: Disconnected = "broken_geodesic",
) -> jax.Array:
    """Distance on the pseudo-hyperboloid ``<x, x>_{p,q} = -1`` of signature ``(p, q)``.

    ``<a, b>_{p,q} = -sum(a[:p] * b[:p]) + sum(a[p:] * b[p:])``: ``p`` timelike coordinates first,
    then ``q`` spacelike ones. Signature ``(1, n)`` is the hyperboloid model of hyperbolic space,
    ``(n, 0)`` the unit sphere, and ``(2, n - 1)`` anti-de Sitter space.

    **The three regimes.** With ``u = -<a, b>_{p,q}`` (Bengtsson, *Anti-de Sitter space*, eqs. 19,
    20 and 299; Law & Stam, NeurIPS 2020, eq. 8):

    * ``u >= 1``: a spacelike geodesic joins the points, of length ``arccosh(u)``;
    * ``-1 <= u < 1``: a timelike geodesic, of length ``arccos(u)``;
    * ``u < -1``: **no geodesic joins them** (the manifold is geodesically disconnected for
      ``p >= 2``, and the two sheets of the ``(1, n)`` hyperboloid are not joined at all), so the
      geodesic distance is undefined. ``disconnected`` says what to return there.

    **``disconnected="broken_geodesic"``, the default:** ``pi + d(-a, b)``, the length of the path
    from ``a`` to its antipode ``-a`` (a timelike geodesic of length ``pi``) and on to ``b``, which
    ``-a`` reaches by a spacelike geodesic. Law & Stam (2020, p. 5) propose it and QGCN (Xiong et
    al., NeurIPS 2022), Q-GT and HyperCore compute it. Effects: always finite, so a loss built on it
    stays finite when an optimiser carries points into the disconnected region, and continuous with
    the timelike branch at ``u = -1``, where both are ``pi``. Caveats: in that region it is a path
    length, not a geodesic distance, and not the shortest such path in general; it makes the
    function a symmetric premetric, not a metric.

    **``disconnected="nan"``:** NaN there, the undefined value, as calibrax returns NaN for a
    domain error under ``jit`` rather than a number that could be mistaken for a distance.
    Effects: a pair no geodesic joins cannot pass unnoticed; caveat: a NaN reaching a loss poisons
    every parameter its gradient touches.

    **Evaluation.** Each branch is computed in half-angle form from ``<a - b, a - b> = 2(u - 1)`` and
    ``<a + b, a + b> = -2(u + 1)``, which keeps the small differences ``u - 1`` and ``u + 1`` from
    being formed by subtraction: ``2 asinh(sqrt(<a - b, a - b>) / 2)`` for the spacelike branch (the
    hyperbolic haversine), ``2 atan2(sqrt(-<a - b, a - b>), sqrt(-<a + b, a + b>))`` for the
    timelike one (Kahan's half-angle ``atan2`` form, *Mindless*, sec. 12, applied to the indefinite
    form), and the spacelike form between ``-a`` and ``b`` for the broken geodesic. At signature
    ``(1, n)`` the spacelike branch is :func:`~calibrax.metrics.functional.distance.lorentz_distance`'s
    cancellation-free haversine (hyperbolix), the same computation. Identical points give exactly 0
    with an exactly-zero gradient.

    **Assumptions and caveats.** The points lie on the pseudo-hyperboloid; nothing projects them.
    For ``p >= 2`` the form ``<a - b, a - b>`` still subtracts its timelike part from its spacelike
    part, so points far from the origin lose float32 digits (no cancellation-free form for a general
    signature was found; at ``(1, n)`` the haversine has none). Two distinct points on each other's
    light cone (``u = 1``) are at distance 0, as in Law & Stam's eq. 8. The derivative is unbounded
    as a pair approaches ``u = 1`` or ``u = -1`` from a distance, as ``arccosh`` and ``arccos`` are
    there.

    Args:
        a: First point on the pseudo-hyperboloid.
        b: Second point on the pseudo-hyperboloid.
        signature: ``(p, q)``, ``p >= 1`` timelike and ``q`` spacelike coordinates, ``p + q`` the
            points' size. Static under ``jit``.
        disconnected: ``"broken_geodesic"`` (default) or ``"nan"``, for a pair no geodesic joins.
            Static under ``jit``.

    Returns:
        The distance. Lower is better. 0.0 for identical points.

    Examples:
        >>> import jax.numpy as jnp
        >>> a = jnp.array([jnp.sqrt(2.0), 1.0, 0.0])
        >>> ultrahyperbolic_distance(a, a, signature=(1, 2))
        0.0

    Raises:
        ValueError: If ``a`` and ``b`` do not have the same shape, the signature has no points or
            does not match their size, or ``disconnected`` is not a choice.
    """
    # Two points on one pseudo-hyperboloid; the slicing below assumes a single point.
    a, b = _prepare_arrays(a, b, dtype=jnp.float32)
    _check_ultrahyperbolic_arguments(a.shape[-1], signature, disconnected)
    p, _ = signature

    difference = _minkowski_square(a - b, p)
    total = _minkowski_square(a + b, p)
    is_spacelike = difference >= 0.0
    is_timelike = ~is_spacelike & (total <= 0.0)

    if p == 1:
        # The haversine reads the upper sheet: a pair on the lower one is mirrored onto it, and
        # the broken geodesic's second leg, from -a to b, is carried onto b's sheet.
        sheet_a = jnp.where(a[0] < 0.0, -1.0, 1.0)
        sheet_b = jnp.where(b[0] < 0.0, -1.0, 1.0)
        spacelike = hyperboloid_haversine_distance(sheet_a * a, sheet_a * b)
        across = hyperboloid_haversine_distance(-sheet_b * a, sheet_b * b)
    else:
        # Each branch reads only its own regime's argument (inner where): an untaken branch's
        # root never meets a negative value, so its gradient cannot be NaN.
        spacelike = 2.0 * safe_asinh(safe_root(jnp.where(is_spacelike, difference, 0.0)) / 2.0)
        across = 2.0 * safe_asinh(
            safe_root(jnp.where(is_spacelike | is_timelike, 0.0, total)) / 2.0
        )
    timelike = 2.0 * jnp.arctan2(
        safe_root(jnp.where(is_timelike, -difference, 1.0)),
        safe_root(jnp.where(is_timelike, -total, 1.0)),
    )
    unjoined = (
        jnp.pi + across if disconnected == "broken_geodesic" else jnp.full_like(across, jnp.nan)
    )
    return jnp.where(is_spacelike, spacelike, jnp.where(is_timelike, timelike, unjoined))
