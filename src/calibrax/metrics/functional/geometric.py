"""Geometric and point cloud metrics -- pure math.

Distance metrics for comparing point sets and shapes. All operations
are pure mathematical -- no pretrained models or external libraries.

Includes: chamfer_distance, earth_movers_distance_1d, directed_hausdorff,
hausdorff_distance.
Registered with ``domain="geometric"``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON


def chamfer_distance(set_a: Any, set_b: Any) -> Any:
    """Chamfer distance between two point sets.

    Mean of nearest-neighbor distances in both directions.
    Symmetric: CD(A, B) = CD(B, A).

    Args:
        set_a: First point set, shape (n, d).
        set_b: Second point set, shape (m, d).

    Returns:
        Chamfer distance. Lower is better. 0.0 = identical sets.

    Examples:
        >>> import jax.numpy as jnp
        >>> pts = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> chamfer_distance(pts, pts)
        0.0
    """
    a = jnp.asarray(set_a, dtype=jnp.float32)
    b = jnp.asarray(set_b, dtype=jnp.float32)

    # Pairwise distances (n, m)
    diff = a[:, None, :] - b[None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + _EPSILON)

    # Mean of min distances in both directions
    mean_a_to_b = jnp.mean(jnp.min(dists, axis=1))
    mean_b_to_a = jnp.mean(jnp.min(dists, axis=0))

    return (mean_a_to_b + mean_b_to_a) / 2.0


def earth_movers_distance_1d(a: Any, b: Any) -> Any:
    """1D Earth Mover's Distance (Wasserstein-1).

    Delegates to wasserstein_1d from the divergence module (DRY).

    Args:
        a: First 1D distribution/sample array.
        b: Second 1D distribution/sample array.

    Returns:
        EMD value. Lower is better. 0.0 = identical distributions.

    Examples:
        >>> import jax.numpy as jnp
        >>> earth_movers_distance_1d(jnp.array([1.0, 2.0, 3.0]), jnp.array([1.0, 2.0, 3.0]))
        0.0
    """
    from calibrax.metrics.functional.divergence import wasserstein_1d

    return wasserstein_1d(a, b)


def directed_hausdorff(set_a: Any, set_b: Any) -> Any:
    """Directed Hausdorff distance from set_a to set_b.

    max_{a in A} min_{b in B} d(a, b). NOT symmetric:
    directed_hausdorff(A, B) != directed_hausdorff(B, A) in general.

    Measures the worst-case nearest-neighbor distance from A to B.

    Args:
        set_a: Source point set, shape (n, d).
        set_b: Target point set, shape (m, d).

    Returns:
        Directed Hausdorff distance. Lower is better.

    Examples:
        >>> import jax.numpy as jnp
        >>> a = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        >>> b = jnp.array([[0.0, 0.0]])
        >>> directed_hausdorff(a, b)  # 1.0 (point [1,0] is 1.0 from [0,0])
        ...
    """
    a = jnp.asarray(set_a, dtype=jnp.float32)
    b = jnp.asarray(set_b, dtype=jnp.float32)

    diff = a[:, None, :] - b[None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + _EPSILON)

    # For each point in A, find min distance to B, then take max
    min_dists = jnp.min(dists, axis=1)
    return jnp.max(min_dists)


def hausdorff_distance(set_a: Any, set_b: Any) -> Any:
    """Hausdorff distance (symmetric) between two point sets.

    max(directed_hausdorff(A, B), directed_hausdorff(B, A)).
    A true metric on non-empty compact subsets.

    Args:
        set_a: First point set, shape (n, d).
        set_b: Second point set, shape (m, d).

    Returns:
        Hausdorff distance. Lower is better. 0.0 = identical sets.

    Examples:
        >>> import jax.numpy as jnp
        >>> pts = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> hausdorff_distance(pts, pts)
        0.0
    """
    return jnp.maximum(
        directed_hausdorff(set_a, set_b),
        directed_hausdorff(set_b, set_a),
    )
