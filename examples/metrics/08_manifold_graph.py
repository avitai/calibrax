# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Manifold and Graph Metrics
#
# | | |
# |---|---|
# | **Level** | Tier 3: Advanced Guide |
# | **Time** | ~30 minutes |
# | **Prerequisites** | `04_distances.py`, linear algebra basics |
# | **Metrics covered** | SPD affine-invariant, Grassmann, spectral, resistance distance |
# | **Key concepts** | Riemannian geometry, graph Laplacian, invariance verification |

# %%
"""Graph and manifold metrics: distances on structured mathematical objects.

Demonstrates:
- Spectral distance between adjacency matrices
- Resistance distance on a network
- Shortest path computation (Floyd-Warshall)
- SPD affine-invariant vs log-Euclidean distance
- Grassmann distance between subspaces
- Ultrahyperbolic distance with different signatures
"""

import jax.numpy as jnp

from calibrax.metrics.functional.graph import (
    resistance_distance,
    shortest_path_distance,
    spectral_distance,
)
from calibrax.metrics.functional.manifold import (
    grassmann_distance,
    spd_affine_invariant_distance,
    spd_log_euclidean_distance,
    ultrahyperbolic_distance,
)


def main() -> None:
    """Run graph and manifold metric examples."""
    # == GRAPH METRICS =====================================================
    print("=" * 60)
    print("GRAPH METRICS")
    print("=" * 60)

    # -- Path graph: 0 -- 1 -- 2 -- 3 ------------------------------------
    print("\n=== Path Graph (4 nodes) ===")
    path_adj = jnp.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=jnp.float32,
    )
    print("  Adjacency matrix:")
    for row in path_adj.tolist():
        print(f"    {row}")

    # Spectral distance (same graph -> 0)
    print(f"\n  Spectral distance (path vs itself): {spectral_distance(path_adj, path_adj):.6f}")

    # -- Cycle graph: 0 -- 1 -- 2 -- 3 -- 0 ------------------------------
    print("\n=== Cycle Graph (4 nodes) ===")
    cycle_adj = jnp.array(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=jnp.float32,
    )

    spec_dist = spectral_distance(path_adj, cycle_adj)
    print(f"  Spectral distance (path vs cycle): {spec_dist:.6f}")
    print("  Non-zero because the Laplacian eigenvalue spectra differ.")

    # -- Shortest path distance (Floyd-Warshall) --------------------------
    print("\n=== Shortest Path Distance (Floyd-Warshall) ===")
    sp_matrix = shortest_path_distance(path_adj)
    print("  Path graph shortest distances:")
    for i in range(4):
        row = [f"{float(sp_matrix[i, j]):.0f}" for j in range(4)]
        print(f"    node {i}: {row}")
    print(f"  d(0, 3) = {float(sp_matrix[0, 3]):.0f} (3 hops along the path)")

    sp_cycle = shortest_path_distance(cycle_adj)
    print("\n  Cycle graph shortest distances:")
    for i in range(4):
        row = [f"{float(sp_cycle[i, j]):.0f}" for j in range(4)]
        print(f"    node {i}: {row}")
    print(f"  d(0, 2) = {float(sp_cycle[0, 2]):.0f} (2 hops either direction)")

    # -- Resistance distance -----------------------------------------------
    print("\n=== Resistance Distance ===")
    omega_path = resistance_distance(path_adj)
    print("  Path graph resistance distances:")
    for i in range(4):
        row = [f"{float(omega_path[i, j]):.2f}" for j in range(4)]
        print(f"    node {i}: {row}")

    omega_cycle = resistance_distance(cycle_adj)
    print("\n  Cycle graph resistance distances:")
    for i in range(4):
        row = [f"{float(omega_cycle[i, j]):.2f}" for j in range(4)]
        print(f"    node {i}: {row}")

    print("\n  Resistance distance captures topology: nodes with more")
    print("  parallel paths have lower resistance distance.")
    print(f"  Path d(0,3) = {float(omega_path[0, 3]):.2f} (single path)")
    print(f"  Cycle d(0,2) = {float(omega_cycle[0, 2]):.2f} (two parallel paths)")

    # == MANIFOLD METRICS ==================================================
    print()
    print("=" * 60)
    print("MANIFOLD METRICS")
    print("=" * 60)

    # -- SPD matrices (covariance matrices) --------------------------------
    print("\n=== SPD Manifold: Covariance Matrix Distances ===")

    # Three 3x3 SPD matrices (symmetric positive definite)
    spd_identity = jnp.eye(3)
    spd_scaled = jnp.eye(3) * 4.0  # uniformly scaled
    spd_aniso = jnp.array(
        [
            [2.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )  # anisotropic

    print("  Affine-invariant distance (geometrically exact):")
    print(f"    d(I, I)         = {spd_affine_invariant_distance(spd_identity, spd_identity):.6f}")
    print(f"    d(I, 4*I)       = {spd_affine_invariant_distance(spd_identity, spd_scaled):.6f}")
    print(f"    d(I, aniso)     = {spd_affine_invariant_distance(spd_identity, spd_aniso):.6f}")
    print(f"    d(4*I, aniso)   = {spd_affine_invariant_distance(spd_scaled, spd_aniso):.6f}")

    print("\n  Log-Euclidean distance (faster approximation):")
    print(f"    d(I, I)         = {spd_log_euclidean_distance(spd_identity, spd_identity):.6f}")
    print(f"    d(I, 4*I)       = {spd_log_euclidean_distance(spd_identity, spd_scaled):.6f}")
    print(f"    d(I, aniso)     = {spd_log_euclidean_distance(spd_identity, spd_aniso):.6f}")
    print(f"    d(4*I, aniso)   = {spd_log_euclidean_distance(spd_scaled, spd_aniso):.6f}")

    print("\n  Affine-invariant is congruence-invariant: d(A,B) = d(MAM',MBM').")
    print("  Log-Euclidean is faster but only orthogonally invariant.")

    # -- Grassmann manifold (subspace comparison) --------------------------
    print("\n=== Grassmann Manifold: Subspace Distances ===")

    # Two 2-dimensional subspaces of R^4 (orthonormal bases)
    # Subspace 1: span of first two standard basis vectors
    u1 = jnp.eye(4, 2)  # shape (4, 2)
    # Subspace 2: rotated version
    angle = jnp.pi / 6  # 30 degrees
    rotation = jnp.array(
        [
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    u2 = rotation
    # Orthonormalize u2 via QR
    u2, _ = jnp.linalg.qr(u2)

    # Subspace 3: orthogonal complement (last two basis vectors)
    u3 = jnp.eye(4, 2, k=2)  # columns 2 and 3

    print(f"  d(U1, U1) = {grassmann_distance(u1, u1):.6f}  (same subspace)")
    print(f"  d(U1, U2) = {grassmann_distance(u1, u2):.6f}  (30-deg rotation)")
    print(f"  d(U1, U3) = {grassmann_distance(u1, u3):.6f}  (orthogonal complement)")
    print("  Grassmann distance is basis-independent: depends only on the subspace.")

    # -- Ultrahyperbolic distance (mixed signature) ------------------------
    print("\n=== Ultrahyperbolic Distance (Mixed Signature) ===")

    # Signature (1, 2): 1 timelike + 2 spacelike (standard hyperboloid)
    # Point on hyperboloid: -t^2 + x^2 + y^2 = -1, so t = sqrt(1 + x^2 + y^2)
    def hyperboloid_point(x: float, y: float) -> jnp.ndarray:
        """Create a point on the (1,2) hyperboloid."""
        t = jnp.sqrt(1.0 + x**2 + y**2)
        return jnp.array([t, x, y])

    origin = hyperboloid_point(0.0, 0.0)
    p1 = hyperboloid_point(0.5, 0.3)
    p2 = hyperboloid_point(1.0, 0.0)

    print("  Signature (1, 2) -- standard hyperboloid:")
    d_12_self = ultrahyperbolic_distance(origin, origin, signature=(1, 2))
    d_12_near = ultrahyperbolic_distance(origin, p1, signature=(1, 2))
    d_12_far = ultrahyperbolic_distance(origin, p2, signature=(1, 2))
    print(f"    d(origin, origin) = {d_12_self:.6f}")
    print(f"    d(origin, p1)     = {d_12_near:.6f}")
    print(f"    d(origin, p2)     = {d_12_far:.6f}")

    # Signature (2, 2): 2 timelike + 2 spacelike (ultrahyperbolic)
    # Point satisfying -t1^2 - t2^2 + x1^2 + x2^2 = -1
    def ultra_point(x1: float, x2: float) -> jnp.ndarray:
        """Create a point on the (2,2) pseudo-hyperboloid."""
        t_sq = 1.0 + x1**2 + x2**2
        t1 = jnp.sqrt(t_sq / 2.0)
        t2 = jnp.sqrt(t_sq / 2.0)
        return jnp.array([t1, t2, x1, x2])

    origin_ultra = ultra_point(0.0, 0.0)
    q1 = ultra_point(0.3, 0.4)
    q2 = ultra_point(0.8, 0.6)

    print("\n  Signature (2, 2) -- ultrahyperbolic space:")
    d_22_self = ultrahyperbolic_distance(origin_ultra, origin_ultra, signature=(2, 2))
    d_22_near = ultrahyperbolic_distance(origin_ultra, q1, signature=(2, 2))
    d_22_far = ultrahyperbolic_distance(origin_ultra, q2, signature=(2, 2))
    print(f"    d(origin, origin) = {d_22_self:.6f}")
    print(f"    d(origin, q1)     = {d_22_near:.6f}")
    print(f"    d(origin, q2)     = {d_22_far:.6f}")

    print("\n  Ultrahyperbolic spaces generalize hyperbolic geometry.")
    print("  Signature (1, n) recovers standard Lorentz/hyperbolic distance.")
    print("  Mixed signatures model spaces with multiple curvature directions,")
    print("  useful for knowledge graph embeddings (UltraE).")


if __name__ == "__main__":
    main()
