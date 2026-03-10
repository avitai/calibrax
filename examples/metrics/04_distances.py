# ---
# jupyter:
#   jupytext:
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
# # Distances and Divergences Across Geometric Spaces
#
# | | |
# |---|---|
# | **Level** | Tier 2: Tutorial |
# | **Time** | ~20 minutes |
# | **Prerequisites** | `01_quickstart.py`, basic JAX arrays |
# | **Metrics covered** | euclidean, cosine, poincare, lorentz, KL, JS, Wasserstein, Sinkhorn |
# | **Key concepts** | Geometric hierarchy, curvature matching, divergence asymmetry |

# %%
"""Distance and divergence metrics across geometric spaces.

Demonstrates:
- Euclidean, cosine, Manhattan distances on vectors
- Poincare and Lorentz distances on hyperbolic embeddings
- KL divergence, JS divergence between distributions
- Wasserstein distance, Sinkhorn divergence between samples
- Shannon entropy and mutual information
"""

import jax.numpy as jnp

from calibrax.metrics.functional.distance import (
    cosine_distance,
    euclidean_distance,
    lorentz_distance,
    manhattan_distance,
    poincare_distance,
)
from calibrax.metrics.functional.divergence import (
    js_divergence,
    kl_divergence,
    sinkhorn_divergence,
    wasserstein_1d,
)
from calibrax.metrics.functional.information import (
    entropy,
    mutual_information,
)


def main() -> None:
    """Run distance and divergence examples."""
    # -- Vector distances --------------------------------------------------
    print("=== Vector Distances ===")
    a = jnp.array([1.0, 0.0, 0.0])
    b = jnp.array([0.0, 1.0, 0.0])
    c = jnp.array([1.0, 1.0, 0.0])

    print(f"  a = {a.tolist()}")
    print(f"  b = {b.tolist()}")
    print(f"  c = {c.tolist()}")

    print(f"\n  Euclidean(a, b) = {euclidean_distance(a, b):.4f}")
    print(f"  Euclidean(a, c) = {euclidean_distance(a, c):.4f}")
    print(f"  Euclidean(b, c) = {euclidean_distance(b, c):.4f}")

    print(f"\n  Cosine(a, b)    = {cosine_distance(a, b):.4f}  (orthogonal -> 1.0)")
    print(f"  Cosine(a, c)    = {cosine_distance(a, c):.4f}  (45 degrees)")
    print(f"  Cosine(a, a)    = {cosine_distance(a, a):.4f}  (identical -> 0.0)")

    print(f"\n  Manhattan(a, b) = {manhattan_distance(a, b):.4f}")
    print(f"  Manhattan(a, c) = {manhattan_distance(a, c):.4f}")

    # -- Hyperbolic distances (negative curvature) -------------------------
    print("\n=== Hyperbolic Distances ===")
    print("  Poincare ball model (points must satisfy ||x|| < 1):")
    origin = jnp.array([0.0, 0.0])
    near = jnp.array([0.3, 0.0])
    far = jnp.array([0.8, 0.0])

    print(f"    d(origin, near) = {poincare_distance(origin, near):.4f}")
    print(f"    d(origin, far)  = {poincare_distance(origin, far):.4f}")
    print(f"    d(near, far)    = {poincare_distance(near, far):.4f}")
    print("    Distance grows rapidly as points approach the ball boundary.")

    print("\n  Lorentz hyperboloid model (first component is timelike):")
    # Points on hyperboloid: x0 = sqrt(1 + ||x_spatial||^2)
    p1 = jnp.array([1.0, 0.0, 0.0])  # origin
    spatial_2 = jnp.array([0.5, 0.3])
    p2 = jnp.concatenate([jnp.sqrt(1.0 + jnp.sum(spatial_2**2))[None], spatial_2])
    spatial_3 = jnp.array([0.9, 0.1])
    p3 = jnp.concatenate([jnp.sqrt(1.0 + jnp.sum(spatial_3**2))[None], spatial_3])

    print(f"    d(p1, p2) = {lorentz_distance(p1, p2):.4f}")
    print(f"    d(p1, p3) = {lorentz_distance(p1, p3):.4f}")
    print(f"    d(p2, p3) = {lorentz_distance(p2, p3):.4f}")
    print("    Lorentz model is numerically more stable near the boundary.")

    # -- Distribution divergences ------------------------------------------
    print("\n=== Distribution Divergences ===")
    # Two probability distributions over 4 outcomes
    p = jnp.array([0.4, 0.3, 0.2, 0.1])
    q = jnp.array([0.25, 0.25, 0.25, 0.25])  # uniform
    r = jnp.array([0.1, 0.2, 0.3, 0.4])

    print(f"  p = {p.tolist()}")
    print(f"  q = {q.tolist()} (uniform)")
    print(f"  r = {r.tolist()}")

    print(f"\n  KL(p || q) = {kl_divergence(p, q):.6f}")
    print(f"  KL(q || p) = {kl_divergence(q, p):.6f}  (asymmetric!)")
    print(f"  KL(p || r) = {kl_divergence(p, r):.6f}")

    print(f"\n  JS(p, q)   = {js_divergence(p, q):.6f}  (symmetric)")
    print(f"  JS(q, p)   = {js_divergence(q, p):.6f}  (same value)")
    print(f"  JS(p, r)   = {js_divergence(p, r):.6f}")

    # -- Sample-based distances (Wasserstein, Sinkhorn) --------------------
    print("\n=== Sample-Based Distances ===")
    # Two 1D sample distributions
    samples_a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    samples_b = jnp.array([2.0, 3.0, 4.0, 5.0, 6.0])  # shifted by 1
    samples_c = jnp.array([1.0, 1.5, 3.0, 4.5, 5.0])  # different shape

    print(f"  samples_a = {samples_a.tolist()}")
    print(f"  samples_b = {samples_b.tolist()} (shifted)")
    print(f"  samples_c = {samples_c.tolist()} (different shape)")

    print(f"\n  Wasserstein(a, b) = {wasserstein_1d(samples_a, samples_b):.4f}")
    print(f"  Wasserstein(a, c) = {wasserstein_1d(samples_a, samples_c):.4f}")
    print(f"  Wasserstein(a, a) = {wasserstein_1d(samples_a, samples_a):.4f}")

    # Sinkhorn on 2D point clouds
    points_x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    points_y = jnp.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])

    sink_val = sinkhorn_divergence(points_x, points_y, regularization=0.1)
    print(f"\n  Sinkhorn(2D cloud X, Y) = {sink_val:.6f}")
    sink_self = sinkhorn_divergence(points_x, points_x, regularization=0.1)
    print(f"  Sinkhorn(X, X)          = {sink_self:.6f}  (debiased -> ~0)")

    # -- Information-theoretic metrics -------------------------------------
    print("\n=== Information-Theoretic Metrics ===")

    # Entropy of distributions
    uniform4 = jnp.array([0.25, 0.25, 0.25, 0.25])
    peaked = jnp.array([0.9, 0.05, 0.03, 0.02])
    print(f"  Entropy(uniform)  = {entropy(uniform4):.4f}  (max entropy for 4 outcomes)")
    print(f"  Entropy(peaked)   = {entropy(peaked):.4f}  (low uncertainty)")

    # Mutual information from joint probability table
    # Independent: MI = 0
    independent_joint = jnp.array(
        [
            [0.25, 0.25],
            [0.25, 0.25],
        ]
    )
    # Dependent: MI > 0
    dependent_joint = jnp.array(
        [
            [0.45, 0.05],
            [0.05, 0.45],
        ]
    )

    print(f"\n  MI(independent) = {mutual_information(independent_joint):.6f}  (~0)")
    print(f"  MI(dependent)   = {mutual_information(dependent_joint):.6f}  (strong dependence)")


if __name__ == "__main__":
    main()
