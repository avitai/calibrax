# %% [markdown]
"""
# Distances and Divergences Across Geometric Spaces

| Metadata | Value |
| --- | --- |
| **Tier** | Tutorial |
| **Devices** | CPU-compatible |
| **Prerequisites** | The quickstart example; basic JAX arrays |

## Overview

calibrax organises its distance and divergence functions along a geometric hierarchy: flat
(Euclidean, Manhattan, cosine), hyperbolic (Poincare, Lorentz), distributional (KL, JS,
Wasserstein, Sinkhorn) and information-theoretic (entropy, mutual information). This example
computes each family on small inputs and names the property that decides when each fits. These
functions underlie embedding evaluation, distribution comparison and metric learning, and the
difference between a true distance, which satisfies the triangle inequality, and a divergence,
which may be asymmetric, decides which one a task needs.

## Learning Goals

1. Compute Euclidean, cosine and Manhattan distances between vectors
2. Measure hyperbolic distances in the Poincare ball and on the Lorentz hyperboloid
3. Compare probability distributions with KL and JS divergence
4. Compare samples and point clouds with Wasserstein distance and Sinkhorn divergence
5. Quantify uncertainty and dependence with entropy and mutual information
"""

# %% [markdown]
"""
## Setup
"""

# %%
import jax
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
from calibrax.metrics.functional.information import entropy, mutual_information


# %% [markdown]
"""
## Core Concepts

- Euclidean distance is rotation-invariant and sensitive to magnitude; cosine distance,
  `1 - cosine similarity`, is scale-invariant and measures the angle alone; Manhattan distance is
  axis-aligned.
- Hyperbolic space has negative curvature, which suits embeddings of hierarchies such as trees
  and taxonomies. calibrax supports two equivalent models: the Poincare ball, whose points lie
  inside the unit ball and whose distances grow without bound towards its boundary, and the
  Lorentz hyperboloid, whose first coordinate is timelike, `x_0 = sqrt(1 + ||x_spatial||^2)`, and
  which is numerically more stable near the boundary.
- KL divergence measures the information lost when `q` approximates `p`; it is asymmetric and
  unbounded. JS divergence symmetrises it and is bounded by `log(2)`.
- Sinkhorn divergence debiases the entropic optimal-transport cost, so the divergence of a point
  cloud from itself is zero.

## Implementation

### Step 1: Measure flat vector distances
"""

# %%
A = jnp.array([1.0, 0.0, 0.0])
B = jnp.array([0.0, 1.0, 0.0])
C = jnp.array([1.0, 1.0, 0.0])


def flat_distances() -> dict[str, float]:
    """Compute the flat distances between three vectors.

    Returns:
        Each distance by the pair it measures.
    """
    return {
        "Euclidean(a, b)": float(euclidean_distance(A, B)),
        "Euclidean(a, c)": float(euclidean_distance(A, C)),
        "Euclidean(b, c)": float(euclidean_distance(B, C)),
        "Cosine(a, b), orthogonal": float(cosine_distance(A, B)),
        "Cosine(a, c), 45 degrees": float(cosine_distance(A, C)),
        "Cosine(a, a), identical": float(cosine_distance(A, A)),
        "Manhattan(a, b)": float(manhattan_distance(A, B)),
        "Manhattan(a, c)": float(manhattan_distance(A, C)),
    }


# %% [markdown]
"""
### Step 2: Measure distances in the Poincare ball

Three points on one axis of the unit disc: the origin, one near it and one near the boundary.
"""

# %%
ORIGIN = jnp.array([0.0, 0.0])
NEAR = jnp.array([0.3, 0.0])
FAR = jnp.array([0.8, 0.0])


def poincare_distances() -> dict[str, float]:
    """Compute the Poincare distances between the three points.

    Returns:
        Each distance by the pair it measures.
    """
    return {
        "d(origin, near)": float(poincare_distance(ORIGIN, NEAR)),
        "d(origin, far)": float(poincare_distance(ORIGIN, FAR)),
        "d(near, far)": float(poincare_distance(NEAR, FAR)),
    }


# %% [markdown]
"""
### Step 3: Measure distances on the Lorentz hyperboloid

A point is lifted onto the hyperboloid from its spatial coordinates.
"""


# %%
def on_hyperboloid(spatial: jax.Array) -> jax.Array:
    """Lift spatial coordinates onto the Lorentz hyperboloid.

    Args:
        spatial: The point's spatial coordinates.

    Returns:
        The point, its timelike coordinate first.
    """
    return jnp.concatenate([jnp.sqrt(1.0 + jnp.sum(spatial**2))[None], spatial])


P1 = on_hyperboloid(jnp.array([0.0, 0.0]))
P2 = on_hyperboloid(jnp.array([0.5, 0.3]))
P3 = on_hyperboloid(jnp.array([0.9, 0.1]))


def lorentz_distances() -> dict[str, float]:
    """Compute the Lorentz distances between three points on the hyperboloid.

    Returns:
        Each distance by the pair it measures.
    """
    return {
        "d(p1, p2)": float(lorentz_distance(P1, P2)),
        "d(p1, p3)": float(lorentz_distance(P1, P3)),
        "d(p2, p3)": float(lorentz_distance(P2, P3)),
    }


# %% [markdown]
"""
### Step 4: Compare distributions with KL and JS divergence

Three distributions over four outcomes: `q` is uniform and `r` is `p` reversed.
"""

# %%
P = jnp.array([0.4, 0.3, 0.2, 0.1])
Q = jnp.array([0.25, 0.25, 0.25, 0.25])
R = jnp.array([0.1, 0.2, 0.3, 0.4])


def divergences() -> dict[str, float]:
    """Compute KL and JS divergences between the three distributions.

    Returns:
        Each divergence by the pair it measures.
    """
    return {
        "KL(p || q)": float(kl_divergence(P, Q)),
        "KL(q || p)": float(kl_divergence(Q, P)),
        "KL(p || r)": float(kl_divergence(P, R)),
        "JS(p, q)": float(js_divergence(P, Q)),
        "JS(q, p)": float(js_divergence(Q, P)),
        "JS(p, r)": float(js_divergence(P, R)),
    }


# %% [markdown]
"""
### Step 5: Compare samples with Wasserstein and Sinkhorn

`wasserstein_1d` compares one-dimensional samples; Sinkhorn divergence compares point clouds of
any dimension.
"""

# %%
SAMPLES_A = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
SAMPLES_SHIFTED = SAMPLES_A + 1.0
SAMPLES_RESHAPED = jnp.array([1.0, 1.5, 3.0, 4.5, 5.0])
CLOUD_X = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
CLOUD_Y = CLOUD_X + 0.5


def sample_distances() -> dict[str, float]:
    """Compute the transport distances between samples and point clouds.

    Returns:
        Each distance by the pair it measures.
    """
    return {
        "Wasserstein(a, shifted)": float(wasserstein_1d(SAMPLES_A, SAMPLES_SHIFTED)),
        "Wasserstein(a, reshaped)": float(wasserstein_1d(SAMPLES_A, SAMPLES_RESHAPED)),
        "Wasserstein(a, a)": float(wasserstein_1d(SAMPLES_A, SAMPLES_A)),
        "Sinkhorn(X, Y)": float(sinkhorn_divergence(CLOUD_X, CLOUD_Y, regularization=0.1)),
        "Sinkhorn(X, X)": float(sinkhorn_divergence(CLOUD_X, CLOUD_X, regularization=0.1)),
    }


# %% [markdown]
"""
### Step 6: Quantify uncertainty and dependence

Entropy reads one distribution; mutual information reads a joint probability table.
"""

# %%
PEAKED = jnp.array([0.9, 0.05, 0.03, 0.02])
INDEPENDENT_JOINT = jnp.array([[0.25, 0.25], [0.25, 0.25]])
DEPENDENT_JOINT = jnp.array([[0.45, 0.05], [0.05, 0.45]])


def information() -> dict[str, float]:
    """Compute the entropy of two distributions and the mutual information of two tables.

    Returns:
        Each quantity by what it measures.
    """
    return {
        "Entropy(uniform)": float(entropy(Q)),
        "Entropy(peaked)": float(entropy(PEAKED)),
        "MI(independent)": float(mutual_information(INDEPENDENT_JOINT)),
        "MI(dependent)": float(mutual_information(DEPENDENT_JOINT)),
    }


# %% [markdown]
"""
## Results

### Flat and hyperbolic distances

Orthogonal vectors are at cosine distance 1 and identical ones at 0. In the Poincare ball the far
point is more than three times as far from the origin as the near one, though it is less than
three times as far in the plane.

### Divergences, transport and information

KL divergence changes when its arguments swap and JS divergence does not. `JS(p, r)` equals
`KL(p || q)` because the average of `p` and its reverse is the uniform `q`. Shifting every sample
by 1 moves the Wasserstein distance to 1, the debiased Sinkhorn divergence of a cloud from itself
is 0, and the uniform distribution has the largest entropy over four outcomes, `log(4)`.
"""


# %%
def print_table(title: str, values: dict[str, float]) -> None:
    """Print one table of values.

    Args:
        title: The table's title.
        values: Each value by name.
    """
    print(f"=== {title} ===")
    for name, value in values.items():
        print(f"  {name:28s} = {value:.6f}")


def main() -> dict[str, float]:
    """Run the example and return its results.

    Returns:
        Every distance, divergence and information quantity the example computes.
    """
    tables = {
        "Vector Distances": flat_distances(),
        "Poincare Ball": poincare_distances(),
        "Lorentz Hyperboloid": lorentz_distances(),
        "Distribution Divergences": divergences(),
        "Sample-Based Distances": sample_distances(),
        "Information-Theoretic Metrics": information(),
    }
    for title, values in tables.items():
        print_table(title, values)
    return {name: value for values in tables.values() for name, value in values.items()}


if __name__ == "__main__":
    main()


# %% [markdown]
"""
## Next Steps

- [Metric composition](composition.md): combine metrics into collections, suites and quality
  gates
- [Manifold and graph metrics](advanced-manifold.md): SPD, Grassmann and graph-theoretic
  distances
- [Distance API reference](../../api-reference/metrics/distance.md): every distance function's
  signature
"""
