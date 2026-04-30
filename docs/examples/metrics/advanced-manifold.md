# Advanced Manifold and Graph Metrics

| | |
|---|---|
| **Level** | Advanced |
| **Time** | ~15 minutes |
| **Prerequisites** | [Distances and Spaces](distances-and-spaces.md), [Metric Learning](metric-learning.md) |
| **Format** | Python + Jupyter |

## Overview

This example covers metrics on structured mathematical objects: graph adjacency matrices and Riemannian manifolds. For graphs, it demonstrates spectral distance (comparing Laplacian eigenvalue spectra), resistance distance (effective resistance in electrical networks), and shortest-path distance (Floyd-Warshall). For manifolds, it covers SPD matrix distances (affine-invariant and log-Euclidean), Grassmann distance between subspaces, and ultrahyperbolic distance with mixed metric signatures.

These metrics are used in graph neural network evaluation, covariance matrix comparison (e.g., BCI, finance), subspace tracking, and knowledge graph embedding in mixed-curvature spaces.

## What You'll Learn

1. Compare graph topology with spectral distance between adjacency matrices
2. Compute effective resistance distance and shortest-path distance on graphs
3. Measure SPD matrix dissimilarity with affine-invariant and log-Euclidean distances
4. Quantify subspace separation on the Grassmann manifold
5. Work with ultrahyperbolic distances under different metric signatures

## Files

- **Python Script**: [`examples/metrics/08_manifold_graph.py`](https://github.com/avitai/calibrax/blob/main/examples/metrics/08_manifold_graph.py)
- **Jupyter Notebook**: [`examples/metrics/08_manifold_graph.ipynb`](https://github.com/avitai/calibrax/blob/main/examples/metrics/08_manifold_graph.ipynb)

## Quick Start

```bash
source activate.sh && uv run python examples/metrics/08_manifold_graph.py
```

## Key Concepts

### Graph Metrics

#### Spectral Distance

Spectral distance compares two graphs by the difference between their Laplacian eigenvalue spectra. Isomorphic graphs have zero spectral distance; structurally different graphs have positive distance.

```python
from calibrax.metrics.functional.graph import spectral_distance

# Path graph: 0 -- 1 -- 2 -- 3
path_adj = jnp.array([
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
], dtype=jnp.float32)

# Cycle graph: 0 -- 1 -- 2 -- 3 -- 0
cycle_adj = jnp.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
], dtype=jnp.float32)

spectral_distance(path_adj, path_adj)   # 0.0 (identical)
spectral_distance(path_adj, cycle_adj)  # > 0 (different topology)
```

#### Shortest Path Distance (Floyd-Warshall)

Computes all-pairs shortest path distances from an adjacency matrix. Returns a matrix where entry `(i, j)` is the minimum number of hops between nodes `i` and `j`.

```python
from calibrax.metrics.functional.graph import shortest_path_distance

sp_matrix = shortest_path_distance(path_adj)
# sp_matrix[0, 3] = 3 (three hops along the path)

sp_cycle = shortest_path_distance(cycle_adj)
# sp_cycle[0, 2] = 2 (two hops either direction around the cycle)
```

#### Resistance Distance

Resistance distance treats the graph as an electrical network where each edge has unit resistance. Nodes connected by more parallel paths have lower resistance distance. It captures richer topological information than shortest-path distance.

```python
from calibrax.metrics.functional.graph import resistance_distance

omega_path = resistance_distance(path_adj)
omega_cycle = resistance_distance(cycle_adj)

# Path d(0,3) > Cycle d(0,2): the cycle has parallel paths
```

### Manifold Metrics

#### SPD Matrix Distances

Symmetric Positive Definite (SPD) matrices (e.g., covariance matrices) form a Riemannian manifold. Calibrax provides two distance functions:

- **Affine-invariant**: the geometrically exact geodesic distance. It is invariant under congruence transformations (`d(A, B) = d(MAM', MBM')` for any invertible `M`).
- **Log-Euclidean**: a faster approximation that maps SPD matrices to a flat space via the matrix logarithm. Invariant only under orthogonal transformations.

```python
from calibrax.metrics.functional.manifold import (
    spd_affine_invariant_distance,
    spd_log_euclidean_distance,
)

identity = jnp.eye(3)
scaled = jnp.eye(3) * 4.0
anisotropic = jnp.array([
    [2.0, 0.5, 0.0],
    [0.5, 1.0, 0.0],
    [0.0, 0.0, 3.0],
])

spd_affine_invariant_distance(identity, scaled)
spd_log_euclidean_distance(identity, scaled)
```

Use affine-invariant when congruence invariance matters (e.g., comparing covariance matrices under different coordinate systems). Use log-Euclidean when speed is the priority and orthogonal invariance suffices.

#### Grassmann Distance

The Grassmann manifold is the space of all k-dimensional subspaces of R^n. Grassmann distance measures how far apart two subspaces are, independent of the specific basis vectors chosen to represent them.

```python
from calibrax.metrics.functional.manifold import grassmann_distance

# Two 2D subspaces of R^4 (represented as orthonormal bases)
u1 = jnp.eye(4, 2)      # span of first two standard basis vectors
u3 = jnp.eye(4, 2, k=2) # span of last two standard basis vectors

grassmann_distance(u1, u1)  # 0.0 (same subspace)
grassmann_distance(u1, u3)  # maximum (orthogonal complement)
```

Applications include subspace tracking in signal processing, PCA comparison, and dimensionality reduction evaluation.

#### Ultrahyperbolic Distance

Ultrahyperbolic spaces generalise hyperbolic geometry to mixed metric signatures with multiple timelike and spacelike dimensions. Signature `(p, q)` means `p` timelike and `q` spacelike dimensions.

```python
from calibrax.metrics.functional.manifold import ultrahyperbolic_distance

# Standard hyperboloid: signature (1, 2)
origin = jnp.array([1.0, 0.0, 0.0])
point = jnp.array([jnp.sqrt(1.25), 0.5, 0.0])

ultrahyperbolic_distance(origin, point, signature=(1, 2))

# Mixed signature (2, 2) for knowledge graph embeddings
origin_4d = jnp.array([1.0, 0.0, 0.0, 0.0])         # satisfies <x,x>_{2,2} = -1
point_4d = jnp.array([jnp.sqrt(1.25), 0.0, 0.5, 0.0])
ultrahyperbolic_distance(origin_4d, point_4d, signature=(2, 2))
```

Signature `(1, n)` recovers standard Lorentz/hyperbolic distance. Mixed signatures model spaces with multiple curvature directions, useful in knowledge graph embedding frameworks like UltraE.

## Example Code

The script constructs valid points on the hyperboloid to demonstrate the ultrahyperbolic distance:

```python
def hyperboloid_point(x: float, y: float) -> jnp.ndarray:
    """Create a point on the (1,2) hyperboloid."""
    t = jnp.sqrt(1.0 + x**2 + y**2)
    return jnp.array([t, x, y])

origin = hyperboloid_point(0.0, 0.0)
p1 = hyperboloid_point(0.5, 0.3)
p2 = hyperboloid_point(1.0, 0.0)

ultrahyperbolic_distance(origin, p1, signature=(1, 2))  # moderate
ultrahyperbolic_distance(origin, p2, signature=(1, 2))  # larger
```

For the (2, 2) signature, points must satisfy `-t1^2 - t2^2 + x1^2 + x2^2 = -1`:

```python
def ultra_point(x1: float, x2: float) -> jnp.ndarray:
    """Create a point on the (2,2) pseudo-hyperboloid."""
    t_sq = 1.0 + x1**2 + x2**2
    t1 = jnp.sqrt(t_sq / 2.0)
    t2 = jnp.sqrt(t_sq / 2.0)
    return jnp.array([t1, t2, x1, x2])
```

## Next Steps

- [Distances and Spaces](distances-and-spaces.md) -- revisit flat and hyperbolic distances for comparison
- [Metric Learning Losses](metric-learning.md) -- losses that operate on the embedding spaces measured by these metrics
- [API Reference: `calibrax.metrics.functional.graph`](../../api-reference/metrics/graph.md) -- full graph metric signatures
- [API Reference: `calibrax.metrics.functional.manifold`](../../api-reference/metrics/manifold.md) -- full manifold metric signatures
