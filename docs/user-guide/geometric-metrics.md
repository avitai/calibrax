# Geometric Metrics

Calibrax organizes its distance functions along a geometric hierarchy. Choosing
the right distance for your data requires understanding what geometry your data
lives in. Using Euclidean distance on inherently non-Euclidean data (e.g.,
hierarchical embeddings, covariance matrices) produces geometrically
meaningless results.

## Geometric Hierarchy

Each level generalizes the one above it:

$$
\text{Euclidean} \subset \text{Riemannian} \subset \text{pseudo-Riemannian} \subset \text{Finsler}
$$

- **Euclidean**: Flat space with the standard $L_2$ norm. Distances are
  translation- and rotation-invariant.
- **Riemannian**: Curved space with a positive-definite metric tensor at each
  point. Geodesics replace straight lines. Examples: sphere, Poincare ball,
  SPD manifold.
- **Pseudo-Riemannian**: The metric tensor can be indefinite (mixed-signature).
  Examples: Lorentz hyperboloid, ultrahyperbolic space.
- **Finsler**: The distance need not be symmetric -- $d(x, y) \neq d(y, x)$.
  Examples: Randers distance for directed relationships.

## Constant-Curvature Geometries

The three constant-curvature spaces cover the most common embedding scenarios:

### Euclidean (Zero Curvature)

Standard $L_p$ norms for flat data. Translation-invariant by definition.

```python
from calibrax.metrics.functional.distance import (
    euclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    minkowski_distance,
)

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])

# L2 distance -- invariant under rotation and translation
d = euclidean_distance(x, y)

# L1 distance -- more robust to outliers
d = manhattan_distance(x, y)

# L-infinity distance -- dominated by largest coordinate difference
d = chebyshev_distance(x, y)

# General Lp distance
d = minkowski_distance(x, y, p=3.0)
```

### Spherical / Cosine (Positive Curvature)

For data on the unit sphere -- normalized embeddings, directional features,
angular relationships.

```python
from calibrax.metrics.functional.distance import cosine_distance

embeddings_a = jnp.array([1.0, 0.0, 0.0])
embeddings_b = jnp.array([0.0, 1.0, 0.0])

# Scale-invariant: only measures angle between vectors
d = cosine_distance(embeddings_a, embeddings_b)
```

Cosine distance is invariant under uniform scaling (the `"scale"` invariance
in the registry). It is *not* a true metric -- it violates the triangle
inequality -- but it is symmetric and non-negative.

**When to use cosine distance:**

- Sentence/word embeddings (where magnitude is meaningless)
- Any setting where vectors are L2-normalized before comparison
- Angular similarity tasks (document retrieval, recommendation)

### Hyperbolic (Negative Curvature)

Hyperbolic spaces have exponentially growing volume with radius, making them
natural for embedding hierarchical and tree-like structures.

Calibrax provides two equivalent models of hyperbolic space:

```python
from calibrax.metrics.functional.distance import (
    poincare_distance,
    lorentz_distance,
)

# Poincare ball model -- points inside the unit ball
# Geodesic: d = arccosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
x_ball = jnp.array([0.1, 0.2])
y_ball = jnp.array([0.3, 0.4])
d = poincare_distance(x_ball, y_ball)

# Lorentz (hyperboloid) model -- numerically more stable
# Points on the upper sheet of x0^2 - x1^2 - ... = 1
x_hyp = jnp.array([1.0, 0.0, 0.0])  # origin on hyperboloid
y_hyp = jnp.array([jnp.cosh(0.5), jnp.sinh(0.5), 0.0])
d = lorentz_distance(x_hyp, y_hyp)
```

!!! tip "Poincare vs Lorentz"

    Both models represent the same geometry. The Poincare ball is more
    intuitive (points live in a bounded ball) but suffers from numerical
    instability near the boundary. The Lorentz model avoids this issue at
    the cost of one extra dimension. For training hyperbolic embeddings,
    prefer Lorentz.

**When to use hyperbolic distance:**

- Taxonomy/ontology embeddings
- Hierarchical clustering
- Knowledge graph link prediction
- Any data with tree-like or power-law structure

### Finsler (Asymmetric)

Finsler geometry generalizes Riemannian geometry by dropping the symmetry
requirement. The Randers distance models directed relationships where the
"cost" of traveling from A to B differs from B to A.

```python
from calibrax.metrics.functional.distance import randers_distance

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])
drift_vector = jnp.array([0.1, 0.2, 0.0])  # ||drift|| < 1

# Asymmetric: d(x, y) != d(y, x) in general
# The drift vector breaks symmetry
d = randers_distance(x, y, drift=drift_vector)
```

**When to use Randers distance:**

- Directed graph embeddings
- Asymmetric similarity (e.g., "A is similar to B" but not vice versa)
- Physical systems with a preferred direction (wind, current)

## Correlation-Aware Distances

### Mahalanobis Distance

Euclidean distance weighted by the inverse covariance matrix. Accounts for
feature correlations and differing scales.

```python
from calibrax.metrics.functional.distance import mahalanobis_distance
import jax.numpy as jnp

x = jnp.array([1.0, 2.0])
y = jnp.array([3.0, 4.0])

# Covariance-weighted distance
cov = jnp.array([[2.0, 0.5], [0.5, 1.0]])
d = mahalanobis_distance(x, y, precision_matrix=jnp.linalg.inv(cov))
```

**When to use Mahalanobis:**

- Features with different scales and correlations
- Anomaly detection (distance from cluster center)
- Multivariate Gaussian comparisons

## Manifold Distances

For structured mathematical objects that live on curved manifolds. These
are registered under `domain="manifold"`.

### SPD Matrices

Symmetric Positive Definite matrices appear as covariance matrices in BCI,
diffusion tensors in neuroimaging, and kernel matrices in ML.

```python
from calibrax.metrics.functional.manifold import (
    spd_affine_invariant_distance,
    spd_log_euclidean_distance,
)

# Symmetric positive definite matrices (e.g. covariance matrices)
cov_a = jnp.array([[2.0, 0.5], [0.5, 1.0]])
cov_b = jnp.array([[1.5, 0.3], [0.3, 1.2]])

# Affine-invariant: d(A, B) = d(MAM^T, MBM^T) for any invertible M
# Geometrically exact but O(n^3) per pair
d = spd_affine_invariant_distance(cov_a, cov_b)

# Log-Euclidean: faster approximation via matrix logarithm
# Invariant under orthogonal transformations only
d = spd_log_euclidean_distance(cov_a, cov_b)
```

!!! note "Affine-Invariant vs Log-Euclidean"

    The affine-invariant distance is the geodesic on the SPD manifold and
    is invariant under both `"affine"` and `"congruence"` transformations.
    The log-Euclidean distance is only `"orthogonal"`-invariant but runs
    faster. For most ML applications, log-Euclidean is sufficient.

### Grassmann Manifold

The Grassmann manifold $\text{Gr}(p, n)$ is the space of $p$-dimensional
subspaces of $\mathbb{R}^n$. Distances between subspaces are
basis-independent.

```python
from calibrax.metrics.functional.manifold import grassmann_distance

# (n, p) matrices with orthonormal columns: two 2D subspaces of R^4
U = jnp.eye(4, 2)       # span of first two standard basis vectors
V = jnp.eye(4, 2, k=2)  # span of last two standard basis vectors

# Compare two subspaces given by orthonormal bases
d = grassmann_distance(U, V)
```

**When to use Grassmann distance:**

- Comparing PCA subspaces across datasets
- Feature space drift detection
- Subspace tracking in streaming data

### Stiefel Manifold

The Stiefel manifold $\text{St}(p, n)$ is the space of orthonormal $p$-frames
in $\mathbb{R}^n$. Unlike Grassmann, the distance is basis-*dependent* --
it distinguishes between different orientations of the same subspace.

```python
from calibrax.metrics.functional.manifold import stiefel_distance

# Orthonormal frames: (n, p) matrices with orthonormal columns
frame_a = jnp.eye(4, 2)
frame_b = jnp.eye(4, 2, k=1)

# Compare two orthonormal frames
d = stiefel_distance(frame_a, frame_b)
```

### Pseudo-Riemannian (Mixed Signature)

For embeddings in spaces with mixed-signature metric tensors, such as
knowledge graph embeddings that combine hyperbolic and spherical components.

```python
from calibrax.metrics.functional.manifold import ultrahyperbolic_distance

# Points on pseudo-hyperboloid with <x,x>_{p,q} = -1
# signature (3, 2): 3 timelike + 2 spacelike = 5 dimensions
x = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])  # satisfies -1-0-0+0+0 = -1
y = jnp.array([jnp.sqrt(1.25), 0.0, 0.0, 0.5, 0.0])

# Pseudo-hyperboloid with p positive and q negative dimensions
d = ultrahyperbolic_distance(x, y, signature=(3, 2))
```

## Graph Distances

Distance functions on graph structures, operating on adjacency matrices.
Registered under `domain="graph"`.

### Between-Graph Distances

Compare two different graphs:

```python
from calibrax.metrics.functional.graph import (
    spectral_distance,
    graph_edit_distance_approx,
)
import jax.numpy as jnp

adj_a = jnp.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
adj_b = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

# Laplacian eigenvalue spectrum comparison
# Permutation-invariant (isomorphism-invariant)
d = spectral_distance(adj_a, adj_b)

# Approximate graph edit distance via spectral relaxation
d = graph_edit_distance_approx(adj_a, adj_b)
```

### Within-Graph Distances

Compute a distance matrix for nodes in a single graph:

```python
from calibrax.metrics.functional.graph import (
    resistance_distance,
    shortest_path_distance,
)

adjacency_matrix = jnp.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

# Effective electrical resistance between all node pairs
# Considers ALL paths (not just shortest)
R = resistance_distance(adjacency_matrix)

# Floyd-Warshall shortest paths (hop count)
D = shortest_path_distance(adjacency_matrix)
```

**Spectral vs resistance vs shortest path:**

- Spectral distance captures global structure (eigenvalue distribution)
- Resistance distance accounts for multiple paths and connectivity
- Shortest path counts minimum hops, ignoring alternative routes

## Invariance-Based Selection

The `list_by_invariance()` method lets you find metrics based on what
transformations they are invariant under:

```python
from calibrax.metrics import MetricRegistry

registry = MetricRegistry()

# Metrics unchanged by rotation (Euclidean isometries)
rotation_inv = registry.list_by_invariance("rotation")
for m in rotation_inv:
    print(f"{m.name}: {m.properties.invariances}")
# euclidean_distance: ('rotation', 'translation')

# Metrics unchanged by scaling
scale_inv = registry.list_by_invariance("scale")
# cosine_distance: ('scale',)

# Metrics with permutation invariance (graph metrics)
perm_inv = registry.list_by_invariance("permutation")

# Affine-invariant metrics
affine_inv = registry.list_by_invariance("affine")
# spd_affine_invariant_distance: ('affine', 'congruence')
```

Common invariance tags in the registry:

| Invariance | Meaning | Metrics |
|-----------|---------|---------|
| `translation` | Shift-invariant | Euclidean, Manhattan, Chebyshev, Minkowski |
| `rotation` | Orientation-invariant | Euclidean |
| `scale` | Magnitude-invariant | Cosine distance |
| `permutation` | Node-order invariant | Spectral distance, GED |
| `affine` | Affine-transform invariant | SPD affine-invariant distance |
| `congruence` | Congruence-invariant | SPD affine-invariant distance |
| `orthogonal` | Orthogonal-transform invariant | SPD log-Euclidean, Grassmann |
| `lorentz` | Lorentz-boost invariant | Lorentz distance |

## Choosing a Distance Function

A quick decision guide based on your data:

| Data type | Recommended distance |
|-----------|---------------------|
| Normalized embeddings | Cosine distance |
| Raw feature vectors (uniform scale) | Euclidean distance |
| Features with different scales | Mahalanobis distance |
| Hierarchical / tree-structured data | Poincare or Lorentz distance |
| Directed relationships | Randers distance |
| Covariance matrices | SPD affine-invariant or log-Euclidean |
| PCA subspaces | Grassmann distance |
| Orthonormal frames | Stiefel distance |
| Graph structure (global) | Spectral distance |
| Graph node distances | Resistance or shortest path |
| Point clouds / shapes | Chamfer or Hausdorff (see `domain="geometric"`) |

## Next Steps

<div class="grid cards" markdown>

-   **Metrics Overview**

    ---

    Registry, tiers, and domain reference

    [:octicons-arrow-right-24: Metrics Overview](metrics-overview.md)

-   **Composition & Wrappers**

    ---

    Combine distances with other metrics, add confidence intervals

    [:octicons-arrow-right-24: Metric Composition](metric-composition.md)

</div>
