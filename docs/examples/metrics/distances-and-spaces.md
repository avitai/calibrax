# Distances and Spaces

| | |
|---|---|
| **Level** | Intermediate |
| **Time** | ~15 minutes |
| **Prerequisites** | [Quickstart](quickstart.md), [Regression Metrics](regression-metrics.md) |
| **Format** | Python + Jupyter |

## Overview

Calibrax organises distance and divergence functions along a geometric hierarchy: flat (Euclidean, Manhattan, cosine), hyperbolic (Poincare, Lorentz), distributional (KL, JS, Wasserstein, Sinkhorn), and information-theoretic (entropy, mutual information). This example walks through each family with concrete computations, explaining the mathematical properties that determine when each is appropriate.

These metrics form the foundation for embedding evaluation, distribution comparison, and metric learning. Understanding the distinction between true distances (satisfying the triangle inequality) and divergences (which may be asymmetric) is key to selecting the right tool for a given task.

## What You'll Learn

1. Compute Euclidean, cosine, and Manhattan distances on flat vector spaces
2. Measure hyperbolic distances in both Poincare ball and Lorentz hyperboloid models
3. Compare probability distributions with KL divergence, JS divergence, and Wasserstein distance
4. Evaluate point-cloud similarity with Sinkhorn divergence
5. Quantify uncertainty and dependence with entropy and mutual information

## Files

- **Python Script**: [`examples/metrics/04_distances.py`](https://github.com/avitai/calibrax/blob/main/examples/metrics/04_distances.py)
- **Jupyter Notebook**: [`examples/metrics/04_distances.ipynb`](https://github.com/avitai/calibrax/blob/main/examples/metrics/04_distances.ipynb)

## Quick Start

```bash
source activate.sh && uv run python examples/metrics/04_distances.py
```

## Key Concepts

### Flat Vector Distances

The simplest distance family operates on vectors in Euclidean space.

```python
from calibrax.metrics.functional.distance import (
    euclidean_distance, cosine_distance, manhattan_distance,
)

a = jnp.array([1.0, 0.0, 0.0])
b = jnp.array([0.0, 1.0, 0.0])

euclidean_distance(a, b)  # L2 norm of (a - b)
cosine_distance(a, b)     # 1 - cosine_similarity; 0 = identical, 1 = orthogonal
manhattan_distance(a, b)  # L1 norm of (a - b)
```

**Terminal Output:**
```
=== Vector Distances ===
  a = [1.0, 0.0, 0.0]
  b = [0.0, 1.0, 0.0]
  c = [1.0, 1.0, 0.0]

  Euclidean(a, b) = 1.4142
  Euclidean(a, c) = 1.0000
  Euclidean(b, c) = 1.0000

  Cosine(a, b)    = 1.0000  (orthogonal -> 1.0)
  Cosine(a, c)    = 0.2929  (45 degrees)
  Cosine(a, a)    = 0.0000  (identical -> 0.0)

  Manhattan(a, b) = 2.0000
  Manhattan(a, c) = 1.0000
```

- **Euclidean**: rotation-invariant, sensitive to magnitude.
- **Cosine**: scale-invariant, measures angular separation only.
- **Manhattan**: axis-aligned, more robust to high-dimensional noise.

### Hyperbolic Distances

Hyperbolic space has negative curvature, making it naturally suited for embedding hierarchical structures (trees, taxonomies). Calibrax supports two equivalent models.

**Poincare ball**: points lie inside the unit ball (`||x|| < 1`). Distance grows exponentially as points approach the boundary.

```python
from calibrax.metrics.functional.distance import poincare_distance

origin = jnp.array([0.0, 0.0])
near = jnp.array([0.3, 0.0])
far = jnp.array([0.8, 0.0])

poincare_distance(origin, near)  # moderate
poincare_distance(origin, far)   # much larger -- exponential growth near boundary
```

**Terminal Output:**
```
=== Hyperbolic Distances ===
  Poincare ball model (points must satisfy ||x|| < 1):
    d(origin, near) = 0.6190
    d(origin, far)  = 2.1972
    d(near, far)    = 1.5782
    Distance grows rapidly as points approach the ball boundary.
```

**Lorentz hyperboloid**: the first component is timelike (`x_0 = sqrt(1 + ||x_spatial||^2)`). This model is numerically more stable near the boundary.

```python
from calibrax.metrics.functional.distance import lorentz_distance

p1 = jnp.array([1.0, 0.0, 0.0])  # origin on hyperboloid
spatial = jnp.array([0.5, 0.3])
p2 = jnp.concatenate([jnp.sqrt(1.0 + jnp.sum(spatial**2))[None], spatial])

lorentz_distance(p1, p2)
```

**Terminal Output:**
```
  Lorentz hyperboloid model (first component is timelike):
    d(p1, p2) = 0.5543
    d(p1, p3) = 0.8130
    d(p2, p3) = 0.4014
    Lorentz model is numerically more stable near the boundary.
```

### Distribution Divergences

Divergences measure how different two probability distributions are. Unlike true distances, they may be asymmetric.

```python
from calibrax.metrics.functional.divergence import (
    kl_divergence, js_divergence, wasserstein_1d, sinkhorn_divergence,
)

p = jnp.array([0.4, 0.3, 0.2, 0.1])
q = jnp.array([0.25, 0.25, 0.25, 0.25])

kl_divergence(p, q)  # asymmetric: KL(p||q) != KL(q||p)
js_divergence(p, q)  # symmetric: JS(p,q) == JS(q,p)
```

**Terminal Output:**
```
=== Distribution Divergences ===
  p = [0.4000000059604645, 0.30000001192092896, 0.20000000298023224, 0.10000000149011612]
  q = [0.25, 0.25, 0.25, 0.25] (uniform)
  r = [0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.4000000059604645]

  KL(p || q) = 0.106440
  KL(q || p) = 0.121777  (asymmetric!)
  KL(p || r) = 0.456435

  JS(p, q)   = 0.027866  (symmetric)
  JS(q, p)   = 0.027866  (same value)
  JS(p, r)   = 0.106440
```

- **KL divergence**: measures information lost when `q` is used to approximate `p`. Asymmetric and unbounded.
- **JS divergence**: symmetrised KL, bounded in `[0, log(2)]`. Often preferred for comparing distributions.

### Sample-Based Distances

When you have samples rather than explicit distributions, use Wasserstein or Sinkhorn:

```python
samples_a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
samples_b = jnp.array([2.0, 3.0, 4.0, 5.0, 6.0])

wasserstein_1d(samples_a, samples_b)  # 1D optimal transport

# For multidimensional point clouds, use Sinkhorn divergence
points_x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
points_y = jnp.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
sinkhorn_divergence(points_x, points_y, regularization=0.1)
```

**Terminal Output:**
```
=== Sample-Based Distances ===
  samples_a = [1.0, 2.0, 3.0, 4.0, 5.0]
  samples_b = [2.0, 3.0, 4.0, 5.0, 6.0] (shifted)
  samples_c = [1.0, 1.5, 3.0, 4.5, 5.0] (different shape)

  Wasserstein(a, b) = 1.0000
  Wasserstein(a, c) = 0.2000
  Wasserstein(a, a) = 0.0000

  Sinkhorn(2D cloud X, Y) = 0.499959
  Sinkhorn(X, X)          = 0.000000  (debiased -> ~0)
```

Sinkhorn divergence is a debiased version of the entropic optimal transport cost. It satisfies `S(X, X) = 0` (unlike raw Sinkhorn distance).

### Information-Theoretic Metrics

Entropy and mutual information quantify uncertainty and statistical dependence.

```python
from calibrax.metrics.functional.information import entropy, mutual_information

uniform = jnp.array([0.25, 0.25, 0.25, 0.25])
peaked = jnp.array([0.9, 0.05, 0.03, 0.02])

entropy(uniform)  # maximum for 4 outcomes: log(4)
entropy(peaked)   # low -- most mass on one outcome

# Mutual information from a joint probability table
joint = jnp.array([[0.45, 0.05], [0.05, 0.45]])
mutual_information(joint)  # high -- strong dependence
```

**Terminal Output:**
```
=== Information-Theoretic Metrics ===
  Entropy(uniform)  = 1.3863  (max entropy for 4 outcomes)
  Entropy(peaked)   = 0.4280  (low uncertainty)

  MI(independent) = 0.000000  (~0)
  MI(dependent)   = 0.368064  (strong dependence)
```

## Example Code

The script demonstrates the asymmetry of KL divergence concretely:

```python
p = jnp.array([0.4, 0.3, 0.2, 0.1])
q = jnp.array([0.25, 0.25, 0.25, 0.25])  # uniform

kl_divergence(p, q)  # KL(p || q)
kl_divergence(q, p)  # KL(q || p) -- different value

js_divergence(p, q)  # symmetric
js_divergence(q, p)  # same value
```

## Next Steps

- [Model Evaluation with Composition](model-evaluation.md) -- combine metrics into collections, suites, and quality gates
- [Advanced Manifold and Graph Metrics](advanced-manifold.md) -- SPD, Grassmann, and graph-theoretic distances
- [API Reference: `calibrax.metrics.functional.distance`](../../api-reference/metrics/distance.md) -- full signatures
