# Metrics Overview

Calibrax provides a 4-tier metric system that covers everything from stateless
pure functions to trainable embedding losses. Every metric is registered in a
singleton `MetricRegistry` with rich metadata -- domain, axioms, invariances,
differentiability -- enabling programmatic discovery and filtering.

## The 4-Tier System

Each tier represents a different computational pattern:

| Tier | Class | Pattern | Example |
|------|-------|---------|---------|
| 0 | `MetricTier.PURE_FUNCTION` | `fn(predictions, targets) -> scalar` | MSE, cosine distance |
| 1 | `MetricTier.FROZEN_BACKBONE` | `update() -> compute() -> reset()` | FID, BERTScore |
| 2 | `MetricTier.LEARNED` | `nnx.Module` with trainable weights | LPIPS |
| 3 | `MetricTier.METRIC_LEARNING` | Differentiable loss on embeddings | NTXent, ArcFace |

**Tier 0: Pure functions** are stateless JAX functions. Most are JIT-compatible
and differentiable. Use them when you have predictions and ground truth in
matching arrays.

**Tier 1: Frozen backbone** metrics use pretrained models (InceptionV3, BERT)
to extract features, accumulate statistics across batches, then produce a
single score. The backbone weights are never updated.

**Tier 2: Learned** metrics extend `nnx.Module` with trainable calibration
parameters. LPIPS, for example, learns linear weights on VGG features that
align with human perceptual judgments.

**Tier 3: Metric learning** losses train the embedding space itself. They are
differentiable loss functions used in training loops with `jax.grad()`.

### When to Use Which Tier

- **Comparing arrays of numbers** (regression, classification, distances):
  Tier 0
- **Evaluating generative models** where you need feature statistics over many
  batches: Tier 1
- **Perceptual similarity** where the metric itself has learned parameters:
  Tier 2
- **Training an encoder** to produce good embeddings: Tier 3

## MetricRegistry

All metrics are registered in a singleton `MetricRegistry` at import time.
Query it to discover available metrics by domain, tier, mathematical
properties, or invariance.

### Listing Metrics

```python
from calibrax.metrics import MetricRegistry, MetricTier

registry = MetricRegistry()

# All registered metric names
all_names = registry.list_names()

# All Tier 0 pure functions
tier0 = registry.list_by_tier(MetricTier.PURE_FUNCTION)

# Metrics in a specific domain
regression = registry.list_by_domain("general")
distances = registry.list_by_domain("distance")
```

### Querying by Mathematical Properties

Each `MetricEntry` carries boolean axiom fields that describe the metric's
mathematical guarantees:

| Field | Meaning |
|-------|---------|
| `properties.is_true_metric` | Satisfies identity, symmetry, and triangle inequality |
| `properties.is_symmetric` | `d(x, y) = d(y, x)` |
| `properties.is_proper` | Proper scoring rule -- minimized by the true distribution |
| `properties.is_differentiable` | Compatible with `jax.grad` |
| `properties.is_jit_compatible` | Compatible with `jax.jit` |
| `invariances` | Tuple of transformation groups (e.g., `"rotation"`, `"scale"`) |

```python
# True metrics (satisfy metric space axioms)
true_metrics = registry.list_true_metrics()
for m in true_metrics:
    print(f"{m.name}: {m.properties.invariances}")

# Proper scoring rules (for probabilistic calibration)
proper = registry.list_proper_scoring_rules()

# JIT-compatible metrics (safe inside jax.jit)
jit_safe = registry.list_jit_compatible()

# Metrics invariant under rotation
rotation_inv = registry.list_by_invariance("rotation")
```

### Retrieving a Metric Function

```python
# Get the entry with all metadata
entry = registry.get("euclidean_distance")
print(entry.domain)        # "distance"
print(entry.properties.is_true_metric) # True
print(entry.properties.invariances)   # ("rotation", "translation")

# Get just the callable
fn = registry.get_function("mse")
value = fn(predictions, targets)
```

### Batch Computation

`calculate_all` computes multiple Tier 0 metrics in one call:

```python
from calibrax.metrics import calculate_all

# Default: all general-domain metrics
results = calculate_all(predictions, targets)
# {"mse": 0.01, "mae": 0.05, "rmse": 0.1, ...}

# Specific subset
results = calculate_all(predictions, targets, metrics=["mse", "mae", "r_squared"])
```

## Registering Custom Metrics

Use the `@register_metric` decorator to add your own functions to the registry:

```python
from calibrax.metrics import register_metric, MetricProperties
from calibrax.core.models import MetricDirection

@register_metric(
    "weighted_mse",
    domain="general",
    direction=MetricDirection.LOWER,
    description="Sample-weighted mean squared error",
    properties=MetricProperties(
        is_symmetric=True,
        invariances=("translation",),
    ),
)
def weighted_mse(predictions, targets, *, weights=None):
    import jax.numpy as jnp
    diff = jnp.asarray(predictions) - jnp.asarray(targets)
    if weights is not None:
        return float(jnp.mean(jnp.asarray(weights) * diff**2))
    return float(jnp.mean(diff**2))
```

## Domain Reference

The registry organizes metrics into 17 domains. Each domain groups metrics
with a shared evaluation context.

| Domain | Count | Examples |
|--------|-------|---------|
| `general` | 12 | MSE, MAE, RMSE, R-squared, Huber, SMAPE |
| `classification` | 12 | Accuracy, F1, ROC-AUC, Matthews correlation |
| `calibration` | 5 | Brier score, ECE, MCE, adaptive ECE |
| `segmentation` | 3 | IoU, Dice coefficient, pixel accuracy |
| `distance` | 11 | Euclidean, cosine, Poincare, Lorentz, Mahalanobis |
| `divergence` | 13 | KL, JS, Wasserstein, MMD, Sinkhorn, Bregman |
| `information` | 5 | Entropy, cross-entropy, mutual information |
| `ranking` | 8 | NDCG, MAP, MRR, precision@k, recall@k |
| `statistical` | 5 | Pearson, Spearman, Kendall, concordance |
| `clustering` | 7 | Adjusted Rand, silhouette, Davies-Bouldin |
| `fairness` | 4 | Demographic parity, equalized odds, disparate impact |
| `image` | 4 | PSNR, SSIM, MS-SSIM, Vendi score |
| `text` | 5 | BLEU, ROUGE-N, ROUGE-L, perplexity, distinct-N |
| `audio` | 3 | Spectral convergence, MCD, SNR |
| `geometric` | 4 | Chamfer, Hausdorff, Earth Mover's distance |
| `graph` | 4 | Spectral distance, resistance, shortest path, GED |
| `manifold` | 5 | SPD affine-invariant, log-Euclidean, Grassmann, Stiefel |

!!! tip "Domain-Based Collection"

    Use `MetricCollection.from_registry(domain="classification")` to build a
    collection of all Tier 0 metrics for a domain, or
    `MetricSuite.from_registry_domains()` to get one group per domain. See
    [Metric Composition](metric-composition.md) for details.

## Input Signatures

Metrics expect different input shapes depending on their `MetricSignature`:

| Signature | Inputs | Used by |
|-----------|--------|---------|
| `PREDICTIONS_TARGETS` | Two matching arrays | Regression, classification, distance |
| `SAMPLES` | Two sample sets (different sizes OK) | Wasserstein, MMD, Hausdorff |
| `FEATURES_LABELS` | Feature matrix + label vector | Clustering metrics |
| `SINGLE_INPUT` | One array or matrix | Entropy, resistance distance |
| `CUSTOM` | Varies per metric | BLEU (token lists), fairness (group arrays) |

## Next Steps

<div class="grid cards" markdown>

-   **Geometric Metrics**

    ---

    Distance functions across Euclidean, spherical, hyperbolic, and
    manifold geometries

    [:octicons-arrow-right-24: Geometric Metrics](geometric-metrics.md)

-   **Composition & Wrappers**

    ---

    Group, weight, threshold, and track metrics

    [:octicons-arrow-right-24: Metric Composition](metric-composition.md)

-   **Stateful Metrics**

    ---

    Frozen backbone, learned, and metric learning losses

    [:octicons-arrow-right-24: Stateful Metrics](stateful-metrics.md)

</div>
