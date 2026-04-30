# Stateful Metrics

Tier 0 pure functions work when you have matching prediction and target arrays
available in a single call. Many evaluation scenarios require more:
accumulating feature statistics over batches (FID) or applying learned
calibration weights (LPIPS). Tiers 1 and 2 handle these cases. For Tier 3
metric learning losses, see [Metric Learning](metric-learning.md).

## Tier 1: Frozen Backbone Metrics

Tier 1 metrics use pretrained models to extract features, accumulate
statistics across multiple `update()` calls, and produce a final score with
`compute()`.

### Lifecycle

Every `FrozenBackboneMetric` subclass follows the same three-step lifecycle:

```
update(**batch_data)  -->  compute()  -->  reset()
     (repeat)              (final)        (new eval)
```

1. **`update(**kwargs)`** -- Feed a batch. The metric extracts features from
   the frozen backbone and accumulates internal statistics.
2. **`compute()`** -- Return a `dict[str, float]` with the final metric values
   computed from all accumulated data.
3. **`reset()`** -- Clear internal state for a new evaluation run.

### FID (Frechet Inception Distance)

Measures the similarity between real and generated image distributions using
InceptionV3 features. Lower is better.

```python
import jax
from calibrax.metrics.plugins.image import FIDMetric

fid = FIDMetric(feature_dim=2048)

# Simulate accumulating pre-extracted features over batches
dataloader = [(None, None) for _ in range(3)]  # 3 batches
for real_batch, gen_batch in dataloader:
    real_features = jax.random.normal(jax.random.PRNGKey(0), (16, 2048))
    gen_features = jax.random.normal(jax.random.PRNGKey(1), (16, 2048))
    fid.update(real=real_features, generated=gen_features)

result = fid.compute()
print(f"FID: {result['fid']:.2f}")

figure_path = fid.plot(output_dir="figures")
print(f"Saved metric plot to: {figure_path}")

fid.reset()  # Ready for next evaluation
```

!!! note "Feature Extraction"

    `FIDMetric` accepts pre-extracted features (2D arrays) or raw images.
    For raw images, install `calibrax[image]` for InceptionV3 backbone
    support. In testing, pass pre-extracted features directly.

### Plotting Computed Values

Stateful metrics expose `.plot()` for quick scalar summaries. The method calls
`compute()`, uses the publication exporter, and returns the generated path when
`matplotlib` is installed.

```python
metric = FIDMetric(feature_dim=2048)
metric.update(real=real_features, generated=gen_features)

figure_path = metric.plot(output_dir="figures")
```

The method returns `None` if plotting dependencies are unavailable, matching the
publication exporter's other plotting methods.

### Inception Score

Measures both quality (low per-image entropy) and diversity (high marginal
entropy) of generated images. Higher is better.

```python
from calibrax.metrics.plugins.image import InceptionScoreMetric

is_metric = InceptionScoreMetric()

dataloader = [None for _ in range(3)]  # 3 batches
for batch in dataloader:
    class_probs = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(0), (16, 10)), axis=-1)
    is_metric.update(probabilities=class_probs)

result = is_metric.compute()
print(f"IS: {result['inception_score']:.2f}")

is_metric.reset()
```

### BERTScore

Computes precision, recall, and F1 between candidate and reference texts
using frozen BERT token embeddings.

```python
from calibrax.metrics.plugins.text import BERTScoreMetric

bertscore = BERTScoreMetric()

eval_pairs = [
    (jax.random.normal(jax.random.PRNGKey(i), (8, 64)),
     jax.random.normal(jax.random.PRNGKey(i + 10), (8, 64)))
    for i in range(3)
]
for candidate_emb, reference_emb in eval_pairs:
    bertscore.update(
        candidate_embeddings=candidate_emb,
        reference_embeddings=reference_emb,
    )

result = bertscore.compute()
print(f"BERTScore F1: {result['bertscore_f1']:.3f}")
print(f"Precision: {result['bertscore_precision']:.3f}")
print(f"Recall: {result['bertscore_recall']:.3f}")
```

### Writing a Custom Tier 1 Metric

Subclass `FrozenBackboneMetric` and implement three abstract methods:

```python
from calibrax.metrics.stateful import FrozenBackboneMetric
from typing import Any
import jax.numpy as jnp


class MeanFeatureNorm(FrozenBackboneMetric):
    """Track the mean L2 norm of backbone features."""

    def __init__(self) -> None:
        super().__init__(name="mean_feature_norm")
        self._norms: list[float] = []

    def reset(self) -> None:
        self._norms = []

    def _extract_features(self, **kwargs: Any) -> Any:
        # Accept pre-extracted features
        return jnp.asarray(kwargs["features"])

    def _accumulate(self, features: Any) -> None:
        batch_mean_norm = float(jnp.mean(jnp.linalg.norm(features, axis=-1)))
        self._norms.append(batch_mean_norm)

    def _compute_from_accumulated(self) -> dict[str, float]:
        if not self._norms:
            return {"mean_feature_norm": 0.0}
        return {"mean_feature_norm": sum(self._norms) / len(self._norms)}
```

## Tier 2: Learned Metrics

Tier 2 metrics extend `flax.nnx.Module`, giving them trainable parameters
that participate in JAX transformations (`jit`, `grad`, `vmap`).

### LPIPS (Learned Perceptual Image Patch Similarity)

LPIPS uses VGG features with *learned* linear weights trained on human
perceptual similarity judgments. Unlike FID (Tier 1), the calibration
weights are trainable.

```python
import flax.nnx as nnx
from calibrax.metrics.plugins.image import LPIPSMetric

lpips = LPIPSMetric(rngs=nnx.Rngs(0))

# Per-layer VGG features: last dim must match feature_channels (64, 128, 256, 512, 512)
layer1_a, layer1_b = jnp.ones((4, 64)), jnp.ones((4, 64)) * 0.9
layer2_a, layer2_b = jnp.ones((4, 128)), jnp.ones((4, 128)) * 0.9
layer3_a, layer3_b = jnp.ones((4, 256)), jnp.ones((4, 256)) * 0.9
layer4_a, layer4_b = jnp.ones((4, 512)), jnp.ones((4, 512)) * 0.9
layer5_a, layer5_b = jnp.ones((4, 512)), jnp.ones((4, 512)) * 0.9

# Pass per-layer VGG feature differences
lpips.update(
    features_a=[layer1_a, layer2_a, layer3_a, layer4_a, layer5_a],
    features_b=[layer1_b, layer2_b, layer3_b, layer4_b, layer5_b],
)

result = lpips.compute()
print(f"LPIPS: {result['lpips']:.4f}")

lpips.reset()
```

### Writing a Custom Tier 2 Metric

Subclass `LearnedMetric` (which inherits from `nnx.Module`) and add
trainable parameters via `nnx.Param`:

```python
import flax.nnx as nnx
import jax.numpy as jnp
from calibrax.metrics.stateful import LearnedMetric


class LearnedWeightedMSE(LearnedMetric):
    """MSE with learned per-feature importance weights."""

    def __init__(self, num_features: int, *, rngs: nnx.Rngs) -> None:
        super().__init__(name="learned_weighted_mse", rngs=rngs)
        self._feature_weights = nnx.Param(jnp.ones(num_features))
        self._scores: list[float] = []

    def reset(self) -> None:
        self._scores = []

    def update(self, predictions: jnp.ndarray, targets: jnp.ndarray) -> None:
        weights = nnx.softmax(self._feature_weights.value)
        self._scores.append(float(jnp.mean((predictions - targets) ** 2 * weights)))

    def compute(self) -> dict[str, float]:
        if not self._scores:
            return {"learned_weighted_mse": 0.0}
        return {"learned_weighted_mse": sum(self._scores) / len(self._scores)}
```

Because it inherits from `nnx.Module`, its parameters are visible to
`nnx.state()` and can be optimized with standard NNX optimizers.

## When to Use Each Tier

| Scenario | Tier | Class |
|----------|------|-------|
| Comparing arrays of numbers (regression, classification) | 0 | Pure function |
| Evaluating generative models with feature statistics over batches | 1 | `FrozenBackboneMetric` |
| Perceptual similarity where the metric has learned calibration weights | 2 | `LearnedMetric` |
| Training an embedding space | 3 | See [Metric Learning](metric-learning.md) |

Start with Tier 0 pure functions. Move to Tier 1 when evaluation requires
accumulating statistics across batches. Use Tier 2 when the metric itself
needs trainable parameters. Use Tier 3 when training the embedding space is
the objective -- see [Metric Learning](metric-learning.md) for all 7 losses
and 2 miners.

## Optional Dependencies

Tier 1 and 2 plugins require optional extras:

| Extra | Metrics | Install |
|-------|---------|---------|
| `calibrax[image]` | FID, InceptionScore, LPIPS | `uv pip install "calibrax[image]"` |
| `calibrax[text]` | BERTScore | `uv pip install "calibrax[text]"` |

Tier 3 losses and all Tier 0 functional metrics require only core calibrax.

## Next Steps

<div class="grid cards" markdown>

-   **Metric Learning**

    ---

    Contrastive, triplet, angular margin, and proxy-based losses (Tier 3)

    [:octicons-arrow-right-24: Metric Learning](metric-learning.md)

-   **Metric Composition**

    ---

    Group, weight, threshold, and track metrics

    [:octicons-arrow-right-24: Metric Composition](metric-composition.md)

</div>
