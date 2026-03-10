# Metric Learning

Tier 3 losses train the *embedding space itself*. They are differentiable
loss functions that work with `jax.grad()` for end-to-end training.

All losses follow the same interface:

```python
from calibrax.metrics.learning import ContrastiveLoss

loss_fn = ContrastiveLoss(margin=1.0)
loss = loss_fn(embeddings, labels)  # Returns a differentiable JAX scalar
```

## Contrastive Losses

### ContrastiveLoss

Pair-based loss with a margin. Pulls same-class pairs together and pushes
different-class pairs apart.

$$
L = y \cdot d^2 + (1-y) \cdot \max(0,\, m - d)^2
$$

```python
from calibrax.metrics.learning import ContrastiveLoss

loss_fn = ContrastiveLoss(margin=1.0)
loss = loss_fn(embeddings, labels)
```

### TripletMarginLoss

Mines all valid (anchor, positive, negative) triplets from the batch.

$$
L = \max(0,\, d(a, p) - d(a, n) + m)
$$

```python
from calibrax.metrics.learning import TripletMarginLoss

loss_fn = TripletMarginLoss(margin=0.2)
loss = loss_fn(embeddings, labels)
```

### NTXentLoss (InfoNCE)

Temperature-scaled softmax cross-entropy on cosine similarities.
The standard loss for self-supervised contrastive learning (SimCLR, MoCo).

```python
from calibrax.metrics.learning import NTXentLoss

loss_fn = NTXentLoss(temperature=0.5)
loss = loss_fn(embeddings, labels)
```

## Angular Margin Losses

These losses add angular penalties in cosine space and maintain *learnable
weight matrices*. They are `nnx.Module` subclasses with trainable parameters.

### ArcFaceLoss

Additive angular margin: $\cos(\theta + m)$ for the target class.

```python
import jax
import flax.nnx as nnx
from calibrax.metrics.learning import ArcFaceLoss

# Angular margin losses need embeddings matching embedding_dim
embeddings = jax.random.normal(jax.random.PRNGKey(0), (4, 128))
labels = jnp.array([0, 0, 1, 1])

loss_fn = ArcFaceLoss(
    num_classes=100,
    embedding_dim=128,
    margin=0.5,
    scale=64.0,
    rngs=nnx.Rngs(0),
)
loss = loss_fn(embeddings, labels)
```

### CosFaceLoss

Additive cosine margin: $\cos(\theta) - m$ for the target class.

```python
from calibrax.metrics.learning import CosFaceLoss

loss_fn = CosFaceLoss(
    num_classes=100,
    embedding_dim=128,
    margin=0.35,
    scale=64.0,
    rngs=nnx.Rngs(0),
)
loss = loss_fn(embeddings, labels)
```

## Proxy-Based Losses

Proxy methods learn *class-representative vectors* in embedding space.
$O(MC)$ complexity instead of $O(N^2)$ for pair-based methods, making them
practical for large class counts.

### ProxyNCALoss

Pushes each sample toward its class proxy and away from other proxies:

```python
from calibrax.metrics.learning import ProxyNCALoss

loss_fn = ProxyNCALoss(
    num_classes=100,
    embedding_dim=128,
    rngs=nnx.Rngs(0),
)
loss = loss_fn(embeddings, labels)
```

### ProxyAnchorLoss

Uses LogSumExp for smooth hard mining with stable gradients:

```python
from calibrax.metrics.learning import ProxyAnchorLoss

loss_fn = ProxyAnchorLoss(
    num_classes=100,
    embedding_dim=128,
    margin=0.1,
    scale=32.0,
    rngs=nnx.Rngs(0),
)
loss = loss_fn(embeddings, labels)
```

## Miners

Miners identify informative triplets from a batch before passing them to a
loss function. This improves training efficiency by focusing on hard examples.

```python
from calibrax.metrics.learning import HardNegativeMiner, SemiHardMiner

# Hard negatives: closest sample from a different class
miner = HardNegativeMiner()
indices = miner.mine(embeddings, labels)
# indices.anchors, indices.positives, indices.negatives

# Semi-hard: farther than positive but within margin
miner = SemiHardMiner(margin=1.0)
indices = miner.mine(embeddings, labels)
```

`MinedIndices` is a frozen dataclass with `anchors`, `positives`, and
`negatives` arrays of matching length.

## Training Loop Integration

All Tier 3 losses return differentiable JAX arrays. Use `nnx.grad()` for
end-to-end training:

```python
import flax.nnx as nnx
import optax
from calibrax.metrics.learning import TripletMarginLoss

# Simple encoder: Linear layer projecting to embedding space
encoder = nnx.Linear(128, 32, rngs=nnx.Rngs(0))
loss_fn = TripletMarginLoss(margin=0.2)
optimizer = nnx.Optimizer(encoder, optax.adam(1e-3), wrt=nnx.Param)

def train_step(encoder, optimizer, batch_x, batch_labels):
    def loss(encoder):
        return loss_fn(encoder(batch_x), batch_labels)

    grads = nnx.grad(loss, wrt=nnx.Param)(encoder)
    optimizer.update(grads)
```

For losses with trainable parameters (ArcFace, CosFace, ProxyNCA,
ProxyAnchor), include the loss module in the gradient computation so its
weights are jointly optimized:

```python
arcface = ArcFaceLoss(num_classes=100, embedding_dim=128, rngs=nnx.Rngs(1))

def train_step(encoder, arcface, optimizer, batch_x, batch_labels):
    def loss(encoder, arcface):
        return arcface(encoder(batch_x), batch_labels)

    grads = nnx.grad(loss)(encoder, arcface)
    optimizer.update(grads)
```

## Choosing a Loss

| Loss | Best for | Complexity |
|------|----------|-----------|
| ContrastiveLoss | Small batches, pairwise similarity | $O(N^2)$ |
| TripletMarginLoss | Baseline metric learning | $O(N^3)$ triplets |
| NTXentLoss | Self-supervised contrastive (SimCLR) | $O(N^2)$ |
| ArcFaceLoss | Face recognition, fine-grained classification | $O(NC)$ |
| CosFaceLoss | Similar to ArcFace, simpler margin | $O(NC)$ |
| ProxyNCALoss | Large class count, fast convergence | $O(MC)$ |
| ProxyAnchorLoss | Hardest mining with smooth gradients | $O(MC)$ |

Where $N$ = batch size, $C$ = number of classes, $M$ = number of proxies.

!!! tip "Reduction"

    All `MetricLearningLoss` subclasses accept a `reduction` parameter
    (`"mean"` or `"sum"`). Default is `"mean"`. Angular margin and proxy
    losses handle reduction internally.

## Next Steps

<div class="grid cards" markdown>

-   **Stateful Metrics**

    ---

    Frozen backbone (Tier 1) and learned (Tier 2) metrics

    [:octicons-arrow-right-24: Stateful Metrics](stateful-metrics.md)

-   **Metric Composition**

    ---

    Group, weight, threshold, and track metrics

    [:octicons-arrow-right-24: Metric Composition](metric-composition.md)

</div>
