# Metric Learning Losses

| | |
|---|---|
| **Level** | Advanced |
| **Time** | ~15 minutes |
| **Prerequisites** | [Quickstart](quickstart.md), [Distances and Spaces](distances-and-spaces.md), [Model Evaluation](model-evaluation.md) |
| **Format** | Python + Jupyter |

## Overview

Metric learning trains embedding spaces where semantically similar items are close together and dissimilar items are far apart. Calibrax provides seven differentiable loss functions -- contrastive, triplet margin, NTXent (InfoNCE), ArcFace, CosFace, ProxyNCA, and ProxyAnchor -- along with hard negative and semi-hard miners. All losses are JIT-compatible and produce gradients suitable for training with JAX/Flax.

This example computes each loss on a shared batch of embeddings, verifies gradient flow with `jax.grad()`, and mines hard negatives for triplet training. It shows how hyperparameters (margin, temperature, scale) affect loss behaviour.

## What You'll Learn

1. Compute contrastive loss on embedding pairs with configurable margin
2. Apply triplet margin loss with anchor/positive/negative selection
3. Use NTXent (InfoNCE) for self-supervised contrastive learning
4. Train angular-margin classifiers with ArcFace loss (Flax NNX module)
5. Verify gradient flow through each loss with `jax.grad()` and `nnx.grad()`
6. Mine hard negatives with `HardNegativeMiner`

## Files

- **Python Script**: [`examples/metrics/07_metric_learning.py`](https://github.com/avitai/calibrax/blob/main/examples/metrics/07_metric_learning.py)
- **Jupyter Notebook**: [`examples/metrics/07_metric_learning.ipynb`](https://github.com/avitai/calibrax/blob/main/examples/metrics/07_metric_learning.ipynb)

## Quick Start

```bash
source activate.sh && uv run python examples/metrics/07_metric_learning.py
```

## Key Concepts

### ContrastiveLoss

Contrastive loss pulls same-class pairs together and pushes different-class pairs apart by at least `margin`. Given embeddings and integer labels, it forms all valid pairs within the batch.

```python
from calibrax.metrics.learning import ContrastiveLoss

contrastive = ContrastiveLoss(margin=1.0)
loss = contrastive(embeddings, labels)
```

Larger margins enforce wider separation between classes but require embeddings to be more spread out to satisfy the constraint.

### TripletMarginLoss

Triplet loss operates on (anchor, positive, negative) triplets. It enforces that the anchor-positive distance is smaller than the anchor-negative distance by at least `margin`.

```python
from calibrax.metrics.learning import TripletMarginLoss

triplet = TripletMarginLoss(margin=0.2)
loss = triplet(embeddings, labels)
```

The loss automatically mines triplets from the batch using the provided labels.

### NTXentLoss (InfoNCE)

NTXent is the normalised temperature-scaled cross-entropy loss used in SimCLR and other self-supervised frameworks. It treats each sample's positive pair as the target class in a softmax over all negatives in the batch.

```python
from calibrax.metrics.learning import NTXentLoss

ntxent = NTXentLoss(temperature=0.5)
loss = ntxent(embeddings, labels)
```

Temperature controls the sharpness of the softmax distribution. Lower values produce sharper distributions that focus more on the hardest negatives.

### ArcFaceLoss

ArcFace adds an angular margin penalty in the cosine similarity space. It is an `nnx.Module` with a learnable weight matrix, making it suitable for large-scale face recognition and fine-grained classification.

```python
import flax.nnx as nnx
from calibrax.metrics.learning import ArcFaceLoss

# ArcFace needs embeddings matching embedding_dim
embeddings = jax.random.normal(jax.random.PRNGKey(0), (4, 16))
labels = jnp.array([0, 0, 1, 1])

arcface = ArcFaceLoss(
    num_classes=4,
    embedding_dim=16,
    margin=0.5,
    scale=64.0,
    rngs=nnx.Rngs(0),
)
loss = arcface(embeddings, labels)

# Inspect trainable parameters
graph_def, state = nnx.split(arcface)
param_count = sum(p.size for p in jax.tree.leaves(state))
```

### Gradient Verification

All losses support differentiation. Pure-function losses (contrastive, triplet, NTXent) work with `jax.grad()`. The ArcFace module uses `nnx.grad()`:

```python
# Pure-function loss
grad_fn = jax.grad(lambda emb: ContrastiveLoss(margin=1.0)(emb, labels))
grads = grad_fn(embeddings)

# NNX module loss
arcface_grad_fn = nnx.grad(
    lambda model, emb: model(emb, labels), argnums=1
)
arcface_grads = arcface_grad_fn(arcface, embeddings)
```

### HardNegativeMiner

Hard negative mining selects the closest different-class sample for each anchor, producing the most informative triplets for training.

```python
from calibrax.metrics.learning import HardNegativeMiner

miner = HardNegativeMiner()
mined = miner.mine(embeddings, labels)

# mined.anchors, mined.positives, mined.negatives -- index arrays
# All triplets satisfy: label[anchor] == label[positive] != label[negative]
```

## Example Code

The gradient verification section confirms that gradients flow correctly through each loss:

```python
def contrastive_loss_fn(emb: jax.Array) -> jax.Array:
    return ContrastiveLoss(margin=1.0)(emb, labels)

grads = jax.grad(contrastive_loss_fn)(embeddings)
grad_norm = float(jnp.linalg.norm(grads))
nonzero_count = int(jnp.sum(jnp.abs(grads) > 1e-10))
print(f"Gradient norm: {grad_norm:.6f}")
print(f"Non-zero elements: {nonzero_count}/{grads.size}")
```

The miner verification validates that all produced triplets are semantically correct:

```python
mined = miner.mine(embeddings, labels)
all_valid = all(
    labels[int(mined.anchors[i])] == labels[int(mined.positives[i])]
    and labels[int(mined.anchors[i])] != labels[int(mined.negatives[i])]
    for i in range(len(mined.anchors))
)
```

## Next Steps

- [Advanced Manifold and Graph Metrics](advanced-manifold.md) -- SPD, Grassmann, and graph-theoretic distances
- [Distances and Spaces](distances-and-spaces.md) -- the distance functions underlying metric learning losses
- [API Reference: `calibrax.metrics.learning`](../../api-reference/metrics/learning.md) -- full loss function signatures and parameters
