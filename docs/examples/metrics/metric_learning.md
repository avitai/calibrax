# Metric Learning Losses

> Ran in 5.2 s on 1 cpu (python 3.12.14, jax 0.11.1, substrax 0.1.27, calibrax 0.1.17), from [`examples/metrics/metric_learning.py`](https://github.com/avitai/calibrax/blob/main/examples/metrics/metric_learning.py) ([notebook](https://github.com/avitai/calibrax/blob/main/examples/metrics/metric_learning.ipynb)). Run it from the repository root:

```bash
source ./activate.sh
uv run python examples/metrics/metric_learning.py
```

| Metadata | Value |
| --- | --- |
| **Tier** | Advanced Guide |
| **Devices** | CPU-compatible |
| **Prerequisites** | The composition example; embedding spaces |

## Overview

Metric learning losses train an embedding space in which similar items sit close together and
dissimilar ones far apart. This example computes calibrax's four loss families on one batch of
toy embeddings: contrastive and triplet losses on Euclidean distances, NTXent (InfoNCE) on cosine
similarity with a temperature, and ArcFace with an angular margin and trainable class proxies. It
checks that gradients flow through each loss, mines hard and semi-hard negatives, and compares
the triplet loss on embeddings of different quality.

## Learning Goals

1. Compute the contrastive loss across margins
2. Compute the triplet margin loss across margins
3. Compute the NTXent loss across temperatures
4. Create an ArcFace loss and inspect its trainable class proxies
5. Verify that gradients flow through every loss
6. Mine hard and semi-hard negatives from a batch
7. Compare the triplet loss on random, loosely and tightly clustered embeddings

## Setup


```python
import jax
import jax.numpy as jnp
from flax import nnx

from calibrax.metrics.learning import (
    ArcFaceLoss,
    ContrastiveLoss,
    HardNegativeMiner,
    NTXentLoss,
    SemiHardMiner,
    TripletMarginLoss,
)
```

## Core Concepts

- The contrastive loss reads pairs: a positive pair (same class, `y = 1`) is penalised by its
  squared distance, and a negative pair by how far it falls inside the margin,
  `L = y * d^2 + (1 - y) * max(0, margin - d)^2`.
- The triplet loss reads (anchor, positive, negative) triplets, every valid one in the batch, and
  asks the anchor to be closer to the positive than to the negative by the margin,
  `L = max(0, d(a, p) - d(a, n) + margin)`.
- NTXent treats each sample as an anchor and its same-class counterpart as the positive, every
  other sample as a negative, on cosine similarity scaled by a temperature:
  `L = -log(exp(sim(a, p) / t) / sum_k exp(sim(a, k) / t))`. A lower temperature sharpens the
  softmax, so hard negatives weigh more.
- ArcFace adds an angular margin to the target class's logit in cosine space against a learnable
  matrix of class proxies. As an `nnx.Module` it is trained jointly with the embedding network.
- A hard-negative miner picks each anchor's closest sample of another class; a semi-hard miner
  picks negatives farther than the positive but within a margin of it, which trains more stably.
- Losses sit on different scales: compare within a family, not across families.

## Implementation

### Step 1: Create clustered embeddings

Eight embeddings in 16 dimensions, two per class: each class has a random centroid, and each
sample adds noise to its centroid. The centroids are spread little enough that the losses'
margins, from 0.1 to 2.0, bind: classes far apart would satisfy every margin, and each loss and
its gradient would then be 0.


```python
LABELS = jnp.array([0, 0, 1, 1, 2, 2, 3, 3])
EMBEDDING_DIM = 16
CENTROID_SCALE = 0.15
NOISE = 0.1


def clustered_embeddings(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Draw two embeddings around each of four class centroids.

    Args:
        key: The key the centroids and noise are drawn from.

    Returns:
        The embeddings, and the key left for later draws.
    """
    key, centroid_key = jax.random.split(key)
    centroids = jax.random.normal(centroid_key, (4, EMBEDDING_DIM)) * CENTROID_SCALE
    key, noise_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, (8, EMBEDDING_DIM)) * NOISE
    return centroids[LABELS] + noise, key
```

### Step 2: Sweep the contrastive margin


```python
CONTRASTIVE_MARGINS = (0.5, 1.0, 2.0)


def contrastive_sweep(embeddings: jax.Array) -> dict[float, float]:
    """Compute the contrastive loss at each margin.

    Args:
        embeddings: The batch's embeddings.

    Returns:
        The loss at each margin.
    """
    return {m: float(ContrastiveLoss(margin=m)(embeddings, LABELS)) for m in CONTRASTIVE_MARGINS}
```

### Step 3: Sweep the triplet margin


```python
TRIPLET_MARGINS = (0.1, 0.2, 0.5, 1.0)


def triplet_sweep(embeddings: jax.Array) -> dict[float, float]:
    """Compute the triplet margin loss at each margin.

    Args:
        embeddings: The batch's embeddings.

    Returns:
        The loss at each margin.
    """
    return {m: float(TripletMarginLoss(margin=m)(embeddings, LABELS)) for m in TRIPLET_MARGINS}
```

### Step 4: Sweep the NTXent temperature


```python
TEMPERATURES = (0.1, 0.5, 1.0, 2.0)


def ntxent_sweep(embeddings: jax.Array) -> dict[float, float]:
    """Compute the NTXent loss at each temperature.

    Args:
        embeddings: The batch's embeddings.

    Returns:
        The loss at each temperature.
    """
    return {t: float(NTXentLoss(temperature=t)(embeddings, LABELS)) for t in TEMPERATURES}
```

### Step 5: Create an ArcFace loss and inspect its proxies

ArcFace holds one proxy per class as trainable parameters.


```python
def arcface_loss() -> ArcFaceLoss:
    """Create the ArcFace loss the example uses.

    Returns:
        An ArcFace loss over four classes, margin 0.5 and scale 64.
    """
    return ArcFaceLoss(
        num_classes=4, embedding_dim=EMBEDDING_DIM, margin=0.5, scale=64.0, rngs=nnx.Rngs(0)
    )


def arcface_summary(embeddings: jax.Array) -> dict[str, float | int | list[int]]:
    """Compute the ArcFace loss and describe its trainable parameters.

    Args:
        embeddings: The batch's embeddings.

    Returns:
        The loss, the number of trainable parameters and the proxy matrix's shape.
    """
    arcface = arcface_loss()
    proxies = jax.tree.leaves(nnx.state(arcface, nnx.Param))
    return {
        "loss": float(arcface(embeddings, LABELS)),
        "trainable parameters": sum(int(p.size) for p in proxies),
        "proxy matrix shape": list(proxies[0].shape),
    }
```

### Step 6: Verify that gradients flow

Each loss is differentiated with respect to the embeddings; ArcFace through `nnx.grad`, as it
is a module.


```python
def gradient_norms(embeddings: jax.Array) -> dict[str, float]:
    """Compute the norm of each loss's gradient with respect to the embeddings.

    Args:
        embeddings: The batch's embeddings.

    Returns:
        Each loss's gradient norm.
    """
    losses = {
        "ContrastiveLoss": ContrastiveLoss(margin=1.0),
        "TripletMarginLoss": TripletMarginLoss(margin=0.2),
        "NTXentLoss": NTXentLoss(temperature=0.5),
    }
    norms = {
        name: float(jnp.linalg.norm(jax.grad(lambda e, loss=loss: loss(e, LABELS))(embeddings)))
        for name, loss in losses.items()
    }

    def arcface_of(model: ArcFaceLoss, batch: jax.Array) -> jax.Array:
        """ArcFace's loss on ``batch``, a function of the model so ``nnx.grad`` can split it."""
        return model(batch, LABELS)

    arcface_grad = nnx.grad(arcface_of, argnums=1)(arcface_loss(), embeddings)
    norms["ArcFaceLoss"] = float(jnp.linalg.norm(arcface_grad))
    return norms
```

### Step 7: Mine hard and semi-hard negatives


```python
SEMI_HARD_MARGIN = 0.2


def mined_triplets(embeddings: jax.Array) -> dict[str, int | bool | list[list[int]]]:
    """Mine hard and semi-hard triplets and check the hard ones.

    Args:
        embeddings: The batch's embeddings.

    Returns:
        The number of triplets each miner finds, whether every hard triplet pairs a same-class
        positive with an other-class negative, and the first five hard triplets.
    """
    hard = HardNegativeMiner().mine(embeddings, LABELS)
    semi_hard = SemiHardMiner(margin=SEMI_HARD_MARGIN).mine(embeddings, LABELS)
    triplets = [
        [int(a), int(p), int(n)]
        for a, p, n in zip(hard.anchors, hard.positives, hard.negatives, strict=True)
    ]
    valid = all(LABELS[a] == LABELS[p] and LABELS[a] != LABELS[n] for a, p, n in triplets)
    return {
        "hard triplets": len(triplets),
        "hard triplets valid": bool(valid),
        "first hard triplets": triplets[:5],
        "semi-hard triplets": len(semi_hard.anchors),
    }
```

### Step 8: Compare embeddings of different quality

The triplet loss on random embeddings, and on new clusters drawn loosely and tightly around one
set of centroids.


```python
RANDOM_SCALE = 0.15
LOOSE_NOISE = 0.15
TIGHT_NOISE = 0.03


def loss_by_quality(key: jax.Array) -> dict[str, float]:
    """Compute the triplet loss on embeddings of three qualities.

    Args:
        key: The key the embeddings are drawn from.

    Returns:
        The loss on random, loosely clustered and tightly clustered embeddings.
    """
    loss = TripletMarginLoss(margin=0.2)
    key, random_key = jax.random.split(key)
    random = jax.random.normal(random_key, (8, EMBEDDING_DIM)) * RANDOM_SCALE
    key, centroid_key = jax.random.split(key)
    centroids = jax.random.normal(centroid_key, (4, EMBEDDING_DIM)) * CENTROID_SCALE
    key, loose_key = jax.random.split(key)
    loose = centroids[LABELS] + jax.random.normal(loose_key, (8, EMBEDDING_DIM)) * LOOSE_NOISE
    _, tight_key = jax.random.split(key)
    tight = centroids[LABELS] + jax.random.normal(tight_key, (8, EMBEDDING_DIM)) * TIGHT_NOISE
    return {
        "Random embeddings": float(loss(random, LABELS)),
        "Loosely clustered": float(loss(loose, LABELS)),
        "Tightly clustered": float(loss(tight, LABELS)),
    }
```

## Results

### The four losses

The classes sit close enough together that the margins bind: the contrastive loss grows with its
margin as more negative pairs fall inside it, and the triplet loss grows with its margin as more
triplets fall short of it. The NTXent loss grows with the temperature, which flattens the softmax
over the negatives.

### Gradients, mining and quality

Every loss has a nonzero gradient with respect to the embeddings, so each can train them. The
hard-negative miner finds a valid triplet for every anchor, and the semi-hard miner keeps only
the negatives within its margin of the positive. The triplet loss falls from random to loosely to
tightly clustered embeddings, as within-class distances shrink against between-class ones.


```python
def print_sweep(title: str, parameter: str, values: dict[float, float]) -> None:
    """Print one loss at each value of a parameter.

    Args:
        title: The table's title.
        parameter: The parameter's name.
        values: The loss at each value.
    """
    print(f"=== {title} ===")
    for value, loss in values.items():
        print(f"  {parameter}={value:.1f}  loss={loss:.6f}")


def main() -> dict[str, float | int | bool | list[int] | list[list[int]]]:
    """Run the example and return its results.

    Returns:
        Each loss sweep, ArcFace's summary, the gradient norms, the mined triplets and the losses
        by embedding quality.
    """
    embeddings, key = clustered_embeddings(jax.random.key(42))
    print(f"Embeddings {embeddings.shape}, labels {LABELS.tolist()}")
    contrastive = contrastive_sweep(embeddings)
    triplet = triplet_sweep(embeddings)
    ntxent = ntxent_sweep(embeddings)
    print_sweep("ContrastiveLoss", "margin", contrastive)
    print_sweep("TripletMarginLoss", "margin", triplet)
    print_sweep("NTXentLoss (InfoNCE)", "temperature", ntxent)
    sections = {
        "ArcFaceLoss (Angular Margin)": arcface_summary(embeddings),
        "Gradient Norms": gradient_norms(embeddings),
        "Negative Mining": mined_triplets(embeddings),
        "Triplet Loss by Embedding Quality": loss_by_quality(key),
    }
    for title, values in sections.items():
        print(f"=== {title} ===")
        for name, value in values.items():
            print(
                f"  {name:28s} {value:.6f}" if isinstance(value, float) else f"  {name:28s} {value}"
            )
    return {
        **{f"contrastive margin {m}": v for m, v in contrastive.items()},
        **{f"triplet margin {m}": v for m, v in triplet.items()},
        **{f"ntxent temperature {t}": v for t, v in ntxent.items()},
        **{name: value for values in sections.values() for name, value in values.items()},
    }


if __name__ == "__main__":
    main()
```

    Embeddings (8, 16), labels [0, 0, 1, 1, 2, 2, 3, 3]
    === ContrastiveLoss ===
      margin=0.5  loss=0.048266
      margin=1.0  loss=0.073710
      margin=2.0  loss=1.180608
    === TripletMarginLoss ===
      margin=0.1  loss=0.000471
      margin=0.2  loss=0.010976
      margin=0.5  loss=0.224624
      margin=1.0  loss=0.724624
    === NTXentLoss (InfoNCE) ===
      temperature=0.1  loss=0.154258
      temperature=0.5  loss=1.244618
      temperature=1.0  loss=1.567968
      temperature=2.0  loss=1.750083
    === ArcFaceLoss (Angular Margin) ===
      loss                         46.815022
      trainable parameters         64
      proxy matrix shape           [16, 4]
    === Gradient Norms ===
      ContrastiveLoss              0.149052
      TripletMarginLoss            0.220947
      NTXentLoss                   1.633766
      ArcFaceLoss                  52.484394
    === Negative Mining ===
      hard triplets                8
      hard triplets valid          True
      first hard triplets          [[0, 1, 3], [1, 0, 7], [2, 3, 7], [3, 2, 7], [4, 5, 0]]
      semi-hard triplets           7
    === Triplet Loss by Embedding Quality ===
      Random embeddings            0.203612
      Loosely clustered            0.063251
      Tightly clustered            0.000000


## Next Steps

- [Manifold and graph metrics](manifold_graph.md): SPD, Grassmann and graph-theoretic
  distances
- [Distances and divergences](distances.md): the distances these losses build on
- [Metric learning guide](../../user-guide/metric-learning.md): training an embedding network with
  these losses
- [Learning API reference](../../api-reference/metrics/learning.md): every loss and miner's
  signature

<!-- substrax-notebooks: {"page": {"body": "a8ef6807b8ea2b9336449c2c72d15e1e3580c8ab74bfe27efbb865544cd502d5", "figures": {}, "record": "7f3b2d39e91bc4c3529a968514d89cba0934df6ce00ff67912dfad9fcd56f7d0"}, "provenance": {"device": "1 cpu", "outputs": [[], [], [], [], [], [], [], [], [], ["text"]], "seconds": 5.2, "sha256": "07ef26e3066b41626d54c1ecafaef5fab8875450cd95377a05a239918b3b192d", "source": "examples/metrics/metric_learning.py", "versions": {"calibrax": "0.1.17", "jax": "0.11.1", "python": "3.12.14", "substrax": "0.1.27"}}} -->
