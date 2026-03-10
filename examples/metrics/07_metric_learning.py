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
# # Metric Learning Losses
#
# | | |
# |---|---|
# | **Level** | Tier 3: Advanced Guide |
# | **Time** | ~30 minutes |
# | **Prerequisites** | `05_composition.py`, embedding space concepts |
# | **Metrics covered** | contrastive_loss, triplet_margin_loss, ntxent_loss, arcface_loss |
# | **Key concepts** | Pairwise losses, triplet mining, temperature scaling, angular margins |
#
# Metric learning losses train embedding spaces where semantically similar
# items are close and dissimilar items are far apart. This tutorial covers
# four loss families -- contrastive, triplet, InfoNCE, and angular margin --
# and demonstrates hard negative mining for improved training.

# %%
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from calibrax.metrics.learning import (
    ArcFaceLoss,
    ContrastiveLoss,
    HardNegativeMiner,
    NTXentLoss,
    SemiHardMiner,
    TripletMarginLoss,
)


# %% [markdown]
# ## 1. Shared Synthetic Data
#
# We create a batch of 8 embeddings in R^16 with 4 classes (2 samples per
# class). This small batch lets us inspect all pairs and triplets directly.


# %%
def create_synthetic_embeddings(
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Create clustered embeddings where same-class points are nearby.

    Returns:
        Tuple of (embeddings, labels, key) with 8 samples in 4 classes.
    """
    embedding_dim = 16
    labels = jnp.array([0, 0, 1, 1, 2, 2, 3, 3])

    # Create class centroids, then add noise to make 2 samples per class
    key, subkey = jax.random.split(key)
    centroids = jax.random.normal(subkey, (4, embedding_dim)) * 2.0

    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, (8, embedding_dim)) * 0.3
    embeddings = centroids[labels] + noise

    print("=== Synthetic Embedding Data ===")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Labels: {labels.tolist()}")
    print("  Classes: 4 (2 samples each)")

    return embeddings, labels, key


# %% [markdown]
# ## 2. Contrastive Loss
#
# The contrastive loss operates on pairs. For positive pairs (same class),
# it penalizes large distances. For negative pairs (different class), it
# penalizes distances smaller than a margin:
#
# L = y * d^2 + (1-y) * max(0, margin - d)^2
#
# The margin controls the minimum separation between negative pairs.


# %%
def demonstrate_contrastive_loss(embeddings: jax.Array, labels: jax.Array) -> None:
    """Show contrastive loss with varying margins."""
    print("\n=== ContrastiveLoss ===")
    print("  Formula: L = y*d^2 + (1-y)*max(0, margin-d)^2")

    for margin in [0.5, 1.0, 2.0]:
        loss_fn = ContrastiveLoss(margin=margin)
        loss_val = loss_fn(embeddings, labels)
        print(f"  margin={margin:.1f}  loss={float(loss_val):.6f}")

    print("  Larger margins push negative pairs further apart.")


# %% [markdown]
# ## 3. Triplet Margin Loss
#
# The triplet loss operates on (anchor, positive, negative) triplets and
# enforces that the anchor is closer to the positive than the negative
# by at least a margin:
#
# L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
#
# All valid triplets are mined from the batch automatically.


# %%
def demonstrate_triplet_loss(embeddings: jax.Array, labels: jax.Array) -> None:
    """Show triplet margin loss with varying margins."""
    print("\n=== TripletMarginLoss ===")
    print("  Formula: L = max(0, d(a,p) - d(a,n) + margin)")

    for margin in [0.1, 0.2, 0.5, 1.0]:
        loss_fn = TripletMarginLoss(margin=margin)
        loss_val = loss_fn(embeddings, labels)
        print(f"  margin={margin:.1f}  loss={float(loss_val):.6f}")

    print("  Triplet loss mines all valid (a, p, n) triplets from the batch.")


# %% [markdown]
# ## 4. NTXent Loss (InfoNCE)
#
# The NT-Xent (Normalized Temperature-scaled Cross Entropy) loss uses
# cosine similarity with temperature scaling. It treats each sample as
# an anchor and its same-class counterpart as the positive, with all
# other samples as negatives:
#
# L = -log( exp(sim(a, p) / t) / sum_k( exp(sim(a, k) / t) ) )
#
# Lower temperature sharpens the softmax, making the loss more sensitive
# to hard negatives.


# %%
def demonstrate_ntxent_loss(embeddings: jax.Array, labels: jax.Array) -> None:
    """Show NTXent loss with varying temperatures."""
    print("\n=== NTXentLoss (InfoNCE) ===")
    print("  Formula: L = -log(exp(sim(a,p)/t) / sum(exp(sim(a,k)/t)))")

    for temperature in [0.1, 0.5, 1.0, 2.0]:
        loss_fn = NTXentLoss(temperature=temperature)
        loss_val = loss_fn(embeddings, labels)
        print(f"  temperature={temperature:.1f}  loss={float(loss_val):.6f}")

    print("  Lower temperature sharpens the softmax distribution.")


# %% [markdown]
# ## 5. ArcFace Loss (Angular Margin)
#
# ArcFace adds an angular margin to the target class logit in cosine
# space. It maintains a learnable weight matrix (class proxies) and
# penalizes cos(theta + m) for the target class. As an `nnx.Module`,
# ArcFace has trainable parameters updated during training alongside
# the embedding network.


# %%
def demonstrate_arcface_loss(embeddings: jax.Array, labels: jax.Array) -> None:
    """Show ArcFace loss with trainable proxy weights."""
    print("\n=== ArcFaceLoss (Angular Margin) ===")

    num_classes = 4
    embedding_dim = embeddings.shape[1]
    arcface = ArcFaceLoss(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        margin=0.5,
        scale=64.0,
        rngs=nnx.Rngs(0),
    )

    loss_val = arcface(embeddings, labels)
    print(f"  Loss (margin=0.5, scale=64): {float(loss_val):.6f}")

    # Inspect trainable parameters
    graph_def, state = nnx.split(arcface)
    param_count = sum(p.size for p in jax.tree.leaves(state))
    print(f"  Trainable parameters: {param_count}")
    print(f"  Weight matrix shape: {arcface._weight[...].shape}")
    print("  ArcFace learns class proxies jointly with the embedding network.")


# %% [markdown]
# ## 6. Comparing Loss Functions
#
# Different losses have different strengths. Contrastive and triplet losses
# operate in Euclidean space; NTXent uses cosine similarity; ArcFace adds
# angular margins. The choice depends on the downstream task and the
# desired embedding geometry.


# %%
def compare_losses(embeddings: jax.Array, labels: jax.Array) -> None:
    """Side-by-side comparison of all four loss families."""
    print("\n=== Loss Comparison (same embeddings) ===")

    contrastive_val = float(ContrastiveLoss(margin=1.0)(embeddings, labels))
    triplet_val = float(TripletMarginLoss(margin=0.2)(embeddings, labels))
    ntxent_val = float(NTXentLoss(temperature=0.5)(embeddings, labels))

    arcface = ArcFaceLoss(
        num_classes=4,
        embedding_dim=embeddings.shape[1],
        margin=0.5,
        scale=64.0,
        rngs=nnx.Rngs(0),
    )
    arcface_val = float(arcface(embeddings, labels))

    print(f"  ContrastiveLoss (margin=1.0):   {contrastive_val:.6f}")
    print(f"  TripletMarginLoss (margin=0.2): {triplet_val:.6f}")
    print(f"  NTXentLoss (temperature=0.5):   {ntxent_val:.6f}")
    print(f"  ArcFaceLoss (margin=0.5):       {arcface_val:.6f}")
    print("  Losses are on different scales -- compare within a family, not across.")


# %% [markdown]
# ## 7. Gradient Verification
#
# All losses must produce nonzero gradients for training to work. We verify
# that `jax.grad` flows through each loss function by checking the gradient
# norm with respect to the embeddings.


# %%
def verify_gradients(embeddings: jax.Array, labels: jax.Array) -> None:
    """Verify that gradients flow through each loss."""
    print("\n=== Gradient Verification ===")

    # Contrastive
    def contrastive_fn(emb: jax.Array) -> jax.Array:
        return ContrastiveLoss(margin=1.0)(emb, labels)

    grads = jax.grad(contrastive_fn)(embeddings)
    grad_norm = float(jnp.linalg.norm(grads))
    nonzero_count = int(float(jnp.sum(jnp.abs(grads) > 1e-10)))
    print(f"  ContrastiveLoss gradient norm: {grad_norm:.6f}")
    print(f"    Non-zero elements: {nonzero_count}/{grads.size}")

    # Triplet
    def triplet_fn(emb: jax.Array) -> jax.Array:
        return TripletMarginLoss(margin=0.2)(emb, labels)

    grad_norm = float(jnp.linalg.norm(jax.grad(triplet_fn)(embeddings)))
    print(f"  TripletMarginLoss gradient norm: {grad_norm:.6f}")

    # NTXent
    def ntxent_fn(emb: jax.Array) -> jax.Array:
        return NTXentLoss(temperature=0.5)(emb, labels)

    grad_norm = float(jnp.linalg.norm(jax.grad(ntxent_fn)(embeddings)))
    print(f"  NTXentLoss gradient norm:        {grad_norm:.6f}")

    # ArcFace (gradient w.r.t. embeddings via nnx.grad)
    arcface = ArcFaceLoss(
        num_classes=4,
        embedding_dim=embeddings.shape[1],
        margin=0.5,
        scale=64.0,
        rngs=nnx.Rngs(0),
    )

    def arcface_fn(model: ArcFaceLoss, emb: jax.Array) -> jax.Array:
        return model(emb, labels)

    arcface_grads = nnx.grad(arcface_fn, argnums=1)(arcface, embeddings)
    grad_norm = float(jnp.linalg.norm(arcface_grads))
    print(f"  ArcFaceLoss gradient norm:       {grad_norm:.6f}")
    print("  All gradient norms > 0 confirms differentiability.")


# %% [markdown]
# ## 8. Hard Negative Mining
#
# Mining selects informative triplets from a batch to improve training
# efficiency. Two strategies:
#
# - **HardNegativeMiner**: selects the closest different-class sample as
#   the negative (hardest examples, fastest convergence, risk of collapse).
# - **SemiHardMiner**: selects negatives that are farther than the positive
#   but within a margin (more stable training).


# %%
def demonstrate_mining(embeddings: jax.Array, labels: jax.Array) -> None:
    """Show hard and semi-hard negative mining."""
    print("\n=== Hard Negative Mining ===")
    miner = HardNegativeMiner()
    mined = miner.mine(embeddings, labels)

    print(f"  HardNegativeMiner: {len(mined.anchors)} triplets mined")
    if len(mined.anchors) > 0:
        print("  First 5 triplets (anchor, positive, negative):")
        for i in range(min(5, len(mined.anchors))):
            a_idx = int(mined.anchors[i])
            p_idx = int(mined.positives[i])
            n_idx = int(mined.negatives[i])
            print(
                f"    ({a_idx}, {p_idx}, {n_idx}) -- "
                f"labels=({int(labels[a_idx])}, {int(labels[p_idx])}, {int(labels[n_idx])})"
            )

        # Verify triplet validity
        all_valid = all(
            labels[int(mined.anchors[i])] == labels[int(mined.positives[i])]
            and labels[int(mined.anchors[i])] != labels[int(mined.negatives[i])]
            for i in range(len(mined.anchors))
        )
        print(f"  All triplets valid: {all_valid}")

    # Show hardness: distance to mined negative
    if len(mined.anchors) > 0:
        sample_anchor = int(mined.anchors[0])
        sample_neg = int(mined.negatives[0])
        dist = float(jnp.linalg.norm(embeddings[sample_anchor] - embeddings[sample_neg]))
        print("\n  Example hard negative:")
        print(f"    Anchor idx={sample_anchor} (label={int(labels[sample_anchor])})")
        print(f"    Hard neg idx={sample_neg} (label={int(labels[sample_neg])})")
        print(f"    Distance: {dist:.4f}")

    # Semi-hard mining
    print("\n=== Semi-Hard Negative Mining ===")
    semi_miner = SemiHardMiner(margin=2.0)
    semi_mined = semi_miner.mine(embeddings, labels)
    print(f"  SemiHardMiner (margin=2.0): {len(semi_mined.anchors)} triplets mined")
    print("  Semi-hard negatives satisfy: d(a,p) < d(a,n) < d(a,p) + margin.")
    print("  This balances informativeness with training stability.")


# %% [markdown]
# ## 9. How Losses Relate to Embedding Quality
#
# Lower loss values generally indicate better-separated embeddings. Here
# we compare loss values on well-clustered vs randomly scattered embeddings
# to illustrate this relationship.


# %%
def demonstrate_loss_vs_quality(key: jax.Array) -> None:
    """Show that losses decrease as embeddings become better separated."""
    print("\n=== Loss vs Embedding Quality ===")
    labels = jnp.array([0, 0, 1, 1, 2, 2, 3, 3])
    embedding_dim = 16
    loss_fn = TripletMarginLoss(margin=0.2)

    # Random embeddings (poor quality)
    key, subkey = jax.random.split(key)
    random_emb = jax.random.normal(subkey, (8, embedding_dim))
    loss_random = float(loss_fn(random_emb, labels))

    # Loosely clustered
    key, subkey = jax.random.split(key)
    centroids = jax.random.normal(subkey, (4, embedding_dim)) * 2.0
    key, subkey = jax.random.split(key)
    loose_emb = centroids[labels] + jax.random.normal(subkey, (8, embedding_dim)) * 0.5
    loss_loose = float(loss_fn(loose_emb, labels))

    # Tightly clustered (high quality)
    key, subkey = jax.random.split(key)
    tight_emb = centroids[labels] + jax.random.normal(subkey, (8, embedding_dim)) * 0.05
    loss_tight = float(loss_fn(tight_emb, labels))

    print(f"  Random embeddings:  TripletLoss = {loss_random:.6f}")
    print(f"  Loosely clustered:  TripletLoss = {loss_loose:.6f}")
    print(f"  Tightly clustered:  TripletLoss = {loss_tight:.6f}")
    print("  Loss decreases as within-class distances shrink relative to between-class.")


# %% [markdown]
# ## Main


# %%
def main() -> None:
    """Run metric learning loss examples."""
    key = jax.random.PRNGKey(42)
    embeddings, labels, key = create_synthetic_embeddings(key)

    demonstrate_contrastive_loss(embeddings, labels)
    demonstrate_triplet_loss(embeddings, labels)
    demonstrate_ntxent_loss(embeddings, labels)
    demonstrate_arcface_loss(embeddings, labels)
    compare_losses(embeddings, labels)
    verify_gradients(embeddings, labels)
    demonstrate_mining(embeddings, labels)
    demonstrate_loss_vs_quality(key)


if __name__ == "__main__":
    main()
