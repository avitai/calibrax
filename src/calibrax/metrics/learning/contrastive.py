"""Contrastive metric learning losses.

Includes pair-based, triplet-based, and InfoNCE losses.
All return differentiable JAX arrays for gradient flow.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON
from calibrax.metrics.learning._base import _pairwise_distances, MetricLearningLoss


class ContrastiveLoss(MetricLearningLoss):
    """Pair-based contrastive loss with margin.

    L = y * d^2 + (1-y) * max(0, margin - d)^2

    where y=1 for positive pairs (same class) and y=0 for negative pairs.

    Args:
        margin: Margin for negative pairs. Defaults to 1.0.
        reduction: Loss reduction ("mean" or "sum").

    Examples:
        >>> loss_fn = ContrastiveLoss(margin=1.0)
        >>> loss = loss_fn(embeddings, labels)
    """

    def __init__(self, margin: float = 1.0, *, reduction: str = "mean") -> None:
        """Initialize contrastive loss.

        Args:
            margin: Distance margin for negative pairs.
            reduction: Reduction method.
        """
        super().__init__(reduction=reduction)
        self._margin = margin

    def _compute_loss(self, embeddings: jax.Array, labels: jax.Array, **kwargs: Any) -> jax.Array:
        """Compute contrastive loss for all pairs.

        Args:
            embeddings: (batch_size, dim) embedding vectors.
            labels: (batch_size,) class labels.

        Returns:
            Per-pair loss array.
        """
        n = embeddings.shape[0]
        distances = _pairwise_distances(embeddings)

        # Positive/negative masks
        same_class = labels[:, None] == labels[None, :]
        # Upper triangle only (avoid duplicates and self-pairs)
        mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)

        pos_loss = same_class * distances**2
        neg_loss = (~same_class) * jnp.maximum(self._margin - distances, 0.0) ** 2

        pair_losses = (pos_loss + neg_loss) * mask
        valid_pairs = jnp.sum(mask)
        return jnp.where(valid_pairs > 0, pair_losses.sum() / valid_pairs, 0.0)[None]


class TripletMarginLoss(MetricLearningLoss):
    """Triplet loss with margin.

    L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    Mines all valid triplets from the batch.

    Args:
        margin: Triplet margin. Defaults to 0.2.
        reduction: Loss reduction ("mean" or "sum").

    Examples:
        >>> loss_fn = TripletMarginLoss(margin=0.2)
        >>> loss = loss_fn(embeddings, labels)
    """

    def __init__(self, margin: float = 0.2, *, reduction: str = "mean") -> None:
        """Initialize triplet margin loss.

        Args:
            margin: Distance margin.
            reduction: Reduction method.
        """
        super().__init__(reduction=reduction)
        self._margin = margin

    def _compute_loss(self, embeddings: jax.Array, labels: jax.Array, **kwargs: Any) -> jax.Array:
        """Compute triplet loss for all valid triplets.

        Args:
            embeddings: (batch_size, dim) embedding vectors.
            labels: (batch_size,) class labels.

        Returns:
            Per-triplet loss array (reduced to scalar via batch mean).
        """
        distances = _pairwise_distances(embeddings)

        n = embeddings.shape[0]
        same_class = labels[:, None] == labels[None, :]

        # For each anchor, find positive and negative distances
        # Anchor-positive: same class, not self
        # Anchor-negative: different class
        ap_mask = same_class & ~jnp.eye(n, dtype=bool)
        an_mask = ~same_class

        # Use broadcasting for all triplets
        ap_distances = distances[:, :, None]  # (n, n, 1)
        an_distances = distances[:, None, :]  # (n, 1, n)

        triplet_loss = jnp.maximum(ap_distances - an_distances + self._margin, 0.0)

        # Valid triplet mask: anchor-positive pair AND anchor-negative pair
        valid_mask = ap_mask[:, :, None] & an_mask[:, None, :]
        triplet_loss = triplet_loss * valid_mask

        num_valid = jnp.maximum(jnp.sum(valid_mask), 1.0)
        return (jnp.sum(triplet_loss) / num_valid)[None]


class NTXentLoss(MetricLearningLoss):
    """Normalized Temperature-scaled Cross Entropy (InfoNCE) loss.

    L = -log(exp(sim(a, p) / t) / sum(exp(sim(a, k) / t)))

    Temperature-scaled softmax cross-entropy on cosine similarities.

    Args:
        temperature: Temperature scaling factor. Defaults to 0.5.
        reduction: Loss reduction ("mean" or "sum").

    Examples:
        >>> loss_fn = NTXentLoss(temperature=0.5)
        >>> loss = loss_fn(embeddings, labels)
    """

    def __init__(self, temperature: float = 0.5, *, reduction: str = "mean") -> None:
        """Initialize NT-Xent loss.

        Args:
            temperature: Temperature parameter.
            reduction: Reduction method.
        """
        super().__init__(reduction=reduction)
        self._temperature = temperature

    def _compute_loss(self, embeddings: jax.Array, labels: jax.Array, **kwargs: Any) -> jax.Array:
        """Compute NT-Xent loss.

        Args:
            embeddings: (batch_size, dim) embedding vectors.
            labels: (batch_size,) class labels.

        Returns:
            Per-anchor loss array.
        """
        # Normalize embeddings
        norms = jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
        normalized = embeddings / (norms + _EPSILON)

        # Cosine similarity matrix
        sim_matrix = normalized @ normalized.T / self._temperature

        n = embeddings.shape[0]
        same_class = labels[:, None] == labels[None, :]
        # Positive mask: same class, not self
        pos_mask = same_class & ~jnp.eye(n, dtype=bool)

        # For numerical stability, subtract max
        sim_matrix = sim_matrix - jnp.max(sim_matrix, axis=1, keepdims=True)

        # Denominator: sum over all except self
        neg_mask = ~jnp.eye(n, dtype=bool)
        exp_sim = jnp.exp(sim_matrix) * neg_mask
        log_denom = jnp.log(jnp.sum(exp_sim, axis=1) + _EPSILON)

        # Numerator: log of positive pair similarities
        pos_sim = sim_matrix * pos_mask
        # Average over positive pairs for each anchor
        num_positives = jnp.maximum(jnp.sum(pos_mask, axis=1), 1.0)
        log_num = jnp.sum(pos_sim, axis=1) / num_positives

        losses = -log_num + log_denom
        return losses
