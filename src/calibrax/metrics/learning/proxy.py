"""Proxy-based metric learning losses.

Proxy methods learn class-representative proxy vectors in embedding space.
O(MC) complexity vs O(N^2) for pair-based methods, making them the most
scalable metric learning approach for large class counts.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON


class ProxyNCALoss(nnx.Module):
    """Proxy Neighborhood Component Analysis loss.

    Learns proxy vectors (one per class) and pushes each sample toward
    its class proxy and away from other proxies.

    L = -log(exp(-d(x, p+)) / sum(exp(-d(x, p-))))

    Args:
        num_classes: Number of classes.
        embedding_dim: Dimensionality of embedding space.
        rngs: RNG streams for parameter initialization.

    Examples:
        >>> loss_fn = ProxyNCALoss(num_classes=10, embedding_dim=128, rngs=nnx.Rngs(0))
        >>> loss = loss_fn(embeddings, labels)
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize ProxyNCA loss.

        Args:
            num_classes: Number of proxy vectors to learn.
            embedding_dim: Embedding dimensionality.
            rngs: RNG streams.
        """
        super().__init__()
        self._proxies = nnx.Param(
            jax.random.normal(rngs.params(), (num_classes, embedding_dim)) * 0.01
        )

    def __call__(self, embeddings: jax.Array, labels: jax.Array) -> jax.Array:
        """Compute ProxyNCA loss.

        Args:
            embeddings: (batch_size, embedding_dim) embedding vectors.
            labels: (batch_size,) integer class labels.

        Returns:
            Scalar loss value.
        """
        # Normalize
        emb_norm = embeddings / (jnp.linalg.norm(embeddings, axis=-1, keepdims=True) + _EPSILON)
        proxy_norm = self._proxies[...] / (
            jnp.linalg.norm(self._proxies[...], axis=-1, keepdims=True) + _EPSILON
        )

        # Distances to all proxies (batch_size, num_classes)
        distances = jnp.sqrt(
            jnp.sum((emb_norm[:, None, :] - proxy_norm[None, :, :]) ** 2, axis=-1) + _EPSILON
        )

        # Negative distances for softmax
        neg_distances = -distances

        # Cross-entropy: push toward positive proxy
        log_probs = jax.nn.log_softmax(neg_distances, axis=-1)
        one_hot = jax.nn.one_hot(labels, self._proxies[...].shape[0])
        loss = -jnp.sum(one_hot * log_probs, axis=-1)

        return jnp.mean(loss)


class ProxyAnchorLoss(nnx.Module):
    """Proxy Anchor loss with smooth hard mining.

    Each proxy acts as an anchor. Aggregates positive/negative samples
    via LogSumExp for smooth hard mining with stable gradients.

    Args:
        num_classes: Number of classes.
        embedding_dim: Dimensionality of embedding space.
        margin: Angular margin. Defaults to 0.1.
        scale: Logit scale factor. Defaults to 32.0.
        rngs: RNG streams for parameter initialization.

    Examples:
        >>> loss_fn = ProxyAnchorLoss(num_classes=10, embedding_dim=128, rngs=nnx.Rngs(0))
        >>> loss = loss_fn(embeddings, labels)
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        *,
        margin: float = 0.1,
        scale: float = 32.0,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize Proxy Anchor loss.

        Args:
            num_classes: Number of proxy vectors.
            embedding_dim: Embedding dimensionality.
            margin: Margin for positive/negative separation.
            scale: Scale factor.
            rngs: RNG streams.
        """
        super().__init__()
        self._margin = margin
        self._scale = scale
        self._proxies = nnx.Param(
            jax.random.normal(rngs.params(), (num_classes, embedding_dim)) * 0.01
        )

    def __call__(self, embeddings: jax.Array, labels: jax.Array) -> jax.Array:
        """Compute Proxy Anchor loss.

        Args:
            embeddings: (batch_size, embedding_dim) embedding vectors.
            labels: (batch_size,) integer class labels.

        Returns:
            Scalar loss value.
        """
        num_classes = self._proxies[...].shape[0]

        # Normalize
        emb_norm = embeddings / (jnp.linalg.norm(embeddings, axis=-1, keepdims=True) + _EPSILON)
        proxy_norm = self._proxies[...] / (
            jnp.linalg.norm(self._proxies[...], axis=-1, keepdims=True) + _EPSILON
        )

        # Cosine similarity (num_classes, batch_size)
        similarity = proxy_norm @ emb_norm.T

        # For each proxy, compute positive and negative losses
        one_hot = jax.nn.one_hot(labels, num_classes)  # (batch, num_classes)
        pos_mask = one_hot.T  # (num_classes, batch)
        neg_mask = 1.0 - pos_mask

        # Positive loss: LogSumExp of -(similarity - margin) for positive samples
        pos_sim = self._scale * (similarity - self._margin)
        pos_loss = jnp.log(1.0 + jnp.sum(jnp.exp(-pos_sim) * pos_mask, axis=1) + _EPSILON)

        # Negative loss: LogSumExp of (similarity + margin) for negative samples
        neg_sim = self._scale * (similarity + self._margin)
        neg_loss = jnp.log(1.0 + jnp.sum(jnp.exp(neg_sim) * neg_mask, axis=1) + _EPSILON)

        # Average over classes that have at least one positive sample
        has_positives = jnp.sum(pos_mask, axis=1) > 0
        num_valid = jnp.maximum(jnp.sum(has_positives), 1.0)

        total_loss = jnp.sum((pos_loss + neg_loss) * has_positives) / num_valid
        return total_loss
