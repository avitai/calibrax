"""Angular margin metric learning losses.

ArcFace and CosFace losses with learnable weight matrices.
Both are nnx.Module subclasses with trainable parameters.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON, _EPSILON_CLIP


class ArcFaceLoss(nnx.Module):
    """ArcFace loss with additive angular margin.

    Adds an angular margin penalty to the target class logit in cosine space:
    cos(theta + m) for the target class, cos(theta) for other classes.

    Args:
        num_classes: Number of target classes.
        embedding_dim: Dimensionality of input embeddings.
        margin: Angular margin in radians. Defaults to 0.5.
        scale: Logit scaling factor. Defaults to 64.0.
        rngs: RNG streams for parameter initialization.

    Examples:
        >>> loss_fn = ArcFaceLoss(num_classes=10, embedding_dim=128, rngs=nnx.Rngs(0))
        >>> loss = loss_fn(embeddings, labels)
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        *,
        margin: float = 0.5,
        scale: float = 64.0,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize ArcFace loss.

        Args:
            num_classes: Number of classes.
            embedding_dim: Embedding dimensionality.
            margin: Angular margin in radians.
            scale: Logit scale factor.
            rngs: RNG streams.
        """
        super().__init__()
        self._margin = margin
        self._scale = scale
        self._weight = nnx.Param(
            jax.random.normal(rngs.params(), (embedding_dim, num_classes)) * 0.01
        )

    def __call__(self, embeddings: jax.Array, labels: jax.Array) -> jax.Array:
        """Compute ArcFace loss.

        Args:
            embeddings: (batch_size, embedding_dim) normalized embeddings.
            labels: (batch_size,) integer class labels.

        Returns:
            Scalar loss value.
        """
        # Normalize embeddings and weights
        emb_norm = embeddings / (jnp.linalg.norm(embeddings, axis=-1, keepdims=True) + _EPSILON)
        w_norm = self._weight[...] / (
            jnp.linalg.norm(self._weight[...], axis=0, keepdims=True) + _EPSILON
        )

        # Cosine similarity
        cos_theta = emb_norm @ w_norm
        cos_theta = jnp.clip(cos_theta, -1.0 + _EPSILON_CLIP, 1.0 - _EPSILON_CLIP)

        # ArcFace: cos(theta + m) for target class
        theta = jnp.arccos(cos_theta)
        one_hot = jax.nn.one_hot(labels, cos_theta.shape[-1])
        target_logits = jnp.cos(theta + self._margin * one_hot)

        # Scale and cross-entropy
        logits = target_logits * self._scale
        return jnp.mean(-jnp.sum(one_hot * jax.nn.log_softmax(logits, axis=-1), axis=-1))


class CosFaceLoss(nnx.Module):
    """CosFace loss with additive cosine margin.

    Subtracts a margin from the target class cosine similarity:
    cos(theta) - m for the target class.

    Args:
        num_classes: Number of target classes.
        embedding_dim: Dimensionality of input embeddings.
        margin: Cosine margin. Defaults to 0.35.
        scale: Logit scaling factor. Defaults to 64.0.
        rngs: RNG streams for parameter initialization.

    Examples:
        >>> loss_fn = CosFaceLoss(num_classes=10, embedding_dim=128, rngs=nnx.Rngs(0))
        >>> loss = loss_fn(embeddings, labels)
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        *,
        margin: float = 0.35,
        scale: float = 64.0,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize CosFace loss.

        Args:
            num_classes: Number of classes.
            embedding_dim: Embedding dimensionality.
            margin: Cosine margin.
            scale: Logit scale factor.
            rngs: RNG streams.
        """
        super().__init__()
        self._margin = margin
        self._scale = scale
        self._weight = nnx.Param(
            jax.random.normal(rngs.params(), (embedding_dim, num_classes)) * 0.01
        )

    def __call__(self, embeddings: jax.Array, labels: jax.Array) -> jax.Array:
        """Compute CosFace loss.

        Args:
            embeddings: (batch_size, embedding_dim) normalized embeddings.
            labels: (batch_size,) integer class labels.

        Returns:
            Scalar loss value.
        """
        emb_norm = embeddings / (jnp.linalg.norm(embeddings, axis=-1, keepdims=True) + _EPSILON)
        w_norm = self._weight[...] / (
            jnp.linalg.norm(self._weight[...], axis=0, keepdims=True) + _EPSILON
        )

        cos_theta = emb_norm @ w_norm
        one_hot = jax.nn.one_hot(labels, cos_theta.shape[-1])
        target_logits = cos_theta - self._margin * one_hot

        logits = target_logits * self._scale
        return jnp.mean(-jnp.sum(one_hot * jax.nn.log_softmax(logits, axis=-1), axis=-1))
