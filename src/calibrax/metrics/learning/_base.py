"""Base classes for metric learning losses (Tier 3).

All losses return differentiable JAX arrays for gradient flow.
They learn distance functions via backpropagation on embedding spaces.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import jax
import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON


def _pairwise_distances(embeddings: jax.Array) -> jax.Array:
    """Compute pairwise Euclidean distances between all embedding pairs.

    Args:
        embeddings: (n, dim) embedding matrix.

    Returns:
        (n, n) symmetric distance matrix.
    """
    diff = embeddings[:, None, :] - embeddings[None, :, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1) + _EPSILON)


class Reducer:
    """Reduction strategy for per-element losses.

    Args:
        reduction: One of "mean" or "sum".
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Initialize reducer.

        Args:
            reduction: Reduction method ("mean" or "sum").
        """
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")
        self._reduction = reduction

    def __call__(self, losses: jax.Array) -> jax.Array:
        """Apply reduction to per-element losses.

        Args:
            losses: Array of per-element loss values.

        Returns:
            Reduced scalar loss.
        """
        if self._reduction == "mean":
            return jnp.mean(losses)
        return jnp.sum(losses)


class MetricLearningLoss:
    """Abstract base for metric learning losses.

    Subclasses implement _compute_loss to return per-element losses,
    which are then reduced to a scalar via the Reducer.

    Args:
        reduction: Loss reduction strategy ("mean" or "sum").

    Examples:
        >>> class MyLoss(MetricLearningLoss):
        ...     def _compute_loss(self, embeddings, labels):
        ...         return jnp.zeros(embeddings.shape[0])
        >>> loss_fn = MyLoss()
        >>> loss_fn(embeddings, labels)
    """

    def __init__(self, *, reduction: str = "mean") -> None:
        """Initialize metric learning loss.

        Args:
            reduction: Reduction method ("mean" or "sum").
        """
        self._reducer = Reducer(reduction)

    def __call__(self, embeddings: jax.Array, labels: jax.Array, **kwargs: Any) -> jax.Array:
        """Compute the metric learning loss.

        Args:
            embeddings: Batch of embedding vectors (batch_size, embedding_dim).
            labels: Integer class labels (batch_size,).
            **kwargs: Additional arguments for subclass losses.

        Returns:
            Scalar loss value as a differentiable JAX array.
        """
        per_element = self._compute_loss(embeddings, labels, **kwargs)
        return self._reducer(per_element)

    @abstractmethod
    def _compute_loss(self, embeddings: jax.Array, labels: jax.Array, **kwargs: Any) -> jax.Array:
        """Compute per-element losses before reduction.

        Args:
            embeddings: Batch of embedding vectors.
            labels: Integer class labels.
            **kwargs: Additional arguments.

        Returns:
            Per-element loss array.
        """
        ...
