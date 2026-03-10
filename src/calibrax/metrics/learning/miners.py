"""Metric learning miners for triplet selection.

Miners identify informative triplets (anchor, positive, negative)
from a batch to improve training efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from calibrax.metrics.learning._base import _pairwise_distances


@dataclass(frozen=True, slots=True)
class MinedIndices:
    """Indices of mined triplets.

    Args:
        anchors: Anchor sample indices.
        positives: Positive sample indices (same class as anchor).
        negatives: Negative sample indices (different class from anchor).
    """

    anchors: jnp.ndarray
    positives: jnp.ndarray
    negatives: jnp.ndarray


class HardNegativeMiner:
    """Mines hardest negatives: closest embedding from a different class.

    For each anchor-positive pair, selects the negative with minimum
    distance to the anchor.

    Examples:
        >>> miner = HardNegativeMiner()
        >>> indices = miner.mine(embeddings, labels)
    """

    def mine(self, embeddings: jnp.ndarray, labels: jnp.ndarray) -> MinedIndices:
        """Mine hard negative triplets.

        Args:
            embeddings: (batch_size, dim) embedding vectors.
            labels: (batch_size,) integer class labels.

        Returns:
            MinedIndices with anchor, positive, and hard negative indices.
        """
        n = embeddings.shape[0]
        distances = _pairwise_distances(embeddings)

        same_class = labels[:, None] == labels[None, :]
        diff_class = ~same_class

        anchors_list = []
        positives_list = []
        negatives_list = []

        for i in range(n):
            # Find positives (same class, not self)
            pos_mask = same_class[i] & (jnp.arange(n) != i)
            if not jnp.any(pos_mask):
                continue

            # Find hard negative (closest different class)
            neg_mask = diff_class[i]
            if not jnp.any(neg_mask):
                continue

            # Hard negative: min distance among different class
            neg_distances = jnp.where(neg_mask, distances[i], jnp.inf)
            hard_neg = int(jnp.argmin(neg_distances))

            # Use all positives with this hard negative
            pos_indices = jnp.where(pos_mask)[0]
            for pos_idx in pos_indices:
                anchors_list.append(i)
                positives_list.append(int(pos_idx))
                negatives_list.append(hard_neg)

        if not anchors_list:
            return MinedIndices(
                anchors=jnp.array([], dtype=jnp.int32),
                positives=jnp.array([], dtype=jnp.int32),
                negatives=jnp.array([], dtype=jnp.int32),
            )

        return MinedIndices(
            anchors=jnp.array(anchors_list),
            positives=jnp.array(positives_list),
            negatives=jnp.array(negatives_list),
        )


class SemiHardMiner:
    """Mines semi-hard negatives: farther than positive but within margin.

    Semi-hard negatives satisfy: d(a, p) < d(a, n) < d(a, p) + margin.

    Args:
        margin: Margin for semi-hard selection.

    Examples:
        >>> miner = SemiHardMiner(margin=1.0)
        >>> indices = miner.mine(embeddings, labels)
    """

    def __init__(self, margin: float = 1.0) -> None:
        """Initialize semi-hard miner.

        Args:
            margin: Distance margin for semi-hard criterion.
        """
        self._margin = margin

    def mine(self, embeddings: jnp.ndarray, labels: jnp.ndarray) -> MinedIndices:
        """Mine semi-hard negative triplets.

        Args:
            embeddings: (batch_size, dim) embedding vectors.
            labels: (batch_size,) integer class labels.

        Returns:
            MinedIndices with anchor, positive, and semi-hard negative indices.
        """
        n = embeddings.shape[0]
        distances = _pairwise_distances(embeddings)

        same_class = labels[:, None] == labels[None, :]
        diff_class = ~same_class

        anchors_list = []
        positives_list = []
        negatives_list = []

        for i in range(n):
            pos_mask = same_class[i] & (jnp.arange(n) != i)
            if not jnp.any(pos_mask):
                continue

            neg_mask = diff_class[i]
            if not jnp.any(neg_mask):
                continue

            pos_indices = jnp.where(pos_mask)[0]
            for pos_idx in pos_indices:
                d_ap = distances[i, int(pos_idx)]
                # Semi-hard: d_ap < d_an < d_ap + margin
                semi_hard_mask = (
                    neg_mask & (distances[i] > d_ap) & (distances[i] < d_ap + self._margin)
                )
                if jnp.any(semi_hard_mask):
                    # Pick closest semi-hard negative
                    semi_distances = jnp.where(semi_hard_mask, distances[i], jnp.inf)
                    neg_idx = int(jnp.argmin(semi_distances))
                    anchors_list.append(i)
                    positives_list.append(int(pos_idx))
                    negatives_list.append(neg_idx)

        if not anchors_list:
            return MinedIndices(
                anchors=jnp.array([], dtype=jnp.int32),
                positives=jnp.array([], dtype=jnp.int32),
                negatives=jnp.array([], dtype=jnp.int32),
            )

        return MinedIndices(
            anchors=jnp.array(anchors_list),
            positives=jnp.array(positives_list),
            negatives=jnp.array(negatives_list),
        )
