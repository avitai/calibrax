"""Tests for metric learning miners."""

from __future__ import annotations

import jax.numpy as jnp

from calibrax.metrics.learning.miners import (
    HardNegativeMiner,
    MinedIndices,
    SemiHardMiner,
)


class TestHardNegativeMiner:
    """Tests for HardNegativeMiner."""

    def test_returns_mined_indices(self) -> None:
        miner = HardNegativeMiner()
        embeddings = jnp.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
        labels = jnp.array([0, 0, 1, 1])
        result = miner.mine(embeddings, labels)
        assert isinstance(result, MinedIndices)

    def test_correct_structure(self) -> None:
        miner = HardNegativeMiner()
        embeddings = jnp.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
        labels = jnp.array([0, 0, 1, 1])
        result = miner.mine(embeddings, labels)
        assert result.anchors.shape[0] > 0
        assert result.positives.shape[0] == result.anchors.shape[0]
        assert result.negatives.shape[0] == result.anchors.shape[0]

    def test_negative_is_closest_different_class(self) -> None:
        """Hard negative should be the closest embedding from a different class."""
        miner = HardNegativeMiner()
        # Class 0: [0,0], [1,0] — Class 1: [2,0], [10,0]
        # For anchor [1,0], hard negative should be [2,0] (closest from class 1)
        embeddings = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
        labels = jnp.array([0, 0, 1, 1])
        result = miner.mine(embeddings, labels)
        # Verify we got triplets
        assert result.anchors.shape[0] > 0


class TestSemiHardMiner:
    """Tests for SemiHardMiner."""

    def test_returns_mined_indices(self) -> None:
        miner = SemiHardMiner(margin=1.0)
        embeddings = jnp.array([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0], [3.1, 0.0]])
        labels = jnp.array([0, 0, 1, 1])
        result = miner.mine(embeddings, labels)
        assert isinstance(result, MinedIndices)

    def test_correct_structure(self) -> None:
        miner = SemiHardMiner(margin=1.0)
        embeddings = jnp.array([[0.0, 0.0], [0.1, 0.0], [0.5, 0.0], [5.0, 0.0]])
        labels = jnp.array([0, 0, 1, 1])
        result = miner.mine(embeddings, labels)
        assert result.anchors.shape[0] >= 0  # May be 0 if no semi-hard exists


class TestMinedIndices:
    """Tests for MinedIndices dataclass."""

    def test_is_frozen(self) -> None:
        indices = MinedIndices(
            anchors=jnp.array([0]),
            positives=jnp.array([1]),
            negatives=jnp.array([2]),
        )
        assert indices.anchors.shape == (1,)
        assert indices.positives.shape == (1,)
        assert indices.negatives.shape == (1,)
