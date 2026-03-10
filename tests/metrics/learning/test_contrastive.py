"""Tests for contrastive metric learning losses."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.learning.contrastive import (
    ContrastiveLoss,
    NTXentLoss,
    TripletMarginLoss,
)


class TestContrastiveLoss:
    """Tests for ContrastiveLoss."""

    def test_positive_pair_loss(self) -> None:
        """Positive pairs (same class) should have distance-based loss."""
        loss_fn = ContrastiveLoss(margin=1.0)
        # 2 embeddings, same label → positive pair
        embeddings = jnp.array([[1.0, 0.0], [0.9, 0.1]])
        labels = jnp.array([0, 0])
        result = loss_fn(embeddings, labels)
        assert float(result) > 0.0

    def test_negative_pair_loss(self) -> None:
        """Negative pairs beyond margin should have zero loss."""
        loss_fn = ContrastiveLoss(margin=0.1)
        # 2 embeddings, different labels, far apart
        embeddings = jnp.array([[0.0, 0.0], [10.0, 10.0]])
        labels = jnp.array([0, 1])
        result = loss_fn(embeddings, labels)
        assert float(result) >= 0.0

    def test_gradient_flow(self) -> None:
        """Gradients should be non-zero."""
        loss_fn = ContrastiveLoss(margin=1.0)
        embeddings = jnp.array([[1.0, 0.0], [0.5, 0.5]])
        labels = jnp.array([0, 0])

        def loss(emb: jax.Array) -> jax.Array:
            return loss_fn(emb, labels)

        grads = jax.grad(loss)(embeddings)
        assert jnp.any(grads != 0.0)

    def test_returns_scalar(self) -> None:
        loss_fn = ContrastiveLoss()
        result = loss_fn(jnp.eye(3), jnp.array([0, 1, 0]))
        assert result.shape == ()


class TestTripletMarginLoss:
    """Tests for TripletMarginLoss."""

    def test_easy_triplet_zero_loss(self) -> None:
        """When anchor-positive is much closer than anchor-negative."""
        loss_fn = TripletMarginLoss(margin=0.2)
        # anchor, positive (close), negative (far)
        embeddings = jnp.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]])
        labels = jnp.array([0, 0, 1])
        result = loss_fn(embeddings, labels)
        assert float(result) == pytest.approx(0.0, abs=0.1)

    def test_hard_triplet_positive_loss(self) -> None:
        """When anchor-negative is closer than anchor-positive + margin."""
        loss_fn = TripletMarginLoss(margin=1.0)
        embeddings = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.3, 0.0]])
        labels = jnp.array([0, 0, 1])
        result = loss_fn(embeddings, labels)
        assert float(result) > 0.0

    def test_gradient_flow(self) -> None:
        loss_fn = TripletMarginLoss(margin=1.0)
        embeddings = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]])
        labels = jnp.array([0, 0, 1])

        def loss(emb: jax.Array) -> jax.Array:
            return loss_fn(emb, labels)

        grads = jax.grad(loss)(embeddings)
        assert jnp.any(grads != 0.0)

    def test_returns_scalar(self) -> None:
        loss_fn = TripletMarginLoss()
        result = loss_fn(jnp.eye(4), jnp.array([0, 0, 1, 1]))
        assert result.shape == ()


class TestNTXentLoss:
    """Tests for NTXentLoss (InfoNCE)."""

    def test_identical_pairs_low_loss(self) -> None:
        """Pairs of identical embeddings should have relatively low loss."""
        loss_fn = NTXentLoss(temperature=0.5)
        # 4 samples, classes 0,0,1,1
        embeddings = jnp.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        labels = jnp.array([0, 0, 1, 1])
        result = loss_fn(embeddings, labels)
        assert float(result) >= 0.0

    def test_gradient_flow(self) -> None:
        loss_fn = NTXentLoss(temperature=0.5)
        embeddings = jnp.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        labels = jnp.array([0, 0, 1, 1])

        def loss(emb: jax.Array) -> jax.Array:
            return loss_fn(emb, labels)

        grads = jax.grad(loss)(embeddings)
        assert jnp.any(grads != 0.0)

    def test_temperature_effect(self) -> None:
        """Lower temperature should produce sharper distribution."""
        embeddings = jnp.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.2, 0.8],
            ]
        )
        labels = jnp.array([0, 0, 1, 1])
        loss_low_t = NTXentLoss(temperature=0.1)
        loss_high_t = NTXentLoss(temperature=10.0)
        result_low = float(loss_low_t(embeddings, labels))
        result_high = float(loss_high_t(embeddings, labels))
        # Different temperatures should produce different loss values
        assert result_low != pytest.approx(result_high, abs=0.01)

    def test_returns_scalar(self) -> None:
        loss_fn = NTXentLoss()
        result = loss_fn(jnp.eye(4), jnp.array([0, 0, 1, 1]))
        assert result.shape == ()
