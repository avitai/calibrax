"""Tests for angular margin metric learning losses."""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from calibrax.metrics.learning.angular import ArcFaceLoss, CosFaceLoss


class TestArcFaceLoss:
    """Tests for ArcFaceLoss."""

    def test_is_nnx_module(self) -> None:
        loss_fn = ArcFaceLoss(num_classes=5, embedding_dim=8, rngs=nnx.Rngs(0))
        assert isinstance(loss_fn, nnx.Module)

    def test_gradient_flow(self) -> None:
        loss_fn = ArcFaceLoss(num_classes=3, embedding_dim=4, rngs=nnx.Rngs(0))
        embeddings = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        labels = jnp.array([0, 1])

        def loss(emb: jax.Array) -> jax.Array:
            return loss_fn(emb, labels)

        grads = jax.grad(loss)(embeddings)
        assert jnp.any(grads != 0.0)

    def test_margin_effect(self) -> None:
        """Larger margin should increase loss for misaligned embeddings."""
        embeddings = jnp.array([[1.0, 0.0, 0.0, 0.0]])
        labels = jnp.array([0])
        loss_small = ArcFaceLoss(num_classes=3, embedding_dim=4, margin=0.1, rngs=nnx.Rngs(42))
        loss_large = ArcFaceLoss(num_classes=3, embedding_dim=4, margin=0.8, rngs=nnx.Rngs(42))
        val_small = float(loss_small(embeddings, labels))
        val_large = float(loss_large(embeddings, labels))
        # Different margins should produce different values
        assert val_small != val_large

    def test_returns_scalar(self) -> None:
        loss_fn = ArcFaceLoss(num_classes=3, embedding_dim=4, rngs=nnx.Rngs(0))
        result = loss_fn(jnp.ones((2, 4)), jnp.array([0, 1]))
        assert result.shape == ()


class TestCosFaceLoss:
    """Tests for CosFaceLoss."""

    def test_is_nnx_module(self) -> None:
        loss_fn = CosFaceLoss(num_classes=5, embedding_dim=8, rngs=nnx.Rngs(0))
        assert isinstance(loss_fn, nnx.Module)

    def test_gradient_flow(self) -> None:
        loss_fn = CosFaceLoss(num_classes=3, embedding_dim=4, rngs=nnx.Rngs(0))
        embeddings = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        labels = jnp.array([0, 1])

        def loss(emb: jax.Array) -> jax.Array:
            return loss_fn(emb, labels)

        grads = jax.grad(loss)(embeddings)
        assert jnp.any(grads != 0.0)

    def test_returns_scalar(self) -> None:
        loss_fn = CosFaceLoss(num_classes=3, embedding_dim=4, rngs=nnx.Rngs(0))
        result = loss_fn(jnp.ones((2, 4)), jnp.array([0, 1]))
        assert result.shape == ()
