"""Tests for proxy-based metric learning losses."""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from calibrax.metrics.learning.proxy import ProxyAnchorLoss, ProxyNCALoss


class TestProxyNCALoss:
    """Tests for ProxyNCALoss."""

    def test_gradient_flow(self) -> None:
        loss_fn = ProxyNCALoss(num_classes=3, embedding_dim=4, rngs=nnx.Rngs(0))
        embeddings = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        labels = jnp.array([0, 1])

        def loss(emb: jax.Array) -> jax.Array:
            return loss_fn(emb, labels)

        grads = jax.grad(loss)(embeddings)
        assert jnp.any(grads != 0.0)

    def test_is_nnx_module(self) -> None:
        loss_fn = ProxyNCALoss(num_classes=3, embedding_dim=4, rngs=nnx.Rngs(0))
        assert isinstance(loss_fn, nnx.Module)

    def test_correct_num_proxies(self) -> None:
        loss_fn = ProxyNCALoss(num_classes=5, embedding_dim=8, rngs=nnx.Rngs(0))
        assert loss_fn._proxies.value.shape == (5, 8)

    def test_returns_scalar(self) -> None:
        loss_fn = ProxyNCALoss(num_classes=3, embedding_dim=4, rngs=nnx.Rngs(0))
        result = loss_fn(jnp.ones((2, 4)), jnp.array([0, 1]))
        assert result.shape == ()


class TestProxyAnchorLoss:
    """Tests for ProxyAnchorLoss."""

    def test_gradient_flow(self) -> None:
        loss_fn = ProxyAnchorLoss(num_classes=3, embedding_dim=4, rngs=nnx.Rngs(0))
        embeddings = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        labels = jnp.array([0, 1])

        def loss(emb: jax.Array) -> jax.Array:
            return loss_fn(emb, labels)

        grads = jax.grad(loss)(embeddings)
        assert jnp.any(grads != 0.0)

    def test_is_nnx_module(self) -> None:
        loss_fn = ProxyAnchorLoss(num_classes=3, embedding_dim=4, rngs=nnx.Rngs(0))
        assert isinstance(loss_fn, nnx.Module)

    def test_returns_scalar(self) -> None:
        loss_fn = ProxyAnchorLoss(num_classes=3, embedding_dim=4, rngs=nnx.Rngs(0))
        result = loss_fn(jnp.ones((2, 4)), jnp.array([0, 1]))
        assert result.shape == ()
