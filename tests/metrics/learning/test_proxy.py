"""Tests for proxy-based metric learning losses."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from calibrax.metrics.learning.proxy import ProxyAnchorLoss, ProxyNCALoss


@pytest.mark.parametrize("loss_class", [ProxyNCALoss, ProxyAnchorLoss])
def test_one_trainable_proxy_per_class(loss_class: type[ProxyNCALoss | ProxyAnchorLoss]) -> None:
    """The trainable state, as an optimizer reads it, is one embedding-sized proxy per class."""
    loss_fn = loss_class(num_classes=5, embedding_dim=8, rngs=nnx.Rngs(0))
    params = jax.tree.leaves(nnx.state(loss_fn, nnx.Param))
    assert [leaf.shape for leaf in params] == [(5, 8)]


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
