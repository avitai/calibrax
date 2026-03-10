"""Tests for image backbone plugins (FID, Inception Score, LPIPS).

All tests use pre-extracted features / mock probabilities to avoid
requiring InceptionV3 weights in CI.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from calibrax.core.protocols import StatefulMetricProtocol
from calibrax.metrics.plugins.image import FIDMetric, InceptionScoreMetric, LPIPSMetric


class TestFIDMetric:
    """Tests for FIDMetric."""

    def test_identical_distributions(self) -> None:
        fid = FIDMetric()
        features = jnp.ones((50, 64))
        fid.update(real=features, generated=features)
        result = fid.compute()
        assert result["fid"] == pytest.approx(0.0, abs=1e-3)

    def test_different_distributions(self) -> None:
        fid = FIDMetric()
        real = jnp.zeros((50, 64))
        generated = jnp.ones((50, 64))
        fid.update(real=real, generated=generated)
        result = fid.compute()
        assert result["fid"] > 0.0

    def test_reset_clears_state(self) -> None:
        fid = FIDMetric()
        features = jnp.ones((10, 32))
        fid.update(real=features, generated=features)
        fid.reset()
        result = fid.compute()
        assert result["fid"] == float("inf")

    def test_multiple_batches(self) -> None:
        fid = FIDMetric()
        batch1 = jnp.ones((10, 32))
        batch2 = jnp.ones((15, 32)) * 1.01
        fid.update(real=batch1, generated=batch1)
        fid.update(real=batch2, generated=batch2)
        result = fid.compute()
        assert result["fid"] == pytest.approx(0.0, abs=0.1)

    def test_compute_without_update(self) -> None:
        fid = FIDMetric()
        result = fid.compute()
        assert result["fid"] == float("inf")

    def test_conforms_to_protocol(self) -> None:
        fid = FIDMetric()
        assert isinstance(fid, StatefulMetricProtocol)


class TestInceptionScoreMetric:
    """Tests for InceptionScoreMetric."""

    def test_confident_diverse(self) -> None:
        """High IS when predictions are confident and diverse."""
        is_metric = InceptionScoreMetric()
        # Each sample is confident (one-hot) but different classes
        probs = jnp.eye(10)  # 10 samples, each confident in a different class
        is_metric.update(probabilities=probs)
        result = is_metric.compute()
        assert result["inception_score"] > 5.0

    def test_uniform_predictions(self) -> None:
        """Low IS when all predictions are uniform."""
        is_metric = InceptionScoreMetric()
        probs = jnp.ones((20, 10)) / 10
        is_metric.update(probabilities=probs)
        result = is_metric.compute()
        assert result["inception_score"] == pytest.approx(1.0, abs=0.1)

    def test_reset(self) -> None:
        is_metric = InceptionScoreMetric()
        probs = jnp.eye(5)
        is_metric.update(probabilities=probs)
        is_metric.reset()
        result = is_metric.compute()
        assert result["inception_score"] == 0.0

    def test_compute_without_update(self) -> None:
        is_metric = InceptionScoreMetric()
        result = is_metric.compute()
        assert result["inception_score"] == 0.0

    def test_conforms_to_protocol(self) -> None:
        is_metric = InceptionScoreMetric()
        assert isinstance(is_metric, StatefulMetricProtocol)


class TestLPIPSMetric:
    """Tests for LPIPSMetric."""

    def test_creation_with_rngs(self) -> None:
        lpips = LPIPSMetric(rngs=nnx.Rngs(0))
        assert lpips.name == "lpips"

    def test_is_nnx_module(self) -> None:
        lpips = LPIPSMetric(rngs=nnx.Rngs(0))
        assert isinstance(lpips, nnx.Module)

    def test_has_learned_weights(self) -> None:
        lpips = LPIPSMetric(rngs=nnx.Rngs(0))
        assert len(lpips._layer_weights) == 5  # Default 5 VGG layers

    def test_inherits_nnx_mode_switching(self) -> None:
        """nnx.Module provides train() for mode switching."""
        lpips = LPIPSMetric(rngs=nnx.Rngs(0))
        lpips.train()  # inherited from nnx.Module
