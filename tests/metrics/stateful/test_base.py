"""Tests for FrozenBackboneMetric and LearnedMetric base classes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from calibrax.core.protocols import StatefulMetricProtocol
from calibrax.metrics.stateful._base import FrozenBackboneMetric, LearnedMetric


class MockBackboneMetric(FrozenBackboneMetric):
    """Mock implementation for testing the base class."""

    def __init__(self) -> None:
        super().__init__(name="mock_metric")
        self._accumulated: list[float] = []

    def reset(self) -> None:
        """Reset accumulated state."""
        self._accumulated = []

    def _extract_features(self, **kwargs: Any) -> float:
        """Extract mean of values."""
        values = kwargs.get("values", jnp.array([0.0]))
        return float(jnp.mean(values))

    def _accumulate(self, features: Any) -> None:
        """Accumulate extracted features."""
        self._accumulated.append(features)

    def _compute_from_accumulated(self) -> dict[str, float]:
        """Compute mean of accumulated features."""
        if not self._accumulated:
            return {"mock_metric": 0.0}
        return {"mock_metric": sum(self._accumulated) / len(self._accumulated)}


class TestFrozenBackboneMetric:
    """Tests for FrozenBackboneMetric."""

    def test_name_property(self) -> None:
        metric = MockBackboneMetric()
        assert metric.name == "mock_metric"

    def test_update_accumulates(self) -> None:
        metric = MockBackboneMetric()
        metric.update(values=jnp.array([1.0, 2.0, 3.0]))
        metric.update(values=jnp.array([4.0, 5.0, 6.0]))
        assert len(metric._accumulated) == 2

    def test_compute_returns_dict(self) -> None:
        metric = MockBackboneMetric()
        metric.update(values=jnp.array([2.0, 4.0]))
        result = metric.compute()
        assert isinstance(result, dict)
        assert "mock_metric" in result
        assert result["mock_metric"] == pytest.approx(3.0)

    def test_reset_clears_state(self) -> None:
        metric = MockBackboneMetric()
        metric.update(values=jnp.array([1.0]))
        metric.reset()
        result = metric.compute()
        assert result["mock_metric"] == 0.0

    def test_lifecycle(self) -> None:
        """Full lifecycle: update multiple batches, compute, reset, reuse."""
        metric = MockBackboneMetric()
        metric.update(values=jnp.array([1.0]))
        metric.update(values=jnp.array([3.0]))
        result = metric.compute()
        assert result["mock_metric"] == pytest.approx(2.0)
        metric.reset()
        metric.update(values=jnp.array([10.0]))
        result = metric.compute()
        assert result["mock_metric"] == pytest.approx(10.0)

    def test_conforms_to_protocol(self) -> None:
        """MockBackboneMetric should satisfy StatefulMetricProtocol."""
        metric = MockBackboneMetric()
        assert isinstance(metric, StatefulMetricProtocol)

    def test_compute_without_update(self) -> None:
        """Compute without any updates should return default."""
        metric = MockBackboneMetric()
        result = metric.compute()
        assert result["mock_metric"] == 0.0

    def test_plot_returns_output_path(self, tmp_path: Path) -> None:
        """Frozen metrics should plot computed scalar values."""
        metric = MockBackboneMetric()
        metric.update(values=jnp.array([1.0, 3.0]))
        result = metric.plot(output_dir=tmp_path)
        assert result is not None
        assert result.exists()
        assert result.suffix == ".png"


class MockLearnedMetric(LearnedMetric):
    """Mock implementation for testing LearnedMetric base."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__(name="mock_learned", rngs=rngs)
        self._linear = nnx.Linear(in_features=4, out_features=1, rngs=rngs)
        self._accumulated: list[float] = []

    def reset(self) -> None:
        """Reset accumulated state."""
        self._accumulated = []

    def update(self, **kwargs: Any) -> None:
        """Update with mock computation."""
        values = jnp.asarray(kwargs["values"])
        output = self._linear(values)
        self._accumulated.append(float(jnp.mean(output)))

    def compute(self) -> dict[str, float]:
        """Compute mean of accumulated values."""
        if not self._accumulated:
            return {"mock_learned": 0.0}
        return {"mock_learned": sum(self._accumulated) / len(self._accumulated)}


class TestLearnedMetric:
    """Tests for LearnedMetric."""

    def test_is_nnx_module(self) -> None:
        metric = MockLearnedMetric(rngs=nnx.Rngs(0))
        assert isinstance(metric, nnx.Module)

    def test_name_property(self) -> None:
        metric = MockLearnedMetric(rngs=nnx.Rngs(0))
        assert metric.name == "mock_learned"

    def test_inherits_nnx_mode_switching(self) -> None:
        """nnx.Module provides train() and set_attributes() for mode switching."""
        metric = MockLearnedMetric(rngs=nnx.Rngs(0))
        metric.train()  # inherited from nnx.Module

    def test_has_trainable_params(self) -> None:
        metric = MockLearnedMetric(rngs=nnx.Rngs(0))
        # nnx.Module should have state
        assert metric._linear is not None

    def test_update_compute(self) -> None:
        metric = MockLearnedMetric(rngs=nnx.Rngs(42))
        metric.update(values=jnp.ones(4))
        result = metric.compute()
        assert isinstance(result, dict)
        assert "mock_learned" in result

    def test_plot_returns_output_path(self, tmp_path: Path) -> None:
        """Learned metrics should share the same plotting behavior."""
        metric = MockLearnedMetric(rngs=nnx.Rngs(42))
        metric.update(values=jnp.ones(4))
        result = metric.plot(output_dir=tmp_path)
        assert result is not None
        assert result.exists()
        assert result.suffix == ".png"
