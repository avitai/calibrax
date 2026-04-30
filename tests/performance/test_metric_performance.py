"""Performance benchmarks for core metric paths."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.regression import mse


@pytest.mark.performance
def test_mse_large_vector_performance(benchmark: Any) -> None:
    """Benchmark MSE on a large vector and assert the computed value."""
    predictions = jnp.ones((100_000,), dtype=jnp.float32)
    targets = jnp.zeros((100_000,), dtype=jnp.float32)

    result = benchmark(mse, predictions, targets)

    assert result == pytest.approx(1.0)
