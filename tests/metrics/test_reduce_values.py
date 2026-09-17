"""``reduce_values`` is public: consumers reduce their own losses the way calibrax's do."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from calibrax.metrics import reduce_values as from_metrics
from calibrax.metrics.functional import reduce_values


def test_exported_from_both_packages() -> None:
    assert from_metrics is reduce_values


def test_plain_reductions() -> None:
    values = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    assert jnp.allclose(reduce_values(values), 2.5)
    assert jnp.allclose(reduce_values(values, reduction="sum"), 10.0)
    assert jnp.allclose(reduce_values(values, reduction="none"), values)
    assert jnp.allclose(reduce_values(values, reduction="batch_sum"), 5.0)  # rows sum 3 and 7
    assert jnp.allclose(reduce_values(values, axis=0), jnp.array([2.0, 3.0]))


def test_mask_and_weights() -> None:
    values = jnp.array([1.0, 2.0, 3.0, 4.0])
    mask = jnp.array([True, True, False, True])
    weights = jnp.array([1.0, 3.0, 1.0, 1.0])
    assert jnp.allclose(reduce_values(values, mask=mask), (1.0 + 2.0 + 4.0) / 3)
    assert jnp.allclose(reduce_values(values, weights=weights), (1.0 + 6.0 + 3.0 + 4.0) / 6)
    assert jnp.allclose(reduce_values(values, mask=mask, weights=weights), (1.0 + 6.0 + 4.0) / 5)
    assert jnp.allclose(
        reduce_values(values, mask=mask, reduction="none"), jnp.array([1.0, 2.0, 0.0, 4.0])
    )
    assert reduce_values(values, mask=jnp.zeros(4, dtype=bool)) == 0.0


def test_rejects_an_unknown_reduction() -> None:
    with pytest.raises(ValueError, match="reduction"):
        reduce_values(jnp.ones(3), reduction="median")
