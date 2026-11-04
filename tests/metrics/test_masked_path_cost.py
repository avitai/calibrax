"""A mask alone costs the one selection it needs, and nothing more.

With a mask and no weights the excluded elements are exactly the ones the mask drops, so
zeroing them before the arithmetic takes one ``jnp.where``. The general rule, which also keeps
a zero-weighted element's value for its weight derivative, needs the folded weights, an
``isfinite`` test and two boolean operations for every input. With no weights given it
computes nothing the mask does not already say.

The cost is real on the compiled path. On the CPU backend XLA runs a reduction as an XNNPACK
custom fusion, which takes no select, so every selection before the reduction is a pass over
memory of its own: substituting two inputs is two passes, guarding their difference is one.
Each test compiles the metric and the minimal NaN-safe program written out beside it, in one
process, and compares their fusion counts (never a recorded number, which a jax upgrade would
move). NaN safety itself, value and gradient, is held by ``test_masked_nan_safety.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.metrics._utils import _EPSILON, reduce_values, weighted_mean
from calibrax.metrics.functional.regression import (
    _huber_elementwise,
    CHARBONNIER_ALPHA,
    CHARBONNIER_EPSILON,
    charbonnier_loss,
    HUBER_DELTA,
    huber_loss,
    mae,
    mse,
    per_sample_relative_l2,
    r_squared,
    relative_l2_error,
)
from tests.factories import fusion_count


_RNG = np.random.default_rng(3)
_SIZE = 4096
_P = jnp.asarray(_RNG.normal(size=_SIZE), dtype=jnp.float32)
_T = jnp.asarray(_RNG.normal(size=_SIZE), dtype=jnp.float32)
_KEEP = jnp.asarray(_RNG.random(_SIZE) > 0.3)
_FIELD_P = jnp.asarray(_RNG.normal(size=(64, 32)), dtype=jnp.float32)
_FIELD_T = jnp.asarray(_RNG.normal(size=(64, 32)), dtype=jnp.float32)
_SAMPLE_KEEP = jnp.asarray(_RNG.random(64) > 0.3)


def _guarded_difference(p: jax.Array, t: jax.Array, keep: jax.Array) -> jax.Array:
    """The difference with excluded elements zeroed: the one selection a loss needs."""
    return jnp.where(keep, p - t, 0.0)


#: Each element-wise loss beside its minimal masked program: one selection on the difference,
#: then the loss's own element-wise expression and the shared reduction.
_LOSSES: list[tuple[str, Callable[..., jax.Array], Callable[..., jax.Array]]] = [
    (
        "mse",
        lambda p, t, keep: mse(p, t, mask=keep),
        lambda p, t, keep: reduce_values(_guarded_difference(p, t, keep) ** 2, mask=keep),
    ),
    (
        "mae",
        lambda p, t, keep: mae(p, t, mask=keep),
        lambda p, t, keep: reduce_values(jnp.abs(_guarded_difference(p, t, keep)), mask=keep),
    ),
    (
        "huber_loss",
        lambda p, t, keep: huber_loss(p, t, mask=keep),
        lambda p, t, keep: reduce_values(
            _huber_elementwise(
                _guarded_difference(p, t, keep) ** 2,
                jnp.abs(_guarded_difference(p, t, keep)),
                HUBER_DELTA,
            ),
            mask=keep,
        ),
    ),
    (
        "charbonnier_loss",
        lambda p, t, keep: charbonnier_loss(p, t, mask=keep),
        lambda p, t, keep: reduce_values(
            jnp.sqrt(_guarded_difference(p, t, keep) ** 2 + CHARBONNIER_EPSILON**2)
            ** CHARBONNIER_ALPHA,
            mask=keep,
        ),
    ),
]


@pytest.mark.parametrize(("name", "metric", "minimal"), _LOSSES, ids=[c[0] for c in _LOSSES])
def test_a_masked_loss_compiles_to_one_selection(
    name: str, metric: Callable[..., jax.Array], minimal: Callable[..., jax.Array]
) -> None:
    """The loss guards its difference once, not each input by the general rule."""
    assert fusion_count(metric, _P, _T, _KEEP) == fusion_count(minimal, _P, _T, _KEEP), name


def test_a_masked_relative_l2_error_selects_each_input_once() -> None:
    """Both inputs are read on their own, the target as the denominator, so each needs one
    selection at the sample axis and no more."""

    def minimal(p: jax.Array, t: jax.Array, keep: jax.Array) -> jax.Array:
        field = keep[:, None]
        return reduce_values(
            per_sample_relative_l2(jnp.where(field, p, 0.0), jnp.where(field, t, 0.0)),
            mask=keep,
        )

    def metric(p: jax.Array, t: jax.Array, keep: jax.Array) -> jax.Array:
        return relative_l2_error(p, t, mask=keep)

    assert fusion_count(metric, _FIELD_P, _FIELD_T, _SAMPLE_KEEP) == fusion_count(
        minimal, _FIELD_P, _FIELD_T, _SAMPLE_KEEP
    )


def test_a_masked_statistic_selects_each_input_once() -> None:
    """A ratio statistic reads both inputs on their own, through the folded weights."""

    def minimal(p: jax.Array, t: jax.Array, keep: jax.Array) -> jax.Array:
        w = jnp.where(keep, 1.0, 0.0)
        p = jnp.where(keep, p, 0.0)
        t = jnp.where(keep, t, 0.0)
        ss_res = jnp.sum(w * (p - t) ** 2)
        ss_tot = jnp.sum(w * (t - weighted_mean(t, w)) ** 2)
        return jnp.where(jnp.sum(w) > 0.0, 1.0 - ss_res / (ss_tot + _EPSILON), jnp.nan)

    def metric(p: jax.Array, t: jax.Array, keep: jax.Array) -> jax.Array:
        return r_squared(p, t, mask=keep)

    assert fusion_count(metric, _P, _T, _KEEP) == fusion_count(minimal, _P, _T, _KEEP)
