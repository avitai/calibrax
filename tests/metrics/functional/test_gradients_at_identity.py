"""A distance built on a root has a finite gradient where it is 0.

The root's derivative is infinite at 0, so a bare ``jnp.sqrt`` makes ``jax.grad`` NaN exactly at
a perfect match, the point a training loss converges to. Each of these distances takes its root
through ``safe_root``, whose derivative at 0 is 0.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.distance import (
    euclidean_distance,
    mahalanobis_distance,
    minkowski_distance,
)
from calibrax.metrics.functional.divergence import hellinger_distance, mmd
from calibrax.metrics.functional.forecasting import energy_score
from calibrax.metrics.functional.geometric import rmsd
from calibrax.metrics.functional.graph import graph_edit_distance_approx, spectral_distance
from calibrax.metrics.functional.manifold import spd_log_euclidean_distance, stiefel_distance
from calibrax.metrics.functional.regression import relative_error, relative_l2_error


_VEC = jnp.array([0.3, -1.2, 2.0])
_PROB = jnp.array([0.2, 0.3, 0.5])
_PROB_WITH_ZERO = jnp.array([0.0, 0.4, 0.6])
_SAMPLES = jax.random.normal(jax.random.key(0), (6, 2))
_COORDS = jax.random.normal(jax.random.key(1), (5, 3))
_ADJ = jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
_SPD = jnp.array([[2.0, 0.3], [0.3, 1.0]])
_BASIS = jnp.linalg.qr(jax.random.normal(jax.random.key(2), (4, 2)))[0]
_FIELDS = jax.random.normal(jax.random.key(3), (3, 4))


def _pattern(like: Any, scale: float = 0.05) -> Any:
    """A non-uniform perturbation: a uniform shift is no difference to a centred RMSD."""
    return scale * jnp.arange(1.0, like.size + 1.0).reshape(like.shape) / like.size


# name -> (distance to its identity point, the identity point, a point off it)
CASES: dict[str, tuple[Callable[[Any], Any], Any, Any]] = {
    "mahalanobis_distance": (lambda a: mahalanobis_distance(a, _VEC), _VEC, _VEC + _pattern(_VEC)),
    "mahalanobis_distance_with_precision": (
        lambda a: mahalanobis_distance(
            a, _VEC, precision_matrix=jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        ),
        _VEC,
        _VEC + _pattern(_VEC),
    ),
    "minkowski_distance": (lambda a: minkowski_distance(a, _VEC), _VEC, _VEC + _pattern(_VEC)),
    "minkowski_distance_p3": (
        lambda a: minkowski_distance(a, _VEC, p=3.0),
        _VEC,
        _VEC + _pattern(_VEC),
    ),
    "hellinger_distance": (lambda a: hellinger_distance(a, _PROB), _PROB, _PROB + _pattern(_PROB)),
    "hellinger_distance_with_a_zero_entry": (
        lambda a: hellinger_distance(a, _PROB_WITH_ZERO),
        _PROB_WITH_ZERO,
        _PROB_WITH_ZERO + _pattern(_PROB_WITH_ZERO),
    ),
    # The unbiased MMD^2 estimate is clamped at 0 and needs a real change of distribution.
    "mmd": (lambda a: mmd(a, _SAMPLES), _SAMPLES, _SAMPLES + 1.5),
    "rmsd": (lambda a: rmsd(a, _COORDS), _COORDS, _COORDS + _pattern(_COORDS)),
    "graph_edit_distance_approx": (
        lambda a: graph_edit_distance_approx(a, _ADJ),
        _ADJ,
        _ADJ + _pattern(_ADJ),
    ),
    "relative_error": (lambda a: relative_error(a, _VEC), _VEC, _VEC + _pattern(_VEC)),
    "euclidean_distance": (lambda a: euclidean_distance(a, _VEC), _VEC, _VEC + _pattern(_VEC)),
    "spectral_distance": (lambda a: spectral_distance(a, _ADJ), _ADJ, _ADJ + _pattern(_ADJ)),
    "spd_log_euclidean_distance": (
        lambda a: spd_log_euclidean_distance(a, _SPD),
        _SPD,
        _SPD + jnp.array([[0.2, 0.05], [0.05, 0.1]]),
    ),
    "stiefel_distance": (lambda a: stiefel_distance(a, _BASIS), _BASIS, _BASIS + _pattern(_BASIS)),
    "relative_l2_error": (
        lambda a: relative_l2_error(a, _FIELDS),
        _FIELDS,
        _FIELDS + _pattern(_FIELDS),
    ),
}


def test_the_energy_score_gradient_is_finite_for_any_ensemble() -> None:
    """Each member's distance to itself enters the spread term, so a bare norm was NaN always."""
    members = jax.random.normal(jax.random.key(4), (4, 5, 3))
    observed = jax.random.normal(jax.random.key(5), (4, 3))

    grad = jax.grad(lambda m: energy_score(m, observed))(members)

    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.abs(grad).sum()) > 0.0


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_gradient_is_finite_at_identity_and_off_it(name: str) -> None:
    fn, at, off = CASES[name]

    assert float(fn(at)) == pytest.approx(0.0, abs=1e-6)
    assert bool(jnp.all(jnp.isfinite(jax.grad(fn)(at))))
    grad_off = jax.grad(fn)(off)
    assert bool(jnp.all(jnp.isfinite(grad_off)))
    assert float(jnp.abs(grad_off).sum()) > 0.0


@pytest.mark.parametrize("name", sorted(CASES))
def test_values_are_unchanged_off_identity_under_jit(name: str) -> None:
    fn, _, off = CASES[name]

    assert float(jax.jit(fn)(off)) == pytest.approx(float(fn(off)), rel=1e-5)
    assert float(fn(off)) > 0.0
