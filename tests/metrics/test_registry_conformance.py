"""Every registered metric is what its entry says: its signature, ``jax.jit`` and ``jax.grad``.

``MetricProperties`` promises two capabilities and ``MetricSignature`` a calling convention, and a
promise nothing checks goes stale. Each entry's function is called on inputs its signature
admits (or a fixture below, for inputs the signature cannot describe) and:

- ``is_jit_compatible`` holds exactly when ``jax.jit`` of the call, keyword arguments closed
  over, reproduces the eager value;
- ``is_differentiable`` holds exactly when ``jax.grad`` with respect to the first argument is
  finite and not identically zero;
- a signature other than ``CUSTOM`` is callable as ``fn(*arrays)`` with no keyword.
"""

from __future__ import annotations

import functools
import inspect
import zlib
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.metrics import MetricRegistry
from calibrax.metrics._types import MetricEntry, MetricFn, MetricSignature, MetricTier


class _Stream:
    """The random stream every input builder draws from; reseeded for each metric."""

    rng = np.random.default_rng(0)


_N = 32
_LABELLED = {"classification", "calibration", "segmentation", "fairness", "ranking"}
_POSITIONAL = {
    MetricSignature.PREDICTIONS_TARGETS: 2,
    MetricSignature.ENSEMBLE_PREDICTIONS_TARGETS: 2,
    MetricSignature.SAMPLES: 2,
    MetricSignature.FEATURES_LABELS: 2,
    MetricSignature.DISTRIBUTIONS: 2,
    MetricSignature.SINGLE_INPUT: 1,
}

Call = tuple[tuple[object, ...], Mapping[str, object]]


def _f32(values: np.ndarray) -> jax.Array:
    return jnp.asarray(values, jnp.float32)


def _probabilities(size: int = _N) -> jax.Array:
    return _f32(_Stream.rng.uniform(0.05, 0.95, size))


def _binary(size: int = _N) -> jax.Array:
    return _f32(_Stream.rng.random(size) > 0.5)


def _spd(dim: int = 3) -> jax.Array:
    base = _Stream.rng.normal(size=(dim, dim))
    return _f32(base @ base.T + dim * np.eye(dim))


def _adjacency(nodes: int = 6) -> jax.Array:
    weights = np.triu(_Stream.rng.uniform(0.2, 1.0, (nodes, nodes)), 1)
    return _f32(weights + weights.T)


def _hyperboloid(dim: int = 2) -> jax.Array:
    spatial = _Stream.rng.normal(size=dim) * 0.5
    return _f32(np.concatenate([[np.sqrt(1.0 + spatial @ spatial)], spatial]))


def _image(size: int = 48) -> jax.Array:
    return _f32(_Stream.rng.uniform(0.0, 1.0, (size, size)))


def _labels(classes: int) -> jax.Array:
    return jnp.asarray(_Stream.rng.integers(0, classes, _N), jnp.int32)


def _covariances(count: int, dim: int = 2) -> jax.Array:
    return jnp.stack([_spd(dim) for _ in range(count)])


def _joint() -> jax.Array:
    table = _Stream.rng.uniform(0.1, 1.0, (4, 3))
    return _f32(table / table.sum())


def _gram() -> jax.Array:
    vectors = _Stream.rng.normal(size=(6, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return _f32(vectors @ vectors.T)


def _ranking(k: int = 5) -> Call:
    return (_f32(_Stream.rng.normal(size=_N)), _binary()), {"k": k}


# Inputs a signature cannot describe: arrays of a particular structure, or keyword arguments.
# The clustering agreements are called as the registry calls them: predicted labels first.
FIXTURES: dict[str, Callable[[], Call]] = {
    "adjusted_mutual_information": lambda: (
        (_labels(3), _labels(4)),
        {"num_classes": 4, "num_clusters": 3},
    ),
    "adjusted_rand_index": lambda: (
        (_labels(3), _labels(4)),
        {"num_classes": 4, "num_clusters": 3},
    ),
    "normalized_mutual_information_clustering": lambda: (
        (_labels(3), _labels(4)),
        {"num_classes": 4, "num_clusters": 3},
    ),
    "v_measure": lambda: ((_labels(3), _labels(4)), {"num_classes": 4, "num_clusters": 3}),
    "silhouette_score": lambda: (
        (_f32(_Stream.rng.normal(size=(_N, 2))), _labels(3)),
        {"num_clusters": 3},
    ),
    "calinski_harabasz_score": lambda: (
        (_f32(_Stream.rng.normal(size=(_N, 2))), _labels(3)),
        {"num_clusters": 3},
    ),
    "davies_bouldin_score": lambda: (
        (_f32(_Stream.rng.normal(size=(_N, 2))), _labels(3)),
        {"num_clusters": 3},
    ),
    "balanced_accuracy": lambda: ((_labels(3), _labels(3)), {"num_classes": 3}),
    "cohen_kappa": lambda: ((_labels(3), _labels(3)), {"num_classes": 3}),
    "classwise_ece": lambda: (
        (jax.nn.softmax(_f32(_Stream.rng.normal(size=(_N, 3)))), _labels(3)),
        {"num_classes": 3},
    ),
    "softmax_cross_entropy": lambda: ((_f32(_Stream.rng.normal(size=(_N, 3))), _labels(3)), {}),
    "conditional_entropy": lambda: ((_joint(),), {}),
    "mutual_information": lambda: ((_joint(),), {}),
    "normalized_mutual_information": lambda: ((_joint(),), {}),
    "energy_score": lambda: (
        (_f32(_Stream.rng.normal(size=(8, 5, 2))), _f32(_Stream.rng.normal(size=(8, 2)))),
        {},
    ),
    "ensemble_ranked_probability_score": lambda: (
        (_f32(_Stream.rng.normal(size=(_N, 6))), _f32(_Stream.rng.normal(size=_N))),
        {"thresholds": jnp.array([-0.5, 0.0, 0.5])},
    ),
    "graph_edit_distance_approx": lambda: ((_adjacency(), _adjacency()), {}),
    "spectral_distance": lambda: ((_adjacency(), _adjacency()), {}),
    "resistance_distance": lambda: ((_adjacency(),), {}),
    "shortest_path_distance": lambda: ((_adjacency(),), {}),
    "hit_rate": _ranking,
    "ndcg_at_k": _ranking,
    "precision_at_k": _ranking,
    "recall_at_k": _ranking,
    "coverage": lambda: ((jnp.asarray(_Stream.rng.integers(0, 20, 10)),), {"catalog_size": 20}),
    "ssim": lambda: ((_image(), _image()), {}),
    "ms_ssim": lambda: ((_image(96), _image(96)), {"power_factors": (0.5, 0.5)}),
    "r_squared_adjusted": lambda: (
        (_f32(_Stream.rng.normal(size=_N)), _f32(_Stream.rng.normal(size=_N))),
        {"num_predictors": 2},
    ),
    "randers_distance": lambda: (
        (_f32(_Stream.rng.normal(size=3)), _f32(_Stream.rng.normal(size=3))),
        {"direction": jnp.array([1.0, 0.0, 0.0]), "magnitude": 0.5},
    ),
    "spd_affine_invariant_distance": lambda: ((_spd(), _spd()), {}),
    "spd_log_euclidean_distance": lambda: ((_spd(), _spd()), {}),
    "grassmann_distance": lambda: (
        (
            _f32(np.linalg.qr(_Stream.rng.normal(size=(5, 2)))[0]),
            _f32(np.linalg.qr(_Stream.rng.normal(size=(5, 2)))[0]),
        ),
        {},
    ),
    "stiefel_distance": lambda: (
        (
            _f32(np.linalg.qr(_Stream.rng.normal(size=(5, 2)))[0]),
            _f32(np.linalg.qr(_Stream.rng.normal(size=(5, 2)))[0]),
        ),
        {},
    ),
    "ultrahyperbolic_distance": lambda: ((_hyperboloid(), _hyperboloid()), {"signature": (1, 2)}),
    # Points on the hyperboloid and inside the unit ball: the models these distances are defined on.
    "lorentz_distance": lambda: ((_hyperboloid(), _hyperboloid()), {}),
    "poincare_distance": lambda: (
        (_f32(_Stream.rng.uniform(-0.4, 0.4, 3)), _f32(_Stream.rng.uniform(-0.4, 0.4, 3))),
        {},
    ),
    # A similarity matrix: the Gram matrix of unit vectors.
    "vendi_score": lambda: ((_gram(),), {}),
    "anees": lambda: (
        (
            _f32(_Stream.rng.normal(size=(8, 2))),
            _covariances(8),
            _f32(_Stream.rng.normal(size=(8, 2))),
        ),
        {},
    ),
    "non_credibility_index": lambda: (
        (
            _f32(_Stream.rng.normal(size=(8, 2))),
            _covariances(8),
            _f32(_Stream.rng.normal(size=(8, 2))),
            _covariances(8),
        ),
        {},
    ),
    "bleu": lambda: (("the cat sat on the mat", ["the cat is on the mat"]), {}),
    "rouge_l": lambda: (("the cat sat on the mat", "the cat is on the mat"), {}),
    "rouge_n": lambda: (("the cat sat on the mat", "the cat is on the mat"), {}),
    "distinct_n": lambda: ((["the", "cat", "the", "mat"],), {}),
    "perplexity": lambda: ((_f32(-_Stream.rng.uniform(0.1, 3.0, _N)),), {}),
    "bregman_divergence": lambda: (
        (_probabilities(8), _probabilities(8)),
        {"generator": lambda u: jnp.sum(u * jnp.log(u))},
    ),
    "f_divergence": lambda: (
        (
            jax.nn.softmax(_f32(_Stream.rng.normal(size=8))),
            jax.nn.softmax(_f32(_Stream.rng.normal(size=8))),
        ),
        {"generator": lambda u: u * jnp.log(u)},
    ),
    "gaussian_nll": lambda: (
        (
            _f32(_Stream.rng.normal(size=_N)),
            _f32(_Stream.rng.uniform(0.5, 2.0, _N)),
            _f32(_Stream.rng.normal(size=_N)),
        ),
        {},
    ),
    "regression_calibration_error": lambda: (
        (
            _f32(_Stream.rng.normal(size=_N)),
            _f32(_Stream.rng.uniform(0.5, 2.0, _N)),
            _f32(_Stream.rng.normal(size=_N)),
        ),
        {"quantile_levels": jnp.array([0.1, 0.5, 0.9])},
    ),
    "interval_score": lambda: (
        (
            _f32(_Stream.rng.normal(size=_N)) - 1.0,
            _f32(_Stream.rng.normal(size=_N)) + 1.0,
            _f32(_Stream.rng.normal(size=_N)),
        ),
        {"alpha": 0.1},
    ),
    "picp": lambda: (
        (
            _f32(_Stream.rng.normal(size=_N)) - 1.0,
            _f32(_Stream.rng.normal(size=_N)) + 1.0,
            _f32(_Stream.rng.normal(size=_N)),
        ),
        {},
    ),
    "mpiw": lambda: (
        (_f32(_Stream.rng.normal(size=_N)) - 1.0, _f32(_Stream.rng.normal(size=_N)) + 1.0),
        {},
    ),
    "demographic_parity_ratio": lambda: ((_probabilities(), _labels(3)), {"num_groups": 3}),
    "disparate_impact_ratio": lambda: ((_probabilities(), _labels(3)), {"num_groups": 3}),
    "equalized_odds_difference": lambda: (
        (_probabilities(), _binary(), _labels(3)),
        {"num_groups": 3},
    ),
    "equal_opportunity_difference": lambda: (
        (_probabilities(), _binary(), _labels(3)),
        {"num_groups": 3},
    ),
}


def _signature_call(entry: MetricEntry) -> Call:
    """Generic inputs for an entry whose signature describes them."""
    signature = entry.signature
    if signature is MetricSignature.PREDICTIONS_TARGETS:
        targets = _binary() if entry.domain in _LABELLED else _f32(_Stream.rng.normal(size=_N))
        return (_probabilities(), targets), {}
    if signature is MetricSignature.ENSEMBLE_PREDICTIONS_TARGETS:
        return (_f32(_Stream.rng.normal(size=(_N, 8))), _f32(_Stream.rng.normal(size=_N))), {}
    if signature is MetricSignature.SAMPLES:
        return (
            _f32(_Stream.rng.normal(size=(_N, 3))),
            _f32(_Stream.rng.normal(size=(_N, 3)) + 0.5),
        ), {}
    if signature is MetricSignature.DISTRIBUTIONS:
        return (
            jax.nn.softmax(_f32(_Stream.rng.normal(size=16))),
            jax.nn.softmax(_f32(_Stream.rng.normal(size=16))),
        ), {}
    if signature is MetricSignature.SINGLE_INPUT:
        return (_probabilities(),), {}
    msg = f"{entry.name}: a {signature.value} entry needs a fixture"
    raise LookupError(msg)


def _entries() -> list[MetricEntry]:
    registry = MetricRegistry()
    return sorted(
        (entry for tier in MetricTier for entry in registry.list_by_tier(tier) if entry.fn),
        key=lambda entry: entry.name,
    )


ENTRIES = _entries()
IDS = [entry.name for entry in ENTRIES]


def _call(entry: MetricEntry) -> Call:
    """The entry's inputs, drawn from a stream seeded by its name: the same whatever runs first."""
    _Stream.rng = np.random.default_rng(zlib.crc32(entry.name.encode()))
    return FIXTURES[entry.name]() if entry.name in FIXTURES else _signature_call(entry)


def _fn(entry: MetricEntry) -> MetricFn:
    assert entry.fn is not None
    return entry.fn


def _close(a: object, b: object) -> bool:
    left = jnp.asarray(a, jnp.float32)
    right = jnp.asarray(b, jnp.float32)
    return bool(jnp.allclose(left, right, rtol=1e-4, atol=1e-5, equal_nan=True))


def test_every_registered_metric_is_covered() -> None:
    assert len(ENTRIES) == 139
    stale = sorted(set(FIXTURES) - set(IDS))
    assert stale == [], f"fixtures for unregistered metrics: {stale}"


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_the_signature_matches_the_function(entry: MetricEntry) -> None:
    if entry.signature is MetricSignature.CUSTOM:
        return
    parameters = inspect.signature(_fn(entry)).parameters.values()
    required = [p for p in parameters if p.default is inspect.Parameter.empty]
    positional = [p for p in required if p.kind is not inspect.Parameter.KEYWORD_ONLY]
    keyword = [p.name for p in required if p.kind is inspect.Parameter.KEYWORD_ONLY]

    assert keyword == [], f"required keywords {keyword}: a {entry.signature.value} call has none"
    assert len(positional) == _POSITIONAL[entry.signature]


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_jit_compatibility_is_as_declared(entry: MetricEntry) -> None:
    args, kwargs = _call(entry)
    bound = functools.partial(_fn(entry), **kwargs)
    eager = bound(*args)
    try:
        traced_ok = _close(jax.jit(bound)(*args), eager)
    except (TypeError, ValueError, jax.errors.ConcretizationTypeError):
        traced_ok = False

    assert traced_ok is entry.properties.is_jit_compatible


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_differentiability_is_as_declared(entry: MetricEntry) -> None:
    args, kwargs = _call(entry)
    bound = functools.partial(_fn(entry), **kwargs)
    first, rest = args[0], args[1:]
    try:
        gradient = jax.grad(lambda x: jnp.sum(jnp.asarray(bound(x, *rest), jnp.float32)))(first)
        informative = bool(jnp.all(jnp.isfinite(gradient))) and bool(jnp.any(gradient != 0))
    except (TypeError, ValueError, jax.errors.ConcretizationTypeError):
        informative = False

    assert informative is entry.properties.is_differentiable
