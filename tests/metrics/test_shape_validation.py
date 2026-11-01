"""A metric either refuses a mismatched pair of inputs, or documents why it accepts one.

``predictions`` of shape ``(n, 1)`` against ``targets`` of shape ``(n,)`` is not one sample
described two ways: the two broadcast to ``(n, n)``, pairing every element of one with every
element of the other, so the answer comes from a square expansion of a vector comparison and
nothing downstream can tell. Measured across the registry with each metric's own fixture,
twenty metrics accept that pair and return a number different from the flat call.

Three groups, because the contracts differ and the remedy follows the contract:

* **Same shape by documentation, no validation at all.** Nine metrics whose docstrings give
  both inputs the same shape -- two label assignments over the same points, two signals, two
  images, two points on a manifold, a prediction beside the group it belongs to -- and which
  convert with a bare ``jnp.asarray``. They take the preamble their siblings use.
* **The shared distance helper.** ``_batch_or_single`` dispatches on ``a.ndim`` and never
  compares ``a`` with ``b``, so every distance routed through it inherits the gap.
* **Different shapes are the contract.** Four metrics document a probability array against
  class indices, so a trailing axis is a legitimate call and refusing it would be the defect.
  They are pinned here so that a later sweep does not "fix" them.

The reference for every refusal is the sibling that already refuses: ``mse`` and the other
twenty-four metrics that prepare their inputs with
:func:`calibrax.metrics._utils._prepare_arrays` raise ``ValueError: Shape mismatch``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.metrics import functional


@dataclass(frozen=True, slots=True, kw_only=True)
class ShapeCase:
    """A metric, a call inside its documented contract, and any extra arguments it needs.

    Attributes:
        name: The metric's registered name, used as the parametrisation id.
        metric: The callable under test, taking the two inputs positionally.
        first: The first input, shaped as the metric documents it.
        second: The second input, shaped as the metric documents it.
        extra: Keyword arguments the call requires beyond the two inputs.
    """

    name: str
    metric: Callable[..., jnp.ndarray]
    first: jnp.ndarray
    second: jnp.ndarray
    extra: Mapping[str, object] = field(default_factory=dict)


def _ids(cases: tuple[ShapeCase, ...]) -> list[str]:
    """Parametrisation ids, so a failure names the metric rather than an index."""
    return [case.name for case in cases]


#: One generator for every fixture, so the inputs are reproducible and the file has one seed.
_RNG = np.random.default_rng(7)
_SIZE = 12
_CLASSES = 3

#: Two label assignments over the same twelve points: what the clustering metrics document.
_LABELS_TRUE = jnp.asarray(_RNG.integers(0, _CLASSES, size=_SIZE))
_LABELS_PRED = jnp.asarray(_RNG.integers(0, _CLASSES, size=_SIZE))

#: Two signals, and two images, on the unit interval.
_SIGNAL = jnp.asarray(_RNG.random(_SIZE), dtype=jnp.float32)
_SIGNAL_OTHER = jnp.asarray(_RNG.random(_SIZE), dtype=jnp.float32)

#: Two vectors in the same space, for the distances.
_VECTOR = jnp.asarray(_RNG.normal(size=_SIZE), dtype=jnp.float32)
_VECTOR_OTHER = jnp.asarray(_RNG.normal(size=_SIZE), dtype=jnp.float32)


def _inside_the_unit_ball(values: jnp.ndarray) -> jnp.ndarray:
    """Scale to norm 0.5, because the Poincare ball excludes its own boundary."""
    return values / (2.0 * jnp.linalg.norm(values))


_BALL = _inside_the_unit_ball(jnp.asarray(_RNG.normal(size=_SIZE), dtype=jnp.float32))
_BALL_OTHER = _inside_the_unit_ball(jnp.asarray(_RNG.normal(size=_SIZE), dtype=jnp.float32))

#: A binary prediction beside the group each sample belongs to, for the fairness ratios.
_BINARY = jnp.asarray((_RNG.random(_SIZE) > 0.5).astype(np.float32))
_GROUPS = jnp.asarray(_RNG.integers(0, 2, size=_SIZE))

#: Per-class probabilities against class indices: the documented *different* shapes.
_PROBABILITIES = jnp.asarray(_RNG.dirichlet(np.ones(_CLASSES), size=_SIZE), dtype=jnp.float32)
_INDICES = jnp.asarray(_RNG.integers(0, _CLASSES, size=_SIZE))


#: Nine metrics whose docstrings give both inputs the same shape and which validate nothing.
_SAME_SHAPE_BY_DOCUMENTATION: tuple[ShapeCase, ...] = (
    ShapeCase(
        name="adjusted_mutual_information",
        metric=functional.adjusted_mutual_information,
        first=_LABELS_TRUE,
        second=_LABELS_PRED,
    ),
    ShapeCase(
        name="adjusted_rand_index",
        metric=functional.adjusted_rand_index,
        first=_LABELS_TRUE,
        second=_LABELS_PRED,
    ),
    ShapeCase(
        name="normalized_mutual_information_clustering",
        metric=functional.normalized_mutual_information_clustering,
        first=_LABELS_TRUE,
        second=_LABELS_PRED,
    ),
    ShapeCase(
        name="v_measure", metric=functional.v_measure, first=_LABELS_TRUE, second=_LABELS_PRED
    ),
    ShapeCase(
        name="spectral_convergence",
        metric=functional.spectral_convergence,
        first=_SIGNAL,
        second=_SIGNAL_OTHER,
    ),
    ShapeCase(name="psnr", metric=functional.psnr, first=_SIGNAL, second=_SIGNAL_OTHER),
    ShapeCase(
        name="ultrahyperbolic_distance",
        metric=functional.ultrahyperbolic_distance,
        first=_VECTOR,
        second=_VECTOR_OTHER,
        extra={"signature": (4, 8)},
    ),
    ShapeCase(
        name="demographic_parity_ratio",
        metric=functional.demographic_parity_ratio,
        first=_BINARY,
        second=_GROUPS,
    ),
    ShapeCase(
        name="disparate_impact_ratio",
        metric=functional.disparate_impact_ratio,
        first=_BINARY,
        second=_GROUPS,
    ),
)

#: A drift direction inside the unit ball, which ``randers_distance`` requires.
_DRIFT = _inside_the_unit_ball(jnp.asarray(_RNG.normal(size=_SIZE), dtype=jnp.float32))

#: Every distance that reaches the shared helper: eight of them, seven routed through it and
#: ``mahalanobis_distance`` through a hand-written copy of the same dispatch. The check lands
#: in the helper, so all eight are pinned here rather than only the ones a value sweep finds.
#: ``lorentz_distance`` and ``randers_distance`` are the two the sweep could not classify --
#: they raise ``IndexError`` and ``TypeError`` on a mismatch today, which is a refusal that
#: names neither the argument nor the contract.
_THROUGH_THE_DISTANCE_HELPER: tuple[ShapeCase, ...] = (
    ShapeCase(
        name="chebyshev_distance",
        metric=functional.chebyshev_distance,
        first=_VECTOR,
        second=_VECTOR_OTHER,
    ),
    ShapeCase(
        name="cosine_distance",
        metric=functional.cosine_distance,
        first=_VECTOR,
        second=_VECTOR_OTHER,
    ),
    ShapeCase(
        name="euclidean_distance",
        metric=functional.euclidean_distance,
        first=_VECTOR,
        second=_VECTOR_OTHER,
    ),
    ShapeCase(
        name="manhattan_distance",
        metric=functional.manhattan_distance,
        first=_VECTOR,
        second=_VECTOR_OTHER,
    ),
    ShapeCase(
        name="minkowski_distance",
        metric=functional.minkowski_distance,
        first=_VECTOR,
        second=_VECTOR_OTHER,
    ),
    ShapeCase(
        name="poincare_distance",
        metric=functional.poincare_distance,
        first=_BALL,
        second=_BALL_OTHER,
    ),
    ShapeCase(
        name="mahalanobis_distance",
        metric=functional.mahalanobis_distance,
        first=_VECTOR,
        second=_VECTOR_OTHER,
    ),
    ShapeCase(
        name="lorentz_distance",
        metric=functional.lorentz_distance,
        first=_VECTOR,
        second=_VECTOR_OTHER,
    ),
    ShapeCase(
        name="randers_distance",
        metric=functional.randers_distance,
        first=_VECTOR,
        second=_VECTOR_OTHER,
        extra={"direction": _DRIFT, "magnitude": 0.3},
    ),
)

_REFUSES = _SAME_SHAPE_BY_DOCUMENTATION + _THROUGH_THE_DISTANCE_HELPER

#: Four metrics documenting a probability array against class indices, where the shapes differ
#: by contract. Each is called the way its docstring describes.
_ACCEPTS_DIFFERENT_SHAPES: tuple[ShapeCase, ...] = (
    ShapeCase(
        name="ranked_probability_score",
        metric=functional.ranked_probability_score,
        first=_PROBABILITIES,
        second=_INDICES,
    ),
    ShapeCase(name="log_loss", metric=functional.log_loss, first=_PROBABILITIES, second=_INDICES),
    ShapeCase(
        name="balanced_accuracy",
        metric=functional.balanced_accuracy,
        first=_PROBABILITIES,
        second=_INDICES,
    ),
    ShapeCase(
        name="cohen_kappa", metric=functional.cohen_kappa, first=_PROBABILITIES, second=_INDICES
    ),
)


@pytest.mark.parametrize("case", _REFUSES, ids=_ids(_REFUSES))
def test_a_trailing_axis_on_the_first_input_is_refused(case: ShapeCase) -> None:
    """``(n, 1)`` against ``(n,)`` is not one sample, so it cannot be answered.

    The message is the one the twenty-four metrics that already validate produce, because a
    caller should not have to learn which metric refused them in which words.
    """
    with pytest.raises(ValueError, match=r"[Ss]hape mismatch"):
        case.metric(case.first[:, None], case.second, **case.extra)


@pytest.mark.parametrize("case", _REFUSES, ids=_ids(_REFUSES))
def test_the_in_contract_call_still_answers(case: ShapeCase) -> None:
    """Guard the fixtures: the refusal must be specific to the mismatch, not to the inputs.

    Without this, a fixture that is out of contract would make the test above pass for the
    wrong reason -- the metric refusing because the data is wrong rather than the shape.
    """
    value = float(case.metric(case.first, case.second, **case.extra))
    assert np.isfinite(value), f"{case.name}: the in-contract fixture is not a valid call"


@pytest.mark.parametrize("case", _REFUSES, ids=_ids(_REFUSES))
def test_the_refusal_is_symmetric_in_the_inputs(case: ShapeCase) -> None:
    """A trailing axis on either side is the same mismatch and gets the same answer."""
    with pytest.raises(ValueError, match=r"[Ss]hape mismatch"):
        case.metric(case.first, case.second[:, None], **case.extra)


@pytest.mark.parametrize("case", _ACCEPTS_DIFFERENT_SHAPES, ids=_ids(_ACCEPTS_DIFFERENT_SHAPES))
def test_a_documented_difference_in_shape_is_accepted(case: ShapeCase) -> None:
    """These four document a probability array against class indices, so they must not refuse.

    Pinned because a sweep for the defect above finds them too: they accept a trailing axis
    and return a different number, which for them is the contract rather than a gap. Refusing
    here would break the multiclass call every one of them documents.
    """
    assert case.first.ndim != case.second.ndim, (
        f"{case.name}: this case exists to pin differing shapes, so the fixture must differ"
    )
    value = float(case.metric(case.first, case.second, **case.extra))
    assert np.isfinite(value), f"{case.name} refused the shape its docstring documents"
