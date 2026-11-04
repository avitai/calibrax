"""What an eager call does that compilation would absorb: counted, not timed.

On the eager path every ``jnp`` call is Python-level dispatch, so work that compiles to
nothing still costs. Two such costs are held here, each by counting the one operation it is
made of (substrax ``counted_calls``, whose weights see every call made from Python):

* **A shape check reads the arrays the metric converts anyway.** Reading a raw input's shape
  with ``jnp.shape`` and converting it on the next line pays twice for one fact; comparing the
  converted arrays' shapes pays only the comparison.
* **A mask alone costs nothing but the mask.** With no weights, the excluded elements are
  exactly the ones the mask drops, so the general rule's ``isfinite`` test -- which only a
  zero weight needs, to keep a finite observation's weight derivative -- has nothing to do.

Counts, never times (they do not depend on the machine). Inputs are NumPy arrays, so that a
conversion is a call that can be seen.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from substrax.testing import by_package, counted_calls

from calibrax.metrics.functional.regression import (
    charbonnier_loss,
    huber_loss,
    mae,
    mse,
    r_squared,
    r_squared_adjusted,
    relative_l2_error,
)
from calibrax.metrics.functional.statistical import (
    concordance_correlation,
    kendall_tau,
    pearson_correlation,
    spearman_rank_correlation,
)
from tests.metrics.test_shape_validation import (
    _SAME_SHAPE_BY_DOCUMENTATION,
    _THROUGH_THE_DISTANCE_HELPER,
    ShapeCase,
)


_KEYS = by_package("calibrax")
_SHAPE_CASES = (*_SAME_SHAPE_BY_DOCUMENTATION, *_THROUGH_THE_DISTANCE_HELPER)


def _counts(
    call: Callable[[], object], **weights: Callable[[object, object], int]
) -> dict[str, int]:
    """Weighted counts over one eager call, after a warm-up call."""
    call()
    with counted_calls(_KEYS, weights=weights) as counted:
        call()
    return {name: counted.get(name, 0) for name in weights}


@pytest.mark.parametrize("case", _SHAPE_CASES, ids=[c.name for c in _SHAPE_CASES])
def test_a_shape_check_reads_the_converted_arrays(case: ShapeCase) -> None:
    """No input's shape is read on its own, and each input is converted once."""
    first, second = np.asarray(case.first), np.asarray(case.second)
    inputs = (first, second)

    def shape_reads(callee: object, argument: object) -> int:
        return int(callee is jnp.shape and any(argument is x for x in inputs))

    def conversions(callee: object, argument: object) -> int:
        return int(callee is jnp.asarray and any(argument is x for x in inputs))

    found = _counts(
        lambda: case.metric(first, second, **case.extra),
        shape_reads=shape_reads,
        conversions=conversions,
    )
    assert found == {"shape_reads": 0, "conversions": 2}, case.name


def _isfinite(callee: object, argument: object) -> int:
    del argument
    return int(callee is jnp.isfinite)


_RNG = np.random.default_rng(5)
_P = _RNG.normal(size=64).astype(np.float32)
_T = _RNG.normal(size=64).astype(np.float32)
_KEEP = _RNG.random(64) > 0.3
_W = _RNG.uniform(0.5, 2.0, size=64).astype(np.float32)
_FIELD_P = _RNG.normal(size=(16, 8)).astype(np.float32)
_FIELD_T = _RNG.normal(size=(16, 8)).astype(np.float32)
_SAMPLE_KEEP = _RNG.random(16) > 0.3

#: Every masked metric, called with a mask alone and, where it takes them, with weights.
_MASKED: list[tuple[str, Callable[..., object], Callable[..., object] | None]] = [
    ("mse", lambda: mse(_P, _T, mask=_KEEP), lambda: mse(_P, _T, weights=_W)),
    ("mae", lambda: mae(_P, _T, mask=_KEEP), lambda: mae(_P, _T, weights=_W)),
    ("huber_loss", lambda: huber_loss(_P, _T, mask=_KEEP), lambda: huber_loss(_P, _T, weights=_W)),
    (
        "charbonnier_loss",
        lambda: charbonnier_loss(_P, _T, mask=_KEEP),
        lambda: charbonnier_loss(_P, _T, weights=_W),
    ),
    ("r_squared", lambda: r_squared(_P, _T, mask=_KEEP), lambda: r_squared(_P, _T, weights=_W)),
    (
        "r_squared_adjusted",
        lambda: r_squared_adjusted(_P, _T, num_predictors=2, mask=_KEEP),
        lambda: r_squared_adjusted(_P, _T, num_predictors=2, weights=_W),
    ),
    (
        "pearson_correlation",
        lambda: pearson_correlation(_P, _T, mask=_KEEP),
        lambda: pearson_correlation(_P, _T, weights=_W),
    ),
    (
        "concordance_correlation",
        lambda: concordance_correlation(_P, _T, mask=_KEEP),
        lambda: concordance_correlation(_P, _T, weights=_W),
    ),
    (
        "kendall_tau",
        lambda: kendall_tau(_P, _T, mask=_KEEP),
        lambda: kendall_tau(_P, _T, weights=_W),
    ),
    ("spearman_rank_correlation", lambda: spearman_rank_correlation(_P, _T, mask=_KEEP), None),
    (
        "relative_l2_error",
        lambda: relative_l2_error(_FIELD_P, _FIELD_T, mask=_SAMPLE_KEEP),
        lambda: relative_l2_error(_FIELD_P, _FIELD_T, weights=_SAMPLE_KEEP.astype(np.float32)),
    ),
]


@pytest.mark.parametrize(("name", "masked", "weighted"), _MASKED, ids=[m[0] for m in _MASKED])
def test_a_mask_alone_runs_no_finiteness_test(
    name: str, masked: Callable[..., object], weighted: Callable[..., object] | None
) -> None:
    """With no weights the mask says everything; with weights the test still runs."""
    assert _counts(masked, isfinite=_isfinite) == {"isfinite": 0}, f"{name} with a mask alone"
    if weighted is not None:
        assert _counts(weighted, isfinite=_isfinite)["isfinite"] > 0, (
            f"{name} with weights no longer guards a non-finite value at a zero weight"
        )
