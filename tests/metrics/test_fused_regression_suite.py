"""The fused regression suite must cover what ``calculate_all`` asks of it.

Written before the fix, so these fail against the gap they describe.

``calculate_all(metrics=None)`` builds its candidate set from the general domain and the
``PREDICTIONS_TARGETS`` signature, then takes the fused single-pass path **only on an exact
match** against ``_FUSED_REGRESSION_NAMES``, falling back to dynamic dispatch otherwise. The
fallback is silent and returns correct answers, so a metric arriving in the general domain
without being added to the fused suite disables the fast path with nothing failing.

That is what had happened. ``charbonnier_loss`` was registered under general with the
same-shape signature and was absent from both the fused name set and the fused function, so
the candidate set held 14 names against the fused set's 13 and the default call had been
using dynamic dispatch. Observed directly by spying on the fused function rather than by
re-deriving the decision: ``calculate_all(metrics=None)`` did not call it, while
``calculate_all(metrics=<the 13>)`` did.

The tests here close the gap and keep it closed: one pins the set equality that selects the
fast path, one pins that the fused value equals the function it stands in for, and one pins
that fused and dynamic dispatch agree on every metric, which is the property that makes the
fast path a substitution rather than a second implementation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.metrics._builtin_registrations import (
    _calculate_regression_fused,
    _FUSED_REGRESSION_NAMES,
    calculate_all,
)
from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import MetricSignature, MetricTier
from calibrax.metrics.functional.regression import charbonnier_loss


#: Float32 agreement budget. The fused forms are the same expressions, so the values are
#: expected to be bit-equal; this leaves room only for XLA reassociating a shared reduction.
TOLERANCE = 1e-6

SIZE = 4096


@pytest.fixture
def series() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Predictions and targets, reproducible from one seed."""
    rng = np.random.default_rng(0)
    targets = jnp.asarray(rng.normal(size=SIZE), dtype=jnp.float32)
    predictions = targets + jnp.asarray(rng.normal(scale=0.4, size=SIZE), dtype=jnp.float32)
    return predictions, targets


def _candidate_names() -> set[str]:
    """The set ``calculate_all`` compares against ``_FUSED_REGRESSION_NAMES``."""
    return {
        entry.name
        for entry in MetricRegistry().list_by_tier(MetricTier.PURE_FUNCTION)
        if entry.domain == "general" and entry.signature == MetricSignature.PREDICTIONS_TARGETS
    }


class TestTheFastPathIsReachable:
    """The selection condition, asserted rather than assumed."""

    def test_the_candidate_set_matches_the_fused_set(self) -> None:
        """Membership comparison, not a count, so test ordering cannot disturb it."""
        assert _candidate_names() == set(_FUSED_REGRESSION_NAMES)

    def test_the_fused_suite_returns_exactly_the_names_it_declares(self) -> None:
        returned = set(_calculate_regression_fused(jnp.ones(4), jnp.zeros(4)))
        assert returned == set(_FUSED_REGRESSION_NAMES)

    def test_the_default_call_reaches_the_fused_suite(
        self, series: tuple[jnp.ndarray, jnp.ndarray], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spy on the fused function instead of re-deriving the decision logic.

        Reproducing the condition in a test can agree with a broken implementation. This
        observes the call.
        """
        predictions, targets = series
        calls: list[int] = []
        module = __import__(
            "calibrax.metrics._builtin_registrations", fromlist=["_calculate_regression_fused"]
        )
        original = module._calculate_regression_fused

        def spy(p: jnp.ndarray, t: jnp.ndarray) -> dict[str, jnp.ndarray]:
            calls.append(1)
            return original(p, t)

        monkeypatch.setattr(module, "_calculate_regression_fused", spy)
        calculate_all(predictions, targets)
        assert calls, "calculate_all(metrics=None) did not use the fused single-pass path"


class TestTheFusedValuesAreTheFunctionsTheyReplace:
    """A fast path is only a substitution if it agrees with what it substitutes for."""

    def test_charbonnier_matches_its_functional_form(
        self, series: tuple[jnp.ndarray, jnp.ndarray]
    ) -> None:
        """At the registered defaults, which is how ``calculate_all`` calls it."""
        predictions, targets = series
        fused = float(_calculate_regression_fused(predictions, targets)["charbonnier_loss"])
        assert fused == pytest.approx(float(charbonnier_loss(predictions, targets)), abs=TOLERANCE)

    def test_fused_and_dynamic_dispatch_agree_on_every_metric(
        self, series: tuple[jnp.ndarray, jnp.ndarray]
    ) -> None:
        """Every fused entry against the registry's own function for that name."""
        predictions, targets = series
        fused = _calculate_regression_fused(predictions, targets)
        registry = MetricRegistry()
        for name in sorted(_FUSED_REGRESSION_NAMES):
            entry = registry.get(name)
            assert entry is not None, f"{name} is in the fused set but not registered"
            assert entry.fn is not None, f"{name} has no registered function"
            assert float(fused[name]) == pytest.approx(
                float(entry.fn(predictions, targets)), abs=TOLERANCE
            ), f"fused {name} disagrees with its registered function"
