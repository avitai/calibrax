"""``r_squared_adjusted`` belongs beside ``r_squared``, not in the correlation module.

Written before the move, so these tests fail against the old placement and pass after it.

The two are forms of one quantity and now share one implementation
(:func:`calibrax.metrics._utils.coefficient_of_determination`), yet they lived in different
modules and were registered under different domains. The split was accidental rather than
designed, and three things said so:

- ``statistical.py`` describes itself as covering "linear correlation, rank correlation, and
  agreement measures". Adjusted R-squared is a goodness-of-fit measure with a
  degrees-of-freedom penalty, and is none of those.
- ``regression.py`` lists "R-squared" among the metrics it holds.
- ``docs/user-guide/metrics-overview.md`` recorded the statistical domain as holding six
  metrics while naming five. The unnamed sixth was this one, forgotten in the domain it had
  been filed under.

The load-bearing test here is the last one. ``calculate_all`` picks a fused single-pass path
only when the general domain's same-shape metric names match ``_FUSED_REGRESSION_NAMES``
exactly, and **falls back to dynamic dispatch silently** otherwise. Moving a metric into
``general`` is exactly the change that could break that match, and nothing else in the suite
would fail if it did -- the results would stay correct and merely stop being fused.
"""

from __future__ import annotations

import importlib

import pytest

from calibrax.metrics._builtin_registrations import _FUSED_REGRESSION_NAMES
from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import MetricSignature, MetricTier


class TestItLivesInTheRegressionModule:
    """The module that holds ``r_squared`` holds its adjusted form too."""

    def test_importable_from_regression(self) -> None:
        module = importlib.import_module("calibrax.metrics.functional.regression")
        assert hasattr(module, "r_squared_adjusted")

    def test_gone_from_statistical(self) -> None:
        """A clean break: the old home does not keep a re-export.

        Leaving an alias behind would make the split survive the move in every way that
        matters, since both import paths would keep working and neither would be wrong.
        """
        module = importlib.import_module("calibrax.metrics.functional.statistical")
        assert not hasattr(module, "r_squared_adjusted")

    def test_still_exported_from_the_functional_package(self) -> None:
        """The public name is unchanged; only its module and domain move."""
        functional = importlib.import_module("calibrax.metrics.functional")
        assert "r_squared_adjusted" in functional.__all__
        assert hasattr(functional, "r_squared_adjusted")

    def test_the_two_forms_share_a_module(self) -> None:
        regression = importlib.import_module("calibrax.metrics.functional.regression")
        assert regression.r_squared.__module__ == regression.r_squared_adjusted.__module__


class TestItIsRegisteredUnderGeneral:
    """One quantity, one domain.

    **Membership, never a count.** The first draft of this class asserted
    ``len(list_by_domain("general")) == 16``, and it was flaky by construction: the
    registry is a process-wide singleton, ``tests/test_doc_examples.py`` executes
    ``docs/user-guide/metrics-overview.md``, and that document's example registers a
    ``weighted_mse`` metric into the general domain. With ``pytest-randomly`` deciding the
    order, the assertion saw 16 or 17 depending on the seed -- it failed in one full run and
    passed in the next with no code change between them.

    ``scripts/derive_status.py`` already measures these counts correctly, in a subprocess,
    and says why in its own docstring. Counts belong there; this file asserts membership,
    which no ordering can disturb.
    """

    def test_domain_is_general(self) -> None:
        entry = MetricRegistry().get("r_squared_adjusted")
        assert entry is not None, "r_squared_adjusted is no longer registered at all"
        assert entry.domain == "general"

    def test_it_shares_r_squared_s_domain(self) -> None:
        registry = MetricRegistry()
        plain = registry.get("r_squared")
        adjusted = registry.get("r_squared_adjusted")
        assert plain is not None
        assert adjusted is not None
        assert plain.domain == adjusted.domain

    def test_the_general_domain_holds_it(self) -> None:
        names = {entry.name for entry in MetricRegistry().list_by_domain("general")}
        assert "r_squared_adjusted" in names

    def test_the_statistical_domain_does_not(self) -> None:
        names = {entry.name for entry in MetricRegistry().list_by_domain("statistical")}
        assert "r_squared_adjusted" not in names


class TestTheFusedPathSurvives:
    """The silent regression the move could cause, asserted rather than hoped for."""

    @staticmethod
    def _general_same_shape_names() -> set[str]:
        """The set ``calculate_all`` builds to decide whether to take the fused path."""
        registry = MetricRegistry()
        return {
            entry.name
            for entry in registry.list_by_tier(MetricTier.PURE_FUNCTION)
            if entry.domain == "general" and entry.signature == MetricSignature.PREDICTIONS_TARGETS
        }

    def test_the_moved_metric_does_not_join_the_fused_decision_set(self) -> None:
        """The invariant this move could break.

        ``calculate_all`` takes the fused path only when this set matches
        ``_FUSED_REGRESSION_NAMES`` exactly, so a metric arriving in the general domain can
        abandon the fast path with nothing failing and no answer changing. Adjusted
        R-squared must stay out of the set, which it does by signature rather than by
        domain -- see the next test for the mechanism.
        """
        assert "r_squared_adjusted" not in self._general_same_shape_names()

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Pre-existing and unrelated to this move: charbonnier_loss is registered under "
            "general with the PREDICTIONS_TARGETS signature but is absent from "
            "_FUSED_REGRESSION_NAMES and from _calculate_regression_fused, so the decision "
            "set holds 14 names against the fused set's 13 and calculate_all(metrics=None) "
            "silently uses dynamic dispatch. Recorded strict so that fusing charbonnier_loss "
            "turns this test from xpass into a failure, forcing the marker to be removed "
            "rather than leaving a stale exemption behind."
        ),
    )
    def test_the_fused_fast_path_is_reachable_by_default(self) -> None:
        """``calculate_all(metrics=None)`` should reach ``_calculate_regression_fused``."""
        assert self._general_same_shape_names() == set(_FUSED_REGRESSION_NAMES)

    def test_the_moved_metric_is_excluded_by_its_signature(self) -> None:
        """It stays out of the fused set because it needs ``num_predictors``.

        That is the mechanism the test above depends on, so it is named here: the exclusion
        is by signature, not by domain, and it must not quietly become ``PREDICTIONS_TARGETS``.
        """
        entry = MetricRegistry().get("r_squared_adjusted")
        assert entry is not None
        assert entry.signature == MetricSignature.CUSTOM
        assert "r_squared_adjusted" not in _FUSED_REGRESSION_NAMES
