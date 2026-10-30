"""The fused suite and the loss functions must share one expression, not two.

Written before the extraction, so these fail against the duplicated forms.

Three metrics in the fused regression suite carry a numeric shape parameter:
``charbonnier_loss`` (``epsilon``, ``alpha``), ``huber_loss`` (``delta``) and
``quantile_loss`` (``quantile``). The other eleven have none. The fused suite evaluates
every entry at its registered default, so each of those defaults was written twice -- once
as the signature default and once as a literal in the fused body.

**The duplicates were not merely duplicated, they were algebraically collapsed**, which is
why replacing the literals with named constants would have been wrong:

- Huber's linear branch is ``delta * (|e| - 0.5 * delta)``. The fused body had
  ``abs_diff - 0.5``, which equals it only at ``delta == 1.0``.
- Quantile's branches are ``q * r`` and ``(q - 1) * r``. The fused body used ``0.5`` for
  both, which coincides only because ``q == 1 - q`` at ``q == 0.5``.

So the fix shares the elementwise kernel instead. The functions and the fused suite compute
the same residual quantities under different names -- ``err``/``abs_err``/``err**2`` against
``diff``/``abs_diff``/``sq_diff`` -- so a kernel taking those pieces serves both with no
extra work and leaves one copy of each formula.

The load-bearing tests here are the ones that drive a kernel at a **non-default** parameter.
A collapsed expression agrees with its function at the default and nowhere else, so only an
off-default comparison can tell a shared formula from a coincidence.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import huber as scipy_huber
from sklearn.metrics import mean_pinball_loss

from calibrax.metrics._builtin_registrations import _calculate_regression_fused
from calibrax.metrics.functional.regression import (
    CHARBONNIER_ALPHA,
    CHARBONNIER_EPSILON,
    charbonnier_loss,
    HUBER_DELTA,
    huber_loss,
    QUANTILE_LEVEL,
    quantile_loss,
)


#: Float32 budget. The kernels are the same expressions the functions use, so agreement is
#: expected to be exact; this leaves room only for XLA reassociating a shared reduction.
TOLERANCE = 1e-6

SIZE = 512


def _build_series() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Predictions and targets with errors on both sides of any sensible threshold."""
    rng = np.random.default_rng(0)
    targets = jnp.asarray(rng.normal(size=SIZE), dtype=jnp.float32)
    # scale 1.5 puts a good fraction of |error| above and below delta = 1.0
    predictions = targets + jnp.asarray(rng.normal(scale=1.5, size=SIZE), dtype=jnp.float32)
    return predictions, targets


@pytest.fixture
def series() -> tuple[jnp.ndarray, jnp.ndarray]:
    """The shared fixture, injected into the comparison tests."""
    return _build_series()


class TestTheDefaultsAreDeclaredOnce:
    """Each shape parameter's default is a named constant, used as the signature default."""

    @pytest.mark.parametrize(
        ("constant", "expected"),
        [
            (HUBER_DELTA, 1.0),
            (QUANTILE_LEVEL, 0.5),
            (CHARBONNIER_EPSILON, 1e-3),
            (CHARBONNIER_ALPHA, 1.0),
        ],
    )
    def test_the_constant_holds_the_documented_value(
        self, constant: float, expected: float
    ) -> None:
        assert constant == expected

    def test_the_signature_defaults_come_from_the_constants(self) -> None:
        """So a change to the constant moves the function and the fused suite together."""
        assert inspect.signature(huber_loss).parameters["delta"].default == HUBER_DELTA
        assert inspect.signature(quantile_loss).parameters["quantile"].default == QUANTILE_LEVEL
        assert (
            inspect.signature(charbonnier_loss).parameters["epsilon"].default == CHARBONNIER_EPSILON
        )


class TestTheKernelsGeneralise:
    """Off-default comparison against **independent** implementations.

    The first draft of this class compared the kernel against the function that calls it.
    That proves nothing once they share an expression: collapsing Huber's linear branch to
    ``|e| - 0.5`` and giving both quantile branches the same coefficient left every
    assertion passing, because each broken kernel was being compared with itself. Positive
    controls caught it; the comparison did not.

    So the references here are outside calibrax. ``scipy.special.huber`` implements the
    Huber function directly, and ``sklearn.metrics.mean_pinball_loss`` the pinball loss.
    Both are driven away from the defaults, which is the only way to tell a general formula
    from one that happens to agree at a single parameter value.

    Cross-checked against a third implementation each before these tests were written.
    Huber agreed across ``calibrax``, ``scipy.special.huber`` and ``optax.huber_loss`` at
    deltas 0.25, 0.5, 1, 2 and 7. Quantile agreed across ``calibrax``,
    ``sklearn.metrics.mean_pinball_loss`` and the closed form
    ``mean(max(q * r, (q - 1) * r))`` at levels 0.05, 0.1, 0.5, 0.9 and 0.95.

    **Huber has two conventions in the wild, and they agree at the default.** The form here,
    shared by scipy and optax, is ``0.5 * e^2`` against ``delta * (|e| - 0.5 * delta)``.
    PyTorch's ``smooth_l1_loss`` instead divides the quadratic branch by its ``beta``, which
    is this loss divided by ``delta``. Measured, the two differ at every delta tried except
    ``delta == 1.0``, where they coincide to every digit -- so a test that only ever used
    the default could not tell which of the two was implemented.
    """

    @pytest.mark.parametrize("delta", [0.25, 0.5, 1.0, 2.0, 3.0])
    def test_huber_matches_scipy_at_any_delta(
        self, series: tuple[jnp.ndarray, jnp.ndarray], delta: float
    ) -> None:
        """A linear branch written ``|e| - 0.5`` passes only at ``delta == 1.0``.

        Every delta here puts a real share of ``|e|`` above the knee on this fixture, from
        86% at 0.25 down to 3% at 3.0, so each case exercises the linear branch. An earlier
        version used 7.0, where **no** error exceeds the threshold: both forms then take the
        quadratic branch and the case cannot detect a wrong linear branch at all. Huber
        degenerating to half the mean squared error at large delta is a real property, but
        it is not this test's job.
        """
        predictions, targets = series
        error = np.asarray(predictions) - np.asarray(targets)
        reference = float(np.mean(scipy_huber(delta, error)))
        assert float(huber_loss(predictions, targets, delta=delta)) == pytest.approx(
            reference, abs=TOLERANCE
        )

    @pytest.mark.parametrize("quantile", [0.05, 0.1, 0.5, 0.9, 0.95])
    def test_quantile_matches_sklearn_at_any_level(
        self, series: tuple[jnp.ndarray, jnp.ndarray], quantile: float
    ) -> None:
        """One coefficient for both branches passes only at ``q == 0.5``."""
        predictions, targets = series
        reference = float(
            mean_pinball_loss(np.asarray(targets), np.asarray(predictions), alpha=quantile)
        )
        assert float(quantile_loss(predictions, targets, quantile=quantile)) == pytest.approx(
            reference, abs=TOLERANCE
        )

    def test_the_fixture_would_expose_a_collapsed_expression(self) -> None:
        """Guards the fixture, not the code: an in-test positive control.

        If the fixture's errors all sat on one side of every threshold, or on one side of
        zero, the parametrised tests above would pass against a collapsed expression. So
        rather than merely checking that the references move with the parameter, this
        computes what each collapsed form would give and asserts the gap is far larger than
        the tolerance those tests use.

        Measured on this fixture: the smallest gap is 0.48 for quantile at 0.1 or 0.9, and
        0.18 for Huber at delta 2.0, against a tolerance of 1e-6. Both are five orders of
        magnitude of headroom, so the comparisons are not passing on luck.
        """
        predictions, targets = _build_series()
        error = np.asarray(predictions) - np.asarray(targets)

        # Huber, collapsed to the delta == 1 form, at a delta where that is wrong.
        absolute = np.abs(error)
        for delta in (0.25, 0.5, 2.0):
            correct = float(np.mean(scipy_huber(delta, error)))
            collapsed = float(np.mean(np.where(absolute <= delta, 0.5 * error**2, absolute - 0.5)))
            assert abs(correct - collapsed) > 1000 * TOLERANCE

        # Quantile, collapsed to one coefficient for both branches, away from one half.
        for quantile in (0.1, 0.9):
            correct = float(
                mean_pinball_loss(np.asarray(targets), np.asarray(predictions), alpha=quantile)
            )
            collapsed = float(np.mean(np.where(error <= 0, -quantile * error, quantile * error)))
            assert abs(correct - collapsed) > 1000 * TOLERANCE


class TestTheFusedSuiteUsesTheKernels:
    """The point of the extraction: the fused values are the functions, not copies."""

    def test_fused_huber_and_quantile_match_their_functions(
        self, series: tuple[jnp.ndarray, jnp.ndarray]
    ) -> None:
        predictions, targets = series
        fused = _calculate_regression_fused(predictions, targets)
        assert float(fused["huber_loss"]) == pytest.approx(
            float(huber_loss(predictions, targets)), abs=TOLERANCE
        )
        assert float(fused["quantile_loss"]) == pytest.approx(
            float(quantile_loss(predictions, targets)), abs=TOLERANCE
        )

    def test_no_shape_parameter_literal_remains_in_the_fused_body(self) -> None:
        """A grep-style guard, so a future metric cannot quietly re-introduce one.

        Checks the fused function's own source for the collapsed literals that were there.
        A positive control is built in: the constants' names must appear, so the check
        cannot pass by reading the wrong function.
        """
        source = inspect.getsource(_calculate_regression_fused)
        body = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))
        assert "HUBER_DELTA" in body, "positive control: read the wrong function"
        assert "QUANTILE_LEVEL" in body, "positive control: read the wrong function"
        assert "abs_diff - 0.5" not in body
        assert "0.5 * (-diff)" not in body
