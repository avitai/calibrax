"""One R-squared formula, reached from all three of its callers.

Before this change the formula existed three times: in ``r_squared``, inline in the fused
regression suite, and again in ``r_squared_adjusted``. Only the first accepted weights, so
weighting ``r_squared`` and then reading ``r_squared_adjusted`` on the same data returned a
weighted number next to an unweighted one.

The tests here pin the agreement between the three callers, the degrees-of-freedom
convention, and that the fused path still produces what it produced before.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import calibrax.metrics._builtin_registrations as registrations
from calibrax.metrics._builtin_registrations import _calculate_regression_fused
from calibrax.metrics._utils import _EPSILON, coefficient_of_determination
from calibrax.metrics.functional.regression import r_squared, r_squared_adjusted
from tests.factories import WeightedSample


#: Predictors for the adjusted variant, matching the recorded references.
NUM_PREDICTORS = 3

#: ``statsmodels``-convention references for the fixture: a mask reduces ``n``, weights do
#: not. Computed with ``sklearn 1.8.0`` and the textbook correction, outside JAX.
ADJUSTED_UNWEIGHTED = 0.758324604304144
ADJUSTED_WEIGHTED = 0.7652735412256604
ADJUSTED_MASKED = 0.7904727522980777

#: What the rejected convention would give, kept so the choice stays observable.
ADJUSTED_IF_N_WERE_SUM_OF_WEIGHTS = 0.7570401171729566


def fusion_count(
    function: Callable[[jnp.ndarray, jnp.ndarray], object],
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
) -> int:
    """Fusions in the HLO ``function`` compiles to, for these argument shapes.

    A proxy for how much work the compiler kept, used only to compare two spellings of
    the same computation against each other in one process -- never against a recorded
    number, which a jax upgrade would invalidate.

    Args:
        function: The function to trace and compile.
        predictions: Predicted values, for shape and dtype.
        targets: Ground truth values, for shape and dtype.

    Returns:
        How many times ``fusion`` appears in the compiled HLO.
    """
    text = jax.jit(function).lower(predictions, targets).compile().as_text()
    assert text is not None, "the compiler returned no HLO text to count"
    return text.count("fusion")


class TestTheThreeCallersAgree:
    """One formula means the three callers cannot disagree about it."""

    def test_r_squared_is_the_shared_core(self, sample: WeightedSample) -> None:
        targets, predictions, weights, mask = sample
        assert float(r_squared(predictions, targets, mask=mask, weights=weights)) == (
            float(coefficient_of_determination(predictions, targets, mask=mask, weights=weights))
        )

    def test_the_fused_suite_matches_r_squared(self, sample: WeightedSample) -> None:
        """The inline copy in the fused path is gone, so these must be identical."""
        targets, predictions, _, _ = sample
        fused = _calculate_regression_fused(predictions, targets)
        assert float(fused["r_squared"]) == float(r_squared(predictions, targets))

    def test_adjusted_reduces_to_r_squared_when_predictors_vanish(
        self, sample: WeightedSample
    ) -> None:
        """With ``p = 0`` the correction is ``(n-1)/(n-1)``, so the two coincide."""
        targets, predictions, weights, _ = sample
        adjusted = r_squared_adjusted(predictions, targets, num_predictors=0, weights=weights)
        plain = r_squared(predictions, targets, weights=weights)
        assert float(adjusted) == pytest.approx(float(plain), abs=1e-6)

    def test_adjusted_is_weighted_when_r_squared_is(self, sample: WeightedSample) -> None:
        """The defect this change fixes: the two used to disagree under weights."""
        targets, predictions, weights, _ = sample
        weighted = float(
            r_squared_adjusted(predictions, targets, num_predictors=NUM_PREDICTORS, weights=weights)
        )
        unweighted = float(r_squared_adjusted(predictions, targets, num_predictors=NUM_PREDICTORS))
        assert weighted != unweighted, (
            "adjusted R-squared ignores weights again, which is the regression this change removed"
        )


class TestTheDegreesOfFreedomConvention:
    """A mask changes ``n``; weights do not. Both halves are observable."""

    def test_unweighted_matches_the_reference(self, sample: WeightedSample) -> None:
        targets, predictions, _, _ = sample
        assert float(
            r_squared_adjusted(predictions, targets, num_predictors=NUM_PREDICTORS)
        ) == pytest.approx(ADJUSTED_UNWEIGHTED, abs=1e-6)

    def test_weights_do_not_change_n(self, sample: WeightedSample) -> None:
        """``statsmodels`` WLS reports ``nobs`` as the observation count, not the weight
        sum. Measured on statsmodels 0.15.0, a separate 40-point WLS fit whose weights
        summed to 25.21 reports ``nobs == 40.0``. This fixture has 80 points and weights
        summing to 43.51, so under the rejected convention its ``n`` would be 43.51."""
        targets, predictions, weights, _ = sample
        assert float(
            r_squared_adjusted(predictions, targets, num_predictors=NUM_PREDICTORS, weights=weights)
        ) == pytest.approx(ADJUSTED_WEIGHTED, abs=1e-6)

    def test_the_rejected_convention_is_measurably_different(self, sample: WeightedSample) -> None:
        """So "weights do not change n" is a decision and not an accident."""
        targets, predictions, weights, _ = sample
        actual = float(
            r_squared_adjusted(predictions, targets, num_predictors=NUM_PREDICTORS, weights=weights)
        )
        assert abs(actual - ADJUSTED_IF_N_WERE_SUM_OF_WEIGHTS) > 1e-3, (
            "the two conventions now agree on this fixture, so it no longer shows which "
            "one is implemented"
        )

    def test_a_mask_does_change_n(self, sample: WeightedSample) -> None:
        """An excluded element is not an observation, so the penalty shrinks with it."""
        targets, predictions, _, mask = sample
        assert float(
            r_squared_adjusted(predictions, targets, num_predictors=NUM_PREDICTORS, mask=mask)
        ) == pytest.approx(ADJUSTED_MASKED, abs=1e-6)

    def test_a_mask_equals_subsetting(self, sample: WeightedSample) -> None:
        targets, predictions, _, mask = sample
        keep = np.asarray(mask)
        assert float(
            r_squared_adjusted(predictions, targets, num_predictors=NUM_PREDICTORS, mask=mask)
        ) == pytest.approx(
            float(
                r_squared_adjusted(
                    jnp.asarray(np.asarray(predictions)[keep]),
                    jnp.asarray(np.asarray(targets)[keep]),
                    num_predictors=NUM_PREDICTORS,
                )
            ),
            abs=1e-6,
        )

    def test_scaling_every_weight_leaves_the_answer_alone(self, sample: WeightedSample) -> None:
        """The property that taking ``sum(weights)`` as ``n`` would break."""
        targets, predictions, weights, _ = sample
        base = float(
            r_squared_adjusted(predictions, targets, num_predictors=NUM_PREDICTORS, weights=weights)
        )
        doubled = float(
            r_squared_adjusted(
                predictions, targets, num_predictors=NUM_PREDICTORS, weights=weights * 2.0
            )
        )
        assert base == pytest.approx(doubled, abs=1e-6)


class TestTransformations:
    """Delegation must not cost traceability, and the fused path must still trace."""

    def test_adjusted_jits_vmaps_and_differentiates(self, sample: WeightedSample) -> None:
        targets, predictions, weights, _ = sample

        def metric(p: jnp.ndarray, t: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
            return r_squared_adjusted(p, t, num_predictors=NUM_PREDICTORS, weights=w)

        assert jnp.isfinite(jax.jit(metric)(predictions, targets, weights))
        batched = jax.jit(jax.vmap(metric))(
            jnp.stack([predictions] * 3), jnp.stack([targets] * 3), jnp.stack([weights] * 3)
        )
        assert batched.shape == (3,)
        assert jnp.all(jnp.isfinite(jax.grad(lambda x: metric(x, targets, weights))(predictions)))

    def test_adjusted_traces_with_a_mask(self, sample: WeightedSample) -> None:
        """``n`` becomes a traced count under a mask, which must not break the trace."""
        targets, predictions, _, mask = sample
        traced = jax.jit(
            lambda p, t, m: r_squared_adjusted(p, t, num_predictors=NUM_PREDICTORS, mask=m)
        )(predictions, targets, mask)
        assert jnp.isfinite(traced)

    def test_the_fused_suite_still_jits(self, sample: WeightedSample) -> None:
        """Delegation was adopted because XLA removes the apparent duplicate reduction."""
        targets, predictions, _, _ = sample
        fused = jax.jit(_calculate_regression_fused)(predictions, targets)
        assert jnp.isfinite(fused["r_squared"])
        assert float(fused["r_squared"]) == pytest.approx(
            float(_calculate_regression_fused(predictions, targets)["r_squared"]), abs=0.0
        )


class TestTheUnweightedPathIsUnchanged:
    """Sharing the core must not move a single existing result, and does not.

    The core takes a separate unweighted path when neither ``mask`` nor ``weights`` is
    given, computing the identical expression the three former copies did -- centred on
    :func:`jax.numpy.mean`, not on ``weighted_mean`` against a ones vector. The earlier
    draft of this change did use a ones vector, and it shifted the answer by up to 2.5
    float32 ULP on 7 of these 240 cases; these tests are what made that visible.
    """

    @staticmethod
    def _former_inline_formula(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        """What all three call sites computed before they shared one core."""
        ss_res = jnp.sum((targets - predictions) ** 2)
        ss_tot = jnp.sum((targets - jnp.mean(targets)) ** 2)
        return 1.0 - ss_res / (ss_tot + _EPSILON)

    @pytest.mark.parametrize("seed", [7, 34])
    @pytest.mark.parametrize("size", [8, 80, 4096])
    @pytest.mark.parametrize("offset", [1.0, 1e4])
    def test_bit_identical_to_the_former_formula(self, seed: int, size: int, offset: float) -> None:
        """Bit-identical, not approximately equal: ``abs=0.0`` is the point of the test.

        Seed 34 at size 80 with the 1e4 offset is the case the ones-vector draft got
        wrong, so it is named rather than left to a grid that might stop covering it.
        """
        rng = np.random.default_rng(seed)
        targets = jnp.asarray(rng.normal(size=size) + offset, dtype=jnp.float32)
        predictions = jnp.asarray(
            np.asarray(targets) + rng.normal(scale=0.4, size=size), dtype=jnp.float32
        )
        assert float(coefficient_of_determination(predictions, targets)) == float(
            self._former_inline_formula(predictions, targets)
        )

    def test_the_residual_spelling_lets_a_suite_reuse_its_own_binding(self) -> None:
        """``p - t``, not ``t - p``, is what lets XLA reuse a caller's own residual.

        The two are bit-equal once squared, so nothing numerical depends on the choice --
        but a fused suite binds ``diff = p - t`` for its other metrics, and XLA reuses that
        binding only for the spelling it already has. Written the other way round the real
        regression suite grew from 118 fusions to 122.

        The comparison must happen inside a function that *has* such a binding: measured on
        the core alone the spelling makes no difference at all, and an earlier version of
        this test compared it standalone and passed with the spelling reversed. Both sides
        are compiled in this process, so a jax upgrade that shifts fusion counts globally
        does not fail this.
        """
        rng = np.random.default_rng(3)
        targets = jnp.asarray(rng.normal(size=4096), dtype=jnp.float32)
        predictions = jnp.asarray(
            np.asarray(targets) + rng.normal(scale=0.4, size=4096), dtype=jnp.float32
        )

        def suite_delegating(p: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
            """A caller shaped like the fused suite: it binds the residual, then delegates."""
            diff = p - t
            return jnp.mean(diff**2) + coefficient_of_determination(p, t)

        def suite_inline(p: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
            """The same, with R-squared reusing the binding directly -- the best case."""
            diff = p - t
            sq_diff = diff**2
            r2 = 1.0 - jnp.sum(sq_diff) / (jnp.sum((t - jnp.mean(t)) ** 2) + _EPSILON)
            return jnp.mean(sq_diff) + r2

        assert fusion_count(suite_delegating, predictions, targets) == fusion_count(
            suite_inline, predictions, targets
        )

    def test_the_real_fused_suite_is_no_larger_for_delegating(self) -> None:
        """The same invariant on the shipped suite, which is what actually ships."""
        rng = np.random.default_rng(3)
        targets = jnp.asarray(rng.normal(size=4096), dtype=jnp.float32)
        predictions = jnp.asarray(
            np.asarray(targets) + rng.normal(scale=0.4, size=4096), dtype=jnp.float32
        )
        shipped = fusion_count(_calculate_regression_fused, predictions, targets)
        source = inspect.getsource(_calculate_regression_fused)
        inlined = source.replace(
            '"r_squared": coefficient_of_determination(p, t),',
            '"r_squared": 1.0 - jnp.sum(sq_diff) / (jnp.sum((t - jnp.mean(t)) ** 2) + _EPSILON),',
        ).replace("def _calculate_regression_fused", "def _reference_fused", 1)
        assert inlined != source, "the delegation line moved; this reference no longer applies"
        namespace = dict(vars(registrations))
        exec(compile(inlined, "<reference>", "exec"), namespace)  # noqa: S102
        reference = fusion_count(namespace["_reference_fused"], predictions, targets)
        assert shipped == reference
