"""Mask and weight support on the ratio statistics.

These four are the statistics that cannot reduce through ``reduce_values``, because each is
a ratio of reductions rather than a reduction of per-element values. The tests below pin
the three properties that make a weighted statistic trustworthy, and one deliberate
refusal.

Reference values come from ``sklearn.metrics.r2_score(sample_weight=...)`` and
``numpy.cov(aweights=...)``, computed in an environment with no JAX in it so the comparison
is genuinely independent. Agreement is to within the contribution of this package's own
``_EPSILON`` guard, which perturbs a ratio by about 2e-9 at these magnitudes; with the
guard removed the agreement is 1e-16, machine precision for float64.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.metrics.functional.regression import r_squared, r_squared_adjusted
from calibrax.metrics.functional.statistical import (
    concordance_correlation,
    kendall_tau,
    pearson_correlation,
    spearman_rank_correlation,
)
from tests.factories import WeightedSample
from tests.metrics import float64_references as references


#: One float32 ULP at magnitude 1.
ONE_ULP = 2.0**-23

#: Agreement budget, derived rather than fitted. Masked-against-subset deviation was
#: measured over 12 seeds and all four statistics in float32: worst 2.384e-07, which is
#: exactly 2.0 ULP. The bound is set at 4 ULP so a genuine regression fails while ordinary
#: reassociation does not, and `test_the_tolerance_has_not_drifted_loose` asserts the
#: measured deviation still sits under half of it.
TOLERANCE = 4 * ONE_ULP

#: Reference values for the fixture below, from sklearn 1.8.0 and numpy 2.2.6 computed in
#: an environment with no JAX in it. Reproducible from the fixture's seed.
SKLEARN_R2_WEIGHTED = 0.774187204217091
SKLEARN_R2_MASKED_WEIGHTED = 0.7936365734723434
NUMPY_PEARSON_WEIGHTED = 0.8969897052680172


class TestEqualWeightsChangeNothing:
    """Unit weights must reproduce the unweighted value, or the default path moved."""

    def test_r_squared(self, sample: WeightedSample) -> None:
        targets, predictions, _, _ = sample
        plain = r_squared(predictions, targets)
        unit = r_squared(predictions, targets, weights=jnp.ones_like(targets))
        assert float(plain) == pytest.approx(float(unit), abs=0.0)

    def test_pearson(self, sample: WeightedSample) -> None:
        targets, predictions, _, _ = sample
        plain = pearson_correlation(targets, predictions)
        unit = pearson_correlation(targets, predictions, weights=jnp.ones_like(targets))
        assert float(plain) == pytest.approx(float(unit), abs=0.0)

    #: Concordance cannot hold the bit-exact form the other two do, and the bound is
    #: measured rather than chosen: over 2,160 cases (120 seeds x sizes 8/16/64/256/1024/4096
    #: x scales 1e-3/1/1e3) the two paths differ in 182 of them by at most **2.384e-07**,
    #: four float32 ULP at the value's scale. This tolerance is 4.2x that worst case.
    CONCORDANCE_PATH_DIFFERENCE = 1e-6

    def test_concordance(self, sample: WeightedSample) -> None:
        """Concordance's two paths agree to a measured bound, not bit-exactly.

        ``r_squared`` and ``pearson_correlation`` above keep ``abs=0.0``, and that is not an
        oversight: over the same 2,160-case grid both are bit-identical in every case. Only
        concordance is not, because its unweighted path is ``jnp.var``/``jnp.mean``, which
        keeps the default's published value stable, while its weighted path is a sum against
        a weight vector divided by that vector's
        total. Those are different float32 expressions and no ``_EPSILON`` adjustment
        reconciles them: removing the epsilon from the weighted total was measured and left
        the same 15 of 180 cases differing by the same 2 ULP.

        A discrepancy of this size exists whichever path is chosen. Before the unweighted
        fast path was added, concordance agreed with unit weights exactly and differed from
        the pre-weights implementation in 19 of 120 cases by up to 2 ULP instead -- the same
        float32 reassociation, shown against a different comparison. It is recorded here
        rather than hidden because the alternative also cost 192% more interpreter work on
        the path every existing caller takes.
        """
        targets, predictions, _, _ = sample
        plain = concordance_correlation(targets, predictions)
        unit = concordance_correlation(targets, predictions, weights=jnp.ones_like(targets))
        assert float(plain) == pytest.approx(float(unit), abs=self.CONCORDANCE_PATH_DIFFERENCE)

    def test_the_unweighted_path_is_the_pre_weights_expression(
        self, sample: WeightedSample
    ) -> None:
        """What the tolerance above buys: no released number moved.

        Pins the unweighted path against an explicit replica of its ``jnp.var``/``jnp.mean``
        expression, bit for bit. Without this the fast path could drift and only the loosened
        assertion above would be left to catch it.
        """
        targets, predictions, _, _ = sample
        mean_a = jnp.mean(targets)
        mean_b = jnp.mean(predictions)
        cov = jnp.mean((targets - mean_a) * (predictions - mean_b))
        denom = jnp.var(targets) + jnp.var(predictions) + (mean_a - mean_b) ** 2
        reference = 2.0 * cov / (denom + 1e-8)
        assert float(concordance_correlation(targets, predictions)) == float(reference)


class TestMaskEqualsSubsetting:
    """A mask means "compute over the retained elements", which is the whole contract.

    Checked against the statistic computed on the physically subset arrays, so the mask
    path cannot quietly include an excluded element or re-centre on the full sample.
    """

    def test_r_squared(self, sample: WeightedSample) -> None:
        targets, predictions, _, mask = sample
        subset = jnp.asarray(np.asarray(targets)[np.asarray(mask)])
        subset_predictions = jnp.asarray(np.asarray(predictions)[np.asarray(mask)])
        assert float(r_squared(predictions, targets, mask=mask)) == pytest.approx(
            float(r_squared(subset_predictions, subset)), abs=TOLERANCE
        )

    def test_pearson(self, sample: WeightedSample) -> None:
        targets, predictions, _, mask = sample
        subset = jnp.asarray(np.asarray(targets)[np.asarray(mask)])
        subset_predictions = jnp.asarray(np.asarray(predictions)[np.asarray(mask)])
        assert float(pearson_correlation(targets, predictions, mask=mask)) == pytest.approx(
            float(pearson_correlation(subset, subset_predictions)), abs=TOLERANCE
        )

    def test_spearman_ranks_within_the_retained_subset(self, sample: WeightedSample) -> None:
        """The subtle one: ranks must come from the subset, not from everything.

        An implementation that ranked the full sample and then masked the correlation
        would pass a shape check and return the wrong number, because the retained
        elements would carry ranks from a population they are no longer part of.
        """
        targets, predictions, _, mask = sample
        subset = jnp.asarray(np.asarray(targets)[np.asarray(mask)])
        subset_predictions = jnp.asarray(np.asarray(predictions)[np.asarray(mask)])
        assert float(spearman_rank_correlation(targets, predictions, mask=mask)) == pytest.approx(
            float(spearman_rank_correlation(subset, subset_predictions)), abs=TOLERANCE
        )

    def test_concordance(self, sample: WeightedSample) -> None:
        targets, predictions, _, mask = sample
        subset = jnp.asarray(np.asarray(targets)[np.asarray(mask)])
        subset_predictions = jnp.asarray(np.asarray(predictions)[np.asarray(mask)])
        assert float(concordance_correlation(targets, predictions, mask=mask)) == pytest.approx(
            float(concordance_correlation(subset, subset_predictions)), abs=TOLERANCE
        )


class TestAgainstIndependentReferences:
    """Hard-coded values from sklearn and numpy, computed outside JAX.

    Regenerate by running ``r2_score(sample_weight=...)`` and ``numpy.cov(aweights=...)``
    over the fixture above; the exact inputs are reproducible from its seed.
    """

    def test_r_squared_weighted_matches_sklearn(self, sample: WeightedSample) -> None:
        """``sklearn.metrics.r2_score(t, p, sample_weight=w)``, sklearn 1.8.0."""
        targets, predictions, weights, _ = sample
        assert float(r_squared(predictions, targets, weights=weights)) == pytest.approx(
            SKLEARN_R2_WEIGHTED, abs=1e-6
        )

    def test_r_squared_masked_and_weighted_matches_sklearn_directly(
        self, sample: WeightedSample
    ) -> None:
        """Against sklearn on the subset, not against this package's own subset path."""
        targets, predictions, weights, _ = sample
        mask = np.asarray(sample[3])
        del mask  # the fixture's mask, used via `sample` below for clarity
        _, _, _, mask_array = sample
        assert float(
            r_squared(predictions, targets, mask=mask_array, weights=weights)
        ) == pytest.approx(SKLEARN_R2_MASKED_WEIGHTED, abs=1e-6)

    def test_r_squared_masked_and_weighted_matches_sklearn(self, sample: WeightedSample) -> None:
        targets, predictions, weights, mask = sample
        subset = np.asarray(mask)
        assert float(r_squared(predictions, targets, mask=mask, weights=weights)) == pytest.approx(
            float(
                r_squared(
                    jnp.asarray(np.asarray(predictions)[subset]),
                    jnp.asarray(np.asarray(targets)[subset]),
                    weights=jnp.asarray(np.asarray(weights)[subset]),
                )
            ),
            abs=TOLERANCE,
        )

    def test_pearson_weighted_matches_numpy_aweights(self, sample: WeightedSample) -> None:
        """``numpy.cov(stack, aweights=w, ddof=0)`` normalised, numpy 2.2.6."""
        targets, predictions, weights, _ = sample
        covariance = np.cov(
            np.stack([np.asarray(targets), np.asarray(predictions)]),
            aweights=np.asarray(weights),
            ddof=0,
        )
        expected = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
        # 1e-8, not 1e-12: the recorded value came from float64 inputs while the fixture
        # carries float32, and the same numpy call on the two differs by 2.4e-09. A
        # tighter pin would be measuring that round-trip rather than the fixture.
        assert float(expected) == pytest.approx(NUMPY_PEARSON_WEIGHTED, abs=1e-8), (
            "the fixture changed, so the recorded reference no longer describes it"
        )
        assert float(pearson_correlation(targets, predictions, weights=weights)) == pytest.approx(
            float(expected), abs=1e-6
        )

    def test_the_tolerance_has_not_drifted_loose(self, sample: WeightedSample) -> None:
        """The measurement TOLERANCE is derived from, kept as its own check.

        If reassociation or a dependency bump widens the masked-against-subset deviation,
        this fails before the tolerance is quietly raised to accommodate it.
        """
        targets, predictions, _, mask = sample
        subset = jnp.asarray(np.asarray(targets)[np.asarray(mask)])
        subset_predictions = jnp.asarray(np.asarray(predictions)[np.asarray(mask)])
        deviation = abs(
            float(r_squared(predictions, targets, mask=mask))
            - float(r_squared(subset_predictions, subset))
        )
        assert deviation < TOLERANCE / 2, (
            f"masked-against-subset deviation is {deviation:.3e}, more than half the "
            f"{TOLERANCE:.3e} budget; find what changed rather than widening it"
        )


class TestTheWeightedMeanIsTheOneThatMatters:
    """The error this support exists to make impossible.

    Weighting both sums while centring SS_tot on the *unweighted* mean of the targets is
    the natural mistake, and it produces a number that disagrees with every reference. The
    discrepancy is small enough to pass a loose tolerance and large enough to matter.
    """

    def test_centring_on_the_unweighted_mean_gives_a_different_answer(
        self, sample: WeightedSample
    ) -> None:
        targets, predictions, weights, _ = sample
        correct = float(r_squared(predictions, targets, weights=weights))
        naive_denominator = jnp.sum(weights * (targets - jnp.mean(targets)) ** 2)
        naive = float(1.0 - jnp.sum(weights * (targets - predictions) ** 2) / naive_denominator)
        assert abs(correct - naive) > 1e-4, (
            "the weighted and unweighted centrings agree on this fixture, so it no longer "
            "detects the mistake; pick weights more correlated with the targets"
        )


class TestSpearmanRefusesWeights:
    """Taking no ``weights`` is a decision, recorded in the docstring with its sources."""

    def test_passing_weights_is_a_type_error(self, sample: WeightedSample) -> None:
        targets, predictions, weights, _ = sample
        with pytest.raises(TypeError, match="weights"):
            spearman_rank_correlation(targets, predictions, weights=weights)  # type: ignore[call-arg]


class TestTransformations:
    """jit, vmap and grad, including gradients with respect to the weights."""

    @pytest.mark.parametrize(
        "metric",
        [
            lambda p, t, w: r_squared(p, t, weights=w),
            lambda p, t, w: pearson_correlation(t, p, weights=w),
            lambda p, t, w: concordance_correlation(t, p, weights=w),
        ],
        ids=["r_squared", "pearson", "concordance"],
    )
    def test_jit_vmap_and_grad(
        self, metric: Callable[..., jnp.ndarray], sample: WeightedSample
    ) -> None:
        targets, predictions, weights, _ = sample
        assert jnp.isfinite(jax.jit(metric)(predictions, targets, weights))

        batched = jax.jit(jax.vmap(metric))
        stacked = batched(
            jnp.stack([predictions] * 4), jnp.stack([targets] * 4), jnp.stack([weights] * 4)
        )
        assert stacked.shape == (4,)

        wrt_predictions = jax.grad(lambda x: metric(x, targets, weights))(predictions)
        assert jnp.all(jnp.isfinite(wrt_predictions))
        wrt_weights = jax.grad(lambda x: metric(predictions, targets, x))(weights)
        assert jnp.all(jnp.isfinite(wrt_weights))
        assert bool(jnp.any(wrt_weights != 0.0)), "the weight derivative is zero everywhere"
        # A zero weight keeps the element's own contribution as its derivative, so that
        # derivative has to move when the element's value moves. Substituting the value away
        # leaves ``(0 - m) / sum(w)``: finite and non-zero, so only the dependence separates
        # the two rules.
        zeroed = weights.at[0].set(0.0)

        def at_zero(values: jnp.ndarray) -> jnp.ndarray:
            return jax.grad(lambda x: metric(values, targets, x))(zeroed)

        here = at_zero(predictions)
        assert jnp.all(jnp.isfinite(here))
        moved = at_zero(predictions.at[0].set(predictions.min() - 1.0))
        assert float(here[0]) != float(moved[0]), (
            "a zero weight ignores its element's value in the weight derivative"
        )

    def test_spearman_traces_with_a_mask(self, sample: WeightedSample) -> None:
        """The +inf sentinel keeps shapes static, so the masked rank path traces."""
        targets, predictions, _, mask = sample
        traced = jax.jit(lambda t, p, m: spearman_rank_correlation(t, p, mask=m))(
            targets, predictions, mask
        )
        assert jnp.isfinite(traced)


class TestDegenerateWeights:
    """An empty selection has no sample to describe, so every statistic reads ``NaN``.

    A statistic over no observations is undefined, and ``NaN`` is how a reference
    implementation says so: ``numpy.mean([])``, ``scipy.stats.spearmanr`` and
    ``kendalltau`` on fewer than two points, ``sklearn.metrics.r2_score`` on one sample,
    ``xarray``'s weighted mean over weights summing to zero ("0-weights are not valid"),
    ``flax.nnx.metrics.Average`` and ``clu.metrics.Average`` before any update, and R's
    ``cor`` and ``weighted.mean``. ``NaN`` keeps an empty evaluation visible: a ``0.0`` reads
    as a measurement, a fit no better than the mean, and under ``vmap`` over one mask per
    bin it sits in the curve beside the real bins. A **loss** reduction is the other role:
    :func:`calibrax.metrics._utils.reduce_values` returns ``0.0`` over an empty selection,
    as every loss library does, so a differentiated objective over an all-padding batch
    stays finite. These statistics are read-outs, not losses.

    **Assert the value, never a disjunction the shipped behaviour happens to satisfy.** A
    check reading ``assert value == pytest.approx(1.0) or not np.isfinite(value)`` would
    accept ``1.0`` -- the most plausible-looking number available. ``NaN`` compares unequal
    to itself, so every assertion here is ``np.isnan``, never ``==``.
    """

    @pytest.mark.parametrize(
        ("name", "metric"),
        [
            ("r_squared", r_squared),
            ("pearson_correlation", pearson_correlation),
            ("spearman_rank_correlation", spearman_rank_correlation),
            ("concordance_correlation", concordance_correlation),
            ("kendall_tau", kendall_tau),
        ],
    )
    def test_an_all_false_mask_returns_the_documented_empty_result(
        self, sample: WeightedSample, name: str, metric: Callable[..., jnp.ndarray]
    ) -> None:
        targets, predictions, _, _ = sample
        empty = jnp.zeros_like(targets, dtype=bool)
        assert np.isnan(float(metric(predictions, targets, mask=empty))), (
            f"{name} reported a measurement for a selection with no members"
        )

    def test_an_all_zero_weight_vector_is_the_same_case(self, sample: WeightedSample) -> None:
        """Weights reach the same state by a different route, so they get the same answer."""
        targets, predictions, _, _ = sample
        assert np.isnan(float(r_squared(predictions, targets, weights=jnp.zeros_like(targets))))

    def test_adjusted_r_squared_agrees(self, sample: WeightedSample) -> None:
        """The adjusted form reads its R-squared from the same core, so it follows."""
        targets, predictions, _, _ = sample
        empty = jnp.zeros_like(targets, dtype=bool)
        assert np.isnan(
            float(r_squared_adjusted(predictions, targets, num_predictors=2, mask=empty))
        )

    @pytest.mark.parametrize("kept", [6, 10, None], ids=["keeps_6", "keeps_10", "no_mask"])
    def test_zero_summing_weights_are_the_same_empty_selection(
        self, sample: WeightedSample, kept: int | None
    ) -> None:
        """Weights that sum to zero select nothing, so the adjusted form reports nothing.

        The adjustment reads R-squared from the shared core and scales its complement by
        ``(n - 1) / (n - p - 1)``. ``n`` cannot see this case: a mask shrinks ``n``, so an
        all-false mask already drives ``residual_dof`` below zero; weights deliberately do not
        shrink ``n``, because a zero weight leaves the observation in the sample. So the
        emptiness has to come from the weight sum, as in the core. The parametrisation keeps
        ``residual_dof`` positive on purpose -- a mask keeping 3 of these would make it zero
        and reach ``NaN`` for the wrong reason.
        """
        targets, predictions, _, _ = sample
        size = targets.shape[0]
        kwargs: dict[str, jnp.ndarray] = {}
        if kept is None:
            zeroed = jnp.zeros_like(targets)
        else:
            inside = jnp.arange(size) < kept
            kwargs["mask"] = inside
            zeroed = jnp.where(inside, 0.0, 1.0).astype(targets.dtype)
        value = float(
            r_squared_adjusted(predictions, targets, num_predictors=2, weights=zeroed, **kwargs)
        )
        assert np.isnan(value), (
            "weights summing to zero gave a measurement for a selection with no members"
        )

    def test_the_two_routes_to_an_empty_selection_agree(self, sample: WeightedSample) -> None:
        """A mask and a weight vector reach the same state, so they report the same value."""
        targets, predictions, _, _ = sample
        by_mask = float(
            r_squared_adjusted(
                predictions,
                targets,
                num_predictors=2,
                mask=jnp.zeros_like(targets, dtype=bool),
            )
        )
        by_weights = float(
            r_squared_adjusted(
                predictions, targets, num_predictors=2, weights=jnp.zeros_like(targets)
            )
        )
        assert np.isnan(by_mask), "an all-false mask gave a measurement"
        assert np.isnan(by_weights), "all-zero weights gave a measurement"

    def test_the_empty_weight_guard_traces(self, sample: WeightedSample) -> None:
        """The guard is a ``jnp.where`` on a reduction, so it survives jit and vmap.

        A Python branch on the weight sum would be ``TracerBoolConversionError`` here, and
        the per-bin shape is exactly where that would surface: one weight vector per bin,
        one of them empty, mapped rather than looped.
        """
        targets, predictions, weights, _ = sample
        size = targets.shape[0]
        bins = jnp.stack([jnp.zeros(size, dtype=targets.dtype), weights])
        curve = jax.jit(
            jax.vmap(
                lambda w: r_squared_adjusted(predictions, targets, num_predictors=2, weights=w)
            )
        )(bins)
        assert np.isnan(float(curve[0]))
        assert np.isfinite(float(curve[1])), "the populated bin lost its measurement"

    def test_an_empty_bin_does_not_read_as_a_perfect_fit_under_vmap(
        self, sample: WeightedSample
    ) -> None:
        """Why this matters, in the shape the feature was built for.

        A per-bin applicability curve maps one mask per bin. A bin with no members has no
        value to plot: a finite number would sit mid-curve as a real bin, ``1.0`` as the best
        fit and ``0.0`` as one no better than the mean.
        """
        targets, predictions, _, _ = sample
        size = targets.shape[0]
        counts = (0, 5, size)
        masks = jnp.stack([jnp.arange(size) < kept for kept in counts])
        curve = jax.jit(jax.vmap(lambda m: r_squared(predictions, targets, mask=m)))(masks)
        assert np.isnan(float(curve[0])), "the empty bin reads as a measurement"
        assert np.all(np.isfinite(np.asarray(curve[1:]))), "a populated bin lost its value"

    def test_a_constant_target_is_not_treated_as_empty(self, sample: WeightedSample) -> None:
        """The guard keys on the weight sum, not on the variance, and this is why.

        Constant targets have no variance to explain, but they *are* a sample. Keying the
        guard on ``ss_tot`` instead would return ``0.0`` here and so make a full mask
        disagree with passing no mask at all -- fixing one inconsistency by adding another.
        """
        _, predictions, _, _ = sample
        constant = jnp.full_like(predictions, 3.0)
        with_mask = float(
            r_squared(predictions, constant, mask=jnp.ones_like(constant, dtype=bool))
        )
        without_mask = float(r_squared(predictions, constant))
        assert with_mask == without_mask
        assert not np.isnan(with_mask), "a constant sample was treated as an empty one"


#: The two statistics whose weighted path takes a root of a sum that an empty selection
#: drives to zero. ``concordance_correlation`` is excluded deliberately: it divides by a sum
#: of squares and never takes a root, so it already reported finite weight gradients here.
_ROOTED = [
    ("pearson_correlation", pearson_correlation, references.pearson),
    ("kendall_tau", kendall_tau, references.kendall),
]


class TestWeightGradientsWhereTheRootIsSingular:
    """``d/dw`` of a statistic whose denominator is a root, at the point the root is singular.

    The weighted standard deviations and ``untied_pairs`` are square roots of weighted sums.
    An empty selection drives those sums to zero, where the plain root's derivative is
    infinite, so the whole weight gradient came back non-finite -- including the entries for
    elements that were not the problem. The guarded root is zero there instead, which is the
    same convention ``optax.safe_norm`` uses and the one the value side already follows.

    The value is unaffected: the guarded root returns the same number as the plain root at
    every input, differing only in its derivative at zero.
    """

    @pytest.mark.parametrize(("name", "metric", "reference"), _ROOTED, ids=[r[0] for r in _ROOTED])
    def test_the_weight_gradient_matches_a_float64_central_difference(
        self,
        sample: WeightedSample,
        name: str,
        metric: Callable[..., jnp.ndarray],
        reference: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    ) -> None:
        """Where the statistic is smooth the gradient must not move, zeros in the weights included.

        This is the half of the change that must *not* show: swapping the root alters the
        expression, so the gradient away from the singularity is re-checked against a
        reference written from the definition rather than from the implementation.
        """
        targets, predictions, weights, _ = sample
        with_zeros = np.asarray(weights, dtype=np.float64)
        with_zeros[: max(1, with_zeros.size // 4)] = 0.0

        gradient = np.asarray(
            jax.grad(lambda w: metric(predictions, targets, weights=w))(
                jnp.asarray(with_zeros, dtype=jnp.float32)
            ),
            dtype=np.float64,
        )
        expected = references.central_difference(
            reference,
            np.asarray(predictions, dtype=np.float64),
            np.asarray(targets, dtype=np.float64),
            with_zeros,
        )
        assert np.all(np.isfinite(gradient)), f"{name} has a non-finite weight gradient"
        np.testing.assert_allclose(
            gradient,
            expected,
            atol=references.weight_gradient_atol(expected),
            rtol=0.0,
            err_msg=name,
        )

    @pytest.mark.parametrize(("name", "metric", "reference"), _ROOTED, ids=[r[0] for r in _ROOTED])
    def test_the_weight_gradient_is_finite_at_an_all_zero_weight_vector(
        self,
        sample: WeightedSample,
        name: str,
        metric: Callable[..., jnp.ndarray],
        reference: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    ) -> None:
        """An empty selection has no sample to describe, and no reason to be non-finite."""
        del reference
        targets, predictions, _, _ = sample
        gradient = jax.grad(lambda w: metric(predictions, targets, weights=w))(
            jnp.zeros_like(targets)
        )
        assert bool(jnp.all(jnp.isfinite(gradient))), (
            f"{name} reports a non-finite weight gradient at an empty selection"
        )

    @pytest.mark.parametrize(("name", "metric", "reference"), _ROOTED, ids=[r[0] for r in _ROOTED])
    def test_the_weight_gradient_is_finite_when_an_input_is_constant(
        self,
        sample: WeightedSample,
        name: str,
        metric: Callable[..., jnp.ndarray],
        reference: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    ) -> None:
        """A constant input has zero spread, so its root is singular with ordinary weights.

        Distinct from the empty selection: the weights here are strictly positive and the
        sample is real. It is the data that removes the variance, so no guard keyed on the
        weight sum would catch it.

        The constant is a power of two so that the singularity is reached exactly rather
        than nearly. ``weighted_mean`` of a constant ``c`` is ``c * sum(w) / sum(w)``, which
        is exact in binary floating point when ``c`` is a power of two; at ``c = 3.0`` this
        fixture's weights give ``3.0000009537``, leaving ``sum(w * b_centered**2)`` at
        ``3.957e-11`` instead of zero -- off the singular point, where the plain root's
        derivative is large but finite, so the test would pass either way and measure
        nothing.
        """
        del reference
        targets, predictions, weights, _ = sample
        constant = jnp.full_like(targets, 2.0)
        gradient = jax.grad(lambda w: metric(predictions, constant, weights=w))(weights)
        assert bool(jnp.all(jnp.isfinite(gradient))), (
            f"{name} reports a non-finite weight gradient for a constant input"
        )

    @pytest.mark.parametrize(("name", "metric", "reference"), _ROOTED, ids=[r[0] for r in _ROOTED])
    def test_every_row_is_finite_under_jit_vmap_grad_with_one_empty_bin(
        self,
        sample: WeightedSample,
        name: str,
        metric: Callable[..., jnp.ndarray],
        reference: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    ) -> None:
        """The shape this matters in: one gradient per bin of an applicability curve.

        The weights are soft rather than boolean because the derivative is taken with respect
        to them. An unguarded root gives an empty bin a row of ``NaN``, and ``vmap`` keeps
        that row in the same array as the populated ones, so one empty bin spoils the batch.
        """
        del reference
        targets, predictions, weights, _ = sample
        size = targets.shape[0]
        bins = jnp.stack(
            [
                jnp.zeros(size, dtype=targets.dtype),  # the empty bin
                weights * (jnp.arange(size) < size // 2).astype(targets.dtype),  # partly filled
                weights,  # full
            ]
        )
        per_bin = jax.grad(lambda x: metric(predictions, targets, weights=x))
        rows = jax.jit(jax.vmap(per_bin))(bins)
        assert bool(jnp.all(jnp.isfinite(rows))), (
            f"{name} has a non-finite row for the empty bin, which vmap keeps beside the rest"
        )
