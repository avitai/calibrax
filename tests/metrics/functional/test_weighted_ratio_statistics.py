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

from calibrax.metrics._utils import _EPSILON
from calibrax.metrics.functional.regression import r_squared, r_squared_adjusted
from calibrax.metrics.functional.statistical import (
    concordance_correlation,
    kendall_tau,
    pearson_correlation,
    spearman_rank_correlation,
)
from tests.factories import WeightedSample


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
        concordance is not, because its unweighted path is ``jnp.var``/``jnp.mean`` -- the
        expression it had before weights existed, kept so that no released number moved --
        while its weighted path is a sum against a weight vector divided by that vector's
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

        Pins the unweighted path against an explicit replica of the expression concordance
        had before it accepted weights, bit for bit. Without this the fast path could drift
        and only the loosened assertion above would be left to catch it.
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
    """An empty selection has no sample to describe, so every statistic returns ``0.0``.

    That is the value :func:`calibrax.metrics._utils.reduce_values` documents for the case --
    "one documented, finite result, so a caller that treats an empty selection as an error
    can check the mask at the host boundary instead of finding a NaN later" -- and the value
    :func:`calibrax.metrics._utils.weighted_mean` already returns when its scale sums to
    zero. Pinning it here for all five keeps one convention rather than one per statistic.

    **An earlier version of this class asserted the wrong thing.** It read
    ``assert value == pytest.approx(1.0) or not np.isfinite(value)``, under this same
    docstring, which accepted ``1.0`` -- the most plausible-looking number available, and
    what ``r_squared`` actually returned -- while *failing* for ``0.0``. It did not merely
    miss the defect, it forbade the fix. Assert the value, never a disjunction that the
    shipped behaviour happens to satisfy.
    """

    #: The documented result for a selection with no members.
    EMPTY_RESULT = 0.0

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
        assert float(metric(predictions, targets, mask=empty)) == self.EMPTY_RESULT, (
            f"{name} reported a measurement for a selection with no members"
        )

    def test_an_all_zero_weight_vector_is_the_same_case(self, sample: WeightedSample) -> None:
        """Weights reach the same state by a different route, so they get the same answer."""
        targets, predictions, _, _ = sample
        assert (
            float(r_squared(predictions, targets, weights=jnp.zeros_like(targets)))
            == self.EMPTY_RESULT
        )

    def test_adjusted_r_squared_agrees(self, sample: WeightedSample) -> None:
        """The adjusted form reads its R-squared from the same core, so it follows."""
        targets, predictions, _, _ = sample
        empty = jnp.zeros_like(targets, dtype=bool)
        assert (
            float(r_squared_adjusted(predictions, targets, num_predictors=2, mask=empty))
            == self.EMPTY_RESULT
        )

    def test_an_empty_bin_does_not_read_as_a_perfect_fit_under_vmap(
        self, sample: WeightedSample
    ) -> None:
        """Why this matters, in the shape the feature was built for.

        A per-bin applicability curve maps one mask per bin. A bin with no members used to
        return 1.0, so an empty bin appeared mid-curve as the best-fitting bin in the plot.
        """
        targets, predictions, _, _ = sample
        size = targets.shape[0]
        counts = (0, 5, size)
        masks = jnp.stack([jnp.arange(size) < kept for kept in counts])
        curve = jax.jit(jax.vmap(lambda m: r_squared(predictions, targets, mask=m)))(masks)
        assert float(curve[0]) == self.EMPTY_RESULT
        assert float(curve[0]) < float(curve[-1]), (
            "the empty bin outranks the full one, which is what the 1.0 reading did"
        )

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
        assert with_mask != self.EMPTY_RESULT


#: Tolerance for the weight gradient against a float64 central difference, in float32 ULP at
#: the gradient's own scale. The comparison is a float32 gradient against a float64
#: difference, so the discrepancy is float32 evaluation error rather than truncation: at
#: ``h = 1e-5`` the reference's own truncation is near 1e-10. Measured over 20 seeds x sizes
#: 8/12/24 with a quarter of the weights zeroed, the worst was 4.37 ULP for
#: ``pearson_correlation`` and 0.87 for ``kendall_tau``; the bound allows 16. An absolute
#: constant cannot express it: the same readings span 8.438e-10 to 5.208e-07, so one number
#: is fixed by the worst statistic and runs loose by nearly three orders of magnitude for the
#: other. This is the convention ``tests/metrics/test_zero_weight_gradients.py`` records.
_GRADIENT_ULPS = 16

#: The central difference's step. Both reference formulas are rational in the weights away
#: from an all-zero vector, so a +/-h step stays in their domain.
_STEP = 1e-5


def _gradient_atol(reference: np.ndarray) -> float:
    """Float32 ULP at the scale of the gradient being compared."""
    return (
        _GRADIENT_ULPS * float(np.finfo(np.float32).eps) * max(1.0, float(np.abs(reference).max()))
    )


def _reference_pearson(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson r in float64, written from the definition rather than from the code."""
    a, b, w = (np.asarray(v, dtype=np.float64) for v in (a, b, w))
    total = w.sum()
    a_centered = a - (w * a).sum() / total
    b_centered = b - (w * b).sum() / total
    covariance = (w * a_centered * b_centered).sum()
    std_a = np.sqrt((w * a_centered**2).sum())
    std_b = np.sqrt((w * b_centered**2).sum())
    return float(covariance / (std_a * std_b + _EPSILON))


def _reference_kendall(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted tau-b in float64: one weight per pair, from the definition."""
    a, b, w = (np.asarray(v, dtype=np.float64) for v in (a, b, w))
    upper = np.triu(np.ones((a.size, a.size), dtype=bool), k=1)
    pair_weight = np.where(upper, np.outer(w, w), 0.0)
    a_diff = a[:, None] - a[None, :]
    b_diff = b[:, None] - b[None, :]
    concordant = (pair_weight * np.sign(a_diff) * np.sign(b_diff)).sum()
    total = pair_weight.sum()
    untied_a = total - np.where(a_diff == 0, pair_weight, 0.0).sum()
    untied_b = total - np.where(b_diff == 0, pair_weight, 0.0).sum()
    return float(concordant / (np.sqrt(untied_a * untied_b) + _EPSILON))


def _central_difference(
    reference: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    a: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """``d/dw`` of ``reference`` by a float64 central difference, one coordinate at a time."""
    gradient = np.zeros(w.size, dtype=np.float64)
    for i in range(w.size):
        up, down = np.array(w, dtype=np.float64), np.array(w, dtype=np.float64)
        up[i] += _STEP
        down[i] -= _STEP
        gradient[i] = (reference(a, b, up) - reference(a, b, down)) / (2.0 * _STEP)
    return gradient


#: The two statistics whose weighted path takes a root of a sum that an empty selection
#: drives to zero. ``concordance_correlation`` is excluded deliberately: it divides by a sum
#: of squares and never takes a root, so it already reported finite weight gradients here.
_ROOTED = [
    ("pearson_correlation", pearson_correlation, _reference_pearson),
    ("kendall_tau", kendall_tau, _reference_kendall),
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
        expected = _central_difference(
            reference,
            np.asarray(predictions, dtype=np.float64),
            np.asarray(targets, dtype=np.float64),
            with_zeros,
        )
        assert np.all(np.isfinite(gradient)), f"{name} has a non-finite weight gradient"
        np.testing.assert_allclose(
            gradient, expected, atol=_gradient_atol(expected), rtol=0.0, err_msg=name
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
