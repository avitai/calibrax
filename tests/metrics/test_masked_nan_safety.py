"""An excluded element cannot poison a statistic, in value or in gradient.

``reduce_values`` documents that a masked element "contribute[s] nothing to a sum and
[is] not counted in a mean". An element that turns the whole result into ``NaN``
contributes everything, so the behaviour contradicted the convention this change set
exists to extend.

The denominator was enumerated from the signatures rather than from memory: of everything in
``calibrax.metrics.functional`` and ``_utils`` that accepts ``mask``, ``weights``, ``mask_a``
or ``mask_b``, seventeen are metrics -- the rest are the shared primitives, plus ``bleu``,
whose ``weights`` are n-gram orders and not sample weights. Measured against ``9b0a951`` with
a ``NaN`` at one excluded position, **sixteen of those seventeen failed at least one of the
three criteria below**, in three distinct ways: most returned ``NaN``; ``mae`` and
``kendall_tau`` were wrong in value with a finite gradient; ``relative_l2_error`` returned the
*correct* value with a ``NaN`` gradient, and ``rmse``, ``rmsd`` and ``pairwise_rmsd`` returned
``0.0`` -- a perfect score -- because ``safe_root`` took ``NaN`` to zero at that commit,
which a later one fixed. Only
:func:`spearman_rank_correlation` was correct, and it is left untouched here as the control.

**Substitution happens on the inputs, before any arithmetic.** A ``jnp.where`` placed
after the multiply repairs the value and leaves the gradient ``NaN``, because ``where``
still differentiates the branch it did not take: measured, the post-hoc form gave the right
number with ``grad`` of ``[0.2, nan, -0.2, 0.4]``. The package already documents the
remedy in :func:`calibrax.metrics._utils.safe_root` -- "the root is taken of a value
guarded by an inner ``where`` and the zero case is selected by an outer one; the outer
``where`` alone keeps the NaN" -- citing the JAX FAQ and ``optax.safe_norm``. This is that
idiom applied to masking, so the library follows one convention rather than two.

The guarantee is therefore about the **cotangent**, not merely finiteness, which is how
PyTorch frames ``ignore_index``: an ignored target "does not contribute to the input
gradient". Every test here asserts the excluded slot receives exactly ``0.0``.

A zero *weight* neutralises identically to ``mask=False``. The two are one vector by the
time they reach a reduction, since :func:`sample_weights` folds them, so they cannot be
given different guarantees. ``tf.math.multiply_no_nan`` makes the same choice: it returns
zero when the weight is zero "even if x is NaN or infinite".
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.metrics._utils import (
    coefficient_of_determination,
    reduce_values,
    weighted_mean,
)
from calibrax.metrics.functional.classification import softmax_cross_entropy
from calibrax.metrics.functional.geometric import pairwise_rmsd, rmsd
from calibrax.metrics.functional.regression import (
    charbonnier_loss,
    huber_loss,
    mae,
    mse,
    r_squared,
    r_squared_adjusted,
    relative_l2_error,
    rmse,
)
from calibrax.metrics.functional.statistical import (
    concordance_correlation,
    kendall_tau,
    pearson_correlation,
    spearman_rank_correlation,
)
from calibrax.metrics.functional.text import perplexity


#: Float32 budget. The guarded form computes the same reductions over the same elements as
#: the physical subset, so only reassociation can separate them.
TOLERANCE = 1e-5

#: The excluded position, and the sample size.
EXCLUDED = 1
SIZE = 8

#: Every statistic and loss over a pair of series that takes ``mask`` and ``weights``.
PAIRED: list[tuple[str, Callable[..., jnp.ndarray], dict[str, object]]] = [
    ("coefficient_of_determination", coefficient_of_determination, {}),
    ("pearson_correlation", pearson_correlation, {}),
    ("concordance_correlation", concordance_correlation, {}),
    ("kendall_tau", kendall_tau, {}),
    ("mse", mse, {}),
    ("mae", mae, {}),
    ("rmse", rmse, {}),
    ("huber_loss", huber_loss, {}),
    ("charbonnier_loss", charbonnier_loss, {}),
    ("relative_l2_error", relative_l2_error, {}),
    ("r_squared", r_squared, {}),
    ("r_squared_adjusted", r_squared_adjusted, {"num_predictors": 2}),
]

#: ``r_squared_adjusted`` is excluded from the zero-weight equivalence, and that is the
#: one place the two parameters are documented to diverge: "**A mask changes ``n``; weights
#: do not.**" A masked element leaves the sample and shrinks the degrees-of-freedom
#: correction; a zero-weighted element is still an observation. The convention follows
#: ``statsmodels``, whose ``WLS.nobs`` is ``float(self.wexog.shape[0])``, and
#: ``tests/metrics/test_one_r_squared_formula.py`` pins both halves of it.
#: :class:`TestTheAdjustedPenaltyKeepsItsOwnConvention` asserts the divergence rather than
#: leaving a silent hole here.
ZERO_WEIGHT_EQUIVALENT = [p for p in PAIRED if p[0] != "r_squared_adjusted"]


def _series() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Clean inputs, the same inputs with a NaN at ``EXCLUDED``, a keep-mask, and indices."""
    rng = np.random.default_rng(0)
    targets = jnp.asarray(rng.normal(size=SIZE), dtype=jnp.float32)
    clean = targets + jnp.asarray(rng.normal(scale=0.4, size=SIZE), dtype=jnp.float32)
    dirty = clean.at[EXCLUDED].set(jnp.nan)
    keep = jnp.asarray([i != EXCLUDED for i in range(SIZE)])
    kept = jnp.asarray([i for i in range(SIZE) if i != EXCLUDED])
    return clean, dirty, keep, kept


class TestPairedStatistics:
    """Value equals the subset, and the excluded slot receives a zero cotangent."""

    @pytest.mark.parametrize(("name", "metric", "extra"), PAIRED, ids=[p[0] for p in PAIRED])
    def test_a_masked_nan_matches_the_physical_subset(
        self, name: str, metric: Callable[..., jnp.ndarray], extra: dict[str, object]
    ) -> None:
        clean, dirty, keep, kept = _series()
        targets = clean - jnp.asarray(0.3, dtype=jnp.float32)
        masked = float(metric(dirty, targets, mask=keep, **extra))
        subset = float(metric(clean[kept], targets[kept], **extra))
        assert masked == pytest.approx(subset, abs=TOLERANCE), (
            f"{name} let a NaN at an excluded position change the result"
        )

    @pytest.mark.parametrize(("name", "metric", "extra"), PAIRED, ids=[p[0] for p in PAIRED])
    def test_the_excluded_slot_receives_a_zero_cotangent(
        self, name: str, metric: Callable[..., jnp.ndarray], extra: dict[str, object]
    ) -> None:
        """Finiteness is not enough: the excluded element must contribute exactly nothing.

        A ``jnp.where`` after the arithmetic yields the right value with a ``NaN`` here,
        so this is the assertion that distinguishes the two placements.
        """
        clean, dirty, keep, _ = _series()
        targets = clean - jnp.asarray(0.3, dtype=jnp.float32)
        grad = jax.grad(lambda x: metric(x, targets, mask=keep, **extra))(dirty)
        assert jnp.all(jnp.isfinite(grad)), f"{name} has a non-finite gradient"
        assert float(grad[EXCLUDED]) == 0.0, (
            f"{name} gives the excluded element a non-zero cotangent"
        )

    @pytest.mark.parametrize(
        ("name", "metric", "extra"),
        ZERO_WEIGHT_EQUIVALENT,
        ids=[p[0] for p in ZERO_WEIGHT_EQUIVALENT],
    )
    def test_a_zero_weight_neutralises_identically_to_a_mask(
        self, name: str, metric: Callable[..., jnp.ndarray], extra: dict[str, object]
    ) -> None:
        """``sample_weights`` folds both into one vector, so they cannot differ."""
        clean, dirty, keep, kept = _series()
        targets = clean - jnp.asarray(0.3, dtype=jnp.float32)
        weights = jnp.where(keep, 1.0, 0.0).astype(jnp.float32)
        by_weight = float(metric(dirty, targets, weights=weights, **extra))
        subset = float(metric(clean[kept], targets[kept], **extra))
        assert by_weight == pytest.approx(subset, abs=TOLERANCE), (
            f"{name} treats a zero weight differently from an excluded element"
        )

    @pytest.mark.parametrize(("name", "metric", "extra"), PAIRED, ids=[p[0] for p in PAIRED])
    def test_an_unmasked_nan_still_propagates(
        self, name: str, metric: Callable[..., jnp.ndarray], extra: dict[str, object]
    ) -> None:
        """The fix must not silently swallow a NaN the caller did not exclude.

        Without this, substituting zeros everywhere would pass every test above while
        hiding real data corruption.
        """
        clean, dirty, _, _ = _series()
        targets = clean - jnp.asarray(0.3, dtype=jnp.float32)
        everything = jnp.ones(SIZE, dtype=bool)
        assert not np.isfinite(float(metric(dirty, targets, mask=everything, **extra))), (
            f"{name} absorbed a NaN that was not excluded"
        )


class TestTheReducersDeliberatelyDoNotGuard:
    """The primitives propagate a NaN, and that is the design rather than an omission.

    An earlier version of this file asserted the opposite -- that ``reduce_values`` and
    ``weighted_mean`` absorb an excluded NaN. They were guarded, and the tests passed, and
    the design was still wrong: a guard inside a reduction repairs the number while the
    gradient stays poisoned, because by then the NaN has already reached the derivative and
    ``0 * dvalues/dp`` is ``NaN`` when ``dvalues/dp`` is. Measured, guarding only the
    primitive left ``mse`` returning the correct value with a ``NaN`` gradient -- a worse
    state than the plain ``NaN`` it replaced, since a forward test then passes.

    It was also redundant: with the boundary guard in place the primitive's guard changed no
    result and only added work. Re-measured at 4,096 weighted elements, putting it back costs
    6 fusions in ``pearson_correlation`` (55 to 61), 6 in ``concordance_correlation`` (58 to
    64) and 3 in ``coefficient_of_determination`` (34 to 37).

    So the guarantee lives at the function boundary, and these primitives are left exact.
    A direct caller handing them unsanitised values gets a ``NaN``, which is the honest
    answer: ``weatherbenchX`` defaults to ``skipna=False`` for the same reason, calling the
    tolerant setting "not recommended, as it won't catch unexpected NaNs".
    """

    def test_reduce_values_propagates_an_excluded_nan(self) -> None:
        values = jnp.asarray([1.0, jnp.nan, 3.0, 4.0], dtype=jnp.float32)
        keep = jnp.asarray([True, False, True, True])
        assert not np.isfinite(float(reduce_values(values, mask=keep, reduction="mean"))), (
            "reduce_values absorbed a NaN; the guarantee belongs at the caller, not here"
        )

    def test_weighted_mean_propagates_an_excluded_nan(self) -> None:
        values = jnp.asarray([1.0, jnp.nan, 3.0, 4.0], dtype=jnp.float32)
        weights = jnp.asarray([1.0, 0.0, 1.0, 1.0], dtype=jnp.float32)
        assert not np.isfinite(float(weighted_mean(values, weights)))

    def test_the_losses_that_use_them_are_nonetheless_safe(self) -> None:
        """Which is the point: every in-repo caller substitutes before reducing."""
        clean, dirty, keep, kept = _series()
        targets = clean - jnp.asarray(0.3, dtype=jnp.float32)
        for metric in (mse, mae, rmse, huber_loss, charbonnier_loss, relative_l2_error):
            masked = float(metric(dirty, targets, mask=keep))
            subset = float(metric(clean[kept], targets[kept]))
            assert masked == pytest.approx(subset, abs=TOLERANCE), metric.__name__


class TestSpearmanIsTheUntouchedControl:
    """It was already correct, so this change must leave it exactly as it was.

    It reaches ``pearson_correlation`` twice internally, so it is the function most
    exposed to a double substitution.
    """

    def test_it_still_matches_the_physical_subset(self) -> None:
        clean, dirty, keep, kept = _series()
        targets = clean - jnp.asarray(0.3, dtype=jnp.float32)
        assert float(spearman_rank_correlation(dirty, targets, mask=keep)) == pytest.approx(
            float(spearman_rank_correlation(clean[kept], targets[kept])), abs=TOLERANCE
        )

    def test_its_excluded_slot_still_receives_a_zero_cotangent(self) -> None:
        clean, dirty, keep, _ = _series()
        targets = clean - jnp.asarray(0.3, dtype=jnp.float32)
        grad = jax.grad(lambda x: spearman_rank_correlation(x, targets, mask=keep))(dirty)
        assert jnp.all(jnp.isfinite(grad))
        assert float(grad[EXCLUDED]) == 0.0


class TestTransformations:
    """Substitution must not cost traceability, gradients with respect to weights included."""

    @pytest.mark.parametrize(("name", "metric", "extra"), PAIRED, ids=[p[0] for p in PAIRED])
    def test_jit_vmap_and_weight_gradients(
        self, name: str, metric: Callable[..., jnp.ndarray], extra: dict[str, object]
    ) -> None:
        clean, dirty, keep, _ = _series()
        targets = clean - jnp.asarray(0.3, dtype=jnp.float32)
        weights = jnp.where(keep, 1.0, 0.0).astype(jnp.float32)

        def call(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
            return metric(x, targets, weights=w, **extra)

        assert jnp.isfinite(jax.jit(call)(dirty, weights))
        batched = jax.jit(jax.vmap(call))(jnp.stack([dirty] * 3), jnp.stack([weights] * 3))
        assert batched.shape == (3,)
        grad_w = jax.grad(lambda w: call(dirty, w))(weights)
        assert jnp.all(jnp.isfinite(grad_w)), (
            f"{name} has a non-finite gradient with respect to its weights"
        )


#: Log-probabilities, a keep-mask, and the physical subset. ``perplexity`` names its single
#: parameter ``mask`` and documents it as weights, so one vector serves as both.
_LOG_PROBS = jnp.asarray([-1.0, jnp.nan, -2.0, -3.0], dtype=jnp.float32)
_LOG_PROBS_KEPT = jnp.asarray([-1.0, -2.0, -3.0], dtype=jnp.float32)
_POSITION_MASK = jnp.asarray([1.0, 0.0, 1.0, 1.0], dtype=jnp.float32)


class TestPerplexity:
    """``exp(-sum(log_probs * weights) / sum(weights))`` reads every position.

    A padding position's log-probability is exactly what a caller is most likely to leave
    unwritten, so ``NaN`` there is the expected input rather than an exotic one.
    """

    def test_a_masked_nan_matches_the_physical_subset(self) -> None:
        masked = float(perplexity(_LOG_PROBS, mask=_POSITION_MASK))
        assert masked == pytest.approx(float(perplexity(_LOG_PROBS_KEPT)), abs=TOLERANCE)

    def test_the_excluded_position_receives_a_zero_cotangent(self) -> None:
        grad = jax.grad(lambda x: perplexity(x, mask=_POSITION_MASK))(_LOG_PROBS)
        assert jnp.all(jnp.isfinite(grad))
        assert float(grad[EXCLUDED]) == 0.0

    def test_a_fractional_weight_neutralises_the_same_way(self) -> None:
        """The parameter is documented as weights, so a zero among floats must also hold."""
        weights = jnp.asarray([2.0, 0.0, 1.0, 3.0], dtype=jnp.float32)
        masked = float(perplexity(_LOG_PROBS, mask=weights))
        subset = float(perplexity(_LOG_PROBS_KEPT, mask=jnp.asarray([2.0, 1.0, 3.0])))
        assert masked == pytest.approx(subset, abs=TOLERANCE)

    def test_an_unmasked_nan_still_propagates(self) -> None:
        keep_all = jnp.ones(4, dtype=jnp.float32)
        assert not np.isfinite(float(perplexity(_LOG_PROBS, mask=keep_all)))

    def test_it_survives_jit_and_vmap(self) -> None:
        call = jax.jit(perplexity)
        assert jnp.isfinite(call(_LOG_PROBS, mask=_POSITION_MASK))
        batched = jax.jit(jax.vmap(lambda x, m: perplexity(x, mask=m)))(
            jnp.stack([_LOG_PROBS] * 3), jnp.stack([_POSITION_MASK] * 3)
        )
        assert batched.shape == (3,)
        assert jnp.all(jnp.isfinite(batched))


#: Logits with a NaN in the row the mask drops, the labels, and the kept rows.
_LOGITS = jnp.asarray([[1.0, 2.0], [jnp.nan, 0.5], [0.3, 1.7], [2.0, -1.0]], dtype=jnp.float32)
_LOGITS_KEPT = jnp.asarray([[1.0, 2.0], [0.3, 1.7], [2.0, -1.0]], dtype=jnp.float32)
_LABELS = jnp.asarray([0, 1, 1, 0])
_LABELS_KEPT = jnp.asarray([0, 1, 0])
_ROW_MASK = jnp.asarray([True, False, True, True])


class TestSoftmaxCrossEntropy:
    """The mask is at the label shape and the ``NaN`` is one axis deeper.

    ``log_softmax`` runs along the class axis, so a single bad logit poisons its whole row
    and then the picked value, which ``reduce_values`` can no longer neutralise. The
    substitution therefore folds at the label shape and restores the trailing axis.
    """

    def test_a_masked_nan_matches_the_physical_subset(self) -> None:
        masked = float(softmax_cross_entropy(_LOGITS, _LABELS, mask=_ROW_MASK))
        subset = float(softmax_cross_entropy(_LOGITS_KEPT, _LABELS_KEPT))
        assert masked == pytest.approx(subset, abs=TOLERANCE)

    def test_the_whole_excluded_row_receives_a_zero_cotangent(self) -> None:
        grad = jax.grad(lambda x: softmax_cross_entropy(x, _LABELS, mask=_ROW_MASK))(_LOGITS)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.all(grad[EXCLUDED] == 0.0), "the excluded logit row must contribute nothing"

    def test_a_zero_weight_neutralises_identically_to_a_mask(self) -> None:
        weights = jnp.asarray([2.0, 0.0, 1.0, 3.0], dtype=jnp.float32)
        masked = float(softmax_cross_entropy(_LOGITS, _LABELS, weights=weights))
        subset = float(
            softmax_cross_entropy(_LOGITS_KEPT, _LABELS_KEPT, weights=jnp.asarray([2.0, 1.0, 3.0]))
        )
        assert masked == pytest.approx(subset, abs=TOLERANCE)

    def test_the_other_reductions_agree_too(self) -> None:
        """``sum`` and ``none`` read the same substituted rows, so both must be clean."""
        total = float(softmax_cross_entropy(_LOGITS, _LABELS, mask=_ROW_MASK, reduction="sum"))
        assert total == pytest.approx(
            float(softmax_cross_entropy(_LOGITS_KEPT, _LABELS_KEPT, reduction="sum")),
            abs=TOLERANCE,
        )
        elementwise = softmax_cross_entropy(_LOGITS, _LABELS, mask=_ROW_MASK, reduction="none")
        assert jnp.all(jnp.isfinite(elementwise))
        assert float(elementwise[EXCLUDED]) == 0.0

    def test_an_unmasked_nan_still_propagates(self) -> None:
        keep_all = jnp.ones(4, dtype=bool)
        assert not np.isfinite(float(softmax_cross_entropy(_LOGITS, _LABELS, mask=keep_all)))

    def test_it_survives_jit_and_vmap(self) -> None:
        def call(x: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
            return softmax_cross_entropy(x, _LABELS, mask=m)

        assert jnp.isfinite(jax.jit(call)(_LOGITS, _ROW_MASK))
        batched = jax.jit(jax.vmap(call))(jnp.stack([_LOGITS] * 3), jnp.stack([_ROW_MASK] * 3))
        assert batched.shape == (3,)
        assert jnp.all(jnp.isfinite(batched))


#: Two conformations whose shared atoms are 1.0 apart after centring, with a NaN at the
#: atom neither mask keeps. The subset answer must not be 0.0, or the value assertion
#: cannot tell a repaired result from the swallowed one: before the fix this returned
#: exactly 0.0, which reads as two identical conformations.
_COORDS_A = jnp.asarray([[0.0, 0.0], [jnp.nan, jnp.nan], [1.0, 0.0]], dtype=jnp.float32)
_COORDS_B = jnp.asarray([[0.0, 1.0], [5.0, 5.0], [3.0, 1.0]], dtype=jnp.float32)
_ATOM_MASK = jnp.asarray([True, False, True])
_SHARED_A = jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
_SHARED_B = jnp.asarray([[0.0, 1.0], [3.0, 1.0]], dtype=jnp.float32)


class TestRmsd:
    """An absent atom's coordinates are the canonical ``NaN``: there is no atom there.

    Both centroids sum over every atom, so ``0 * nan`` poisons them and every squared term
    after them. At ``9b0a951`` ``safe_root`` then took the ``NaN`` to ``0.0``, which is why
    this one reported a perfect match rather than a ``NaN`` -- the quietest of the failures,
    and the reason the value assertion below still needs a non-zero reference even now that
    the primitive propagates.
    """

    def test_the_reference_is_discriminating(self) -> None:
        """Guard the fixture itself: a zero reference would make the next test vacuous."""
        assert float(rmsd(_SHARED_A, _SHARED_B)) == pytest.approx(1.0, abs=TOLERANCE)

    def test_a_masked_nan_matches_the_physical_subset(self) -> None:
        masked = float(rmsd(_COORDS_A, _COORDS_B, mask_a=_ATOM_MASK, mask_b=_ATOM_MASK))
        assert masked == pytest.approx(float(rmsd(_SHARED_A, _SHARED_B)), abs=TOLERANCE)

    def test_one_sided_exclusion_is_enough(self) -> None:
        """The pair's weight is ``mask_a & mask_b``, so either mask alone must neutralise."""
        masked = float(rmsd(_COORDS_A, _COORDS_B, mask_a=_ATOM_MASK))
        assert masked == pytest.approx(float(rmsd(_SHARED_A, _SHARED_B)), abs=TOLERANCE)

    def test_the_excluded_atom_receives_a_zero_cotangent(self) -> None:
        grad = jax.grad(lambda x: rmsd(x, _COORDS_B, mask_a=_ATOM_MASK, mask_b=_ATOM_MASK))(
            _COORDS_A
        )
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.all(grad[EXCLUDED] == 0.0)

    def test_pairwise_rmsd_inherits_it_through_vmap(self) -> None:
        conformations = jnp.stack([_COORDS_A, _COORDS_B])
        masks = jnp.stack([_ATOM_MASK, _ATOM_MASK])
        matrix = pairwise_rmsd(conformations, masks)
        assert jnp.all(jnp.isfinite(matrix))
        assert float(matrix[0, 1]) == pytest.approx(
            float(rmsd(_SHARED_A, _SHARED_B)), abs=TOLERANCE
        )
        grad = jax.grad(lambda c: pairwise_rmsd(c, masks)[0, 1])(conformations)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.all(grad[0, EXCLUDED] == 0.0)

    def test_an_unmasked_nan_still_propagates(self) -> None:
        keep_all = jnp.ones(3, dtype=bool)
        assert not np.isfinite(float(rmsd(_COORDS_A, _COORDS_B, mask_a=keep_all, mask_b=keep_all)))

    def test_pairwise_rmsd_propagates_an_unmasked_nan_too(self) -> None:
        """It reaches ``safe_root`` through ``vmap``, so it needs its own assertion.

        Until the predicate was fixed this returned a matrix of zeros -- every pair of
        conformations reported identical. No xfail covered it: the three markers that
        covered ``rmse``, ``relative_l2_error`` and ``rmsd`` named four affected functions
        between them but only ever asserted three.
        """
        keep_all = jnp.ones((2, 3), dtype=bool)
        matrix = pairwise_rmsd(jnp.stack([_COORDS_A, _COORDS_B]), keep_all)
        assert bool(jnp.any(jnp.isnan(matrix))), "a kept NaN atom must reach the result"

    def test_it_survives_jit_and_vmap(self) -> None:
        def call(x: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
            return rmsd(x, _COORDS_B, mask_a=m, mask_b=m)

        assert jnp.isfinite(jax.jit(call)(_COORDS_A, _ATOM_MASK))
        batched = jax.jit(jax.vmap(call))(jnp.stack([_COORDS_A] * 3), jnp.stack([_ATOM_MASK] * 3))
        assert batched.shape == (3,)
        assert jnp.all(jnp.isfinite(batched))


class TestTheAdjustedPenaltyKeepsItsOwnConvention:
    """The one documented divergence between a mask and a zero weight.

    A first draft of this file asserted the blanket equivalence over every paired function
    and ``r_squared_adjusted`` failed it. The implementation was right: the docstring says
    "**A mask changes ``n``; weights do not**", citing ``statsmodels``' ``WLS.nobs ==
    float(self.wexog.shape[0])``. Measured on the fixture below, the two agree on the
    underlying ratio and differ only in the degrees-of-freedom factor, exactly as that
    convention predicts -- so the test was the thing that was wrong.

    This class exists so the exclusion above is a claim with evidence rather than a gap.
    """

    def test_the_two_differ_only_by_the_documented_factor(self) -> None:
        clean, dirty, keep, _ = _series()
        targets = clean - jnp.asarray(0.3, dtype=jnp.float32)
        weights = jnp.where(keep, 1.0, 0.0).astype(jnp.float32)
        ratio = float(r_squared(dirty, targets, mask=keep))

        by_mask = float(r_squared_adjusted(dirty, targets, num_predictors=2, mask=keep))
        by_weight = float(r_squared_adjusted(dirty, targets, num_predictors=2, weights=weights))

        kept_n = SIZE - 1
        # A mask drops the element from the sample: n = 7, residual dof = 7 - 2 - 1.
        assert by_mask == pytest.approx(
            1.0 - (1.0 - ratio) * (kept_n - 1) / (kept_n - 3), abs=TOLERANCE
        )
        # A zero weight keeps it: n = 8, residual dof = 8 - 2 - 1.
        assert by_weight == pytest.approx(
            1.0 - (1.0 - ratio) * (SIZE - 1) / (SIZE - 3), abs=TOLERANCE
        )
        assert by_mask != pytest.approx(by_weight, abs=TOLERANCE), (
            "if these agree the convention has been lost, not the test"
        )

    def test_the_mask_form_is_the_one_that_matches_the_physical_subset(self) -> None:
        """Which is why the mask form, not the weight form, is in ``PAIRED``."""
        clean, dirty, keep, kept = _series()
        targets = clean - jnp.asarray(0.3, dtype=jnp.float32)
        assert float(
            r_squared_adjusted(dirty, targets, num_predictors=2, mask=keep)
        ) == pytest.approx(
            float(r_squared_adjusted(clean[kept], targets[kept], num_predictors=2)),
            abs=TOLERANCE,
        )

    def test_the_excluded_slot_still_receives_a_zero_cotangent_either_way(self) -> None:
        """The divergence is in ``n`` alone; the NaN must be neutralised on both paths."""
        clean, dirty, keep, _ = _series()
        targets = clean - jnp.asarray(0.3, dtype=jnp.float32)
        weights = jnp.where(keep, 1.0, 0.0).astype(jnp.float32)
        for grad in (
            jax.grad(lambda x: r_squared_adjusted(x, targets, num_predictors=2, mask=keep))(dirty),
            jax.grad(lambda x: r_squared_adjusted(x, targets, num_predictors=2, weights=weights))(
                dirty
            ),
        ):
            assert jnp.all(jnp.isfinite(grad))
            assert float(grad[EXCLUDED]) == 0.0


class TestRelativeL2ErrorMasksBySample:
    """Its mask is dimensioned against its *output*, which no other function here is.

    ``relative_l2_error`` reduces to one value per sample -- ``||p_i - t_i|| / ||t_i||`` --
    so its documented mask is ``(n_samples,)`` while its input is ``(n_samples, *field)``.
    A first draft substituted at the input's own shape, as every other function here does,
    and the documented mask stopped being accepted at all: ``ValueError: Incompatible types
    for broadcasting: input type=float32[4] and requested type=float32[4,3]``. The full
    suite caught it; the targeted tests above did not, because they are all one-dimensional.
    """

    FIELD = (4, 3)

    def _fixture(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        rng = np.random.default_rng(0)
        targets = jnp.asarray(rng.normal(size=self.FIELD), dtype=jnp.float32)
        clean = targets + jnp.asarray(rng.normal(scale=0.4, size=self.FIELD), dtype=jnp.float32)
        dirty = clean.at[2, 1].set(jnp.nan)  # inside the sample the mask drops
        keep = jnp.asarray([True, True, False, True])
        return clean, dirty, targets, keep

    def test_a_per_sample_mask_is_accepted_over_a_two_dimensional_field(self) -> None:
        clean, dirty, targets, keep = self._fixture()
        masked = float(relative_l2_error(dirty, targets, mask=keep))
        subset = float(relative_l2_error(clean[keep], targets[keep]))
        assert masked == pytest.approx(subset, abs=TOLERANCE)

    def test_the_dropped_samples_field_receives_a_zero_cotangent(self) -> None:
        _, dirty, targets, keep = self._fixture()
        grad = jax.grad(lambda x: relative_l2_error(x, targets, mask=keep))(dirty)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.all(grad[2] == 0.0), "the whole excluded sample must contribute nothing"

    def test_a_field_shaped_mask_is_refused_rather_than_broadcast(self) -> None:
        """``9b0a951`` accepted ``(n_samples, 1)`` here and answered from an outer product.

        Measured on that commit with a clean ``(4, 3)`` fixture: a ``(4, 1)`` mask against
        the ``(4,)`` per-sample values broadcast to ``(4, 4)`` inside ``reduce_values``, and
        the function returned 0.719671 where the masked mean is 0.874025 -- the mean of a
        sixteen-element outer product. Folding at the sample axis makes the shape mismatch
        an error instead, which is the only honest answer to a mask that does not fit.
        """
        _, dirty, targets, keep = self._fixture()
        with pytest.raises(ValueError, match="must be per sample"):
            relative_l2_error(dirty, targets, mask=keep[:, None])

    def test_per_sample_weights_behave_the_same(self) -> None:
        clean, dirty, targets, keep = self._fixture()
        weights = jnp.where(keep, 1.0, 0.0).astype(jnp.float32)
        by_weight = float(relative_l2_error(dirty, targets, weights=weights))
        subset = float(relative_l2_error(clean[keep], targets[keep]))
        assert by_weight == pytest.approx(subset, abs=TOLERANCE)
