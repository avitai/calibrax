"""A zero weight keeps the element's derivative with respect to its own weight.

``d/dw_i`` of a weighted statistic is the element's **influence**, and evaluating it at
``w_i = 0`` is case deletion -- Jaeckel's infinitesimal jackknife, and what
leave-one-out diagnostics compute. It is also what sample-reweighting methods
differentiate: Ren et al. (2018) initialise example weights at zero and take a meta-gradient
step on them, so ``w_i = 0`` is the configuration they evaluate at, not an edge case.

``sample_weights`` promises "gradients with respect to ``weights`` included", and substituting
zero wherever the folded weight is zero broke that promise: the derivative was taken for a
point moved to the origin, so it read ``0`` instead of the element's contribution.

**The rule these tests pin.** An element is replaced by zero when the *mask* excludes it
(whatever it holds), or when its weight is zero *and* its value is not finite. A finite
element with a zero weight keeps its value: it is an observation carrying weight zero, so its
derivative is its contribution. The mask and the weights cannot be collapsed into one vector
for this, because a mask must discard an element whose value is finite but whose square
overflows -- measured, folding them first makes ``mask``-ing a ``1e30`` read ``NaN``.

Its one documented cost: a **zero weight** on such a value still reads ``NaN``, as it does
without any of this. To discard an element, mask it.

Every assertion here is on a gradient *value* against an independent reference, never on
finiteness: a finite-but-wrong derivative is exactly the failure that shipped.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.metrics.functional.classification import softmax_cross_entropy
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
from tests.metrics import float64_references as references


#: Losses that reduce element-wise, so ``reduction="none"`` is the per-element loss.
ELEMENTWISE = [
    ("mse", mse, {}),
    ("mae", mae, {}),
    ("huber_loss", huber_loss, {}),
    ("charbonnier_loss", charbonnier_loss, {}),
]


def _sample(n: int = 6, seed: int = 0) -> tuple[jnp.ndarray, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    targets = jnp.asarray(rng.normal(size=n), dtype=jnp.float32)
    predictions = targets + jnp.asarray(rng.normal(scale=0.7, size=n), dtype=jnp.float32)
    return predictions, targets


class TestSumReductionGivesTheElementsOwnLoss:
    """``d/dw_i`` of ``sum(w_j l_j)`` is ``l_i``, exactly, including where ``w_i = 0``."""

    @pytest.mark.parametrize(
        ("name", "loss", "extra"), ELEMENTWISE, ids=[e[0] for e in ELEMENTWISE]
    )
    def test_some_weights_zero(
        self, name: str, loss: Callable[..., jnp.ndarray], extra: dict[str, object]
    ) -> None:
        p, t = _sample()
        weights = jnp.asarray([1.0, 0.0, 2.0, 0.0, 0.5, 1.0], dtype=jnp.float32)
        grad = jax.grad(lambda w: loss(p, t, weights=w, reduction="sum", **extra))(weights)
        elementwise = loss(p, t, reduction="none", **extra)
        np.testing.assert_array_equal(np.asarray(grad), np.asarray(elementwise), err_msg=name)

    @pytest.mark.parametrize(
        ("name", "loss", "extra"), ELEMENTWISE, ids=[e[0] for e in ELEMENTWISE]
    )
    def test_all_weights_zero(
        self, name: str, loss: Callable[..., jnp.ndarray], extra: dict[str, object]
    ) -> None:
        """The configuration Ren et al. start from: every weight exactly zero."""
        p, t = _sample()
        grad = jax.grad(lambda w: loss(p, t, weights=w, reduction="sum", **extra))(
            jnp.zeros(6, dtype=jnp.float32)
        )
        np.testing.assert_array_equal(
            np.asarray(grad), np.asarray(loss(p, t, reduction="none", **extra)), err_msg=name
        )


class TestMeanReductionMatchesTheAnalyticDerivative:
    """``d/dw_i`` of ``sum(w l)/sum(w)`` is ``(l_i S - sum(w l)) / S^2``, computed in float64."""

    @staticmethod
    def _reference(elementwise: np.ndarray, weights: np.ndarray) -> np.ndarray:
        loss64 = elementwise.astype(np.float64)
        w64 = weights.astype(np.float64)
        total = w64.sum()
        return (loss64 * total - (w64 * loss64).sum()) / total**2

    @pytest.mark.parametrize(
        ("name", "loss", "extra"), ELEMENTWISE, ids=[e[0] for e in ELEMENTWISE]
    )
    def test_elementwise_losses(
        self, name: str, loss: Callable[..., jnp.ndarray], extra: dict[str, object]
    ) -> None:
        p, t = _sample()
        weights = jnp.asarray([1.0, 0.0, 2.0, 0.0, 0.5, 1.0], dtype=jnp.float32)
        grad = jax.grad(lambda w: loss(p, t, weights=w, **extra))(weights)
        reference = self._reference(
            np.asarray(loss(p, t, reduction="none", **extra)), np.asarray(weights)
        )
        np.testing.assert_allclose(np.asarray(grad), reference, rtol=0, atol=2e-7, err_msg=name)

    def test_rmse_carries_the_chain_rule_factor(self) -> None:
        """``rmse`` is the root of that mean, so its reference gains ``1/(2*rmse)``."""
        p, t = _sample()
        weights = jnp.asarray([1.0, 0.0, 2.0, 0.0, 0.5, 1.0], dtype=jnp.float32)
        grad = jax.grad(lambda w: rmse(p, t, weights=w))(weights)
        inner = self._reference(np.asarray(mse(p, t, reduction="none")), np.asarray(weights))
        reference = inner / (2.0 * float(rmse(p, t, weights=weights)))
        np.testing.assert_allclose(np.asarray(grad), reference, rtol=0, atol=2e-7)

    def test_relative_l2_error_is_per_sample(self) -> None:
        """Its mask and weights are dimensioned per sample, so ``l`` is the per-sample ratio."""
        rng = np.random.default_rng(1)
        targets = jnp.asarray(rng.normal(size=(5, 4)), dtype=jnp.float32)
        predictions = targets + jnp.asarray(rng.normal(scale=0.5, size=(5, 4)), dtype=jnp.float32)
        weights = jnp.asarray([1.0, 0.0, 2.0, 0.0, 1.0], dtype=jnp.float32)
        grad = jax.grad(lambda w: relative_l2_error(predictions, targets, weights=w))(weights)
        per_sample = np.linalg.norm(np.asarray(predictions - targets, dtype=np.float64), axis=1) / (
            np.linalg.norm(np.asarray(targets, dtype=np.float64), axis=1) + 1e-8
        )
        np.testing.assert_allclose(
            np.asarray(grad), self._reference(per_sample, np.asarray(weights)), rtol=0, atol=5e-7
        )

    def test_softmax_cross_entropy(self) -> None:
        rng = np.random.default_rng(2)
        logits = jnp.asarray(rng.normal(size=(6, 4)), dtype=jnp.float32)
        labels = jnp.asarray(rng.integers(0, 4, size=6))
        weights = jnp.asarray([1.0, 0.0, 2.0, 0.0, 0.5, 1.0], dtype=jnp.float32)
        grad = jax.grad(lambda w: softmax_cross_entropy(logits, labels, weights=w))(weights)
        elementwise = np.asarray(softmax_cross_entropy(logits, labels, reduction="none"))
        np.testing.assert_allclose(
            np.asarray(grad), self._reference(elementwise, np.asarray(weights)), rtol=0, atol=2e-7
        )


class TestAMaskedElementContributesNothingToTheWeightGradient:
    """A mask says the element is not an observation, so its weight derivative is zero.

    Exactly zero, for every value a masked slot can hold -- including the finite ones a
    folded-weight rule would have to keep, and the overflowing value that is the reason the
    mask cannot be folded into the weights before the finiteness test runs.
    """

    @pytest.mark.parametrize("value", [2.0, float("nan"), float("inf"), float("-inf"), 1e30])
    def test_every_value_a_masked_slot_can_hold(self, value: float) -> None:
        p, t = _sample()
        p = p.at[1].set(value)
        keep = jnp.asarray([True, False, True, True, True, True])
        weights = jnp.ones(6, dtype=jnp.float32)
        grad = jax.grad(lambda w: mse(p, t, mask=keep, weights=w, reduction="sum"))(weights)
        assert float(grad[1]) == 0.0
        assert bool(jnp.all(jnp.isfinite(grad))), "a masked slot must not poison the others"

    @pytest.mark.parametrize("value", [2.0, float("nan"), 1e30])
    def test_the_masked_value_does_not_reach_the_result(self, value: float) -> None:
        p, t = _sample()
        keep = jnp.asarray([True, False, True, True, True, True])
        clean = float(mse(p.at[1].set(0.0), t, mask=keep, reduction="sum"))
        assert float(mse(p.at[1].set(value), t, mask=keep, reduction="sum")) == clean


class TestTheDocumentedCost:
    """A zero weight on a value whose square overflows reads ``NaN``. Mask it instead."""

    def test_zero_weight_on_an_overflowing_value(self) -> None:
        p, t = _sample()
        p = p.at[1].set(1e30)
        weights = jnp.asarray([1.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        assert not np.isfinite(float(mse(p, t, weights=weights, reduction="sum")))

    def test_masking_it_is_the_remedy(self) -> None:
        p, t = _sample()
        p = p.at[1].set(1e30)
        keep = jnp.asarray([True, False, True, True, True, True])
        assert np.isfinite(float(mse(p, t, mask=keep, reduction="sum")))


class TestTransforms:
    """The rule must survive the transforms, gradients with respect to weights included."""

    def test_jit_vmap_and_grad_over_weight_vectors(self) -> None:
        p, t = _sample()
        keep = jnp.asarray([True, False, True, True, True, True])

        def call(w: jnp.ndarray) -> jnp.ndarray:
            return mse(p, t, mask=keep, weights=w, reduction="sum")

        weights = jnp.asarray([1.0, 1.0, 2.0, 0.0, 0.5, 1.0], dtype=jnp.float32)
        assert float(jax.jit(call)(weights)) == pytest.approx(float(call(weights)))
        batched = jax.jit(jax.vmap(jax.grad(call)))(
            jnp.stack([weights, jnp.zeros(6, dtype=jnp.float32)])
        )
        assert bool(jnp.all(jnp.isfinite(batched)))
        # The all-zero row must still carry each element's own loss at the kept slots.
        elementwise = np.asarray(mse(p, t, reduction="none"))
        np.testing.assert_allclose(
            np.asarray(batched[1])[[0, 2, 3, 4, 5]], elementwise[[0, 2, 3, 4, 5]], atol=0
        )
        assert float(batched[1][1]) == 0.0, "the masked slot stays zero under vmap"


#: Each statistic with its float64 reference and the keyword its jax form needs.
STATISTICS = [
    ("r_squared", r_squared, references.r_squared, {}),
    (
        "r_squared_adjusted",
        r_squared_adjusted,
        references.r_squared_adjusted,
        {"num_predictors": 2},
    ),
    ("pearson_correlation", pearson_correlation, references.pearson, {}),
    ("concordance_correlation", concordance_correlation, references.concordance, {}),
    ("kendall_tau", kendall_tau, references.kendall, {}),
    ("spearman_rank_correlation", spearman_rank_correlation, references.spearman, {}),
]

#: The same table for assertions that need no reference formula.
STATISTICS_WITHOUT_REFERENCE = [(n, m, e) for n, m, _ref, e in STATISTICS]


class TestStatisticsMatchAFiniteDifferenceOfTheirOwnFormula:
    """The ratio statistics have no closed form as simple as a loss, so differentiate the
    reference numerically instead -- in float64, against a formula written independently of
    the implementation, at weights containing zeros.
    """

    @pytest.mark.parametrize(
        ("name", "metric", "reference", "extra"), STATISTICS, ids=[s[0] for s in STATISTICS]
    )
    def test_weight_gradient(
        self,
        name: str,
        metric: Callable[..., jnp.ndarray],
        reference: Callable[..., float],
        extra: dict[str, object],
    ) -> None:
        rng = np.random.default_rng(3)
        n = 24
        t64 = rng.normal(size=n)
        p64 = t64 + rng.normal(scale=0.6, size=n)
        w64 = rng.random(n)
        w64[[2, 7, 15]] = 0.0
        p = jnp.asarray(p64, dtype=jnp.float32)
        t = jnp.asarray(t64, dtype=jnp.float32)
        weights = jnp.asarray(w64, dtype=jnp.float32)

        got = np.asarray(
            jax.grad(lambda w: metric(p, t, weights=w, **extra))(weights), dtype=np.float64
        )
        finite_difference = references.central_difference(reference, p64, t64, w64)

        np.testing.assert_allclose(
            got,
            finite_difference,
            rtol=0,
            atol=references.weight_gradient_atol(finite_difference),
            err_msg=name,
        )

    @pytest.mark.parametrize(
        ("name", "metric", "extra"),
        STATISTICS_WITHOUT_REFERENCE,
        ids=[s[0] for s in STATISTICS_WITHOUT_REFERENCE],
    )
    def test_the_zero_weight_slots_are_not_all_zero(
        self, name: str, metric: Callable[..., jnp.ndarray], extra: dict[str, object]
    ) -> None:
        """A rule that substitutes the value reads exactly zero at those slots."""
        rng = np.random.default_rng(3)
        n = 24
        t64 = rng.normal(size=n)
        p64 = t64 + rng.normal(scale=0.6, size=n)
        w64 = rng.random(n)
        w64[[2, 7, 15]] = 0.0
        got = jax.grad(
            lambda w: metric(
                jnp.asarray(p64, jnp.float32), jnp.asarray(t64, jnp.float32), weights=w, **extra
            )
        )(jnp.asarray(w64, dtype=jnp.float32))
        at_zero = np.asarray(got)[[2, 7, 15]]
        assert np.all(at_zero != 0.0), f"{name} returned a zero derivative at a zero weight"
