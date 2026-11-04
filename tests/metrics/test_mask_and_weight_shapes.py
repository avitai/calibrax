"""``mask`` and ``weights`` broadcast against the input's own shape, for every statistic.

Six statistics take a ``mask`` and ``weights``. All six document the parameters as
"broadcastable", and these tests hold every one of them to it. Ravelling the
inputs first and broadcasting the mask against the **raveled** shape breaks that contract: a
same-shape 2-D mask over 2-D data then raises
``ValueError: Cannot broadcast to shape with fewer dimensions``.

The convention is settled by the package, not by preference.
:func:`calibrax.metrics._utils.reduce_values` -- the function whose semantics this whole
change set exists to extend to the ratio statistics -- broadcasts with
``jnp.broadcast_to(weights, values.shape)`` against the values' own shape, and documents the
parameters as "broadcastable to ``values``". ``r_squared`` already matched it. So the five
are brought to the sixth rather than the reverse.

``sklearn`` argues the same way by exclusion: ``r2_score`` rejects a 2-D ``sample_weight``
outright ("Sample weights must be 1D array or scalar") and weights *rows* for multioutput
targets. That model cannot apply to a statistic that ravels everything into one number, and
it offers no support for broadcasting against a raveled shape either.

What that yields for ``(8, 5)`` data, by ordinary numpy rules: a ``(8, 5)`` mask works, a
length-5 row vector broadcasts across rows, a scalar works, and a length-8 column vector
raises -- correctly, because aligning it would require guessing which axis it indexes.
"""

from __future__ import annotations

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


#: Float32 agreement budget. A broadcast-then-ravel and an explicit ravel compute the same
#: reductions over the same elements, so only reassociation can separate them.
TOLERANCE = 1e-6

#: Deliberately non-square, so a length-8 and a length-5 vector cannot both broadcast and
#: the row/column distinction is observable.
SHAPE = (8, 5)

#: Every statistic taking both parameters, and the keyword arguments it additionally requires.
STATISTICS: list[tuple[str, Callable[..., jnp.ndarray], dict[str, object]]] = [
    ("r_squared", r_squared, {}),
    ("r_squared_adjusted", r_squared_adjusted, {"num_predictors": 2}),
    ("pearson_correlation", pearson_correlation, {}),
    ("concordance_correlation", concordance_correlation, {}),
    ("kendall_tau", kendall_tau, {}),
    ("spearman_rank_correlation", spearman_rank_correlation, {}),
]


@pytest.fixture
def two_dimensional() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Correlated predictions and targets of shape ``SHAPE``."""
    rng = np.random.default_rng(0)
    targets = jnp.asarray(rng.normal(size=SHAPE), dtype=jnp.float32)
    predictions = targets + jnp.asarray(rng.normal(scale=0.4, size=SHAPE), dtype=jnp.float32)
    return predictions, targets


def _row_mask() -> jnp.ndarray:
    """A length-5 boolean over the trailing axis, which broadcasts across rows."""
    return jnp.asarray([True, False, True, True, False])


def _row_weights() -> jnp.ndarray:
    """A length-5 weight vector over the trailing axis."""
    return jnp.asarray([0.5, 1.0, 2.0, 0.25, 1.5], dtype=jnp.float32)


class TestTheMaskBroadcastsAgainstTheInput:
    """A mask shaped like the data, or broadcastable to it, is accepted and correct."""

    @pytest.mark.parametrize(
        ("name", "metric", "extra"), STATISTICS, ids=[m[0] for m in STATISTICS]
    )
    def test_a_same_shape_mask_equals_the_explicitly_raveled_computation(
        self,
        two_dimensional: tuple[jnp.ndarray, jnp.ndarray],
        name: str,
        metric: Callable[..., jnp.ndarray],
        extra: dict[str, object],
    ) -> None:
        """The statistic is over all elements, so the 2-D call must match the flat one."""
        predictions, targets = two_dimensional
        rng = np.random.default_rng(1)
        mask = jnp.asarray(rng.random(SHAPE) > 0.3)
        flat = float(metric(predictions.ravel(), targets.ravel(), mask=mask.ravel(), **extra))
        assert float(metric(predictions, targets, mask=mask, **extra)) == pytest.approx(
            flat, abs=TOLERANCE
        ), f"{name} disagrees with its own raveled form"

    @pytest.mark.parametrize(
        ("name", "metric", "extra"), STATISTICS, ids=[m[0] for m in STATISTICS]
    )
    def test_a_row_mask_broadcasts_across_rows(
        self,
        two_dimensional: tuple[jnp.ndarray, jnp.ndarray],
        name: str,
        metric: Callable[..., jnp.ndarray],
        extra: dict[str, object],
    ) -> None:
        """A length-5 mask selects columns, the same way numpy would broadcast it."""
        predictions, targets = two_dimensional
        row = _row_mask()
        expanded = jnp.broadcast_to(row, SHAPE)
        flat = float(metric(predictions.ravel(), targets.ravel(), mask=expanded.ravel(), **extra))
        assert float(metric(predictions, targets, mask=row, **extra)) == pytest.approx(
            flat, abs=TOLERANCE
        ), f"{name} does not broadcast a row mask the way numpy does"

    @pytest.mark.parametrize(
        ("name", "metric", "extra"), STATISTICS, ids=[m[0] for m in STATISTICS]
    )
    def test_a_column_mask_is_refused(
        self,
        two_dimensional: tuple[jnp.ndarray, jnp.ndarray],
        name: str,
        metric: Callable[..., jnp.ndarray],
        extra: dict[str, object],
    ) -> None:
        """A length-8 mask over ``(8, 5)`` data is ambiguous and must raise, not guess.

        Numpy aligns trailing axes, so this cannot broadcast. Accepting it by ravelling the
        mask instead would silently reinterpret it as the first 8 elements in row-major
        order, which is not what a caller passing a per-row mask means.
        """
        predictions, targets = two_dimensional
        column = jnp.asarray(np.arange(SHAPE[0]) % 2 == 0)
        with pytest.raises(ValueError, match=r"(?i)broadcast") as raised:
            metric(predictions, targets, mask=column, **extra)
        assert "broadcast" in str(raised.value).lower(), (
            f"{name} rejected the ambiguous mask, but not with a shape error"
        )


class TestAFloatMaskIsAMask:
    """A 0/1 float mask means what the boolean mask means, in every statistic.

    :func:`calibrax.metrics._utils.reduce_values` folds a mask with ``jnp.where``, so a
    numeric mask's nonzero entries keep and its zeros exclude, and ``perplexity`` documents
    "Booleans and 0/1 floats are both accepted". A statistic that refuses the float form
    breaks the one convention the masking change set extends.
    """

    @pytest.mark.parametrize(
        ("name", "metric", "extra"), STATISTICS, ids=[m[0] for m in STATISTICS]
    )
    def test_a_float_mask_equals_its_boolean_form(
        self,
        two_dimensional: tuple[jnp.ndarray, jnp.ndarray],
        name: str,
        metric: Callable[..., jnp.ndarray],
        extra: dict[str, object],
    ) -> None:
        """Same elements kept, so the same value, bit for bit."""
        predictions, targets = two_dimensional
        keep = np.random.default_rng(2).random(SHAPE) > 0.3
        as_bool = float(metric(predictions, targets, mask=jnp.asarray(keep), **extra))
        as_float = float(
            metric(predictions, targets, mask=jnp.asarray(keep, dtype=jnp.float32), **extra)
        )
        assert as_float == as_bool, f"{name} reads a 0/1 float mask unlike its boolean form"


class TestTheWeightsBroadcastAgainstTheInput:
    """Identically for weights, on every statistic."""

    @pytest.mark.parametrize(
        ("name", "metric", "extra"), STATISTICS, ids=[m[0] for m in STATISTICS]
    )
    def test_same_shape_weights_equal_the_explicitly_raveled_computation(
        self,
        two_dimensional: tuple[jnp.ndarray, jnp.ndarray],
        name: str,
        metric: Callable[..., jnp.ndarray],
        extra: dict[str, object],
    ) -> None:
        predictions, targets = two_dimensional
        rng = np.random.default_rng(2)
        weights = jnp.asarray(rng.random(SHAPE) + 0.05, dtype=jnp.float32)
        flat = float(metric(predictions.ravel(), targets.ravel(), weights=weights.ravel(), **extra))
        assert float(metric(predictions, targets, weights=weights, **extra)) == pytest.approx(
            flat, abs=TOLERANCE
        ), f"{name} disagrees with its own raveled form"

    @pytest.mark.parametrize(
        ("name", "metric", "extra"), STATISTICS, ids=[m[0] for m in STATISTICS]
    )
    def test_row_weights_broadcast_across_rows(
        self,
        two_dimensional: tuple[jnp.ndarray, jnp.ndarray],
        name: str,
        metric: Callable[..., jnp.ndarray],
        extra: dict[str, object],
    ) -> None:
        predictions, targets = two_dimensional
        row = _row_weights()
        flat = float(
            metric(
                predictions.ravel(),
                targets.ravel(),
                weights=jnp.broadcast_to(row, SHAPE).ravel(),
                **extra,
            )
        )
        assert float(metric(predictions, targets, weights=row, **extra)) == pytest.approx(
            flat, abs=TOLERANCE
        ), f"{name} does not broadcast row weights the way numpy does"

    @pytest.mark.parametrize(
        ("name", "metric", "extra"), STATISTICS, ids=[m[0] for m in STATISTICS]
    )
    def test_scalar_weights_are_accepted(
        self,
        two_dimensional: tuple[jnp.ndarray, jnp.ndarray],
        name: str,
        metric: Callable[..., jnp.ndarray],
        extra: dict[str, object],
    ) -> None:
        """A scalar scales every element, so it must not change a scale-free statistic."""
        predictions, targets = two_dimensional
        scaled = float(metric(predictions, targets, weights=jnp.asarray(2.0), **extra))
        unit = float(metric(predictions, targets, weights=jnp.ones(SHAPE), **extra))
        assert scaled == pytest.approx(unit, abs=TOLERANCE), (
            f"{name} depends on the overall scale of its weights"
        )


class TestOneDimensionalInputIsUnchanged:
    """The common case must not move, since every existing caller is one-dimensional."""

    @pytest.mark.parametrize(
        ("name", "metric", "extra"), STATISTICS, ids=[m[0] for m in STATISTICS]
    )
    def test_a_flat_mask_over_flat_data_still_works(
        self,
        name: str,
        metric: Callable[..., jnp.ndarray],
        extra: dict[str, object],
    ) -> None:
        rng = np.random.default_rng(3)
        size = SHAPE[0] * SHAPE[1]
        targets = jnp.asarray(rng.normal(size=size), dtype=jnp.float32)
        predictions = targets + jnp.asarray(rng.normal(scale=0.4, size=size), dtype=jnp.float32)
        mask = jnp.asarray(rng.random(size) > 0.3)
        assert jnp.isfinite(metric(predictions, targets, mask=mask, **extra)), (
            f"{name} stopped accepting a flat mask over flat data"
        )


class TestTransformations:
    """The broadcast must not cost traceability, weights gradients included."""

    @pytest.mark.parametrize(
        ("name", "metric", "extra"), STATISTICS, ids=[m[0] for m in STATISTICS]
    )
    def test_jit_vmap_and_grad_over_two_dimensional_input(
        self,
        two_dimensional: tuple[jnp.ndarray, jnp.ndarray],
        name: str,
        metric: Callable[..., jnp.ndarray],
        extra: dict[str, object],
    ) -> None:
        predictions, targets = two_dimensional
        weights = jnp.full(SHAPE, 1.5, dtype=jnp.float32)

        def call(p: jnp.ndarray, t: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
            return metric(p, t, weights=w, **extra)

        assert jnp.isfinite(jax.jit(call)(predictions, targets, weights))
        batched = jax.jit(jax.vmap(call))(
            jnp.stack([predictions] * 3), jnp.stack([targets] * 3), jnp.stack([weights] * 3)
        )
        assert batched.shape == (3,)
        grad_w = jax.grad(lambda w: call(predictions, targets, w))(weights)
        assert jnp.all(jnp.isfinite(grad_w)), (
            f"{name} has a non-finite gradient with respect to its weights"
        )
        # A zero weight is an observation carrying weight zero, so ``d/dw`` there is that
        # element's own contribution -- which means it must depend on the element's value.
        # A rule that substitutes the value away leaves a derivative that does not: for a
        # mean-reduced statistic it reads ``(0 - m) / sum(w)``, which is finite and non-zero,
        # so neither a finiteness check nor a ``!= 0.0`` check can tell the two apart. Moving
        # the value is what separates them.
        zeroed = weights.at[0, 0].set(0.0)

        def at_zero(values: jnp.ndarray) -> jnp.ndarray:
            return jax.grad(lambda w: call(values, targets, w))(zeroed)

        here = at_zero(predictions)
        assert jnp.all(jnp.isfinite(here)), f"{name} is non-finite at a zero weight"
        # The unique minimum, so the rank statistic moves as well: measured on this fixture,
        # ``kendall_tau`` shifts 0.0666 when the element is lowered and nothing at all when it
        # is raised, because raising it past the maximum flips no pair.
        moved = at_zero(predictions.at[0, 0].set(predictions.min() - 1.0))
        assert float(here[0, 0]) != float(moved[0, 0]), (
            f"{name} ignores a zero-weighted element's value in its weight derivative"
        )
