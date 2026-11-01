"""Shared utilities for metric functions.

Provides numerical stability guards and validation helpers used across
all metric modules. Private module — not part of the public API.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


_EPSILON = 1e-8
_EPSILON_CLIP = 1e-7
#: The one root order with a dedicated primitive: `jnp.sqrt` rather than a general power.
_SQUARE_ROOT_ORDER = 2.0


def _validate_shapes(predictions: ArrayLike, targets: ArrayLike) -> None:
    """Validate that predictions and targets have matching shapes.

    Args:
        predictions: Predicted values array.
        targets: Ground truth values array.

    Raises:
        ValueError: If shapes do not match.
    """
    p = jnp.asarray(predictions)
    t = jnp.asarray(targets)
    if p.shape != t.shape:
        msg = f"Shape mismatch: predictions {p.shape} vs targets {t.shape}"
        raise ValueError(msg)


def _prepare_arrays(predictions: ArrayLike, targets: ArrayLike) -> tuple[jax.Array, jax.Array]:  # noqa: DOC502  # raised by _validate_shapes
    """Validate shapes and convert predictions/targets to JAX arrays.

    Combines shape validation with array conversion — the standard
    preamble for regression, calibration, and similar metrics.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Tuple of (predictions_array, targets_array) as JAX arrays.

    Raises:
        ValueError: If shapes do not match.
    """
    _validate_shapes(predictions, targets)
    return jnp.asarray(predictions), jnp.asarray(targets)


def _prepare_class_arrays(  # noqa: DOC502  # raised by _validate_shapes
    predictions: ArrayLike, targets: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """Validate shapes and convert to int32 class index arrays.

    Standard preamble for classification, segmentation, and clustering
    metrics that operate on discrete class labels.

    Args:
        predictions: Predicted class labels.
        targets: Ground truth class labels.

    Returns:
        Tuple of (predictions_array, targets_array) as int32 JAX arrays.

    Raises:
        ValueError: If shapes do not match.
    """
    _validate_shapes(predictions, targets)
    return (
        jnp.asarray(predictions).astype(jnp.int32),
        jnp.asarray(targets).astype(jnp.int32),
    )


def safe_divide(
    numerator: ArrayLike,
    denominator: ArrayLike,
    *,
    eps: float = _EPSILON,
) -> jax.Array:
    """Division guarded against zero/near-zero denominators.

    Replaces scattered ``x / (y + _EPSILON)`` patterns with a centralized,
    consistent numerical stability guard. All metric modules should use
    this instead of manual epsilon addition.

    Args:
        numerator: Dividend array or scalar.
        denominator: Divisor array or scalar.
        eps: Small constant added to denominator to prevent division by zero.

    Returns:
        Result of numerator / (denominator + eps).
    """
    return jnp.asarray(numerator) / (jnp.asarray(denominator) + eps)


def safe_root(x: ArrayLike, *, order: float = 2.0) -> jax.Array:
    """The ``order``-th root of non-negative ``x``, with derivative 0 where ``x`` is 0.

    The plain root's derivative is infinite at 0, so ``jax.grad`` of a distance built on it
    is NaN at a perfect match. Following the JAX FAQ ("Gradients contain NaN where using
    where"), the root is taken of a value guarded by an inner ``where`` and the zero case is
    selected by an outer one; the outer ``where`` alone keeps the NaN. The derivative at 0 is
    0, the convention of ``optax.safe_norm``.

    **The predicate selects the degenerate branch, so a ``NaN`` reaches the root.** Written
    the other way round -- selecting the root branch on ``x > 0.0`` -- a ``NaN`` fails that
    comparison too and leaves as ``0.0``, silently converting missing data into a perfect
    score. The convention this function cites does not do that: measured on optax 0.2.8,
    ``safe_norm([nan, 1.0], 0.0)`` returns ``nan`` while ``safe_norm([0.0, 0.0], 0.0)``
    returns ``0.0`` with a zero gradient, because its predicate is ``norm <= min_norm`` and
    ``NaN`` fails ``<=``, landing on the unguarded side. This is that predicate.

    **A negative ``x`` is out of contract and yields ``NaN``**, which is what the references
    do; clamping it to zero instead would hide a domain error. Measured: ``numpy.sqrt(-4.0)`` is ``nan`` with
    "invalid value encountered in sqrt", ``jax.numpy.sqrt(-4.0)`` is ``nan``,
    ``scipy.spatial.distance.mahalanobis`` with an indefinite inverse-covariance is ``nan``
    (scipy 1.17.0), and ``sklearn.metrics.pairwise_distances(metric="mahalanobis")`` with the
    same matrix is ``nan`` (1.8.0), as is ``sklearn.covariance.EmpiricalCovariance``'s
    squared distance. None of them raises and none clamps. ``optax`` cannot be consulted:
    every one of its safe helpers norms or squares its input first, so none takes an
    already-negative radicand.

    Raising was not an option rather than a rejected one. A Python branch on the value is
    ``TracerBoolConversionError`` under both ``jax.jit`` and ``jax.vmap``;
    ``jax.experimental.checkify`` works but changes the return to ``(err, value)``, which
    would propagate through all 24 call sites and every public metric built on them; and
    ``jax.debug.print`` is a host callback that cannot stop the computation.

    The guard is therefore exactly the singular point and nothing else: the derivative of
    the root is infinite at zero, which is the one case needing a branch. A negative is not
    a singularity, it is a domain error, and it now reads as one. Measured over 450
    adversarial calls across all nine consumer modules -- magnitudes from 1e-20 to 1e20,
    identical and near-identical inputs, ill-conditioned and singular matrices -- the only
    radicand that ever goes negative is ``mahalanobis_distance``'s ``diff @ prec @ diff``
    with a caller-supplied matrix that is **not** positive semi-definite. A PSD matrix never
    does: over 200 trials per cell, identical points give exactly ``0`` and points 1e-7
    apart give ``+4.4e-14``, and scipy returns no ``NaN`` in 500 near-identical PSD pairs
    either. So the inputs this newly rejects are exactly the ones every reference rejects.
    Sums of squares, absolute values raised to a power, and the two radicands ``mmd`` and
    ``hellinger_distance`` already clamp with ``jnp.maximum(..., 0.0)`` are all unaffected.

    **At order 2 the root is ``jnp.sqrt``, not ``x ** 0.5``.** The two are the same number
    once compiled and not when interpreted: measured, jitted ``pow(x, 0.5)`` and jitted
    ``sqrt(x)`` agree on all 1,000,000 float32 inputs tried, while eager ``pow`` lands one
    ULP away on 691 of them. A power-based root therefore disagrees with its own compiled
    path, which is what breaks an exact comparison between a jitted metric and an eager
    reference. Their agreement is below HLO rather than in it: a ``sqrt`` module and a
    ``pow(x, 0.5)`` module are two different graphs that compute the same float32. So
    choosing the root does not preserve the graph -- it preserves every fusion count (21 of
    21 compiled metrics, and 17 of 17 including the gradient and ``vmap`` forms) and every
    jitted value, which is what callers can observe.

    Eager is also faster, since ``sqrt`` is a hardware instruction and ``pow`` a
    transcendental: on 262,144 elements the eager call was faster in all five paired rounds,
    by a factor between 1.7 and 2.5 depending on machine load. At eight elements the
    difference sits inside the measurement's own noise floor.

    ``-inf`` turns on the choice of root. IEEE defines ``pow(-inf, 0.5)`` as
    ``+inf``, so a power-based root reads a negative radicand of infinite magnitude as
    *infinitely far apart* rather than as the domain error every smaller negative gives;
    ``jnp.sqrt`` makes it ``NaN``, as does ``numpy.sqrt``. Only ``mahalanobis_distance``
    with a matrix that is not positive semi-definite can form such a radicand.

    Args:
        x: Non-negative values.
        order: The root's order; 2 is the square root, and is taken as ``jnp.sqrt`` when it
            is a static Python number. A traced ``order`` takes the general power path.

    Returns:
        ``x ** (1 / order)`` element-wise; ``NaN`` where ``x`` is ``NaN`` or negative, and
        ``+0.0`` at either signed zero, where the guard overrides the root's own sign.
    """
    x = jnp.asarray(x)
    at_the_singularity = x == 0.0
    safe = jnp.where(at_the_singularity, jnp.ones_like(x), x)
    # `order` is a Python float by contract, but `minkowski_distance` and
    # `sliced_wasserstein` forward a caller's `p`, which `jit` over that argument turns into
    # a tracer. `isinstance` decides the branch from the static value alone, so a traced
    # order takes the general power path rather than raising on a Python `if`.
    takes_the_square_root = isinstance(order, (int, float)) and order == _SQUARE_ROOT_ORDER
    root = jnp.sqrt(safe) if takes_the_square_root else safe ** (1.0 / order)
    return jnp.where(at_the_singularity, jnp.zeros_like(x), root)


def safe_norm(x: ArrayLike, *, axis: int | tuple[int, ...] | None = None) -> jax.Array:
    """The Euclidean norm of ``x`` over ``axis`` (all elements when ``None``), zero-safe.

    ``jnp.linalg.norm`` has a NaN gradient at the zero vector, which a distance reaches at a
    perfect match; this is ``safe_root`` of the sum of squares, with gradient 0 there.

    The radicand is ``sum(square(x))``, non-negative by construction, so the domain error
    :func:`safe_root` reports for a negative is unreachable from here. A ``NaN`` in ``x``
    still propagates, as it should.

    Args:
        x: Values.
        axis: Axis or axes to reduce.

    Returns:
        The norm, reduced over ``axis``.
    """
    return safe_root(jnp.sum(jnp.square(x), axis=axis))


def dense_labels(labels: ArrayLike, num_labels: int | None) -> tuple[jax.Array, int]:
    """Labels as ids ``0..k-1``, and ``k``: labels are names, so their values carry no order.

    With ``num_labels`` given, ``labels`` are taken as ids in ``[0, num_labels)`` and ``k`` is
    that static count, so a metric built on it traces; ids with no member are empty. With
    ``None``, the distinct values are renumbered in sorted order and counted from the data,
    which runs only eagerly (the count is a shape).

    Args:
        labels: Integer labels.
        num_labels: The number of ids, or None to count the distinct labels.

    Returns:
        The ids and their count.
    """
    ids = jnp.asarray(labels, dtype=jnp.int32)
    if num_labels is not None:
        return ids, num_labels
    values, inverse = jnp.unique(ids, return_inverse=True)
    return inverse.reshape(ids.shape), int(values.shape[0])


def label_masks(labels: ArrayLike, num_labels: int | None) -> tuple[jax.Array, jax.Array]:
    """One-hot membership of each label id, ``(k, n)`` as floats, and each id's size ``(k,)``.

    Ids come from :func:`dense_labels`: ``num_labels`` given traces, None counts eagerly.

    Args:
        labels: Integer labels, shape ``(n,)``.
        num_labels: The number of ids, or None to count the distinct labels.

    Returns:
        The membership masks and the members per id; an unused id has size 0.
    """
    ids, count = dense_labels(labels, num_labels)
    masks = (ids[None, :] == jnp.arange(count)[:, None]).astype(jnp.float32)
    return masks, jnp.sum(masks, axis=1)


def safe_log(x: ArrayLike, *, eps: float = _EPSILON) -> jax.Array:
    """Logarithm guarded against zero/negative inputs.

    Clamps input to ``[eps, inf)`` before taking log. Essential for
    divergence, information-theoretic, and calibration metrics that
    compute log of probabilities or ratios.

    Args:
        x: Input array or scalar.
        eps: Minimum value to clamp to before log.

    Returns:
        ``jnp.log(max(x, eps))``.
    """
    return jnp.log(jnp.maximum(x, eps))


# Ensemble forecasts are (samples, members) with at least two members.
_ENSEMBLE_NDIM = 2
# A matrix of records or features: (n_samples, n_features).
_FEATURE_MATRIX_NDIM = 2
# A batch of series: (batch, sequence, features).
_SERIES_NDIM = 3
# Feature correlations need at least two features.
_MIN_CORRELATED_FEATURES = 2
_MIN_ENSEMBLE_MEMBERS = 2
_MULTIVARIATE_ENSEMBLE_NDIM = 3


def _prepare_ensemble_arrays(
    predictions: ArrayLike, targets: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """Validate and convert an ensemble forecast and its targets.

    Args:
        predictions: Forecast ensemble with shape ``(n_samples, n_members)``.
        targets: Observed targets with shape ``(n_samples,)``; a scalar is accepted
            for a single sample.

    Returns:
        Tuple of float32 JAX arrays ``(predictions, targets)``.

    Raises:
        ValueError: If the ensemble is not two-dimensional with at least two
            members, or the targets do not match its sample count.
    """
    pred = jnp.asarray(predictions, dtype=jnp.float32)
    target = jnp.asarray(targets, dtype=jnp.float32)
    if pred.ndim != _ENSEMBLE_NDIM:
        msg = f"predictions must be 2-dimensional, got shape {pred.shape}"
        raise ValueError(msg)
    if pred.shape[1] < _MIN_ENSEMBLE_MEMBERS:
        msg = f"predictions must contain at least two ensemble members, got {pred.shape[1]}"
        raise ValueError(msg)
    if target.ndim == 0:
        target = target[None]
    if target.ndim != 1:
        msg = f"targets must be scalar or 1-dimensional, got shape {target.shape}"
        raise ValueError(msg)
    if pred.shape[0] != target.shape[0]:
        msg = (
            "predictions and targets must have matching sample count: "
            f"{pred.shape[0]} != {target.shape[0]}"
        )
        raise ValueError(msg)
    return pred, target


def _prepare_multivariate_ensemble_arrays(
    predictions: ArrayLike, targets: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """Validate and convert a multivariate ensemble forecast and its targets.

    Args:
        predictions: Ensemble with shape ``(n_samples, n_members, n_outputs)``.
        targets: Observed targets with shape ``(n_samples, n_outputs)``.

    Returns:
        Tuple of float32 JAX arrays ``(predictions, targets)``.

    Raises:
        ValueError: If the shapes are not ``(n, m, d)`` and ``(n, d)`` with at
            least two members.
    """
    pred = jnp.asarray(predictions, dtype=jnp.float32)
    target = jnp.asarray(targets, dtype=jnp.float32)
    if pred.ndim != _MULTIVARIATE_ENSEMBLE_NDIM:
        msg = f"predictions must be 3-dimensional (samples, members, outputs), got {pred.shape}"
        raise ValueError(msg)
    if pred.shape[1] < _MIN_ENSEMBLE_MEMBERS:
        msg = f"predictions must contain at least two ensemble members, got {pred.shape[1]}"
        raise ValueError(msg)
    if target.shape != (pred.shape[0], pred.shape[2]):
        msg = (
            f"targets must have shape (n_samples, n_outputs) = {(pred.shape[0], pred.shape[2])}, "
            f"got {target.shape}"
        )
        raise ValueError(msg)
    return pred, target


def matrix_sqrtm(matrix: ArrayLike, *, eps: float = _EPSILON) -> jax.Array:
    """Square root of a symmetric positive semi-definite matrix by eigendecomposition.

    Eigenvalues are floored at ``eps`` before the square root, which keeps the
    result real and finite for covariance matrices that are only semi-definite in
    floating point. Stays on backends where ``jax.scipy.linalg.sqrtm`` has no kernel.

    Args:
        matrix: Symmetric positive semi-definite matrix with shape ``(d, d)``.
        eps: Floor applied to the eigenvalues.

    Returns:
        The matrix square root with shape ``(d, d)``.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(jnp.asarray(matrix))
    sqrt_eigenvalues = jnp.sqrt(jnp.maximum(eigenvalues, eps))
    return eigenvectors @ jnp.diag(sqrt_eigenvalues) @ eigenvectors.T


_REDUCTIONS = ("none", "mean", "sum", "batch_sum")


def sample_weights(
    mask: ArrayLike | None,
    weights: ArrayLike | None,
    shape: tuple[int, ...],
) -> jax.Array:
    """Fold a mask and per-element weights into the single weight vector both mean.

    The semantics :func:`reduce_values` documents, made available to statistics that cannot
    reduce through it. A mask excludes an element by giving it zero weight, which is exactly
    what "contributes nothing to a sum and is not counted in a mean" means, so one weight
    vector expresses both and no statistic has to handle them separately.

    Traceable: the ``None`` checks are on static arguments and resolve at trace time, so the
    result works under ``jit``, ``vmap`` and ``grad``, gradients with respect to ``weights``
    included.

    Args:
        mask: Elements to keep (boolean, broadcastable), or ``None`` for all.
        weights: Element weights (broadcastable), or ``None`` for unit weights.
        shape: Shape the weights must broadcast to.

    Returns:
        One weight array of shape ``shape``, zero wherever the mask excludes.
    """
    combined = jnp.ones(shape) if weights is None else jnp.broadcast_to(jnp.asarray(weights), shape)
    if mask is None:
        return combined
    return jnp.where(jnp.broadcast_to(jnp.asarray(mask), shape), combined, 0.0)


def weighted_mean(values: ArrayLike, weights: ArrayLike) -> jax.Array:
    """The weighted mean ``sum(w * x) / sum(w)``.

    Factored out because three statistics need it, and each deriving it separately is a
    familiar way for a weighted metric to drift from its reference.

    Args:
        values: Values to average.
        weights: Weights, broadcastable to ``values``.

    Returns:
        The weighted mean as a scalar array.
    """
    w = jnp.asarray(weights)
    return jnp.sum(w * jnp.asarray(values)) / (jnp.sum(w) + _EPSILON)


def coefficient_of_determination(
    predictions: jax.Array,
    targets: jax.Array,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> jax.Array:
    """``1 - SS_res / SS_tot``, the one place this package computes R-squared.

    Lives here rather than in :mod:`calibrax.metrics.functional.regression` because three
    callers need it and two of them are in other modules: ``r_squared``,
    ``r_squared_adjusted`` and the fused regression suite. Every module that needs it
    already imports this one, so sharing it adds no dependency edge between siblings.

    **Two paths, chosen at trace time.** With neither ``mask`` nor ``weights`` the sums are
    unweighted and the centre is :func:`jax.numpy.mean`, which is exactly what the three
    former copies computed -- so sharing this function changes no existing result by a
    single bit, measured over 240 cases. Weighting instead through a vector of ones was
    tried and rejected: it shifted the unweighted answer by up to 2.5 float32 ULP and left
    the compiled regression suite 4 fusions larger, all to express "no weights". The
    ``None`` checks are static, as in :func:`sample_weights`, so both paths trace.

    The residual is written ``p - t`` rather than ``t - p`` on purpose. They are bit-equal
    once squared, but the fused regression suite binds ``diff = p - t``, and XLA reuses
    that binding only for the spelling it already has: written the other way round the
    compiled suite grew from 118 fusions to 122. Keeping the spelling is what lets the
    fused path delegate here for free.

    When weights are given, ``SS_tot`` is centred on the **weighted** mean of the targets.
    Weighting both sums while centring on the unweighted mean is the usual error and gives
    a different number; the weighted mean is what
    ``sklearn.metrics.r2_score(sample_weight=...)`` uses.

    A selection with no members returns ``0.0``, the result :func:`reduce_values` documents
    for that case and the one :func:`weighted_mean` already returns when its scale sums to
    zero. The guard keys on the **weight sum**, not on ``SS_tot``: constant targets have no
    variance to explain but are still a sample, and keying on ``SS_tot`` would return
    ``0.0`` for them, making a full mask disagree with passing no mask at all. Only the
    weighted branch needs this -- the unweighted branch has no mask and so cannot be
    empty, which is why the fused suite's call is untouched.

    Two passes over the targets, deliberately. The one-pass identity
    ``sum(w t^2) - sum(w) * mean^2`` was measured and rejected: in float32 with targets
    offset by 1e4 it returns 0.9875 where this returns 0.9116 against a float64 truth of
    0.9116, and it is no faster on GPU either (72.1 us against 73.7 us at 4,096 points),
    so it trades accuracy for nothing.

    Args:
        predictions: Predicted values, already converted. Every caller has them from
            :func:`_prepare_arrays`, and converting again is not free on the eager path:
            re-wrapping here cost ``r_squared`` 127 extra Python-level calls per call,
            22% more interpreter work for an identical compiled graph.
        targets: Ground truth values, already converted.
        mask: Elements to keep (boolean, broadcastable), or ``None`` for all.
        weights: Element weights (broadcastable), or ``None`` for unit weights.

    Returns:
        The coefficient of determination as a scalar array.
    """
    p = predictions
    t = targets
    if mask is None and weights is None:
        return 1.0 - jnp.sum((p - t) ** 2) / (jnp.sum((t - jnp.mean(t)) ** 2) + _EPSILON)
    w = sample_weights(mask, weights, t.shape)
    # Before any arithmetic, so an excluded NaN reaches neither the sums nor the gradient.
    p = substitute_excluded(p, w, mask)
    t = substitute_excluded(t, w, mask)
    ss_res = jnp.sum(w * (p - t) ** 2)
    ss_tot = jnp.sum(w * (t - weighted_mean(t, w)) ** 2)
    # An empty selection has no sample to describe. Without this it reported exactly 1.0 --
    # both sums are zero, so the ratio is 0/_EPSILON -- a perfect fit from no observations,
    # which under `vmap` put an empty bin at the top of a per-bin curve.
    return jnp.where(jnp.sum(w) > 0.0, 1.0 - ss_res / (ss_tot + _EPSILON), 0.0)


def substitute_excluded(values: ArrayLike, weights: jax.Array, mask: ArrayLike | None) -> jax.Array:
    """``values`` with every element the weights exclude replaced by zero.

    Applied to the **inputs**, before any arithmetic reads them. That placement is the
    whole point: a zero weight does not neutralise a bad value in a product, because
    ``0 * nan`` is ``nan`` under IEEE 754, so one excluded ``NaN`` poisons an entire
    masked reduction. Substituting afterwards repairs the value and leaves the gradient
    ``NaN``, since :func:`jax.numpy.where` still differentiates the branch it did not
    take -- measured, the post-hoc form returned the correct number with a gradient of
    ``[0.2, nan, -0.2, 0.4]``.

    This is the idiom :func:`safe_root` already documents for the root at zero, quoting
    the JAX FAQ and ``optax.safe_norm``: guard the input with an inner ``where``, select
    with an outer one. Here only the inner half is needed, because the excluded element's
    contribution is zero by construction rather than something to select afterwards.

    **A zero weight and an excluded element are not the same thing**, which is why this takes
    the mask as well as the folded weights. A mask says the element is not an observation, so
    it goes whatever it holds. A zero weight says the observation is present carrying weight
    zero, so its value stays and ``d/dw_i`` remains its contribution -- the element's
    influence, the quantity case-deletion diagnostics and sample-reweighting methods
    differentiate. Substituting it computes that derivative for a point moved to the origin
    and returns ``0``.

    The two cannot be recovered from the folded vector, so they are not folded first. The case
    that forces it is a value that is finite but whose square overflows float32: masking such
    an element has to discard it, and a rule keyed on the folded weight alone leaves it in --
    measured, ``0 * inf`` then makes the *masked* result ``NaN``, so masking stops working.

    A ``NaN`` at a zero weight is still substituted, since ``0 * NaN`` is ``NaN`` and the
    derivative of a ``NaN`` observation is not worth preserving. ``tf.math.multiply_no_nan``
    goes further, returning zero "even if x is NaN or infinite", which would discard the
    weight derivative of every zero-weighted element.

    One cost, and it buys the gradient: a **zero weight** on a finite value whose square
    overflows reads ``NaN``. To discard an element, mask it.

    This is the design Google Research's ``weatherbenchX`` arrived at for the same problem,
    in a domain where missing observations are the norm rather than the exception. Its
    aggregator substitutes before reducing, for the reason named in its own comment::

        # Set masked values to Zero for stat and weights, which will therefore
        # be ignored in mean_statistics(). this is equivalent to multiplying by
        # the mask, but avoids NaN * 0 -> NaN in cases where there are NaNs in
        # masked positions.

    It also records the hazard this function exists to avoid, as a known wart of its own
    unmasked path:
    "even if the input NaNs are outside the binning mask ... the aggregated statistics will
    still be NaN. Use the masking option to avoid this." By contrast ``metrax``, a
    general-purpose JAX metrics library, multiplies (``absolute_error * sample_weights``)
    and carries the hazard -- so the library that had to solve it, did, and the one that did
    not, did not.

    A caller that passes neither argument skips this entirely, by a trace-time branch rather
    than by relying on constant folding. On the weighted path the cost is one extra pass over
    memory per array, and it is small enough that measuring it needs care. Timing the two
    trees in separate processes was rejected on its own control: ``pearson_correlation`` with
    neither parameter compiles to the same HLO either way, 36 fusions in both, and still read
    1.25x at 262,144 elements. Within one process, arms strictly alternating, best of three
    per round over fifteen rounds, the same unchanged function timed twice reads 1.06x, 1.02x
    and 0.96x at 4,096 / 262,144 / 1,048,576 -- so roughly six per cent is this machine's
    floor. Against that, substituting both inputs of ``pearson_correlation`` costs nothing
    measurable at 4,096 (1.02x) and between nothing and about fifteen per cent at the two
    larger sizes: 1.10x and 1.15x on the fastest sample, 1.12x and 1.01x on the median.

    Threading ``where=`` into the reductions instead, as ``optax`` does for
    ``softmax_cross_entropy``, is rejected on correctness, which does not depend on the
    timing: measured, it returns the *right* value, 0.99527 on a fixture with a ``NaN`` at a
    zero-weighted slot, with a gradient of ``[nan, nan, nan, nan]``. The value passes through
    the arithmetic before the reduction skips it, so the derivative is already poisoned. It is
    also never the faster of the two -- 1.14x and 1.47x against the unguarded form at the same
    two sizes -- but that is not why it is not used here.

    Args:
        values: The array to neutralise.
        weights: The folded weight vector, zero wherever an element is excluded.
        mask: Elements to keep, or ``None``. Needed separately from ``weights``: a masked
            element goes whatever it holds, where a zero-weighted one keeps a finite value.

    Returns:
        ``values`` with the excluded elements set to zero.
    """
    v = jnp.asarray(values)
    # A zero weight alone is no reason to substitute: the element is an observation
    # carrying weight zero, and `d/dw_i` there is its contribution. Only a value that
    # would survive the multiply as a NaN has to go.
    drop = (weights == 0.0) & ~jnp.isfinite(v)
    if mask is not None:
        drop = drop | ~jnp.broadcast_to(jnp.asarray(mask, dtype=bool), v.shape)
    return jnp.where(drop, 0.0, v)


def exclude_inputs(
    mask: ArrayLike | None, weights: ArrayLike | None, *arrays: jax.Array
) -> tuple[jax.Array, ...]:
    """Each array with the elements ``mask`` and ``weights`` exclude replaced by zero.

    For the element-wise losses, whose ``mask`` and ``weights`` broadcast against the input
    itself and which need the folded vector for nothing else: ``mse``, ``mae``,
    ``huber_loss`` and ``charbonnier_loss``, with ``rmse`` through ``mse``. One line each,
    and a caller passing neither argument emits no extra work at all.

    Everything else calls :func:`substitute_excluded` directly, which is not duplication but
    the absence of it. A statistic that already folds a weight vector for its own formula --
    ``pearson_correlation``, ``concordance_correlation``, ``kendall_tau``,
    :func:`coefficient_of_determination` -- reuses that vector rather than folding a second
    one. The rest need a different shape entirely: ``softmax_cross_entropy`` folds at the
    label shape because its ``NaN`` sits one axis deeper, ``relative_l2_error`` at the sample
    axis because its mask is dimensioned against its output, ``rmsd`` from
    ``mask_a & mask_b`` rather than from a mask and weights at all, and ``perplexity`` from a
    vector it has already validated and cast.

    Substitution belongs here, on the raw inputs, rather than in :func:`reduce_values`: by
    the time a loss has formed its element-wise values the ``NaN`` has already reached the
    derivative, and ``0 * dvalues/dp`` is ``NaN`` when ``dvalues/dp`` is. Guarding the
    reduction repairs the number and leaves the gradient poisoned, which is worse than a
    plain ``NaN`` because a forward test then passes.

    ``arrays`` must already be :class:`jax.Array`, which every caller has from
    :func:`_prepare_arrays` one line earlier. Converting again here is not free on the eager
    path, where there is no compilation to absorb it: measured, re-wrapping them with
    ``jnp.asarray`` takes an eager ``mse`` over 100,000 elements to 585 Python-level calls
    against 455, a 28.6% increase in interpreter work for a compiled graph that is identical
    either way -- 7 fusions with the conversion and 7 without. Wall clock could not resolve it
    (6 microseconds on 92, inside a 7% noise floor) but CodSpeed's instruction-count
    simulation did, reporting a 11.8% regression on exactly that benchmark.

    Args:
        mask: Elements to keep, or ``None``.
        weights: Element weights, or ``None``.
        *arrays: The inputs to neutralise, already converted. The first fixes the shape the
            others broadcast against, which is the shape the mask and weights are folded at.

    Returns:
        The arrays, in order, returned untouched when neither ``mask`` nor ``weights`` is
        given.
    """
    if mask is None and weights is None:
        return arrays
    folded = sample_weights(mask, weights, arrays[0].shape)
    return tuple(substitute_excluded(a, folded, mask) for a in arrays)


def reduce_values(
    values: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    reduction: str = "mean",
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """Reduce element-wise loss values under an optional mask and weights.

    Every loss in the functional tier reduces through this one function, so a mask,
    weights, a reduction and an axis mean the same thing everywhere:

    - ``mask`` (boolean, broadcastable to ``values``) excludes elements: they contribute
      nothing to a sum and are not counted in a mean;
    - ``weights`` (broadcastable to ``values``) scale elements, and a mean is the weighted
      mean ``sum(w * x) / sum(w)`` over the unmasked elements;
    - ``reduction`` is ``"none"`` (element-wise, masked elements are 0), ``"mean"``,
      ``"sum"`` or ``"batch_sum"`` (sum over the non-batch axes, mean over the leading
      batch axis; ``axis`` is ignored);
    - ``axis`` restricts ``"mean"`` and ``"sum"`` to the given axes.

    A mean over no unmasked element (an all-false mask, or weights that sum to zero) is
    ``0.0``: one documented, finite result, so a caller that treats an empty selection as
    an error can check the mask at the host boundary instead of finding a NaN later.

    Args:
        values: Element-wise loss values.
        mask: Elements to keep, or ``None`` for all of them.
        weights: Element weights, or ``None`` for unit weights.
        reduction: One of ``"none"``, ``"mean"``, ``"sum"``, ``"batch_sum"``.
        axis: Axis or axes for ``"mean"`` and ``"sum"``.

    Returns:
        The reduced value(s).

    Raises:
        ValueError: If ``reduction`` is not one of the supported modes.
    """
    if reduction not in _REDUCTIONS:
        msg = f"Unknown reduction: {reduction!r}. Use one of {_REDUCTIONS}."
        raise ValueError(msg)
    values = jnp.asarray(values)
    if mask is None and weights is None:
        # The plain reductions: no scale array, no guarded division, so a loss without a
        # mask or weights costs what the bare formula costs.
        return _plain_reduction(values, reduction, axis)
    scale = jnp.ones_like(values) if weights is None else jnp.broadcast_to(weights, values.shape)
    if mask is not None:
        scale = jnp.where(jnp.asarray(mask), scale, jnp.zeros_like(scale))
    # No guard here: by the time a loss reaches this point its `values` are already
    # derived from the inputs, so neutralising them would repair the result and leave the
    # gradient NaN -- measured, `0 * dvalues/dp` is NaN when the derivative itself is. Each
    # caller substitutes its own inputs instead, which is the only placement that fixes
    # both. Every call site in this package does so.
    return _scaled_reduction(values * scale, scale, reduction, axis)


def _plain_reduction(
    values: jax.Array, reduction: str, axis: int | tuple[int, ...] | None
) -> jax.Array:
    """Reduce unscaled values."""
    if reduction == "none":
        return values
    if reduction == "sum":
        return jnp.sum(values, axis=axis)
    if reduction == "batch_sum":
        return _batch_sum(values)
    return jnp.mean(values, axis=axis)


def _scaled_reduction(
    scaled: jax.Array, scale: jax.Array, reduction: str, axis: int | tuple[int, ...] | None
) -> jax.Array:
    """Reduce masked or weighted values; a mean divides by the scale that survived the mask."""
    if reduction == "none":
        return scaled
    if reduction == "sum":
        return jnp.sum(scaled, axis=axis)
    if reduction == "batch_sum":
        return _batch_sum(scaled) if scaled.ndim > 1 else _mean_of_scaled(scaled, scale, axis=None)
    return _mean_of_scaled(scaled, scale, axis=axis)


def _batch_sum(values: jax.Array) -> jax.Array:
    """Sum over the non-batch axes, mean over the leading batch axis."""
    if values.ndim <= 1:
        return jnp.mean(values)
    batch = values.shape[0]
    return jnp.mean(jnp.sum(values.reshape(batch, -1), axis=-1))


def _mean_of_scaled(
    scaled: jax.Array, scale: jax.Array, *, axis: int | tuple[int, ...] | None
) -> jax.Array:
    """``sum(scaled) / sum(scale)`` with ``0.0`` where the scale sums to zero."""
    numerator = jnp.sum(scaled, axis=axis)
    denominator = jnp.sum(scale, axis=axis)
    empty = denominator == 0
    # The guarded operand goes inside the select too, so the gradient of the empty
    # branch is finite rather than the derivative of a division by zero.
    safe_denominator = jnp.where(empty, jnp.ones_like(denominator), denominator)
    return jnp.where(empty, jnp.zeros_like(numerator), numerator / safe_denominator)
