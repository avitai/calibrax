"""Seismic misfits for full-waveform inversion (FWI): data and model misfits on traces and models.

Pure functions on seismic gathers and velocity models, each a scalar that ``jax.grad`` and
``jax.jit`` go through, so an inversion loop can descend on it. Gathers are shaped
``(..., n_time)``: the last axis is time and every leading axis an independent trace (a receiver,
a shot). Velocity models are ``(H, W)`` images, or ``(H, W, C)``.

Registered with ``domain="seismic"``.

Includes 7 functions: seismic_l2_misfit, seismic_l2_misfit_normalized,
seismic_relative_model_misfit, seismic_cross_correlation_misfit,
seismic_velocity_ssim_misfit, seismic_wasserstein2_misfit, seismic_soft_dtw_misfit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike

from calibrax.metrics._utils import _EPSILON, _prepare_arrays, safe_divide, safe_norm
from calibrax.metrics.functional.image import ssim


def _traces(predictions: ArrayLike, targets: ArrayLike) -> tuple[jax.Array, jax.Array]:
    """Both gathers as float32 ``(n_traces, n_time)`` arrays of the same shape.

    Args:
        predictions: Predicted gather, shape ``(..., n_time)``.
        targets: Observed gather, the same shape.

    Returns:
        The two gathers, one trace per row.
    """
    pred, target = _prepare_arrays(predictions, targets, dtype=jnp.float32)
    return pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1])


def seismic_l2_misfit(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:  # noqa: DOC502  # raised by _prepare_arrays
    """L2 data misfit: the mean squared difference of predicted and observed gathers.

    The least-squares data term of FWI, ``mean((predictions - targets)**2)``.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        Symmetric. Differentiable everywhere.

    Args:
        predictions: Predicted (modelled) data, shape ``(..., n_time)``.
        targets: Observed data, the same shape.

    Returns:
        The mean squared difference, a scalar. 0.0 for identical inputs.

    Raises:
        ValueError: If the shapes differ.

    Examples:
        >>> import jax.numpy as jnp
        >>> d = jnp.sin(jnp.linspace(0, 10, 256))
        >>> float(seismic_l2_misfit(d, d))
        0.0
    """
    pred, target = _prepare_arrays(predictions, targets, dtype=jnp.float32)
    return jnp.mean(jnp.square(pred - target))


def seismic_l2_misfit_normalized(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:  # noqa: DOC502  # raised by _prepare_arrays
    """Normalized L2 data misfit: ``mean((p - t)^2) / mean(t^2)``.

    :func:`seismic_l2_misfit` over the observed data's energy, so misfits compare across shots of
    different amplitude.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        Not symmetric (normalized by the targets' energy).

    Args:
        predictions: Predicted (modelled) data, shape ``(..., n_time)``.
        targets: Observed data, the same shape.

    Returns:
        The normalized mean squared difference, a scalar. 0.0 for identical inputs.

    Raises:
        ValueError: If the shapes differ.

    Examples:
        >>> import jax.numpy as jnp
        >>> d = jnp.sin(jnp.linspace(0, 10, 256))
        >>> float(seismic_l2_misfit_normalized(d, d))
        0.0
    """
    pred, target = _prepare_arrays(predictions, targets, dtype=jnp.float32)
    return safe_divide(jnp.mean(jnp.square(pred - target)), jnp.mean(jnp.square(target)))


def seismic_relative_model_misfit(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:  # noqa: DOC502  # raised by _prepare_arrays
    """Relative model misfit: ``||m_pred - m_true||_2 / ||m_true||_2``.

    The velocity-model error an inversion result is scored by against the true model: one ratio of
    norms over the whole model, the global form of the ``relative_error`` regression metric.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        Not symmetric (normalized by the true model's norm), and not a true metric (the
        normalization breaks the triangle inequality).

    Args:
        predictions: Predicted model, any shape.
        targets: True model, the same shape.

    Returns:
        The relative L2 model error, a scalar. 0.0 for identical inputs.

    Raises:
        ValueError: If the shapes differ.

    Examples:
        >>> import jax.numpy as jnp
        >>> m = jnp.linspace(1500.0, 6000.0, 256)
        >>> float(seismic_relative_model_misfit(m, m))
        0.0
    """
    pred, target = _prepare_arrays(predictions, targets, dtype=jnp.float32)
    return safe_divide(safe_norm(pred - target), safe_norm(target))


def _ncc_misfit_trace(target: jax.Array, prediction: jax.Array) -> jax.Array:
    """``1 - max`` normalized (demeaned) cross-correlation over every linear lag, for one trace.

    Args:
        target: An observed trace.
        prediction: A predicted trace of the same length.

    Returns:
        The trace's misfit, in [0, 2].
    """
    tc = target - jnp.mean(target)
    pc = prediction - jnp.mean(prediction)
    denom = safe_norm(tc) * safe_norm(pc)
    corr = jnp.correlate(tc, pc, mode="full") / jnp.maximum(denom, _EPSILON)
    # A normalized cross-correlation lies in [-1, 1]; the clip removes rounding past the bounds.
    misfit = 1.0 - jnp.clip(jnp.max(corr), -1.0, 1.0)
    # Two identical traces correlate exactly 1 at lag 0, the maximum, so the misfit and its
    # gradient are exactly 0 there; the rounded quotient reads 1 - ulp on some traces. A flat
    # trace has no correlation structure, so against any other trace it scores the worst, 1.
    flat = denom < _EPSILON
    misfit = jnp.where(flat, 1.0, misfit)
    return jnp.where(jnp.all(target == prediction), 0.0, misfit)


def seismic_cross_correlation_misfit(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:  # noqa: DOC502  # raised by _prepare_arrays
    """Normalized cross-correlation misfit: ``1 - max`` NCC for each trace, then the mean.

    Each trace is demeaned, its linear cross-correlation with the predicted trace is divided by
    the two norms, and the maximum over every lag is taken. A pure time shift moves the
    correlation peak to the shift's lag, so the misfit stays near 0 where the L2 misfit grows.

    The correlation is linear, not circular: a shifted trace loses the samples that leave the
    window, so for a shift of ``s`` samples the maximum NCC is slightly below 1. Zero-pad the
    traces for an exactly shift-invariant score.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, 2].
        Symmetric. Differentiable: the maximum over lags is piecewise smooth, attained at one lag
        almost surely.

    Args:
        predictions: Predicted traces, shape ``(..., n_time)``.
        targets: Observed traces, the same shape.

    Returns:
        The mean ``1 - max NCC`` over traces, a scalar. 0.0 for identical inputs.

    Raises:
        ValueError: If the shapes differ.

    Examples:
        >>> import jax.numpy as jnp
        >>> d = jnp.sin(jnp.linspace(0, 10, 256))
        >>> float(seismic_cross_correlation_misfit(d, d))
        0.0
    """
    pred, target = _traces(predictions, targets)
    return jnp.mean(jax.vmap(_ncc_misfit_trace)(target, pred))


def seismic_velocity_ssim_misfit(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    max_val: float | jax.Array | None = None,
) -> jax.Array:
    """SSIM misfit on velocity models: ``1 - SSIM``, with the image SSIM.

    :func:`calibrax.metrics.functional.image.ssim` (Wang et al. 2004, Gaussian window). A velocity
    model's values are physical (m/s), so ``max_val``, the dynamic range in SSIM's stability
    constants, defaults to the targets' range rather than the image default of 1.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, 2] (SSIM in [-1, 1]; [0, 1] in practice).
        Symmetric only for a given ``max_val``: the default reads the targets' range, so
        swapping models whose ranges differ changes the value. Differentiable, as the
        image SSIM is.

    Args:
        predictions: Predicted velocity model, shape ``(H, W)`` or ``(H, W, C)``.
        targets: True velocity model, the same shape.
        max_val: The dynamic range; ``max(targets) - min(targets)`` by default.

    Returns:
        ``1 - SSIM``, a scalar. 0.0 for identical models.

    Examples:
        >>> import jax.numpy as jnp
        >>> m = jnp.linspace(1500.0, 6000.0, 48 * 48).reshape(48, 48)
        >>> float(seismic_velocity_ssim_misfit(m, m))
        0.0
    """
    pred = jnp.asarray(predictions, dtype=jnp.float32)
    target = jnp.asarray(targets, dtype=jnp.float32)
    if max_val is None:
        max_val = jnp.maximum(jnp.max(target) - jnp.min(target), _EPSILON)
    return 1.0 - ssim(pred, target, max_val=max_val)


#: The normalizations ``seismic_wasserstein2_misfit`` turns a signed trace into a density with.
_W2_NORMALIZATIONS = ("linear", "softplus")


def _softplus_difference(a: jax.Array, b: jax.Array) -> jax.Array:
    """``softplus(a) - softplus(b)``, computed from ``a - b`` rather than as a difference.

    ``(1 + e^a) / (1 + e^b) = 1 + sigmoid(b) expm1(a - b)``, so the difference is
    ``log1p(sigmoid(b) expm1(a - b))``: no cancellation when ``a`` and ``b`` are close. The branch
    taken keeps the ``expm1`` argument non-positive, so it never overflows, and the argument is
    clamped inside each branch so the one not taken has a finite gradient too.

    Args:
        a: First argument.
        b: Second argument, the same shape.

    Returns:
        ``softplus(a) - softplus(b)``, elementwise.
    """
    delta = a - b
    below = jnp.log1p(jax.nn.sigmoid(b) * jnp.expm1(jnp.minimum(delta, 0.0)))
    above = -jnp.log1p(jax.nn.sigmoid(a) * jnp.expm1(jnp.minimum(-delta, 0.0)))
    return jnp.where(delta <= 0.0, below, above)


def _w2_cell_masses(
    pred: jax.Array,
    target: jax.Array,
    scale: float | None,
    c: float | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Each trace's cell masses under Engquist & Yang's eq. 4.1, and their difference.

    ``(sigma(f) + c) / S`` with ``sigma`` the identity or ``softplus(scale .)``. The difference
    ``q - p`` is carried from ``sigma(t) - sigma(y)``, not taken between two masses: under the
    linear scaling ``c`` dominates every sample, both densities sit near uniform, and subtracting
    them would leave only rounding.

    Args:
        pred: Predicted traces, ``(n_traces, n_time)``.
        target: Observed traces, the same shape.
        scale: The softplus scale ``b``, or None for the linear scaling.
        c: The buffer, at least 0.

    Returns:
        The predicted masses ``p``, the observed masses ``q``, ``q - p``, and for each trace
        whether both are densities (non-negative with a positive total).
    """
    if scale is None:
        h_pred, h_target = pred + c, target + c
        rise = target - pred
    else:
        h_pred = jax.nn.softplus(scale * pred) + c
        h_target = jax.nn.softplus(scale * target) + c
        rise = _softplus_difference(scale * target, scale * pred)
    total_pred = jnp.sum(h_pred, axis=-1, keepdims=True)
    total_target = jnp.sum(h_target, axis=-1, keepdims=True)
    valid = (
        jnp.all(h_pred >= 0.0, axis=-1)
        & jnp.all(h_target >= 0.0, axis=-1)
        & (total_pred[:, 0] > 0.0)
        & (total_target[:, 0] > 0.0)
    )
    # A trace outside the domain divides by a stand-in (inner where); its misfit is NaN below.
    safe_pred = jnp.where(total_pred > 0.0, total_pred, 1.0)
    safe_target = jnp.where(total_target > 0.0, total_target, 1.0)
    p, q = h_pred / safe_pred, h_target / safe_target
    # q - p = (h_t - h_p) / S_t + p (S_p - S_t) / S_t, with S_p - S_t = -sum(h_t - h_p).
    gap = (rise - p * jnp.sum(rise, axis=-1, keepdims=True)) / safe_target
    return p, q, gap, valid


def _wasserstein2_squared_trace(p: jax.Array, q: jax.Array, gap: jax.Array, dt: float) -> jax.Array:
    """``int_0^1 (F^-1 - G^-1)^2 dz`` for one trace's cell masses, exactly.

    Each sample is the mass of its cell ``[i dt, (i + 1) dt]``, so the CDFs ``F`` and ``G`` are
    piecewise linear and so are their inverses: the inverse CDF interpolated linearly between
    samples (Yang et al., Geophysics 2018, p. R49). Between consecutive knots of the two CDFs
    merged (the structure of POT's ``ot.lp.wasserstein_1d``), both quantile functions are linear,
    so the squared gap is a quadratic and ``dz (d0^2 + d0 d1 + d1^2) / 3`` integrates it exactly.

    Each interval's two segments are the ones after the last knot of each CDF in the merged
    order; a zero-mass cell, a jump of the inverse, only ever bounds an interval of zero width.
    A quantile ``e_k + (z - F_k) dt / p_k`` needs ``z - F_k``, which
    as a difference of two CDF values near 1 over a mass near ``1 / n`` would lose all float32
    digits; it is instead a difference of one CDF over the few cells between the two segments,
    plus the carried CDF gap ``D_k = G_k - F_k``.

    Args:
        p: Predicted cell masses, summing to 1.
        q: Observed cell masses, summing to 1.
        gap: ``q - p``, computed without cancellation.
        dt: The sample spacing.

    Returns:
        ``W2^2`` in the units of ``dt`` squared.
    """
    n = p.shape[0]
    zero = jnp.zeros((1,), dtype=p.dtype)
    cdf_p = jnp.concatenate([zero, jnp.cumsum(p)])
    cdf_q = jnp.concatenate([zero, jnp.cumsum(q)])
    cdf_gap = jnp.concatenate([zero, jnp.cumsum(gap)])  # G_k - F_k
    knots = jnp.concatenate([cdf_p, cdf_q])
    from_p = jnp.arange(2 * (n + 1)) < n + 1
    index = jnp.concatenate([jnp.arange(n + 1), jnp.arange(n + 1)])
    order = jnp.argsort(knots, stable=True)
    from_p, index = from_p[order], index[order]
    # The interval after merged knot j lies in the segment of F after the last F knot so far,
    # and likewise for G: a running count, read off the order without comparing floats.
    k = jnp.clip(jnp.cumsum(from_p)[:-1] - 1, 0, n - 1)
    m = jnp.clip(jnp.cumsum(~from_p)[:-1] - 1, 0, n - 1)

    def offsets(i: jax.Array, own: jax.Array) -> tuple[jax.Array, jax.Array]:
        """``z - F_k`` and ``z - G_m`` at the knots ``i``, ``own`` marking knots of ``F``."""
        to_p = jnp.where(own, cdf_p[i] - cdf_p[k], (cdf_q[i] - cdf_q[k]) + cdf_gap[k])
        to_q = jnp.where(own, (cdf_p[i] - cdf_p[m]) - cdf_gap[m], cdf_q[i] - cdf_q[m])
        return to_p, to_q

    lo_p, lo_q = offsets(index[:-1], from_p[:-1])
    hi_p, hi_q = offsets(index[1:], from_p[1:])
    # Each segment's mass is its CDF's own step, so a whole step over its mass is exactly 1 and
    # equal CDFs give an exactly-zero gap, value and gradient. A zero-mass segment bounds only
    # zero-width intervals; the stand-in keeps them finite.
    step_p = cdf_p[k + 1] - cdf_p[k]
    step_q = cdf_q[m + 1] - cdf_q[m]
    mass_p = jnp.where(step_p > 0.0, step_p, 1.0)
    mass_q = jnp.where(step_q > 0.0, step_q, 1.0)
    cells = (k - m).astype(p.dtype)
    d0 = (cells + lo_p / mass_p - lo_q / mass_q) * dt
    d1 = (cells + hi_p / mass_p - hi_q / mass_q) * dt
    return jnp.sum((hi_p - lo_p) * (d0 * d0 + d0 * d1 + d1 * d1)) / 3.0


def seismic_wasserstein2_misfit(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    normalization: str = "linear",
    b: float | None = None,
    c: float | None = None,
    dt: float = 1.0,
) -> jax.Array:
    """Squared quadratic Wasserstein misfit for each trace, then the mean over traces.

    For each trace, the 1-D ``W2^2(f, g) = int_0^1 |F^-1(z) - G^-1(z)|^2 dz`` (Engquist, Froese
    & Yang, Commun. Math. Sci. 2016, eq. 13; Yang et al., Geophysics 2018, eqs. 9 and 28-29)
    between the predicted and observed traces turned into densities. A pure time shift by ``s``
    scores ``s^2``, so unlike L2 the misfit keeps growing once two wavelets no longer overlap:
    the cycle-skipping defense W2 is used for in full-waveform inversion.

    **Discretization.** Each sample is the mass of its cell, so the CDFs are piecewise linear and
    their inverses interpolate linearly between samples (Yang et al. 2018, p. R49); the integral
    of the squared quantile gap is then exact. It converges at second order in ``dt``, sub-sample
    shifts included, where W2 between the samples as point masses does not converge below one
    sample. The gradient is JAX's own derivative of that exact integral (no custom rule), so
    ``jax.grad``, ``jax.jvp``, ``jax.jacfwd``, ``jax.vmap`` and ``jax.jit`` all apply; in the
    continuum the first variation of ``W2^2`` is the Kantorovich potential (Santambrogio 2015,
    Prop. 7.17).

    **Normalization** (Engquist & Yang, Commun. Pure Appl. Math. 2022, eq. 4.1):
    ``(sigma(f) + c) / S``, with ``S`` each trace's total.

    - ``"linear"`` (default): ``sigma(f) = f``, eq. 1.2. ``c`` defaults to one constant, the l1
      norm of all the targets, as the authors set it for their Marmousi inversions, and is held
      fixed (no gradient), as "fixed in inversion" requires (Yang et al. 2018, p. R48). It keeps
      polarity, since the scaling is one-to-one, but is not strictly convex in a shift (Yang et
      al. 2018, Fig. 3a). A prediction below ``-c`` has no density and scores NaN, as do all-zero
      targets, which leave ``c = 0``.
    - ``"softplus"``: ``sigma(f) = log(1 + e^(b f))``, eq. 1.4, convex in a shift for a proper
      ``b`` (Corollary 4.4); the authors choose ``b`` with ``0.2 <= ||b f||_inf <= 6``. ``b`` has no
      default. ``c`` defaults to 0, since softplus is already positive. It suppresses the negative
      half of a trace; the authors note that adding the same misfit on ``-f`` avoids that bias.

    Squared, magnitude-only densities (``|f|``, ``f^2``, envelopes) are not offered: the same
    authors reject them for FWI as not one-to-one (Yang et al. 2018, p. R48; Engquist & Yang 2022,
    Section 4). The *squared* W2 is returned, smooth at a perfect match. The mean over traces is
    calibrax's reduction; Yang et al. 2018 eq. 28 sums.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf), in units of ``dt`` squared.
        Not symmetric under the linear default, whose ``c`` comes from the targets alone.
        Differentiable. Each trace sorts ``2 n_time + 2`` CDF knots.

    Args:
        predictions: Predicted traces, shape ``(..., n_time)``.
        targets: Observed traces, the same shape.
        normalization: ``"linear"`` or ``"softplus"``. Static under ``jax.jit``.
        b: The softplus scale, positive; required for ``"softplus"``, refused otherwise.
        c: The buffer, at least 0. None means the l1 norm of the targets for ``"linear"`` and 0
            for ``"softplus"``.
        dt: The sample spacing, positive; the misfit is in its units squared.

    Returns:
        The mean ``W2^2`` over traces, a scalar. 0.0 for identical inputs.

    Raises:
        ValueError: If the shapes differ, ``normalization`` is unknown, ``b`` is missing, not
            positive or given for ``"linear"``, ``c`` is negative, or ``dt`` is not positive.

    Examples:
        >>> import jax.numpy as jnp
        >>> d = jnp.sin(jnp.linspace(0, 10, 256))
        >>> float(seismic_wasserstein2_misfit(d, d))
        0.0
    """
    if normalization not in _W2_NORMALIZATIONS:
        msg = f"normalization must be one of {_W2_NORMALIZATIONS}, got {normalization!r}"
        raise ValueError(msg)
    if normalization == "softplus" and (b is None or not b > 0):
        msg = f"the softplus normalization needs a positive scale b, got {b!r}"
        raise ValueError(msg)
    if normalization == "linear" and b is not None:
        msg = f"b scales the softplus normalization only; the linear one takes c, got b={b!r}"
        raise ValueError(msg)
    if c is not None and not c >= 0:
        msg = f"the buffer c must be at least 0, got {c!r}"
        raise ValueError(msg)
    if not dt > 0:
        msg = f"the sample spacing dt must be positive, got {dt!r}"
        raise ValueError(msg)
    pred, target = _traces(predictions, targets)
    if c is None:
        buffer = lax.stop_gradient(jnp.sum(jnp.abs(target))) if normalization == "linear" else 0.0
    else:
        buffer = c
    scale = b if normalization == "softplus" else None
    p, q, gap, valid = _w2_cell_masses(pred, target, scale, buffer)
    per_trace = jax.vmap(_wasserstein2_squared_trace, in_axes=(0, 0, 0, None))(p, q, gap, dt)
    return jnp.mean(jnp.where(valid, per_trace, jnp.nan))


def _soft_dtw(x: jax.Array, y: jax.Array, gamma: float) -> jax.Array:
    """Soft-DTW of two traces (Cuturi & Blondel, ICML 2017), sweeping anti-diagonals.

    Cell ``R[i, j]`` depends on ``R[i-1, j]``, ``R[i, j-1]`` and ``R[i-1, j-1]``, all on the two
    previous anti-diagonals, so one diagonal is computed at once: ``n + m - 1`` vectorized steps
    carrying two diagonals, where a cell-by-cell recursion takes ``n m`` steps and reverse mode
    stores the whole table at each. This is the schedule GPU implementations use (Maghoumi et
    al., "DeepNAG", 2021). Diagonal ``k`` holds ``R[i, k - i]`` for ``i`` in ``0..n``.

    Args:
        x: A trace, length ``n``.
        y: A trace, length ``m``.
        gamma: The softmin temperature, positive.

    Returns:
        ``R[n, m]``, the soft-DTW cost.
    """
    n, m = x.shape[0], y.shape[0]
    cost = jnp.square(x[:, None] - y[None, :])
    rows = jnp.arange(n + 1)
    inf = jnp.asarray(jnp.inf, dtype=cost.dtype)
    first = jnp.where(rows == 0, 0.0, inf)  # R[0, 0] = 0 anchors the recursion
    second = jnp.full((n + 1,), inf)
    pad = jnp.full((1,), inf)

    def diagonal(
        carry: tuple[jax.Array, jax.Array], k: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        before, last = carry
        cols = k - rows
        inside = (rows >= 1) & (cols >= 1) & (cols <= m)
        up = jnp.concatenate([pad, last[:-1]])  # R[i-1, j]
        across = jnp.concatenate([pad, before[:-1]])  # R[i-1, j-1]
        # A cell outside the table reads a finite stand-in, so its softmin's gradient is finite.
        neighbours = jnp.where(inside, jnp.stack([up, last, across]), 0.0)
        softmin = -gamma * jax.scipy.special.logsumexp(-neighbours / gamma, axis=0)
        step = cost[jnp.clip(rows - 1, 0, n - 1), jnp.clip(cols - 1, 0, m - 1)]
        return (last, jnp.where(inside, step + softmin, inf)), None

    (_, final), _ = lax.scan(diagonal, (first, second), jnp.arange(2, n + m + 1))
    return final[n]


def seismic_soft_dtw_misfit(  # noqa: DOC502  # the shape refusal is raised by _prepare_arrays
    predictions: ArrayLike, targets: ArrayLike, *, gamma: float = 0.01
) -> jax.Array:
    """Soft-DTW divergence for each trace, then the mean: dynamic time warping, differentiable.

    The hard ``min`` of DTW's Bellman recursion becomes a soft minimum at temperature ``gamma``
    (Cuturi & Blondel, ICML 2017), so the alignment is differentiable throughout. A stretched or
    shifted trace aligns along a warping path off the diagonal and costs little where the L2
    misfit is large.

    The soft minimum sits below the hard one, so raw soft-DTW carries a ``gamma``-dependent bias
    along the path. This returns the soft-DTW divergence ``SDTW(p, t) - (SDTW(p, p) + SDTW(t,
    t)) / 2`` (Blondel, Mensch & Vert, AISTATS 2021), non-negative and exactly 0 for identical
    inputs; a final clip at 0 removes rounding below it.

    On misaligned traces the divergence's residual grows about linearly with ``gamma``, so
    ``gamma`` should be small against the typical squared per-sample cost: the default 0.01 suits
    unit-scale traces, and ``gamma ~ 0.01 * mean(t**2)`` others.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        Symmetric. Differentiable. Each trace costs three soft-DTW sweeps of ``n_time^2`` cells
        in ``2 n_time - 1`` steps.

    Args:
        predictions: Predicted traces, shape ``(..., n_time)``.
        targets: Observed traces, the same shape.
        gamma: The softmin temperature, positive; smaller is closer to hard DTW. Static under
            ``jax.jit``.

    Returns:
        The mean soft-DTW divergence over traces, a scalar. 0.0 for identical inputs.

    Raises:
        ValueError: If ``gamma`` is not positive.

    Examples:
        >>> import jax.numpy as jnp
        >>> d = jnp.sin(jnp.linspace(0, 10, 64))
        >>> float(seismic_soft_dtw_misfit(d, d))
        0.0
    """
    if not gamma > 0:
        msg = f"seismic_soft_dtw_misfit needs a positive temperature gamma, got {gamma!r}"
        raise ValueError(msg)
    pred, target = _traces(predictions, targets)

    def divergence(p: jax.Array, t: jax.Array) -> jax.Array:
        own = 0.5 * (_soft_dtw(p, p, gamma) + _soft_dtw(t, t, gamma))
        return jnp.maximum(_soft_dtw(p, t, gamma) - own, 0.0)

    return jnp.mean(jax.vmap(divergence)(pred, target))
