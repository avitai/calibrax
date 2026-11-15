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


def _envelope_density(trace: jax.Array) -> jax.Array:
    """A trace's normalized positive envelope ``|t| / sum|t|``, uniform for a silent trace.

    Args:
        trace: One trace.

    Returns:
        A probability density over the trace's samples.
    """
    env = jnp.abs(trace)
    total = jnp.sum(env)
    silent = total <= _EPSILON
    # A silent trace divides by a stand-in (inner where), so its gradient stays finite.
    density = env / jnp.where(silent, 1.0, total)
    return jnp.where(silent, jnp.full_like(env, 1.0 / env.shape[0]), density)


def _wasserstein2_squared_trace(target: jax.Array, prediction: jax.Array) -> jax.Array:
    """The exact 1-D ``W2^2`` of two traces' envelope densities, by the quantile formula.

    Args:
        target: An observed trace.
        prediction: A predicted trace of the same length.

    Returns:
        The mean squared gap of the sorted densities.
    """
    u = jnp.sort(_envelope_density(target))
    v = jnp.sort(_envelope_density(prediction))
    return jnp.mean(jnp.square(u - v))


def seismic_wasserstein2_misfit(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:  # noqa: DOC502  # raised by _prepare_arrays
    """Squared Wasserstein-2 misfit for each trace on normalized positive envelopes, then the mean.

    Each trace becomes a probability density through its normalized positive envelope (Metivier
    et al., Geophys. J. Int. 2016), and the 1-D ``W2^2`` is the mean squared difference of the
    sorted quantile functions, exactly. ``jnp.sort`` carries gradients through the permutation.

    Positive envelopes discard polarity: two traces whose envelopes agree but whose signs differ
    are at distance 0. Keeping polarity needs transport between signed measures.

    The *squared* W2 is returned: it is smooth at a perfect match, where W2 itself, like any
    norm, has a singular gradient.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        Symmetric. Differentiable.

    Args:
        predictions: Predicted traces, shape ``(..., n_time)``.
        targets: Observed traces, the same shape.

    Returns:
        The mean squared W2 over traces, a scalar. 0.0 for identical inputs.

    Raises:
        ValueError: If the shapes differ.

    Examples:
        >>> import jax.numpy as jnp
        >>> d = jnp.sin(jnp.linspace(0, 10, 256)) ** 2 + 0.1
        >>> float(seismic_wasserstein2_misfit(d, d))
        0.0
    """
    pred, target = _traces(predictions, targets)
    return jnp.mean(jax.vmap(_wasserstein2_squared_trace)(target, pred))


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
