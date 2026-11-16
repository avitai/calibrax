"""Seismic (full-waveform inversion) misfits against float64 references of each quantity.

Each metric is compared, eagerly and under ``jax.jit``, with the quantity it defines computed in
float64 NumPy from the same float32 inputs. Every tolerance is a rounding bound for the
computation, stated where it is used: a recursive sum of ``n`` terms is within ``(n - 1) eps`` of
its value relative to the sum of the terms' magnitudes (Higham, *Accuracy and Stability of
Numerical Algorithms*, 2nd ed., 2002, eq. 4.4), plus one ``eps`` for each further rounding.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike

import calibrax.metrics.functional.seismic as seismic_module
from calibrax.metrics import MetricRegistry, MetricSignature, MetricTier
from calibrax.metrics.functional.image import ssim
from calibrax.metrics.functional.seismic import (
    seismic_cross_correlation_misfit,
    seismic_l2_misfit,
    seismic_l2_misfit_normalized,
    seismic_relative_model_misfit,
    seismic_soft_dtw_misfit,
    seismic_velocity_ssim_misfit,
    seismic_wasserstein2_misfit,
)


EPS = float(np.finfo(np.float32).eps)
GUARD = 1e-8  # calibrax's guarded-division epsilon, which every relative metric adds
TRACE_METRICS = [
    "seismic_l2_misfit",
    "seismic_l2_misfit_normalized",
    "seismic_relative_model_misfit",
    "seismic_cross_correlation_misfit",
    "seismic_wasserstein2_misfit",
    "seismic_soft_dtw_misfit",
]
ALL_METRICS = [*TRACE_METRICS, "seismic_velocity_ssim_misfit"]
Metric = Callable[..., jax.Array]
FUNCTIONS: dict[str, Metric] = {
    "seismic_l2_misfit": seismic_l2_misfit,
    "seismic_l2_misfit_normalized": seismic_l2_misfit_normalized,
    "seismic_relative_model_misfit": seismic_relative_model_misfit,
    "seismic_cross_correlation_misfit": seismic_cross_correlation_misfit,
    "seismic_wasserstein2_misfit": seismic_wasserstein2_misfit,
    "seismic_soft_dtw_misfit": seismic_soft_dtw_misfit,
    "seismic_velocity_ssim_misfit": seismic_velocity_ssim_misfit,
}


def _ricker(n: int = 128, peak: int = 48, width: float = 6.0) -> np.ndarray:
    """A Ricker-like wavelet, tapered so a shift loses little energy."""
    t = np.arange(n, dtype=np.float64) - peak
    return ((1.0 - (t / width) ** 2) * np.exp(-0.5 * (t / width) ** 2)).astype(np.float32)


def _shifted(d: np.ndarray, shift: int = 10) -> np.ndarray:
    """A zero-padded time shift, the recording geometry (no wrap-around)."""
    return np.concatenate([np.zeros(shift, dtype=np.float32), d[:-shift]])


def _velocity_model(n: int = 48) -> np.ndarray:
    """A smooth layered velocity model in m/s, constant along the second axis."""
    z = np.linspace(0.0, 1.0, n)
    column = 1500.0 + 3000.0 * z**1.5 + 200.0 * np.sin(6.0 * np.pi * z)
    return (column[:, None] * np.ones(n)).astype(np.float32)


def _noise(seed: int, n: int = 128) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=n).astype(np.float32)


# --- float64 references of each quantity, with the rounding bound of its float32 computation ---


def _l2(p: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """``mean((p - t)^2)``: a sum of ``n`` non-negative terms, each from two roundings."""
    terms = (p - t) ** 2
    return float(np.mean(terms)), (terms.size + 3) * EPS * float(np.mean(terms))


def _l2_normalized(p: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """A ratio of two such sums (each ``n + 3`` roundings), plus the guard's bias."""
    num, den = float(np.mean((p - t) ** 2)), float(np.mean(t**2))
    value = num / den
    return value, (2 * (p.size + 3) * EPS + GUARD / den) * value


def _relative(p: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """``||p - t|| / ||t||``: two norms of ``n``-term sums and a division, plus the guard."""
    num, den = float(np.linalg.norm(p - t)), float(np.linalg.norm(t))
    value = num / den
    return value, (2 * (p.size + 3) * EPS + GUARD / den) * value


def _cc_trace(p: np.ndarray, t: np.ndarray) -> float:
    if np.array_equal(p, t):
        return 0.0
    tc, pc = t - t.mean(), p - p.mean()
    return float(
        1.0 - np.max(np.correlate(tc, pc, mode="full")) / (np.linalg.norm(tc) * np.linalg.norm(pc))
    )


def _cross_correlation(p: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """``1 - max NCC``: each lag is a dot product of at most ``n`` terms over the two norms.

    By Cauchy-Schwarz a dot product's rounding error over ``||x|| ||y||`` is within ``n eps``,
    and the demeaning and the two norms add a few more roundings: ``(n + 6) eps`` absolute.
    """
    p2, t2 = p.reshape(-1, p.shape[-1]), t.reshape(-1, t.shape[-1])
    value = float(np.mean([_cc_trace(a, b) for a, b in zip(p2, t2, strict=True)]))
    return value, (p.shape[-1] + 6) * EPS


def _w2_cell_masses(x: np.ndarray, normalization: str, b: float | None, c: float) -> np.ndarray:
    """Engquist & Yang (CPAM 2022) eq. 4.1: ``(sigma(x) + c) / S`` for one trace."""
    h = x + c if normalization == "linear" else np.logaddexp(0.0, b * x) + c
    return h / h.sum()


def _w2_transport_form(f: np.ndarray, g: np.ndarray, dt: float) -> float:
    """``int (t - G^-1(F(t)))^2 f(t) dt`` (Yang et al., Geophysics 2018, eq. 29), exactly.

    Each sample is the mass of its cell ``[i dt, (i + 1) dt]``, so ``F`` and ``G`` are piecewise
    linear. Between the cell edges and the times where ``F`` reaches a knot of ``G``, both ``F``
    and ``G^-1`` are linear, so ``t - T(t)`` is linear and the integral of its square against the
    cell's constant density is exact. The form integrates over time, where the implementation
    integrates the quantile gap over probability, so the two agree only if both are right.
    """
    edges = np.arange(f.size + 1) * dt
    cdf_f = np.concatenate([[0.0], np.cumsum(f)])
    cdf_g = np.concatenate([[0.0], np.cumsum(g)])
    knots = np.unique(np.concatenate([edges, np.interp(cdf_g, cdf_f, edges)]))
    total = 0.0
    for lo, hi in itertools.pairwise(knots):
        cell = min(int((lo + hi) / 2 // dt), f.size - 1)
        if f[cell] == 0.0:
            continue  # no mass to transport
        # F is linear on the cell; G^-1 is linear on the segment of G holding F(mid), whose mass
        # is positive. Both are taken from the piece itself, never from a neighbour across a jump.
        z_lo, z_mid, z_hi = (
            cdf_f[cell] + (x - edges[cell]) * f[cell] / dt for x in (lo, (lo + hi) / 2, hi)
        )
        seg = min(max(int(np.searchsorted(cdf_g, z_mid, side="right")) - 1, 0), g.size - 1)
        gap_lo = lo - (edges[seg] + (z_lo - cdf_g[seg]) * dt / g[seg])
        gap_hi = hi - (edges[seg] + (z_hi - cdf_g[seg]) * dt / g[seg])
        total += (f[cell] / dt) * (hi - lo) * (gap_lo**2 + gap_lo * gap_hi + gap_hi**2) / 3.0
    return float(total)


def _wasserstein2(
    p: np.ndarray,
    t: np.ndarray,
    normalization: str = "linear",
    b: float | None = None,
    c: float | None = None,
    dt: float = 1.0,
) -> tuple[float, float]:
    """The mean over traces of ``W2^2`` of the normalized cell densities, and its float32 bound.

    ``c`` defaults as the implementation's does: for ``"linear"`` one constant, the l1 norm of all
    the targets; for ``"softplus"`` 0.

    Rounding bound for the implementation, which carries the CDF gap ``D_k = G_k - F_k`` as a
    cumulative sum of ``q - p`` rather than subtracting two CDFs near 1. A quantile offset
    ``z - F_k`` is then either a difference of one CDF over ``s + 1`` of its own steps (``s`` the
    largest displacement in cells between the two quantile functions), each step rounding by at
    most ``eps`` of a CDF value at most 1 (Higham eq. 4.4), or such a difference plus ``D_k``, a sum
    of ``n`` terms of ``q - p`` within ``(n + 4) eps sum |q - p|``. Divided by the cell's mass, a
    quantile is within ``dQ = ((s + 4) + (n + 4) sum |q - p|) eps dt / p_min``, ``p_min`` the least
    positive mass. Both quantiles enter the gap, which moves ``W2^2`` by at most
    ``2 sqrt(W2^2) (2 dQ) + (2 dQ)^2``, and the sum over ``2 n + 1`` intervals adds ``4 n eps``
    relative.
    """
    p2, t2 = p.reshape(-1, p.shape[-1]), t.reshape(-1, t.shape[-1])
    if c is None:
        c = float(np.abs(t2).sum()) if normalization == "linear" else 0.0
    values, bounds = [], []
    for a, d in zip(p2, t2, strict=True):
        f = _w2_cell_masses(a, normalization, b, c)
        g = _w2_cell_masses(d, normalization, b, c)
        value = _w2_transport_form(f, g, dt)
        n = a.size
        edges = np.arange(n + 1) * dt
        quantile_f = np.interp(np.cumsum(g), np.cumsum(f), edges[1:])
        displacement = int(np.ceil(np.max(np.abs(quantile_f - edges[1:])) / dt)) + 1
        gap_mass = float(np.abs(g - f).sum())
        p_min = min(f[f > 0].min(), g[g > 0].min())
        d_q = ((displacement + 4) + (n + 4) * gap_mass) * EPS * dt / p_min
        values.append(value)
        bounds.append(2 * np.sqrt(value) * 2 * d_q + (2 * d_q) ** 2 + 4 * n * EPS * value)
    return float(np.mean(values)), float(np.mean(bounds))


def _soft_dtw(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Cuturi & Blondel (ICML 2017), Algorithm 1, in float64."""
    n, m = len(x), len(y)
    cost = (x[:, None] - y[None, :]) ** 2
    table = np.full((n + 1, m + 1), np.inf)
    table[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            v = -np.array([table[i - 1, j], table[i, j - 1], table[i - 1, j - 1]]) / gamma
            top = np.max(v)
            table[i, j] = cost[i - 1, j - 1] - gamma * (top + np.log(np.sum(np.exp(v - top))))
    return float(table[n, m])


def _soft_dtw_divergence(p: np.ndarray, t: np.ndarray, gamma: float = 0.01) -> tuple[float, float]:
    """``SDTW(p, t) - (SDTW(p, p) + SDTW(t, t)) / 2`` (Blondel, Mensch & Vert, AISTATS 2021).

    Each soft-DTW value accumulates ``n + m`` rounded additions along its path, so it is within
    ``(n + m) eps`` of itself relative to its magnitude; the divergence subtracts three of them.
    """
    p2, t2 = (
        p.astype(np.float64).reshape(-1, p.shape[-1]),
        t.astype(np.float64).reshape(-1, t.shape[-1]),
    )
    values, bounds = [], []
    for a, b in zip(p2, t2, strict=True):
        cross, own_a, own_b = _soft_dtw(a, b, gamma), _soft_dtw(a, a, gamma), _soft_dtw(b, b, gamma)
        values.append(max(cross - 0.5 * (own_a + own_b), 0.0))
        bounds.append((2 * a.size) * EPS * (abs(cross) + 0.5 * (abs(own_a) + abs(own_b))))
    return float(np.mean(values)), float(np.mean(bounds))


REFERENCES: dict[str, Callable[[np.ndarray, np.ndarray], tuple[float, float]]] = {
    "seismic_l2_misfit": _l2,
    "seismic_l2_misfit_normalized": _l2_normalized,
    "seismic_relative_model_misfit": _relative,
    "seismic_cross_correlation_misfit": _cross_correlation,
    "seismic_wasserstein2_misfit": _wasserstein2,
    "seismic_soft_dtw_misfit": _soft_dtw_divergence,
}

_D = _ricker()
CASES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "shifted": (_shifted(_D), _D),
    "perturbed": (_D + 0.05 * np.sin(np.linspace(0, 20, _D.size)).astype(np.float32), _D),
    "noise": (_noise(1, 64), _noise(2, 64)),
    "gather": (np.stack([_D, 2.0 * _D, np.roll(_D, 5)]), np.stack([_D, _D, _D])),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("name", TRACE_METRICS)
def test_matches_the_float64_reference_eagerly_and_under_jit(name: str, case: str) -> None:
    predictions, targets = CASES[case]
    if name == "seismic_soft_dtw_misfit" and case == "gather":
        predictions, targets = predictions[:, :64], targets[:, :64]  # the oracle is O(n^2) Python
    expected, bound = REFERENCES[name](predictions.astype(np.float64), targets.astype(np.float64))
    fn = FUNCTIONS[name]
    for got in (fn(predictions, targets), jax.jit(fn)(predictions, targets)):
        assert got.shape == ()
        assert abs(float(got) - expected) <= bound, (name, case, float(got), expected, bound)


#: Misfits whose gradient at a perfect match is a product with an exactly-zero factor: ``p - t``,
#: the quantile gap ``Q_p - Q_t``, or the identical-trace branch of the correlation.
EXACT_ZERO_GRADIENT = [
    "seismic_l2_misfit",
    "seismic_l2_misfit_normalized",
    "seismic_relative_model_misfit",
    "seismic_cross_correlation_misfit",
    "seismic_wasserstein2_misfit",
]


@pytest.mark.parametrize("name", ALL_METRICS)
def test_identical_inputs_are_exactly_zero_with_a_finite_gradient(name: str) -> None:
    """Each docstring's "0.0 for identical inputs", with no rounding residue in the value.

    Both arguments are traced, as a loss's are: a target closed over as a constant is folded at
    compile time and rounds apart from the prediction's runtime path. The gradient is finite
    everywhere, calibrax's contract at identity (``test_gradients_at_identity.py``), and exactly 0
    where a factor of it is exactly 0. Soft-DTW's and SSIM's identity gradients are sums of
    nonzero terms that cancel, so a rounding residue (about 5e-9 against gradients of order 0.2)
    remains there.
    """
    x = jnp.asarray(_velocity_model() if name == "seismic_velocity_ssim_misfit" else _D)
    fn = FUNCTIONS[name]
    assert float(fn(x, x)) == 0.0
    gradient = np.asarray(jax.jit(jax.grad(fn))(x, x))
    assert np.all(np.isfinite(gradient))
    if name in EXACT_ZERO_GRADIENT:
        assert np.max(np.abs(gradient)) == 0.0


@pytest.mark.parametrize("name", ALL_METRICS)
def test_gradients_are_finite_and_nonzero_off_a_match(name: str) -> None:
    if name == "seismic_velocity_ssim_misfit":
        target = jnp.asarray(_velocity_model())
        prediction = target + 50.0
    else:
        target = jnp.asarray(_D)
        prediction = target + 0.05 * jnp.sin(jnp.linspace(0, 20, _D.size))
    gradient = jax.jit(jax.grad(FUNCTIONS[name]))(prediction, target)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_a_constant_trace_against_another_scores_the_worst_correlation() -> None:
    """A flat trace has no correlation structure: two equal ones match, any other pair scores 1."""
    flat = jnp.ones(64)
    assert float(seismic_cross_correlation_misfit(flat, flat)) == 0.0
    assert float(seismic_cross_correlation_misfit(2.0 * flat, flat)) == 1.0


def _bump(n: int = 128, start: int = 40, width: int = 32) -> np.ndarray:
    """A raised-cosine pulse: positive inside ``(start, start + width)``, exactly 0 outside."""
    i = np.arange(n, dtype=np.float64) - start
    inside = (i > 0) & (i < width)
    return np.where(inside, 1.0 - np.cos(2.0 * np.pi * i / width), 0.0).astype(np.float32)


class W2Options(TypedDict, total=False):
    """Keyword settings of ``seismic_wasserstein2_misfit`` and its reference."""

    normalization: str
    b: float
    c: float
    dt: float


SOFTPLUS: W2Options = {"normalization": "softplus", "b": 2.0}  # ||b f||_inf = 2 at a unit peak


@pytest.mark.parametrize("dt", [1.0, 0.004])
@pytest.mark.parametrize("shift", [1, 10, 40])
def test_a_shifted_non_negative_trace_scores_the_squared_shift(shift: int, dt: float) -> None:
    """``W2^2`` of a density and its translate by ``s`` is ``s^2`` (Engquist & Froese 2014).

    With ``c = 0`` a non-negative trace is its own density, so a shift by whole cells is exactly a
    translate. The float64 reference meets ``(s dt)^2`` within its own rounding bound, the float32
    bound scaled to float64's ``eps``, and the implementation meets the reference within its own.
    """
    d = _bump()
    shifted = _shifted(d, shift)
    expected, bound = _wasserstein2(shifted.astype(np.float64), d.astype(np.float64), c=0.0, dt=dt)
    assert abs(expected - (shift * dt) ** 2) <= bound * float(np.finfo(np.float64).eps) / EPS
    for got in (
        seismic_wasserstein2_misfit(shifted, d, c=0.0, dt=dt),
        jax.jit(lambda p, t: seismic_wasserstein2_misfit(p, t, c=0.0, dt=dt))(shifted, d),
    ):
        assert abs(float(got) - expected) <= bound, (shift, dt, float(got), expected, bound)


@pytest.mark.parametrize("case", sorted(CASES))
def test_softplus_matches_the_float64_reference(case: str) -> None:
    predictions, targets = CASES[case]
    expected, bound = _wasserstein2(
        predictions.astype(np.float64), targets.astype(np.float64), **SOFTPLUS
    )

    def fn(p: ArrayLike, t: ArrayLike) -> jax.Array:
        return seismic_wasserstein2_misfit(p, t, **SOFTPLUS)

    for got in (fn(predictions, targets), jax.jit(fn)(predictions, targets)):
        assert abs(float(got) - expected) <= bound, (case, float(got), expected, bound)


def test_softplus_w2_grows_with_the_shift_where_l2_does_not() -> None:
    """The cycle-skipping property W2 is used for (Engquist & Yang, CPAM 2022, Cor. 4.4).

    Past the wavelet's width the L2 misfit stops growing, so a descent on it stalls; W2 under the
    softplus scaling at ``||b f||_inf = 2``, inside the authors' range ``[0.2, 6]``, keeps growing.
    """
    d = _ricker(n=256, peak=64)
    shifts = [2, 5, 10, 20, 40]
    w2 = [float(seismic_wasserstein2_misfit(_shifted(d, s), d, **SOFTPLUS)) for s in shifts]
    l2 = [float(seismic_l2_misfit(_shifted(d, s), d)) for s in shifts]
    assert all(later > earlier for earlier, later in itertools.pairwise(w2))
    assert l2[-1] < max(l2)


def test_the_linear_normalization_keeps_polarity() -> None:
    """A one-to-one scaling (Assumption 4.1): ``-d`` and ``d`` are different densities."""
    d = jnp.asarray(_D)
    assert float(seismic_wasserstein2_misfit(-d, d)) > 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"normalization": "square"}, "normalization"),
        ({"normalization": "softplus"}, "b"),
        ({"normalization": "softplus", "b": 0.0}, "b"),
        ({"b": 1.0}, "b"),
        ({"c": -1.0}, "c"),
        ({"dt": 0.0}, "dt"),
    ],
)
def test_w2_refuses_arguments_outside_their_domain(kwargs: W2Options, match: str) -> None:
    d = jnp.asarray(_D)
    with pytest.raises(ValueError, match=match):
        seismic_wasserstein2_misfit(d, d, **kwargs)


def test_a_linear_density_that_goes_negative_is_nan() -> None:
    """Outside eq. 4.1's domain, ``sigma(f) + c >= 0``, a trace has no density."""
    d = jnp.asarray(_D)
    below = d - 2.0 * float(jnp.sum(jnp.abs(d)))
    assert np.isnan(float(seismic_wasserstein2_misfit(below, d)))


def test_all_zero_targets_under_the_linear_default_are_nan() -> None:
    """``c`` is the targets' l1 norm, so zero targets leave ``c = 0``, outside eq. 4.1's ``c > 0``."""
    assert np.isnan(float(seismic_wasserstein2_misfit(jnp.ones(64), jnp.zeros(64))))


@pytest.mark.parametrize("kwargs", [{}, SOFTPLUS], ids=["linear", "softplus"])
def test_a_silent_trace_in_a_gather_is_uniform_with_a_finite_gradient(
    kwargs: W2Options,
) -> None:
    """Eq. 4.1's buffer (or softplus's floor) makes a silent trace uniform, not 0/0."""
    targets = np.stack([_D, np.zeros_like(_D)])
    predictions = targets + 0.05 * np.sin(np.linspace(0, 20, _D.size)).astype(np.float32)
    expected, bound = _wasserstein2(
        predictions.astype(np.float64), targets.astype(np.float64), **kwargs
    )
    value, gradient = jax.value_and_grad(
        lambda p: seismic_wasserstein2_misfit(p, jnp.asarray(targets), **kwargs)
    )(jnp.asarray(predictions))
    assert abs(float(value) - expected) <= bound
    assert np.all(np.isfinite(np.asarray(gradient)))


def _reference_gradient(
    p: np.ndarray, t: np.ndarray, kwargs: W2Options, step: float = 1e-6
) -> np.ndarray:
    """``d/dp`` of the float64 reference by central differences, one sample at a time."""
    gradient = np.empty(p.size)
    for i in range(p.size):
        up, down = p.copy(), p.copy()
        up.flat[i] += step
        down.flat[i] -= step
        gradient[i] = (_wasserstein2(up, t, **kwargs)[0] - _wasserstein2(down, t, **kwargs)[0]) / (
            2.0 * step
        )
    return gradient.reshape(p.shape)


@pytest.mark.parametrize("kwargs", [{}, SOFTPLUS], ids=["linear", "softplus"])
def test_the_gradient_matches_a_central_difference_of_the_reference(
    kwargs: W2Options,
) -> None:
    """The autodiff gradient is the gradient of the computed objective, held to the reference's.

    Every entry of the gradient is a sum of the same quantile gaps the value is, scaled by cell
    masses, so it carries the value's relative rounding, ``bound / value``, at the gradient's
    scale. A float64 central difference at step ``1e-6`` is exact to about ``1e-10`` relative.
    """
    targets = _noise(5, 24)
    predictions = targets + 0.3 * _noise(6, 24)
    p64, t64 = predictions.astype(np.float64), targets.astype(np.float64)
    value, bound = _wasserstein2(p64, t64, **kwargs)
    expected = _reference_gradient(p64, t64, kwargs)
    got = np.asarray(
        jax.grad(lambda p: seismic_wasserstein2_misfit(p, jnp.asarray(targets), **kwargs))(
            jnp.asarray(predictions)
        )
    )
    tolerance = (bound / value) * float(np.abs(expected).max())
    assert float(np.abs(got - expected).max()) <= tolerance


def test_the_softplus_difference_keeps_the_digits_subtraction_loses() -> None:
    """``log1p(sigmoid(b) expm1(a - b))`` against ``softplus(a) - softplus(b)`` for close arguments.

    With ``a`` and ``b`` within a factor of 2, ``a - b`` is exact (Sterbenz), and the rest is
    ``expm1``, the three operations of ``sigmoid``, a product and ``log1p``, each within one
    rounding and each well conditioned near 0: 8 ``eps`` relative to the difference. Subtracting
    two softplus values of order 1 errs by ``eps`` of them, which over a difference of ``1e-4``
    is about ``1e4 eps``, so the control below must fail the same bound.
    """
    a = np.linspace(0.5, 3.0, 101, dtype=np.float32)
    b = (a + np.float32(1e-4)).astype(np.float32)
    exact = np.logaddexp(0.0, a.astype(np.float64)) - np.logaddexp(0.0, b.astype(np.float64))
    stable = np.asarray(seismic_module._softplus_difference(jnp.asarray(a), jnp.asarray(b)))
    subtracted = np.asarray(jax.nn.softplus(jnp.asarray(a)) - jax.nn.softplus(jnp.asarray(b)))
    assert float(np.max(np.abs(stable - exact) / np.abs(exact))) <= 8 * EPS
    assert float(np.max(np.abs(subtracted - exact) / np.abs(exact))) > 8 * EPS


def test_the_linear_mass_gap_is_carried_not_subtracted() -> None:
    """``q - p`` from ``t - y``, where both masses sit near ``1 / n`` under a large buffer.

    ``(h_t - h_p - p sum(h_t - h_p)) / S_t``: one rounding of ``t - y``, an ``n``-term sum within
    ``(n - 1) eps`` of its terms' magnitudes (Higham eq. 4.4), and three more roundings, so each
    entry is within ``(n + 4) eps (|t_i - y_i| + p_i sum|t - y|) / S_t``. Subtracting the two
    masses errs by ``eps`` of a mass, ``1 / n``, which against a gap ``c`` times smaller is far
    outside that, so the control must fail it.
    """
    targets = _noise(8, 128)[None, :]
    predictions = (targets + 0.05 * _noise(9, 128)).astype(np.float32)
    buffer = 1e3
    p, q, gap, valid = seismic_module._w2_cell_masses(
        jnp.asarray(predictions), jnp.asarray(targets), None, buffer
    )
    t64, y64 = targets.astype(np.float64), predictions.astype(np.float64)
    exact = (t64 + buffer) / (t64 + buffer).sum() - (y64 + buffer) / (y64 + buffer).sum()
    rise = np.abs(t64 - y64)
    n = targets.shape[-1]
    bound = (n + 4) * EPS * (rise + np.asarray(p) * rise.sum()) / (t64 + buffer).sum()
    assert bool(valid[0])
    assert np.all(np.abs(np.asarray(gap) - exact) <= bound)
    assert not np.all(np.abs(np.asarray(q - p) - exact) <= bound)


def test_w2_composes_with_jax_transformations() -> None:
    """vmap, grad, jvp, jacfwd and jit of grad all go through it; the two modes agree.

    ``jvp`` and ``grad . v`` are the same linearization summed in different orders, so they agree
    within a sum of ``2 n`` products, ``2 n eps sum |g_i v_i|`` (Higham eq. 4.4).
    """
    targets = jnp.asarray(np.stack([_D, np.roll(_D, 3)]))
    predictions = targets + 0.05 * jnp.sin(jnp.linspace(0, 20, _D.size))
    direction = jnp.asarray(_noise(7, targets.size).reshape(targets.shape))

    def fn(p: jax.Array) -> jax.Array:
        return seismic_wasserstein2_misfit(p, targets, **SOFTPLUS)

    gradient = jax.grad(fn)(predictions)
    _, tangent = jax.jvp(fn, (predictions,), (direction,))
    products = float(jnp.sum(jnp.abs(gradient * direction)))
    assert abs(float(tangent) - float(jnp.vdot(gradient, direction))) <= (
        2 * targets.size * EPS * products
    )
    forward = jax.jacfwd(fn)(predictions)
    assert float(jnp.max(jnp.abs(forward - gradient))) <= (
        2 * targets.size * EPS * float(jnp.max(jnp.abs(gradient)))
    )
    assert np.isfinite(float(jax.jit(jax.grad(fn))(predictions).sum()))
    batched = jax.vmap(lambda p: seismic_wasserstein2_misfit(p, targets))(
        jnp.stack([predictions, targets])
    )
    assert batched.shape == (2,)
    assert float(batched[1]) == 0.0


def test_the_velocity_ssim_misfit_is_one_minus_ssim_at_the_targets_range() -> None:
    model = jnp.asarray(_velocity_model())
    prediction = model + 300.0
    data_range = float(jnp.max(model) - jnp.min(model))
    expected = 1.0 - ssim(prediction, model, max_val=data_range)
    assert float(seismic_velocity_ssim_misfit(prediction, model)) == float(expected)
    assert float(seismic_velocity_ssim_misfit(prediction, model, max_val=data_range)) == float(
        expected
    )


def test_ssim_takes_a_traced_dynamic_range() -> None:
    """A data range read from the data is traced under ``jit``; it gives the float's value."""
    model = jnp.asarray(_velocity_model())
    prediction = model + 300.0
    data_range = jnp.max(model) - jnp.min(model)
    traced = jax.jit(lambda p, t, r: ssim(p, t, max_val=r))(prediction, model, data_range)
    assert float(traced) == float(ssim(prediction, model, max_val=float(data_range)))


@pytest.mark.parametrize("gamma", [0.0, -0.1])
def test_soft_dtw_refuses_a_temperature_that_is_not_positive(gamma: float) -> None:
    d = jnp.asarray(_D[:16])
    with pytest.raises(ValueError, match="gamma"):
        seismic_soft_dtw_misfit(d, d, gamma=gamma)


def test_soft_dtw_temperature_matches_the_reference() -> None:
    p, t = _noise(3, 24), _noise(4, 24)
    for gamma in (0.1, 1.0):
        expected, bound = _soft_dtw_divergence(p.astype(np.float64), t.astype(np.float64), gamma)
        assert abs(float(seismic_soft_dtw_misfit(p, t, gamma=gamma)) - expected) <= bound


@pytest.mark.parametrize("name", TRACE_METRICS)
def test_symmetry_is_as_declared(name: str) -> None:
    """A metric registered symmetric scores ``(p, t)`` as ``(t, p)``, within both orders' bounds."""
    predictions, targets = CASES["perturbed"]
    forward, backward = FUNCTIONS[name](predictions, targets), FUNCTIONS[name](targets, predictions)
    _, bound = REFERENCES[name](predictions.astype(np.float64), targets.astype(np.float64))
    _, bound_back = REFERENCES[name](targets.astype(np.float64), predictions.astype(np.float64))
    symmetric = abs(float(forward) - float(backward)) <= bound + bound_back
    assert symmetric is MetricRegistry().get(name).properties.is_symmetric


def test_the_velocity_ssim_misfit_is_symmetric_only_at_a_fixed_range() -> None:
    """Its default range is the targets', so swapping models with different ranges changes it."""
    model = jnp.asarray(_velocity_model())
    prediction = 1.3 * model + 50.0
    assert float(seismic_velocity_ssim_misfit(prediction, model)) != float(
        seismic_velocity_ssim_misfit(model, prediction)
    )
    fixed = float(jnp.max(model) - jnp.min(model))
    assert float(seismic_velocity_ssim_misfit(prediction, model, max_val=fixed)) == float(
        seismic_velocity_ssim_misfit(model, prediction, max_val=fixed)
    )
    assert MetricRegistry().get("seismic_velocity_ssim_misfit").properties.is_symmetric is False


class TestRegistry:
    """The seven metrics are registered with Tier 0 metadata in the ``seismic`` domain."""

    def test_the_domain_holds_the_seven_metrics(self) -> None:
        names = [entry.name for entry in MetricRegistry().list_by_domain("seismic")]
        assert sorted(names) == sorted(ALL_METRICS)

    @pytest.mark.parametrize("name", ALL_METRICS)
    def test_metadata(self, name: str) -> None:
        entry = MetricRegistry().get(name)
        assert entry.tier == MetricTier.PURE_FUNCTION
        assert entry.domain == "seismic"
        assert entry.direction.value == "lower"
        assert entry.signature == MetricSignature.PREDICTIONS_TARGETS
        assert entry.fn is FUNCTIONS[name]
        assert entry.properties.is_differentiable is True
        assert entry.properties.is_jit_compatible is True
