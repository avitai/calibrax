"""Seismic (full-waveform inversion) misfits against float64 references of each quantity.

Each metric is compared, eagerly and under ``jax.jit``, with the quantity it defines computed in
float64 NumPy from the same float32 inputs. Every tolerance is a rounding bound for the
computation, stated where it is used: a recursive sum of ``n`` terms is within ``(n - 1) eps`` of
its value relative to the sum of the terms' magnitudes (Higham, *Accuracy and Stability of
Numerical Algorithms*, 2nd ed., 2002, eq. 4.4), plus one ``eps`` for each further rounding.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def _density(x: np.ndarray) -> np.ndarray:
    total = np.abs(x).sum()
    return np.abs(x) / total if total > 0 else np.full_like(x, 1.0 / x.size)


def _wasserstein2(p: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Mean squared gap of the sorted densities; each density entry is within ``(n + 2) eps``.

    With ``delta_i = (n + 2) eps (u_i + v_i)`` bounding the error of ``u_i - v_i``, the squared
    gap moves by at most ``2 |u_i - v_i| delta_i + delta_i^2``, and the mean adds ``n eps``.
    """
    p2, t2 = p.reshape(-1, p.shape[-1]), t.reshape(-1, t.shape[-1])
    values, bounds = [], []
    for a, b in zip(p2, t2, strict=True):
        u, v = np.sort(_density(b.astype(np.float64))), np.sort(_density(a.astype(np.float64)))
        delta = (a.size + 2) * EPS * (u + v)
        values.append(float(np.mean((u - v) ** 2)))
        bounds.append(
            float(np.mean(2 * np.abs(u - v) * delta + delta**2)) + a.size * EPS * values[-1]
        )
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
#: ``u - v`` of the sorted densities, or the identical-trace branch of the correlation.
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


def test_the_envelope_density_discards_polarity() -> None:
    """Documented formulation: positive envelopes, so ``-d`` and ``d`` share one density."""
    d = jnp.asarray(_D)
    assert float(seismic_wasserstein2_misfit(-d, d)) == 0.0


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
