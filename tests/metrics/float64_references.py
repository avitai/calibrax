"""Float64 reference formulas for the weighted ratio statistics, and their weight gradients.

Each formula is written from the statistic's definition rather than from the implementation,
in numpy float64, so a test comparing a float32 gradient against a central difference of it
measures the implementation's float32 evaluation error and nothing else.

**The tolerance is in float32 ULP at the gradient's own scale, not an absolute constant.** At
``h = 1e-5`` in float64 the reference's own truncation is near 1e-10 and its roundoff near
1e-11, four orders below what is measured, so ``eps`` for float32 (1.19e-07) is the unit the
error is in. Measured over 20 seeds by sizes 8/32/128 (8/32 for ``kendall_tau``, quadratic in
the sample), the worst error in those units was ``r_squared`` 7.29, ``r_squared_adjusted``
6.99, ``pearson_correlation`` 6.00, ``concordance_correlation`` 5.91 and ``kendall_tau`` 1.23,
and over 20 seeds by sizes 8/12/24 with a quarter of the weights zeroed it was 4.37 for
``pearson_correlation`` and 0.87 for ``kendall_tau``. All sit under 8; the bound allows 16.

An absolute constant cannot express it: those readings span 8.438e-10 to 1.142e-06, so one
number is fixed by the worst statistic and runs loose by orders of magnitude for the others. A
relative tolerance cannot either: the relative error reaches 3.3e+01 wherever a reference
derivative passes near zero, so it would be set by whichever element landed closest to it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


#: Float32 ULP allowed between a weight gradient and its float64 central difference.
WEIGHT_GRADIENT_ULPS = 16

#: The central difference's step. The reference formulas are rational in the weights away from
#: an all-zero vector, so a +/-h step at ``w_i = 0`` stays in their domain.
STEP = 1e-5

#: The same guard as ``calibrax.metrics._utils._EPSILON``, so a reference and the implementation
#: guard their ratios alike.
EPSILON = 1e-8

#: A reference statistic of two series and a weight vector.
Reference = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def weight_gradient_atol(reference: np.ndarray) -> float:
    """Float32 ULP at the scale of the gradient being compared.

    Args:
        reference: The float64 gradient the float32 one is compared against.

    Returns:
        The absolute tolerance, ``WEIGHT_GRADIENT_ULPS`` float32 ULP at that scale.
    """
    scale = max(1.0, float(np.abs(reference).max()))
    return WEIGHT_GRADIENT_ULPS * float(np.finfo(np.float32).eps) * scale


def central_difference(
    reference: Reference, a: np.ndarray, b: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """``d/dw`` of ``reference`` by a float64 central difference, one coordinate at a time.

    Args:
        reference: The statistic to differentiate.
        a: First series.
        b: Second series.
        w: Weights at which to differentiate.

    Returns:
        The float64 gradient with respect to ``w``.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    gradient = np.empty(w.size, dtype=np.float64)
    for i in range(w.size):
        up, down = np.array(w, dtype=np.float64), np.array(w, dtype=np.float64)
        up[i] += STEP
        down[i] -= STEP
        gradient[i] = (reference(a, b, up) - reference(a, b, down)) / (2.0 * STEP)
    return gradient


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    """``sum(w x) / (sum(w) + EPSILON)``, the implementation's guarded weighted mean."""
    return float((w * x).sum() / (w.sum() + EPSILON))


def r_squared(p: np.ndarray, t: np.ndarray, w: np.ndarray) -> float:
    """Weighted R-squared, ``SS_tot`` centred on the weighted mean; NaN for no weight."""
    if w.sum() <= 0:
        return float("nan")
    ss_res = float((w * (p - t) ** 2).sum())
    ss_tot = float((w * (t - weighted_mean(t, w)) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + EPSILON)


def r_squared_adjusted(p: np.ndarray, t: np.ndarray, w: np.ndarray, k: int = 2) -> float:
    """Adjusted R-squared with ``k`` predictors, ``n`` the element count; NaN without a
    residual degree of freedom."""
    n = float(t.size)
    dof = n - k - 1.0
    if dof <= 0:
        return float("nan")
    return 1.0 - (1.0 - r_squared(p, t, w)) * (n - 1.0) / (dof + EPSILON)


def pearson(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson r: weighted means, covariance and standard deviations."""
    ac = a - weighted_mean(a, w)
    bc = b - weighted_mean(b, w)
    denominator = np.sqrt((w * ac**2).sum()) * np.sqrt((w * bc**2).sum()) + EPSILON
    return float((w * ac * bc).sum() / denominator)


def concordance(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted concordance correlation coefficient."""
    total = w.sum() + EPSILON
    mean_a, mean_b = weighted_mean(a, w), weighted_mean(b, w)
    var_a = (w * (a - mean_a) ** 2).sum() / total
    var_b = (w * (b - mean_b) ** 2).sum() / total
    cov = (w * (a - mean_a) * (b - mean_b)).sum() / total
    return float(2.0 * cov / (var_a + var_b + (mean_a - mean_b) ** 2 + EPSILON))


def kendall(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted tau-b: one weight ``w_i * w_j`` per pair."""
    n = a.size
    a_diff, b_diff = a[:, None] - a[None, :], b[:, None] - b[None, :]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair = np.where(upper, w[:, None] * w[None, :], 0.0)
    concordant = (pair * np.sign(a_diff) * np.sign(b_diff)).sum()
    total = pair.sum()
    tied_a = np.where(a_diff == 0, pair, 0.0).sum()
    tied_b = np.where(b_diff == 0, pair, 0.0).sum()
    return float(concordant / (np.sqrt((total - tied_a) * (total - tied_b)) + EPSILON))
