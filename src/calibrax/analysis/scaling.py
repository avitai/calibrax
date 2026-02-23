"""Scaling law fitting via log-linear regression.

Fits power-law relationships (value = a * size^b) using pure Python
log-linear regression. No external dependencies required.
"""

from __future__ import annotations

import math

from calibrax.core.models import ScalingLaw


_COMPLEXITY_TOLERANCE = 0.15
_COMPLEXITY_CLASSES: list[tuple[float, str]] = [
    (0.0, "O(1)"),
    (0.5, "O(sqrt(n))"),
    (1.0, "O(n)"),
    (1.5, "O(n^1.5)"),
    (2.0, "O(n^2)"),
    (3.0, "O(n^3)"),
]


def _validate_inputs(sizes: list[float], values: list[float]) -> None:
    """Validate scaling law inputs.

    Args:
        sizes: Input sizes.
        values: Measured values.

    Raises:
        ValueError: If inputs are empty or have different lengths.
    """
    if not sizes or not values:
        raise ValueError("Cannot fit scaling law on empty data")
    if len(sizes) != len(values):
        raise ValueError(f"Mismatched lengths: len(sizes)={len(sizes)}, len(values)={len(values)}")


def _fit_log_linear(
    log_sizes: list[float],
    log_values: list[float],
) -> tuple[float, float, float]:
    """Fit a linear regression on log-transformed data.

    Args:
        log_sizes: Log-transformed input sizes.
        log_values: Log-transformed measured values.

    Returns:
        Tuple of (slope, intercept, r_squared).
    """
    m = len(log_sizes)
    mean_x = sum(log_sizes) / m
    mean_y = sum(log_values) / m

    ss_xx = sum((x - mean_x) ** 2 for x in log_sizes)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_sizes, log_values))
    ss_yy = sum((y - mean_y) ** 2 for y in log_values)

    if ss_xx == 0:
        return 0.0, mean_y, 0.0

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(log_sizes, log_values))
    r_squared = 1.0 - (ss_res / ss_yy) if ss_yy > 0 else 0.0

    return slope, intercept, r_squared


def scaling_fit(sizes: list[float], values: list[float]) -> ScalingLaw:
    """Fit power-law: value = a * size^b using log-linear regression.

    Takes log of both sides: log(value) = log(a) + b * log(size),
    then fits a linear regression. Pure Python (no scipy/numpy needed).

    Args:
        sizes: Input sizes (e.g., batch sizes, dataset sizes).
        values: Measured values (e.g., throughput, latency).

    Returns:
        ScalingLaw with coefficient (a), exponent (b), r_squared, and
        complexity classification string.

    Raises:
        ValueError: If inputs are empty or have different lengths.
    """
    _validate_inputs(sizes, values)

    pairs = [(s, v) for s, v in zip(sizes, values) if s > 0 and v > 0]
    if not pairs:
        return ScalingLaw(coefficient=0.0, exponent=0.0, r_squared=0.0, complexity="O(1)")

    log_sizes = [math.log(s) for s, _ in pairs]
    log_values = [math.log(v) for _, v in pairs]

    if all(v == log_values[0] for v in log_values):
        return ScalingLaw(
            coefficient=math.exp(log_values[0]),
            exponent=0.0,
            r_squared=1.0,
            complexity="O(1)",
        )

    slope, intercept, r_squared = _fit_log_linear(log_sizes, log_values)

    return ScalingLaw(
        coefficient=math.exp(intercept),
        exponent=round(slope, 4),
        r_squared=round(r_squared, 6),
        complexity=_classify_complexity(slope),
    )


def _classify_complexity(exponent: float) -> str:
    """Classify a power-law exponent into Big-O notation.

    Args:
        exponent: The power-law exponent to classify.

    Returns:
        Big-O notation string (e.g., "O(n)", "O(n^2)").
    """
    for target, label in _COMPLEXITY_CLASSES:
        if abs(exponent - target) < _COMPLEXITY_TOLERANCE:
            return label
    return f"O(n^{exponent:.1f})"
