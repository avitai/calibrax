"""Shared numerical rounding constants for host-side statistical comparison modules."""

from __future__ import annotations

import numpy as np


_MIN_SAMPLE_SIZE = 2


def _higham_gamma(k: int, dtype: np.dtype[np.floating]) -> float:
    """Higham (2nd ed., Eq. 3.4) rounding constant ``k * u / (1 - k * u)`` for ``u = eps / 2``."""
    u = 0.5 * float(np.finfo(dtype).eps)
    ku = min(float(k) * u, 0.5)
    return ku / (1.0 - ku)
