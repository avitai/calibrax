"""Types for the parts of ruptures 1.1 calibrax uses; ruptures ships no type information.

Signatures follow ``ruptures.detection`` (Pelt, Binseg, Window). ``predict`` returns the
breakpoints as integers, Python ``int`` or a NumPy integer (``Window`` takes them from a NumPy
array), which ``SupportsInt`` covers.
"""

from typing import Self, SupportsInt

import numpy as np

class Pelt:
    def __init__(
        self,
        model: str = "l2",
        custom_cost: object | None = None,
        min_size: int = 2,
        jump: int = 5,
        params: dict[str, object] | None = None,
    ) -> None: ...
    def fit(self, signal: np.ndarray) -> Self: ...
    def predict(self, pen: float) -> list[SupportsInt]: ...

class Binseg:
    def __init__(
        self,
        model: str = "l2",
        custom_cost: object | None = None,
        min_size: int = 2,
        jump: int = 5,
        params: dict[str, object] | None = None,
    ) -> None: ...
    def fit(self, signal: np.ndarray) -> Self: ...
    def predict(
        self, n_bkps: int | None = None, pen: float | None = None, epsilon: float | None = None
    ) -> list[SupportsInt]: ...

class Window:
    def __init__(
        self,
        width: int = 100,
        model: str = "l2",
        custom_cost: object | None = None,
        min_size: int = 2,
        jump: int = 5,
        params: dict[str, object] | None = None,
    ) -> None: ...
    def fit(self, signal: np.ndarray) -> Self: ...
    def predict(
        self, n_bkps: int | None = None, pen: float | None = None, epsilon: float | None = None
    ) -> list[SupportsInt]: ...
