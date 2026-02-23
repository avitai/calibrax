"""Carbon emissions tracking via CodeCarbon integration.

Wraps the ``codecarbon.EmissionsTracker`` as a context manager, exposing
emissions data as a frozen ``CarbonResult`` dataclass. Requires the
optional ``codecarbon`` dependency (``uv pip install "calibrax[codecarbon]"``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any


try:
    from codecarbon import EmissionsTracker

    CODECARBON_AVAILABLE = True
except ImportError:
    EmissionsTracker = None  # type: ignore[assignment, misc]
    CODECARBON_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class CarbonResult:
    """Result of carbon emissions measurement.

    Attributes:
        emissions_kg_co2: Total CO2 emissions in kilograms.
        energy_consumed_kwh: Total energy consumed in kilowatt-hours.
        duration_sec: Duration of the tracked period in seconds.
        country_iso_code: ISO code of the country used for carbon intensity.
    """

    emissions_kg_co2: float
    energy_consumed_kwh: float
    duration_sec: float
    country_iso_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        d: dict[str, Any] = {
            "emissions_kg_co2": float(self.emissions_kg_co2),
            "energy_consumed_kwh": float(self.energy_consumed_kwh),
            "duration_sec": float(self.duration_sec),
        }
        if self.country_iso_code is not None:
            d["country_iso_code"] = self.country_iso_code
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CarbonResult:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with carbon result fields.

        Returns:
            Reconstructed CarbonResult instance.
        """
        return cls(
            emissions_kg_co2=data["emissions_kg_co2"],
            energy_consumed_kwh=data["energy_consumed_kwh"],
            duration_sec=data["duration_sec"],
            country_iso_code=data.get("country_iso_code"),
        )


class CarbonTracker:
    """Context manager for tracking carbon emissions via CodeCarbon.

    Requires the ``codecarbon`` package. Install with::

        uv pip install "calibrax[codecarbon]"

    Usage::

        with CarbonTracker() as tracker:
            # ... run workload ...
        result = tracker.result()
        print(f"Emissions: {result.emissions_kg_co2:.6f} kg CO2")

    Args:
        country_iso_code: Optional ISO country code for regional carbon intensity.
        log_level: Logging level for CodeCarbon (default: "warning").

    Raises:
        ImportError: If codecarbon is not installed.
    """

    def __init__(
        self,
        country_iso_code: str | None = None,
        log_level: str = "warning",
    ) -> None:
        """Initialize the carbon tracker.

        Args:
            country_iso_code: Optional ISO country code.
            log_level: CodeCarbon logging level.

        Raises:
            ImportError: If codecarbon is not installed.
        """
        if not CODECARBON_AVAILABLE:
            msg = 'codecarbon is required for CarbonTracker: uv pip install "calibrax[codecarbon]"'
            raise ImportError(msg)

        self._country_iso_code = country_iso_code
        self._log_level = log_level
        self._tracker: Any = None
        self._emissions: float = 0.0
        self._energy: float = 0.0
        self._duration: float = 0.0

    def __enter__(self) -> CarbonTracker:
        """Start emissions tracking."""
        kwargs: dict[str, Any] = {
            "log_level": self._log_level,
            "save_to_file": False,
        }
        if self._country_iso_code is not None:
            kwargs["country_iso_code"] = self._country_iso_code

        try:
            self._tracker = EmissionsTracker(**kwargs)  # type: ignore[misc]
        except TypeError:
            # Handle codecarbon versions that don't accept country_iso_code
            kwargs.pop("country_iso_code", None)
            self._tracker = EmissionsTracker(**kwargs)  # type: ignore[misc]
        self._tracker.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop emissions tracking and record results."""
        if self._tracker is not None:
            self._emissions = self._tracker.stop() or 0.0
            data = self._tracker.final_emissions_data
            if data:
                self._energy = getattr(data, "energy_consumed", 0.0) or 0.0
                self._duration = getattr(data, "duration", 0.0) or 0.0
            else:
                self._energy = 0.0
                self._duration = 0.0

    def result(self) -> CarbonResult:
        """Get the carbon emissions result.

        Call this after exiting the context manager.

        Returns:
            CarbonResult with emissions, energy, and duration data.
        """
        return CarbonResult(
            emissions_kg_co2=float(self._emissions),
            energy_consumed_kwh=float(self._energy),
            duration_sec=float(self._duration),
            country_iso_code=self._country_iso_code,
        )
