"""Carbon emissions tracking via CodeCarbon integration.

Wraps codecarbon's trackers as a context manager, exposing emissions data as a frozen
``CarbonResult`` dataclass: ``EmissionsTracker``, which locates the machine itself, or, when a
country is given, ``OfflineEmissionsTracker``, the tracker codecarbon takes ``country_iso_code``
on. codecarbon is imported when a tracker starts, since importing it loads the NVML bindings.
Requires the optional ``codecarbon`` dependency (``uv pip install "calibrax[codecarbon]"``).
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from substrax.records import read_record
from substrax.typing import JsonValue


CODECARBON_AVAILABLE = importlib.util.find_spec("codecarbon") is not None


class _EmissionsData(Protocol):
    """The emissions record fields CarbonTracker reads."""

    @property
    def energy_consumed(self) -> float: ...

    @property
    def duration(self) -> float: ...


class _Tracker(Protocol):
    """The part of a codecarbon tracker CarbonTracker uses."""

    @property
    def final_emissions_data(self) -> _EmissionsData | None: ...

    def start(self) -> None: ...

    def stop(self) -> float | None: ...


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

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        d: dict[str, JsonValue] = {
            "emissions_kg_co2": float(self.emissions_kg_co2),
            "energy_consumed_kwh": float(self.energy_consumed_kwh),
            "duration_sec": float(self.duration_sec),
        }
        if self.country_iso_code is not None:
            d["country_iso_code"] = self.country_iso_code
        return d

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> CarbonResult:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds a value its annotation
                does not admit.
        """
        return read_record(cls, data)


class CarbonTracker:
    """Context manager for tracking carbon emissions via CodeCarbon.

    Requires the ``codecarbon`` package. Install with:

    ```bash
    uv pip install "calibrax[codecarbon]"
    ```

    Usage:

    ```python
    with CarbonTracker() as tracker:
        # ... run workload ...
    result = tracker.result()
    print(f"Emissions: {result.emissions_kg_co2:.6f} kg CO2")
    ```
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
        self._tracker: _Tracker | None = None
        self._emissions: float = 0.0
        self._energy: float = 0.0
        self._duration: float = 0.0

    def __enter__(self) -> CarbonTracker:
        """Start emissions tracking."""
        from codecarbon import EmissionsTracker, OfflineEmissionsTracker

        tracker: _Tracker
        if self._country_iso_code is None:
            tracker = EmissionsTracker(log_level=self._log_level, save_to_file=False)
        else:
            tracker = OfflineEmissionsTracker(
                country_iso_code=self._country_iso_code,
                log_level=self._log_level,
                save_to_file=False,
            )
        tracker.start()
        self._tracker = tracker
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
