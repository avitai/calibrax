"""Tests for calibrax.profiling.carbon module."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pytest
from substrax.testing import run_python

from calibrax.profiling.carbon import CarbonResult, CarbonTracker


class TestCarbonResult:
    """Tests for CarbonResult frozen dataclass."""

    def test_creation(self) -> None:
        """Should create CarbonResult with all required fields."""
        result = CarbonResult(
            emissions_kg_co2=0.005,
            energy_consumed_kwh=0.01,
            duration_sec=60.0,
        )
        assert result.emissions_kg_co2 == 0.005
        assert result.energy_consumed_kwh == 0.01
        assert result.duration_sec == 60.0
        assert result.country_iso_code is None

    def test_creation_with_country(self) -> None:
        """Should create CarbonResult with country_iso_code."""
        result = CarbonResult(
            emissions_kg_co2=0.1,
            energy_consumed_kwh=0.5,
            duration_sec=120.0,
            country_iso_code="DEU",
        )
        assert result.country_iso_code == "DEU"

    def test_frozen_immutability(self) -> None:
        """Should raise FrozenInstanceError on attribute mutation."""
        result = CarbonResult(
            emissions_kg_co2=0.005,
            energy_consumed_kwh=0.01,
            duration_sec=60.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.emissions_kg_co2 = 0.0  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.country_iso_code = "USA"  # type: ignore[misc]

    def test_to_dict_without_country(self) -> None:
        """to_dict should omit country_iso_code when None."""
        result = CarbonResult(
            emissions_kg_co2=0.005,
            energy_consumed_kwh=0.01,
            duration_sec=60.0,
        )
        d = result.to_dict()
        assert d == {
            "emissions_kg_co2": 0.005,
            "energy_consumed_kwh": 0.01,
            "duration_sec": 60.0,
        }
        assert "country_iso_code" not in d

    def test_to_dict_with_country(self) -> None:
        """to_dict should include country_iso_code when set."""
        result = CarbonResult(
            emissions_kg_co2=0.1,
            energy_consumed_kwh=0.5,
            duration_sec=120.0,
            country_iso_code="FRA",
        )
        d = result.to_dict()
        assert d["country_iso_code"] == "FRA"

    def test_from_dict_without_country(self) -> None:
        """from_dict should handle missing country_iso_code as None."""
        result = CarbonResult.from_dict(
            {
                "emissions_kg_co2": 0.002,
                "energy_consumed_kwh": 0.005,
                "duration_sec": 30.0,
            }
        )
        assert result.emissions_kg_co2 == 0.002
        assert result.country_iso_code is None

    def test_from_dict_with_country(self) -> None:
        """from_dict should restore country_iso_code."""
        result = CarbonResult.from_dict(
            {
                "emissions_kg_co2": 0.1,
                "energy_consumed_kwh": 0.5,
                "duration_sec": 120.0,
                "country_iso_code": "GBR",
            }
        )
        assert result.country_iso_code == "GBR"

    def test_to_dict_from_dict_round_trip(self) -> None:
        """to_dict/from_dict should produce an equivalent object."""
        original = CarbonResult(
            emissions_kg_co2=0.123,
            energy_consumed_kwh=0.456,
            duration_sec=789.0,
            country_iso_code="USA",
        )
        reconstructed = CarbonResult.from_dict(original.to_dict())
        assert reconstructed.emissions_kg_co2 == original.emissions_kg_co2
        assert reconstructed.energy_consumed_kwh == original.energy_consumed_kwh
        assert reconstructed.duration_sec == original.duration_sec
        assert reconstructed.country_iso_code == original.country_iso_code

    def test_to_dict_from_dict_round_trip_no_country(self) -> None:
        """Round-trip should work for CarbonResult without country."""
        original = CarbonResult(
            emissions_kg_co2=0.001,
            energy_consumed_kwh=0.002,
            duration_sec=10.0,
        )
        reconstructed = CarbonResult.from_dict(original.to_dict())
        assert reconstructed.emissions_kg_co2 == original.emissions_kg_co2
        assert reconstructed.country_iso_code is None


def _tracker_double(
    emissions: float = 0.0, energy: float = 0.0, duration: float = 0.0
) -> MagicMock:
    tracker = MagicMock()
    tracker.stop.return_value = emissions
    data = MagicMock()
    data.energy_consumed = energy
    data.duration = duration
    tracker.final_emissions_data = data
    return tracker


class TestCarbonTracker:
    """CarbonTracker wraps codecarbon's online tracker, or its offline tracker for a country."""

    def test_without_a_country_the_online_tracker_measures(self) -> None:
        online = _tracker_double(emissions=0.042, energy=0.015, duration=30.0)
        with (
            patch("calibrax.profiling.carbon.EmissionsTracker", return_value=online) as online_cls,
            patch("calibrax.profiling.carbon.OfflineEmissionsTracker") as offline_cls,
            CarbonTracker() as tracker,
        ):
            online.start.assert_called_once()

        result = tracker.result()
        assert (result.emissions_kg_co2, result.energy_consumed_kwh, result.duration_sec) == (
            0.042,
            0.015,
            30.0,
        )
        assert "country_iso_code" not in online_cls.call_args.kwargs
        offline_cls.assert_not_called()

    def test_a_country_uses_the_offline_tracker_with_that_country(self) -> None:
        """codecarbon's online tracker refuses country_iso_code; the offline one takes it."""
        offline = _tracker_double(emissions=0.01)
        with (
            patch("calibrax.profiling.carbon.EmissionsTracker") as online_cls,
            patch(
                "calibrax.profiling.carbon.OfflineEmissionsTracker", return_value=offline
            ) as offline_cls,
            CarbonTracker(country_iso_code="DEU") as tracker,
        ):
            pass

        assert offline_cls.call_args.kwargs["country_iso_code"] == "DEU"
        online_cls.assert_not_called()
        assert tracker.result().country_iso_code == "DEU"
        assert tracker.result().emissions_kg_co2 == 0.01

    def test_no_final_data_gives_zero_energy_and_duration(self) -> None:
        online = _tracker_double()
        online.final_emissions_data = None
        with (
            patch("calibrax.profiling.carbon.EmissionsTracker", return_value=online),
            CarbonTracker() as tracker,
        ):
            pass

        assert (tracker.result().energy_consumed_kwh, tracker.result().duration_sec) == (0.0, 0.0)

    def test_result_before_the_context_is_zero(self) -> None:
        result = CarbonTracker().result()
        assert (result.emissions_kg_co2, result.energy_consumed_kwh, result.duration_sec) == (
            0.0,
            0.0,
            0.0,
        )

    def test_exit_without_enter_is_noop(self) -> None:
        tracker = CarbonTracker()
        tracker.__exit__(None, None, None)
        assert tracker.result().emissions_kg_co2 == 0.0


def test_importing_without_codecarbon_names_the_extra() -> None:
    result = run_python(
        "import sys; sys.modules['codecarbon'] = None\n"
        "try:\n"
        "    import calibrax.profiling.carbon\n"
        "except ImportError as error:\n"
        "    print(error)\n",
        timeout=120,
    )

    assert "calibrax[codecarbon]" in result.stdout


def test_profiling_imports_without_codecarbon() -> None:
    result = run_python(
        "import sys; sys.modules['codecarbon'] = None\nimport calibrax.profiling\nprint('ok')\n",
        timeout=120,
    )

    assert result.stdout.strip() == "ok"
