"""Tests for calibrax.profiling.carbon module."""

from __future__ import annotations

import builtins
import dataclasses
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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


class TestCarbonTracker:
    """Tests for CarbonTracker context manager."""

    def test_raises_import_error_when_unavailable(self) -> None:
        """Should raise ImportError when codecarbon is not installed."""
        with patch("calibrax.profiling.carbon.CODECARBON_AVAILABLE", False):
            with pytest.raises(ImportError, match="codecarbon is required"):
                CarbonTracker()

    def test_context_manager_with_mocked_tracker(self) -> None:
        """Should wrap EmissionsTracker and yield CarbonResult."""
        mock_emissions_tracker = MagicMock()
        mock_emissions_tracker.stop.return_value = 0.042
        mock_emissions_data = MagicMock()
        mock_emissions_data.energy_consumed = 0.015
        mock_emissions_data.duration = 30.0
        mock_emissions_tracker.final_emissions_data = mock_emissions_data

        with (
            patch("calibrax.profiling.carbon.CODECARBON_AVAILABLE", True),
            patch(
                "calibrax.profiling.carbon.EmissionsTracker",
                return_value=mock_emissions_tracker,
            ),
        ):
            with CarbonTracker() as tracker:
                mock_emissions_tracker.start.assert_called_once()

            result = tracker.result()

        assert isinstance(result, CarbonResult)
        assert result.emissions_kg_co2 == 0.042
        assert result.energy_consumed_kwh == 0.015
        assert result.duration_sec == 30.0

    def test_context_manager_with_country_code(self) -> None:
        """Should pass country_iso_code to result and EmissionsTracker kwargs."""
        mock_emissions_tracker = MagicMock()
        mock_emissions_tracker.stop.return_value = 0.01
        mock_emissions_tracker.final_emissions_data = None

        with (
            patch("calibrax.profiling.carbon.CODECARBON_AVAILABLE", True),
            patch(
                "calibrax.profiling.carbon.EmissionsTracker",
                return_value=mock_emissions_tracker,
            ) as mock_cls,
        ):
            with CarbonTracker(country_iso_code="DEU") as tracker:
                pass

            result = tracker.result()

        assert result.country_iso_code == "DEU"
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["country_iso_code"] == "DEU"

    def test_context_manager_handles_no_final_data(self) -> None:
        """Should handle None final_emissions_data gracefully."""
        mock_emissions_tracker = MagicMock()
        mock_emissions_tracker.stop.return_value = 0.0
        mock_emissions_tracker.final_emissions_data = None

        with (
            patch("calibrax.profiling.carbon.CODECARBON_AVAILABLE", True),
            patch(
                "calibrax.profiling.carbon.EmissionsTracker",
                return_value=mock_emissions_tracker,
            ),
        ):
            with CarbonTracker():
                pass

    def test_result_before_context_manager(self) -> None:
        """result() before entering context should return zeros."""
        with (
            patch("calibrax.profiling.carbon.CODECARBON_AVAILABLE", True),
            patch("calibrax.profiling.carbon.EmissionsTracker"),
        ):
            tracker = CarbonTracker()
            result = tracker.result()
            assert result.emissions_kg_co2 == 0.0
            assert result.energy_consumed_kwh == 0.0
            assert result.duration_sec == 0.0

    def test_exit_without_enter_is_noop(self) -> None:
        """__exit__ should be safe when tracker was never started."""
        with (
            patch("calibrax.profiling.carbon.CODECARBON_AVAILABLE", True),
            patch("calibrax.profiling.carbon.EmissionsTracker"),
        ):
            tracker = CarbonTracker()
            tracker.__exit__(None, None, None)
            result = tracker.result()
            assert result.emissions_kg_co2 == 0.0
            assert result.energy_consumed_kwh == 0.0
            assert result.duration_sec == 0.0

    def test_enter_falls_back_when_country_not_supported(self) -> None:
        """__enter__ should retry without country_iso_code on TypeError."""
        mock_tracker = MagicMock()

        with (
            patch("calibrax.profiling.carbon.CODECARBON_AVAILABLE", True),
            patch(
                "calibrax.profiling.carbon.EmissionsTracker",
                side_effect=[TypeError("unsupported kwarg"), mock_tracker],
            ) as tracker_cls,
        ):
            with CarbonTracker(country_iso_code="USA"):
                pass

        assert tracker_cls.call_count == 2
        first_kwargs = tracker_cls.call_args_list[0].kwargs
        second_kwargs = tracker_cls.call_args_list[1].kwargs
        assert "country_iso_code" in first_kwargs
        assert "country_iso_code" not in second_kwargs

    def test_module_import_sets_unavailable_when_codecarbon_missing(self) -> None:
        """Module import guard should set CODECARBON_AVAILABLE=False."""
        import calibrax.profiling.carbon as carbon_mod

        module_path = Path(carbon_mod.__file__)
        spec = importlib.util.spec_from_file_location("carbon_import_probe", module_path)
        assert spec is not None
        assert spec.loader is not None
        probe_module = importlib.util.module_from_spec(spec)

        real_import = builtins.__import__

        def _import_hook(name: str, *args: object, **kwargs: object) -> object:
            if name == "codecarbon":
                raise ImportError("missing codecarbon")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import_hook):
            sys.modules[spec.name] = probe_module
            try:
                spec.loader.exec_module(probe_module)
            finally:
                sys.modules.pop(spec.name, None)

        assert probe_module.CODECARBON_AVAILABLE is False
        assert probe_module.EmissionsTracker is None
