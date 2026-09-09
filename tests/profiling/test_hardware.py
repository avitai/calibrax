"""Tests for hardware specifications and detection.

Verifies HARDWARE_SPECS contents, backend-based detection via mocking,
execution time measurement, and synchronization barriers.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
from substrax.devices import DeviceInfo, DeviceKind

from calibrax.profiling.hardware import (
    _block_until_ready,
    detect_hardware_specs,
    HARDWARE_SPECS,
    measure_execution_time,
)


class TestHardwareSpecs:
    """Tests for the HARDWARE_SPECS dictionary."""

    def test_contains_expected_keys(self) -> None:
        expected_keys = {"tpu_v5e", "a100_80g", "h100", "cpu_generic"}
        assert set(HARDWARE_SPECS.keys()) == expected_keys

    @pytest.mark.parametrize("spec_name", ["tpu_v5e", "a100_80g", "h100", "cpu_generic"])
    def test_entries_have_required_fields(self, spec_name: str) -> None:
        spec = HARDWARE_SPECS[spec_name]
        assert "peak_flops" in spec
        assert "memory_bandwidth" in spec
        assert "critical_intensity" in spec

    @pytest.mark.parametrize(
        ("spec_name", "peak_flops", "memory_bandwidth"),
        [
            # Vendor figures: dense bf16 TFLOPS and HBM bandwidth per chip.
            ("tpu_v5e", 197.0e12, 819.0e9),
            ("a100_80g", 312.0e12, 2039.0e9),
            ("h100", 989.0e12, 3350.0e9),
            ("cpu_generic", 2.0e12, 200.0e9),
        ],
    )
    def test_entries_carry_the_published_figures(
        self, spec_name: str, peak_flops: float, memory_bandwidth: float
    ) -> None:
        spec = HARDWARE_SPECS[spec_name]
        assert spec["peak_flops"] == peak_flops
        assert spec["peak_flops_bf16"] == peak_flops
        assert spec["memory_bandwidth"] == memory_bandwidth

    @pytest.mark.parametrize("spec_name", ["tpu_v5e", "a100_80g", "h100", "cpu_generic"])
    def test_critical_intensity_is_the_ridge_point(self, spec_name: str) -> None:
        spec = HARDWARE_SPECS[spec_name]
        assert spec["critical_intensity"] == pytest.approx(
            spec["peak_flops"] / spec["memory_bandwidth"]
        )

    def test_tpu_v5e_ridge_point_matches_the_scaling_book(self) -> None:
        # 197 TFLOPS over 819 GB/s: the 240 FLOPs/byte the JAX scaling book quotes.
        assert HARDWARE_SPECS["tpu_v5e"]["critical_intensity"] == pytest.approx(240, rel=0.01)

    @pytest.mark.parametrize("spec_name", ["tpu_v5e", "a100_80g", "h100", "cpu_generic"])
    def test_entries_have_peak_flops_bf16(self, spec_name: str) -> None:
        assert "peak_flops_bf16" in HARDWARE_SPECS[spec_name]

    def test_a100_has_tensor_core_shapes(self) -> None:
        assert "tensor_core_shapes" in HARDWARE_SPECS["a100_80g"]

    def test_h100_has_tensor_core_shapes(self) -> None:
        assert "tensor_core_shapes" in HARDWARE_SPECS["h100"]

    def test_cpu_generic_has_simd_width(self) -> None:
        assert "simd_width" in HARDWARE_SPECS["cpu_generic"]


class TestDetectHardwareSpecs:
    """Tests for detect_hardware_specs against a fake device snapshot."""

    def _detect_with_backend(self, backend: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Run detect_hardware_specs against a fake substrax device snapshot."""
        import calibrax.profiling.hardware as hw_module

        info = DeviceInfo(
            platform=backend,
            kind=DeviceKind.from_platform(backend),
            count=1,
            device_kinds=(backend,),
        )
        monkeypatch.setattr(hw_module, "detect_devices", lambda: info)
        return detect_hardware_specs()

    def test_returns_cpu_generic_on_cpu_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._detect_with_backend("cpu", monkeypatch)
        assert result is HARDWARE_SPECS["cpu_generic"]

    def test_returns_a100_on_gpu_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._detect_with_backend("gpu", monkeypatch)
        assert result is HARDWARE_SPECS["a100_80g"]

    def test_returns_tpu_v5e_on_tpu_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._detect_with_backend("tpu", monkeypatch)
        assert result is HARDWARE_SPECS["tpu_v5e"]

    def test_returns_cpu_generic_on_unknown_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._detect_with_backend("metal", monkeypatch)
        assert result is HARDWARE_SPECS["cpu_generic"]


class TestMeasureExecutionTime:
    """Tests for measure_execution_time."""

    def test_returns_positive_float(self) -> None:
        x = jnp.ones((4, 4))
        result = measure_execution_time(lambda a: a + 1, [x])

        assert isinstance(result, float)
        assert result > 0

    def test_custom_warmup_and_iterations(self) -> None:
        x = jnp.ones((2, 2))
        result = measure_execution_time(lambda a: a * 2, [x], warmup=1, iterations=5)

        assert isinstance(result, float)
        assert result > 0

    @patch("calibrax.profiling.hardware._block_until_ready")
    def test_calls_block_until_ready(self, mock_block: MagicMock) -> None:
        x = jnp.ones((2,))
        measure_execution_time(lambda a: a + 1, [x], warmup=2, iterations=3)

        # warmup (2) + iterations (3) = 5 calls
        assert mock_block.call_count == 5

    def test_returns_average_time(self) -> None:
        x = jnp.ones((8, 8))
        t1 = measure_execution_time(lambda a: a + 1, [x], warmup=1, iterations=20)
        t2 = measure_execution_time(lambda a: a + 1, [x], warmup=1, iterations=20)

        # Both should be small and roughly similar magnitude
        assert t1 > 0
        assert t2 > 0


class TestBlockUntilReady:
    """Tests for the _block_until_ready helper."""

    def test_calls_block_until_ready_on_array(self) -> None:
        mock_result = MagicMock()
        mock_result.block_until_ready = MagicMock()
        _block_until_ready(mock_result)
        mock_result.block_until_ready.assert_called_once()

    def test_handles_tuple_of_arrays(self) -> None:
        mock_a = MagicMock()
        mock_b = MagicMock()
        _block_until_ready((mock_a, mock_b))
        mock_a.block_until_ready.assert_called_once()
        mock_b.block_until_ready.assert_called_once()

    def test_handles_list_of_arrays(self) -> None:
        mock_a = MagicMock()
        mock_b = MagicMock()
        _block_until_ready([mock_a, mock_b])
        mock_a.block_until_ready.assert_called_once()
        mock_b.block_until_ready.assert_called_once()

    def test_handles_non_array_result(self) -> None:
        # Should not raise for plain Python values
        _block_until_ready(42)
        _block_until_ready("hello")
        _block_until_ready(None)

    def test_handles_mixed_tuple(self) -> None:
        mock_arr = MagicMock()
        plain_val = 42  # No block_until_ready attribute
        _block_until_ready((mock_arr, plain_val))
        mock_arr.block_until_ready.assert_called_once()
