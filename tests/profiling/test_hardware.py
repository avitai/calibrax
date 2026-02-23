"""Tests for hardware specifications and detection.

Verifies HARDWARE_SPECS contents, backend-based detection via mocking,
execution time measurement, and synchronization barriers.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

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

    @pytest.mark.parametrize("spec_name", ["tpu_v5e", "a100_80g", "h100", "cpu_generic"])
    def test_peak_flops_positive(self, spec_name: str) -> None:
        assert HARDWARE_SPECS[spec_name]["peak_flops"] > 0

    @pytest.mark.parametrize("spec_name", ["tpu_v5e", "a100_80g", "h100", "cpu_generic"])
    def test_memory_bandwidth_positive(self, spec_name: str) -> None:
        assert HARDWARE_SPECS[spec_name]["memory_bandwidth"] > 0

    @pytest.mark.parametrize("spec_name", ["tpu_v5e", "a100_80g", "h100", "cpu_generic"])
    def test_critical_intensity_positive(self, spec_name: str) -> None:
        assert HARDWARE_SPECS[spec_name]["critical_intensity"] > 0

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
    """Tests for detect_hardware_specs with mocked backends.

    Replaces the ``jax`` reference on the hardware module with a mock
    to avoid corrupting JAX's internal backend state.
    """

    def _detect_with_backend(self, backend: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Run detect_hardware_specs with a fake jax.default_backend."""
        import calibrax.profiling.hardware as hw_module

        mock_jax = MagicMock()
        mock_jax.default_backend.return_value = backend
        monkeypatch.setattr(hw_module, "jax", mock_jax)
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
