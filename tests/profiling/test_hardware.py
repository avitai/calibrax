"""Tests for hardware specifications and detection.

Verifies HARDWARE_SPECS contents, backend-based detection via mocking,
execution time measurement, and synchronization barriers.
"""

from typing import Any

import pytest
from substrax.devices import DeviceInfo, DeviceKind

from calibrax.profiling.hardware import (
    detect_hardware_specs,
    HARDWARE_SPECS,
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
