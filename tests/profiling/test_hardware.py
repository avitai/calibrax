"""Hardware specifications and their detection from JAX's device kind."""

from __future__ import annotations

import json

import pytest
from substrax.devices import DeviceInfo, DeviceKind

import calibrax.profiling.hardware as hw_module
from calibrax.profiling.hardware import (
    detect_hardware_specs,
    HARDWARE_SPECS,
    HardwareSpec,
    spec_for_device_kind,
)


# Vendor figures per chip: dense BF16 peak (FP32 accumulate) and memory bandwidth; the sources
# are cited beside the table in calibrax.profiling.hardware.
PUBLISHED = [
    ("a100_sxm4_40gb", 312.0e12, 1555.0e9),
    ("a100_pcie_40gb", 312.0e12, 1555.0e9),
    ("a100_sxm4_80gb", 312.0e12, 2039.0e9),
    ("a100_pcie_80gb", 312.0e12, 1935.0e9),
    ("h100_sxm", 989.4e12, 3350.0e9),
    ("h100_pcie", 756.0e12, 2000.0e9),
    ("rtx_4090", 165.2e12, 1008.0e9),
    ("tpu_v4", 275.0e12, 1200.0e9),
    ("tpu_v5e", 197.0e12, 819.0e9),
    ("tpu_v5p", 459.0e12, 2765.0e9),
    ("tpu_v6e", 918.0e12, 1638.0e9),
    ("cpu_generic", 2.0e12, 200.0e9),
]


class TestHardwareSpecs:
    def test_the_table_holds_the_published_accelerators(self) -> None:
        assert set(HARDWARE_SPECS) == {name for name, _, _ in PUBLISHED}

    @pytest.mark.parametrize(("name", "peak_flops", "memory_bandwidth"), PUBLISHED)
    def test_entries_carry_the_published_figures(
        self, name: str, peak_flops: float, memory_bandwidth: float
    ) -> None:
        spec = HARDWARE_SPECS[name]
        assert spec.name == name
        assert spec.peak_flops == peak_flops
        assert spec.memory_bandwidth == memory_bandwidth

    @pytest.mark.parametrize("name", sorted(HARDWARE_SPECS))
    def test_critical_intensity_is_the_ridge_point(self, name: str) -> None:
        spec = HARDWARE_SPECS[name]
        assert spec.critical_intensity == pytest.approx(spec.peak_flops / spec.memory_bandwidth)

    def test_tpu_v5e_ridge_point_matches_the_scaling_book(self) -> None:
        # 197 TFLOPS over 819 GB/s: the 240 FLOPs/byte the JAX scaling book quotes.
        assert HARDWARE_SPECS["tpu_v5e"].critical_intensity == pytest.approx(240, rel=0.01)

    def test_to_dict_is_json_with_the_ridge_point(self) -> None:
        record = HARDWARE_SPECS["a100_sxm4_80gb"].to_dict()
        assert json.loads(json.dumps(record)) == record
        assert record["critical_intensity"] == pytest.approx(312.0e12 / 2039.0e9)
        assert record["tensor_core_shapes"] == [[16, 16, 16], [16, 16, 8]]


class TestSpecForDeviceKind:
    @pytest.mark.parametrize(
        ("device_kind", "name"),
        [
            ("NVIDIA A100-SXM4-40GB", "a100_sxm4_40gb"),
            ("NVIDIA A100-PCIE-40GB", "a100_pcie_40gb"),
            ("NVIDIA A100-SXM4-80GB", "a100_sxm4_80gb"),
            ("NVIDIA A100 80GB PCIe", "a100_pcie_80gb"),
            ("NVIDIA H100 80GB HBM3", "h100_sxm"),
            ("NVIDIA H100 PCIe", "h100_pcie"),
            ("NVIDIA GeForce RTX 4090", "rtx_4090"),
            ("TPU v4", "tpu_v4"),
            ("TPU v5 lite", "tpu_v5e"),
            ("TPU v5e", "tpu_v5e"),
            ("TPU v5", "tpu_v5p"),
            ("TPU v5p", "tpu_v5p"),
            ("TPU v6 lite", "tpu_v6e"),
            ("TPU v6e", "tpu_v6e"),
            ("cpu", "cpu_generic"),
        ],
    )
    def test_a_known_device_kind_names_its_spec(self, device_kind: str, name: str) -> None:
        assert spec_for_device_kind(device_kind) is HARDWARE_SPECS[name]

    @pytest.mark.parametrize(
        "device_kind", ["NVIDIA L4", "Tesla T4", "NVIDIA GeForce RTX 3090", "TPU v4 lite", "METAL"]
    )
    def test_an_unknown_device_kind_has_no_spec(self, device_kind: str) -> None:
        assert spec_for_device_kind(device_kind) is None


class TestDetectHardwareSpecs:
    """detect_hardware_specs against a fake device snapshot."""

    @staticmethod
    def _detect(
        platform: str, device_kinds: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
    ) -> HardwareSpec | None:
        info = DeviceInfo(
            platform=platform,
            kind=DeviceKind.from_platform(platform),
            count=len(device_kinds),
            device_kinds=device_kinds,
        )
        monkeypatch.setattr(hw_module, "detect_devices", lambda: info)
        return detect_hardware_specs()

    def test_the_cpu_backend_gets_the_cpu_stand_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._detect("cpu", ("cpu",), monkeypatch) is HARDWARE_SPECS["cpu_generic"]

    def test_a_gpu_is_detected_by_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        detected = self._detect("gpu", ("NVIDIA H100 80GB HBM3",), monkeypatch)
        assert detected is HARDWARE_SPECS["h100_sxm"]

    def test_a_tpu_is_detected_by_generation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._detect("tpu", ("TPU v6 lite",), monkeypatch) is HARDWARE_SPECS["tpu_v6e"]

    def test_an_unlisted_accelerator_is_not_guessed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._detect("gpu", ("NVIDIA L4",), monkeypatch) is None
        assert self._detect("METAL", ("METAL",), monkeypatch) is None

    def test_mixed_device_kinds_are_not_guessed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kinds = ("NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3")
        assert self._detect("gpu", kinds, monkeypatch) is None
