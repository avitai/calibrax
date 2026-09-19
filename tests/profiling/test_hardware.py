"""Hardware specifications and their detection from JAX's device kind."""

from __future__ import annotations

import json
from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from substrax.devices import DeviceInfo, DeviceKind
from substrax.typing import PyTree

import calibrax.profiling.hardware as hw_module
from calibrax.profiling.hardware import (
    detect_hardware_specs,
    HARDWARE_SPECS,
    HardwareSpec,
    measure_hardware_spec,
    spec_for_device_kind,
)
from calibrax.profiling.roofline import RooflineAnalyzer
from calibrax.profiling.timing_records import CallTiming


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
        assert record["tensor_core_shapes"] == [[16, 16, 16], [32, 8, 16], [8, 32, 16], [16, 16, 8]]

    @pytest.mark.parametrize(
        "name", [n for n in sorted(HARDWARE_SPECS) if n[:4] in {"a100", "h100"}]
    )
    def test_ampere_and_later_gpus_hold_the_wmma_shapes(self, name: str) -> None:
        # CUDA C++ Programming Guide, Element Types and Matrix Sizes: bf16 16x16x16, 32x8x16 and
        # 8x32x16, tf32 16x16x8, on compute capability 8.0 and higher.
        assert HARDWARE_SPECS[name].tensor_core_shapes == (
            (16, 16, 16),
            (32, 8, 16),
            (8, 32, 16),
            (16, 16, 8),
        )

    def test_the_rtx_4090_has_the_same_tensor_cores(self) -> None:
        # Ada is compute capability 8.9.
        assert (
            HARDWARE_SPECS["rtx_4090"].tensor_core_shapes
            == HARDWARE_SPECS["h100_sxm"].tensor_core_shapes
        )

    @pytest.mark.parametrize(
        "name", [n for n in sorted(HARDWARE_SPECS) if not n.startswith(("a100", "h100", "rtx"))]
    )
    def test_chips_without_cuda_tensor_cores_list_no_shapes(self, name: str) -> None:
        assert HARDWARE_SPECS[name].tensor_core_shapes == ()


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
        ],
    )
    def test_a_known_device_kind_names_its_spec(self, device_kind: str, name: str) -> None:
        assert spec_for_device_kind(device_kind) is HARDWARE_SPECS[name]

    @pytest.mark.parametrize(
        "device_kind",
        ["NVIDIA L4", "Tesla T4", "NVIDIA GeForce RTX 3090", "TPU v4 lite", "METAL", "cpu"],
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

    def test_a_cpu_has_no_published_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A CPU's ceilings are measured (measure_hardware_spec), not looked up.
        assert self._detect("cpu", ("cpu",), monkeypatch) is None

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


class TestMeasureHardwareSpec:
    """measure_hardware_spec: the default device's attainable ceilings, measured through XLA."""

    def test_the_ceilings_are_the_counted_work_over_the_median_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        medians = iter([0.5, 0.25])  # the matmul is timed first, then the triad

        def fixed_median(call: Callable[[], PyTree], **_options: object) -> CallTiming:
            call()
            median = next(medians)
            return CallTiming(
                samples_sec=(median,), median_sec=median, percentiles_sec={}, warmup=0
            )

        monkeypatch.setattr(hw_module, "time_calls", fixed_median)

        spec = measure_hardware_spec(dtype=jnp.float32, matmul_size=64, triad_length=1024)

        assert spec.peak_flops == 2 * 64**3 / 0.5  # XLA's count of a 64x64 matmul
        assert spec.memory_bandwidth == 3 * 1024 * 4 / 0.25  # read b and c, write a
        assert spec.critical_intensity == pytest.approx(spec.peak_flops / spec.memory_bandwidth)

    def test_the_spec_names_the_measured_device_and_dtype(self) -> None:
        spec = measure_hardware_spec(dtype=jnp.float32, matmul_size=128, triad_length=2**16)

        assert spec.name == f"measured:{jax.devices()[0].device_kind}:float32"
        assert spec.peak_flops > 0
        assert spec.memory_bandwidth > 0
        assert spec.tensor_core_shapes == ()

    @pytest.mark.parametrize(("matmul_size", "triad_length"), [(0, 1024), (64, 0), (-1, 1024)])
    def test_sizes_must_be_positive(self, matmul_size: int, triad_length: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            measure_hardware_spec(
                dtype=jnp.float32, matmul_size=matmul_size, triad_length=triad_length
            )

    def test_a_measured_spec_drives_the_roofline(self) -> None:
        spec = measure_hardware_spec(dtype=jnp.float32, matmul_size=128, triad_length=2**16)

        result = RooflineAnalyzer(hardware_specs=spec).analyze_operation(
            lambda x: x @ x, [jnp.ones((64, 64))]
        )

        assert result.arithmetic_intensity > 0
