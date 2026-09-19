"""Accelerator specifications for roofline analysis, and their detection from JAX's device kind.

Each entry holds a chip's dense BF16 peak (FP32 accumulate) and its memory bandwidth, from the
vendor's published specification; the ridge point of the roofline, ``critical_intensity`` in
FLOPs per byte, is their ratio. ``detect_hardware_specs`` names the chip by the
``jax.Device.device_kind`` JAX reports (the CUDA device name for a GPU) and returns ``None``
for an accelerator the table does not hold: a spec is a measurement, so an unlisted chip is
passed in by the caller rather than guessed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from substrax.devices import detect_devices
from substrax.typing import JsonValue


@dataclass(frozen=True, slots=True, kw_only=True)
class HardwareSpec:
    """One accelerator's roofline figures.

    Attributes:
        name: The table name (``"h100_sxm"``).
        peak_flops: Dense BF16 peak with FP32 accumulate, in FLOP/s.
        memory_bandwidth: Memory bandwidth, in bytes/s.
        tensor_core_shapes: Matrix-unit tile shapes ``(m, n, k)`` a kernel aligns to.
        simd_width: Vector width of a CPU, in lanes.
    """

    name: str
    peak_flops: float
    memory_bandwidth: float
    tensor_core_shapes: tuple[tuple[int, int, int], ...] = ()
    simd_width: int | None = None

    @property
    def critical_intensity(self) -> float:
        """The roofline's ridge point: FLOPs per byte where compute and bandwidth balance."""
        return self.peak_flops / self.memory_bandwidth

    def to_dict(self) -> dict[str, JsonValue]:
        """The JSON form, with the ridge point, as a run's environment records it."""
        return {
            "name": self.name,
            "peak_flops": self.peak_flops,
            "memory_bandwidth": self.memory_bandwidth,
            "critical_intensity": self.critical_intensity,
            "tensor_core_shapes": [list(shape) for shape in self.tensor_core_shapes],
            "simd_width": self.simd_width,
        }


def _specs(*specs: HardwareSpec) -> Mapping[str, HardwareSpec]:
    return MappingProxyType({spec.name: spec for spec in specs})


# Sources, per chip:
# A100: NVIDIA A100 datasheet (nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf), BFLOAT16 Tensor
#   Core 312 TFLOPS dense; GPU memory bandwidth 1,555 GB/s (40GB PCIe, 40GB SXM), 1,935 GB/s
#   (80GB PCIe), 2,039 GB/s (80GB SXM).
# H100 SXM5: NVIDIA H100 Tensor Core GPU Architecture whitepaper v1.04, Table 3, "Peak BF16 Tensor
#   TFLOPS with FP32 Accumulate" 989.4 dense; H100 product page, GPU memory bandwidth 3.35 TB/s.
# H100 PCIe: the same whitepaper table, 756 dense; H100 PCIe product brief PB-11133-001_v02,
#   Table 2, peak memory bandwidth 2,000 GB/s (the whitepaper table gives 2,039 GB/s).
# RTX 4090: NVIDIA Ada GPU Architecture whitepaper, Appendix A, "Peak BF16 Tensor TFLOPS with FP32
#   Accumulate" 165.2 dense; memory bandwidth 1,008 GB/s.
# TPU v4, v5e, v5p, v6e: Cloud TPU documentation, "System architecture" of each version: 275, 197,
#   459 and 918 TFLOPS BF16 per chip; 1,200, 819, 2,765 and 1,638 GB/s HBM bandwidth. The v5e
#   page's English edition now prints "800 GiBps"; the other editions and JAX's own table
#   (jax/_src/tpu_info.py) give 819 GB/s, kept here.
# cpu_generic is a stand-in for a modern server socket, not a measurement.
# The tensor-core shapes are carried over from the earlier table; no vendor source is recorded
# for them.
HARDWARE_SPECS: Mapping[str, HardwareSpec] = _specs(
    HardwareSpec(
        name="a100_sxm4_40gb",
        peak_flops=312.0e12,
        memory_bandwidth=1555.0e9,
        tensor_core_shapes=((16, 16, 16), (16, 16, 8)),
    ),
    HardwareSpec(
        name="a100_pcie_40gb",
        peak_flops=312.0e12,
        memory_bandwidth=1555.0e9,
        tensor_core_shapes=((16, 16, 16), (16, 16, 8)),
    ),
    HardwareSpec(
        name="a100_sxm4_80gb",
        peak_flops=312.0e12,
        memory_bandwidth=2039.0e9,
        tensor_core_shapes=((16, 16, 16), (16, 16, 8)),
    ),
    HardwareSpec(
        name="a100_pcie_80gb",
        peak_flops=312.0e12,
        memory_bandwidth=1935.0e9,
        tensor_core_shapes=((16, 16, 16), (16, 16, 8)),
    ),
    HardwareSpec(
        name="h100_sxm",
        peak_flops=989.4e12,
        memory_bandwidth=3350.0e9,
        tensor_core_shapes=((16, 16, 16),),
    ),
    HardwareSpec(
        name="h100_pcie",
        peak_flops=756.0e12,
        memory_bandwidth=2000.0e9,
        tensor_core_shapes=((16, 16, 16),),
    ),
    HardwareSpec(name="rtx_4090", peak_flops=165.2e12, memory_bandwidth=1008.0e9),
    HardwareSpec(name="tpu_v4", peak_flops=275.0e12, memory_bandwidth=1200.0e9),
    HardwareSpec(name="tpu_v5e", peak_flops=197.0e12, memory_bandwidth=819.0e9),
    HardwareSpec(name="tpu_v5p", peak_flops=459.0e12, memory_bandwidth=2765.0e9),
    HardwareSpec(name="tpu_v6e", peak_flops=918.0e12, memory_bandwidth=1638.0e9),
    HardwareSpec(name="cpu_generic", peak_flops=2.0e12, memory_bandwidth=200.0e9, simd_width=8),
)

# The device_kind JAX reports for each chip. GPU names are the CUDA device names JAX's own tests
# use (tests/pallas/triton_gpu_info_test.py), the RTX 4090's as measured; TPU names are those
# jax/_src/tpu_info.py accepts, the reported one and its alias.
_BY_DEVICE_KIND: Mapping[str, str] = MappingProxyType(
    {
        "NVIDIA A100-SXM4-40GB": "a100_sxm4_40gb",
        "NVIDIA A100-PCIE-40GB": "a100_pcie_40gb",
        "NVIDIA A100-SXM4-80GB": "a100_sxm4_80gb",
        "NVIDIA A100 80GB PCIe": "a100_pcie_80gb",
        "NVIDIA H100 80GB HBM3": "h100_sxm",
        "NVIDIA H100 PCIe": "h100_pcie",
        "NVIDIA GeForce RTX 4090": "rtx_4090",
        "TPU v4": "tpu_v4",
        "TPU v5 lite": "tpu_v5e",
        "TPU v5e": "tpu_v5e",
        "TPU v5": "tpu_v5p",
        "TPU v5p": "tpu_v5p",
        "TPU v6 lite": "tpu_v6e",
        "TPU v6e": "tpu_v6e",
        "cpu": "cpu_generic",
    }
)


class UnknownHardwareError(LookupError):
    """The accelerator in use is not in ``HARDWARE_SPECS`` and no spec was given."""


def spec_for_device_kind(device_kind: str) -> HardwareSpec | None:
    """The spec of the chip JAX names ``device_kind``, or ``None`` when the table lacks it.

    Args:
        device_kind: A ``jax.Device.device_kind``.

    Returns:
        The chip's spec, or ``None``.
    """
    name = _BY_DEVICE_KIND.get(device_kind)
    return None if name is None else HARDWARE_SPECS[name]


def detect_hardware_specs() -> HardwareSpec | None:
    """The spec of the devices this JAX process sees, when the table holds them.

    Returns:
        The spec shared by every visible device, or ``None`` when a device is not in the
        table or the devices are of different kinds.
    """
    kinds = set(detect_devices().device_kinds)
    if len(kinds) != 1:
        return None
    (kind,) = kinds
    return spec_for_device_kind(kind)
