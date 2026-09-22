"""Accelerator specifications for roofline analysis: published, detected, or measured.

Each table entry holds a chip's dense BF16 peak (FP32 accumulate) and its memory bandwidth, from
the vendor's published specification; the ridge point of the roofline, ``critical_intensity`` in
FLOPs per byte, is their ratio. ``detect_hardware_specs`` names the chip by the
``jax.Device.device_kind`` JAX reports (the CUDA device name for a GPU) and returns ``None`` for a
device the table does not hold, a CPU among them: a spec is a measurement, never a guess.
``measure_hardware_spec`` measures any device's attainable ceilings through XLA instead, an
empirical roofline (the approach of LBNL's Empirical Roofline Tool, Lo et al. 2014).
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from substrax.devices import detect_devices
from substrax.typing import JsonValue

from calibrax.profiling.flops import FlopsCounter
from calibrax.profiling.timing import time_calls


@dataclass(frozen=True, slots=True, kw_only=True)
class HardwareSpec:
    """One accelerator's roofline figures.

    Attributes:
        name: The table name (``"h100_sxm"``).
        peak_flops: Dense BF16 peak with FP32 accumulate, in FLOP/s; a measured spec's peak is
            for the dtype it was measured in.
        memory_bandwidth: Memory bandwidth, in bytes/s.
        tensor_core_shapes: The CUDA tensor-core (WMMA) fragment shapes ``(m, n, k)`` for
            bf16 and tf32; empty for a chip without CUDA tensor cores.
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
# Tensor-core shapes: NVIDIA CUDA C++ Programming Guide (CUDA 12.3), "Element Types and Matrix
#   Sizes" of the warp matrix functions: __nv_bfloat16 at 16x16x16, 32x8x16 and 8x32x16, tf32 at
#   16x16x8, on compute capability 8.0 and higher ("Alternate Floating Point"); the A100 is 8.0,
#   the RTX 4090 8.9 and the H100 9.0. TPUs and CPUs have no CUDA tensor cores.
_WMMA_SHAPES: tuple[tuple[int, int, int], ...] = (
    (16, 16, 16),
    (32, 8, 16),
    (8, 32, 16),
    (16, 16, 8),
)

HARDWARE_SPECS: Mapping[str, HardwareSpec] = _specs(
    HardwareSpec(
        name="a100_sxm4_40gb",
        peak_flops=312.0e12,
        memory_bandwidth=1555.0e9,
        tensor_core_shapes=_WMMA_SHAPES,
    ),
    HardwareSpec(
        name="a100_pcie_40gb",
        peak_flops=312.0e12,
        memory_bandwidth=1555.0e9,
        tensor_core_shapes=_WMMA_SHAPES,
    ),
    HardwareSpec(
        name="a100_sxm4_80gb",
        peak_flops=312.0e12,
        memory_bandwidth=2039.0e9,
        tensor_core_shapes=_WMMA_SHAPES,
    ),
    HardwareSpec(
        name="a100_pcie_80gb",
        peak_flops=312.0e12,
        memory_bandwidth=1935.0e9,
        tensor_core_shapes=_WMMA_SHAPES,
    ),
    HardwareSpec(
        name="h100_sxm",
        peak_flops=989.4e12,
        memory_bandwidth=3350.0e9,
        tensor_core_shapes=_WMMA_SHAPES,
    ),
    HardwareSpec(
        name="h100_pcie",
        peak_flops=756.0e12,
        memory_bandwidth=2000.0e9,
        tensor_core_shapes=_WMMA_SHAPES,
    ),
    HardwareSpec(
        name="rtx_4090",
        peak_flops=165.2e12,
        memory_bandwidth=1008.0e9,
        tensor_core_shapes=_WMMA_SHAPES,
    ),
    HardwareSpec(name="tpu_v4", peak_flops=275.0e12, memory_bandwidth=1200.0e9),
    HardwareSpec(name="tpu_v5e", peak_flops=197.0e12, memory_bandwidth=819.0e9),
    HardwareSpec(name="tpu_v5p", peak_flops=459.0e12, memory_bandwidth=2765.0e9),
    HardwareSpec(name="tpu_v6e", peak_flops=918.0e12, memory_bandwidth=1638.0e9),
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


# Calls made before timing (compilation, caches) and timed calls; the median is reported.
_MEASUREMENT_WARMUP = 3
_MEASUREMENT_ITERATIONS = 10


def measure_hardware_spec(
    *, dtype: DTypeLike, matmul_size: int = 4096, triad_length: int = 2**26
) -> HardwareSpec:
    """The default device's attainable roofline ceilings, measured through XLA.

    The peak is a square ``matmul_size`` matmul: XLA's own FLOP count (``FlopsCounter``) over
    the median time (``time_calls``). The bandwidth is STREAM's triad ``a = b + s * c``
    (McCalpin) over ``triad_length`` elements: three arrays moved over the median time; the
    default of 2**26 elements (768 MiB of float32) is far above any last-level cache. These are
    the ceilings XLA's kernels reach on this device, the ones JAX code meets, and they are
    below a vendor's peak. Choose the device with ``jax.default_device``.

    Args:
        dtype: The element type measured; the peak depends on it.
        matmul_size: Side of the square matrices.
        triad_length: Elements per triad array.

    Returns:
        A spec named ``measured:<device kind>:<dtype>``.

    Raises:
        ValueError: If a size is not positive.
    """
    if matmul_size <= 0 or triad_length <= 0:
        msg = f"sizes must be positive, got matmul_size={matmul_size}, triad_length={triad_length}"
        raise ValueError(msg)
    element = jnp.dtype(dtype)

    matrix = jnp.ones((matmul_size, matmul_size), element)
    matmul = jax.jit(jnp.matmul)
    flops = FlopsCounter().count(matmul, matrix, matrix).total_flops
    matmul_sec = time_calls(
        lambda: matmul(matrix, matrix),
        warmup=_MEASUREMENT_WARMUP,
        iterations=_MEASUREMENT_ITERATIONS,
    ).median_sec

    b, c = jnp.ones((triad_length,), element), jnp.ones((triad_length,), element)
    triad = jax.jit(lambda x, y: x + 3.0 * y)
    triad_sec = time_calls(
        lambda: triad(b, c), warmup=_MEASUREMENT_WARMUP, iterations=_MEASUREMENT_ITERATIONS
    ).median_sec

    (device,) = matrix.devices()
    return HardwareSpec(
        name=f"measured:{device.device_kind}:{element.name}",
        peak_flops=flops / matmul_sec,
        memory_bandwidth=3 * triad_length * element.itemsize / triad_sec,
    )


def resolve_hardware_spec(*, dtype: DTypeLike) -> HardwareSpec:
    """The roofline figures of the devices this JAX process sees.

    The published spec when ``HARDWARE_SPECS`` lists the devices (:func:`detect_hardware_specs`),
    else their ceilings measured for ``dtype`` (:func:`measure_hardware_spec`). A measurement
    runs once per dtype per process, because the devices a process sees do not change.

    Args:
        dtype: The element type the workload computes in; a measured peak depends on it.

    Returns:
        The listed or the measured spec.
    """
    return _resolved_spec(jnp.dtype(dtype).name)


@functools.cache
def _resolved_spec(dtype_name: str) -> HardwareSpec:
    """The spec for one dtype, cached by its name so ``jnp.float32`` and ``"float32"`` share it."""
    return detect_hardware_specs() or measure_hardware_spec(dtype=dtype_name)
