"""Hardware specifications and detection for profiling.

Provides accelerator specs (TPU v5e, A100, H100, CPU) and utility
functions for hardware detection and synchronized execution timing.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import jax
from substrax.devices import detect_devices, DeviceKind


def _spec(peak_flops: float, memory_bandwidth: float, **extra: Any) -> dict[str, Any]:
    """Build one hardware entry from its two measured figures.

    The ridge point of the roofline, ``critical_intensity`` in FLOPs per byte, is the
    ratio of the two and is derived rather than typed so the three cannot disagree.

    Args:
        peak_flops: Dense bf16 peak in FLOP/s.
        memory_bandwidth: HBM (or system memory) bandwidth in bytes/s.
        **extra: Accelerator-specific fields such as tensor core shapes.

    Returns:
        The specification dictionary the roofline analyzer reads.
    """
    return {
        "peak_flops": peak_flops,
        "peak_flops_bf16": peak_flops,
        "memory_bandwidth": memory_bandwidth,
        "critical_intensity": peak_flops / memory_bandwidth,
        **extra,
    }


# Dense bf16 peak and memory bandwidth per chip, from the vendors' published specifications:
# TPU v5e: 197 TFLOPS, 819 GB/s HBM2 (cloud.google.com/tpu/docs/v5e).
# A100 80GB SXM: 312 TFLOPS, 2039 GB/s (NVIDIA A100 datasheet).
# H100 SXM: 989 TFLOPS, 3.35 TB/s (NVIDIA H100 datasheet).
# cpu_generic is a stand-in for a modern server socket, not a measurement.
HARDWARE_SPECS: dict[str, dict[str, Any]] = {
    "tpu_v5e": _spec(197.0e12, 819.0e9),
    "a100_80g": _spec(312.0e12, 2039.0e9, tensor_core_shapes=[(16, 16, 16), (16, 16, 8)]),
    "h100": _spec(989.0e12, 3350.0e9, tensor_core_shapes=[(16, 16, 16)]),
    "cpu_generic": _spec(2.0e12, 200.0e9, simd_width=8),
}


def detect_hardware_specs() -> dict[str, Any]:
    """Detect current hardware and return appropriate specifications.

    Reads the accelerator class from ``substrax.devices.detect_devices`` and
    returns the pre-configured specs for that platform.

    Returns:
        Hardware specification dictionary with peak_flops, memory_bandwidth,
        and critical_intensity keys (among others).
    """
    kind = detect_devices().kind

    if kind is DeviceKind.TPU:
        return HARDWARE_SPECS["tpu_v5e"]
    if kind is DeviceKind.GPU:
        return HARDWARE_SPECS["a100_80g"]

    return HARDWARE_SPECS["cpu_generic"]


def measure_execution_time(
    func: Callable[..., Any],
    inputs: list[jax.Array],
    warmup: int = 3,
    iterations: int = 10,
) -> float:
    """Measure execution time of a JAX function with synchronization.

    JIT-compiles the function, runs warmup iterations, then times
    ``iterations`` executions with ``block_until_ready()`` barriers.

    Args:
        func: JAX function to benchmark.
        inputs: Input arguments as a list of arrays.
        warmup: Number of warmup iterations (for JIT compilation).
        iterations: Number of timed iterations.

    Returns:
        Average execution time in seconds.
    """
    compiled_func = jax.jit(func)

    for _ in range(warmup):
        result = compiled_func(*inputs)
        _block_until_ready(result)

    start_time = time.perf_counter()
    for _ in range(iterations):
        result = compiled_func(*inputs)
        _block_until_ready(result)

    total_time = time.perf_counter() - start_time
    return total_time / iterations


def _block_until_ready(result: Any) -> None:
    """Block until a JAX result is materialized.

    Handles single arrays, tuples, and lists of arrays.

    Args:
        result: JAX computation result.
    """
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    elif isinstance(result, tuple | list):
        for r in result:
            if hasattr(r, "block_until_ready"):
                r.block_until_ready()
