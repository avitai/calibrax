"""Hardware specifications and detection for profiling.

Provides accelerator specs (TPU v5e, A100, H100, CPU) and utility
functions for hardware detection and synchronized execution timing.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import jax


# Hardware specifications for common accelerators
HARDWARE_SPECS: dict[str, dict[str, Any]] = {
    "tpu_v5e": {
        "peak_flops": 197.0e12,  # 197 TFLOPS (bf16)
        "peak_flops_bf16": 197.0e12,
        "memory_bandwidth": 1600.0e9,  # 1.6 TB/s
        "critical_intensity": 123.125,  # FLOPs/byte
    },
    "a100_80g": {
        "peak_flops": 312.0e12,  # 312 TFLOPS (bf16/fp16)
        "peak_flops_bf16": 312.0e12,
        "memory_bandwidth": 2039.0e9,  # 2.0 TB/s
        "critical_intensity": 153.0,
        "tensor_core_shapes": [(16, 16, 16), (16, 16, 8)],
    },
    "h100": {
        "peak_flops": 989.0e12,  # 989 TFLOPS (bf16/fp16)
        "peak_flops_bf16": 989.0e12,
        "memory_bandwidth": 3350.0e9,  # 3.35 TB/s
        "critical_intensity": 295.0,
        "tensor_core_shapes": [(16, 16, 16)],
    },
    "cpu_generic": {
        "peak_flops": 2.0e12,  # ~2 TFLOPS (optimistic)
        "peak_flops_bf16": 2.0e12,
        "memory_bandwidth": 200.0e9,  # ~200 GB/s
        "critical_intensity": 10.0,
        "simd_width": 8,
    },
}


def detect_hardware_specs() -> dict[str, Any]:
    """Detect current hardware and return appropriate specifications.

    Uses ``jax.default_backend()`` to determine the accelerator type
    and returns pre-configured specs for that platform.

    Returns:
        Hardware specification dictionary with peak_flops, memory_bandwidth,
        and critical_intensity keys (among others).
    """
    backend = jax.default_backend()

    if backend == "tpu":
        return HARDWARE_SPECS["tpu_v5e"]
    if backend == "gpu":
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
