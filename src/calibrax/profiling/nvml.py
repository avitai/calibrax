"""GPU readings through NVIDIA's NVML: memory, compute utilization, clocks and power.

This module is the NVML integration: it needs NVIDIA's nvidia-ml-py (the ``cuda12`` extra),
importing it without that raises ``ImportError`` naming the extra, and it is not re-exported from
``calibrax.profiling``. ``NvmlDevice`` satisfies ``GPUProfilerProtocol``, so a
``ResourceMonitor`` or an ``EnergyMonitor`` takes one to read the GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Self


try:
    import pynvml
except ImportError as error:
    msg = 'calibrax.profiling.nvml needs nvidia-ml-py: uv pip install "calibrax[cuda12]"'
    raise ImportError(msg) from error

from calibrax.profiling.resources import GpuClocks, GpuMemory, GpuPower


_BYTES_PER_MB = 1024 * 1024
_MILLIWATTS_PER_WATT = 1000.0


class NvmlDevice:
    """One GPU read through NVML, initialised once for the device's lifetime.

    A reading the GPU does not support (NVML's ``NOT_SUPPORTED``) is ``None``; any other NVML
    error is raised.

    ```python
    with NvmlDevice(0) as gpu, ResourceMonitor(gpu_profiler=gpu) as monitor:
        ...  # run the benchmark
    ```
    """

    def __init__(self, index: int = 0) -> None:
        """Initialise NVML and take the handle of GPU ``index``.

        Args:
            index: The GPU's NVML index.

        Raises:
            pynvml.NVMLError: If NVML cannot start (no driver) or has no GPU ``index``.
        """
        pynvml.nvmlInit()
        try:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except pynvml.NVMLError:
            pynvml.nvmlShutdown()
            raise

    def __enter__(self) -> Self:
        """The device."""
        return self

    def __exit__(self, *args: object) -> None:
        """Shut NVML down."""
        self.close()

    def close(self) -> None:
        """Shut NVML down; the device takes no readings after."""
        pynvml.nvmlShutdown()

    def memory(self) -> GpuMemory | None:
        """The GPU's memory in use and in total."""
        info = _supported(lambda: pynvml.nvmlDeviceGetMemoryInfo(self._handle))
        if info is None:
            return None
        return GpuMemory(
            used_mb=float(info.used) / _BYTES_PER_MB, total_mb=float(info.total) / _BYTES_PER_MB
        )

    def utilization(self) -> float | None:
        """The GPU's compute utilization over NVML's last sample period, in percent."""
        rates = _supported(lambda: pynvml.nvmlDeviceGetUtilizationRates(self._handle))
        return None if rates is None else float(rates.gpu)

    def clocks(self) -> GpuClocks | None:
        """The GPU's current graphics and memory clocks."""
        graphics = _supported(
            lambda: pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_GRAPHICS)
        )
        memory = _supported(
            lambda: pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_MEM)
        )
        if graphics is None or memory is None:
            return None
        return GpuClocks(graphics_mhz=float(graphics), memory_mhz=float(memory))

    def power(self) -> GpuPower | None:
        """The GPU's current power draw and management limit."""
        draw = _supported(lambda: pynvml.nvmlDeviceGetPowerUsage(self._handle))
        limit = _supported(lambda: pynvml.nvmlDeviceGetPowerManagementLimit(self._handle))
        if draw is None or limit is None:
            return None
        return GpuPower(
            draw_w=float(draw) / _MILLIWATTS_PER_WATT, limit_w=float(limit) / _MILLIWATTS_PER_WATT
        )


def _supported[T](query: Callable[[], T]) -> T | None:
    """The query's answer, or ``None`` when NVML reports the query unsupported on this GPU.

    Args:
        query: The NVML call.

    Returns:
        Its answer, or ``None`` when unsupported.

    Raises:
        pynvml.NVMLError: For any NVML error other than ``NOT_SUPPORTED``.
    """
    try:
        return query()
    except pynvml.NVMLError as error:
        if error.value == pynvml.NVML_ERROR_NOT_SUPPORTED:
            return None
        raise
