"""Energy monitoring: GPU power through a power source such as NVML, CPU energy through RAPL.

``EnergyMonitor`` samples in a background thread during a benchmark. GPU power comes from the
source it is given (``calibrax.profiling.nvml.NvmlDevice`` reads NVML); CPU energy comes from the
Linux RAPL package counter. A reading that is not available leaves its fields ``None``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from calibrax.profiling._sampling import SamplingThread
from calibrax.profiling.resources import GpuPower


logger = logging.getLogger(__name__)

# RAPL sysfs files for the CPU package: the cumulative energy counter and the value it wraps at
# (Linux powercap sysfs ABI, ``energy_uj`` and ``max_energy_range_uj``).
_RAPL_DIRECTORY = Path("/sys/class/powercap/intel-rapl:0")
_MICROJOULES_PER_JOULE = 1_000_000.0


class GpuPowerSource(Protocol):
    """Anything that reads a GPU's power; ``NvmlDevice`` is one."""

    def power(self) -> GpuPower | None:
        """The GPU's current power draw and limit, or None when it cannot be read."""
        ...


def _read_rapl_uj(name: str) -> int | None:
    """Read one RAPL counter file in microjoules.

    Args:
        name: The file under the package's powercap directory.

    Returns:
        The value, or None when the file is absent or unreadable (non-root on most systems).
    """
    try:
        return int((_RAPL_DIRECTORY / name).read_text().strip())
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def _read_rapl_energy_uj() -> int | None:
    """The CPU package's cumulative energy counter in microjoules."""
    return _read_rapl_uj("energy_uj")


def _read_rapl_range_uj() -> int | None:
    """The value the CPU package's energy counter wraps at, in microjoules."""
    return _read_rapl_uj("max_energy_range_uj")


@dataclass(frozen=True, slots=True, kw_only=True)
class EnergySample:
    """Single energy measurement at a point in time.

    Attributes:
        timestamp: Time of measurement (perf_counter).
        gpu_power_watts: Instantaneous GPU power (None if unavailable).
        cpu_energy_joules: Cumulative CPU energy since monitoring start.
        gpu_energy_joules: Cumulative GPU energy since monitoring start.
    """

    timestamp: float
    gpu_power_watts: float | None
    cpu_energy_joules: float | None
    gpu_energy_joules: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class EnergySummary:
    """Aggregated energy usage over a monitoring period.

    Attributes:
        total_gpu_energy_joules: Total GPU energy consumed.
        total_cpu_energy_joules: Total CPU energy consumed.
        total_combined_energy_joules: GPU + CPU combined.
        mean_gpu_power_watts: Average GPU power draw.
        peak_gpu_power_watts: Maximum GPU power draw.
        duration_sec: Monitoring duration.
        num_samples: Total samples collected.
    """

    total_gpu_energy_joules: float | None
    total_cpu_energy_joules: float | None
    total_combined_energy_joules: float | None
    mean_gpu_power_watts: float | None
    peak_gpu_power_watts: float | None
    duration_sec: float
    num_samples: int


def _combine_energy(
    gpu_energy: float | None,
    cpu_energy: float | None,
) -> float | None:
    """Combine GPU and CPU energy values.

    Args:
        gpu_energy: GPU energy in joules, or None.
        cpu_energy: CPU energy in joules, or None.

    Returns:
        Combined energy, or None if both are None.
    """
    if gpu_energy is not None and cpu_energy is not None:
        return gpu_energy + cpu_energy
    if gpu_energy is not None:
        return gpu_energy
    if cpu_energy is not None:
        return cpu_energy
    return None


class EnergyMonitor:
    """Background energy monitoring of a GPU power source and the CPU's RAPL counter.

    GPU energy integrates the source's power draw over the sampling interval (rectangular
    rule); CPU energy is the RAPL counter's increase, across its wraparound. Without a GPU
    source, or where RAPL is unreadable, those fields stay ``None``.

    ```python
    with NvmlDevice(0) as gpu, EnergyMonitor(gpu=gpu) as monitor:
        ...  # run the benchmark
    summary = monitor.summary
    ```
    """

    def __init__(
        self, sample_interval_sec: float = 0.1, *, gpu: GpuPowerSource | None = None
    ) -> None:
        """Initialize EnergyMonitor.

        Args:
            sample_interval_sec: Seconds between energy samples.
            gpu: The GPU power source; None measures the CPU only.
        """
        self._interval = sample_interval_sec
        self._gpu = gpu
        self._samples: list[EnergySample] = []
        self._sampling_thread = SamplingThread(target=self._sample_loop)

        self._last_gpu_reading: tuple[float, float] | None = None
        self._gpu_energy_j: float | None = None
        self._rapl_range_uj: int | None = None
        self._rapl_prev_uj: int | None = None
        self._rapl_accumulated_uj = 0

    def __enter__(self) -> EnergyMonitor:
        """Start background energy sampling thread."""
        self._samples.clear()
        self._last_gpu_reading = None
        self._gpu_energy_j = None
        self._rapl_accumulated_uj = 0
        self._rapl_prev_uj = _read_rapl_energy_uj()
        self._rapl_range_uj = _read_rapl_range_uj()

        self._sampling_thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop background energy sampling thread."""
        self._sampling_thread.stop()

    def _sample_loop(self) -> None:
        """Collect energy samples until stopped."""
        while not self._sampling_thread.stop_event.is_set():
            now = time.perf_counter()
            gpu_power_w = self._read_gpu_power()
            if gpu_power_w is not None:
                self._integrate_gpu_power(now, gpu_power_w)
            self._update_rapl()

            self._samples.append(
                EnergySample(
                    timestamp=now,
                    gpu_power_watts=gpu_power_w,
                    cpu_energy_joules=self._compute_cpu_energy(),
                    gpu_energy_joules=self._gpu_energy_j,
                ),
            )

            self._sampling_thread.stop_event.wait(timeout=self._interval)

    def _read_gpu_power(self) -> float | None:
        """The GPU power source's draw in watts, or None without one or a reading."""
        if self._gpu is None:
            return None
        power = self._gpu.power()
        return None if power is None else power.draw_w

    def _integrate_gpu_power(self, now: float, power_w: float) -> None:
        """Add the previous reading's power over the interval since it (rectangular rule)."""
        if self._last_gpu_reading is not None:
            then, previous_w = self._last_gpu_reading
            self._gpu_energy_j = (self._gpu_energy_j or 0.0) + previous_w * (now - then)
        self._last_gpu_reading = (now, power_w)

    def _update_rapl(self) -> None:
        """Add the RAPL counter's increase since the last reading, across a wraparound.

        The counter wraps at ``max_energy_range_uj``; when that range is unreadable a wrap
        cannot be measured, so CPU energy becomes unknown rather than undercounted.
        """
        current_uj = _read_rapl_energy_uj()
        if current_uj is None or self._rapl_prev_uj is None:
            return

        delta = current_uj - self._rapl_prev_uj
        if delta < 0:
            if self._rapl_range_uj is None:
                logger.warning("RAPL counter wrapped with no readable range; CPU energy unknown")
                self._rapl_prev_uj = None
                return
            delta += self._rapl_range_uj
        self._rapl_accumulated_uj += delta
        self._rapl_prev_uj = current_uj

    def _compute_cpu_energy(self) -> float | None:
        """CPU energy in joules since the monitor started, or None where RAPL is unreadable."""
        if self._rapl_prev_uj is None:
            return None
        return self._rapl_accumulated_uj / _MICROJOULES_PER_JOULE

    @property
    def samples(self) -> list[EnergySample]:
        """Return a copy of all collected samples."""
        return list(self._samples)

    @property
    def summary(self) -> EnergySummary:
        """Compute aggregated energy summary.

        Returns:
            EnergySummary with totals, or None fields when unavailable.
        """
        if not self._samples:
            return EnergySummary(
                total_gpu_energy_joules=None,
                total_cpu_energy_joules=None,
                total_combined_energy_joules=None,
                mean_gpu_power_watts=None,
                peak_gpu_power_watts=None,
                duration_sec=0.0,
                num_samples=0,
            )

        duration = (
            self._samples[-1].timestamp - self._samples[0].timestamp
            if len(self._samples) > 1
            else 0.0
        )

        gpu_energy = self._gpu_energy_j
        cpu_energy = self._compute_cpu_energy()
        mean_power, peak_power = self._compute_gpu_power_stats()

        return EnergySummary(
            total_gpu_energy_joules=gpu_energy,
            total_cpu_energy_joules=cpu_energy,
            total_combined_energy_joules=_combine_energy(gpu_energy, cpu_energy),
            mean_gpu_power_watts=mean_power,
            peak_gpu_power_watts=peak_power,
            duration_sec=duration,
            num_samples=len(self._samples),
        )

    def _compute_gpu_power_stats(self) -> tuple[float | None, float | None]:
        """Compute mean and peak GPU power from samples.

        Returns:
            Tuple of (mean_power, peak_power), both None if no GPU data.
        """
        gpu_powers = [s.gpu_power_watts for s in self._samples if s.gpu_power_watts is not None]
        if not gpu_powers:
            return None, None
        return sum(gpu_powers) / len(gpu_powers), max(gpu_powers)
