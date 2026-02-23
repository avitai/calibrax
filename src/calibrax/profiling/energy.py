"""Energy monitoring via NVML (GPU) and RAPL (CPU).

Provides EnergyMonitor context manager for tracking power consumption
during benchmark execution. Gracefully degrades when hardware
interfaces are unavailable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from calibrax.profiling._sampling import SamplingThread


logger = logging.getLogger(__name__)

# RAPL sysfs path for CPU package energy
_RAPL_ENERGY_PATH = Path(
    "/sys/class/powercap/intel-rapl:0/energy_uj",
)


def _get_nvml_power_mw() -> int | None:
    """Read GPU power draw in milliwatts via pynvml.

    Returns:
        Power draw in milliwatts, or None if unavailable.
    """
    try:
        import pynvml  # type: ignore[import-untyped]
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        power = pynvml.nvmlDeviceGetPowerUsage(handle)
        return int(power)
    except (
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        OSError,
        pynvml.NVMLError,  # type: ignore[attr-defined]
    ):
        return None


def _read_rapl_energy_uj() -> int | None:
    """Read CPU energy counter in microjoules from Linux RAPL sysfs.

    Returns:
        Cumulative energy in microjoules, or None if unavailable.
    """
    try:
        return int(_RAPL_ENERGY_PATH.read_text().strip())
    except (FileNotFoundError, PermissionError, ValueError):
        return None


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
    """Background energy monitoring via NVML and RAPL.

    Uses daemon thread sampling at configurable interval.
    Gracefully degrades when NVML or RAPL is unavailable.

    Usage::

        with EnergyMonitor() as mon:
            # ... run benchmark ...
        summary = mon.summary
    """

    def __init__(self, sample_interval_sec: float = 0.1) -> None:
        """Initialize EnergyMonitor.

        Args:
            sample_interval_sec: Seconds between energy samples.
        """
        self._interval = sample_interval_sec
        self._samples: list[EnergySample] = []
        self._sampling_thread = SamplingThread(target=self._sample_loop)

        # Tracked state for integration
        self._gpu_power_readings: list[tuple[float, float]] = []
        self._rapl_start_uj: int | None = None
        self._rapl_prev_uj: int | None = None
        self._rapl_accumulated_uj: int = 0

    def __enter__(self) -> EnergyMonitor:
        """Start background energy sampling thread."""
        self._samples.clear()
        self._gpu_power_readings.clear()
        self._rapl_accumulated_uj = 0

        # Initialize RAPL baseline
        rapl_uj = _read_rapl_energy_uj()
        self._rapl_start_uj = rapl_uj
        self._rapl_prev_uj = rapl_uj

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
                self._gpu_power_readings.append((now, gpu_power_w))

            self._update_rapl()

            gpu_energy = self._compute_gpu_energy()
            cpu_energy = self._compute_cpu_energy()

            self._samples.append(
                EnergySample(
                    timestamp=now,
                    gpu_power_watts=gpu_power_w,
                    cpu_energy_joules=cpu_energy,
                    gpu_energy_joules=gpu_energy,
                ),
            )

            self._sampling_thread.stop_event.wait(timeout=self._interval)

    def _read_gpu_power(self) -> float | None:
        """Read GPU power in watts."""
        power_mw = _get_nvml_power_mw()
        if power_mw is None:
            return None
        return power_mw / 1000.0

    def _update_rapl(self) -> None:
        """Update RAPL accumulated energy with wraparound handling."""
        current_uj = _read_rapl_energy_uj()
        if current_uj is None or self._rapl_prev_uj is None:
            return

        delta = current_uj - self._rapl_prev_uj
        if delta < 0:
            # Counter wrapped around — use absolute value as approximation
            delta = current_uj
        self._rapl_accumulated_uj += delta
        self._rapl_prev_uj = current_uj

    def _compute_gpu_energy(self) -> float | None:
        """Compute cumulative GPU energy via rectangular integration."""
        readings = self._gpu_power_readings
        if len(readings) < 2:
            return None

        total = 0.0
        for i in range(1, len(readings)):
            dt = readings[i][0] - readings[i - 1][0]
            power = readings[i - 1][1]
            total += power * dt
        return total

    def _compute_cpu_energy(self) -> float | None:
        """Compute CPU energy from RAPL counters."""
        if self._rapl_start_uj is None:
            return None
        if self._rapl_accumulated_uj == 0:
            return 0.0
        return self._rapl_accumulated_uj / 1_000_000.0

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

        gpu_energy = self._compute_gpu_energy()
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
