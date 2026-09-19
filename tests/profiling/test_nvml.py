"""NvmlDevice: GPU readings through NVIDIA's NVML, the nvidia-ml-py integration."""

from __future__ import annotations

import types
from collections.abc import Iterator
from typing import NoReturn
from unittest.mock import patch

import pytest
from substrax.testing import run_python

from calibrax.profiling.nvml import NvmlDevice
from calibrax.profiling.resources import GpuClocks, GpuMemory, GpuPower, GPUProfilerProtocol


_NOT_SUPPORTED = 3
_UNKNOWN = 999


class _FakeNvmlError(Exception):
    def __init__(self, value: int) -> None:
        super().__init__(value)
        self.value = value


def _unsupported(*_args: object) -> NoReturn:
    raise _FakeNvmlError(_NOT_SUPPORTED)


def _fake_nvml() -> types.SimpleNamespace:
    """The NVML calls NvmlDevice makes, answering for one GPU."""
    calls: list[str] = []
    return types.SimpleNamespace(
        calls=calls,
        NVMLError=_FakeNvmlError,
        NVML_ERROR_NOT_SUPPORTED=_NOT_SUPPORTED,
        NVML_CLOCK_GRAPHICS=0,
        NVML_CLOCK_MEM=2,
        nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetHandleByIndex=lambda index: f"gpu{index}",
        nvmlDeviceGetMemoryInfo=lambda _h: types.SimpleNamespace(used=2 * 2**30, total=8 * 2**30),
        nvmlDeviceGetUtilizationRates=lambda _h: types.SimpleNamespace(gpu=73, memory=40),
        nvmlDeviceGetClockInfo=lambda _h, kind: {0: 1500, 2: 10_000}[kind],
        nvmlDeviceGetPowerUsage=lambda _h: 240_000,
        nvmlDeviceGetPowerManagementLimit=lambda _h: 450_000,
    )


@pytest.fixture
def fake_nvml() -> Iterator[types.SimpleNamespace]:
    fake = _fake_nvml()
    with patch("calibrax.profiling.nvml.pynvml", fake):
        yield fake


def test_readings_are_converted_to_calibrax_units(fake_nvml: types.SimpleNamespace) -> None:
    with NvmlDevice(0) as device:
        assert device.memory() == GpuMemory(used_mb=2048.0, total_mb=8192.0)
        assert device.utilization() == 73.0
        assert device.clocks() == GpuClocks(graphics_mhz=1500.0, memory_mhz=10_000.0)
        assert device.power() == GpuPower(draw_w=240.0, limit_w=450.0)
    assert fake_nvml.calls == ["init", "shutdown"]


def test_nvml_is_initialised_once_however_many_readings(fake_nvml: types.SimpleNamespace) -> None:
    with NvmlDevice() as device:
        for _ in range(5):
            device.power()
    assert fake_nvml.calls.count("init") == 1


@pytest.mark.usefixtures("fake_nvml")
def test_a_device_satisfies_the_gpu_profiler_protocol() -> None:
    with NvmlDevice() as device:
        assert isinstance(device, GPUProfilerProtocol)


@pytest.mark.parametrize(
    ("query", "reading"),
    [
        ("nvmlDeviceGetMemoryInfo", "memory"),
        ("nvmlDeviceGetUtilizationRates", "utilization"),
        ("nvmlDeviceGetClockInfo", "clocks"),
        ("nvmlDeviceGetPowerUsage", "power"),
        ("nvmlDeviceGetPowerManagementLimit", "power"),
    ],
)
def test_a_reading_the_gpu_does_not_support_is_none(query: str, reading: str) -> None:
    fake = _fake_nvml()
    setattr(fake, query, _unsupported)
    with patch("calibrax.profiling.nvml.pynvml", fake), NvmlDevice() as device:
        assert getattr(device, reading)() is None
        others = {"memory", "utilization", "clocks", "power"} - {reading}
        assert all(getattr(device, other)() is not None for other in others)


def test_any_other_nvml_error_is_raised() -> None:
    def failing_utilization(_handle: object) -> types.SimpleNamespace:
        raise _FakeNvmlError(_UNKNOWN)

    fake = _fake_nvml()
    fake.nvmlDeviceGetUtilizationRates = failing_utilization
    with (
        patch("calibrax.profiling.nvml.pynvml", fake),
        NvmlDevice() as device,
        pytest.raises(_FakeNvmlError),
    ):
        device.utilization()


def test_importing_without_nvidia_ml_py_names_the_extra() -> None:
    result = run_python(
        "import sys; sys.modules['pynvml'] = None\n"
        "try:\n"
        "    import calibrax.profiling.nvml\n"
        "except ImportError as error:\n"
        "    print(error)\n",
        timeout=120,
    )

    assert "calibrax[cuda12]" in result.stdout


def test_profiling_imports_without_nvidia_ml_py() -> None:
    result = run_python(
        "import sys; sys.modules['pynvml'] = None\nimport calibrax.profiling\nprint('ok')\n",
        timeout=120,
    )

    assert result.stdout.strip() == "ok"


def test_a_device_nvml_cannot_open_shuts_nvml_down(fake_nvml: types.SimpleNamespace) -> None:
    def no_such_gpu(_index: int) -> str:
        raise _FakeNvmlError(_UNKNOWN)

    fake_nvml.nvmlDeviceGetHandleByIndex = no_such_gpu

    with pytest.raises(_FakeNvmlError):
        NvmlDevice(99)
    assert fake_nvml.calls == ["init", "shutdown"]
