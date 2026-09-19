"""Types for the parts of NVIDIA's nvidia-ml-py (the ``pynvml`` module) calibrax uses.

nvidia-ml-py ships no type information. Signatures follow its ``pynvml.py``: ``NVMLError`` sets
``value``, the NVML return code, in ``__new__``; device handles are opaque ctypes pointers;
memory is in bytes, clocks in MHz and power in milliwatts.
"""

from typing import NewType

_Handle = NewType("_Handle", object)

NVML_ERROR_NOT_SUPPORTED: int
NVML_CLOCK_GRAPHICS: int
NVML_CLOCK_MEM: int

class NVMLError(Exception):
    value: int
    def __init__(self, value: int) -> None: ...

class c_nvmlMemory_t:
    total: int
    free: int
    used: int

class c_nvmlUtilization_t:
    gpu: int
    memory: int

def nvmlInit() -> None: ...
def nvmlShutdown() -> None: ...
def nvmlDeviceGetHandleByIndex(index: int) -> _Handle: ...
def nvmlDeviceGetMemoryInfo(handle: _Handle) -> c_nvmlMemory_t: ...
def nvmlDeviceGetUtilizationRates(handle: _Handle) -> c_nvmlUtilization_t: ...
def nvmlDeviceGetClockInfo(handle: _Handle, type: int) -> int: ...
def nvmlDeviceGetPowerUsage(handle: _Handle) -> int: ...
def nvmlDeviceGetPowerManagementLimit(handle: _Handle) -> int: ...
