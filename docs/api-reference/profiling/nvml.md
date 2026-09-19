# calibrax.profiling.nvml

GPU memory, compute utilization, clocks and power through NVIDIA's NVML. This module needs
the `cuda12` extra (nvidia-ml-py); importing it without that raises `ImportError` naming the
extra. `NvmlDevice` satisfies `GPUProfilerProtocol` and `GpuPowerSource`, so it drives both
`ResourceMonitor` and `EnergyMonitor`.

::: calibrax.profiling.nvml
    options:
      show_root_heading: false
