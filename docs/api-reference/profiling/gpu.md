# calibrax.profiling.gpu

GPU memory profiling. Includes `GPUMemoryProfiler` for memory readings from JAX's device
statistics, `analyze_memory_pattern` for suggestions from a series of readings, and
`MemoryOptimizer` for pipeline memory analysis. An accelerator's ridge point and tensor-core
shapes are in `calibrax.profiling.hardware`.

::: calibrax.profiling.gpu
    options:
      show_root_heading: false
