# calibrax.profiling.timing

Wall-clock timing collector with GPU synchronization support.
`TimingCollector.measure_iteration()` consumes an iterator and records
per-batch times, total wall clock, and element counts. Supports
warm-up iteration exclusion and JIT compilation time measurement.

::: calibrax.profiling.timing
    options:
      show_root_heading: false

## calibrax.profiling.timing_records

`TimingSample` and `CallTiming`, the records the collector and `time_calls` return. They
import without JAX, so reading a stored result does not load it.

::: calibrax.profiling.timing_records
    options:
      show_root_heading: false
