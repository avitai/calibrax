# Profiling: what Calibrax owns

Calibrax is the one home for benchmark measurement in the Avitai JAX stack. This page
states the boundary so that the sibling packages do not grow copies.

## Owned by `calibrax.profiling`

| Concern | Module | Notes |
| --- | --- | --- |
| Wall-clock timing with device synchronisation and warm-up | `timing` | `TimingCollector`, `TimingSample` |
| Host and GPU resource sampling | `resources`, `gpu` | `ResourceMonitor`, `GPUMemoryProfiler`, `MemoryOptimizer` |
| Energy and carbon | `energy`, `carbon` | NVML and RAPL sampling; CodeCarbon-backed emissions |
| FLOP counting | `flops` | XLA's cost analysis of the lowered function, the same estimate `flax.nnx.tabulate` reports |
| Roofline analysis and the accelerator spec table | `roofline`, `hardware` | `RooflineAnalyzer`, `HARDWARE_SPECS`, `detect_hardware_specs` |
| Compilation profiling and XLA optimisation analysis | `compilation` | `CompilationProfiler` |
| Analytic complexity estimates | `complexity` | `analyze_complexity` |
| Linking `jax.profiler` traces to stored runs | `tracing` | `TraceLinker`, `TraceReference` |

## Not owned here

| Concern | Where it lives | Why |
| --- | --- | --- |
| Device identity, placement and meshes | `substrax.devices`, `substrax.mesh` | Infrastructure shared by every package; `detect_hardware_specs` reads the platform from there and maps it to the spec table |
| Data-pipeline throughput, goodput and prefetching | `datarax.performance` | Measures the pipeline, not a function; datarax composes Calibrax's timing for the numbers it stores |
| Training-loop harnesses that schedule profiling | `opifex.benchmarking.profiling` | Composes `CompilationProfiler`, `FlopsCounter` and `RooflineAnalyzer` around a training step |
| Reading a captured trace | [XProf](https://github.com/openxla/xprof) | Calibrax captures and links traces; XProf visualises them |
| Fixed-kernel microbenchmarks of an accelerator | [accelerator-microbenchmarks](https://github.com/AI-Hypercomputer/accelerator-microbenchmarks) | One way to calibrate `HARDWARE_SPECS`; not a benchmark of the caller's code |

## Rules for contributors

- A new measurement that any two packages would want goes here, once, with its test.
- A sibling package that needs a number from a profiler imports the profiler; it does
  not re-implement the estimate (opifex's model-based FLOP estimator was replaced by
  `FlopsCounter` for this reason).
- The spec table is data: extend `HARDWARE_SPECS` with a measured entry rather than
  hard-coding peaks in a consumer.
