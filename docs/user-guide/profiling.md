# Profiling Workloads

Calibrax provides a suite of profiling tools that can be used independently or
composed into a full benchmark result: wall-clock timing (with warmup exclusion and
compilation measurement), resource monitoring (CPU, memory, GPU clock/power), GPU
memory analysis, energy measurement, FLOP counting, hardware detection, roofline
analysis, compilation profiling, complexity analysis, XLA trace linking, and carbon
emissions tracking.

## Timing

`TimingCollector` measures wall-clock time for an iterator-based workload. Pass a
`sync_fn` to ensure GPU operations complete before each timestamp.

```python
import jax
import jax.numpy as jnp
from calibrax.profiling.timing import TimingCollector

collector = TimingCollector(
    sync_fn=lambda batch: jax.block_until_ready(batch),
    warmup_iterations=2,  # exclude first 2 batches from per_batch_times
)
sample = collector.measure_iteration(
    iterator=iter(data_loader),
    num_batches=100,
    count_fn=lambda batch: batch["image"].shape[0],
)

print(f"Wall clock: {sample.wall_clock_sec:.3f}s")
print(f"Elements processed: {sample.num_elements}")
print(f"First batch: {sample.first_batch_time:.4f}s (includes JIT compilation)")
print(f"Warmup excluded: {sample.warmup_batches_excluded} batches")
print(f"Median batch: {sorted(sample.per_batch_times)[len(sample.per_batch_times)//2]:.4f}s")
```

The returned `TimingSample` contains:

- `wall_clock_sec` — total elapsed time (including warmup)
- `per_batch_times` — per-iteration timings with warmup batches excluded
- `first_batch_time` — first iteration time, typically higher due to JIT
- `num_batches` — total iterations consumed (including warmup)
- `num_elements` — total element count (via `count_fn`)
- `warmup_batches_excluded` — number of leading batches excluded from timing
- `compilation_time_sec` — JIT compilation time, if measured separately

### Measuring Compilation Time

Measure XLA compilation overhead separately from execution:

```python
import jax.numpy as jnp

collector = TimingCollector()
comp_time = collector.measure_compilation_time(
    lambda x: jnp.dot(x, x.T),
    jnp.ones((64, 64)),
)
print(f"Compilation: {comp_time:.3f}s")
```

### Serialization

`TimingSample` supports `to_dict()` and `from_dict()` for JSON-compatible
round-trip serialization:

```python
from calibrax.profiling.timing import TimingSample

d = sample.to_dict()
restored = TimingSample.from_dict(d)
```

!!! tip "GPU Synchronization"

    Without `sync_fn`, GPU timing measures only dispatch time, not actual
    computation. Always pass `sync_fn=lambda batch: jax.block_until_ready(batch)`
    for accurate GPU benchmarks.

## Resource Monitoring

`ResourceMonitor` runs a background thread that samples CPU and memory usage
at a configurable interval. Use it as a context manager:

```python
from calibrax.profiling.resources import ResourceMonitor

with ResourceMonitor(sample_interval_sec=0.1) as monitor:
    # Run your workload here
    train(model, data)

summary = monitor.summary
print(f"Peak RSS: {summary.peak_rss_mb:.1f} MB")
print(f"Mean RSS: {summary.mean_rss_mb:.1f} MB")
print(f"Memory growth: {summary.memory_growth_mb:.1f} MB")
print(f"Duration: {summary.duration_sec:.2f}s")
print(f"Samples collected: {summary.num_samples}")
```

To include the GPU, pass a `GPUProfilerProtocol` source. `NvmlDevice` reads memory,
compute utilization, clocks and power through NVIDIA's NVML; it lives in
`calibrax.profiling.nvml` and needs the `cuda12` extra:

```python
from calibrax.profiling.nvml import NvmlDevice

with NvmlDevice(0) as gpu, ResourceMonitor(gpu_profiler=gpu) as monitor:
    train(model, data)

summary = monitor.summary
if summary.mean_gpu_util is not None:
    print(f"Mean GPU utilization: {summary.mean_gpu_util:.1f}%")
if summary.peak_gpu_mem_mb is not None:
    print(f"Peak GPU memory: {summary.peak_gpu_mem_mb:.1f} MB")
```

Without NVML, `GPUMemoryProfiler` reads a GPU's memory from JAX's device statistics; JAX
reports no utilization, clocks or power, so those readings are `None`. An NVML error other
than an unsupported reading is raised when the monitor exits.

## GPU Memory Analysis

`GPUMemoryProfiler` reads GPU memory at any point, and `analyze_memory_pattern` turns a
series of readings into suggestions. `MemoryOptimizer`
analyzes the memory footprint of an entire pipeline, measuring baseline, peak,
and retained memory.

```python
from calibrax.profiling.gpu import GPUMemoryProfiler, MemoryOptimizer

# Quick snapshot; None without a GPU
memory = GPUMemoryProfiler().memory()
if memory is not None:
    print(f"GPU memory used: {memory.used_mb:.1f} of {memory.total_mb:.1f} MB")

# Full pipeline analysis
optimizer = MemoryOptimizer()
analysis = optimizer.analyze_pipeline_memory(pipeline_fn, sample_data)
if analysis is not None:
    print(f"Baseline: {analysis.baseline_memory_mb:.1f} MB")
    print(f"Peak: {analysis.peak_memory_mb:.1f} MB")
    print(f"Efficiency: {analysis.memory_efficiency:.1%}")
    for suggestion in analysis.suggestions:
        print(f"  - {suggestion}")
```

`AdaptiveOperation` auto-detects hardware and optimizes tensor shapes:

```python
from calibrax.profiling.gpu import AdaptiveOperation

adaptive = AdaptiveOperation()
print(f"Platform: {adaptive.config.platform}")
print(f"Precision: {adaptive.config.precision}")

optimized_shapes = adaptive.optimize_shapes((32, 128), (128, 64))
```

## Energy Monitoring

`EnergyMonitor` integrates the power draw of a GPU source such as `NvmlDevice`, and reads
CPU energy from the Linux RAPL counter, during a workload. Use it as a context manager:

```python
from calibrax.profiling.energy import EnergyMonitor
from calibrax.profiling.nvml import NvmlDevice

with NvmlDevice(0) as gpu, EnergyMonitor(sample_interval_sec=0.1, gpu=gpu) as monitor:
    train(model, data)

energy_summary = monitor.summary
print(f"Duration: {energy_summary.duration_sec:.2f}s")
if energy_summary.total_gpu_energy_joules is not None:
    print(f"GPU energy: {energy_summary.total_gpu_energy_joules:.2f} J")
if energy_summary.total_cpu_energy_joules is not None:
    print(f"CPU energy: {energy_summary.total_cpu_energy_joules:.2f} J")
if energy_summary.mean_gpu_power_watts is not None:
    print(f"Mean GPU power: {energy_summary.mean_gpu_power_watts:.1f} W")
```

!!! note

    Without a GPU source the GPU fields are `None`. RAPL's counter is readable only by
    root on most Linux systems; where it is not readable the CPU fields are `None`. A
    counter wraparound is measured against the kernel's `max_energy_range_uj`.

## FLOP Counting

`FlopsCounter` lowers a JAX function from the shapes of its example arguments and
reads XLA's cost analysis of the result, the same estimate `flax.nnx.tabulate`
reports. Nothing is executed and no data is copied:

```python
import jax.numpy as jnp
from calibrax.profiling.flops import FlopsCounter

def matmul_workload(x, w):
    return jnp.dot(x, w)

counter = FlopsCounter()
result = counter.count(matmul_workload, jnp.ones((64, 128)), jnp.ones((128, 32)))

print(f"Total FLOPs: {result.total_flops:,}")
print(f"Transcendentals: {result.transcendentals:,}")
```

XLA's conventions: a matmul `(M, K) @ (K, N)` is `2 * M * K * N`, an elementwise op
is one FLOP per output element, `sin`, `exp` and friends are transcendentals rather
than FLOPs, a conditional costs its most expensive branch, and a loop body is
counted once because the trip count is not part of the HLO. A function containing a
custom call XLA has no cost model for (`jax.pure_callback`, some linear-algebra
kernels) raises `FlopsUnavailableError`.

!!! tip "NNX Models"

    Pass the NNX state as an argument, or close over the module; both lower. For a
    per-module table use `flax.nnx.tabulate(model, *args, compute_flops=True)`,
    which reports the same estimate per submodule.

## Hardware Detection

`HARDWARE_SPECS` holds each accelerator's dense BF16 peak and memory bandwidth from
the vendor's specification (A100 in its four variants, H100 SXM and PCIe, RTX 4090,
TPU v4, v5e, v5p and v6e, and a CPU stand-in). `detect_hardware_specs()` names the chip
by the `device_kind` JAX reports and returns its `HardwareSpec`, or `None` for an
accelerator the table does not hold; pass a `HardwareSpec` for one.

```python
from calibrax.profiling.hardware import HARDWARE_SPECS, HardwareSpec, detect_hardware_specs

spec = detect_hardware_specs()  # HardwareSpec(name="rtx_4090", ...) on an RTX 4090
if spec is not None:
    print(f"{spec.name}: {spec.peak_flops:.3g} FLOP/s, {spec.memory_bandwidth:.3g} B/s, "
          f"ridge point {spec.critical_intensity:.0f} FLOPs/byte")

a100 = HARDWARE_SPECS["a100_sxm4_80gb"]
l4 = HardwareSpec(name="l4", peak_flops=121.0e12, memory_bandwidth=300.0e9)
```

`RooflineAnalyzer` raises `UnknownHardwareError` when it has no spec for the chip in use.

## Roofline Analysis

`RooflineAnalyzer` compares a workload's arithmetic intensity against the
hardware roofline to determine whether it is compute-bound or memory-bound:

```python
import jax.numpy as jnp
from calibrax.profiling.roofline import RooflineAnalyzer

def matmul_fn(x):
    return jnp.dot(x, x.T)

analyzer = RooflineAnalyzer()
result = analyzer.analyze_operation(matmul_fn, [jnp.ones((64, 64))])

print(f"Arithmetic intensity: {result.arithmetic_intensity:.2f} FLOP/byte")
print(f"Bound: {result.bottleneck}")
for rec in result.recommendations:
    print(f"  - {rec}")
```

## Compilation Profiling

`CompilationProfiler` analyzes JIT compilation overhead, shape consistency,
and XLA optimization effectiveness:

```python
from calibrax.profiling.compilation import CompilationProfiler

profiler = CompilationProfiler()

# Instrument a JIT-compiled function (returns a callable wrapper)
instrumented = profiler.profile_jit_compilation(fn)
result = instrumented(*sample_args)

# Get compilation report
report = profiler.get_result()
print(f"Cache hit rate: {report.cache_hit_rate:.2%}")

# Analyze XLA optimization effectiveness
xla_result = profiler.estimate_xla_optimization(fn, *sample_args)
print(f"Optimization score: {xla_result.optimization_score:.2f}")
```

## Complexity Analysis

`analyze_complexity()` examines parameter counts, memory requirements, and
computational cost for Flax NNX modules:

```python
from calibrax.profiling.complexity import analyze_complexity

result = analyze_complexity(model, (4, 128))  # input_shape tuple, not data
print(f"Total parameters: {result.total_parameters:,}")
print(f"Memory (MB): {result.parameter_memory_mb:.1f}")
print(f"Estimated operations: {result.total_estimated_operations:,}")
```

## XLA Trace Linking

`TraceLinker` wraps `jax.profiler.trace()` and records the trace directory
alongside benchmark metadata for later analysis in TensorBoard:

```python
from calibrax.profiling.tracing import TraceLinker

linker = TraceLinker()
with linker.trace("temp/doc-examples/traces/run_001") as ref:
    # Run workload — XLA profiling is active
    train_step(model, batch)

print(f"Trace saved to: {ref.trace_dir}")
# Open with: tensorboard --logdir temp/doc-examples/traces/run_001
```

## Carbon Emissions Tracking

!!! warning "Import Path"

    `CarbonTracker` is **not** re-exported from `calibrax.profiling` to avoid
    loading codecarbon at import time. Import it directly:

    ```python
    from calibrax.profiling.carbon import CarbonTracker
    ```

!!! warning "Optional Dependency"

    Requires codecarbon: `uv pip install "calibrax[codecarbon]"`

`CarbonTracker` measures energy consumption and CO2 emissions during a
workload via CodeCarbon:

```python
from calibrax.profiling.carbon import CarbonTracker

with CarbonTracker(country_iso_code="USA") as tracker:
    train(model, data)

result = tracker.result()
print(f"Emissions: {result.emissions_kg_co2:.4f} kg CO2")
print(f"Energy: {result.energy_consumed_kwh:.4f} kWh")
print(f"Duration: {result.duration_sec:.1f}s")
```

## Composing a BenchmarkResult

Combine profiling outputs into a single `BenchmarkResult` for storage and analysis:

```python
from pathlib import Path
from calibrax.core.result import BenchmarkResult
from calibrax.core.models import Metric

result = BenchmarkResult(
    name="forward_pass",
    domain="training",
    timing=sample,          # from TimingCollector
    resources=summary,      # from ResourceMonitor
    metrics={
        "throughput": Metric(value=sample.num_elements / sample.wall_clock_sec),
    },
)

result.save(Path("temp/doc-examples/results/forward_pass.json"))
```

## Best Practices

- Always use `sync_fn` for GPU timing — without it, measurements reflect dispatch
  time, not actual compute time
- Set `ResourceMonitor` sample interval to at least 10x shorter than the expected
  workload duration to get meaningful statistics
- Run workloads once before timing to warm up JIT compilation, or exclude the
  first batch time from throughput calculations
- Use `MemoryOptimizer` during development to catch memory leaks early

## Next Steps

<div class="grid cards" markdown>

-   :material-chart-bell-curve:{ .lg .middle } **Statistical Analysis**

    ---

    Apply bootstrap CI and outlier detection to your timing samples

    [:octicons-arrow-right-24: Statistics](statistics.md)

-   :material-database:{ .lg .middle } **Storage & Baselines**

    ---

    Save profiling results and establish performance baselines

    [:octicons-arrow-right-24: Storage](storage.md)

</div>
