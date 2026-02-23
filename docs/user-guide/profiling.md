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

To include GPU utilization, pass a `GPUProfilerProtocol`-compatible object:

```python
from calibrax.profiling.gpu import GPUMemoryProfiler

gpu_profiler = GPUMemoryProfiler()
with ResourceMonitor(gpu_profiler=gpu_profiler) as monitor:
    train(model, data)

summary = monitor.summary
if summary.mean_gpu_util is not None:
    print(f"Mean GPU utilization: {summary.mean_gpu_util:.1f}%")
if summary.peak_gpu_mem_mb is not None:
    print(f"Peak GPU memory: {summary.peak_gpu_mem_mb:.1f} MB")
```

## GPU Memory Analysis

`GPUMemoryProfiler` checks GPU memory usage at any point. `MemoryOptimizer`
analyzes the memory footprint of an entire pipeline, measuring baseline, peak,
and retained memory.

```python
from calibrax.profiling.gpu import GPUMemoryProfiler, MemoryOptimizer

# Quick snapshot
profiler = GPUMemoryProfiler()
usage = profiler.get_memory_usage()
print(f"GPU memory used: {usage.get('gpu_memory_used_mb', 0):.1f} MB")

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

`EnergyMonitor` tracks GPU power draw (via NVML) and CPU energy consumption
(via RAPL) during a workload. Use it as a context manager:

```python
from calibrax.profiling.energy import EnergyMonitor

with EnergyMonitor(sample_interval_sec=0.1) as monitor:
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

    Energy monitoring requires hardware support: NVML for GPU power and
    Intel RAPL for CPU energy. On unsupported hardware, energy fields will
    be `None`.

## FLOP Counting

`FlopsCounter` analyzes a JAX function's computation graph (jaxpr) to count
floating-point operations:

```python
import jax.numpy as jnp
from calibrax.profiling.flops import FlopsCounter

def matmul_workload(x, w):
    return jnp.dot(x, w)

counter = FlopsCounter()
result = counter.count(matmul_workload, jnp.ones((64, 128)), jnp.ones((128, 32)))

print(f"Total FLOPs: {result.total_flops:,}")
print(f"Operations: {result.num_operations}")
for op, count in result.flops_by_operation.items():
    print(f"  {op}: {count:,}")
```

!!! tip "NNX Models"

    `FlopsCounter` works with pure JAX functions. For Flax NNX models,
    use `flax.nnx.tabulate(model, *args, compute_flops=True)` instead,
    which handles the NNX state and `Rngs` automatically.

## Hardware Detection

`detect_hardware_specs()` auto-detects the active JAX backend and returns
reference performance specs. `HARDWARE_SPECS` contains peak FLOP/s and memory
bandwidth values for common accelerators.

```python
from calibrax.profiling.hardware import detect_hardware_specs, HARDWARE_SPECS

specs = detect_hardware_specs()
print(f"Peak FLOP/s: {specs.get('peak_flops', 'N/A')}")
print(f"Memory BW (B/s): {specs.get('memory_bandwidth', 'N/A')}")

# Reference specs for specific hardware
a100_specs = HARDWARE_SPECS["a100_80g"]
```

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

!!! note

    Carbon tracking requires the `codecarbon` optional dependency:
    `uv pip install "calibrax[codecarbon]"`

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
