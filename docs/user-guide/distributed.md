# Distributed Benchmarking Patterns

Calibrax is designed for single-device profiling, but its tools integrate
with JAX's multi-device profiling ecosystem. This guide covers patterns
for benchmarking distributed workloads.

## Multi-Device Profiling with JAX Traces

Use `jax.profiler.trace()` to capture XLA execution timelines across
devices. Calibrax's `TraceLinker` wraps this with metadata linkage:

```python
from calibrax.profiling.tracing import TraceLinker

linker = TraceLinker()
with linker.trace("/tmp/traces/run_001", run_id="my_run") as ref:
    # Run distributed workload with mesh annotations
    result = pjit_function(sharded_input)

# ref.trace_dir can be stored in Run.metadata for later analysis
```

## Per-Device Timing with Aggregation

For multi-device workloads, run a `TimingCollector` per device and
aggregate results using Calibrax's analysis tools:

```python
import jax
from calibrax.profiling.timing import TimingCollector

collectors = {}
for device in jax.devices():
    collectors[device.id] = TimingCollector(
        sync_fn=lambda result: jax.numpy.array(0.0).block_until_ready(),
        warmup_iterations=2,
    )

# After collecting per-device samples, compare using analysis module
from calibrax.analysis.comparison import compare_configurations
```

## XProf Pod Viewer Integration

JAX's XProf integration produces TensorBoard-compatible trace files.
Use Calibrax's `TraceLinker` to record trace paths alongside benchmark
results, then view them in TensorBoard:

```bash
tensorboard --logdir=/tmp/traces/run_001
```

The trace files contain:
- Per-device operation timelines
- Memory allocation patterns
- Communication (all-reduce, all-gather) costs
- HLO operation breakdown

## Multi-Process Profile Merging

For multi-process JAX workloads (e.g., using `jax.distributed`), each
process writes its own trace. Use `nsys-jax-combine` to merge profiles:

```bash
nsys-jax-combine --output merged_profile.nsys-rep trace_rank_*.nsys-rep
```

Then link the merged profile path in Calibrax's Store:

```python
from calibrax.storage.store import Store
from calibrax.core.models import Metric, Point, Run

store = Store("/tmp/calibrax-distributed-demo")
run = Run(
    points=(Point(name="distributed", scenario="train",
                  metrics={"time": Metric(value=1.5)}),),
)
store.save(run)
latest = store.latest()
# Store trace reference in run metadata
```

## Resource Monitoring Across Devices

`ResourceMonitor` tracks CPU/memory for the local process. For GPU metrics
across multiple GPUs, use `GPUMemoryProfiler` with device selection or
combine with NVML queries:

```python
from calibrax.profiling.resources import ResourceMonitor
from calibrax.profiling.gpu import GPUMemoryProfiler

profiler = GPUMemoryProfiler()
with ResourceMonitor(gpu_profiler=profiler) as mon:
    # Run distributed workload
    pass

summary = mon.summary
```
