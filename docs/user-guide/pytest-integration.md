# pytest Integration Patterns

Calibrax can be used alongside `pytest-benchmark` for a layered
benchmarking strategy: pytest-benchmark for quick CI checks, Calibrax
for full profiling pipelines.

## Basic pytest Fixture

Wrap Calibrax's `TimingCollector` in a pytest fixture:

```python
import pytest
from calibrax.profiling.timing import TimingCollector


@pytest.fixture
def timing_collector():
    """Provide a TimingCollector with 2 warmup iterations."""
    return TimingCollector(warmup_iterations=2)


def test_matmul_throughput(timing_collector):
    import jax.numpy as jnp

    data = [jnp.ones((64, 64)) for _ in range(20)]
    sample = timing_collector.measure_iteration(iter(data), num_batches=20)

    assert sample.num_batches == 20
    assert sample.warmup_batches_excluded == 2
    mean_time = sum(sample.per_batch_times) / len(sample.per_batch_times)
    assert mean_time < 1.0  # Sanity check
```

## Store-Backed Benchmarking Fixture

Save benchmark results to a Calibrax `Store` for trend tracking:

```python
import pytest
from pathlib import Path
from calibrax.storage.store import Store
from calibrax.core.models import Run, Point, Metric


@pytest.fixture
def benchmark_store(tmp_path):
    """Provide a temporary Store for benchmark results."""
    return Store(tmp_path / "benchmarks")


def test_training_step_speed(benchmark_store):
    import time

    start = time.perf_counter()
    # ... run training step ...
    elapsed = time.perf_counter() - start

    run = Run(
        points=(
            Point(
                name="training_step",
                scenario="speed",
                tags={"framework": "jax"},
                metrics={"wall_time_sec": Metric(value=elapsed)},
            ),
        ),
    )
    benchmark_store.save(run)
```

## Combining with pytest-benchmark

Use pytest-benchmark for quick timing and Calibrax for detailed analysis:

```python
def test_forward_pass(benchmark):
    import jax.numpy as jnp

    x = jnp.ones((32, 128))

    def forward():
        return jnp.dot(x, x.T).block_until_ready()

    # Quick benchmark via pytest-benchmark
    result = benchmark(forward)

    # Detailed profiling via Calibrax (optional, for CI)
    from calibrax.profiling.timing import TimingCollector

    collector = TimingCollector(
        sync_fn=lambda: jnp.array(0.0).block_until_ready(),
        warmup_iterations=3,
    )
    sample = collector.measure_iteration(
        iter(range(50)),
        num_batches=50,
        count_fn=lambda _: (forward(), 1)[1],
    )
    assert len(sample.per_batch_times) == 47  # 50 - 3 warmup
```

## CI Regression Checking

Use Calibrax's `CIGuard` in pytest for automated regression detection:

```python
import pytest
from calibrax.ci.guard import CIGuard
from calibrax.storage.store import Store


@pytest.fixture
def ci_guard(tmp_path):
    store = Store(tmp_path / "benchmarks")
    return CIGuard(store, threshold=0.05)


def test_no_regressions(ci_guard):
    result = ci_guard.check()
    if not result.passed:
        failures = [
            f"{r.metric} on {r.point_name}: {r.delta_pct:+.1f}%"
            for r in result.regressions
        ]
        pytest.fail(f"Performance regressions detected:\n" + "\n".join(failures))
```

## Resource Monitoring in Tests

Monitor resource usage during test execution:

```python
from calibrax.profiling.resources import ResourceMonitor


def test_memory_bounded(benchmark):
    with ResourceMonitor(sample_interval_sec=0.05) as mon:
        # ... run workload ...
        pass

    summary = mon.summary
    assert summary.peak_rss_mb < 2048, f"Peak RSS too high: {summary.peak_rss_mb:.0f} MB"
    assert summary.memory_growth_mb < 100, "Memory leak detected"
```
