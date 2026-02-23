# Storage & Baselines

The `Store` class provides a JSON-per-run file backend for persisting benchmark
runs, managing baselines, extracting trends, and ingesting external data.

## Directory Layout

A store organizes files in a flat structure:

```
benchmark-data/
├── config.json          # Metric definitions
├── runs/
│   ├── a1b2c3d4e5f6.json
│   ├── f6e5d4c3b2a1.json
│   └── ...
└── baselines/
    └── main.json        # Symlink or copy of the baseline run
```

## Creating a Store

```python
from pathlib import Path
from calibrax.storage.store import Store
from calibrax.core.models import MetricDef, MetricDirection, Metric, Point, Run

store = Store(Path("/tmp/calibrax-storage-demo"))
```

The store creates its directory structure automatically on first use.

## Saving and Loading Runs

```python
run = Run(
    points=(Point(name="fwd", scenario="train",
                  metrics={"throughput": Metric(value=1200.0)}),),
    metric_defs={"throughput": MetricDef(
        name="throughput", unit="samples/sec", direction=MetricDirection.HIGHER)},
)

# Save a run — stored as {root}/runs/{run.id}.json
path = store.save(run)
print(f"Saved to {path}")

# Load by ID
loaded = store.load(run.id)

# Get the most recent run
latest = store.latest()

# List all runs (newest first)
all_runs = store.list_runs()

# Filter by branch
main_runs = store.list_runs(branch="main")
```

## Querying by Tags

Find runs whose points match specific tag values:

```python
# Find all runs with a "framework=flax" tag
flax_runs = store.query(framework="flax")

# Multiple tags are ANDed
filtered = store.query(framework="flax", scenario="training")
```

## Baseline Management

Baselines are the reference point for regression detection. A store supports
one active baseline at a time:

```python
# Set a run as the baseline
store.set_baseline(run.id)

# Retrieve the current baseline (None if not set)
baseline = store.get_baseline()

# Update the baseline after a successful CI run
# Note: latest() raises FileNotFoundError if the store is empty
latest_passing_run = store.latest()
store.set_baseline(latest_passing_run.id)
```

!!! tip "Baseline Strategy"

    Set the baseline after a successful CI check passes. This ensures
    regressions are always measured against the last known-good state.

## Trend Extraction

Extract a time series of metric values across runs for a specific point and tags:

```python
trend = store.extract_trend(
    metric="throughput",
    point_name="forward_pass",
    tags={"framework": "flax"},
    n_runs=20,  # limit to last 20 runs
)

print(f"Metric: {trend.metric}, Point: {trend.point_name}")
for tp in trend.points:
    print(f"  {tp.run_id}: {tp.value:.1f} "
          f"[{tp.lower or 0:.1f}, {tp.upper or 0:.1f}]")
```

The returned `TrendSeries` contains `TrendPoint` objects with `run_id`,
`timestamp`, `value`, and optional `lower`/`upper` bounds and `commit` hash.

## Ingesting External Data

Import benchmark results from external JSON files:

```python
import json
from pathlib import Path
from calibrax.core.models import Metric, Point, Run

# Create a sample external results file
external_run = Run(
    points=(Point(name="imported", scenario="test",
                  metrics={"latency": Metric(value=0.42)}),),
)
ext_path = Path("temp/doc-examples/external-results.json")
ext_path.parent.mkdir(parents=True, exist_ok=True)
ext_path.write_text(json.dumps(external_run.to_dict()))

# Ingest the file into the store
run = store.ingest(ext_path)
print(f"Ingested as run {run.id}")
```

The `ingest()` method reads the file, creates a `Run` from the data, saves it
to the store, and returns the resulting `Run` object.

## Best Practices

- Keep one store per project or benchmark suite
- Set baselines explicitly rather than relying on "latest" — this prevents
  a regression from becoming the new baseline
- Use `extract_trend()` to visualize long-term performance changes before
  investigating specific regressions
- Use `query()` to compare runs across branches or configurations

## Next Steps

<div class="grid cards" markdown>

-   :material-alert-decagram:{ .lg .middle } **Regression Detection**

    ---

    Compare new runs against stored baselines

    [:octicons-arrow-right-24: Regressions](regressions.md)

-   :material-export:{ .lg .middle } **Exporting Results**

    ---

    Export stored runs to W&B or generate publication tables

    [:octicons-arrow-right-24: Exporters](exporters.md)

</div>
