# Exporting Results

Calibrax supports exporting benchmark results to Weights & Biases (W&B) and
generating publication-ready plots and tables. Both exporters implement the
`Exporter` ABC, and custom exporters can be built by subclassing it.

## Exporter Interface

All exporters implement two methods:

```python
from abc import ABC
from calibrax.exporters.base import Exporter
from calibrax.core.models import Run

class Exporter(ABC):
    def export_run(self, run: Run) -> str:
        """Export a single run. Returns a URL or identifier."""
        ...

    def export_analysis(self, run: Run, baseline: Run | None = None) -> None:
        """Export analysis comparing run against an optional baseline."""
        ...
```

## Weights & Biases

!!! warning "Import Path"

    `WandBExporter` is **not** re-exported from `calibrax.exporters` to avoid
    loading wandb at import time. Import it directly:

    ```python
    from calibrax.exporters.wandb import WandBExporter
    ```

!!! warning "Optional Dependency"

    Requires wandb: `uv pip install "calibrax[wandb]"`

### Basic Usage

```python
from calibrax.exporters.wandb import WandBExporter
from calibrax.core.models import Metric, MetricDef, MetricDirection, Point, Run

# Create sample runs for export
run = Run(
    points=(Point(name="forward_pass", scenario="training",
                  tags={"framework": "flax"},
                  metrics={"throughput": Metric(value=1200.0),
                           "latency": Metric(value=0.8)}),),
    metric_defs={
        "throughput": MetricDef(name="throughput", unit="samples/sec",
                                direction=MetricDirection.HIGHER),
        "latency": MetricDef(name="latency", unit="ms",
                             direction=MetricDirection.LOWER),
    },
)
baseline_run = Run(
    points=(Point(name="forward_pass", scenario="training",
                  tags={"framework": "flax"},
                  metrics={"throughput": Metric(value=1100.0),
                           "latency": Metric(value=0.9)}),),
    metric_defs=run.metric_defs,
)

exporter = WandBExporter(
    project="my-benchmarks",
    entity="my-team",        # optional
    tags=["nightly", "gpu"],  # optional
)

# Check authentication
if not exporter.check_auth():
    print("Run 'wandb login' first")

# Export a run
url = exporter.export_run(run)
print(f"View at: {url}")

# Export analysis with baseline comparison
exporter.export_analysis(run, baseline=baseline_run)
```

### Exporting Trends

```python
from pathlib import Path
from calibrax.storage.store import Store

store = Store(Path("temp/doc-examples/exporters-store"))
store.save(run)

exporter.export_trends(
    store=store,
    metric="throughput",
    point_name="forward_pass",
    tags={"framework": "flax"},
    n_runs=50,
)
```

### Logging Custom Artifacts

```python
# Log matplotlib figures
exporter.log_figures({"scaling": fig})

# Log HTML artifacts
exporter.log_html_artifacts({"report": html_string})

# Log custom tables
exporter.log_extra_tables({
    "results": (
        ["Framework", "Throughput", "Latency"],  # headers
        [["flax", 1200, 0.8], ["pytorch", 950, 1.2]],  # rows
    ),
})
```

## Publication Generator

!!! warning "Optional Dependency"

    Plot generation requires matplotlib: `uv pip install "calibrax[publication]"`

    Table generation (LaTeX, HTML, CSV) works without matplotlib.

`PublicationGenerator` creates plots and tables suitable for papers and reports:

```python
from pathlib import Path
from calibrax.exporters.publication import PublicationGenerator

pub = PublicationGenerator(output_dir=Path("temp/doc-examples/figures"))
```

### Comparison Plots

Bar charts comparing metrics across configurations:

```python
path = pub.generate_comparison_plot(
    run,
    metrics=["throughput", "latency"],
    output_format="pdf",  # "png", "pdf", or "svg"
)
if path is not None:
    print(f"Plot saved to {path}")
```

### Scaling Plots

Log-log plots with fitted scaling laws:

```python
path = pub.generate_scaling_plot(
    sizes=[100, 500, 1000, 5000],
    values=[0.01, 0.05, 0.10, 0.52],
    metric_name="latency",
    output_format="pdf",
)
```

### Convergence Plots

Time series plots showing metric trends:

```python
trend = store.extract_trend("throughput", "forward_pass", {"framework": "flax"})
path = pub.generate_convergence_plot(trend, output_format="png")
```

### Tables

Generate LaTeX, HTML, or CSV tables:

```python
# LaTeX table
path = pub.generate_table(run, output_format="latex")
print(f"LaTeX table at {path}")  # ./figures/table.tex

# HTML table
path = pub.generate_table(run, output_format="html")

# CSV table
path = pub.generate_table(
    run,
    metrics=["throughput", "latency"],
    output_format="csv",
    group_by_tag="framework",
)
```

## Writing Custom Exporters

Subclass `Exporter` to integrate with other backends:

```python
from calibrax.exporters.base import Exporter
from calibrax.core.models import Run

class SlackExporter(Exporter):
    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url

    def export_run(self, run: Run) -> str:
        # Post summary to Slack
        message = f"Benchmark {run.id}: {len(run.points)} points"
        # ... send to webhook ...
        return self._webhook_url

    def export_analysis(self, run: Run, baseline: Run | None = None) -> None:
        # Post regression summary
        ...
```

## Next Steps

<div class="grid cards" markdown>

-   :material-shield-check:{ .lg .middle } **CI Integration**

    ---

    Automate exports as part of CI pipelines

    [:octicons-arrow-right-24: CI integration](ci-integration.md)

-   :material-database:{ .lg .middle } **Storage & Baselines**

    ---

    Manage the data that feeds into exports

    [:octicons-arrow-right-24: Storage](storage.md)

</div>
