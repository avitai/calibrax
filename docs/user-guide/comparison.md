# Comparing Configurations

Calibrax provides tools for ranking configurations by a single metric, computing
aggregate scores across multiple metrics, generating full comparison reports,
and finding Pareto-optimal tradeoffs.

## Ranking by a Single Metric

`rank_table()` ranks all points in a run by a given metric, grouping by a tag
(default: `"framework"`). Rankings respect `MetricDef.direction` — higher-is-better
metrics rank the highest value first, and lower-is-better metrics rank the lowest
value first.

```python
from calibrax.core.models import (
    MetricDef, MetricDirection, Metric, Point, Run,
)
from calibrax.analysis.ranking import rank_table

run = Run(
    points=(
        Point(name="bench", scenario="train", tags={"framework": "flax"},
              metrics={"throughput": Metric(value=1200.0)}),
        Point(name="bench", scenario="train", tags={"framework": "pytorch"},
              metrics={"throughput": Metric(value=950.0)}),
        Point(name="bench", scenario="train", tags={"framework": "keras"},
              metrics={"throughput": Metric(value=1050.0)}),
    ),
    metric_defs={
        "throughput": MetricDef(
            name="throughput", unit="samples/sec", direction=MetricDirection.HIGHER
        ),
    },
)

rankings = rank_table(run, "throughput")
for entry in rankings:
    print(f"#{entry.rank} {entry.label}: {entry.value:.0f} "
          f"({'best' if entry.is_best else f'{entry.delta_from_best:.1f}% behind'})")
```

```text
#1 flax: 1200 (best)
#2 keras: 1050 (12.5% behind)
#3 pytorch: 950 (20.8% behind)
```

## Aggregate Scoring

`aggregate_score()` computes a weighted, normalized score across multiple
metrics. Scores are in `[0, 1]` where higher is always better, regardless of
individual metric directions:

```python
from calibrax.core.models import MetricDef, MetricDirection, Metric, Point, Run
from calibrax.analysis.ranking import aggregate_score

run_multi = Run(
    points=(
        Point(name="bench", scenario="train", tags={"framework": "flax"},
              metrics={"throughput": Metric(value=1200.0), "latency": Metric(value=0.8)}),
        Point(name="bench", scenario="train", tags={"framework": "pytorch"},
              metrics={"throughput": Metric(value=950.0), "latency": Metric(value=1.2)}),
        Point(name="bench", scenario="train", tags={"framework": "keras"},
              metrics={"throughput": Metric(value=1050.0), "latency": Metric(value=1.0)}),
    ),
    metric_defs={
        "throughput": MetricDef(
            name="throughput", unit="samples/sec", direction=MetricDirection.HIGHER
        ),
        "latency": MetricDef(
            name="latency", unit="ms", direction=MetricDirection.LOWER
        ),
    },
)

scores = aggregate_score(run_multi, weights={"throughput": 0.6, "latency": 0.4})
for label, score in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"{label}: {score:.3f}")
```

```text
flax: 1.000
keras: 0.440
pytorch: 0.000
```

## Cross-Configuration Comparison

`compare_configurations()` takes a dictionary of labeled runs and produces a
`ComparisonReport` with per-metric comparisons, winner determination, and
improvement factors:

```python
from calibrax.core.models import MetricDef, MetricDirection, Metric, Point, Run
from calibrax.analysis.comparison import compare_configurations

metric_defs = {
    "throughput": MetricDef(
        name="throughput", unit="samples/sec", direction=MetricDirection.HIGHER
    ),
    "latency": MetricDef(
        name="latency", unit="ms", direction=MetricDirection.LOWER
    ),
}

flax_run = Run(
    points=(Point(name="bench", scenario="train", tags={"framework": "flax"},
                  metrics={"throughput": Metric(value=1200.0), "latency": Metric(value=0.8)}),),
    metric_defs=metric_defs,
)
pytorch_run = Run(
    points=(Point(name="bench", scenario="train", tags={"framework": "pytorch"},
                  metrics={"throughput": Metric(value=950.0), "latency": Metric(value=1.2)}),),
    metric_defs=metric_defs,
)
keras_run = Run(
    points=(Point(name="bench", scenario="train", tags={"framework": "keras"},
                  metrics={"throughput": Metric(value=1050.0), "latency": Metric(value=1.0)}),),
    metric_defs=metric_defs,
)

report = compare_configurations(
    runs={"flax": flax_run, "pytorch": pytorch_run, "keras": keras_run},
    metrics=["throughput", "latency"],
)

print(f"Overall winner: {report.overall_winner}")
for mc in report.metric_comparisons:
    print(f"\n{mc.metric_name}:")
    print(f"  Best: {mc.best_label}")
    for label, factor in mc.improvement_factors.items():
        print(f"  {label}: {factor:.2f}x of best")
```

```text
Overall winner: flax

throughput:
  Best: flax
  flax: 1.00x of best
  keras: 1.14x of best
  pytorch: 1.26x of best

latency:
  Best: flax
  flax: 1.00x of best
  keras: 1.25x of best
  pytorch: 1.50x of best
```

The report is serializable via `report.to_dict()` and `ComparisonReport.from_dict()`.

## Pareto Front Analysis

`pareto_front()` finds the set of points where no other point is better in
all specified metrics simultaneously — the Pareto-optimal configurations:

```python
from calibrax.analysis.pareto import pareto_front

# All points from a multi-configuration run
optimal = pareto_front(
    points=list(run.points),
    x_metric="throughput",
    y_metric="latency",
    metric_defs=run.metric_defs,
)

print(f"Pareto-optimal configurations: {len(optimal)} of {len(run.points)}")
for p in optimal:
    print(f"  {p.tags.get('framework', 'unknown')}: "
          f"throughput={p.metrics['throughput'].value:.0f}, "
          f"latency={p.metrics['latency'].value:.3f}")
```

The function respects `MetricDef.direction` — for `HIGHER` metrics, larger values
dominate; for `LOWER` metrics, smaller values dominate. When `metric_defs` is
not provided, both metrics default to higher-is-better.

## Best Practices

- Use `rank_table()` for quick single-metric comparisons
- Use `aggregate_score()` when multiple metrics matter and you need a single
  ranking
- Use `compare_configurations()` for detailed multi-metric reports suitable
  for export
- Use `pareto_front()` to identify configurations with fundamentally different
  tradeoffs (e.g., throughput vs. latency)

## Next Steps

<div class="grid cards" markdown>

-   :material-export:{ .lg .middle } **Exporting Results**

    ---

    Export comparison reports to W&B or publication-ready tables

    [:octicons-arrow-right-24: Exporters](exporters.md)

-   :material-chart-bell-curve:{ .lg .middle } **Statistical Analysis**

    ---

    Add statistical rigor with significance tests and confidence intervals

    [:octicons-arrow-right-24: Statistics](statistics.md)

</div>
