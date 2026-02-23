# calibrax.core.models

Data model classes for benchmark results. All dataclasses are frozen and
immutable, with `to_dict()` / `from_dict()` support for JSON serialization.

Key types: `Run`, `Point`, `Metric`, `MetricDef`, `MetricDirection`,
`MetricPriority`, `Regression`, `RankEntry`, `SignificanceResult`, `ScalingLaw`,
`TrendPoint`, `TrendSeries`.

Key functions: `extract_framework_metrics`, `is_higher_better`.

::: calibrax.core.models
    options:
      show_root_heading: false
