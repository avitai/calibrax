# calibrax.metrics.composition

Composition framework for grouping and combining metrics. `MetricCollection`
groups multiple metrics for batch computation, `WeightedMetric` produces
a single weighted score, `MetricSuite` organizes metrics by domain,
and `ThresholdMetric` wraps a metric with a pass/fail threshold for CI gates.

::: calibrax.metrics.composition
    options:
      show_source: false
      show_root_heading: false
      members_order: source
      docstring_style: google
      show_signature_annotations: true
