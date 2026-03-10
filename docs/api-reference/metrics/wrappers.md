# calibrax.metrics.wrappers

Decorator-pattern wrappers that enhance any metric function with additional
behavior. `BootstrapMetric` adds confidence interval estimation,
`ClasswiseWrapper` provides per-class breakdown, `MetricTracker` tracks
historical values with best-value detection, and `MinMaxTracker` maintains
running min/max/current state.

::: calibrax.metrics.wrappers
    options:
      show_source: false
      show_root_heading: false
      members_order: source
      docstring_style: google
      show_signature_annotations: true
