# calibrax.core.adapters

Adapter classes for wrapping external objects (ML models, frameworks) to work
with Calibrax's interfaces. Includes `BenchmarkAdapter` (ABC for non-JAX targets)
and `NNXBenchmarkAdapter` (inherits `nnx.Module` for JIT compatibility).

::: calibrax.core.adapters
    options:
      show_root_heading: false
