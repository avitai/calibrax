# Peer Comparison

Calibrax overlaps with metric libraries and benchmark tools, but its center of
gravity is different: JAX-native scientific ML benchmarking, profiling,
statistical analysis, regression detection, and metric evaluation in one
package.

## Summary

| Tool | Primary focus | Where Calibrax differs |
|------|---------------|------------------------|
| [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/) | PyTorch metric implementations with functional and module APIs, broad domains, wrappers, plotting, and distributed support | Calibrax targets JAX first, adds benchmark storage, profiling, statistical comparison, CI regression gates, and geometry-heavy metric metadata |
| [jax_metrics](https://cgarciae.github.io/jax_metrics/) | JAX metric and loss abstractions with pytree state, distributed-friendly accumulation, and numerical-equivalence discipline | Calibrax has a broader benchmarking system, 111 registered Tier 0 metrics, registry metadata, exporters, storage, and regression analysis |
| [ASV](https://asv.readthedocs.io/en/stable/) | Benchmarking Python packages over time with runtime, memory, custom values, and static web output | Calibrax stores JSON-per-run benchmark results inside the project workflow and adds JAX-specific profiling, statistical tests, and CI gates |
| [CodSpeed](https://codspeed.io/docs) | Hosted and CI-oriented performance testing with PR checks, profiling, and benchmark reports | Calibrax now includes a focused CodSpeed workflow for PR benchmark checks while keeping local storage and analysis in Calibrax |

## TorchMetrics

Use TorchMetrics when the project is PyTorch-based and needs its mature metric
catalog, Lightning integration, module-state API, wrappers, and plotting.

Use Calibrax when the code is JAX-based, when metric metadata needs to be
queried by domain, mathematical properties, or invariance, or when metric
values need to live alongside profiling and benchmark regression records.

TorchMetrics includes mature plotting and distributed wrappers beyond Calibrax's
current stateful metric surface. Calibrax now covers CRPS for JAX ensemble
forecasts and exposes VMAF through an optional FFmpeg/libvmaf boundary rather
than a core dependency.

## jax_metrics

`jax_metrics` is closest to Calibrax on framework choice. It provides
Keras-like metric and loss abstractions, pytree-friendly state, distributed
accumulation, and notes that metrics are usually checked against Keras or
TorchMetrics references.

Calibrax should keep that numerical-equivalence bar for standard metrics, but
its current scope is larger: metric registry discovery, geometry and graph
metrics, benchmarking, profiling, storage, exporters, and CI regression gates.

## ASV

ASV is strongest for long-running benchmark history across commits and
environments. It can publish an interactive static site and is widely used by
scientific Python projects.

Calibrax is lighter-weight for local scientific ML workflows that need JAX
timing, hardware metadata, statistical comparison, baseline storage, and
regression checks without adopting a separate ASV project layout.

## CodSpeed

CodSpeed is strongest when a hosted PR workflow should catch performance
regressions before merge. Its docs cover CI setup, benchmark checks, profiling,
and repository integration.

Calibrax includes a focused CodSpeed workflow for the `tests/performance/`
benchmark suite. Treat CodSpeed as an external reporting layer for PR checks;
Calibrax storage, statistical analysis, and regression gates remain the local
source of benchmark truth.
