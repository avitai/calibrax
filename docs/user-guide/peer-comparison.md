# Peer Comparison

Calibrax overlaps with metric libraries and benchmark tools, but its center of
gravity is different: JAX-native scientific ML benchmarking, profiling,
statistical analysis, regression detection, and metric evaluation in one
package.

## Summary

| Tool | Primary focus | Where Calibrax differs |
|------|---------------|------------------------|
| [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/) | PyTorch metric implementations with functional and module APIs, broad domains, wrappers, plotting, and distributed support | Calibrax targets JAX first, adds benchmark storage, profiling, statistical comparison, CI regression gates, and geometry-heavy metric metadata |
| [jax_metrics](https://cgarciae.github.io/jax_metrics/) | JAX metric and loss abstractions with pytree state, distributed-friendly accumulation, and numerical-equivalence discipline | Calibrax has a broader benchmarking system, 132 registered Tier 0 metrics, registry metadata, exporters, storage, and regression analysis |
| [ASV](https://asv.readthedocs.io/en/stable/) | Benchmarking Python packages over time with runtime, memory, custom values, and static web output | Calibrax stores JSON-per-run benchmark results inside the project workflow and adds JAX-specific profiling, statistical tests, and CI gates |
| [CodSpeed](https://codspeed.io/docs) | Hosted and CI-oriented performance testing with PR checks, profiling, and benchmark reports | Calibrax now includes a focused CodSpeed workflow for PR benchmark checks while keeping local storage and analysis in Calibrax |
| [metrax](https://github.com/google/metrax) | Google's JAX evaluation metrics on CLU: accuracy, AUC, F-beta, MSE/MAE/RMSE, R-squared, Spearman, ranking at k, BLEU/ROUGE/WER/perplexity, PSNR/SSIM/IoU/Dice, SNR, with an `nnx` subpackage | Calibrax covers those as Tier 0 functions plus fourteen more domains (distance, divergence, information, geometric, graph, manifold, fairness, forecasting, uncertainty, generative, ...), registry metadata (direction, signature, invariances, properness), composition, and the benchmarking stack; metrax accumulates distributed state, which Calibrax leaves to the caller |
| [MAPIE](https://github.com/scikit-learn-contrib/MAPIE), [crepes](https://github.com/henrikbostrom/crepes), [puncc](https://github.com/deel-ai/puncc) | Conformal prediction on top of scikit-learn-style estimators: prediction intervals and sets with coverage guarantees, Mondrian and normalised variants, risk control, exchangeability tests | Calibrax does not build conformal predictors; it scores the intervals and distributions they produce (`picp`, `mpiw`, `interval_score`, `regression_calibration_error`, `gaussian_nll`, the forecasting scores) as JAX functions, so a conformal library and Calibrax compose rather than overlap |
| [netcal](https://github.com/EFS-OpenSource/calibration-framework) | Calibration methods (binning, scaling, regularisation) for classification, detection and regression, with ECE/MCE-style metrics and reliability plots | Calibrax has the metrics side (Brier, ECE, MCE, adaptive and classwise ECE, PIT and rank histograms, event reliability) without the recalibration methods; netcal's post-hoc calibrators are a modelling step that belongs to the model library |
| [XProf](https://github.com/openxla/xprof) | The XLA profiler UI: overview page, trace viewer, op profile, memory profile and input pipeline analysis over profiles captured with `jax.profiler` | Calibrax's `TraceLinker` captures those profiles and links them to a benchmark run, and its profilers report the numbers a run stores (timing, FLOPs from XLA's cost analysis, roofline, compilation); XProf is where a captured trace is read |
| [accelerator-microbenchmarks](https://github.com/AI-Hypercomputer/accelerator-microbenchmarks) | Google's YAML-driven JAX microbenchmarks of GEMMs, attention and collectives on TPUs and GPUs, reporting latency, throughput and bandwidth as JSON/CSV | Calibrax benchmarks the caller's own functions and models rather than fixed kernels, stores runs with statistics, baselines and regression gates, and reads its hardware peaks from a spec table those microbenchmarks could calibrate |

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

## metrax

metrax is the JAX-native metric library closest to Calibrax's Tier 0 surface. Its
classes accumulate state through CLU, which fits distributed evaluation loops; Calibrax
keeps metrics as pure functions and leaves accumulation to the composition layer.
Overlapping names (accuracy, AUC, MSE, BLEU, PSNR, SSIM, IoU) are candidates for the
numerical-equivalence tests; the registry, the geometry-heavy domains, and the
benchmarking system are where the two differ.

## Conformal prediction libraries

MAPIE, crepes and puncc produce prediction intervals and sets with coverage guarantees
from scikit-learn-style estimators. Calibrax scores what they produce: `picp` and
`mpiw` for coverage and width, `interval_score` for a proper score, and the
`uncertainty` and `forecasting` domains for distributional forecasts. A workflow
calibrates with one of those libraries and reports with Calibrax.

## netcal

netcal implements post-hoc calibration methods and the metrics that judge them.
Calibrax implements the metrics side in JAX and no recalibration methods; a model
library that ships temperature scaling or histogram binning pairs with Calibrax's
`calibration` domain to measure the result.

## XProf and accelerator-microbenchmarks

XProf reads profiles; Calibrax's `TraceLinker` writes them alongside a run's stored
metrics, and the profiling modules report the scalar numbers a run keeps. Google's
accelerator-microbenchmarks measure fixed kernels on TPUs and GPUs; Calibrax measures
the caller's code against hardware peaks from its spec table, which those kernel
measurements are one way to calibrate.
