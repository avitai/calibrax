# Calibrax

[![CI](https://github.com/avitai/calibrax/actions/workflows/ci.yml/badge.svg)](https://github.com/avitai/calibrax/actions/workflows/ci.yml)
[![Build](https://github.com/avitai/calibrax/actions/workflows/build-verification.yml/badge.svg)](https://github.com/avitai/calibrax/actions/workflows/build-verification.yml)
[![Quality](https://github.com/avitai/calibrax/actions/workflows/quality-checks.yml/badge.svg)](https://github.com/avitai/calibrax/actions/workflows/quality-checks.yml)
[![Security](https://github.com/avitai/calibrax/actions/workflows/security.yml/badge.svg)](https://github.com/avitai/calibrax/actions/workflows/security.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.11+-green.svg)](https://github.com/jax-ml/jax)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Validated against:** scikit-learn and SciPy references for representative regression, classification, distance, and divergence metrics.

[Documentation](https://calibrax.readthedocs.io/en/latest/) - [Issues](https://github.com/avitai/calibrax/issues) - [Contributing](CONTRIBUTING.md)

---

> **Research preview.** The API changes between releases until v1.0, so pin a version if you need
> stability. Calibrax depends on one other Avitai package, substrax, for its record format, typing,
> RNG keys and device detection, so it can be tried on its own.
>
> Issues, questions and pull requests decide what gets built next.

---

**Calibrax** (*Calibrate + JAX*) holds the evaluation metrics, benchmarking, profiling and
statistical analysis that [Datarax](https://github.com/avitai/datarax),
[Artifex](https://github.com/avitai/artifex) and [Opifex](https://github.com/avitai/opifex) share,
in one JAX package.

## Features

### Metrics (156 registered Tier 0 metrics, 22 domains, 4-tier architecture)

Calibrax provides a 4-tier metric system covering the full spectrum of ML
evaluation. The current registry contains 156 Tier 0 pure-function metrics;
Tier 1-3 APIs, optional plugins, and metric-learning losses are part of the
package architecture but are not all registered metric entries today.

| Tier | Name | Pattern | Examples |
|------|------|---------|----------|
| 0 | Pure Functions | `fn(predictions, targets) -> scalar` | MSE, cosine distance, BLEU |
| 1 | Frozen Backbone | `update() -> compute() -> reset()` | FID, BERTScore, Inception Score |
| 2 | Learned | `nnx.Module` with trainable weights | LPIPS |
| 3 | Metric Learning | Differentiable embedding loss | Contrastive, Triplet, ArcFace |

**Functional domains:** general, classification, calibration, segmentation, distance, divergence, information, ranking, statistical, survival, clustering, fairness, forecasting, uncertainty, generative, image, text, audio, geometric, graph, manifold, seismic

**Key capabilities:**

- **MetricRegistry** with axiom-based discovery for registered Tier 0 metrics (`list_true_metrics()`, `list_by_invariance("rotation")`)
- **Geometric distance hierarchy** - Euclidean, Riemannian (SPD, Grassmann, Stiefel), pseudo-Riemannian (ultrahyperbolic), Finsler (Randers)
- **Graph metrics** - spectral distance, resistance distance, Floyd-Warshall shortest paths
- **Reference checks** - representative Tier 0 metrics are tested against scikit-learn and SciPy references with `1e-6` tolerance; see [Peer Comparison](docs/user-guide/peer-comparison.md)
- **Losses with masks and weights** - MSE, MAE, Huber, Charbonnier, relative L2 and softmax cross-entropy take `mask`, `weights`, `reduction` and `axis`, reduced one way
- **Statistics with masks and weights** - R-squared, adjusted R-squared, Pearson, concordance, Kendall's tau and Spearman's rho take `mask` and `weights`. Kendall's and Spearman's weights are frequency weights: integer weights give the coefficient of the sample with each element repeated. A masked or zero-weighted `NaN` reaches neither the value nor the gradient, and a selection with no members reads `NaN` for a statistic and `0.0` for a loss
- **Composition** - `MetricCollection`, `WeightedMetric`, `MetricSuite`, `ThresholdMetric`
- **Wrappers** - `BootstrapMetric` (confidence intervals), `ClasswiseWrapper`, `MetricTracker`, `MinMaxTracker`
- **Metric learning losses** - contrastive, triplet margin, NTXent, ArcFace, CosFace, ProxyNCA, ProxyAnchor, with hard/semi-hard negative mining

### Benchmarking & Profiling

- **Timing** - `time_calls`: median and percentiles of repeated calls, each waited on, with warm-up calls absorbing compilation
- **Resource monitoring** - CPU, memory, GPU memory/clock/power tracking
- **Energy & carbon** - Energy measurement with carbon footprint estimation
- **FLOPS & roofline** - XLA-level FLOP counting, roofline performance analysis
- **Compilation** - XLA compilation profiling and tracing
- **Complexity** - Parameter counts, parameter memory and computational cost of Flax NNX modules
- **Hardware** - Automatic hardware detection and capability reporting

### Analysis & Infrastructure

- **Statistical analysis** - Bootstrap confidence intervals, hypothesis testing, effect sizes, outlier detection
- **Regression detection** - Direction-aware threshold checks against a stored baseline
- **Comparison & ranking** - Cross-configuration comparison, Pareto front analysis, aggregate scoring
- **Validation** - Convergence analysis and accuracy assessment
- **Storage** - JSON-per-run file backend with baseline management
- **Exporters** - W&B and MLflow integration, publication-ready LaTeX/HTML/CSV tables and matplotlib plots
- **CI integration** - Regression gate with git bisect automation
- **Monitoring** - Production alerting with configurable thresholds
- **CLI** - `calibrax ingest|export|check|baseline|trend|summary|profile|profile-gpu`

## Quick Start

```python
import jax.numpy as jnp
from calibrax.metrics import MetricRegistry, calculate_all
from calibrax.metrics.functional.regression import mse, r_squared

predictions = jnp.array([1.1, 2.3, 2.8, 4.2, 4.7])
targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Individual metrics
print(f"MSE: {mse(predictions, targets):.4f}")
print(f"R²:  {r_squared(predictions, targets):.4f}")

# Batch computation of all registered metrics
results = calculate_all(predictions, targets, metrics=["mse", "mae", "rmse", "r_squared"])

# Registry discovery
registry = MetricRegistry()
true_metrics = registry.list_true_metrics()
rotation_inv = registry.list_by_invariance("rotation")
```

## Installation

```bash
# Basic installation
uv pip install calibrax

# With CUDA 12 JAX and NVML GPU monitoring
uv pip install "calibrax[cuda12]"

# With publication export (matplotlib)
uv pip install "calibrax[publication]"
```

## Architecture

```
src/calibrax/
├── core/          Data models, protocols, adapters, result container, registry
├── profiling/     Timing, resources, GPU, energy, FLOPS, roofline, compilation,
│                  complexity, hardware, tracing, carbon
├── statistics/    Summary statistics, outliers, bootstrap, significance testing
├── analysis/      Regression, comparison, ranking, scaling, Pareto, changepoint
├── validation/    Convergence, accuracy, validation framework
├── monitoring/    Alerts, production monitoring
├── storage/       JSON store, baselines
├── exporters/     W&B, MLflow, publication-ready output
├── metrics/
│   ├── functional/   156 Tier 0 pure functions across 22 domains
│   ├── stateful/     Tier 1-2 base classes (FrozenBackboneMetric, LearnedMetric)
│   ├── learning/     Tier 3 metric learning losses and miners
│   ├── plugins/      Optional-dependency metrics (FID, BERTScore, LPIPS)
│   ├── composition.py   MetricCollection, WeightedMetric, MetricSuite, ThresholdMetric
│   ├── wrappers.py      BootstrapMetric, ClasswiseWrapper, MetricTracker, MinMaxTracker
│   └── _registry.py     MetricRegistry singleton with axiom-based discovery
├── ci/            CI regression gate, bisection engine
└── cli/           Command-line interface
```

## Examples

Eight runnable examples live in `examples/metrics/`, each a Python script with its notebook beside
it. [docs/examples.md](docs/examples.md) lists them by tier, with what each covers and how to run
it.

## Contributing

Development setup, the `setup.sh` flags, and the verification commands are in
[CONTRIBUTING.md](CONTRIBUTING.md); the contributor documentation starts at
[docs/contributing](docs/contributing/index.md).

## License

MIT
