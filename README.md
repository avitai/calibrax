# Calibrax

[![CI](https://github.com/avitai/calibrax/actions/workflows/ci.yml/badge.svg)](https://github.com/avitai/calibrax/actions/workflows/ci.yml)
[![Build](https://github.com/avitai/calibrax/actions/workflows/build-verification.yml/badge.svg)](https://github.com/avitai/calibrax/actions/workflows/build-verification.yml)
[![Quality](https://github.com/avitai/calibrax/actions/workflows/quality-checks.yml/badge.svg)](https://github.com/avitai/calibrax/actions/workflows/quality-checks.yml)
[![Security](https://github.com/avitai/calibrax/actions/workflows/security.yml/badge.svg)](https://github.com/avitai/calibrax/actions/workflows/security.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4+-green.svg)](https://github.com/jax-ml/jax)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Validated against:** scikit-learn and SciPy references for representative regression, classification, distance, and divergence metrics.

[Documentation](https://calibrax.readthedocs.io/en/latest/) - [Issues](https://github.com/avitai/calibrax/issues) - [Contributing](CONTRIBUTING.md)

---

> **Research preview.** The API will change while we iterate toward v1.0, so pin a version if you
> need stability. Calibrax is the most standalone library in the Avitai stack: it depends on none
> of the others, so it is a low-commitment way to try one piece.
>
> This is public this early on purpose. Issues, questions and pull requests genuinely steer
> what gets built next, and a star tells us which layer to push on.

---

**Calibrax** (*Calibrate + JAX*) is a unified benchmarking and metrics framework for the JAX scientific ML ecosystem. It extracts and consolidates shared benchmarking, profiling, statistical analysis, and evaluation functionality from
[Datarax](https://github.com/avitai/datarax),
[Artifex](https://github.com/avitai/artifex), and
[Opifex](https://github.com/avitai/opifex).

## Features

### Metrics (111 registered Tier 0 metrics, 17 domains, 4-tier architecture)

Calibrax provides a 4-tier metric system covering the full spectrum of ML
evaluation. The current registry contains 111 Tier 0 pure-function metrics;
Tier 1-3 APIs, optional plugins, and metric-learning losses are part of the
package architecture but are not all registered metric entries today.

| Tier | Name | Pattern | Examples |
|------|------|---------|----------|
| 0 | Pure Functions | `fn(predictions, targets) -> scalar` | MSE, cosine distance, BLEU |
| 1 | Frozen Backbone | `update() -> compute() -> reset()` | FID, BERTScore, Inception Score |
| 2 | Learned | `nnx.Module` with trainable weights | LPIPS |
| 3 | Metric Learning | Differentiable embedding loss | Contrastive, Triplet, ArcFace |

**Functional domains:** regression, classification, calibration, segmentation, distance, divergence, information, ranking, statistical, clustering, fairness, image, text, audio, geometric, graph, manifold

**Key capabilities:**

- **MetricRegistry** with axiom-based discovery for registered Tier 0 metrics (`list_true_metrics()`, `list_by_invariance("rotation")`)
- **Geometric distance hierarchy** - Euclidean, Riemannian (SPD, Grassmann, Stiefel), pseudo-Riemannian (ultrahyperbolic), Finsler (Randers)
- **Graph metrics** - spectral distance, resistance distance, Floyd-Warshall shortest paths
- **Reference checks** - representative Tier 0 metrics are tested against scikit-learn and SciPy references with `1e-6` tolerance; see [Peer Comparison](docs/user-guide/peer-comparison.md)
- **Composition** - `MetricCollection`, `WeightedMetric`, `MetricSuite`, `ThresholdMetric`
- **Wrappers** - `BootstrapMetric` (confidence intervals), `ClasswiseWrapper`, `MetricTracker`, `MinMaxTracker`
- **Metric learning losses** - contrastive, triplet margin, NTXent, ArcFace, CosFace, ProxyNCA, ProxyAnchor, with hard/semi-hard negative mining

### Benchmarking & Profiling

- **Timing** - Warm-up aware timing with JIT compilation separation
- **Resource monitoring** - CPU, memory, GPU memory/clock/power tracking
- **Energy & carbon** - Energy measurement with carbon footprint estimation
- **FLOPS & roofline** - XLA-level FLOP counting, roofline performance analysis
- **Compilation** - XLA compilation profiling and tracing
- **Complexity** - Algorithmic complexity analysis
- **Hardware** - Automatic hardware detection and capability reporting

### Analysis & Infrastructure

- **Statistical analysis** - Bootstrap confidence intervals, hypothesis testing, effect sizes, outlier detection
- **Regression detection** - Direction-aware detection with configurable severity levels
- **Comparison & ranking** - Cross-configuration comparison, Pareto front analysis, aggregate scoring
- **Validation** - Convergence analysis and accuracy assessment
- **Storage** - JSON-per-run file backend with baseline management
- **Exporters** - W&B and MLflow integration, publication-ready LaTeX/HTML/CSV tables and matplotlib plots
- **CI integration** - Regression gate with git bisect automation
- **Monitoring** - Production alerting with configurable thresholds
- **CLI** - `calibrax ingest|export|check|baseline|trend|summary|profile`

## Quick Start

```python
import jax.numpy as jnp
from calibrax.metrics import MetricRegistry, calculate_all
from calibrax.metrics.functional.regression import mse, mae, r_squared

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

# With statistical analysis (scipy)
uv pip install "calibrax[stats]"

# With GPU monitoring
uv pip install "calibrax[gpu]"

# With image quality plugins (FID, Inception Score)
uv pip install "calibrax[image]"

# With text quality plugins (BERTScore)
uv pip install "calibrax[text]"

# With publication export (matplotlib)
uv pip install "calibrax[publication]"
```

## Development Setup

The recommended way to set up a development environment is with the included `setup.sh` script. It auto-detects your platform (Linux CUDA, macOS Intel, Apple Silicon), creates a virtual environment, installs all dependencies, and generates an activation script.

```bash
git clone https://github.com/avitai/calibrax.git
cd calibrax

# Standard setup with automatic GPU detection
./setup.sh

# Activate the environment
source ./activate.sh
```

### setup.sh Options

| Flag | Description |
|------|-------------|
| `--cpu-only` | Force CPU-only setup, skip GPU/Metal detection |
| `--metal` | Enable Metal acceleration on Apple Silicon Macs |
| `--deep-clean` | Clear JAX cache, pip cache, pytest cache, and other artifacts |
| `--force` | Force reinstallation even if environment exists |
| `--verbose`, `-v` | Show detailed output during setup |

```bash
# Examples
./setup.sh --cpu-only         # CPU-only development
./setup.sh --metal            # Apple Silicon with Metal
./setup.sh --force --verbose  # Force reinstall with full output
./setup.sh --deep-clean       # Clean everything and start fresh
```

### Manual Setup

If you prefer to set up manually:

```bash
git clone https://github.com/avitai/calibrax.git
cd calibrax
uv venv
uv pip install -e ".[dev,test,stats]"
uv run pre-commit install
```

## Architecture

```
src/calibrax/
├── core/          Data models, protocols, adapters, result container, registry
├── profiling/     Timing, resources, GPU, energy, FLOPS, roofline, compilation,
│                  complexity, hardware, tracing, carbon
├── statistics/    Statistical analyzer, significance testing
├── analysis/      Regression, comparison, ranking, scaling, Pareto, changepoint
├── validation/    Convergence, accuracy, validation framework
├── monitoring/    Alerts, production monitoring
├── storage/       JSON store, baselines
├── exporters/     W&B, MLflow, publication-ready output
├── metrics/
│   ├── functional/   111 Tier 0 pure functions across 17 domains
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

Runnable examples are in `examples/metrics/`, available as both Python scripts and Jupyter notebooks:

| Example | Level | Topics |
|---------|-------|--------|
| [01_quickstart.py](examples/metrics/01_quickstart.py) | Beginner | Individual metrics, `calculate_all`, registry queries |
| [02_regression_deep_dive.py](examples/metrics/02_regression_deep_dive.py) | Beginner | Same-shape regression metrics, outlier sensitivity |
| [03_classification.py](examples/metrics/03_classification.py) | Intermediate | Classification, calibration, segmentation |
| [04_distances.py](examples/metrics/04_distances.py) | Intermediate | Euclidean, hyperbolic, divergences, information theory |
| [05_composition.py](examples/metrics/05_composition.py) | Intermediate | Collections, weighted metrics, quality gates, tracking |
| [06_image_quality.py](examples/metrics/06_image_quality.py) | Intermediate | PSNR, SSIM, MS-SSIM, BLEU, ROUGE |
| [07_metric_learning.py](examples/metrics/07_metric_learning.py) | Advanced | Contrastive, triplet, NTXent, ArcFace, mining |
| [08_manifold_graph.py](examples/metrics/08_manifold_graph.py) | Advanced | SPD, Grassmann, spectral distance, Floyd-Warshall |

## Development

```bash
# Activate the local environment first
source activate.sh

# Run tests
uv run pytest tests/ -v --cov=calibrax --cov-report=term-missing

# Lint & format
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/

# Type check
uv run pyright src/

# All quality checks
uv run pre-commit run --all-files

# Build documentation
uv run mkdocs build --strict --clean

# Convert examples to Jupyter notebooks
uv run python scripts/jupytext_converter.py batch-py-to-nb examples/metrics/
```

## License

MIT
