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

---

> **Early Development** — API is unstable and subject to breaking changes.
> Pin to specific commits if stability is required.

---

**Calibrax** (*Calibrate + JAX*) is a unified benchmarking framework for the JAX scientific ML ecosystem. It extracts and consolidates shared benchmarking, profiling, and statistical analysis functionality from
[Datarax](https://github.com/avitai/datarax),
[Artifex](https://github.com/avitai/artifex), and
[Opifex](https://github.com/avitai/opifex).

## Features

- **Profiling** — Timing with warm-up awareness, resource monitoring, GPU memory/clock/power tracking, energy measurement, FLOPS counting, roofline analysis, XLA compilation profiling, complexity analysis, hardware detection, carbon tracking
- **Statistical Analysis** — Bootstrap confidence intervals, hypothesis testing, effect sizes, outlier detection
- **Regression Detection** — Direction-aware detection with configurable severity levels
- **Comparison & Ranking** — Cross-configuration comparison, Pareto front analysis, aggregate scoring
- **Validation** — Convergence analysis and accuracy assessment
- **Storage** — JSON-per-run file backend with baseline management
- **Exporters** — W&B and MLflow integration, publication-ready LaTeX/HTML/CSV tables and matplotlib plots
- **CI Integration** — Regression gate with git bisect automation
- **Monitoring** — Production alerting with configurable thresholds
- **CLI** — `calibrax ingest|export|check|baseline|trend|summary|profile`

## Installation

```bash
# Basic installation
uv pip install calibrax

# With statistical analysis (scipy)
uv pip install "calibrax[stats]"

# With GPU monitoring
uv pip install "calibrax[gpu]"

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
├── profiling/     Timing, resources, GPU, energy, FLOPS, roofline, compilation, complexity, hardware, tracing, carbon
├── statistics/    Statistical analyzer, significance testing
├── analysis/      Regression, comparison, ranking, scaling, Pareto, changepoint
├── validation/    Convergence, accuracy, validation framework
├── monitoring/    Alerts, production monitoring
├── storage/       JSON store, baselines
├── exporters/     W&B, MLflow, publication-ready output
├── metrics/       JAX-native evaluation metrics
├── ci/            CI regression gate, bisection engine
└── cli/           Command-line interface
```

## Development

```bash
# Run tests
uv run pytest tests/ -v --cov=calibrax --cov-report=term-missing

# Lint & format
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/

# Type check
uv run pyright src/

# All quality checks
uv run pre-commit run --all-files
```

## License

MIT
