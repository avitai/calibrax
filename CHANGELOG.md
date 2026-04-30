# Changelog

All notable changes to Calibrax are tracked here.

This project follows the spirit of [Keep a Changelog](https://keepachangelog.com/)
and uses semantic versioning while the public API stabilizes.

## [Unreleased]

### Added

- Numerical-equivalence tests for representative regression, classification,
  distance, and divergence metrics against scikit-learn and SciPy references.
- Contributor, release, security, and code-of-conduct documentation.
- Contributor guide pages for project workflow and adding metrics.
- Peer-comparison guide covering TorchMetrics, jax_metrics, ASV, and CodSpeed.
- Dependabot configuration for Python dependencies and GitHub Actions.
- Structured issue templates for bugs, feature requests, and metric requests.

### Changed

- Documentation builds now run with `mkdocs build --strict --clean`.
- Documentation dependencies pin Pygments below 2.20 while the current
  mkdocstrings/pymdown highlighting path passes `filename=None`.
- Metrics documentation now distinguishes the 110 registered Tier 0 metrics
  from Tier 1-3 APIs and metric-learning losses.

## [0.1.0] - 2026-04-25

### Added

- Initial public Calibrax release.
- JAX-native benchmarking, profiling, statistical analysis, storage, exporter,
  CI regression, monitoring, and metric APIs.
