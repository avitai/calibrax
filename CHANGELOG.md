# Changelog

All notable changes to Calibrax are tracked here.

This project follows the spirit of [Keep a Changelog](https://keepachangelog.com/)
and uses semantic versioning while the public API stabilizes.

## [Unreleased]

## [0.1.2] - 2026-08-29

### Changed

- **Requires Python 3.12 or later.** jax 0.11.0 dropped 3.11, and this release
  takes that jax line.
- **The `gpu` extra is renamed `cuda12`.** JAX names its own extras for the CUDA
  major version (`cuda12`, `cuda12-local`, `cuda13`) and publishes no `gpu`
  extra. The hand-named `jax-cuda12-pjrt` and `jax-cuda12-plugin` entries are
  replaced by `jax[cuda12]`, which pulls them in; the stricter NVIDIA floors are
  kept deliberately.
- Resolves to jax 0.11.1, jaxlib 0.11.1 and flax 0.12.9, matching the sibling
  packages.
- `optax` is constrained to 0.2.8 or later. calibrax imports optax nowhere, but
  flax does, and below 0.2.8 optax sets a jax config option removed in jax 0.10,
  so importing flax raises `AttributeError` and takes out collection.
- CI builds every environment from the lockfile, and the pinned actions are
  moved to their current majors.

### Added

- A security policy and issue templates.

## [0.1.1] - 2026-04-30

### Added

- Numerical-equivalence tests for representative regression, classification,
  distance, and divergence metrics against scikit-learn and SciPy references.
- Continuous performance checks with a CodSpeed workflow and focused benchmark
  coverage.
- CRPS as a registered Tier 0 regression metric.
- Optional FFmpeg/libvmaf VMAF video-quality metric boundary.
- Stateful metric plotting via the shared publication plotting infrastructure.
- Contributor, release, security, and code-of-conduct documentation.
- Contributor guide pages for project workflow and adding metrics.
- Peer-comparison guide covering TorchMetrics, jax_metrics, ASV, and CodSpeed.
- Dependabot configuration for Python dependencies and GitHub Actions.
- Structured issue templates for bugs, feature requests, and metric requests.
- Manual generated-release automation in the PyPI publish workflow.

### Changed

- Documentation builds now run with `mkdocs build --strict --clean`.
- Documentation dependencies pin Pygments below 2.20 while the current
  mkdocstrings/pymdown highlighting path passes `filename=None`.
- Metrics documentation now distinguishes the 110 registered Tier 0 metrics
  from Tier 1-3 APIs and metric-learning losses.
- GitHub Actions dependencies were updated through Dependabot.
- Release scheduling now stays under operator control: no commit or tag push
  creates a release by itself.

## [0.1.0] - 2026-04-25

### Added

- Initial public Calibrax release.
- JAX-native benchmarking, profiling, statistical analysis, storage, exporter,
  CI regression, monitoring, and metric APIs.
