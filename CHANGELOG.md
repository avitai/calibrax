# Changelog

All notable changes to Calibrax are tracked here.

This project follows the spirit of [Keep a Changelog](https://keepachangelog.com/)
and uses semantic versioning while the public API stabilizes.

## [Unreleased]

### Added

- Three metric domains, moved from the sibling packages with their tests and given
  the Tier 0 conventions (positional inputs, scalar means, static shape checks):
  `forecasting` (fair CRPS, energy score, rank and PIT histograms, spread-skill ratio,
  ranked probability score with its ensemble and skill-score forms, event
  reliability; from opifex), `uncertainty` (PICP, MPIW, interval and Winkler score,
  Gaussian NLL, regression calibration error, predictive entropy, ensemble mutual
  information, ANEES, non-credibility index, chi-squared credibility interval; from
  opifex) and `generative` (k-NN manifold precision and recall, density-weighted
  variants, distance to closest record, memorization rate; from artifex).
  `relative_l2_error` and `per_sample_relative_l2` join the general domain (from
  opifex) and `kolmogorov_smirnov_distance` the divergence domain (from artifex);
  `matrix_sqrtm` is a shared helper. The registry holds 132 Tier 0 metrics across
  20 domains.

### Fixed

- `HARDWARE_SPECS["tpu_v5e"]` carried a 1.6 TB/s memory bandwidth, the TPU v6e figure;
  the v5e chip has 819 GB/s, so its ridge point is 240 FLOPs/byte rather than 123 and
  the roofline classified memory-bound work on v5e as compute-bound. Each entry's
  `critical_intensity` is now derived from its two published figures instead of typed
  beside them, and the tests pin the figures to the vendor specifications.

### Changed

- **Floors match what is tested.** `jax>=0.11.1`, `jaxlib>=0.11.1` and `flax>=0.12.9` are
  the versions every release since 0.1.2 has resolved and run CI against; the previous
  `>=0.4.0` and `>=0.12.1` floors promised compatibility nothing checked. `numpy` is
  `>=2.1` (jax's own floor) with no ceiling: the `<2.5.0` cap was a snapshot from a
  dependency refresh with no recorded reason, and the lock now resolves numpy 2.5.
- CI checks that `uv.lock` is current, runs the tests on Python 3.13 as well as 3.12,
  uploads coverage from the 3.12 lane (the old condition named 3.11 and never fired),
  checks distributions with `twine check --strict`, and installs bandit and pip-audit
  from the `dev` extra with bandit blocking at medium severity. Pre-commit gains
  import-linter (the layering `cli > ci, metrics > exporters, monitoring > analysis,
  statistics, storage, validation > core > profiling`), interrogate and
  validate-pyproject; the standalone pydocstyle hook is replaced by ruff's docstring
  rules under the Google convention. Pyright runs in `standard` mode on `src/`.
- Lint now enforces the annotation, argument, bugbear, comprehension, complexity, docstring,
  naming, pylint, pytest-style, pathlib, return, security, simplification, print and
  try-except rule families (the same set as substrax), with the reasons for every ignored
  rule recorded in `pyproject.toml`. The code changes it required: every threshold in the
  profiling, monitoring, fairness and analysis modules is a named module constant;
  `BisectionEngine` resolves `git` through `shutil.which` and fails fast when it is
  missing; `zip` calls over paired sequences are strict; `BenchmarkAdapter` is a plain
  base class (it never declared an abstract method); `Store.ingest` lost its unused
  `format` parameter.

### Fixed

- `FlopsCounter` counted 0 FLOPs for anything inside a nested `jax.jit`, a
  `jax.checkpoint` or a custom-derivative call such as `jax.nn.relu`, and skipped
  `lax.cond` branches: its hand-maintained primitive table had not followed jax's
  primitive names (`pjit` became `jit` in jax 0.7). It now reads XLA's cost analysis
  of the function's lowering, the estimate `flax.nnx.tabulate` reports, so nested
  computations count and the table is gone.

### Changed

- `FlopsResult` reports `total_flops` and `transcendentals` (XLA counts `sin`, `exp`
  and friends separately); `flops_by_operation` and `num_operations` are removed,
  since XLA reports no per-primitive breakdown and nothing outside the CLI's print
  read them. A reduction over `n` elements now counts `n - 1`, and a convolution
  counts only the taps that touch real data.
- `FlopsCounter.count` raises `FlopsUnavailableError` (a `ValueError`) when the
  function contains a custom call XLA has no cost model for, instead of counting it
  as zero with a warning.

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
