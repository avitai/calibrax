# Changelog

All notable changes to Calibrax are tracked here.

This project follows the spirit of [Keep a Changelog](https://keepachangelog.com/)
and uses semantic versioning while the public API stabilizes.

## [Unreleased]

### Changed

- The README names calibrax's one Avitai dependency, substrax, instead of saying it depends on
  none of the others, and no longer claims configurable severity levels for regression
  detection, which compares each metric against a threshold.
- `scripts/derive_status.py` checks every documented copy of the Tier 0 metric count and domain
  count across the README and the documentation, the README's domain list, and the per-domain
  table in the metrics overview, and fails when a listed document drops its claim. The copies
  that had drifted are corrected: `docs/contributing/index.md` said 17 domains, the README's
  domain list named `regression` and omitted `general`, `forecasting`, `uncertainty` and
  `generative`, and the metrics-overview table summed to 131 rather than 137.

### Fixed

- `FIDMetric` took the square root of the eigenvalues of the plain product of the two
  covariance matrices. That product is not symmetric, so the trace term was wrong whenever
  the covariances did not commute: between 2.8% and 21.7% high on 4- to 64-dimensional
  features and 49.5% high on 2048-dimensional Inception features. The plugin now computes
  through `generative.frechet_feature_distance`, and `InceptionScoreMetric` through
  `generative.inception_score` with a single split, so each score has one implementation.
- `BERTScoreMetric` reported F1 of the averaged precision and recall. It now averages
  per-pair F1, as the reference BERTScore does; the two differ whenever pairs differ.
- `generative.frechet_feature_distance` returned NaN when a side had one sample. It raises
  `ValueError`, and so does `FIDMetric.compute()` with fewer than two accumulated samples.
- `generative.frechet_feature_distance` lost precision on correlated features, and its result
  depended on the LAPACK build.
  - **Cause.** It formed both covariance matrices in float32 and took eigendecomposition
    square roots of one covariance and of the product `S_r^{1/2} S_g S_r^{1/2}`. Forming a
    covariance from a feature matrix squares that matrix's condition number, and correlated
    features give covariances with condition numbers up to about 5e7. Their smallest
    eigenvalues then sit below float32 resolution, so the square roots of those eigenvalues are
    rounding noise. On such features the result moved by up to 4.3e-4 relative under
    float32-scale input noise, and by up to 4.7e-5 between LAPACK's symmetric eigensolvers.
    The new equivalence test passed on Linux (4.3e-7 off) and failed on both macOS lanes
    (1.3e-4 off). Near-rank-deficient features, and fewer samples than features, were up to
    5.4e-4 off on every platform.
  - **Identity.** The cross term equals the trace norm of `S_r^{1/2} S_g^{1/2}`, the sum of its
    singular values: `tr (S_r^{1/2} S_g S_r^{1/2})^{1/2} = ||S_r^{1/2} S_g^{1/2}||_*`
    (Bhatia, Jain and Lim, "On the Bures-Wasserstein distance between positive definite
    matrices", Expositiones Mathematicae, 2019, arXiv:1712.01504, Remark 1). Mathiasen and
    Hvilshøj, "Fast Fréchet Inception Distance" (arXiv:2009.14075), compute this term from the
    centred feature matrices instead of the covariances. In 32-bit precision they report errors
    at least 1000 times smaller than `scipy.linalg.sqrtm`.
  - **Computation.** Each centred feature matrix is factored with a thin QR, `X - mean = Q R`.
    Since `X_r X_g^T = Q_r (R_r R_g^T) Q_g^T` and each `Q` has orthonormal columns, the cross
    term is the sum of the singular values of `R_r R_g^T` divided by `sqrt((n_r - 1)(n_g - 1))`,
    and `tr S = ||R||_F^2 / (n - 1)`. No covariance is formed. This differs from Fast FID in two
    ways, neither taken from a published method.
    - The QR keeps the matrix at most d x d, whatever the sample count.
    - Singular values are taken directly, rather than eigenvalues of the Gram matrix
      `(C_r^T C_g)(C_g^T C_r)`, which would square the condition number again. This is the same
      reason least-squares solvers prefer QR to the normal equations.
  - **Evidence and its limits.** The float64 reference is SciPy's `sqrtm`, cross-checked
    against the float64 identity.
    - The new form stayed within 1.2e-6 relative of that reference on synthetic Gaussian
      features of 16 to 2048 dimensions, including near-rank-deficient features and fewer
      samples than features.
    - It moved by at most 1e-6 under the same input noise, and between LAPACK's SVD drivers.
    - These measurements used CPU LAPACK on Linux. Real Inception features and GPU backends
      were not measured separately.
    - `generative.frechet_distance`, which receives covariances rather than features, keeps
      the eigendecomposition form and its float32 limit.

## [0.1.5] - 2026-09-09

### Added

- `text.perplexity` takes a `mask` so padding and prompt positions stay out of the mean;
  a mask that keeps nothing scores as infinite perplexity. Language-model evaluations in
  artifex score masked token log-probabilities through it instead of carrying their own
  masked mean.

## [0.1.4] - 2026-09-09

### Added

- `generative.frechet_distance` (from Gaussian statistics) and `frechet_feature_distance`
  (from feature matrices), the core of the Fréchet Inception Distance on the
  eigendecomposition square root; `generative.inception_score` with
  `inception_score_per_split` for its error bar; `statistical.correlation_preservation`
  for synthetic-data fidelity, `statistical.autocorrelation` for sequence batches and
  `statistical.skewness`; `geometric.rmsd` over the atoms two masked conformations share and
  `geometric.pairwise_rmsd`. Moved from artifex's evaluation helpers with their behaviour
  pinned by tests, so artifex's metric classes can wrap calibrax instead of carrying copies.
  Five of them register as Tier 0 metrics.

## [0.1.3] - 2026-09-09

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

- Depends on `substrax>=0.1.0`. Hardware identity (`detect_hardware_specs`, the
  adaptive-operation platform choice) is read from `substrax.devices.detect_devices()`
  rather than from `jax.default_backend()` and `jax.devices()` directly; the spec table
  and the returned dictionary shape are unchanged.
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
