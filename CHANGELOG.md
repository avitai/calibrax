# Changelog

All notable changes to Calibrax are tracked here.

This project follows the spirit of [Keep a Changelog](https://keepachangelog.com/)
and uses semantic versioning while the public API stabilizes.

## [Unreleased]

### Added

- `calibrax.statistics.bootstrap_interval(statistic, *data, key, num_resamples, confidence)`,
  the percentile bootstrap interval (Efron and Tibshirani 1993; `scipy.stats.bootstrap(method=
  "percentile")`, which the tests use as the reference) with arrays resampled together, every
  resample evaluated in one `jax.vmap`, and a `BootstrapInterval` result (`value`, `lower`,
  `upper`, `samples`, a pytree). The quantiles interpolate as `numpy.percentile` does, which is
  exact for equal neighbours where `jnp.quantile` is one float32 ulp off.

### Changed

- Requires `substrax>=0.1.14` and `pydantic>=2.10`; the lock moves substrax from 0.1.11 and
  nothing else (pydantic was already locked through the optional extras). Calibrax's `setup.sh`
  writes its managed environment file with `python -m substrax.runtime.managed_env` (substrax
  0.1.12), records are typed with `substrax.typing.JsonValue` (0.1.13) and read with
  `substrax.records.read_record` (0.1.14).
- Every record's `from_dict` takes `Mapping[str, JsonValue]` and reads through `read_record`:
  each field is checked against its annotation, and a malformed record raises
  `pydantic.ValidationError` naming each field's path, where a string in a number field or a
  number in a string field was stored as given. JSON integers in float fields read as floats.
  `to_dict` returns `dict[str, JsonValue]`. A stored `Run` must carry `id` and `timestamp`, and a
  stored `BenchmarkResult` its `timestamp`; the old reader dated a `BenchmarkResult` without one at
  0.0. Every file `to_dict` writes reads as before: all 86 stored runs and results in datarax and
  cellifex read to the same values.
- `calibrax.core` loads each export on first use (scientific-python SPEC 1 through
  `lazy-loader>=0.5`, from the package's `__init__.pyi`, which type checkers read), so
  `calibrax.core.models`, `calibrax.core.registry`, `calibrax.core.record_values`,
  `calibrax.storage`, `calibrax.ci`, `calibrax.analysis` and `calibrax.validation` import without
  JAX or Flax; importing any of them loaded both through the package's adapters and protocols.
- `calibrax.profiling.flops.cost_mapping(cost)` narrows what `jax.stages.Lowered.cost_analysis()`
  returns, which jax types `Any` and documents as arbitrary (a mapping, a list holding one, or
  `None`), to its numeric fields; `FlopsCounter` reads the analysis through it, and
  `FlopsCounter.count(fn, *args)` is typed with `substrax.typing.PyTree`.
- Monitoring reports are typed: `AdvancedMonitor.get_monitoring_summary()` returns a
  `MonitoringSummary` (`thresholds`, `alert_count`, `metric_history` of `MetricHistorySummary`,
  `is_monitoring`) and `ProductionMonitor.get_pipeline_health_report()` a
  `PipelineHealthReport` (`pipelines` of `PipelineStats`, `overall_health`, `baselines`,
  `total_executions`), with health levels as `PipelineHealth`; read them by attribute where
  they were dicts. `ProductionMonitor` takes `AdvancedMonitor`'s three keyword arguments
  instead of `**kwargs`, and `ProductionMonitor.executions` returns the recorded
  `PipelineExecution`s with their metadata, which nothing could read before.
- `Run.environment`, `Run.metadata`, `BenchmarkResult.metadata` and `BenchmarkResult.config` are
  `calibrax.core.record_values.Metadata`: JSON values and the JAX or NumPy scalars a computation
  produces (`MetadataValue`), written as Python numbers by `to_dict`, in place of
  `dict[str, Any]`. A consumer reading a structured value from them narrows it.
- `sliced_wasserstein` computes `SW_p = (mean over directions of W_p^p)^(1/p)` (Bonneel et al.
  2015; Nadjahi et al. 2020, eq. 5; POT). It returned the mean of the per-direction `W_p`, which
  for `p > 1` is lower: 0.341 against 0.379 on a 10-dimensional Gaussian pair. `key` is
  required, as a key or an `nnx.Rngs` (its `sample` or `default` stream), through
  `substrax.rng.key_from`; the
  silent `PRNGKey(42)` default is gone. `num_projections` defaults to 256
  (`SLICED_WASSERSTEIN_PROJECTIONS`), unequal sample counts raise `ValueError`, and the gradient
  at identical samples is finite.
- The registry's `sliced_wasserstein` entry is `registry_sliced_wasserstein`, which fixes the
  directions with `SLICED_WASSERSTEIN_REGISTRY_SEED` so suite runs compare like with like, and is
  marked `is_true_metric=False`: over a fixed set of directions the value is a pseudometric.
- `rmse` takes the keyword-only `mask`, `weights`, `reduction` and `axis` of `mse`: the root of
  the (masked, weighted) mean over `axis`, then `reduction` over the remaining roots (`"none"`,
  `"mean"`, `"sum"`; `"batch_sum"` raises). Its gradient at a perfect prediction is 0 where it
  was NaN, through the new `safe_root` helper, which `sliced_wasserstein` shares.
- Distances built on a root have a finite gradient at a perfect match, where it was NaN:
  `euclidean_distance`, `mahalanobis_distance`, `minkowski_distance`, `hellinger_distance`,
  `mmd`, `rmsd`, `spectral_distance`, `graph_edit_distance_approx`,
  `spd_log_euclidean_distance`, `stiefel_distance`, `relative_error` and `relative_l2_error`.
  `energy_score`'s gradient was NaN for every ensemble, because each member's distance to
  itself entered the spread term through `jnp.linalg.norm`; it is finite. They take their roots
  through `safe_root` and the new `safe_norm` (the JAX FAQ's inner-and-outer `where`, with
  derivative 0 at 0 as in `optax.safe_norm`); values are unchanged.
- `randers_distance(a, b, *, direction, magnitude)` replaces `drift=`: the drift is
  `magnitude * direction / ||direction||`, `magnitude` is a Python float checked to lie in
  `[0, 1)` (`TypeError` for an array, `ValueError` outside the range), and the direction may be
  traced. The old check read `float(norm(drift))`, which cannot run inside `jax.jit`; the
  magnitude-and-direction form follows Finsler MDS (Dages et al. 2025). Pass the old `drift` as
  `direction=drift, magnitude=float(jnp.linalg.norm(drift))`.
- The functional metrics are typed with `jax.typing.ArrayLike` for array inputs and `jax.Array`
  for array outputs, JAX's recommendation for public APIs, in place of `Any`; private helpers
  that receive converted arrays take `jax.Array`. pyright strict checks them.
- `MetricFn`, a metric function returning a `jax.Array`, types the registry, composition,
  wrappers and fairness helpers; they declared `Callable[..., float]` although every metric
  returns an array; it admits a Python float for the host-side text metrics. `MetricValues` types
  `calculate_all`'s result.
- `MetricLearningLoss.__call__(embeddings, labels)` takes array-likes and no `**kwargs`: every
  loss ignored them (`ContrastiveLoss`, `TripletMarginLoss`, `NTXentLoss`), so they were a
  suppressed unused argument, not an interface.
- `coverage(items, *, catalog_size)` drops the unused `relevance` argument, counts distinct items
  with a fixed-length `jnp.bincount`, so it runs under `jax.jit` (`catalog_size` static) and
  `jax.vmap` where `jnp.unique` could not, and ignores ids outside `[0, catalog_size)`; a negative
  id had counted as item 0. Its registry entry is `MetricSignature.CUSTOM`, since a suite cannot
  call it as `fn(predictions, targets)`.
- `BenchmarkAdapter` is an abstract base class with an abstract `can_adapt`, as the architecture
  notes describe it; the base returned `False` for every target, so an adapter that did not
  override it could never be selected by `AdapterRegistry`. Adapters implement `can_adapt`.
- `LearnedMetric.__init__(name)` no longer takes `rngs`, which it ignored; a subclass creates its
  layers with its own `nnx.Rngs` (`LPIPSMetric` and the user-guide example updated).
- `BootstrapMetric` and `StatisticalAnalyzer` compute through `bootstrap_interval`; they were two
  host-side loops with their own interval rules and fixed default seeds (0 and 42).
  `BootstrapMetric(metric, num_resamples=, confidence=).compute(predictions, targets, *, key)`
  returns a `BootstrapInterval` instead of a dict, and `num_bootstraps` and `seed` are gone.
  `StatisticalAnalyzer(*, key, bootstrap_resamples=)` takes a key in place of `seed` and splits it
  on each call, so successive calls draw fresh resamples and the same key reproduces them.
- `ThresholdMetric.evaluate` returns a `ThresholdResult` (`value`, `passed`, `threshold`,
  `metric_name`) instead of a dict. The composition classes and wrappers type their arrays as
  `ArrayLike`, and a metric collection's pass-through keywords as `object`.
- `FrozenBackboneMetric[FeaturesT]` is generic in the features its `_extract_features` returns
  and `_accumulate` receives (`FIDMetric` and `BERTScoreMetric` use `dict[str, jax.Array]`,
  `InceptionScoreMetric` `jax.Array`); `update` takes array keywords. `DatasetProtocol[ItemT]` and
  `BatchableDatasetProtocol[ItemT]` are generic in their items, and the batchable protocol extends
  the plain one; `get_batch` returns `dict[str, jax.Array]` and `MetricProtocol.compute`
  `jax.Array | float`. `LPIPSMetric.update(*, features_a, features_b)` names its per-layer feature
  sequences instead of reading them from `**kwargs`. The scientific plugin functions take
  array-likes and return arrays.
- `time_calls(call, *, warmup, iterations, percentiles, sync)` times a zero-argument callable:
  close over the inputs (`lambda: step(state, batch)`). It took `func, *args, **kwargs` beside its
  own keyword options, so a function with a `warmup`, `iterations`, `percentiles` or `sync` keyword
  could not be timed. `TimingCollector.measure_iteration` is generic in the batch type, and results
  handed to a sync function are `substrax.typing.PyTree`.
- `CompilationProfiler.profile_jit_compilation` keeps the wrapped function's signature
  (`Callable[P, R]`) and gives each wrapper its own compiled functions. The profiler keyed its
  one cache on the function's name, so two functions named alike (two lambdas) with matching
  input shapes shared a compiled function and the second returned the first's result. Results
  are waited for with `jax.block_until_ready`, through nested pytrees; `reset()` makes each
  wrapper compile again. The unread shape, dtype and timestamp records are gone.

### Fixed

- `CarbonTracker(country_iso_code=...)` uses codecarbon's `OfflineEmissionsTracker`, the tracker
  that takes a country. `EmissionsTracker` refuses `country_iso_code` with `TypeError`, and the
  fallback that caught it retried without the country, so the requested country was dropped
  without a warning and the machine's detected location was used instead.
- `analyze_complexity` counts parameter memory from each parameter's dtype. It assumed four bytes
  per parameter, so a bfloat16 model's parameter memory read twice its size and a float64 model's
  half; parameter counts and the operation estimate use exact Python integers, where a product
  of an input shape computed as an int32 array could overflow.

### Removed

- `scripts/setup_env.py`. `setup.sh` runs `python -m substrax.runtime.managed_env write --prefix
  CALIBRAX`, which writes the memory fraction as `XLA_CLIENT_MEM_FRACTION` and always unsets the
  deprecated `XLA_PYTHON_CLIENT_MEM_FRACTION` (jaxlib refuses both at once); inspect the layering
  with `python -m substrax.runtime.managed_env show --prefix CALIBRAX --env-file .calibrax.env
  --user-env .env --user-env .env.local`. A user-owned `.env` that still exports the deprecated
  name overrides the managed file and stops JAX's CUDA backend from starting.

### Security

- `BisectionEngine.bisect` resolves `good_commit` and `bad_commit` to commit hashes with
  `git rev-parse --verify --end-of-options <ref>^{commit}` before any other git command, and
  refuses a ref that names no commit with `ValueError`. Refs were placed in git's argument list
  as given, so one beginning with `-` was parsed as an option. Branch names and tags keep
  working, the culprit is reported as a full hash, and a range starting at the root commit,
  which failed on `root^`, bisects.
- `vmaf_score` escapes its libvmaf options at both levels FFmpeg's filtergraph syntax defines, so a
  `model` string holding `:`, `,`, `;` or brackets stays one option value instead of adding options
  or filters; it passes inputs as absolute `file:` URLs, so a name holding `:` is not read as
  another protocol; and it resolves `ffmpeg` with `shutil.which`, raising `RuntimeError` when it
  is missing.
- The lock moves anyio from 4.12.1 to 4.14.2 for CVE-2026-63374 and CVE-2026-64847; nothing else moves. 4.14.2 is the first fixed release; 4.15.1 needs typing-extensions 4.16.0, which a single-package upgrade does not allow to move.

## [0.1.9] - 2026-09-18

### Changed

- Requires `substrax>=0.1.11`; the lock moves it from 0.1.9. Calibrax uses substrax's device
  detection only, which 0.1.10 and 0.1.11 leave as it was; 0.1.11 caps jax below 0.11.2, whose
  renamed `jax.experimental.hijax.HiPrimitive` flax 0.12.9 imports at module load, and a
  resolver given `substrax>=0.1.10` keeps jax 0.11.2 and picks 0.1.10 instead, so a fresh
  install of calibrax resolved the failing pair until the floor moved. The build-verification
  smoke installs the wheel with `--refresh` and imports `calibrax.core.adapters`, which loads
  the stack; `import calibrax` alone never did.

### Security

- The lock moves cryptography from 49.0.0 to 50.0.1 for GHSA-g6cj-pr64-35w5. mlflow 3.15.2
  required cryptography below 50, so mlflow (the `mlflow` extra, with its skinny and tracing
  wheels) moves to 3.16.1, which also leaves the affected range of GHSA-h7x2-h6g9-p789 (the
  AI gateway's `api_base` flaw, affected through 3.15.2). The exporter calls the tracking
  client only (`start_run`, `set_tracking_uri`, `set_experiment`, `log_metric`, `log_params`,
  `log_param`, `log_artifact`), which 3.16 leaves as it was; its breaking changes are the
  server's basic-auth default, pyspark below 3.4.4 and the gateway's static prefix.

## [0.1.8] - 2026-09-17

### Added

- `reduce_values(values, *, mask, weights, reduction, axis)` is public, from `calibrax.metrics`
  and `calibrax.metrics.functional`: the reduction every calibrax loss uses (a mask excludes
  elements, weights give the weighted mean, an empty selection is `0.0`, `"none"`, `"mean"`,
  `"sum"` and `"batch_sum"`), so a consumer reduces its own element-wise losses the same way
  instead of keeping a reducer of its own.

## [0.1.7] - 2026-09-17

### Added

- `charbonnier_loss`, the differentiable L1 ``(e^2 + eps^2)^(alpha / 2)``, registered as a
  regression loss; artifex's copy retires onto it.
- `softmax_cross_entropy(logits, labels)`, the cross-entropy of integer labels under the softmax
  of `logits` with the class axis last, registered under the classification domain with the
  `CUSTOM` signature; DiffBio's `cross_entropy_loss` retires onto it.
- Every loss (`mse`, `mae`, `huber_loss`, `charbonnier_loss`, `relative_l2_error`,
  `softmax_cross_entropy`) takes keyword-only `mask`, `weights`, `reduction` (`"none"`, `"mean"`,
  `"sum"`, `"batch_sum"`) and `axis`, reduced by one shared function: a mask excludes elements
  from sums and means, weights give the weighted mean `sum(w x) / sum(w)`, and a mean over no
  element (an all-false mask, weights summing to zero) is `0.0`, one documented finite result a
  caller can check at the host boundary. jit and grad work through the mask.

### Changed

- `time_calls(func, *args, warmup=3, iterations=10, percentiles=(50, 90, 99), sync=..., **kwargs)`
  times a callable and reports the median and percentiles of its timed calls (`CallTiming`),
  with `jax.block_until_ready` over the whole result pytree as the default sync; pass
  `jax.jit(f)` to time the compiled program. `RooflineAnalyzer` measures through it and reports
  the median. `TimingCollector` built without `sync_fn` now waits for each batch result with
  `jax.block_until_ready` where it waited for nothing, so timings it records are not comparable
  with baselines recorded before this release; `TimingSample.to_dict` keeps its fields, which
  the JSON run store persists.
- pytest measures coverage by source directory, `--cov=src/calibrax` in addopts and in the CI
  test command; a package name matched executed code by module, so a package file run as a
  script reported 0%. `tests/ci/test_ci_coverage.py` fails on a `--cov` that is not a directory.
- Requires `substrax>=0.1.9`, the latest release; the lock moves from 0.1.0.
- `fbeta_score` and `f1_score` with `average="macro"` or `"weighted"` return the mean (or
  support-weighted mean) of the per-class F-beta, as scikit-learn and torchmetrics define it. They
  returned the F-beta of the averaged precision and recall, which is not the same number, so
  published macro and weighted F scores change. `precision`, `recall`, `fbeta_score` and
  `f1_score` take `num_classes`; passed statically it makes the per-class averages jit-compatible.
- `MetricCollection.from_registry` admits only metrics with the `PREDICTIONS_TARGETS` signature,
  because `compute_functional` calls every member with that pair; `crps` (ensemble signature)
  no longer joins a `general` collection that would have called it with plain predictions.

### Removed

- `calibrax.profiling.measure_execution_time`; `time_calls` replaces it and reports the median
  rather than a mean, which one slow call moves.

## [0.1.6] - 2026-09-11

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
