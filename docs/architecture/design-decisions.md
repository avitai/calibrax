# Design Decisions

This page documents the key architectural decisions behind Calibrax, the
reasoning that led to each, and their implications.

## 1. A Dedicated Benchmarking Framework for SciML

**Problem:** Scientific ML practitioners working with JAX-based frameworks face
fragmented benchmarking tooling. General-purpose benchmark suites like `pytest-benchmark`
or `airspeed velocity` lack domain-specific concepts — they do not understand that
throughput should be maximized while latency should be minimized, cannot track GPU
memory or energy consumption, and have no built-in support for regression detection
across model training pipelines. Teams end up writing ad-hoc benchmarking scripts
that are inconsistent, hard to compare, and impossible to integrate into CI.

**Decision:** Build a standalone benchmarking framework purpose-built for
scientific ML workloads on JAX. Calibrax provides a unified data model for
benchmark results, direction-aware metric semantics, JAX-native profiling
(GPU sync, FLOP counting via jaxpr analysis, roofline analysis, XLA
compilation profiling, carbon emissions tracking), and a full pipeline from
measurement through statistical analysis to CI gating.

**Implications:**

- Single, well-tested toolkit for defining, collecting, analyzing, and exporting
  benchmark data across any JAX-based project
- Domain-specific metric semantics (direction, units, confidence intervals) are
  built into the data model rather than bolted on
- Projects adopt Calibrax as a dependency and focus on their domain logic — the
  benchmarking infrastructure is handled

## 2. Composition over Inheritance

**Problem:** Benchmark result hierarchies (`BenchmarkResult → TimedResult →
ProfiledResult`) are fragile and hard to extend. Adding a new profiling
dimension (energy, FLOPs) forces changes through the entire class tree, and
every benchmark must carry fields it may not use.

**Decision:** `BenchmarkResult` is a flat dataclass with optional composed
objects (`timing: TimingSample | None`, `resources: ResourceSummary | None`).
Each concern is modeled by a separate, independent dataclass.

**Implications:**

- New profiling dimensions can be added without modifying existing types
- Partial results are natural — not every benchmark needs GPU profiling
- Serialization is straightforward: each component handles its own `to_dict()`

## 3. Protocol-Driven Design

**Problem:** Requiring benchmark classes to inherit from an ABC couples them to
Calibrax and makes wrapping third-party code tedious.

**Decision:** Use Python `typing.Protocol` with `@runtime_checkable` for
`BenchmarkProtocol`, `DatasetProtocol`, `MetricProtocol`, etc. Any object with
matching methods satisfies the protocol via structural subtyping.

**Implications:**

- No base class required — third-party code works without modification
- `isinstance()` checks work at runtime for validation
- Type checkers verify protocol conformance statically

## 4. Direction-Aware Metrics

**Problem:** Regression detection and ranking code frequently has bugs where
higher-is-better and lower-is-better metrics are compared with the wrong
inequality. In scientific ML, throughput should increase while latency, loss,
and energy consumption should decrease — getting this wrong invalidates results.

**Decision:** Every metric declares its direction via `MetricDef.direction`
(`HIGHER`, `LOWER`, or `INFO`). All analysis functions read this field to
determine comparison semantics automatically.

**Implications:**

- Regression detection, ranking, Pareto fronts, and aggregate scoring all
  work correctly without manual polarity flags
- `INFO` metrics (timestamps, commit hashes) are automatically excluded from
  regression checks
- New metrics only need to declare their direction once

## 5. Two Adapter Hierarchies

**Problem:** Flax NNX models need to be JIT-compiled with `nnx.jit`, which
requires the adapter itself to be an `nnx.Module`. But non-JAX targets
(PyTorch, scikit-learn) should not depend on NNX.

**Decision:** Two adapter base classes:

- `BenchmarkAdapter(ABC)` — for non-NNX targets, using standard inheritance
- `NNXBenchmarkAdapter(nnx.Module)` — inherits from NNX, participates in JAX
  transformations

**Implications:**

- NNX adapters can be passed to `nnx.jit`, `nnx.vmap`, and `nnx.grad`
- Non-NNX adapters have no JAX dependency in their implementation
- The `AdapterRegistry` resolves the correct adapter based on `can_adapt()`

## 6. JSON-Per-Run Storage

**Problem:** SQLite and binary formats add deployment complexity (drivers,
migrations, binary compatibility). Benchmark data is small (kilobytes per run)
and accessed sequentially.

**Decision:** Each run is stored as a standalone JSON file in a flat directory.
Baselines are stored in a separate `baselines/` directory.

**Implications:**

- No database driver or schema migrations required
- Files are human-readable and git-friendly (easy to diff and review)
- Trade-off: querying across many runs requires loading all files, which is
  acceptable for typical benchmark stores (hundreds to low thousands of runs)

## 7. Optional Dependencies

**Problem:** scipy, wandb, matplotlib, mlflow, codecarbon, and ruptures are
heavy dependencies that many users do not need. Making them required would
bloat the install and break on minimal environments.

**Decision:** Optional dependencies are guarded by `try/except ImportError` at
the module level, setting availability flags like `WANDB_AVAILABLE`,
`MATPLOTLIB_AVAILABLE`, `CODECARBON_AVAILABLE`, `RUPTURES_AVAILABLE`, and
`MLFLOW_AVAILABLE`. Features degrade gracefully: exporters and trackers raise
`ImportError` on instantiation when their dependency is missing.

**Implications:**

- Base install is lightweight (JAX + standard library)
- Users install only the extras they need (`calibrax[stats]`, `calibrax[wandb]`,
  `calibrax[mlflow]`, `calibrax[codecarbon]`, `calibrax[changepoint]`)
- Heavy optional modules (`WandBExporter`, `MLflowExporter`) are not re-exported
  from their package `__init__.py` to avoid triggering an import-time load
- Change point detection (`ruptures`) and carbon tracking (`codecarbon`) follow
  the same pattern: import guard at module top, `ImportError` on use

## 8. Frozen Dataclasses

**Problem:** Mutable benchmark results are error-prone — accidental mutation
during analysis or export can corrupt data and produce non-reproducible results.

**Decision:** All data model classes use `@dataclass(frozen=True, slots=True,
kw_only=True)`. Results are immutable after construction.

**Implications:**

- Results cannot be accidentally modified during analysis, export, or storage
- Hashable by default, so results can be used as dict keys or set members
- Creating modified versions requires constructing a new instance (use
  `dataclasses.replace()`)
- Compatible with JAX's functional paradigm, which expects immutable data

## 9. Integration-First Ecosystem Design

**Problem:** The JAX benchmarking ecosystem already has mature, specialized tools
for tasks like change point detection, carbon tracking, experiment tracking, and
trace visualization. Reimplementing these capabilities would duplicate effort,
introduce maintenance burden, and produce inferior results compared to the
established tools.

**Decision:** Integrate with existing ecosystem tools rather than building from
scratch. Calibrax provides thin adapter layers that connect its data model to
external tools:

- **ruptures** for change point detection in benchmark trends
- **CodeCarbon** for carbon emissions and energy tracking
- **MLflow** for experiment tracking (alongside the existing W&B exporter)
- **ASV** format export for interoperability with `airspeed velocity`
- **jax.profiler.trace()** for XLA timeline profiling (linked to run metadata)

**Implications:**

- Each integration is a thin wrapper (50-150 lines) that translates between
  Calibrax's data model and the external tool's API
- Users benefit from active upstream development and bug fixes
- Optional dependencies keep the base install clean
- Calibrax focuses on the orchestration layer — connecting measurement, analysis,
  and export — rather than reimplementing domain-specific algorithms

## 10. Hardware Abstraction Layer

**Problem:** Profiling features like roofline analysis, compilation efficiency
scoring, and complexity analysis need hardware-specific constants (peak FLOP/s,
memory bandwidth) to produce meaningful results. Hardcoding these per-module
duplicates values and makes it impossible to extend for new hardware.

**Decision:** Centralize hardware specs in `calibrax.profiling.hardware` with a
`HARDWARE_SPECS` dictionary containing reference values for common accelerators
(TPU v5e, A100, H100, CPU) and a `detect_hardware_specs()` function that
auto-detects the active JAX backend.

**Implications:**

- Roofline analyzer, compilation profiler, and complexity analysis all share
  the same hardware specs — no duplication
- Adding support for new hardware requires a single dictionary entry
- `detect_hardware_specs()` returns sensible defaults for unknown platforms,
  so profiling works everywhere (with reduced accuracy on unrecognized hardware)
