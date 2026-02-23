# CLI Reference

Calibrax provides a command-line interface for common benchmarking operations.
All commands operate on a store directory specified via `--data`.

## General Usage

```bash
calibrax <command> [options]
```

## Commands

### `profile`

Profile a JAX function with timing, resource, and optional energy/FLOP measurement.

```bash
calibrax profile --module <PYTHON.PATH> --function <NAME> \
    [--warmup <N>] [--iterations <N>] [--energy] [--flops] [--data <PATH>]
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--module` | Yes | — | Python module path (e.g. `my_pkg.benchmark`) |
| `--function` | Yes | — | Function name within the module |
| `--warmup` | No | `1` | Number of warmup iterations to exclude |
| `--iterations` | No | `10` | Number of timed iterations |
| `--energy` | No | off | Enable energy monitoring |
| `--flops` | No | off | Enable FLOP counting |
| `--data` | No | None | Store directory to persist profiling results |

```bash
calibrax profile --module my_pkg.benchmark --function train_step \
    --warmup 2 --iterations 50
```

```text
Profiling my_pkg.benchmark.train_step
  Warmup: 2, Iterations: 50

Timing Results:
  Wall clock: 5.2340s
  Batches: 52 (warmup excluded: 2)
  Mean batch time: 0.1047s

Profile complete.
```

---

### `ingest`

Import benchmark results from an external JSON file into the store.

```bash
calibrax ingest --data <PATH> --input <FILE>
```

| Option | Required | Description |
|--------|----------|-------------|
| `--data` | Yes | Path to the store directory |
| `--input` | Yes | Path to the JSON file to import |

```bash
calibrax ingest --data ./benchmark-data --input results.json
```

---

### `export`

Export a run to Weights & Biases.

```bash
calibrax export --data <PATH> [--run <ID>] [--project <NAME>] [--entity <NAME>]
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--data` | Yes | — | Path to the store directory |
| `--run` | No | latest | Run ID to export |
| `--project` | Yes | — | W&B project name |
| `--entity` | No | None | W&B entity (team or user) |

```bash
calibrax export --data ./benchmark-data --project my-benchmarks
```

!!! warning "Requires wandb"

    Install with `uv pip install "calibrax[wandb]"` and run `wandb login` first.

---

### `check`

Run a regression check against the stored baseline. Exits with code 1 if any
regressions exceed the threshold — suitable for CI pipeline gating.

```bash
calibrax check --data <PATH> [--threshold <FLOAT>]
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--data` | Yes | — | Path to the store directory |
| `--threshold` | No | `0.05` | Regression threshold (fraction, e.g. 0.05 = 5%) |

```bash
calibrax check --data ./benchmark-data --threshold 0.05
echo $?  # 0 = pass, 1 = regression detected
```

```text
PASSED: No regressions detected (threshold=0.05)
```

---

### `baseline`

Set a run as the active baseline for regression detection.

```bash
calibrax baseline --data <PATH> [--run <ID>]
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--data` | Yes | — | Path to the store directory |
| `--run` | No | latest | Run ID to set as baseline |

```bash
calibrax baseline --data ./benchmark-data --run a1b2c3d4e5f6
```

---

### `trend`

Show metric values over time for a specific point and framework.

```bash
calibrax trend --data <PATH> --metric <NAME> --point <NAME> --framework <NAME> [--n-runs <N>]
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--data` | Yes | — | Path to the store directory |
| `--metric` | Yes | — | Metric name to track |
| `--point` | Yes | — | Point name to filter by |
| `--framework` | Yes | — | Framework tag value |
| `--n-runs` | No | all | Limit to the last N runs |

```bash
calibrax trend --data ./benchmark-data --metric throughput \
    --point forward_pass --framework flax --n-runs 10
```

```text
Trend: throughput for forward_pass (flax)
Timestamp                           Value Commit
--------------------------------------------------------
2026-02-20 11:34:24.346877      1200.0000 -
```

---

### `summary`

Display a summary of a run's metrics and metadata.

```bash
calibrax summary --data <PATH> [--run <ID>]
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--data` | Yes | — | Path to the store directory |
| `--run` | No | latest | Run ID to summarize |

```bash
calibrax summary --data ./benchmark-data
```

```text
Run: 84cb49fb06a9
  Timestamp: 2026-02-20 11:34:24.346877
  Points: 1

Scenario: training
  flax: latency=0.8000, throughput=1200.0000
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Regression detected (from `check`) or runtime error |

## End-to-End Workflow

A typical workflow using the CLI:

```bash
# 1. Run benchmarks and save results to a JSON file
python run_benchmarks.py --output results.json

# 2. Ingest into the store
calibrax ingest --data ./benchmark-data --input results.json

# 3. Set baseline (first time only)
calibrax baseline --data ./benchmark-data

# 4. Run regression check (in CI)
calibrax check --data ./benchmark-data --threshold 0.05

# 5. View trends
calibrax trend --data ./benchmark-data --metric throughput \
    --point forward_pass --framework flax

# 6. Export to W&B
calibrax export --data ./benchmark-data --project my-benchmarks
```
