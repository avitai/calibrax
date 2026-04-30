# Adding a Metric

Metric changes should keep the registry, tests, and documentation aligned. The
current registry contains 111 Tier 0 pure-function metrics across 17 domains;
Tier 1-3 APIs and losses use separate patterns.

## Choose the Tier

| Tier | Use when | Typical location |
|------|----------|------------------|
| Tier 0 pure function | The metric is a stateless JAX function | `src/calibrax/metrics/functional/` |
| Tier 1 frozen backbone | The metric accumulates features from fixed pretrained weights | `src/calibrax/metrics/stateful/` or `plugins/` |
| Tier 2 learned metric | The metric has trainable parameters | `src/calibrax/metrics/stateful/` or `plugins/` |
| Tier 3 metric learning | The function is a differentiable training loss | `src/calibrax/metrics/learning/` |

## Implementation Steps

1. Add the implementation in the existing domain module, or create a focused
   module if the domain is new.
2. Use JAX arrays and `jax.numpy` for numeric operations.
3. Register Tier 0 functions with `MetricEntry` metadata in
   `src/calibrax/metrics/_builtin_registrations.py` or the local registration
   helper already used by the domain.
4. Set the domain, direction, signature, required extra, and mathematical
   properties deliberately.
5. Export the function from the relevant package `__init__.py` if neighboring
   metrics are exported there.
6. Add unit tests for shape handling, edge cases, JIT compatibility when
   expected, and numerical behavior.
7. Add numerical-equivalence tests against scikit-learn, SciPy, TorchMetrics,
   or a paper reference when a reliable implementation exists.
8. Update user-guide and API docs when the metric changes public behavior.

## Numerical Equivalence

Prefer reference-backed tests for standard metrics. The baseline pattern is:

```python
import numpy as np

ABS_TOL = 1e-6

def assert_close(actual, expected):
    np.testing.assert_allclose(float(actual), float(expected), atol=ABS_TOL)
```

Use deterministic arrays and document any semantic differences from the
reference library, such as label averaging, zero-division handling, smoothing,
or log base.

## Required Checks

```bash
source activate.sh
uv run pytest tests/metrics/ -v
uv run pytest
uv run pre-commit run --all-files
uv run mkdocs build --strict --clean
```

If the targeted test run uses repository-wide coverage options, add `--no-cov`
for quick local iteration and rely on the full `uv run pytest` command for the
coverage gate.
