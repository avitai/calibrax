# Statistical Analysis

Calibrax provides tools for summarizing benchmark measurements with confidence
intervals, detecting outliers, running significance tests, and computing effect
sizes.

## Summarizing Samples

`summarize` describes a sample — where it sits, how far it spreads, and whether that spread is
small enough to call the measurement stable:

```python
import jax.numpy as jnp
from calibrax.statistics import summarize

summary = summarize([0.45, 0.47, 0.44, 0.46, 0.48, 0.43, 0.45])

print(f"Mean: {summary.mean:.4f}")
print(f"Median: {summary.median:.4f}")
print(f"Std: {summary.std:.4f}")
print(f"CV: {summary.cv:.2%}")
print(f"Stable: {bool(summary.is_stable)}")  # True when CV < 10%
```

```text
Mean: 0.4543
Median: 0.4500
Std: 0.0172
CV: 3.78%
Stable: True
```

The `SampleSummary` it returns holds:

| Field | Description |
|-------|-------------|
| `mean`, `median`, `std` | Central tendency and spread |
| `minimum`, `maximum` | Range |
| `cv` | Coefficient of variation (std / mean), zero at a zero mean |
| `is_stable` | `True` when `cv < 0.10` |

Every field is a `jax.Array` and `SampleSummary` is a pytree, so a summary crosses a transform
boundary like any other value: `summarize` traces under `jax.jit`, maps under `jax.vmap` and
differentiates under `jax.grad`.

## Confidence Intervals

An interval is the part that resamples, so it takes the key it draws with — the same key gives
the same interval, and `summarize` stays free of randomness:

```python
import jax
from calibrax.statistics import bootstrap_interval

samples = jnp.asarray([0.45, 0.47, 0.44, 0.46, 0.48, 0.43, 0.45])
interval = bootstrap_interval(jnp.mean, samples, key=jax.random.key(42), confidence=0.99)

print(f"99% CI: [{interval.lower:.4f}, {interval.upper:.4f}]")
```

`bootstrap_interval` takes the statistic, so an interval around a median or a ratio costs the
same call. It resamples several arrays together when they are given together, as scipy's `paired=True` does.

## Outlier Detection

`outlier_mask` measures each observation against the median absolute deviation (MAD) rather
than the standard deviation, so the outliers do not inflate the scale that judges them:

```python
from calibrax.statistics import outlier_mask

samples = [0.45, 0.47, 0.44, 0.46, 1.20, 0.43, 0.45]  # 1.20 is an outlier
mask = outlier_mask(samples, threshold=3.5)

print(f"Outlier indices: {[int(index) for index in jnp.flatnonzero(mask)]}")
```

```text
Outlier indices: [4]
```

The mask has the shape of the sample, so it traces and maps like `summarize`; take the indices
with `jnp.flatnonzero` at the point you need them on the host. A lower `threshold` flags more
observations; the default of 3.5 is conservative.

## Significance Tests

Calibrax provides three significance tests for comparing two sets of measurements:

```python
from calibrax.statistics.significance import (
    welch_t_test,
    mann_whitney_u,
    paired_significance_test,
    effect_size,
)

baseline = [0.45, 0.47, 0.44, 0.46, 0.48]
current = [0.52, 0.54, 0.51, 0.53, 0.55]
```

### Welch's t-test

Use when samples are approximately normally distributed with potentially unequal
variances:

```python
t_stat, p_value = welch_t_test(baseline, current)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.6f}")
```

### Mann-Whitney U test

A non-parametric test — use when normality cannot be assumed:

```python
u_stat, p_value = mann_whitney_u(baseline, current)
print(f"U-statistic: {u_stat:.4f}, p-value: {p_value:.6f}")
```

### Paired significance test

For paired measurements (same workload, before and after a change). Uses the
Wilcoxon signed-rank test when scipy is available, falling back to a sign test
otherwise:

```python
result = paired_significance_test(baseline, current, alpha=0.05)
print(f"Significant: {result.significant}")
print(f"p-value: {result.p_value:.6f}")
print(f"Method: {result.method}")
```

### When to use each test

| Test | Assumptions | Best for |
|------|------------|----------|
| Welch's t-test | Approximate normality | Large samples, parametric comparison |
| Mann-Whitney U | None (non-parametric) | Small samples, unknown distribution |
| Paired test | Paired observations | Before/after comparison on same workload |

## Effect Size

Cohen's d quantifies the magnitude of the difference between two groups,
independent of sample size:

```python
d = effect_size(baseline, current)
print(f"Cohen's d: {d:.2f}")
```

| Cohen's d | Interpretation |
|-----------|---------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |

## Full Workflow Example

```python
import jax
import jax.numpy as jnp
from calibrax.statistics import bootstrap_interval, outlier_mask, summarize
from calibrax.statistics.significance import effect_size, paired_significance_test

baseline_samples = jnp.asarray([0.45, 0.47, 0.44, 0.46, 0.48])
current_samples = jnp.asarray([0.52, 0.54, 0.51, 0.53, 0.55])

# Drop the observations that stand apart before comparing
clean_baseline = baseline_samples[~outlier_mask(baseline_samples)]
clean_current = current_samples[~outlier_mask(current_samples)]

baseline_summary = summarize(clean_baseline)
current_summary = summarize(clean_current)

baseline_interval = bootstrap_interval(jnp.mean, clean_baseline, key=jax.random.key(0))
current_interval = bootstrap_interval(jnp.mean, clean_current, key=jax.random.key(1))

# The significance tests run on the host, so give them Python numbers
significance = paired_significance_test(clean_baseline.tolist(), clean_current.tolist())
d = effect_size(clean_baseline.tolist(), clean_current.tolist())

print(f"Baseline: {baseline_summary.mean:.4f} "
      f"[{baseline_interval.lower:.4f}, {baseline_interval.upper:.4f}]")
print(f"Current:  {current_summary.mean:.4f} "
      f"[{current_interval.lower:.4f}, {current_interval.upper:.4f}]")
print(f"Significant: {significance.significant} (p={significance.p_value:.4f})")
print(f"Effect size: {d:.2f} ({'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'})")
```

## Next Steps

<div class="grid cards" markdown>

-   :material-alert-decagram:{ .lg .middle } **Regression Detection**

    ---

    Use statistical results to detect performance regressions

    [:octicons-arrow-right-24: Regressions](regressions.md)

-   :material-compare:{ .lg .middle } **Comparing Configurations**

    ---

    Rank and compare multiple configurations statistically

    [:octicons-arrow-right-24: Comparison](comparison.md)

</div>
