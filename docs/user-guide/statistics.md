# Statistical Analysis

Calibrax provides tools for summarizing benchmark measurements with confidence
intervals, detecting outliers, running significance tests, and computing effect
sizes.

## Summarizing Samples

`StatisticalAnalyzer` computes descriptive statistics and bootstrap confidence
intervals for a sequence of measurements:

```python
from calibrax.statistics.analyzer import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(bootstrap_resamples=1000, seed=42)
result = analyzer.summarize([0.45, 0.47, 0.44, 0.46, 0.48, 0.43, 0.45])

print(f"Mean: {result.mean:.4f}")
print(f"Median: {result.median:.4f}")
print(f"Std: {result.std:.4f}")
print(f"CV: {result.cv:.2%}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
print(f"Stable: {result.is_stable}")  # True when CV < 10%
```

```text
Mean: 0.4543
Median: 0.4500
Std: 0.0172
CV: 3.78%
95% CI: [0.4429, 0.4671]
Stable: True
```

The `StatisticalResult` dataclass contains:

| Field | Description |
|-------|-------------|
| `mean`, `median`, `std` | Central tendency and spread |
| `min`, `max` | Range |
| `cv` | Coefficient of variation (std / mean) |
| `ci_lower`, `ci_upper` | Bootstrap confidence interval bounds |
| `n` | Sample count |
| `is_stable` | `True` when `cv < 0.10` |

## Bootstrap Confidence Intervals

For more control over the confidence level, use `bootstrap_ci()` directly:

```python
samples = [0.45, 0.47, 0.44, 0.46, 0.48, 0.43, 0.45]
lower, upper = analyzer.bootstrap_ci(samples, confidence=0.99)
print(f"99% CI: [{lower:.4f}, {upper:.4f}]")
```

## Outlier Detection

Detect outliers using the Median Absolute Deviation (MAD) method, which is robust
to skewed distributions:

```python
samples = [0.45, 0.47, 0.44, 0.46, 1.20, 0.43, 0.45]  # 1.20 is an outlier
outlier_indices = analyzer.detect_outliers(samples, threshold=3.5)
print(f"Outlier indices: {outlier_indices}")  # [4]
```

The `threshold` parameter controls sensitivity — lower values flag more samples
as outliers. The default of 3.5 is conservative.

## Significance Tests

!!! warning "Optional Dependency"

    Significance tests require scipy. Install with:

    ```bash
    uv pip install "calibrax[stats]"
    ```

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
from calibrax.statistics.analyzer import StatisticalAnalyzer
from calibrax.statistics.significance import paired_significance_test, effect_size

analyzer = StatisticalAnalyzer()

baseline_samples = [0.45, 0.47, 0.44, 0.46, 0.48]
current_samples = [0.52, 0.54, 0.51, 0.53, 0.55]

# Summarize each group
baseline_stats = analyzer.summarize(baseline_samples)
current_stats = analyzer.summarize(current_samples)

# Remove outliers
clean_baseline = [s for i, s in enumerate(baseline_samples)
                  if i not in analyzer.detect_outliers(baseline_samples)]
clean_current = [s for i, s in enumerate(current_samples)
                 if i not in analyzer.detect_outliers(current_samples)]

# Test significance
sig = paired_significance_test(clean_baseline, clean_current)
d = effect_size(clean_baseline, clean_current)

print(f"Baseline: {baseline_stats.mean:.4f} [{baseline_stats.ci_lower:.4f}, "
      f"{baseline_stats.ci_upper:.4f}]")
print(f"Current:  {current_stats.mean:.4f} [{current_stats.ci_lower:.4f}, "
      f"{current_stats.ci_upper:.4f}]")
print(f"Significant: {sig.significant} (p={sig.p_value:.4f})")
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
