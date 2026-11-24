# calibrax.statistics

Statistical analysis tools for benchmark measurements. Provides summary
statistics, MAD-based outlier detection, percentile and bias-corrected
accelerated (BCa) bootstrap confidence intervals, exact host-side decimal
order-statistic ranks and Benjamini-Hochberg count thresholds, classical and
correlated resample significance tests (Welch's t, Mann-Whitney U, Wilcoxon,
Nadeau-Bengio paired t-test with Student-t ROPE posteriors, and two-way
repeated-measures ANOVA + simultaneous Tukey HSD), Cohen's d effect size,
Benjamini-Hochberg and Foster-Stine sequential alpha-investing FDR control, and
exact non-central Student-t power and minimum detectable effect calculations.

## Summary

::: calibrax.statistics.summary

## Bootstrap

::: calibrax.statistics.bootstrap

## Ranks & Count Thresholds

::: calibrax.statistics.ranks

## Significance Testing

::: calibrax.statistics.significance
    options:
      members:
        - AlphaInvestingResult
        - welch_t_test
        - mann_whitney_u
        - paired_significance_test
        - effect_size
        - benjamini_hochberg_adjust
        - alpha_investing
        - statistical_power
        - minimum_detectable_effect

## Nadeau-Bengio Paired Resample Test

::: calibrax.statistics.nadeau_bengio

## Repeated-Measures ANOVA & Tukey HSD

::: calibrax.statistics.tukey
