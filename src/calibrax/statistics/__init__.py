"""Statistical analysis: summary statistics, bootstrap CI, hypothesis testing, effect sizes."""

from calibrax.statistics.bootstrap import (
    bootstrap_interval,
    BootstrapInterval,
    DEFAULT_RESAMPLES,
)
from calibrax.statistics.significance import (
    alpha_investing,
    AlphaInvestingResult,
    benjamini_hochberg_adjust,
    effect_size,
    mann_whitney_u,
    minimum_detectable_effect,
    paired_significance_test,
    statistical_power,
    welch_t_test,
)
from calibrax.statistics.summary import (
    outlier_mask,
    OUTLIER_Z_THRESHOLD,
    SampleSummary,
    STABILITY_CV_THRESHOLD,
    summarize,
)


__all__ = [
    "AlphaInvestingResult",
    "BootstrapInterval",
    "DEFAULT_RESAMPLES",
    "OUTLIER_Z_THRESHOLD",
    "STABILITY_CV_THRESHOLD",
    "SampleSummary",
    "alpha_investing",
    "benjamini_hochberg_adjust",
    "bootstrap_interval",
    "effect_size",
    "mann_whitney_u",
    "minimum_detectable_effect",
    "outlier_mask",
    "paired_significance_test",
    "statistical_power",
    "summarize",
    "welch_t_test",
]
