"""Statistical analysis: summary statistics, bootstrap CI, hypothesis testing, effect sizes."""

from calibrax.statistics.bootstrap import (
    bootstrap_interval,
    BootstrapInterval,
    DEFAULT_RESAMPLES,
)
from calibrax.statistics.significance import (
    effect_size,
    mann_whitney_u,
    paired_significance_test,
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
    "DEFAULT_RESAMPLES",
    "OUTLIER_Z_THRESHOLD",
    "STABILITY_CV_THRESHOLD",
    "BootstrapInterval",
    "SampleSummary",
    "bootstrap_interval",
    "effect_size",
    "mann_whitney_u",
    "outlier_mask",
    "paired_significance_test",
    "summarize",
    "welch_t_test",
]
