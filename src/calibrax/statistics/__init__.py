"""Statistical analysis: summary statistics, bootstrap CI, hypothesis testing, effect sizes."""

from calibrax.statistics.bootstrap import (
    bootstrap_interval,
    BootstrapInterval,
    DEFAULT_RESAMPLES,
)
from calibrax.statistics.nadeau_bengio import (
    nadeau_bengio_t_test,
    NadeauBengioResult,
)
from calibrax.statistics.ranks import (
    conformal_plus_ranks,
    conformal_quantile_rank,
    decimal_fraction,
    discrete_bh_count_thresholds,
    farey_floor_rational,
    minimum_conformal_calibration_size,
    screening_cutoff_count,
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
from calibrax.statistics.tukey import (
    RepeatedMeasuresTukeyResult,
    rm_tukey_hsd,
    TukeyPairwiseComparison,
)


__all__ = [
    "AlphaInvestingResult",
    "BootstrapInterval",
    "DEFAULT_RESAMPLES",
    "NadeauBengioResult",
    "OUTLIER_Z_THRESHOLD",
    "RepeatedMeasuresTukeyResult",
    "STABILITY_CV_THRESHOLD",
    "SampleSummary",
    "TukeyPairwiseComparison",
    "alpha_investing",
    "benjamini_hochberg_adjust",
    "bootstrap_interval",
    "conformal_plus_ranks",
    "conformal_quantile_rank",
    "decimal_fraction",
    "discrete_bh_count_thresholds",
    "effect_size",
    "farey_floor_rational",
    "mann_whitney_u",
    "minimum_conformal_calibration_size",
    "minimum_detectable_effect",
    "nadeau_bengio_t_test",
    "outlier_mask",
    "paired_significance_test",
    "rm_tukey_hsd",
    "screening_cutoff_count",
    "statistical_power",
    "summarize",
    "welch_t_test",
]
