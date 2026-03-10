"""Tier 0 functional metrics -- pure functions with no model state."""

from calibrax.metrics.functional.audio import (
    mel_cepstral_distortion,
    signal_to_noise_ratio,
    spectral_convergence,
)
from calibrax.metrics.functional.calibration import (
    adaptive_calibration_error,
    brier_decomposition,
    brier_score,
    classwise_ece,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_diagram_bins,
)
from calibrax.metrics.functional.classification import (
    accuracy,
    average_precision,
    balanced_accuracy,
    cohen_kappa,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision,
    recall,
    roc_auc,
    sensitivity,
    specificity,
)
from calibrax.metrics.functional.clustering import (
    adjusted_mutual_information,
    adjusted_rand_index,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_information_clustering,
    silhouette_score,
    v_measure,
)
from calibrax.metrics.functional.distance import (
    chebyshev_distance,
    cosine_distance,
    euclidean_distance,
    hamming_distance,
    jaccard_distance,
    lorentz_distance,
    mahalanobis_distance,
    manhattan_distance,
    minkowski_distance,
    poincare_distance,
    randers_distance,
)
from calibrax.metrics.functional.divergence import (
    bregman_divergence,
    chi_squared_divergence,
    f_divergence,
    hellinger_distance,
    js_divergence,
    kl_divergence,
    mmd,
    renyi_divergence,
    reverse_kl_divergence,
    sinkhorn_divergence,
    sliced_wasserstein,
    total_variation,
    wasserstein_1d,
)
from calibrax.metrics.functional.fairness import (
    demographic_parity_ratio,
    disparate_impact_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    group_metric_breakdown,
)
from calibrax.metrics.functional.geometric import (
    chamfer_distance,
    directed_hausdorff,
    earth_movers_distance_1d,
    hausdorff_distance,
)
from calibrax.metrics.functional.graph import (
    graph_edit_distance_approx,
    resistance_distance,
    shortest_path_distance,
    spectral_distance,
)
from calibrax.metrics.functional.image import (
    ms_ssim,
    psnr,
    ssim,
    vendi_score,
)
from calibrax.metrics.functional.information import (
    conditional_entropy,
    cross_entropy,
    entropy,
    fisher_information_matrix,
    mutual_information,
    normalized_mutual_information,
)
from calibrax.metrics.functional.manifold import (
    grassmann_distance,
    spd_affine_invariant_distance,
    spd_log_euclidean_distance,
    stiefel_distance,
    ultrahyperbolic_distance,
)
from calibrax.metrics.functional.ranking import (
    coverage,
    hit_rate,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from calibrax.metrics.functional.regression import (
    explained_variance,
    huber_loss,
    log_cosh_loss,
    mae,
    mape,
    max_error,
    mse,
    quantile_loss,
    r_squared,
    relative_error,
    rmse,
    smape,
)
from calibrax.metrics.functional.segmentation import (
    dice_coefficient,
    iou,
    pixel_accuracy,
)
from calibrax.metrics.functional.statistical import (
    concordance_correlation,
    kendall_tau,
    pearson_correlation,
    r_squared_adjusted,
    spearman_rank_correlation,
)
from calibrax.metrics.functional.text import (
    bleu,
    distinct_n,
    perplexity,
    rouge_l,
    rouge_n,
)


__all__ = [
    # Audio
    "mel_cepstral_distortion",
    "signal_to_noise_ratio",
    "spectral_convergence",
    # Calibration
    "adaptive_calibration_error",
    "brier_decomposition",
    "brier_score",
    "classwise_ece",
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_diagram_bins",
    # Information
    "conditional_entropy",
    "cross_entropy",
    "entropy",
    "fisher_information_matrix",
    "mutual_information",
    "normalized_mutual_information",
    # Divergence
    "bregman_divergence",
    "chi_squared_divergence",
    "f_divergence",
    "hellinger_distance",
    "js_divergence",
    "kl_divergence",
    "mmd",
    "renyi_divergence",
    "reverse_kl_divergence",
    "sinkhorn_divergence",
    "sliced_wasserstein",
    "total_variation",
    "wasserstein_1d",
    # Image
    "ms_ssim",
    "psnr",
    "ssim",
    "vendi_score",
    # Manifold
    "grassmann_distance",
    "spd_affine_invariant_distance",
    "spd_log_euclidean_distance",
    "stiefel_distance",
    "ultrahyperbolic_distance",
    # Graph
    "graph_edit_distance_approx",
    "resistance_distance",
    "shortest_path_distance",
    "spectral_distance",
    # Geometric
    "chamfer_distance",
    "directed_hausdorff",
    "earth_movers_distance_1d",
    "hausdorff_distance",
    # Fairness
    "demographic_parity_ratio",
    "disparate_impact_ratio",
    "equal_opportunity_difference",
    "equalized_odds_difference",
    "group_metric_breakdown",
    # Distance
    "chebyshev_distance",
    "cosine_distance",
    "euclidean_distance",
    "hamming_distance",
    "jaccard_distance",
    "lorentz_distance",
    "mahalanobis_distance",
    "manhattan_distance",
    "minkowski_distance",
    "poincare_distance",
    "randers_distance",
    # Clustering
    "adjusted_mutual_information",
    "adjusted_rand_index",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "normalized_mutual_information_clustering",
    "silhouette_score",
    "v_measure",
    # Classification
    "accuracy",
    "average_precision",
    "balanced_accuracy",
    "cohen_kappa",
    "confusion_matrix",
    "f1_score",
    "fbeta_score",
    "log_loss",
    "matthews_corrcoef",
    "precision",
    "recall",
    "roc_auc",
    "sensitivity",
    "specificity",
    # Segmentation
    "dice_coefficient",
    "iou",
    "pixel_accuracy",
    # Ranking
    "coverage",
    "hit_rate",
    "mean_average_precision",
    "mean_reciprocal_rank",
    "ndcg",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    # Text
    "bleu",
    "distinct_n",
    "perplexity",
    "rouge_l",
    "rouge_n",
    # Statistical
    "concordance_correlation",
    "kendall_tau",
    "pearson_correlation",
    "r_squared_adjusted",
    "spearman_rank_correlation",
    # Regression
    "explained_variance",
    "huber_loss",
    "log_cosh_loss",
    "mae",
    "mape",
    "max_error",
    "mse",
    "quantile_loss",
    "r_squared",
    "relative_error",
    "rmse",
    "smape",
]
