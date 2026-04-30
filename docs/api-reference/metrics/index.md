# calibrax.metrics

JAX-native evaluation metrics organized in a 4-tier system. All Tier 0
functions accept JAX arrays and return JAX scalar arrays. Higher tiers provide
stateful and learned metric implementations built on Flax NNX.

## Tier System

| Tier | Name | Description | Example |
|------|------|-------------|---------|
| 0 | Pure Functions | Stateless `f(y_pred, y_true) -> scalar` | MSE, BLEU, IoU |
| 1 | Frozen Backbone | Pre-trained feature extractor, no gradient | FID, BERTScore |
| 2 | Learned | Backbone with learned calibration weights | LPIPS |
| 3 | Metric Learning | Differentiable loss for embedding spaces | TripletMarginLoss, ArcFace |

## Registry Usage

All registered metrics can be discovered and computed through the
`MetricRegistry` singleton:

```python
from calibrax.metrics import MetricRegistry, calculate_all

# Discover metrics by domain or tier
registry = MetricRegistry()
regression_metrics = registry.list_by_domain("general")
jit_safe = registry.list_jit_compatible()

# Batch computation of Tier 0 metrics
results = calculate_all(predictions, targets, metrics=["mse", "mae", "r_squared"])
```

## Sub-modules

- [Registry](registry.md) -- MetricRegistry, MetricEntry, MetricTier, MetricSignature
- [Regression](regression.md) -- MSE, MAE, RMSE, R-squared, Huber, quantile loss
- [Classification](classification.md) -- accuracy, precision, recall, F1, ROC-AUC
- [Calibration](calibration.md) -- Brier score, ECE, MCE, adaptive ECE
- [Segmentation](segmentation.md) -- IoU, Dice, pixel accuracy
- [Distance](distance.md) -- Euclidean, cosine, Mahalanobis, Poincare, Lorentz
- [Divergence](divergence.md) -- KL, JS, Wasserstein, Sinkhorn, MMD
- [Information](information.md) -- entropy, cross-entropy, mutual information
- [Ranking](ranking.md) -- NDCG, MAP, MRR, precision/recall at k
- [Statistical](statistical.md) -- Pearson, Spearman, Kendall, concordance
- [Clustering](clustering.md) -- ARI, NMI, silhouette, Davies-Bouldin
- [Fairness](fairness.md) -- demographic parity, equalized odds, disparate impact
- [Image](image.md) -- PSNR, SSIM, MS-SSIM, Vendi Score
- [Video](video.md) -- VMAF via FFmpeg/libvmaf
- [Text](text.md) -- BLEU, ROUGE, perplexity, distinct-N
- [Audio](audio.md) -- SNR, spectral convergence, mel cepstral distortion
- [Geometric](geometric.md) -- Chamfer, Hausdorff, Earth Mover's distance
- [Graph](graph.md) -- spectral distance, graph edit distance, resistance distance
- [Manifold](manifold.md) -- SPD, Grassmann, Stiefel, ultrahyperbolic distances
- [Composition](composition.md) -- MetricCollection, WeightedMetric, MetricSuite
- [Wrappers](wrappers.md) -- BootstrapMetric, ClasswiseWrapper, MetricTracker
- [Stateful](stateful.md) -- FrozenBackboneMetric, LearnedMetric base classes
- [Learning](learning.md) -- contrastive, triplet, ArcFace, proxy losses
- [Scientific](scientific.md) -- chemical validity, binding affinity, conformational

::: calibrax.metrics
    options:
      show_root_heading: false
      members: false
