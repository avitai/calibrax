# Examples

Each example is a Python script under `examples/metrics/` with its notebook beside it, and each
page in this section is generated from a run of that notebook: the code, every output it printed,
and the run's duration, device and versions. All eight run on a CPU in a few seconds.

Run one from the repository root:

```bash
source ./activate.sh
uv run python examples/metrics/quickstart.py
```

or open its notebook (`examples/metrics/quickstart.ipynb`) in Jupyter. `uv run
substrax-notebooks test` runs every example and checks each run against its page.

## Tutorials

A tutorial walks through one capability step by step. Start with the quickstart.

| Example | What it covers |
| --- | --- |
| [Metrics Quickstart](examples/metrics/quickstart.md) | Single metrics, a whole suite with `calculate_all()`, and finding metrics in the `MetricRegistry` by domain, tier and property |
| [Regression Metrics Deep Dive](examples/metrics/regression_deep_dive.md) | Error measures, robust and percentage losses, CRPS for an ensemble forecast, and how an outlier moves MSE, MAE and Huber loss |
| [Classification, Calibration and Segmentation](examples/metrics/classification.md) | Hard and probabilistic classification metrics, the confusion matrix, calibration (Brier score, ECE) and segmentation quality |
| [Distances and Divergences Across Geometric Spaces](examples/metrics/distances.md) | Flat, hyperbolic, distributional and information-theoretic distances, and which property decides when each fits |
| [Image Quality and Text Evaluation Metrics](examples/metrics/image_quality.md) | PSNR, SSIM, MS-SSIM and the Vendi score on synthetic images; BLEU, ROUGE and Distinct-N on text |

## Advanced guides

An advanced guide combines several capabilities into one workload.

| Example | What it covers |
| --- | --- |
| [Metric Composition and Wrappers](examples/metrics/composition.md) | Collections, weighted scores, suites, threshold gates, bootstrap confidence intervals and tracking a metric over epochs |
| [Metric Learning Losses](examples/metrics/metric_learning.md) | Contrastive, triplet, NTXent and ArcFace losses on one batch, their gradients, and hard and semi-hard negative mining |
| [Manifold and Graph Metrics](examples/metrics/manifold_graph.md) | Spectral, shortest-path and resistance distances on graphs; SPD, Grassmann and ultrahyperbolic distances on manifolds |

## Writing an example

Examples follow substrax's
[Writing examples](https://substrax.readthedocs.io/en/latest/guides/writing-examples/) guide. A
commit checks each changed example against it, and `uv run substrax-notebooks page
examples/metrics/<name>.py` regenerates a page after its script changes.
