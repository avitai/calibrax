# Classification Metrics

| | |
|---|---|
| **Level** | Intermediate |
| **Time** | ~12 minutes |
| **Prerequisites** | [Quickstart](quickstart.md), [Regression Metrics](regression-metrics.md) |
| **Format** | Python + Jupyter |

## Overview

This example covers the full classification evaluation pipeline: binary metrics on hard and soft predictions, confusion matrix interpretation, probability calibration assessment, and segmentation quality for pixel-level tasks. It demonstrates 14 classification metrics, 7 calibration metrics, and 3 segmentation metrics.

Classification metrics divide into two families. Hard-prediction metrics (accuracy, precision, recall, F1) require thresholded 0/1 labels. Probabilistic metrics (ROC-AUC, log loss, Brier score, ECE) operate directly on predicted probabilities and measure how well the model ranks and calibrates its confidence.

## What You'll Learn

1. Evaluate binary classifiers with accuracy, precision, recall, F1, and Matthews correlation
2. Build and interpret a confusion matrix (TP, FP, FN, TN)
3. Assess probability calibration with Brier score and Expected Calibration Error (ECE)
4. Measure segmentation quality with IoU, Dice coefficient, and pixel accuracy
5. Extend binary segmentation to multiclass with macro averaging

## Files

- **Python Script**: [`examples/metrics/03_classification.py`](https://github.com/avitai/calibrax/blob/main/examples/metrics/03_classification.py)
- **Jupyter Notebook**: [`examples/metrics/03_classification.ipynb`](https://github.com/avitai/calibrax/blob/main/examples/metrics/03_classification.ipynb)

## Quick Start

```bash
source activate.sh && python examples/metrics/03_classification.py
```

## Key Concepts

### Core Classification Metrics

All metrics follow the `(predictions, targets) -> scalar` signature, consistent with the rest of calibrax.

```python
from calibrax.metrics.functional.classification import (
    accuracy, balanced_accuracy, precision, recall, f1_score,
    sensitivity, specificity, matthews_corrcoef, cohen_kappa,
    roc_auc, log_loss, confusion_matrix,
)

targets = jnp.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
preds_hard = jnp.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0])
preds_soft = jnp.array([0.9, 0.8, 0.4, 0.7, 0.2, 0.6, 0.3, 0.1, 0.85, 0.15])

accuracy(preds_hard, targets)
precision(preds_hard, targets)
recall(preds_hard, targets)
f1_score(preds_hard, targets)
matthews_corrcoef(preds_hard, targets)
roc_auc(preds_soft, targets)   # requires probabilities
log_loss(preds_soft, targets)  # requires probabilities
```

**Balanced accuracy** corrects for class imbalance by averaging per-class recall. **Matthews correlation coefficient** and **Cohen's kappa** account for chance agreement, making them more informative than raw accuracy on skewed datasets.

### Confusion Matrix

The confusion matrix gives a complete picture of classification errors. Calibrax returns a `(num_classes, num_classes)` JAX array.

```python
cm = confusion_matrix(preds_hard, targets, num_classes=2)
# cm[0, 0] = TN, cm[0, 1] = FP
# cm[1, 0] = FN, cm[1, 1] = TP
```

### Calibration Metrics

A well-calibrated model's predicted probabilities match the observed frequency of positive outcomes. Two metrics measure this:

- **Brier score**: mean squared difference between predicted probability and true label. Lower is better.
- **ECE (Expected Calibration Error)**: average absolute gap between predicted confidence and observed accuracy across bins.

```python
from calibrax.metrics.functional.calibration import brier_score, expected_calibration_error

probabilities = preds_soft  # from the classification block above
brier_score(probabilities, targets)
expected_calibration_error(probabilities, targets)
```

The example compares a well-calibrated model against an overconfident one to show how ECE detects miscalibration even when accuracy is high.

### Segmentation Metrics

For pixel-level predictions, accuracy alone is misleading when the background class dominates. IoU and Dice are intersection-based metrics that focus on the overlap between predicted and ground-truth masks.

```python
from calibrax.metrics.functional.segmentation import iou, dice_coefficient, pixel_accuracy

pred_mask = jnp.array([1, 1, 0, 1, 0, 0, 1, 1])
gt_mask = jnp.array([1, 0, 0, 1, 0, 1, 1, 1])

iou(pred_mask, gt_mask)               # intersection / union
dice_coefficient(pred_mask, gt_mask)   # 2 * intersection / (|pred| + |gt|)
pixel_accuracy(pred_mask, gt_mask)     # correct pixels / total pixels
```

For multiclass segmentation, pass `num_classes` and `average="macro"` to get the macro-averaged score across all classes:

```python
pred_multi = jnp.array([0, 1, 2, 0, 1, 2])
gt_multi = jnp.array([0, 1, 1, 0, 2, 2])

iou(pred_multi, gt_multi, num_classes=3, average="macro")
dice_coefficient(pred_multi, gt_multi, num_classes=3, average="macro")
```

## Example Code

The calibration section illustrates the difference between well-calibrated and overconfident models:

```python
well_calibrated_probs = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
overconfident_probs = jnp.array([0.05, 0.05, 0.1, 0.1, 0.9, 0.9, 0.95, 0.95, 0.9, 0.95])
calibration_targets = jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Well-calibrated: low Brier score, low ECE
brier_score(well_calibrated_probs, calibration_targets)
expected_calibration_error(well_calibrated_probs, calibration_targets)

# Overconfident: higher Brier score, higher ECE
brier_score(overconfident_probs, calibration_targets)
expected_calibration_error(overconfident_probs, calibration_targets)
```

## Next Steps

- [Distances and Spaces](distances-and-spaces.md) -- vector distances, hyperbolic geometry, and divergences
- [API Reference: `calibrax.metrics.functional.classification`](../../api-reference/metrics/classification.md) -- full signatures
- [API Reference: `calibrax.metrics.functional.calibration`](../../api-reference/metrics/calibration.md) -- calibration function details
