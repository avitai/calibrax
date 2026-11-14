# Classification, Calibration and Segmentation

> Ran in 3.5 s on 1 cpu (python 3.12.14, jax 0.11.1, substrax 0.1.27, calibrax 0.1.17), from [`examples/metrics/classification.py`](https://github.com/avitai/calibrax/blob/main/examples/metrics/classification.py) ([notebook](https://github.com/avitai/calibrax/blob/main/examples/metrics/classification.ipynb)). Run it from the repository root:

```bash
source ./activate.sh
uv run python examples/metrics/classification.py
```

| Metadata | Value |
| --- | --- |
| **Tier** | Tutorial |
| **Devices** | CPU-compatible |
| **Prerequisites** | The quickstart example; binary classification concepts |

## Overview

This example walks through calibrax's classification evaluation: binary metrics on hard and soft
predictions, the confusion matrix, probability calibration, and segmentation quality for
pixel-level tasks. Classification metrics come in two families. Hard-prediction metrics
(accuracy, precision, recall, F1) need thresholded 0/1 labels; probabilistic metrics (ROC-AUC,
log loss, Brier score, ECE) read the predicted probabilities and measure how well a model ranks
and calibrates its confidence.

## Learning Goals

1. Evaluate a binary classifier with accuracy, precision, recall, F1 and Matthews correlation
2. Build and read a confusion matrix
3. Assess probability calibration with the Brier score and the expected calibration error (ECE)
4. Measure segmentation quality with IoU, the Dice coefficient and pixel accuracy
5. Extend segmentation metrics to several classes with macro averaging

## Setup


```python
import jax
import jax.numpy as jnp

from calibrax.metrics.functional.calibration import brier_score, expected_calibration_error
from calibrax.metrics.functional.classification import (
    accuracy,
    balanced_accuracy,
    cohen_kappa,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision,
    recall,
    roc_auc,
    sensitivity,
    specificity,
)
from calibrax.metrics.functional.segmentation import dice_coefficient, iou, pixel_accuracy
```

## Core Concepts

Every metric follows the `(predictions, targets) -> scalar` signature of the rest of calibrax.
Balanced accuracy corrects for class imbalance by averaging the per-class recall; Matthews
correlation and Cohen's kappa account for agreement by chance, so on skewed data they say more
than raw accuracy. A well-calibrated model's predicted probabilities match the observed
frequency of positives: the Brier score is the mean squared gap between probability and label,
and ECE the average gap between confidence and accuracy across probability bins. For pixel-level
predictions, accuracy misleads when the background dominates; IoU and Dice measure the overlap
of the predicted and true masks instead.

## Implementation

### Step 1: Evaluate hard and soft predictions

The targets are labels, the hard predictions are thresholded labels, and the soft predictions are
probabilities, which ROC-AUC and log loss need.


```python
TARGETS = jnp.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
HARD_PREDICTIONS = jnp.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0])
SOFT_PREDICTIONS = jnp.array([0.9, 0.8, 0.4, 0.7, 0.2, 0.6, 0.3, 0.1, 0.85, 0.15])


def hard_metrics(predictions: jax.Array, targets: jax.Array) -> dict[str, float]:
    """Compute the metrics of thresholded predictions.

    Args:
        predictions: The predicted labels.
        targets: The true labels.

    Returns:
        Each metric's value by name.
    """
    return {
        "Accuracy": float(accuracy(predictions, targets)),
        "Balanced Accuracy": float(balanced_accuracy(predictions, targets)),
        "Precision": float(precision(predictions, targets)),
        "Recall": float(recall(predictions, targets)),
        "F1 Score": float(f1_score(predictions, targets)),
        "Sensitivity (TPR)": float(sensitivity(predictions, targets)),
        "Specificity (TNR)": float(specificity(predictions, targets)),
        "Matthews CC": float(matthews_corrcoef(predictions, targets)),
        "Cohen's Kappa": float(cohen_kappa(predictions, targets)),
    }


def soft_metrics(probabilities: jax.Array, targets: jax.Array) -> dict[str, float]:
    """Compute the metrics of predicted probabilities.

    Args:
        probabilities: The predicted probabilities of the positive class.
        targets: The true labels.

    Returns:
        ROC-AUC and log loss by name.
    """
    return {
        "ROC-AUC": float(roc_auc(probabilities, targets)),
        "Log Loss": float(log_loss(probabilities, targets)),
    }
```

### Step 2: Build the confusion matrix

calibrax returns a `(num_classes, num_classes)` array: row `i`, column `j` counts the samples of
class `i` predicted as `j`.


```python
def confusion_counts(predictions: jax.Array, targets: jax.Array) -> dict[str, int]:
    """Read the four counts of a binary confusion matrix.

    Args:
        predictions: The predicted labels.
        targets: The true labels.

    Returns:
        The true negatives, false positives, false negatives and true positives.
    """
    matrix = confusion_matrix(predictions, targets, num_classes=2)
    return {
        "TN": int(matrix[0, 0]),
        "FP": int(matrix[0, 1]),
        "FN": int(matrix[1, 0]),
        "TP": int(matrix[1, 1]),
    }
```

### Step 3: Compare two models' calibration

Two models rank ten samples the same way, so at a 0.5 threshold they predict the same labels and
have the same accuracy. One in five of the samples the first five probabilities describe is
positive, and four in five of the rest. The calibrated model says so, 0.2 and 0.8; the
overconfident one says 0.02 and 0.98. Lower Brier score and ECE mean better calibration.


```python
CALIBRATION_TARGETS = jnp.array([1, 0, 0, 0, 0, 1, 1, 1, 1, 0])
CALIBRATED = jnp.array([0.2] * 5 + [0.8] * 5)
OVERCONFIDENT = jnp.array([0.02] * 5 + [0.98] * 5)


def calibration(probabilities: jax.Array, targets: jax.Array) -> dict[str, float]:
    """Measure how well predicted probabilities match observed outcomes.

    Args:
        probabilities: The predicted probabilities of the positive class.
        targets: The true labels.

    Returns:
        The Brier score and the ECE.
    """
    return {
        "Brier Score": float(brier_score(probabilities, targets)),
        "ECE": float(expected_calibration_error(probabilities, targets)),
    }
```

### Step 4: Measure a predicted mask

A 4x4 binary mask, flattened: the true mask covers the top-left 2x2 block, and the prediction
adds one pixel to its right and misses one inside it.


```python
TRUE_MASK = jnp.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
PREDICTED_MASK = jnp.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def segmentation(predicted: jax.Array, true: jax.Array) -> dict[str, float]:
    """Measure the overlap of a predicted mask with the true one.

    Args:
        predicted: The predicted mask.
        true: The true mask.

    Returns:
        IoU, the Dice coefficient and pixel accuracy.
    """
    return {
        "IoU (Jaccard)": float(iou(predicted, true)),
        "Dice Coefficient": float(dice_coefficient(predicted, true)),
        "Pixel Accuracy": float(pixel_accuracy(predicted, true)),
    }
```

### Step 5: Average over several classes

With `num_classes` and `average="macro"`, IoU and Dice are computed per class and averaged.


```python
TRUE_CLASSES = jnp.array([0, 0, 1, 1, 2, 2, 0, 1])
PREDICTED_CLASSES = jnp.array([0, 0, 1, 2, 2, 1, 0, 1])


def macro_segmentation(predicted: jax.Array, true: jax.Array) -> dict[str, float]:
    """Macro-average IoU and Dice over three classes.

    Args:
        predicted: The predicted class of each pixel.
        true: The true class of each pixel.

    Returns:
        The macro IoU and the macro Dice.
    """
    return {
        "Macro IoU": float(iou(predicted, true, num_classes=3, average="macro")),
        "Macro Dice": float(dice_coefficient(predicted, true, num_classes=3, average="macro")),
    }
```

## Results

### Classification and the confusion matrix

The hard predictions get eight of ten labels right, one false positive and one false negative.

### Calibration and segmentation

The calibrated model's confidence matches each group's frequency, so its ECE is zero; the
overconfident model, with the same labels and accuracy, has a higher Brier score and ECE. The
segmentation tables show IoU and Dice below pixel accuracy, which the many background pixels
inflate.


```python
def print_table(title: str, values: dict[str, float]) -> None:
    """Print one table of metric values.

    Args:
        title: The table's title.
        values: Each metric's value by name.
    """
    print(f"=== {title} ===")
    for name, value in values.items():
        print(f"  {name:20s} {value:.4f}")


def main() -> dict[str, float | int]:
    """Run the example and return its results.

    Returns:
        The classification metrics, the confusion counts, both models' calibration and the
        segmentation metrics.
    """
    hard = hard_metrics(HARD_PREDICTIONS, TARGETS)
    soft = soft_metrics(SOFT_PREDICTIONS, TARGETS)
    print_table("Core metrics (hard predictions)", hard)
    print_table("Probabilistic metrics (soft predictions)", soft)
    counts = confusion_counts(HARD_PREDICTIONS, TARGETS)
    print("=== Confusion Matrix ===")
    print(f"  [[TN={counts['TN']}, FP={counts['FP']}],")
    print(f"   [FN={counts['FN']}, TP={counts['TP']}]]")
    calibrated = calibration(CALIBRATED, CALIBRATION_TARGETS)
    over = calibration(OVERCONFIDENT, CALIBRATION_TARGETS)
    print_table("Calibration: calibrated model", calibrated)
    print_table("Calibration: overconfident model", over)
    masks = segmentation(PREDICTED_MASK, TRUE_MASK)
    macro = macro_segmentation(PREDICTED_CLASSES, TRUE_CLASSES)
    print_table("Segmentation Metrics", masks)
    print_table("Multiclass Segmentation (macro average)", macro)
    return {
        **hard,
        **soft,
        **counts,
        **{f"calibrated {name}": value for name, value in calibrated.items()},
        **{f"overconfident {name}": value for name, value in over.items()},
        **masks,
        **macro,
    }


if __name__ == "__main__":
    main()
```

    === Core metrics (hard predictions) ===
      Accuracy             0.8000
      Balanced Accuracy    0.8000
      Precision            0.8000
      Recall               0.8000
      F1 Score             0.8000
      Sensitivity (TPR)    0.8000
      Specificity (TNR)    0.8000
      Matthews CC          0.6000
      Cohen's Kappa        0.6000
    === Probabilistic metrics (soft predictions) ===
      ROC-AUC              0.9600
      Log Loss             0.3528
    === Confusion Matrix ===
      [[TN=4, FP=1],
       [FN=1, TP=4]]
    === Calibration: calibrated model ===
      Brier Score          0.1600
      ECE                  0.0000
    === Calibration: overconfident model ===
      Brier Score          0.1924
      ECE                  0.1800
    === Segmentation Metrics ===
      IoU (Jaccard)        0.6000
      Dice Coefficient     0.7500
      Pixel Accuracy       0.8750
    === Multiclass Segmentation (macro average) ===
      Macro IoU            0.6111
      Macro Dice           0.7222


## Next Steps

- [Distances and spaces](distances.md): vector distances, hyperbolic geometry and
  divergences
- [Classification API reference](../../api-reference/metrics/classification.md): every
  classification function's signature
- [Calibration API reference](../../api-reference/metrics/calibration.md): the calibration
  functions

<!-- substrax-notebooks: {"page": {"body": "6228353f94f0355fd80db87db26cc8fb1a420304a36c53b78849e3827f26f353", "figures": {}, "record": "fc04df308630f225e921a01c375fbe38ae5cb4d25abc146b4f360794f9805d6a"}, "provenance": {"device": "1 cpu", "outputs": [[], [], [], [], [], [], ["text"]], "seconds": 3.5, "sha256": "3464b568bf328e3e0c3cc6ecdd461a2765784533dc380ee0126120b89143aca1", "source": "examples/metrics/classification.py", "versions": {"calibrax": "0.1.17", "jax": "0.11.1", "python": "3.12.14", "substrax": "0.1.27"}}} -->
