# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Classification, Calibration, and Segmentation
#
# | | |
# |---|---|
# | **Level** | Tier 2: Tutorial |
# | **Time** | ~15 minutes |
# | **Prerequisites** | `01_quickstart.py`, binary classification concepts |
# | **Metrics covered** | accuracy, precision, recall, F1, ROC-AUC, Brier score, ECE, IoU, Dice |
# | **Key concepts** | Hard vs soft predictions, calibration, segmentation quality |

# %%
"""Classification workflow: binary metrics, calibration, and segmentation.

Demonstrates:
- Binary classification: accuracy, precision, recall, F1, ROC-AUC
- Confusion matrix interpretation
- Calibration metrics: Brier score, ECE
- Segmentation quality: IoU, Dice coefficient on synthetic masks
"""

import jax.numpy as jnp

from calibrax.metrics.functional.calibration import (
    brier_score,
    expected_calibration_error,
)
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
from calibrax.metrics.functional.segmentation import (
    dice_coefficient,
    iou,
    pixel_accuracy,
)


def main() -> None:
    """Run classification and segmentation examples."""
    # -- Binary classification data ----------------------------------------
    # Ground truth: 1 = positive, 0 = negative
    targets = jnp.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
    # Hard predictions (thresholded)
    preds_hard = jnp.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0])
    # Soft predictions (probabilities for ROC-AUC, log loss)
    preds_soft = jnp.array([0.9, 0.8, 0.4, 0.7, 0.2, 0.6, 0.3, 0.1, 0.85, 0.15])

    print("=== Binary Classification Metrics ===")
    print(f"  Targets:     {targets.tolist()}")
    print(f"  Predictions: {preds_hard.tolist()}")
    print(f"  Probabilities: {preds_soft.tolist()}")

    print("\n  --- Core metrics (hard predictions) ---")
    print(f"  Accuracy:          {accuracy(preds_hard, targets):.4f}")
    print(f"  Balanced Accuracy: {balanced_accuracy(preds_hard, targets):.4f}")
    print(f"  Precision:         {precision(preds_hard, targets):.4f}")
    print(f"  Recall:            {recall(preds_hard, targets):.4f}")
    print(f"  F1 Score:          {f1_score(preds_hard, targets):.4f}")
    print(f"  Sensitivity (TPR): {sensitivity(preds_hard, targets):.4f}")
    print(f"  Specificity (TNR): {specificity(preds_hard, targets):.4f}")
    print(f"  Matthews CC:       {matthews_corrcoef(preds_hard, targets):.4f}")
    print(f"  Cohen's Kappa:     {cohen_kappa(preds_hard, targets):.4f}")

    print("\n  --- Probabilistic metrics (soft predictions) ---")
    print(f"  ROC-AUC:           {roc_auc(preds_soft, targets):.4f}")
    print(f"  Log Loss:          {log_loss(preds_soft, targets):.4f}")

    # -- Confusion matrix --------------------------------------------------
    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(preds_hard, targets, num_classes=2)
    print(f"  [[TN={int(cm[0, 0])}, FP={int(cm[0, 1])}],")
    print(f"   [FN={int(cm[1, 0])}, TP={int(cm[1, 1])}]]")
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Negatives:  {tn}")

    # -- Calibration metrics -----------------------------------------------
    print("\n=== Calibration Metrics ===")
    # Well-calibrated model: probabilities match observed frequencies
    well_calibrated_probs = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    calibration_targets = jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    # Overconfident model: always predicts ~0.9 for positives
    overconfident_probs = jnp.array([0.05, 0.05, 0.1, 0.1, 0.9, 0.9, 0.95, 0.95, 0.9, 0.95])

    print("  Well-calibrated model:")
    print(f"    Brier Score: {brier_score(well_calibrated_probs, calibration_targets):.4f}")
    well_ece = expected_calibration_error(well_calibrated_probs, calibration_targets)
    print(f"    ECE:         {well_ece:.4f}")

    print("  Overconfident model:")
    print(f"    Brier Score: {brier_score(overconfident_probs, calibration_targets):.4f}")
    over_ece = expected_calibration_error(overconfident_probs, calibration_targets)
    print(f"    ECE:         {over_ece:.4f}")

    print("  Lower Brier score and ECE indicate better calibration.")

    # -- Segmentation metrics ----------------------------------------------
    print("\n=== Segmentation Metrics ===")
    # Synthetic 4x4 binary mask (flattened for simplicity)
    #  Ground truth:         Prediction:
    #  1 1 0 0               1 1 1 0
    #  1 1 0 0               1 0 0 0
    #  0 0 0 0               0 0 0 0
    #  0 0 0 0               0 0 0 0
    gt_mask = jnp.array(
        [
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    pred_mask = jnp.array(
        [
            1,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )

    print("  Ground truth mask (4x4, flattened):", gt_mask.tolist())
    print("  Predicted mask   (4x4, flattened):", pred_mask.tolist())

    iou_val = iou(pred_mask, gt_mask)
    dice_val = dice_coefficient(pred_mask, gt_mask)
    pixel_acc = pixel_accuracy(pred_mask, gt_mask)

    print(f"\n  IoU (Jaccard):      {iou_val:.4f}")
    print(f"  Dice Coefficient:   {dice_val:.4f}")
    print(f"  Pixel Accuracy:     {pixel_acc:.4f}")
    print("\n  IoU and Dice are more informative than pixel accuracy")
    print("  for imbalanced segmentation tasks (many background pixels).")

    # -- Multiclass segmentation -------------------------------------------
    print("\n=== Multiclass Segmentation (macro average) ===")
    # 3 classes: background=0, class_a=1, class_b=2
    gt_multi = jnp.array([0, 0, 1, 1, 2, 2, 0, 1])
    pred_multi = jnp.array([0, 0, 1, 2, 2, 1, 0, 1])

    iou_macro = iou(pred_multi, gt_multi, num_classes=3, average="macro")
    dice_macro = dice_coefficient(pred_multi, gt_multi, num_classes=3, average="macro")
    print(f"  Macro IoU:  {iou_macro:.4f}")
    print(f"  Macro Dice: {dice_macro:.4f}")


if __name__ == "__main__":
    main()
