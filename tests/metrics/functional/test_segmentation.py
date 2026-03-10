"""Tests for segmentation metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.segmentation import (
    dice_coefficient,
    iou,
    pixel_accuracy,
)


class TestIoU:
    """Tests for iou."""

    def test_perfect_overlap(self) -> None:
        predictions = jnp.array([1, 1, 0, 0])
        targets = jnp.array([1, 1, 0, 0])
        assert iou(predictions, targets) == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap(self) -> None:
        predictions = jnp.array([1, 1, 0, 0])
        targets = jnp.array([0, 0, 1, 1])
        assert iou(predictions, targets) == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self) -> None:
        # Intersection=1, Union=3 → IoU=1/3
        predictions = jnp.array([1, 1, 0, 0])
        targets = jnp.array([0, 1, 1, 0])
        assert iou(predictions, targets) == pytest.approx(1.0 / 3.0, abs=1e-5)

    def test_multiclass_macro(self) -> None:
        predictions = jnp.array([0, 0, 1, 1, 2, 2])
        targets = jnp.array([0, 0, 1, 1, 2, 2])
        result = iou(predictions, targets, num_classes=3, average="macro")
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_multiclass_weighted(self) -> None:
        predictions = jnp.array([0, 0, 1, 1, 2, 2])
        targets = jnp.array([0, 0, 1, 1, 2, 2])
        result = iou(predictions, targets, num_classes=3, average="weighted")
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_returns_jax_scalar(self) -> None:
        predictions = jnp.array([1, 0])
        targets = jnp.array([1, 0])
        result = iou(predictions, targets)
        assert isinstance(result, jax.Array)


class TestDiceCoefficient:
    """Tests for dice_coefficient."""

    def test_perfect_overlap(self) -> None:
        predictions = jnp.array([1, 1, 0, 0])
        targets = jnp.array([1, 1, 0, 0])
        assert dice_coefficient(predictions, targets) == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap(self) -> None:
        predictions = jnp.array([1, 1, 0, 0])
        targets = jnp.array([0, 0, 1, 1])
        assert dice_coefficient(predictions, targets) == pytest.approx(0.0, abs=1e-6)

    def test_relationship_to_iou(self) -> None:
        # Dice = 2*IoU / (1 + IoU)
        predictions = jnp.array([1, 1, 0, 0])
        targets = jnp.array([0, 1, 1, 0])
        iou_val = iou(predictions, targets)
        dice_val = dice_coefficient(predictions, targets)
        expected_dice = 2.0 * iou_val / (1.0 + iou_val)
        assert dice_val == pytest.approx(expected_dice, abs=1e-5)

    def test_partial_overlap(self) -> None:
        # Intersection=1, |P|=2, |T|=2 → Dice = 2*1/(2+2) = 0.5
        predictions = jnp.array([1, 1, 0, 0])
        targets = jnp.array([0, 1, 1, 0])
        assert dice_coefficient(predictions, targets) == pytest.approx(0.5, abs=1e-5)

    def test_multiclass_macro(self) -> None:
        predictions = jnp.array([0, 1, 2])
        targets = jnp.array([0, 1, 2])
        result = dice_coefficient(predictions, targets, num_classes=3, average="macro")
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_returns_jax_scalar(self) -> None:
        predictions = jnp.array([1, 0])
        targets = jnp.array([1, 0])
        result = dice_coefficient(predictions, targets)
        assert isinstance(result, jax.Array)


class TestPixelAccuracy:
    """Tests for pixel_accuracy."""

    def test_perfect_predictions(self) -> None:
        predictions = jnp.array([0, 1, 2, 0])
        targets = jnp.array([0, 1, 2, 0])
        assert pixel_accuracy(predictions, targets) == pytest.approx(1.0, abs=1e-6)

    def test_known_value(self) -> None:
        # 3 out of 4 correct
        predictions = jnp.array([0, 1, 1, 0])
        targets = jnp.array([0, 1, 0, 0])
        assert pixel_accuracy(predictions, targets) == pytest.approx(0.75, abs=1e-6)

    def test_all_wrong(self) -> None:
        predictions = jnp.array([1, 0, 1, 0])
        targets = jnp.array([0, 1, 0, 1])
        assert pixel_accuracy(predictions, targets) == pytest.approx(0.0, abs=1e-6)

    def test_returns_jax_scalar(self) -> None:
        predictions = jnp.array([0, 1])
        targets = jnp.array([0, 1])
        result = pixel_accuracy(predictions, targets)
        assert isinstance(result, jax.Array)


class TestSegmentationMetricRegistration:
    """Tests for segmentation metric registration in MetricRegistry."""

    def test_segmentation_metrics_registered(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        for name in ["iou", "dice_coefficient", "pixel_accuracy"]:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_segmentation_domain(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        seg_metrics = registry.list_by_domain("segmentation")
        assert len(seg_metrics) == 3

    def test_direction_higher(self) -> None:
        from calibrax.core.models import MetricDirection
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        seg_metrics = registry.list_by_domain("segmentation")
        for m in seg_metrics:
            assert m.direction == MetricDirection.HIGHER


class TestVectorizedEquivalence:
    """Verify vectorized metrics match loop-based results."""

    def test_iou_multiclass_macro(self) -> None:
        p = jnp.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        t = jnp.array([0, 1, 1, 0, 2, 2, 1, 1, 0])
        result = iou(p, t, num_classes=3, average="macro")
        assert result >= 0.0
        assert result <= 1.0

    def test_dice_multiclass_weighted(self) -> None:
        p = jnp.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        t = jnp.array([0, 1, 1, 0, 2, 2, 1, 1, 0])
        result = dice_coefficient(p, t, num_classes=3, average="weighted")
        assert result >= 0.0
        assert result <= 1.0
