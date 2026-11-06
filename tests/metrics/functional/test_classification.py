"""Tests for calibrax.metrics.functional.classification module."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, fbeta_score as sk_fbeta, roc_auc_score

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
    softmax_cross_entropy,
    specificity,
)


class TestAccuracy:
    """Tests for accuracy."""

    def test_perfect_predictions(self) -> None:
        """Accuracy should be 1.0 for all correct predictions."""
        targets = jnp.array([0, 1, 2, 1, 0])
        assert accuracy(targets, targets) == pytest.approx(1.0)

    def test_known_value(self) -> None:
        """Accuracy should match hand-calculated value."""
        predictions = jnp.array([0, 1, 0])
        targets = jnp.array([0, 0, 0])
        # 2/3 correct
        assert accuracy(predictions, targets) == pytest.approx(2.0 / 3.0, rel=1e-5)

    def test_returns_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        result = accuracy(jnp.zeros(3, dtype=jnp.int32), jnp.zeros(3, dtype=jnp.int32))
        assert hasattr(result, "shape")
        assert result.shape == ()

    def test_probabilities_input(self) -> None:
        """Should handle 2D probability input via argmax."""
        predictions = jnp.array([[0.9, 0.1], [0.3, 0.7], [0.8, 0.2]])
        targets = jnp.array([0, 1, 0])
        assert accuracy(predictions, targets) == pytest.approx(1.0)


class TestPrecisionRecallF1:
    """Tests for precision, recall, f1_score, fbeta_score."""

    def test_perfect_binary(self) -> None:
        """Perfect binary predictions should give 1.0 for all metrics."""
        predictions = jnp.array([1, 1, 0, 0])
        targets = jnp.array([1, 1, 0, 0])
        assert precision(predictions, targets) == pytest.approx(1.0, abs=1e-5)
        assert recall(predictions, targets) == pytest.approx(1.0, abs=1e-5)
        assert f1_score(predictions, targets) == pytest.approx(1.0, abs=1e-5)

    def test_known_binary_values(self) -> None:
        """Should match hand-calculated TP/FP/FN values."""
        # TP=2, FP=1, FN=1
        predictions = jnp.array([1, 1, 1, 0])
        targets = jnp.array([1, 1, 0, 1])
        assert precision(predictions, targets) == pytest.approx(2.0 / 3.0, rel=1e-3)
        assert recall(predictions, targets) == pytest.approx(2.0 / 3.0, rel=1e-3)

    def test_macro_averaging(self) -> None:
        """Macro averaging should compute per-class then average."""
        predictions = jnp.array([0, 1, 2, 0, 1, 2])
        targets = jnp.array([0, 1, 2, 0, 2, 1])
        result = precision(predictions, targets, average="macro")
        assert 0.0 < result < 1.0  # Not perfect due to 2 errors

    def test_micro_averaging(self) -> None:
        """Micro precision should equal accuracy for multiclass."""
        predictions = jnp.array([0, 1, 2, 0, 1, 2])
        targets = jnp.array([0, 1, 2, 0, 2, 1])
        micro_prec = precision(predictions, targets, average="micro")
        acc = accuracy(predictions, targets)
        assert micro_prec == pytest.approx(acc, rel=1e-5)

    def test_weighted_averaging(self) -> None:
        """Weighted averaging should weight by class frequency."""
        predictions = jnp.array([0, 1, 2, 0])
        targets = jnp.array([0, 1, 2, 0])
        result = precision(predictions, targets, average="weighted")
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_f1_is_harmonic_mean(self) -> None:
        """F1 should equal 2*p*r/(p+r)."""
        predictions = jnp.array([1, 1, 1, 0])
        targets = jnp.array([1, 1, 0, 1])
        p = precision(predictions, targets)
        r = recall(predictions, targets)
        expected_f1 = 2 * p * r / (p + r)
        assert f1_score(predictions, targets) == pytest.approx(expected_f1, rel=1e-4)

    def test_fbeta_with_beta_2(self) -> None:
        """F2 score should weight recall more than precision."""
        # TP=2, FP=1, FN=1
        predictions = jnp.array([1, 1, 1, 0])
        targets = jnp.array([1, 1, 0, 1])
        f1 = f1_score(predictions, targets)
        f2 = fbeta_score(predictions, targets, beta=2.0)
        # F2 weights recall more; with equal P and R, F1 == F2
        assert f1 == pytest.approx(f2, rel=1e-3)


class TestROCAUC:
    """Tests for ROC AUC."""

    def test_perfect_separation(self) -> None:
        """AUC should be 1.0 for perfectly separated predictions."""
        predictions = jnp.array([0.9, 0.8, 0.1, 0.05])
        targets = jnp.array([1, 1, 0, 0])
        assert roc_auc(predictions, targets) == pytest.approx(1.0, abs=1e-5)

    def test_inverted_predictions(self) -> None:
        """Inverted predictions should give AUC near 0."""
        predictions = jnp.array([0.1, 0.2, 0.9, 0.95])
        targets = jnp.array([1, 1, 0, 0])
        assert roc_auc(predictions, targets) == pytest.approx(0.0, abs=1e-5)

    def test_returns_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        result = roc_auc(jnp.array([0.5, 0.5]), jnp.array([0, 1]))
        assert hasattr(result, "shape")
        assert result.shape == ()


def _tied_fixtures(
    count: int, size: int = 12, levels: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Scores drawn from ``levels`` distinct values, so most fixtures tie across both classes."""
    rng = np.random.default_rng(0)
    fixtures = []
    while len(fixtures) < count:
        scores = rng.integers(0, levels, size) / (levels - 1)
        labels = rng.integers(0, 2, size)
        if labels.min() != labels.max():
            fixtures.append((scores.astype(np.float32), labels.astype(np.int32)))
    return fixtures


TIED = _tied_fixtures(200)
_UNIT_ROUNDOFF = float(np.finfo(np.float32).eps) / 2


def _summation_bound(size: int) -> float:
    """The float32 error bound on either metric of ``size`` records, derived, not measured.

    Every count is an exact integer in float32 and each per-record term is one correctly rounded
    division, so the only other error is one sum of at most ``size`` terms in [0, 1] and one final
    division. A sum of n terms in any order is within ``gamma_(n-1) * sum |x_i|`` of the exact
    value, ``gamma_k = k u / (1 - k u)`` (Higham, Accuracy and Stability of Numerical Algorithms,
    2nd ed., sec. 4.2), whatever order eager code or XLA under ``jit`` and ``vmap`` chooses; the
    terms and the result are at most 1, so ``gamma_(size + 2)`` covers the term roundings and the
    division too.
    """
    k = size + 2
    return k * _UNIT_ROUNDOFF / (1 - k * _UNIT_ROUNDOFF)


class TestRankingMetricsWithTiedScores:
    """Tied scores form one point of the curve, as scikit-learn's thresholds do.

    A score shared by a positive and a negative orders them neither way: ROC AUC counts the pair
    half (Mann-Whitney with midranks) and average precision steps once, at the precision after
    the whole tied run. Before, the order of tied records decided it: four tied scores gave 1.0
    or 0.0 by label order.
    """

    def test_tied_scores_do_not_depend_on_record_order(self) -> None:
        scores = jnp.full(4, 0.5)
        for labels in ([1, 1, 0, 0], [0, 0, 1, 1]):
            assert float(roc_auc(scores, jnp.array(labels))) == pytest.approx(0.5)
            assert float(average_precision(scores, jnp.array(labels))) == pytest.approx(0.5)

    @pytest.mark.parametrize("index", range(0, 200, 7))
    def test_roc_auc_matches_scikit_learn_on_tied_scores(self, index: int) -> None:
        scores, labels = TIED[index]
        assert float(roc_auc(scores, labels)) == pytest.approx(
            roc_auc_score(labels, scores), abs=_summation_bound(scores.size)
        )

    @pytest.mark.parametrize("index", range(0, 200, 7))
    def test_average_precision_matches_scikit_learn_on_tied_scores(self, index: int) -> None:
        scores, labels = TIED[index]
        assert float(average_precision(scores, labels)) == pytest.approx(
            average_precision_score(labels, scores), abs=_summation_bound(scores.size)
        )

    def test_a_permutation_of_the_records_changes_neither_metric(self) -> None:
        rng = np.random.default_rng(1)
        for scores, labels in TIED[:40]:
            order = rng.permutation(scores.size)
            assert float(roc_auc(scores[order], labels[order])) == float(roc_auc(scores, labels))
            assert float(average_precision(scores[order], labels[order])) == float(
                average_precision(scores, labels)
            )

    def test_one_class_has_no_roc_auc(self) -> None:
        """ROC AUC is undefined with one class; scikit-learn returns NaN and warns."""
        scores = jnp.array([0.2, 0.5, 0.9])
        assert math.isnan(float(roc_auc(scores, jnp.array([1, 1, 1]))))
        assert math.isnan(float(roc_auc(scores, jnp.array([0, 0, 0]))))

    def test_average_precision_with_one_class_follows_scikit_learn(self) -> None:
        scores = jnp.array([0.2, 0.5, 0.9])
        assert float(average_precision(scores, jnp.array([1, 1, 1]))) == pytest.approx(1.0)
        assert float(average_precision(scores, jnp.array([0, 0, 0]))) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("metric", "reference"),
        [(roc_auc, roc_auc_score), (average_precision, average_precision_score)],
    )
    def test_jit_and_vmap_give_the_metric(self, metric, reference) -> None:  # noqa: ANN001
        """Compiled and batched, each metric is still the defined quantity, to the same bound."""
        scores = np.stack([fixture[0] for fixture in TIED[:8]])
        labels = np.stack([fixture[1] for fixture in TIED[:8]])
        compiled = np.asarray(jax.jit(jax.vmap(metric))(scores, labels))
        expected = np.array([reference(y, s) for s, y in zip(scores, labels, strict=True)])
        np.testing.assert_allclose(
            compiled, expected, rtol=0, atol=_summation_bound(scores.shape[1])
        )

    @pytest.mark.parametrize("metric", [roc_auc, average_precision])
    def test_a_gradient_traces_and_is_zero(self, metric) -> None:  # noqa: ANN001
        """Both are step functions of the scores: the gradient is zero, and finite (DiffBio's
        ExactAUROC differentiates through roc_auc in its tests)."""
        scores, labels = TIED[0]
        grads = jax.grad(lambda s: metric(s, labels))(jnp.asarray(scores))
        assert np.array_equal(np.asarray(grads), np.zeros_like(scores))


class TestLogLoss:
    """Tests for log loss."""

    def test_perfect_predictions(self) -> None:
        """Near-perfect predictions should give near-zero loss."""
        predictions = jnp.array([0.999, 0.001])
        targets = jnp.array([1, 0])
        assert log_loss(predictions, targets) < 0.01

    def test_known_value(self) -> None:
        """Log loss should match hand-calculated value."""

        predictions = jnp.array([0.9])
        targets = jnp.array([1])
        expected = -math.log(0.9)
        assert log_loss(predictions, targets) == pytest.approx(expected, rel=1e-4)

    def test_handles_extreme_probabilities(self) -> None:
        """Should not produce inf for extreme probabilities."""
        predictions = jnp.array([1.0, 0.0])
        targets = jnp.array([1, 0])
        result = log_loss(predictions, targets)
        assert jnp.isfinite(result)

    def test_returns_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        result = log_loss(jnp.array([0.5, 0.5]), jnp.array([0, 1]))
        assert hasattr(result, "shape")
        assert result.shape == ()


class TestMatthewsCorrcoef:
    """Tests for Matthews correlation coefficient."""

    def test_perfect_predictions(self) -> None:
        """MCC should be ~1.0 for perfect predictions."""
        predictions = jnp.array([1, 1, 0, 0])
        targets = jnp.array([1, 1, 0, 0])
        assert matthews_corrcoef(predictions, targets) == pytest.approx(1.0, abs=1e-4)

    def test_range(self) -> None:
        """MCC should be in [-1, 1]."""
        predictions = jnp.array([1, 0, 1, 0, 1])
        targets = jnp.array([0, 1, 0, 1, 0])
        result = matthews_corrcoef(predictions, targets)
        assert -1.0 <= result <= 1.0


class TestCohenKappa:
    """Tests for Cohen's kappa."""

    def test_perfect_agreement(self) -> None:
        """Kappa should be ~1.0 for perfect agreement."""
        predictions = jnp.array([0, 1, 2, 0, 1])
        targets = jnp.array([0, 1, 2, 0, 1])
        assert cohen_kappa(predictions, targets) == pytest.approx(1.0, abs=1e-4)

    def test_returns_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        result = cohen_kappa(jnp.array([0, 1]), jnp.array([0, 1]))
        assert hasattr(result, "shape")
        assert result.shape == ()


class TestConfusionMatrix:
    """Tests for confusion matrix."""

    def test_binary_known_values(self) -> None:
        """Should produce correct 2x2 matrix."""
        predictions = jnp.array([1, 1, 0, 0])
        targets = jnp.array([1, 0, 0, 1])
        cm = confusion_matrix(predictions, targets)
        # Rows=true, Cols=predicted
        # TN=1 FP=1 | class 0
        # FN=1 TP=1 | class 1
        assert cm.shape == (2, 2)
        assert int(cm[0, 0]) == 1  # TN
        assert int(cm[0, 1]) == 1  # FP
        assert int(cm[1, 0]) == 1  # FN
        assert int(cm[1, 1]) == 1  # TP

    def test_multiclass(self) -> None:
        """Should produce correct 3x3 matrix."""
        predictions = jnp.array([0, 1, 2, 0])
        targets = jnp.array([0, 1, 2, 2])
        cm = confusion_matrix(predictions, targets)
        assert cm.shape == (3, 3)
        # Diagonal should have correct counts
        assert int(cm[0, 0]) == 1
        assert int(cm[1, 1]) == 1
        assert int(cm[2, 2]) == 1

    def test_returns_jax_array(self) -> None:
        """Should return JAX array, not float."""
        cm = confusion_matrix(jnp.array([0, 1]), jnp.array([0, 1]))
        assert hasattr(cm, "shape")


class TestBalancedAccuracy:
    """Tests for balanced accuracy."""

    def test_balanced_dataset(self) -> None:
        """On balanced data, balanced accuracy equals standard accuracy."""
        predictions = jnp.array([0, 1, 0, 1])
        targets = jnp.array([0, 1, 0, 1])
        assert balanced_accuracy(predictions, targets) == pytest.approx(1.0, abs=1e-5)

    def test_imbalanced_dataset(self) -> None:
        """On imbalanced data, balanced accuracy differs from accuracy."""
        # 9 class-0, 1 class-1; predict all 0
        predictions = jnp.zeros(10, dtype=jnp.int32)
        targets = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        acc = accuracy(predictions, targets)
        bal_acc = balanced_accuracy(predictions, targets)
        assert acc == pytest.approx(0.9, abs=1e-5)
        assert bal_acc == pytest.approx(0.5, abs=1e-3)  # 50% (1.0 + 0.0) / 2


class TestSpecificitySensitivity:
    """Tests for specificity and sensitivity."""

    def test_known_values(self) -> None:
        """Should match hand-calculated TP/TN/FP/FN values."""
        # TP=2, FP=1, FN=1, TN=2
        predictions = jnp.array([1, 1, 1, 0, 0, 0])
        targets = jnp.array([1, 1, 0, 0, 0, 1])
        assert sensitivity(predictions, targets) == pytest.approx(2.0 / 3.0, rel=1e-3)
        assert specificity(predictions, targets) == pytest.approx(2.0 / 3.0, rel=1e-3)

    def test_sensitivity_equals_recall(self) -> None:
        """Sensitivity should equal recall for binary classification."""
        predictions = jnp.array([1, 0, 1, 1, 0])
        targets = jnp.array([1, 0, 0, 1, 1])
        assert sensitivity(predictions, targets) == pytest.approx(
            recall(predictions, targets), rel=1e-5
        )


class TestSoftmaxCrossEntropy:
    """Logits-and-labels cross-entropy with the shared mask, weights, reduction and axis."""

    logits = jnp.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [1.0, 1.0, 1.0]])
    labels = jnp.array([0, 1, 2, 0])

    def test_uniform_logits_cost_log_of_the_class_count(self) -> None:
        value = softmax_cross_entropy(jnp.zeros((5, 4)), jnp.array([0, 1, 2, 3, 0]))
        assert jnp.allclose(value, jnp.log(4.0))

    def test_matches_the_log_softmax_of_the_true_class(self) -> None:
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        expected = -jnp.mean(log_probs[jnp.arange(4), self.labels])
        assert jnp.allclose(softmax_cross_entropy(self.logits, self.labels), expected)

    def test_mask_weights_and_reduction(self) -> None:
        mask = jnp.array([True, False, True, True])
        per_element = softmax_cross_entropy(self.logits, self.labels, reduction="none")
        assert per_element.shape == (4,)
        masked = softmax_cross_entropy(self.logits, self.labels, mask=mask)
        assert jnp.allclose(masked, jnp.mean(per_element[mask]))
        weights = jnp.array([1.0, 0.0, 2.0, 1.0])
        weighted = softmax_cross_entropy(self.logits, self.labels, weights=weights)
        assert jnp.allclose(weighted, jnp.sum(weights * per_element) / jnp.sum(weights))
        assert jnp.allclose(
            softmax_cross_entropy(self.logits, self.labels, reduction="sum"), jnp.sum(per_element)
        )

    def test_rejects_a_label_shape_that_does_not_match_the_logits(self) -> None:
        with pytest.raises(ValueError, match="labels"):
            softmax_cross_entropy(self.logits, jnp.array([0, 1]))

    def test_jit_and_grad(self) -> None:
        def loss(logits: jax.Array) -> jax.Array:
            return softmax_cross_entropy(
                logits, self.labels, mask=jnp.array([True, True, False, True])
            )

        assert jnp.allclose(jax.jit(loss)(self.logits), loss(self.logits))
        gradient = jax.grad(loss)(self.logits)
        assert jnp.all(jnp.isfinite(gradient))
        assert jnp.all(gradient[2] == 0.0)


class TestFBetaAveraging:
    """Macro and weighted F-beta are the mean of per-class F-beta, as sklearn defines them."""

    @pytest.mark.parametrize(
        ("targets", "predictions", "num_classes"),
        [
            ([0, 1, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0], 2),
            ([0, 0, 0, 1, 2, 2], [0, 1, 0, 1, 2, 0], 3),
            ([0, 0, 0, 0, 0, 1, 2, 3, 4, 4], [0, 0, 1, 1, 0, 1, 2, 2, 4, 4], 5),
            ([0, 1, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], 3),
        ],
    )
    def test_matches_sklearn_on_a_grid(
        self, targets: list[int], predictions: list[int], num_classes: int
    ) -> None:
        labels = list(range(num_classes))
        for average in ("macro", "weighted"):
            for beta in (1.0, 2.0):
                ours = fbeta_score(
                    jnp.array(predictions),
                    jnp.array(targets),
                    beta=beta,
                    average=average,
                    num_classes=num_classes,
                )
                reference = sk_fbeta(
                    targets,
                    predictions,
                    beta=beta,
                    average=average,
                    labels=labels,
                    # sklearn ships no types; pyright infers str from the default "warn".
                    zero_division=0,  # pyright: ignore[reportArgumentType]
                )
                assert jnp.allclose(ours, reference, atol=1e-6), (average, beta)

    def test_f1_macro_is_the_mean_of_per_class_f1(self) -> None:
        targets = jnp.array([0, 0, 1, 1, 2, 2])
        predictions = jnp.array([0, 1, 1, 1, 2, 0])
        per_class = [f1_score(predictions == c, targets == c, average="binary") for c in range(3)]
        macro = f1_score(predictions, targets, average="macro", num_classes=3)
        assert jnp.allclose(macro, jnp.mean(jnp.stack(per_class)))

    def test_jit_compatible_with_a_static_class_count(self) -> None:
        targets = jnp.array([0, 0, 1, 1, 2, 2])
        predictions = jnp.array([0, 1, 1, 1, 2, 0])

        @jax.jit
        def macro_f1(p: jax.Array, t: jax.Array) -> jax.Array:
            return f1_score(p, t, average="macro", num_classes=3)

        assert jnp.allclose(
            macro_f1(predictions, targets),
            f1_score(predictions, targets, average="macro", num_classes=3),
        )
