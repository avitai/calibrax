"""Text metrics requiring pretrained backbones.

Requires: calibrax[text] extra for full BERT feature extraction.
Tests use pre-computed embeddings to avoid requiring model weights.

Tier 1: BERTScoreMetric (frozen BERT for token embedding similarity)
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON
from calibrax.metrics.stateful._base import FrozenBackboneMetric


class BERTScoreMetric(FrozenBackboneMetric):
    """BERTScore using frozen BERT token embeddings.

    Computes precision, recall, and F1 based on cosine similarity
    between token embeddings from candidate and reference texts.

    For testing, accepts pre-computed token embeddings directly.

    Examples:
        >>> bertscore = BERTScoreMetric()
        >>> bertscore.update(
        ...     candidate_embeddings=cand_emb,  # (seq_len_c, hidden_dim)
        ...     reference_embeddings=ref_emb,   # (seq_len_r, hidden_dim)
        ... )
        >>> result = bertscore.compute()
        >>> # {"bertscore_precision": 0.92, "bertscore_recall": 0.88, "bertscore_f1": 0.90}
    """

    def __init__(self) -> None:
        """Initialize BERTScore metric."""
        super().__init__(name="bertscore")
        self._precisions: list[float] = []
        self._recalls: list[float] = []

    def reset(self) -> None:
        """Reset accumulated precision and recall values."""
        self._precisions = []
        self._recalls = []

    def _extract_features(self, **kwargs: Any) -> dict[str, Any]:
        """Accept pre-extracted embeddings.

        Args:
            **kwargs: "candidate_embeddings" (seq_c, dim) and
                "reference_embeddings" (seq_r, dim).

        Returns:
            Dict with candidate and reference embedding arrays.
        """
        return {
            "candidate": jnp.asarray(kwargs["candidate_embeddings"]),
            "reference": jnp.asarray(kwargs["reference_embeddings"]),
        }

    def _accumulate(self, features: Any) -> None:
        """Compute per-pair BERTScore and accumulate.

        Args:
            features: Dict with "candidate" and "reference" embedding arrays.
        """
        cand = features["candidate"]  # (seq_c, dim)
        ref = features["reference"]  # (seq_r, dim)

        # Cosine similarity matrix (seq_c, seq_r)
        cand_norm = cand / (jnp.linalg.norm(cand, axis=-1, keepdims=True) + _EPSILON)
        ref_norm = ref / (jnp.linalg.norm(ref, axis=-1, keepdims=True) + _EPSILON)
        sim_matrix = cand_norm @ ref_norm.T

        # Precision: max similarity for each candidate token
        precision = float(jnp.mean(jnp.max(sim_matrix, axis=1)))
        # Recall: max similarity for each reference token
        recall = float(jnp.mean(jnp.max(sim_matrix, axis=0)))

        self._precisions.append(precision)
        self._recalls.append(recall)

    def _compute_from_accumulated(self) -> dict[str, float]:
        """Compute average BERTScore across accumulated pairs.

        Returns:
            Dictionary with bertscore_precision, bertscore_recall, bertscore_f1.
        """
        if not self._precisions:
            return {
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
            }

        avg_p = sum(self._precisions) / len(self._precisions)
        avg_r = sum(self._recalls) / len(self._recalls)
        f1 = 2 * avg_p * avg_r / (avg_p + avg_r + _EPSILON)

        return {
            "bertscore_precision": avg_p,
            "bertscore_recall": avg_r,
            "bertscore_f1": f1,
        }
