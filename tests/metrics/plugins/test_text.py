"""Tests for text backbone plugin (BERTScore).

All tests use pre-computed embeddings to avoid requiring BERT weights.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from calibrax.core.protocols import StatefulMetricProtocol
from calibrax.metrics.plugins.text import BERTScoreMetric


class TestBERTScoreMetric:
    """Tests for BERTScoreMetric."""

    def test_identical_embeddings(self) -> None:
        """F1 = 1.0 when candidate and reference embeddings are identical."""
        bertscore = BERTScoreMetric()
        emb = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        bertscore.update(candidate_embeddings=emb, reference_embeddings=emb)
        result = bertscore.compute()
        assert result["bertscore_f1"] == pytest.approx(1.0, abs=1e-4)
        assert result["bertscore_precision"] == pytest.approx(1.0, abs=1e-4)
        assert result["bertscore_recall"] == pytest.approx(1.0, abs=1e-4)

    def test_orthogonal_embeddings(self) -> None:
        """Low F1 for orthogonal embeddings."""
        bertscore = BERTScoreMetric()
        cand = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ref = jnp.array([[0.0, 0.0, 1.0]])
        bertscore.update(candidate_embeddings=cand, reference_embeddings=ref)
        result = bertscore.compute()
        assert result["bertscore_f1"] == pytest.approx(0.0, abs=1e-4)

    def test_multiple_pairs(self) -> None:
        """Accumulation averages across multiple update calls."""
        bertscore = BERTScoreMetric()
        # First pair: identical → F1=1.0
        emb1 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        bertscore.update(candidate_embeddings=emb1, reference_embeddings=emb1)
        # Second pair: same again → still F1=1.0
        bertscore.update(candidate_embeddings=emb1, reference_embeddings=emb1)
        result = bertscore.compute()
        assert result["bertscore_f1"] == pytest.approx(1.0, abs=1e-4)

    def test_reset(self) -> None:
        bertscore = BERTScoreMetric()
        emb = jnp.array([[1.0, 0.0]])
        bertscore.update(candidate_embeddings=emb, reference_embeddings=emb)
        bertscore.reset()
        result = bertscore.compute()
        assert result["bertscore_f1"] == 0.0

    def test_compute_without_update(self) -> None:
        bertscore = BERTScoreMetric()
        result = bertscore.compute()
        assert result["bertscore_precision"] == 0.0
        assert result["bertscore_recall"] == 0.0
        assert result["bertscore_f1"] == 0.0

    def test_conforms_to_protocol(self) -> None:
        bertscore = BERTScoreMetric()
        assert isinstance(bertscore, StatefulMetricProtocol)
