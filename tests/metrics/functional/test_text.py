"""Tests for text evaluation metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.text import (
    bleu,
    distinct_n,
    perplexity,
    rouge_l,
    rouge_n,
)


class TestBLEU:
    """Tests for bleu."""

    def test_identical_text(self) -> None:
        candidate = "the cat sat on the mat"
        references = ["the cat sat on the mat"]
        result = bleu(candidate, references)
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self) -> None:
        candidate = "hello world"
        references = ["foo bar baz qux"]
        result = bleu(candidate, references)
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_brevity_penalty(self) -> None:
        candidate = "the cat"
        references = ["the cat sat on the mat"]
        result = bleu(candidate, references)
        assert result < 1.0  # Penalized for being too short

    def test_multiple_references(self) -> None:
        candidate = "the cat sat on the mat"
        references = ["the cat is on the mat", "there is a cat on the mat"]
        # Use max_n=2 since short sentences have 0 4-gram overlap
        result = bleu(candidate, references, max_n=2)
        assert 0.0 < result <= 1.0

    def test_pre_tokenized_input(self) -> None:
        candidate = ["the", "cat", "sat", "on", "the"]
        references = [["the", "cat", "sat", "on", "the"]]
        result = bleu(candidate, references)
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_returns_float(self) -> None:
        result = bleu("hello", ["hello"])
        assert isinstance(result, float)


class TestRougeN:
    """Tests for rouge_n."""

    def test_identical_text(self) -> None:
        text = "the cat sat on the mat"
        assert rouge_n(text, text, n=1) == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self) -> None:
        assert rouge_n("hello world", "foo bar baz", n=1) == pytest.approx(0.0, abs=1e-5)

    def test_known_recall(self) -> None:
        # Reference: "the cat sat" (3 unigrams)
        # Candidate: "the cat ran" → 2/3 overlap
        result = rouge_n("the cat ran", "the cat sat", n=1)
        assert result == pytest.approx(2.0 / 3.0, abs=1e-5)

    def test_bigram(self) -> None:
        text = "the cat sat on the mat"
        assert rouge_n(text, text, n=2) == pytest.approx(1.0, abs=1e-5)


class TestRougeL:
    """Tests for rouge_l."""

    def test_identical_text(self) -> None:
        text = "the cat sat on the mat"
        assert rouge_l(text, text) == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self) -> None:
        assert rouge_l("hello world", "foo bar baz") == pytest.approx(0.0, abs=1e-5)

    def test_known_lcs(self) -> None:
        # LCS of "a b c d" and "a c d e" is "a c d" (length 3)
        result = rouge_l("a b c d", "a c d e")
        assert result > 0.0


class TestPerplexity:
    """Tests for perplexity."""

    def test_perfect_model(self) -> None:
        log_probs = jnp.array([0.0, 0.0, 0.0])
        assert perplexity(log_probs) == pytest.approx(1.0, abs=1e-5)

    def test_known_value(self) -> None:
        # exp(-mean([-1, -1])) = exp(1) ≈ 2.718
        log_probs = jnp.array([-1.0, -1.0])
        import math

        assert perplexity(log_probs) == pytest.approx(math.e, abs=1e-4)

    def test_higher_entropy(self) -> None:
        low_entropy = perplexity(jnp.array([-0.1, -0.1]))
        high_entropy = perplexity(jnp.array([-2.0, -2.0]))
        assert high_entropy > low_entropy

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        result = perplexity(jnp.array([-1.0]))
        assert isinstance(result, jax.Array)


class TestDistinctN:
    """Tests for distinct_n."""

    def test_all_unique(self) -> None:
        tokens = ["the", "cat", "sat", "on"]
        assert distinct_n(tokens, n=1) == pytest.approx(1.0, abs=1e-5)

    def test_all_same(self) -> None:
        tokens = ["the", "the", "the"]
        assert distinct_n(tokens, n=1) == pytest.approx(1.0 / 3.0, abs=1e-5)

    def test_bigram_diversity(self) -> None:
        tokens = ["a", "b", "a", "b"]  # bigrams: (a,b), (b,a), (a,b) → 2/3 unique
        assert distinct_n(tokens, n=2) == pytest.approx(2.0 / 3.0, abs=1e-5)


class TestTextMetricRegistration:
    """Tests for text metric registration."""

    def test_all_registered(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        expected = ["bleu", "rouge_n", "rouge_l", "perplexity", "distinct_n"]
        for name in expected:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_text_domain(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        text_metrics = registry.list_by_domain("text")
        assert len(text_metrics) == 5

    def test_perplexity_direction_lower(self) -> None:
        from calibrax.core.models import MetricDirection
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        assert registry.get("perplexity").direction == MetricDirection.LOWER
