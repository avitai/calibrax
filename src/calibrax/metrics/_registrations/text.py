"""Registrations of the built-in ``text`` metrics."""

from __future__ import annotations

from calibrax.core.models import MetricDirection
from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import (
    MetricEntry,
    MetricFn,
    MetricProperties,
    MetricSignature,
    MetricTier,
)
from calibrax.metrics.functional.text import bleu, distinct_n, perplexity, rouge_l, rouge_n


def register() -> None:
    """Register 5 text metrics at import time."""
    registry = MetricRegistry()
    # (name, fn, description, direction, is_jit_compatible)
    builtins: list[tuple[str, MetricFn, str, MetricDirection, bool]] = [
        ("bleu", bleu, "BLEU score for translation evaluation", MetricDirection.HIGHER, False),
        ("rouge_n", rouge_n, "ROUGE-N recall for summarization", MetricDirection.HIGHER, False),
        ("rouge_l", rouge_l, "ROUGE-L LCS-based F-measure", MetricDirection.HIGHER, False),
        (
            "perplexity",
            perplexity,
            "Perplexity from log-probabilities",
            MetricDirection.LOWER,
            True,
        ),
        ("distinct_n", distinct_n, "Distinct-N lexical diversity", MetricDirection.HIGHER, False),
    ]
    for name, fn, desc, direction, jit_compat in builtins:
        entry = MetricEntry(
            name=name,
            fn=fn,
            tier=MetricTier.PURE_FUNCTION,
            domain="text",
            direction=direction,
            description=desc,
            signature=MetricSignature.CUSTOM,
            properties=MetricProperties(
                is_jit_compatible=jit_compat,
                is_differentiable=jit_compat,
            ),
        )
        registry.register(name, entry)
