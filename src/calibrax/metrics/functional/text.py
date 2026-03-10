"""Text evaluation metrics -- n-gram and math based.

All metrics in this module are n-gram counting or mathematical operations.
No pretrained models, no transformers, no neural networks.

Includes: BLEU, ROUGE-N, ROUGE-L, perplexity, and distinct-n.
Registered with ``domain="text"``.

Note: BLEU/ROUGE/distinct_n operate on Python strings or token lists
and are NOT JAX-traceable. Perplexity operates on JAX arrays.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import jax.numpy as jnp


def _tokenize(text: str | list[str]) -> list[str]:
    """Tokenize input by whitespace if string, or return as-is if already tokenized.

    Args:
        text: String to tokenize or pre-tokenized list.

    Returns:
        List of tokens.
    """
    if isinstance(text, str):
        return text.split()
    return list(text)


def _get_ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    """Extract n-gram counts from a token list.

    Args:
        tokens: List of tokens.
        n: N-gram order.

    Returns:
        Counter of n-gram tuples.
    """
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _modified_precision(
    cand_tokens: list[str],
    ref_token_lists: list[list[str]],
    n: int,
) -> float | None:
    """Compute modified n-gram precision for a single n-gram order.

    Clips candidate n-gram counts by the maximum reference count for each n-gram.

    Args:
        cand_tokens: Tokenized candidate translation.
        ref_token_lists: List of tokenized reference translations.
        n: N-gram order.

    Returns:
        Precision value in (0, 1], or None if candidate has no n-grams
        or precision is zero (caller should short-circuit to 0.0).
    """
    cand_ngrams = _get_ngrams(cand_tokens, n)
    if not cand_ngrams:
        return None

    clipped_count = 0
    total_count = 0
    for ngram, count in cand_ngrams.items():
        max_ref_count = max(
            _get_ngrams(ref_tokens, n).get(ngram, 0) for ref_tokens in ref_token_lists
        )
        clipped_count += min(count, max_ref_count)
        total_count += count

    if total_count == 0:
        return None
    precision = clipped_count / total_count
    if precision == 0:
        return None
    return precision


def bleu(
    candidate: str | list[str],
    references: list[str | list[str]],
    *,
    max_n: int = 4,
    weights: tuple[float, ...] | None = None,
) -> float:
    """BLEU score for machine translation evaluation.

    Computes modified n-gram precision for n=1..max_n with brevity penalty.

    Args:
        candidate: Candidate translation (string or token list).
        references: List of reference translations.
        max_n: Maximum n-gram order (default 4 for BLEU-4).
        weights: Weights for each n-gram order. Default: uniform (1/max_n each).

    Returns:
        BLEU score in [0, 1]. 1.0 = perfect match.

    Examples:
        >>> bleu("the cat sat on the mat", ["the cat is on the mat"])
        ...
    """
    if weights is None:
        weights = tuple(1.0 / max_n for _ in range(max_n))

    cand_tokens = _tokenize(candidate)
    ref_token_lists = [_tokenize(ref) for ref in references]

    if not cand_tokens:
        return 0.0

    # Modified n-gram precisions
    log_precisions = 0.0
    for n in range(1, max_n + 1):
        precision = _modified_precision(cand_tokens, ref_token_lists, n)
        if precision is None:
            return 0.0
        log_precisions += weights[n - 1] * math.log(precision)

    # Brevity penalty
    cand_len = len(cand_tokens)
    ref_lens = [len(ref) for ref in ref_token_lists]
    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - cand_len), r))
    bp = math.exp(min(0, 1 - closest_ref_len / cand_len)) if cand_len > 0 else 0.0

    return bp * math.exp(log_precisions)


def rouge_n(
    candidate: str | list[str],
    reference: str | list[str],
    *,
    n: int = 1,
) -> float:
    """ROUGE-N recall: fraction of reference n-grams found in candidate.

    Args:
        candidate: Candidate text (string or token list).
        reference: Reference text (string or token list).
        n: N-gram order (default 1 for ROUGE-1).

    Returns:
        ROUGE-N recall in [0, 1]. 1.0 = all reference n-grams found.

    Examples:
        >>> rouge_n("the cat sat on the mat", "the cat is on the mat", n=1)
        ...
    """
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)

    ref_ngrams = _get_ngrams(ref_tokens, n)
    cand_ngrams = _get_ngrams(cand_tokens, n)

    if not ref_ngrams:
        return 0.0

    overlap = 0
    for ngram, count in ref_ngrams.items():
        overlap += min(count, cand_ngrams.get(ngram, 0))

    return overlap / sum(ref_ngrams.values())


def rouge_l(
    candidate: str | list[str],
    reference: str | list[str],
) -> float:
    """ROUGE-L: longest common subsequence based F-measure.

    Args:
        candidate: Candidate text (string or token list).
        reference: Reference text (string or token list).

    Returns:
        ROUGE-L F-measure in [0, 1]. 1.0 = identical sequences.

    Examples:
        >>> rouge_l("the cat sat on the mat", "the cat is on the mat")
        ...
    """
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)

    if not cand_tokens or not ref_tokens:
        return 0.0

    # LCS via dynamic programming
    m, n = len(cand_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if cand_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    precision = lcs_len / m
    recall = lcs_len / n

    if precision + recall == 0:
        return 0.0

    # F-measure with beta = recall/precision (balanced)
    beta_sq = (recall / (precision + 1e-12)) ** 2
    f_measure = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-12)
    return f_measure


def perplexity(log_probabilities: Any) -> Any:
    """Perplexity from log-probabilities.

    Computes exp(-mean(log_probs)). Lower perplexity = better model.

    Args:
        log_probabilities: Array of log-probabilities from a language model.

    Returns:
        Perplexity value >= 1.0.

    Examples:
        >>> import jax.numpy as jnp
        >>> perplexity(jnp.array([0.0, 0.0, 0.0]))  # Perfect model
        1.0
    """
    log_probs = jnp.asarray(log_probabilities, dtype=jnp.float32)
    return jnp.exp(-jnp.mean(log_probs))


def distinct_n(tokens: list[str], *, n: int = 1) -> float:
    """Distinct-N: ratio of unique n-grams to total n-grams.

    Measures lexical diversity. Higher = more diverse vocabulary usage.

    Args:
        tokens: List of tokens.
        n: N-gram order (default 1 for unigram diversity).

    Returns:
        Distinct-N ratio in [0, 1]. 1.0 = all n-grams unique.

    Examples:
        >>> distinct_n(["the", "cat", "sat", "on"], n=1)
        1.0
        >>> distinct_n(["the", "the", "the"], n=1)
        0.333...
    """
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)
