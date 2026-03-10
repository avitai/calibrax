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
# # Image Quality and Text Evaluation Metrics
#
# | | |
# |---|---|
# | **Level** | Tier 2: Tutorial |
# | **Time** | ~20 minutes |
# | **Prerequisites** | `01_quickstart.py`, basic JAX arrays |
# | **Metrics covered** | PSNR, SSIM, MS-SSIM, Vendi score, BLEU, ROUGE-N, ROUGE-L |
# | **Key concepts** | Image quality, text evaluation, diversity, registry |
#
# This tutorial demonstrates pure-math image quality metrics (no pretrained
# models), text evaluation metrics for machine translation and summarization,
# and the Vendi score for measuring diversity in a collection of items.

# %%
import jax
import jax.numpy as jnp

from calibrax.metrics import MetricRegistry
from calibrax.metrics.functional.image import ms_ssim, psnr, ssim, vendi_score
from calibrax.metrics.functional.text import bleu, distinct_n, rouge_l, rouge_n


# %% [markdown]
# ## 1. Image Quality Metrics on Synthetic Data
#
# PSNR, SSIM, and MS-SSIM compare a distorted image against a reference.
# Higher values indicate better quality for all three metrics.


# %%
def demonstrate_image_quality(key: jax.Array) -> jax.Array:
    """Show PSNR and SSIM sensitivity to different noise levels."""
    # Smooth gradient image (32x32 grayscale, values in [0, 1])
    x = jnp.linspace(0, 1, 32)
    original = jnp.outer(x, x)

    print("=== Image Quality Metrics ===")
    print("  Image size: 32x32, range [0, 1]")

    # Identical images
    print("\n  --- Original vs itself ---")
    print(f"    PSNR: {psnr(original, original):.2f} dB  (very high = identical)")
    print(f"    SSIM: {ssim(original, original):.6f}  (1.0 = identical)")

    # Slight noise (sigma = 0.02)
    key, subkey = jax.random.split(key)
    noisy_small = jnp.clip(original + jax.random.normal(subkey, original.shape) * 0.02, 0.0, 1.0)
    print("\n  --- Original vs slightly noisy (sigma=0.02) ---")
    print(f"    PSNR: {psnr(noisy_small, original):.2f} dB")
    print(f"    SSIM: {ssim(noisy_small, original):.6f}")

    # Heavy noise (sigma = 0.15)
    key, subkey = jax.random.split(key)
    noisy_large = jnp.clip(original + jax.random.normal(subkey, original.shape) * 0.15, 0.0, 1.0)
    print("\n  --- Original vs very noisy (sigma=0.15) ---")
    print(f"    PSNR: {psnr(noisy_large, original):.2f} dB")
    print(f"    SSIM: {ssim(noisy_large, original):.6f}")

    return key


# %% [markdown]
# ## 2. Multi-Scale SSIM and RGB Images
#
# MS-SSIM evaluates structural similarity at multiple resolutions by
# downsampling. It requires larger images to accommodate the scale pyramid.
# SSIM also works on multi-channel (RGB) images by averaging across channels.


# %%
def demonstrate_ms_ssim_and_rgb(key: jax.Array) -> jax.Array:
    """Show MS-SSIM on large images and SSIM on RGB data."""
    # MS-SSIM requires images large enough for downsampling (min ~160x160 for 5 scales)
    print("\n=== Multi-Scale SSIM (160x160 images, 2 scales) ===")
    x_large = jnp.linspace(0, 1, 160)
    original_large = jnp.outer(x_large, x_large)

    key, subkey = jax.random.split(key)
    noisy_large_img = jnp.clip(
        original_large + jax.random.normal(subkey, original_large.shape) * 0.05,
        0.0,
        1.0,
    )
    ms_ssim_val = ms_ssim(
        noisy_large_img,
        original_large,
        power_factors=(0.5, 0.5),  # 2 scales for efficiency
    )
    print(f"  MS-SSIM (sigma=0.05): {ms_ssim_val:.6f}")

    # RGB SSIM
    print("\n=== RGB SSIM (32x32x3) ===")
    key, subkey = jax.random.split(key)
    rgb_original = jax.random.uniform(subkey, (32, 32, 3))
    key, subkey = jax.random.split(key)
    rgb_noisy = jnp.clip(
        rgb_original + jax.random.normal(subkey, rgb_original.shape) * 0.05,
        0.0,
        1.0,
    )
    print(f"  SSIM(RGB): {ssim(rgb_noisy, rgb_original):.6f}")
    print(f"  PSNR(RGB): {psnr(rgb_noisy, rgb_original):.2f} dB")

    return key


# %% [markdown]
# ## 3. Vendi Score for Diversity Measurement
#
# The Vendi score measures the effective diversity of a set of items using
# the eigenvalue entropy of a similarity matrix. A score of 1 means all
# items are identical; a score of N (the number of items) means maximum
# diversity (all items are orthogonal).


# %%
def demonstrate_vendi_score() -> None:
    """Show Vendi score on similarity matrices with varying diversity."""
    print("\n=== Vendi Score (Diversity) ===")

    # All identical items: similarity matrix is all ones
    identical = jnp.ones((4, 4))
    score_identical = vendi_score(identical)
    print(f"  All identical (4x4 ones):  Vendi = {score_identical:.4f}  (min diversity)")

    # All orthogonal items: similarity matrix is identity
    orthogonal = jnp.eye(4)
    score_orthogonal = vendi_score(orthogonal)
    print(f"  All orthogonal (4x4 eye):  Vendi = {score_orthogonal:.4f}  (max diversity)")

    # Partial similarity: some items are similar, some different
    partial = jnp.array(
        [
            [1.0, 0.8, 0.1, 0.0],
            [0.8, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.7],
            [0.0, 0.1, 0.7, 1.0],
        ]
    )
    score_partial = vendi_score(partial)
    print(f"  Partial similarity:        Vendi = {score_partial:.4f}  (moderate diversity)")

    # Large diverse set
    large_diverse = jnp.eye(8) + 0.1 * jnp.ones((8, 8))
    score_large = vendi_score(large_diverse)
    print(f"  8 nearly-orthogonal items: Vendi = {score_large:.4f}")
    print("  Vendi score ranges from 1 (identical) to N (maximally diverse).")


# %% [markdown]
# ## 4. Text Quality Metrics: BLEU and ROUGE
#
# BLEU measures modified n-gram precision with a brevity penalty, commonly
# used for machine translation. ROUGE measures n-gram recall (ROUGE-N) or
# longest common subsequence (ROUGE-L), used for summarization evaluation.


# %%
def demonstrate_text_metrics() -> None:
    """Show BLEU and ROUGE on sample text pairs."""
    print("\n=== BLEU Score (Machine Translation) ===")

    candidate = "the cat sat on the mat"
    reference_1 = "the cat is on the mat"
    reference_2 = "there is a cat on the mat"

    print(f"  Candidate:   '{candidate}'")
    print(f"  Reference 1: '{reference_1}'")
    print(f"  Reference 2: '{reference_2}'")

    bleu_multi = bleu(candidate, [reference_1, reference_2])
    bleu_single = bleu(candidate, [reference_1])
    print(f"\n  BLEU-4 (multi-ref):  {bleu_multi:.4f}")
    print(f"  BLEU-4 (single ref): {bleu_single:.4f}")
    print("  Multiple references give a higher score by allowing more matches.")

    # ROUGE scores for summarization
    print("\n=== ROUGE Scores (Summarization) ===")
    candidate_summ = "the quick brown fox jumps over the lazy dog"
    reference_summ = "the fast brown fox leaps over a lazy dog"

    print(f"  Candidate: '{candidate_summ}'")
    print(f"  Reference: '{reference_summ}'")

    rouge1 = rouge_n(candidate_summ, reference_summ, n=1)
    rouge2 = rouge_n(candidate_summ, reference_summ, n=2)
    rougel = rouge_l(candidate_summ, reference_summ)
    print(f"\n  ROUGE-1 (unigram recall): {rouge1:.4f}")
    print(f"  ROUGE-2 (bigram recall):  {rouge2:.4f}")
    print(f"  ROUGE-L (LCS F-measure):  {rougel:.4f}")

    # Lexical diversity via Distinct-N
    print("\n=== Lexical Diversity (Distinct-N) ===")
    diverse_tokens = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    repetitive_tokens = ["the", "the", "the", "cat", "the", "the", "cat", "cat"]

    print(f"  Diverse text:    Distinct-1 = {distinct_n(diverse_tokens, n=1):.4f}")
    print(f"  Diverse text:    Distinct-2 = {distinct_n(diverse_tokens, n=2):.4f}")
    print(f"  Repetitive text: Distinct-1 = {distinct_n(repetitive_tokens, n=1):.4f}")
    print(f"  Repetitive text: Distinct-2 = {distinct_n(repetitive_tokens, n=2):.4f}")


# %% [markdown]
# ## 5. Registry Queries for Image and Text Domains
#
# The MetricRegistry tracks all registered metrics with domain tags.
# Querying by domain is useful for discovering which metrics are available
# for a given modality.


# %%
def demonstrate_registry_queries() -> None:
    """Show registry queries for image and text domains."""
    print("\n=== Registry: Image Domain ===")
    registry = MetricRegistry()

    image_entries = registry.list_by_domain("image")
    image_names = [e.name for e in image_entries]
    print(f"  Image metrics ({len(image_names)}): {image_names}")

    for entry in image_entries:
        diff = entry.properties.is_differentiable
        print(f"    {entry.name:20s}  dir={entry.direction}  differentiable={diff}")

    print("\n=== Registry: Text Domain ===")
    text_entries = registry.list_by_domain("text")
    text_names = [e.name for e in text_entries]
    print(f"  Text metrics ({len(text_names)}): {text_names}")

    for entry in text_entries:
        diff = entry.properties.is_differentiable
        print(f"    {entry.name:20s}  dir={entry.direction}  differentiable={diff}")


# %% [markdown]
# ## Main


# %%
def main() -> None:
    """Run image quality and text metric examples."""
    key = jax.random.PRNGKey(0)

    key = demonstrate_image_quality(key)
    key = demonstrate_ms_ssim_and_rgb(key)
    demonstrate_vendi_score()
    demonstrate_text_metrics()
    demonstrate_registry_queries()


if __name__ == "__main__":
    main()
