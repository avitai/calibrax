# %% [markdown]
"""
# Image Quality and Text Evaluation Metrics

| Metadata | Value |
| --- | --- |
| **Tier** | Tutorial |
| **Devices** | CPU-compatible |
| **Prerequisites** | The quickstart and distances examples; basic JAX arrays |

## Overview

This example covers calibrax's image and text quality metrics, all computed from the data itself
with no pretrained model. For images it measures PSNR, SSIM and MS-SSIM on synthetic grayscale
and RGB images at several noise levels, and the Vendi score for the diversity of a set of items;
for text it scores a translation with BLEU, a summary with ROUGE and lexical diversity with
Distinct-N. It ends by asking the registry which image and text metrics calibrax ships. These
metrics judge generative models, where a regression loss says nothing about perceived quality.

## Learning Goals

1. Measure image distortion with PSNR (pixel-level) and SSIM (structural)
2. Evaluate image quality at several resolutions with MS-SSIM, and RGB images with SSIM
3. Measure the diversity of a set of items with the Vendi score
4. Score a translation with BLEU and a summary with ROUGE
5. Compare the lexical diversity of two texts with Distinct-N
6. Query the registry for the image and text metrics
"""

# %% [markdown]
"""
## Setup
"""

# %%
import jax
import jax.numpy as jnp

from calibrax.metrics import MetricRegistry
from calibrax.metrics.functional.image import ms_ssim, psnr, ssim, vendi_score
from calibrax.metrics.functional.text import bleu, distinct_n, rouge_l, rouge_n


# %% [markdown]
"""
## Core Concepts

- PSNR is the ratio of the largest possible signal power to the noise power (the MSE), in
  decibels; higher means less distortion. It is fast but blind to structure: a blurred image and
  a noisy one can share a PSNR and look nothing alike.
- SSIM compares luminance, contrast and structure, from -1 to 1 where 1 is identical, and follows
  human perception more closely than PSNR. On an RGB image it averages over the channels.
  MS-SSIM repeats it at several resolutions by downsampling, so it needs images large enough for
  its scale pyramid.
- The Vendi score is the exponential of the entropy of a similarity matrix's eigenvalues: 1 when
  every item is identical and N, the number of items, when all are orthogonal.
- BLEU is the modified n-gram precision of a candidate translation against one or more
  references, with a brevity penalty. ROUGE-N is n-gram recall against a reference summary, and
  ROUGE-L the F-measure of their longest common subsequence. Distinct-N is the share of a text's
  n-grams that are unique.

## Implementation

### Step 1: Measure PSNR and SSIM at two noise levels

A smooth 32x32 gradient image in [0, 1] is compared with itself and with two noisy copies.
"""

# %%
GRADIENT = jnp.outer(jnp.linspace(0, 1, 32), jnp.linspace(0, 1, 32))


def add_noise(key: jax.Array, image: jax.Array, sigma: float) -> jax.Array:
    """Add Gaussian noise to an image and clip it to [0, 1].

    Args:
        key: The key the noise is drawn from.
        image: The image.
        sigma: The noise's standard deviation.

    Returns:
        The noisy image.
    """
    return jnp.clip(image + jax.random.normal(key, image.shape) * sigma, 0.0, 1.0)


def noise_levels(key: jax.Array) -> tuple[dict[str, tuple[float, float]], jax.Array]:
    """Compute PSNR and SSIM against the gradient image at each noise level.

    Args:
        key: The key the noise is drawn from.

    Returns:
        Each comparison's PSNR and SSIM, and the key left for the next step.
    """
    key, small_key = jax.random.split(key)
    key, large_key = jax.random.split(key)
    images = {
        "itself": GRADIENT,
        "sigma 0.02": add_noise(small_key, GRADIENT, 0.02),
        "sigma 0.15": add_noise(large_key, GRADIENT, 0.15),
    }
    scores = {
        name: (float(psnr(image, GRADIENT)), float(ssim(image, GRADIENT)))
        for name, image in images.items()
    }
    return scores, key


# %% [markdown]
"""
### Step 2: Measure MS-SSIM and RGB SSIM

MS-SSIM runs on a 160x160 gradient at two scales, and SSIM and PSNR on a random 32x32 RGB image.
"""

# %%
LARGE_GRADIENT = jnp.outer(jnp.linspace(0, 1, 160), jnp.linspace(0, 1, 160))


def multi_scale_and_rgb(key: jax.Array) -> dict[str, float]:
    """Compute MS-SSIM on a large image, and SSIM and PSNR on an RGB one.

    Args:
        key: The key the images and noise are drawn from.

    Returns:
        MS-SSIM, RGB SSIM and RGB PSNR.
    """
    key, large_key = jax.random.split(key)
    noisy_large = add_noise(large_key, LARGE_GRADIENT, 0.05)
    key, rgb_key = jax.random.split(key)
    rgb = jax.random.uniform(rgb_key, (32, 32, 3))
    _, rgb_noise_key = jax.random.split(key)
    noisy_rgb = add_noise(rgb_noise_key, rgb, 0.05)
    return {
        "MS-SSIM (sigma=0.05)": float(
            ms_ssim(noisy_large, LARGE_GRADIENT, power_factors=(0.5, 0.5))
        ),
        "SSIM (RGB)": float(ssim(noisy_rgb, rgb)),
        "PSNR (RGB), dB": float(psnr(noisy_rgb, rgb)),
    }


# %% [markdown]
"""
### Step 3: Measure diversity with the Vendi score

Four similarity matrices: all items identical, all orthogonal, two similar pairs, and eight
nearly orthogonal items.
"""

# %%
SIMILARITIES = {
    "All identical (4x4 ones)": jnp.ones((4, 4)),
    "All orthogonal (4x4 eye)": jnp.eye(4),
    "Two similar pairs": jnp.array(
        [
            [1.0, 0.8, 0.1, 0.0],
            [0.8, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.7],
            [0.0, 0.1, 0.7, 1.0],
        ]
    ),
    "8 nearly orthogonal items": jnp.eye(8) + 0.1 * jnp.ones((8, 8)),
}


def diversity() -> dict[str, float]:
    """Compute the Vendi score of each similarity matrix.

    Returns:
        Each matrix's Vendi score.
    """
    return {name: float(vendi_score(matrix)) for name, matrix in SIMILARITIES.items()}


# %% [markdown]
"""
### Step 4: Score a translation with BLEU

One candidate against one reference, then against two. The first reference shares no 4-gram
with the candidate; the second shares "cat sat on the", a phrasing the first lacks. BLEU-2 stops
at bigrams, so it is the same score without the 3-gram and 4-gram precisions.
"""

# %%
CANDIDATE = "the cat sat on the mat"
REFERENCES = ["the cat is on the mat", "a cat sat on the floor"]


def translation_scores() -> dict[str, float]:
    """Compute BLEU-4 and BLEU-2 against one reference and against two.

    Returns:
        Each BLEU score.
    """
    return {
        "BLEU-4 (one reference)": float(bleu(CANDIDATE, REFERENCES[:1])),
        "BLEU-4 (two references)": float(bleu(CANDIDATE, REFERENCES)),
        "BLEU-2 (one reference)": float(bleu(CANDIDATE, REFERENCES[:1], max_n=2)),
        "BLEU-2 (two references)": float(bleu(CANDIDATE, REFERENCES, max_n=2)),
    }


# %% [markdown]
"""
### Step 5: Score a summary with ROUGE
"""

# %%
SUMMARY = "the quick brown fox jumps over the lazy dog"
REFERENCE_SUMMARY = "the fast brown fox leaps over a lazy dog"


def summary_scores() -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2 and ROUGE-L of the summary.

    Returns:
        Each ROUGE score.
    """
    return {
        "ROUGE-1 (unigram recall)": float(rouge_n(SUMMARY, REFERENCE_SUMMARY, n=1)),
        "ROUGE-2 (bigram recall)": float(rouge_n(SUMMARY, REFERENCE_SUMMARY, n=2)),
        "ROUGE-L (LCS F-measure)": float(rouge_l(SUMMARY, REFERENCE_SUMMARY)),
    }


# %% [markdown]
"""
### Step 6: Compare lexical diversity with Distinct-N
"""

# %%
DIVERSE_TOKENS = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
REPETITIVE_TOKENS = ["the", "the", "the", "cat", "the", "the", "cat", "cat"]


def lexical_diversity() -> dict[str, float]:
    """Compute Distinct-1 and Distinct-2 of a diverse and a repetitive text.

    Returns:
        Each text's Distinct-1 and Distinct-2.
    """
    return {
        "Diverse text, Distinct-1": float(distinct_n(DIVERSE_TOKENS, n=1)),
        "Diverse text, Distinct-2": float(distinct_n(DIVERSE_TOKENS, n=2)),
        "Repetitive text, Distinct-1": float(distinct_n(REPETITIVE_TOKENS, n=1)),
        "Repetitive text, Distinct-2": float(distinct_n(REPETITIVE_TOKENS, n=2)),
    }


# %% [markdown]
"""
### Step 7: Query the registry for image and text metrics
"""


# %%
def domain_entries(domain: str) -> list[str]:
    """Describe each metric the registry holds for a domain.

    Args:
        domain: The domain.

    Returns:
        One line per metric: its name, direction and whether it is differentiable.
    """
    return [
        f"{entry.name:20s}  dir={entry.direction}  "
        f"differentiable={entry.properties.is_differentiable}"
        for entry in MetricRegistry().list_by_domain(domain)
    ]


# %% [markdown]
"""
## Results

### Image quality and diversity

PSNR and SSIM both fall as the noise grows, the identical comparison scoring SSIM 1. The Vendi
score is 1 for identical items and 4 for four orthogonal ones, with the two similar pairs between.

### Text quality and the registry

Against the first reference alone BLEU-4 is 0: the candidate shares no 4-gram with it, and
BLEU's geometric mean of the n-gram precisions is 0 when one of them is. The second reference
supplies the missing phrasing and raises both BLEU-4 and BLEU-2, which is why a test set gives a
sentence several references. The repetitive text's
Distinct-N is far below the diverse one's, and the registry lists each image and text metric
with its direction.
"""


# %%
def print_table(title: str, values: dict[str, float]) -> None:
    """Print one table of values.

    Args:
        title: The table's title.
        values: Each value by name.
    """
    print(f"=== {title} ===")
    for name, value in values.items():
        print(f"  {name:30s} {value:.6f}")


def main() -> dict[str, float | int]:
    """Run the example and return its results.

    Returns:
        Every image, diversity and text score, and the number of image and text metrics.
    """
    scores, key = noise_levels(jax.random.key(0))
    print("=== Image Quality Metrics (32x32, range [0, 1]) ===")
    for name, (peak, structural) in scores.items():
        print(f"  Original vs {name:12s} PSNR {peak:7.2f} dB  SSIM {structural:.6f}")
    tables = {
        "Multi-Scale and RGB SSIM": multi_scale_and_rgb(key),
        "Vendi Score (Diversity)": diversity(),
        "BLEU Score (Machine Translation)": translation_scores(),
        "ROUGE Scores (Summarization)": summary_scores(),
        "Lexical Diversity (Distinct-N)": lexical_diversity(),
    }
    for title, values in tables.items():
        print_table(title, values)
    counts: dict[str, int] = {}
    for domain in ("image", "text"):
        lines = domain_entries(domain)
        counts[f"{domain} metrics"] = len(lines)
        print(f"=== Registry: {domain} domain ({len(lines)}) ===")
        for line in lines:
            print(f"  {line}")
    return {
        **{f"PSNR vs {name}": peak for name, (peak, _) in scores.items()},
        **{f"SSIM vs {name}": structural for name, (_, structural) in scores.items()},
        **{name: value for values in tables.values() for name, value in values.items()},
        **counts,
    }


if __name__ == "__main__":
    main()


# %% [markdown]
"""
## Next Steps

- [Metric learning losses](metric_learning.md): contrastive, triplet, NTXent and ArcFace losses
  for training embeddings
- [Metric composition](composition.md): combine quality metrics with gates and tracking
- [Stateful metrics guide](../../user-guide/stateful-metrics.md): FID and Inception Score on
  extracted features
- [Image API reference](../../api-reference/metrics/image.md): every image metric's signature
- [Text API reference](../../api-reference/metrics/text.md): every text metric's signature
"""
