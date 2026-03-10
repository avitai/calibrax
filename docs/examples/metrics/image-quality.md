# Image Quality and Text Metrics

| | |
|---|---|
| **Level** | Intermediate |
| **Time** | ~12 minutes |
| **Prerequisites** | [Quickstart](quickstart.md), [Distances and Spaces](distances-and-spaces.md) |
| **Format** | Python + Jupyter |

## Overview

This example covers perceptual and generative evaluation metrics. For images, it demonstrates PSNR, SSIM, and MS-SSIM on synthetic grayscale and RGB arrays at different noise levels. For text, it shows BLEU and ROUGE scoring for machine translation and summarisation evaluation. Finally, it introduces FID and Inception Score using pre-extracted feature vectors, which is the standard approach when working outside a full image pipeline.

These metrics are essential for evaluating generative models (GANs, diffusion models, language models) where simple regression losses do not capture perceptual quality.

## What You'll Learn

1. Measure image distortion with PSNR (pixel-level) and SSIM (structural similarity)
2. Evaluate multi-scale image quality with MS-SSIM on larger images
3. Score machine translations with BLEU (precision-based) and summarisations with ROUGE (recall-based)
4. Assess lexical diversity with Distinct-N
5. Compare generative distributions using FID and Inception Score on pre-extracted features

## Files

- **Python Script**: [`examples/metrics/06_image_quality.py`](https://github.com/avitai/calibrax/blob/main/examples/metrics/06_image_quality.py)
- **Jupyter Notebook**: [`examples/metrics/06_image_quality.ipynb`](https://github.com/avitai/calibrax/blob/main/examples/metrics/06_image_quality.ipynb)

## Quick Start

```bash
source activate.sh && python examples/metrics/06_image_quality.py
```

## Key Concepts

### PSNR (Peak Signal-to-Noise Ratio)

PSNR measures the ratio between the maximum possible signal power and the noise power (MSE). It is expressed in decibels (dB). Higher values indicate less distortion. Identical images yield a very high PSNR.

```python
import jax
from calibrax.metrics.functional.image import psnr

original = jnp.linspace(0, 1, 64).reshape(8, 8)
noisy_image = jnp.clip(original + jax.random.normal(jax.random.PRNGKey(0), (8, 8)) * 0.05, 0.0, 1.0)

psnr(noisy_image, original)  # returns value in dB
```

PSNR is fast to compute but does not account for structural or perceptual distortion -- a blurred image and a noisy image can have similar PSNR despite looking very different.

### SSIM (Structural Similarity)

SSIM compares luminance, contrast, and structure between two images. Values range from -1 to 1, where 1 means identical. It correlates better with human perception than PSNR.

```python
from calibrax.metrics.functional.image import ssim

ssim(noisy_image, original)  # range [-1, 1], 1 = identical
```

SSIM works on grayscale and multi-channel (RGB) images. For RGB, the score is averaged across channels.

### MS-SSIM (Multi-Scale SSIM)

MS-SSIM evaluates structural similarity at multiple resolutions by repeatedly downsampling the image. It captures both fine detail and coarse structure. Requires larger images (the minimum size depends on the number of scales).

```python
from calibrax.metrics.functional.image import ms_ssim

original_large = jnp.linspace(0, 1, 1024).reshape(32, 32)
noisy_large_image = jnp.clip(
    original_large + jax.random.normal(jax.random.PRNGKey(1), (32, 32)) * 0.05, 0.0, 1.0
)

ms_ssim(
    noisy_large_image, original_large,
    power_factors=(0.5, 0.5),  # 2 scales for efficiency
)
```

### BLEU (Bilingual Evaluation Understudy)

BLEU measures n-gram precision of a candidate translation against one or more references. It is the standard automatic metric for machine translation.

```python
from calibrax.metrics.functional.text import bleu

candidate = "the cat sat on the mat"
references = ["the cat is on the mat", "there is a cat on the mat"]

bleu(candidate, references)  # BLEU-4 by default
```

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE measures n-gram recall, making it suited for summarisation where coverage of reference content matters.

```python
from calibrax.metrics.functional.text import rouge_n, rouge_l

reference = "the cat is on the mat"  # single reference for ROUGE

rouge_n(candidate, reference, n=1)  # ROUGE-1: unigram recall
rouge_n(candidate, reference, n=2)  # ROUGE-2: bigram recall
rouge_l(candidate, reference)       # ROUGE-L: longest common subsequence F-measure
```

### FID and Inception Score

For generative model evaluation, FID (Frechet Inception Distance) compares the distribution of generated features to real features. Inception Score measures both sharpness and diversity of generated samples. Both operate on pre-extracted feature vectors.

```python
from calibrax.metrics.plugins.image import FIDMetric, InceptionScoreMetric

real_features = jax.random.normal(jax.random.PRNGKey(10), (50, 64))
gen_features = jax.random.normal(jax.random.PRNGKey(11), (50, 64))
class_probs = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(12), (50, 10)), axis=-1)

fid = FIDMetric(feature_dim=64)
fid.update(real=real_features, generated=gen_features)
fid_result = fid.compute()  # fid_result["fid"] -- lower is better

is_metric = InceptionScoreMetric()
is_metric.update(probabilities=class_probs)
is_result = is_metric.compute()  # is_result["inception_score"] -- higher is better
```

Both are stateful (Tier 1) metrics: call `reset()` before a new evaluation, `update()` to accumulate data, and `compute()` to produce the final score.

### Distinct-N (Lexical Diversity)

Distinct-N measures the ratio of unique n-grams to total n-grams in a token sequence. Higher values indicate more diverse text.

```python
from calibrax.metrics.functional.text import distinct_n

diverse_tokens = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
distinct_n(diverse_tokens, n=1)  # high: all tokens unique
distinct_n(diverse_tokens, n=2)  # high: all bigrams unique
```

## Example Code

The script compares image quality at two noise levels:

```python
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
x = jnp.linspace(0, 1, 32)
original = jnp.outer(x, x)  # 32x32 gradient image

# Light noise
noise_small = jax.random.normal(subkey, original.shape) * 0.02
noisy_small = jnp.clip(original + noise_small, 0.0, 1.0)

# Heavy noise
noise_large = jax.random.normal(subkey, original.shape) * 0.15
noisy_large = jnp.clip(original + noise_large, 0.0, 1.0)

psnr(noisy_small, original)  # high dB
ssim(noisy_small, original)  # close to 1.0
psnr(noisy_large, original)  # lower dB
ssim(noisy_large, original)  # noticeably below 1.0
```

## Next Steps

- [Metric Learning Losses](metric-learning.md) -- contrastive, triplet, NTXent, and ArcFace losses for embedding training
- [Model Evaluation with Composition](model-evaluation.md) -- combine image quality metrics with quality gates and tracking
- [API Reference: `calibrax.metrics.functional.image`](../../api-reference/metrics/image.md) -- full image metric signatures
- [API Reference: `calibrax.metrics.functional.text`](../../api-reference/metrics/text.md) -- full text metric signatures
