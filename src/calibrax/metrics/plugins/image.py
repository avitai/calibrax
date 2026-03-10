"""Image quality metrics requiring pretrained backbones.

Requires: calibrax[image] extra for full InceptionV3 feature extraction.
Tests use pre-extracted features to avoid requiring model weights.

Tier 1: FIDMetric (InceptionV3), InceptionScoreMetric (InceptionV3)
Tier 2: LPIPSMetric (VGG with learned calibration weights)
"""

from __future__ import annotations

import logging
from typing import Any

import flax.nnx as nnx
import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON
from calibrax.metrics.stateful._base import FrozenBackboneMetric, LearnedMetric


logger = logging.getLogger(__name__)


class FIDMetric(FrozenBackboneMetric):
    """Frechet Inception Distance using InceptionV3 features.

    Accumulates per-batch feature statistics (mean, covariance), then
    computes the Frechet distance between real and generated distributions.

    FID = |mu_r - mu_g|^2 + Tr(Sigma_r + Sigma_g - 2(Sigma_r Sigma_g)^0.5)

    Lower is better. FID = 0 means identical distributions.

    Args:
        feature_dim: Expected dimensionality of features.
            Defaults to 2048 (InceptionV3 pool3 layer).

    Examples:
        >>> fid = FIDMetric()
        >>> fid.update(real=real_features, generated=gen_features)
        >>> result = fid.compute()  # {"fid": 12.5}
    """

    def __init__(self, *, feature_dim: int = 2048) -> None:
        """Initialize FID metric.

        Args:
            feature_dim: Expected dimensionality of features.
        """
        super().__init__(name="fid")
        self._feature_dim = feature_dim
        self._real_features: list[Any] = []
        self._gen_features: list[Any] = []

    def reset(self) -> None:
        """Reset accumulated features."""
        self._real_features = []
        self._gen_features = []

    def _extract_features(self, **kwargs: Any) -> dict[str, Any]:
        """Extract or pass through features.

        Accepts either raw images (with backbone extraction) or
        pre-extracted features. For testing, pre-extracted features
        are used to avoid loading InceptionV3.

        Args:
            **kwargs: Must include "real" and "generated" arrays.
                If arrays have >2 dims, treats as images (would need backbone).
                If 2D, treats as pre-extracted features.

        Returns:
            Dict with "real" and "generated" feature arrays.
        """
        real = jnp.asarray(kwargs["real"])
        generated = jnp.asarray(kwargs["generated"])
        if real.ndim > 2:
            logger.warning(
                "Raw image input detected. Install calibrax[image] for "
                "InceptionV3 feature extraction. Using flattened features."
            )
            real = real.reshape(real.shape[0], -1)
            generated = generated.reshape(generated.shape[0], -1)
        return {"real": real, "generated": generated}

    def _accumulate(self, features: Any) -> None:
        """Accumulate feature batches.

        Args:
            features: Dict with "real" and "generated" feature arrays.
        """
        self._real_features.append(features["real"])
        self._gen_features.append(features["generated"])

    def _compute_from_accumulated(self) -> dict[str, float]:
        """Compute FID from accumulated features.

        Returns:
            {"fid": <value>} or {"fid": inf} if no features accumulated.
        """
        if not self._real_features or not self._gen_features:
            return {"fid": float("inf")}

        real_all = jnp.concatenate(self._real_features, axis=0)
        gen_all = jnp.concatenate(self._gen_features, axis=0)

        mu_real = jnp.mean(real_all, axis=0)
        mu_gen = jnp.mean(gen_all, axis=0)

        real_centered = real_all - mu_real
        gen_centered = gen_all - mu_gen

        n_real = max(real_all.shape[0] - 1, 1)
        n_gen = max(gen_all.shape[0] - 1, 1)
        cov_real = (real_centered.T @ real_centered) / n_real
        cov_gen = (gen_centered.T @ gen_centered) / n_gen

        diff = mu_real - mu_gen
        mean_diff_sq = float(jnp.sum(diff**2))

        # Matrix square root via eigendecomposition
        product = cov_real @ cov_gen
        eigvals = jnp.linalg.eigvalsh(product)
        eigvals = jnp.maximum(eigvals, 0.0)
        sqrt_product_trace = float(jnp.sum(jnp.sqrt(eigvals)))

        trace_sum = float(jnp.trace(cov_real) + jnp.trace(cov_gen))
        fid = mean_diff_sq + trace_sum - 2.0 * sqrt_product_trace
        return {"fid": max(fid, 0.0)}


class InceptionScoreMetric(FrozenBackboneMetric):
    """Inception Score using InceptionV3 class probabilities.

    IS = exp(E[KL(p(y|x) || p(y))])

    Measures quality (low entropy per-image) and diversity
    (high entropy marginal). Higher is better.

    Examples:
        >>> is_metric = InceptionScoreMetric()
        >>> is_metric.update(probabilities=class_probs)
        >>> result = is_metric.compute()  # {"inception_score": 8.5}
    """

    def __init__(self) -> None:
        """Initialize Inception Score metric."""
        super().__init__(name="inception_score")
        self._all_probs: list[Any] = []

    def reset(self) -> None:
        """Reset accumulated probabilities."""
        self._all_probs = []

    def _extract_features(self, **kwargs: Any) -> Any:
        """Extract or pass through class probabilities.

        Args:
            **kwargs: Must include "probabilities" -- softmax output from
                InceptionV3 (or mock probabilities for testing).

        Returns:
            Probability array of shape (batch_size, num_classes).
        """
        return jnp.asarray(kwargs["probabilities"])

    def _accumulate(self, features: Any) -> None:
        """Accumulate probability batches.

        Args:
            features: Probability array of shape (batch_size, num_classes).
        """
        self._all_probs.append(features)

    def _compute_from_accumulated(self) -> dict[str, float]:
        """Compute Inception Score from accumulated probabilities.

        Returns:
            {"inception_score": <value>} or {"inception_score": 0.0} if empty.
        """
        if not self._all_probs:
            return {"inception_score": 0.0}

        all_probs = jnp.concatenate(self._all_probs, axis=0)

        marginal = jnp.mean(all_probs, axis=0, keepdims=True)

        kl = all_probs * (jnp.log(all_probs + _EPSILON) - jnp.log(marginal + _EPSILON))
        kl_per_sample = jnp.sum(kl, axis=1)
        mean_kl = jnp.mean(kl_per_sample)

        inception_score = float(jnp.exp(mean_kl))
        return {"inception_score": inception_score}


class LPIPSMetric(LearnedMetric):
    """Learned Perceptual Image Patch Similarity.

    Tier 2: Uses pretrained VGG features with LEARNED linear calibration
    weights that were trained on human perceptual judgments.

    The calibration weights are trainable -- unlike FID (Tier 1) which
    uses only frozen features.

    Args:
        feature_channels: Number of channels at each VGG layer.
            Defaults to standard VGG16 configuration.
        rngs: RNG streams for parameter initialization.

    Examples:
        >>> lpips = LPIPSMetric(rngs=nnx.Rngs(0))
        >>> lpips.update(image_a=img1, image_b=img2)
        >>> result = lpips.compute()  # {"lpips": 0.12}
    """

    def __init__(
        self,
        *,
        feature_channels: tuple[int, ...] = (64, 128, 256, 512, 512),
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize LPIPS metric.

        Args:
            feature_channels: Number of channels at each VGG layer.
            rngs: RNG streams for parameter initialization.
        """
        super().__init__(name="lpips", rngs=rngs)
        self._layer_weights = nnx.List(
            [nnx.Linear(in_features=ch, out_features=1, rngs=rngs) for ch in feature_channels]
        )
        self._scores: list[float] = []

    def reset(self) -> None:
        """Reset accumulated scores."""
        self._scores = []

    def update(self, **kwargs: Any) -> None:
        """Compute LPIPS between two images.

        Accepts pre-extracted VGG features as lists of per-layer activations,
        or raw image pairs (placeholder for future backbone integration).

        Args:
            **kwargs: Must include "features_a" and "features_b" as lists
                of per-layer feature arrays.
        """
        features_a = kwargs.get("features_a", [])
        features_b = kwargs.get("features_b", [])

        total = 0.0
        for layer_feat_a, layer_feat_b, linear in zip(features_a, features_b, self._layer_weights):
            diff = jnp.asarray(layer_feat_a) - jnp.asarray(layer_feat_b)
            weighted = linear(diff)
            total += float(jnp.mean(weighted**2))

        self._scores.append(total)

    def compute(self) -> dict[str, float]:
        """Compute mean LPIPS from accumulated scores.

        Returns:
            {"lpips": <value>} or {"lpips": 0.0} if no updates.
        """
        if not self._scores:
            return {"lpips": 0.0}
        return {"lpips": sum(self._scores) / len(self._scores)}
