"""Image quality metrics -- pure math on pixel arrays.

All metrics in this module are pure mathematical operations on image arrays.
No pretrained models, no neural networks, no external dependencies beyond JAX.

Includes: PSNR, SSIM, MS-SSIM, and Vendi Score.
Registered with ``domain="image"``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON


def _gaussian_kernel_1d(size: int, sigma: float) -> Any:
    """Create 1D Gaussian kernel.

    Args:
        size: Kernel size (should be odd).
        sigma: Standard deviation.

    Returns:
        1D Gaussian kernel array normalized to sum to 1.
    """
    coords = jnp.arange(size, dtype=jnp.float32) - (size - 1) / 2.0
    kernel = jnp.exp(-0.5 * (coords / sigma) ** 2)
    return kernel / jnp.sum(kernel)


def _gaussian_kernel_2d(size: int, sigma: float) -> Any:
    """Create 2D Gaussian kernel via outer product.

    Args:
        size: Kernel size (should be odd).
        sigma: Standard deviation.

    Returns:
        2D Gaussian kernel array of shape (size, size).
    """
    k1d = _gaussian_kernel_1d(size, sigma)
    return jnp.outer(k1d, k1d)


def _conv2d(image: Any, kernel: Any) -> Any:
    """Apply 2D convolution using JAX's lax.conv.

    Args:
        image: 2D image array of shape (H, W).
        kernel: 2D kernel array of shape (kH, kW).

    Returns:
        Convolved image with same spatial dimensions (via padding).
    """
    # Reshape for lax.conv_general_dilated: (N, C, H, W)
    img = image[None, None, :, :]
    k = kernel[None, None, :, :]
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    result = jnp.pad(img, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="edge")

    from jax import lax

    out = lax.conv_general_dilated(
        result,
        k,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return out[0, 0]


def _ssim_single_channel(
    a: Any,
    b: Any,
    *,
    max_val: float,
    filter_size: int,
    filter_sigma: float,
    k1: float,
    k2: float,
) -> tuple[Any, Any, Any]:
    """Compute SSIM components for a single channel.

    Args:
        a: First image, shape (H, W).
        b: Second image, shape (H, W).
        max_val: Maximum pixel value.
        filter_size: Gaussian window size.
        filter_sigma: Gaussian standard deviation.
        k1: Stability constant for luminance.
        k2: Stability constant for contrast.

    Returns:
        Tuple of (ssim_value, contrast_structure, luminance).
    """
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    kernel = _gaussian_kernel_2d(filter_size, filter_sigma)

    mu_a = _conv2d(a, kernel)
    mu_b = _conv2d(b, kernel)

    mu_a_sq = mu_a**2
    mu_b_sq = mu_b**2
    mu_ab = mu_a * mu_b

    sigma_a_sq = _conv2d(a**2, kernel) - mu_a_sq
    sigma_b_sq = _conv2d(b**2, kernel) - mu_b_sq
    sigma_ab = _conv2d(a * b, kernel) - mu_ab

    # Clamp variances to avoid negative values from numerical errors
    sigma_a_sq = jnp.maximum(sigma_a_sq, 0.0)
    sigma_b_sq = jnp.maximum(sigma_b_sq, 0.0)

    luminance = (2 * mu_ab + c1) / (mu_a_sq + mu_b_sq + c1)
    contrast_structure = (2 * sigma_ab + c2) / (sigma_a_sq + sigma_b_sq + c2)

    ssim_map = luminance * contrast_structure
    return (
        jnp.mean(ssim_map),
        jnp.mean(contrast_structure),
        jnp.mean(luminance),
    )


def psnr(predictions: Any, targets: Any, *, max_val: float = 1.0) -> Any:
    """Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(max_val^2 / MSE). Measured in dB. Higher is better.
    For identical images, returns a very large value (clamped to avoid inf).

    Args:
        predictions: Predicted image array (any shape).
        targets: Ground truth image array (same shape).
        max_val: Maximum pixel value (1.0 for [0,1] range, 255 for uint8).

    Returns:
        PSNR value in dB.

    Examples:
        >>> import jax.numpy as jnp
        >>> img = jnp.ones((8, 8)) * 0.5
        >>> psnr(img, img)  # Very high value (identical images)
        ...
    """
    predictions = jnp.asarray(predictions, dtype=jnp.float32)
    targets = jnp.asarray(targets, dtype=jnp.float32)
    mse_val = jnp.mean((predictions - targets) ** 2)

    # Clamp MSE to avoid inf
    mse_val = jnp.maximum(mse_val, _EPSILON)
    return 10.0 * jnp.log10(max_val**2 / mse_val)


def ssim(
    predictions: Any,
    targets: Any,
    *,
    max_val: float = 1.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Any:
    """Structural Similarity Index Measure.

    Computes luminance, contrast, and structure similarity using a
    Gaussian window. For multi-channel images, averages across channels.

    Args:
        predictions: Predicted image, shape (H, W) or (H, W, C).
        targets: Ground truth image, same shape.
        max_val: Maximum pixel value.
        filter_size: Gaussian window size (should be odd).
        filter_sigma: Gaussian standard deviation.
        k1: Luminance stability constant.
        k2: Contrast stability constant.

    Returns:
        SSIM value in [0, 1]. 1.0 = identical images.

    Examples:
        >>> import jax.numpy as jnp
        >>> img = jnp.ones((32, 32)) * 0.5
        >>> ssim(img, img)
        1.0
    """
    predictions = jnp.asarray(predictions, dtype=jnp.float32)
    targets = jnp.asarray(targets, dtype=jnp.float32)

    if predictions.ndim == 2:
        val, _, _ = _ssim_single_channel(
            predictions,
            targets,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2,
        )
        return val

    # Multi-channel: vectorize over channel axis with vmap
    def _ssim_for_channel(pred_ch: Any, tgt_ch: Any) -> Any:
        val, _, _ = _ssim_single_channel(
            pred_ch,
            tgt_ch,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2,
        )
        return val

    # Transpose to (C, H, W) for vmap over axis 0
    pred_channels = jnp.moveaxis(predictions, -1, 0)
    tgt_channels = jnp.moveaxis(targets, -1, 0)
    channel_ssims = jax.vmap(_ssim_for_channel)(pred_channels, tgt_channels)
    return jnp.mean(channel_ssims)


def ms_ssim(
    predictions: Any,
    targets: Any,
    *,
    max_val: float = 1.0,
    power_factors: tuple[float, ...] | None = None,
) -> Any:
    """Multi-Scale Structural Similarity Index.

    Computes SSIM at multiple downsample scales and combines with
    power weights. Requires images large enough for the number of scales.

    Args:
        predictions: Predicted image, shape (H, W) or (H, W, C).
        targets: Ground truth image, same shape.
        max_val: Maximum pixel value.
        power_factors: Weights per scale. Default: (0.0448, 0.2856, 0.3001, 0.2363, 0.1333).

    Returns:
        MS-SSIM value in [0, 1]. 1.0 = identical images.

    Examples:
        >>> import jax.numpy as jnp
        >>> img = jnp.ones((160, 160)) * 0.5
        >>> ms_ssim(img, img)
        1.0
    """
    if power_factors is None:
        power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

    predictions = jnp.asarray(predictions, dtype=jnp.float32)
    targets = jnp.asarray(targets, dtype=jnp.float32)

    n_scales = len(power_factors)
    is_multichannel = predictions.ndim == 3

    def _ssim_components_for_channel(pred_ch: Any, tgt_ch: Any) -> tuple[Any, Any]:
        val, cs, _ = _ssim_single_channel(
            pred_ch,
            tgt_ch,
            max_val=max_val,
            filter_size=11,
            filter_sigma=1.5,
            k1=0.01,
            k2=0.03,
        )
        return val, cs

    cs_values = []
    for scale in range(n_scales):
        if is_multichannel:
            pred_channels = jnp.moveaxis(predictions, -1, 0)
            tgt_channels = jnp.moveaxis(targets, -1, 0)
            vals, css = jax.vmap(_ssim_components_for_channel)(pred_channels, tgt_channels)
            ssim_val = jnp.mean(vals)
            cs_val = jnp.mean(css)
        else:
            ssim_val, cs_val, lum_val = _ssim_single_channel(
                predictions,
                targets,
                max_val=max_val,
                filter_size=11,
                filter_sigma=1.5,
                k1=0.01,
                k2=0.03,
            )

        if scale < n_scales - 1:
            cs_values.append(cs_val)
            # Downsample by 2x
            h, w = predictions.shape[0], predictions.shape[1]
            new_h, new_w = max(h // 2, 1), max(w // 2, 1)
            if is_multichannel:
                from jax import image as jax_image

                predictions = jax_image.resize(
                    predictions, (new_h, new_w, predictions.shape[2]), method="bilinear"
                )
                targets = jax_image.resize(
                    targets, (new_h, new_w, targets.shape[2]), method="bilinear"
                )
            else:
                from jax import image as jax_image

                predictions = jax_image.resize(predictions, (new_h, new_w), method="bilinear")
                targets = jax_image.resize(targets, (new_h, new_w), method="bilinear")
        else:
            cs_values.append(ssim_val)  # Last scale uses full SSIM

    # Product of contrast-structure raised to power factors
    cs_arr = jnp.array(cs_values)
    pf_arr = jnp.array(power_factors)
    result = jnp.prod(jnp.maximum(cs_arr, _EPSILON) ** pf_arr)

    return result


def vendi_score(similarity_matrix: Any) -> Any:
    """Vendi Score: diversity measure via eigenvalue entropy.

    Computes exp(entropy of eigenvalues) of a similarity matrix.
    Higher values indicate more diversity.

    Args:
        similarity_matrix: Square similarity matrix of shape (n, n).
            Values should be in [0, 1] with 1 on diagonal.

    Returns:
        Vendi score >= 1.0. Score of 1.0 means all items identical,
        score of n means maximum diversity (all items orthogonal).

    Examples:
        >>> import jax.numpy as jnp
        >>> identity = jnp.eye(3)  # 3 orthogonal items
        >>> vendi_score(identity)  # 3.0
        ...
    """
    sim = jnp.asarray(similarity_matrix, dtype=jnp.float32)
    eigenvalues = jnp.linalg.eigvalsh(sim)

    # Normalize eigenvalues
    eigenvalues = jnp.maximum(eigenvalues, 0.0)  # Remove numerical negatives
    total = jnp.sum(eigenvalues)
    probs = eigenvalues / (total + _EPSILON)

    # Entropy: -sum(p * log(p)) for p > 0
    log_probs = jnp.where(probs > _EPSILON, jnp.log(probs), 0.0)
    entropy = -jnp.sum(probs * log_probs)

    return jnp.where(total < _EPSILON, 1.0, jnp.exp(entropy))
