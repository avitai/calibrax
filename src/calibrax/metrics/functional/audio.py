"""Audio quality metrics -- FFT-based, pure math.

All metrics are pure mathematical operations on audio signal arrays.
No pretrained models, no external audio processing libraries.

Includes: spectral_convergence, mel_cepstral_distortion, signal_to_noise_ratio.
Registered with ``domain="audio"``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON


def spectral_convergence(predictions: Any, targets: Any) -> Any:
    """Spectral convergence between predicted and target signals.

    Computes ||STFT(targets) - STFT(predictions)||_F / ||STFT(targets)||_F.

    Args:
        predictions: Predicted signal array, shape (n,).
        targets: Target signal array, shape (n,).

    Returns:
        Spectral convergence ratio. Lower is better. 0.0 = identical spectra.

    Examples:
        >>> import jax.numpy as jnp
        >>> sig = jnp.sin(jnp.linspace(0, 10, 256))
        >>> spectral_convergence(sig, sig)
        0.0
    """
    predictions = jnp.asarray(predictions, dtype=jnp.float32)
    targets = jnp.asarray(targets, dtype=jnp.float32)

    spec_pred = jnp.fft.rfft(predictions)
    spec_tgt = jnp.fft.rfft(targets)

    diff_norm = jnp.sqrt(jnp.sum(jnp.abs(spec_tgt - spec_pred) ** 2))
    tgt_norm = jnp.sqrt(jnp.sum(jnp.abs(spec_tgt) ** 2))

    return diff_norm / (tgt_norm + _EPSILON)


def mel_cepstral_distortion(predictions: Any, targets: Any, *, num_mels: int = 80) -> Any:
    """Mel Cepstral Distortion between two signals.

    Computes (10/ln(10)) * sqrt(2 * sum((c_pred - c_tgt)^2)) on
    cepstral coefficients derived from the signal spectra.

    Args:
        predictions: Predicted signal array, shape (n,).
        targets: Target signal array, shape (n,).
        num_mels: Number of mel-frequency coefficients to use.

    Returns:
        MCD value in dB. Lower is better. 0.0 = identical signals.

    Examples:
        >>> import jax.numpy as jnp
        >>> sig = jnp.sin(jnp.linspace(0, 10, 256))
        >>> mel_cepstral_distortion(sig, sig)
        0.0
    """
    predictions = jnp.asarray(predictions, dtype=jnp.float32)
    targets = jnp.asarray(targets, dtype=jnp.float32)

    # Compute magnitude spectra
    spec_pred = jnp.abs(jnp.fft.rfft(predictions))
    spec_tgt = jnp.abs(jnp.fft.rfft(targets))

    # Log magnitude spectra
    log_pred = jnp.log(spec_pred + _EPSILON)
    log_tgt = jnp.log(spec_tgt + _EPSILON)

    # Cepstral coefficients via inverse FFT (simplified DCT)
    cep_pred = jnp.fft.irfft(log_pred)[:num_mels]
    cep_tgt = jnp.fft.irfft(log_tgt)[:num_mels]

    # MCD formula
    diff_sq = jnp.sum((cep_pred - cep_tgt) ** 2)
    mcd = (10.0 / jnp.log(10.0)) * jnp.sqrt(2.0 * diff_sq + _EPSILON)

    return mcd


def signal_to_noise_ratio(signal: Any, noise: Any) -> Any:
    """Signal-to-Noise Ratio in dB.

    SNR = 10 * log10(|signal|^2 / |noise|^2).

    Args:
        signal: Clean signal array.
        noise: Noise signal array (same shape as signal).

    Returns:
        SNR value in dB. Higher is better.

    Examples:
        >>> import jax.numpy as jnp
        >>> signal = jnp.ones(100)
        >>> noise = jnp.ones(100) * 0.01
        >>> signal_to_noise_ratio(signal, noise)  # ~40 dB
        ...
    """
    signal = jnp.asarray(signal, dtype=jnp.float32)
    noise = jnp.asarray(noise, dtype=jnp.float32)

    signal_power = jnp.mean(signal**2)
    noise_power = jnp.mean(noise**2)

    return 10.0 * jnp.log10(signal_power / (noise_power + _EPSILON))
