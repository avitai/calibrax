"""Tests for audio quality metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.audio import (
    mel_cepstral_distortion,
    signal_to_noise_ratio,
    spectral_convergence,
)


class TestSpectralConvergence:
    """Tests for spectral_convergence."""

    def test_identical_signals(self) -> None:
        sig = jnp.sin(jnp.linspace(0, 10, 256))
        result = spectral_convergence(sig, sig)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_different_signals(self) -> None:
        a = jnp.sin(jnp.linspace(0, 10, 256))
        b = jnp.cos(jnp.linspace(0, 10, 256))
        result = spectral_convergence(a, b)
        assert result > 0.0

    def test_returns_jax_scalar(self) -> None:
        sig = jnp.ones(64)
        result = spectral_convergence(sig, sig)
        assert isinstance(result, jax.Array)


class TestMelCepstralDistortion:
    """Tests for mel_cepstral_distortion."""

    def test_identical_signals(self) -> None:
        sig = jnp.sin(jnp.linspace(0, 10, 256))
        result = mel_cepstral_distortion(sig, sig)
        assert result == pytest.approx(0.0, abs=0.1)

    def test_different_signals(self) -> None:
        a = jnp.sin(jnp.linspace(0, 10, 256))
        b = jnp.cos(jnp.linspace(0, 10, 256))
        result = mel_cepstral_distortion(a, b)
        assert result > 0.0

    def test_returns_jax_scalar(self) -> None:
        sig = jnp.ones(128)
        result = mel_cepstral_distortion(sig, sig)
        assert isinstance(result, jax.Array)


class TestSignalToNoiseRatio:
    """Tests for signal_to_noise_ratio."""

    def test_no_noise(self) -> None:
        signal = jnp.ones(100)
        noise = jnp.ones(100) * 1e-4
        result = signal_to_noise_ratio(signal, noise)
        assert result > 70  # High SNR

    def test_equal_power(self) -> None:
        signal = jnp.ones(100)
        noise = jnp.ones(100)
        result = signal_to_noise_ratio(signal, noise)
        assert result == pytest.approx(0.0, abs=0.1)

    def test_returns_jax_scalar(self) -> None:
        signal = jnp.ones(10)
        noise = jnp.ones(10) * 0.5
        result = signal_to_noise_ratio(signal, noise)
        assert isinstance(result, jax.Array)


class TestAudioMetricRegistration:
    """Tests for audio metric registration."""

    def test_all_registered(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        expected = [
            "spectral_convergence",
            "mel_cepstral_distortion",
            "signal_to_noise_ratio",
        ]
        for name in expected:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_audio_domain(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        audio_metrics = registry.list_by_domain("audio")
        assert len(audio_metrics) == 3
