"""Tests for image quality metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.image import (
    ms_ssim,
    psnr,
    ssim,
    vendi_score,
)


class TestPSNR:
    """Tests for psnr."""

    def test_identical_images(self) -> None:
        img = jnp.ones((16, 16)) * 0.5
        result = psnr(img, img)
        assert result > 60  # Very high for identical images

    def test_known_value(self) -> None:
        # MSE = 0.01, max_val = 1.0 → PSNR = 10*log10(1/0.01) = 20 dB
        a = jnp.zeros((100,))
        b = jnp.ones((100,)) * 0.1  # MSE = 0.01
        result = psnr(a, b, max_val=1.0)
        assert result == pytest.approx(20.0, abs=0.1)

    def test_different_max_val(self) -> None:
        a = jnp.zeros((100,))
        b = jnp.ones((100,)) * 25.5  # MSE ~= 650.25 for uint8 range
        result = psnr(a, b, max_val=255.0)
        assert isinstance(result, jax.Array)

    def test_returns_jax_scalar(self) -> None:
        img = jnp.ones((8, 8))
        result = psnr(img, img)
        assert isinstance(result, jax.Array)


class TestSSIM:
    """Tests for ssim."""

    def test_identical_images(self) -> None:
        img = jnp.ones((32, 32)) * 0.5
        result = ssim(img, img)
        assert result == pytest.approx(1.0, abs=1e-3)

    def test_completely_different(self) -> None:
        a = jnp.zeros((32, 32))
        b = jnp.ones((32, 32))
        result = ssim(a, b)
        assert result < 0.1

    def test_range(self) -> None:
        a = jnp.ones((32, 32)) * 0.3
        b = jnp.ones((32, 32)) * 0.7
        result = ssim(a, b)
        assert -0.01 <= result <= 1.01

    def test_multichannel(self) -> None:
        img = jnp.ones((32, 32, 3)) * 0.5
        result = ssim(img, img)
        assert result == pytest.approx(1.0, abs=1e-3)

    def test_symmetry(self) -> None:
        a = jnp.ones((32, 32)) * 0.3
        b = jnp.ones((32, 32)) * 0.7
        assert ssim(a, b) == pytest.approx(ssim(b, a), abs=1e-6)

    def test_returns_jax_scalar(self) -> None:
        img = jnp.ones((32, 32))
        result = ssim(img, img)
        assert isinstance(result, jax.Array)


class TestMSSSIM:
    """Tests for ms_ssim."""

    def test_identical_images(self) -> None:
        img = jnp.ones((160, 160)) * 0.5
        result = ms_ssim(img, img)
        assert result == pytest.approx(1.0, abs=1e-3)

    def test_range(self) -> None:
        a = jnp.ones((160, 160)) * 0.3
        b = jnp.ones((160, 160)) * 0.7
        result = ms_ssim(a, b)
        assert -0.01 <= result <= 1.01

    def test_returns_jax_scalar(self) -> None:
        img = jnp.ones((160, 160)) * 0.5
        result = ms_ssim(img, img)
        assert isinstance(result, jax.Array)


class TestVendiScore:
    """Tests for vendi_score."""

    def test_identical_items(self) -> None:
        # All-ones matrix: all items identical → score = 1.0
        sim = jnp.ones((3, 3))
        result = vendi_score(sim)
        assert result == pytest.approx(1.0, abs=0.1)

    def test_orthogonal_items(self) -> None:
        # Identity matrix: all items orthogonal → score = n
        sim = jnp.eye(5)
        result = vendi_score(sim)
        assert result == pytest.approx(5.0, abs=0.1)

    def test_returns_jax_scalar(self) -> None:
        sim = jnp.eye(3)
        result = vendi_score(sim)
        assert isinstance(result, jax.Array)

    def test_two_items(self) -> None:
        sim = jnp.eye(2)
        result = vendi_score(sim)
        assert result == pytest.approx(2.0, abs=0.1)


class TestImageMetricRegistration:
    """Tests for image metric registration."""

    def test_all_registered(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        expected = ["psnr", "ssim", "ms_ssim", "vendi_score"]
        for name in expected:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_image_domain(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        image_metrics = registry.list_by_domain("image")
        assert len(image_metrics) == 4

    def test_all_direction_higher(self) -> None:
        from calibrax.core.models import MetricDirection
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        for m in registry.list_by_domain("image"):
            assert m.direction == MetricDirection.HIGHER
