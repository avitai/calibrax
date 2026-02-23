"""Tests for calibrax.analysis.scaling module."""

from __future__ import annotations

import pytest

from calibrax.analysis.scaling import scaling_fit


class TestScalingFit:
    """Tests for scaling_fit."""

    def test_linear_data(self) -> None:
        """Linear data should produce exponent ~ 1.0."""
        sizes = [1.0, 2.0, 4.0, 8.0, 16.0]
        values = [s * 2.5 for s in sizes]  # value = 2.5 * size^1
        result = scaling_fit(sizes, values)
        assert result.exponent == pytest.approx(1.0, abs=0.1)
        assert result.complexity == "O(n)"
        assert result.r_squared > 0.99

    def test_quadratic_data(self) -> None:
        """Quadratic data should produce exponent ~ 2.0."""
        sizes = [1.0, 2.0, 4.0, 8.0, 16.0]
        values = [s**2 for s in sizes]
        result = scaling_fit(sizes, values)
        assert result.exponent == pytest.approx(2.0, abs=0.1)
        assert result.complexity == "O(n^2)"
        assert result.r_squared > 0.99

    def test_constant_data(self) -> None:
        """Constant values should produce exponent 0.0."""
        sizes = [1.0, 2.0, 4.0, 8.0]
        values = [5.0, 5.0, 5.0, 5.0]
        result = scaling_fit(sizes, values)
        assert result.exponent == pytest.approx(0.0)
        assert result.complexity == "O(1)"
        assert result.r_squared == pytest.approx(1.0)

    def test_cubic_data(self) -> None:
        """Cubic data should produce exponent ~ 3.0."""
        sizes = [1.0, 2.0, 4.0, 8.0, 16.0]
        values = [s**3 for s in sizes]
        result = scaling_fit(sizes, values)
        assert result.exponent == pytest.approx(3.0, abs=0.1)
        assert result.complexity == "O(n^3)"

    def test_r_squared_perfect_fit(self) -> None:
        """Perfect power law should have R^2 ~ 1.0."""
        sizes = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        values = [3.0 * s**1.5 for s in sizes]
        result = scaling_fit(sizes, values)
        assert result.r_squared == pytest.approx(1.0, abs=0.001)

    def test_empty_raises_value_error(self) -> None:
        """Empty inputs should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            scaling_fit([], [])

    def test_mismatched_raises_value_error(self) -> None:
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="Mismatched"):
            scaling_fit([1.0, 2.0], [1.0])

    def test_non_positive_filtered(self) -> None:
        """Non-positive values should be filtered out."""
        sizes = [0.0, -1.0, 1.0, 2.0, 4.0]
        values = [0.0, -1.0, 1.0, 2.0, 4.0]
        result = scaling_fit(sizes, values)
        assert result.exponent == pytest.approx(1.0, abs=0.1)

    def test_all_non_positive_returns_zero(self) -> None:
        """All non-positive should return zero exponent."""
        sizes = [0.0, -1.0]
        values = [0.0, -2.0]
        result = scaling_fit(sizes, values)
        assert result.exponent == 0.0
        assert result.r_squared == 0.0

    def test_sqrt_data(self) -> None:
        """Square root data should produce exponent ~ 0.5."""
        import math

        sizes = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0]
        values = [math.sqrt(s) for s in sizes]
        result = scaling_fit(sizes, values)
        assert result.exponent == pytest.approx(0.5, abs=0.1)
        assert result.complexity == "O(sqrt(n))"
