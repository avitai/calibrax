"""Registrations of the built-in ``seismic`` metrics."""

from __future__ import annotations

from calibrax.core.models import MetricDirection
from calibrax.metrics._registrations._spec import _BuiltinMetricSpec, _register_specs
from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import MetricProperties
from calibrax.metrics.functional.seismic import (
    seismic_cross_correlation_misfit,
    seismic_l2_misfit,
    seismic_l2_misfit_normalized,
    seismic_relative_model_misfit,
    seismic_soft_dtw_misfit,
    seismic_velocity_ssim_misfit,
    seismic_wasserstein2_misfit,
)


def register() -> None:
    """Register 7 seismic (FWI) metrics at import time."""
    registry = MetricRegistry()
    _register_specs(
        registry,
        [
            _BuiltinMetricSpec(
                "seismic_l2_misfit",
                seismic_l2_misfit,
                "L2 data misfit: mean squared difference of seismic gathers",
                MetricDirection.LOWER,
                MetricProperties(is_symmetric=True, is_differentiable=True),
                domain="seismic",
            ),
            _BuiltinMetricSpec(
                "seismic_l2_misfit_normalized",
                seismic_l2_misfit_normalized,
                "Normalized L2 data misfit: MSE divided by observed-data energy",
                MetricDirection.LOWER,
                MetricProperties(is_differentiable=True),
                domain="seismic",
            ),
            _BuiltinMetricSpec(
                "seismic_relative_model_misfit",
                seismic_relative_model_misfit,
                "Relative model misfit: ||m_pred - m_true||_2 / ||m_true||_2",
                MetricDirection.LOWER,
                MetricProperties(is_differentiable=True),
                domain="seismic",
            ),
            _BuiltinMetricSpec(
                "seismic_cross_correlation_misfit",
                seismic_cross_correlation_misfit,
                "Cross-correlation misfit: 1 - max normalized cross-correlation per trace",
                MetricDirection.LOWER,
                MetricProperties(is_symmetric=True, is_differentiable=True),
                domain="seismic",
            ),
            _BuiltinMetricSpec(
                "seismic_velocity_ssim_misfit",
                seismic_velocity_ssim_misfit,
                "SSIM misfit on velocity models: 1 - structural similarity",
                MetricDirection.LOWER,
                MetricProperties(is_differentiable=True),
                domain="seismic",
            ),
            _BuiltinMetricSpec(
                "seismic_wasserstein2_misfit",
                seismic_wasserstein2_misfit,
                "Squared Wasserstein-2 misfit per trace on normalized positive envelopes",
                MetricDirection.LOWER,
                MetricProperties(is_symmetric=True, is_differentiable=True),
                domain="seismic",
            ),
            _BuiltinMetricSpec(
                "seismic_soft_dtw_misfit",
                seismic_soft_dtw_misfit,
                "Soft-DTW misfit: differentiable dynamic time warping per trace",
                MetricDirection.LOWER,
                MetricProperties(is_symmetric=True, is_differentiable=True),
                domain="seismic",
            ),
        ],
    )
