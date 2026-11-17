"""Registrations of the built-in ``survival`` metrics."""

from __future__ import annotations

from calibrax.core.models import MetricDirection
from calibrax.metrics._registrations._spec import _BuiltinMetricSpec, _register_specs
from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import MetricProperties, MetricSignature
from calibrax.metrics.functional.survival import harrell_c_index, uno_c_index


def register() -> None:
    """Register censored concordance metrics at import time.

    Note: ``censored_coverage_bounds`` returns a ``(lower_bound, upper_bound)``
    pair rather than a single scalar, so it is not registered in ``MetricRegistry``.
    """
    registry = MetricRegistry()
    rank_concordance = MetricProperties(is_differentiable=False, is_jit_compatible=True)
    _register_specs(
        registry,
        [
            _BuiltinMetricSpec(
                "harrell_c_index",
                harrell_c_index,
                "Harrell concordance index for exact or censored outcomes",
                MetricDirection.HIGHER,
                rank_concordance,
                signature=MetricSignature.PREDICTIONS_TARGETS,
                domain="survival",
            ),
            _BuiltinMetricSpec(
                "uno_c_index",
                uno_c_index,
                "IPCW Kaplan-Meier concordance index (Uno et al. 2011)",
                MetricDirection.HIGHER,
                rank_concordance,
                signature=MetricSignature.CUSTOM,
                domain="survival",
            ),
        ],
    )
