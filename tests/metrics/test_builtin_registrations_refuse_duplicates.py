"""The built-in metric registrations register each name once and hide no clash.

``calibrax.metrics`` registers every built-in metric when it is first imported. A domain's
``register()`` run a second time meets names that are already held, and the registry refuses
them rather than the registration skipping them, so a built-in name defined twice fails the
import instead of keeping whichever came first.
"""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType

import pytest

from calibrax.core import DuplicateRegistrationError
from calibrax.metrics import MetricRegistry
from calibrax.metrics._registrations import (
    audio,
    calibration,
    classification,
    clustering,
    distance,
    divergence,
    fairness,
    forecasting,
    generative,
    geometric,
    graph,
    image,
    information,
    manifold,
    ranking,
    regression,
    segmentation,
    seismic,
    statistical,
    survival,
    text,
    uncertainty,
)


_DOMAIN_MODULES: tuple[ModuleType, ...] = (
    audio,
    calibration,
    classification,
    clustering,
    distance,
    divergence,
    fairness,
    forecasting,
    generative,
    geometric,
    graph,
    image,
    information,
    manifold,
    ranking,
    regression,
    segmentation,
    seismic,
    statistical,
    survival,
    text,
    uncertainty,
)


def _registered_entries() -> dict[str, object]:
    """Map each registered metric name to its entry, through the public API."""
    registry = MetricRegistry()
    return {name: registry.get(name) for name in registry}


def test_importing_calibrax_metrics_in_a_fresh_interpreter_succeeds() -> None:
    """Every built-in name is registered once, so the first import raises nothing."""
    result = subprocess.run(  # noqa: S603  # fixed argv: this interpreter and a literal
        [sys.executable, "-c", "import calibrax.metrics"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "domain", _DOMAIN_MODULES, ids=lambda module: module.__name__.rsplit(".", 1)[-1]
)
def test_registering_a_domain_again_is_refused(domain: ModuleType) -> None:
    """A domain registered a second time is refused and leaves the registry as it was."""
    before = _registered_entries()
    with pytest.raises(DuplicateRegistrationError, match="in MetricRegistry;"):
        domain.register()
    after = _registered_entries()
    assert after.keys() == before.keys()
    assert all(after[name] is before[name] for name in before)
