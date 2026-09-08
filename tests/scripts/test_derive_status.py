"""The derived-status check fails when the README's subpackage table drifts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def derive_status() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "derive_status", REPO_ROOT / "scripts" / "derive_status.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["derive_status"] = module
    spec.loader.exec_module(module)
    return module


def test_the_registry_matches_the_readme(derive_status: ModuleType) -> None:
    metrics = derive_status.collect_metrics(REPO_ROOT, "calibrax")
    drifted = [metric.label for metric in metrics if metric.is_drifted]
    assert drifted == []
    by_label = {metric.label: metric for metric in metrics}
    assert by_label["tier0_metrics"].asserted == by_label["tier0_metrics"].measured
    assert by_label["metric_domains"].asserted == by_label["metric_domains"].measured
    assert int(by_label["tier0_metrics"].measured) > 0


def test_a_stale_readme_count_is_drift(derive_status: ModuleType, tmp_path: Path) -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    stale = readme.replace("(111 registered Tier 0 metrics", "(110 registered Tier 0 metrics", 1)
    edited = tmp_path / "README.md"
    edited.write_text(stale)

    metrics = derive_status.collect_metrics(REPO_ROOT, "calibrax", readme=edited)

    assert [metric.label for metric in metrics if metric.is_drifted] == ["tier0_metrics"]
