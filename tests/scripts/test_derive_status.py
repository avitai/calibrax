"""The derived-status check fails when a documented registry claim drifts from the registry."""

from __future__ import annotations

import importlib.util
import re
import shutil
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


@pytest.fixture
def documented_tree(derive_status: ModuleType, tmp_path: Path) -> Path:
    """Copy the README and every document the check reads into a scratch root."""
    for relative in ("README.md", *derive_status.DOCUMENT_SOURCES):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)
    return tmp_path


def _drifted(derive_status: ModuleType, root: Path) -> list[str]:
    return [m.label for m in derive_status.collect_metrics(root, "calibrax") if m.is_drifted]


def _measured(derive_status: ModuleType, label: str) -> str:
    metrics = derive_status.collect_metrics(REPO_ROOT, "calibrax")
    return next(metric.measured for metric in metrics if metric.label == label)


def _substitute(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text()
    edited, count = re.subn(pattern, replacement, text, count=1)
    assert count == 1, f"{pattern!r} did not match in {path}"
    path.write_text(edited)


def test_the_registry_matches_the_readme(derive_status: ModuleType) -> None:
    metrics = derive_status.collect_metrics(REPO_ROOT, "calibrax")
    drifted = [
        f"{metric.label}: asserted={metric.asserted} measured={metric.measured} "
        f"missing={metric.missing}"
        for metric in metrics
        if metric.is_drifted
    ]
    assert drifted == []
    by_label = {metric.label: metric for metric in metrics}
    assert by_label["tier0_metrics"].asserted == by_label["tier0_metrics"].measured
    assert by_label["metric_domains"].asserted == by_label["metric_domains"].measured
    assert int(by_label["tier0_metrics"].measured) > 0


def test_a_stale_readme_count_is_drift(derive_status: ModuleType, tmp_path: Path) -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    match = re.search(r"\((\d+) registered Tier 0 metrics", readme)
    assert match is not None
    stale = readme.replace(
        match.group(0), f"({int(match.group(1)) + 1} registered Tier 0 metrics", 1
    )
    edited = tmp_path / "README.md"
    edited.write_text(stale)

    metrics = derive_status.collect_metrics(REPO_ROOT, "calibrax", readme=edited)

    assert [metric.label for metric in metrics if metric.is_drifted] == ["tier0_metrics"]


def test_an_unedited_copy_of_the_documents_has_no_drift(
    derive_status: ModuleType, documented_tree: Path
) -> None:
    assert _drifted(derive_status, documented_tree) == []


def test_a_stale_count_in_the_docs_is_drift(
    derive_status: ModuleType, documented_tree: Path
) -> None:
    tier0 = int(_measured(derive_status, "tier0_metrics"))
    _substitute(
        documented_tree / "docs/user-guide/overview.md",
        rf"\b{tier0} registered Tier 0 metrics",
        f"{tier0 + 1} registered Tier 0 metrics",
    )

    assert _drifted(derive_status, documented_tree) == ["tier0_metrics"]


def test_a_stale_count_wrapped_across_lines_is_drift(
    derive_status: ModuleType, documented_tree: Path
) -> None:
    domains = int(_measured(derive_status, "metric_domains"))
    _substitute(
        documented_tree / "docs/contributing/index.md",
        rf"across\s+{domains}\s+domains",
        f"across {domains + 1}\ndomains",
    )

    assert _drifted(derive_status, documented_tree) == ["metric_domains"]


def test_an_equal_second_copy_is_not_drift(
    derive_status: ModuleType, documented_tree: Path
) -> None:
    tier0 = _measured(derive_status, "tier0_metrics")
    overview = documented_tree / "docs/user-guide/overview.md"
    overview.write_text(overview.read_text() + f"\nIt registers {tier0} Tier 0 metrics.\n")

    assert _drifted(derive_status, documented_tree) == []


def test_a_document_that_drops_its_claim_is_drift(
    derive_status: ModuleType, documented_tree: Path
) -> None:
    tier0 = _measured(derive_status, "tier0_metrics")
    _substitute(
        documented_tree / "docs/user-guide/peer-comparison.md",
        rf"\b{tier0} registered Tier 0 metrics, ",
        "",
    )

    metrics = derive_status.collect_metrics(documented_tree, "calibrax")

    drifted = {metric.label: metric for metric in metrics if metric.is_drifted}
    assert list(drifted) == ["tier0_metrics"]
    assert drifted["tier0_metrics"].missing == ("docs/user-guide/peer-comparison.md",)


def test_the_readme_domain_list_must_match_the_registry(
    derive_status: ModuleType, documented_tree: Path
) -> None:
    _substitute(
        documented_tree / "README.md",
        r"(\*\*Functional domains:\*\* [^\n]+)",
        r"\1, regression",
    )

    assert _drifted(derive_status, documented_tree) == ["domain_names"]


def test_the_domain_table_must_match_the_registry(
    derive_status: ModuleType, documented_tree: Path
) -> None:
    table = documented_tree / "docs/user-guide/metrics-overview.md"
    match = re.search(r"^\| `general` \| (\d+) \|", table.read_text(), flags=re.MULTILINE)
    assert match is not None
    _substitute(table, re.escape(match.group(0)), f"| `general` | {int(match.group(1)) + 1} |")

    assert _drifted(derive_status, documented_tree) == ["domain_counts"]
