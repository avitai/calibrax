#!/usr/bin/env python3
"""Derive calibrax's documented status from the package and check every copy against it.

Two run modes::

    uv run python scripts/derive_status.py            # print the derived table
    uv run python scripts/derive_status.py --check     # exit 1 on any drift (CI)

The README and the documentation repeat the metric registry's size in many places.
Each claim is measured from a fresh interpreter and compared with every copy listed
in ``README_CLAIMS`` and ``DOCUMENT_SOURCES``, so a metric or domain added to the
registry fails CI until every document reflects it. A listed document that no longer
carries one of its claims is drift as well: a claim that cannot be found cannot be
checked. Documents are read with whitespace collapsed, so a sentence wrapped across
lines is still found.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger("derive_status")

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_NAME = "calibrax"
VENDOR_PARTS = frozenset({".venv", "site-packages", "node_modules", ".git", "test_venv"})


@dataclass(frozen=True, slots=True, kw_only=True)
class Metric:
    """A derived value paired with what the documents assert for it."""

    label: str
    measured: str
    asserted: str | None
    missing: tuple[str, ...] = ()

    @property
    def is_drifted(self) -> bool:
        """Whether a document lost its claim or asserts a value measurement disagrees with."""
        if self.missing:
            return True
        return self.asserted is not None and self.asserted != self.measured


@dataclass(frozen=True, slots=True, kw_only=True)
class Claim:
    """How one kind of documented claim is read out of a document."""

    patterns: tuple[str, ...]
    reader: Callable[[str, tuple[str, ...]], str | None]


def _flatten(text: str) -> str:
    """Collapse every run of whitespace to one space, joining wrapped lines."""
    return " ".join(text.split())


def _read_counts(text: str, patterns: tuple[str, ...]) -> str | None:
    """Return every distinct number the patterns capture, comma-joined in numeric order."""
    flat = _flatten(text)
    values = {value for pattern in patterns for value in re.findall(pattern, flat)}
    return ",".join(sorted(values, key=int)) if values else None


def _read_name_list(text: str, patterns: tuple[str, ...]) -> str | None:
    """Return the first comma-separated name list the patterns capture, sorted."""
    flat = _flatten(text)
    for pattern in patterns:
        match = re.search(pattern, flat)
        if match:
            return ",".join(sorted(name.strip() for name in match.group(1).split(",")))
    return None


def _read_table_rows(text: str, patterns: tuple[str, ...]) -> str | None:
    """Return every ``name=count`` row the patterns capture, sorted by name."""
    rows = [row for pattern in patterns for row in re.findall(pattern, text, flags=re.MULTILINE)]
    return ",".join(sorted(f"{name}={count}" for name, count in rows)) if rows else None


CLAIMS: dict[str, Claim] = {
    "tier0_metrics": Claim(
        patterns=(
            r"\b(\d+) (?:registered )?Tier 0 (?:pure-function metrics|pure functions|metrics)",
            r"\| (\d+) registered \|",
        ),
        reader=_read_counts,
    ),
    "metric_domains": Claim(
        patterns=(r"\b(\d+) (?:functional )?domains\b",),
        reader=_read_counts,
    ),
    "domain_names": Claim(
        patterns=(r"\*\*Functional domains:\*\* ([a-z_]+(?:, [a-z_]+)*)",),
        reader=_read_name_list,
    ),
    "domain_counts": Claim(
        patterns=(r"^\| `([a-z_]+)` \| (\d+) \|",),
        reader=_read_table_rows,
    ),
}

README_CLAIMS: tuple[str, ...] = ("tier0_metrics", "metric_domains", "domain_names")

# Repository-relative document -> the claims it must carry.
DOCUMENT_SOURCES: dict[str, tuple[str, ...]] = {
    "docs/architecture/module-map.md": ("tier0_metrics", "metric_domains"),
    "docs/contributing/adding-a-metric.md": ("tier0_metrics", "metric_domains"),
    "docs/contributing/example_documentation_design.md": ("tier0_metrics", "metric_domains"),
    "docs/contributing/index.md": ("tier0_metrics", "metric_domains"),
    "docs/user-guide/index.md": ("tier0_metrics", "metric_domains"),
    "docs/user-guide/metrics-overview.md": ("tier0_metrics", "metric_domains", "domain_counts"),
    "docs/user-guide/overview.md": ("tier0_metrics", "metric_domains"),
    "docs/user-guide/peer-comparison.md": ("tier0_metrics",),
}


def _is_vendored(path: Path) -> bool:
    """Whether any path component is a vendored / virtual-env directory."""
    return any(part in VENDOR_PARTS for part in path.parts)


def _count(root: Path, pattern: str) -> int:
    """Count files matching ``pattern`` under ``root``, skipping vendored trees."""
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob(pattern) if not _is_vendored(path))


def measure_version(root: Path, package: str) -> str:
    """Read the version from ``src/<package>/__init__.py`` or ``pyproject.toml``."""
    init = root / "src" / package / "__init__.py"
    if init.is_file():
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)', init.read_text())
        if match:
            return match.group(1)
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        version = tomllib.loads(pyproject.read_text()).get("project", {}).get("version")
        if version is not None:
            return str(version)
    return "unknown"


def measure_tests(root: Path, _package: str) -> str:
    """Count ``test_*.py`` files under ``tests/``."""
    return str(_count(root / "tests", "test_*.py"))


def measure_modules(root: Path, package: str) -> str:
    """Count ``*.py`` source modules under ``src/<package>/``."""
    return str(_count(root / "src" / package, "*.py"))


_REGISTRY_PROBE = (
    "import collections, json\n"
    "from calibrax.metrics import MetricRegistry, MetricTier\n"
    "entries = MetricRegistry().list_by_tier(MetricTier.PURE_FUNCTION)\n"
    "print(json.dumps(collections.Counter(entry.domain for entry in entries)))\n"
)


@functools.cache
def _tier0_domain_counts() -> tuple[tuple[str, int], ...]:
    """Count the Tier 0 metrics per domain in a fresh interpreter.

    The registry is a process-wide singleton that tests register into, so an
    in-process count would depend on what ran before it; a subprocess measures the
    package as shipped.
    """
    result = subprocess.run(  # noqa: S603  # a fixed probe under the running interpreter
        [sys.executable, "-c", _REGISTRY_PROBE],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    counts: dict[str, int] = json.loads(result.stdout)
    return tuple(sorted(counts.items()))


def measure_tier0_metrics(_root: Path, _package: str) -> str:
    """Count the Tier 0 (pure-function) metrics the registry holds after import."""
    return str(sum(count for _, count in _tier0_domain_counts()))


def measure_metric_domains(_root: Path, _package: str) -> str:
    """Count the distinct domains of the Tier 0 metrics."""
    return str(len(_tier0_domain_counts()))


def measure_domain_names(_root: Path, _package: str) -> str:
    """List the Tier 0 domains, comma-joined and sorted."""
    return ",".join(domain for domain, _ in _tier0_domain_counts())


def measure_domain_counts(_root: Path, _package: str) -> str:
    """List ``domain=count`` for every Tier 0 domain, sorted by domain."""
    return ",".join(f"{domain}={count}" for domain, count in _tier0_domain_counts())


def measure_subpackages(root: Path, package: str) -> str:
    """List the importable subpackages under ``src/<package>/``, comma-joined and sorted."""
    source = root / "src" / package
    if not source.exists():
        return ""
    names = sorted(
        path.name for path in source.iterdir() if path.is_dir() and (path / "__init__.py").is_file()
    )
    return ",".join(names)


def measure_todos(root: Path, package: str) -> str:
    """Count TODO / FIXME / XXX / HACK markers under ``src/<package>/``."""
    marker = re.compile(r"TODO|FIXME|XXX|HACK")
    source = root / "src" / package
    if not source.exists():
        return "0"
    total = sum(
        len(marker.findall(path.read_text(errors="ignore")))
        for path in source.rglob("*.py")
        if not _is_vendored(path)
    )
    return str(total)


MEASUREMENTS: dict[str, Callable[[Path, str], str]] = {
    "version": measure_version,
    "tests": measure_tests,
    "modules": measure_modules,
    "subpackages": measure_subpackages,
    "tier0_metrics": measure_tier0_metrics,
    "metric_domains": measure_metric_domains,
    "domain_names": measure_domain_names,
    "domain_counts": measure_domain_counts,
    "todos": measure_todos,
}


def _documents(root: Path, readme: Path) -> dict[str, tuple[Path, tuple[str, ...]]]:
    """Map each checked document's display name to its path and required claims."""
    documents = {"README.md": (readme, README_CLAIMS)}
    for relative, labels in DOCUMENT_SOURCES.items():
        documents[relative] = (root / relative, labels)
    return documents


def _asserted(
    label: str, documents: dict[str, tuple[Path, tuple[str, ...]]]
) -> tuple[str | None, tuple[str, ...]]:
    """Read ``label`` from every document that must carry it.

    Returns:
        The asserted value (distinct per-document readings joined with `` | ``, so any
        disagreement differs from the measurement) and the documents that lack it.
    """
    claim = CLAIMS.get(label)
    if claim is None:
        return None, ()
    readings: set[str] = set()
    missing: list[str] = []
    for name, (path, labels) in documents.items():
        if label not in labels:
            continue
        reading = claim.reader(path.read_text(), claim.patterns) if path.is_file() else None
        if reading is None:
            missing.append(name)
        else:
            readings.add(reading)
    asserted = " | ".join(sorted(readings)) if readings else None
    return asserted, tuple(missing)


def collect_metrics(root: Path, package: str, readme: Path | None = None) -> list[Metric]:
    """Run every measurement and pair it with what the documents under ``root`` assert."""
    documents = _documents(root, readme if readme is not None else root / "README.md")
    metrics: list[Metric] = []
    for label, measure in MEASUREMENTS.items():
        asserted, missing = _asserted(label, documents)
        metrics.append(
            Metric(label=label, measured=measure(root, package), asserted=asserted, missing=missing)
        )
    return metrics


def render_table(metrics: list[Metric]) -> str:
    """Format the metrics as a fixed-width table for human reading."""
    header = f"{'metric':<15} {'drift':<6} measured / asserted"
    rows = []
    for metric in metrics:
        status = "DRIFT" if metric.is_drifted else "ok"
        rows.append(f"{metric.label:<15} {status:<6} {metric.measured}")
        if metric.asserted is not None and metric.asserted != metric.measured:
            rows.append(f"{'':<22} asserted: {metric.asserted}")
        if metric.missing:
            rows.append(f"{'':<22} missing from: {', '.join(metric.missing)}")
    return "\n".join([header, "-" * len(header), *rows])


def main() -> int:
    """Print the derived table; with ``--check``, exit non-zero on drift."""
    parser = argparse.ArgumentParser(description="Derive and verify calibrax's documented status.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any document's claim is missing or disagrees with measurement",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    metrics = collect_metrics(REPO_ROOT, PACKAGE_NAME)
    print(render_table(metrics))

    drifted = [metric for metric in metrics if metric.is_drifted]
    if args.check and drifted:
        for metric in drifted:
            logger.error(
                "drift: %s asserted=%s measured=%s missing=%s",
                metric.label,
                metric.asserted,
                metric.measured,
                ",".join(metric.missing) or "-",
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
