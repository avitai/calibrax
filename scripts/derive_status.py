#!/usr/bin/env python3
"""Derive a repo's status from its source tree and check it against the docs.

Copy this template to ``scripts/derive_status.py`` in a sibling Avitai repo
and edit only the CONFIG block below. Two run modes::

    uv run python scripts/derive_status.py            # print the derived table
    uv run python scripts/derive_status.py --check     # exit 1 on any drift (CI)

Rationale: README / INDEX / badge claims rot as the code evolves. This makes
the claims falsifiable — it walks ``src/`` plus manifests, emits the numbers
the docs assert, and fails loudly when an asserted value disagrees with the
measured one. Wire ``--check`` into CI so drift can never land silently.

See ``audits/UNIFIED-AUDIT-2026-06-01.md`` (Theme 3) in the avitai-portfolio
repo for the cross-cutting rationale this template implements.

This file ends in ``.tmpl`` because the CONFIG block carries
``<placeholders>`` that must be substituted per repo.
"""

from __future__ import annotations

import argparse
import functools
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
VENDOR_PARTS = frozenset({".venv", "site-packages", "node_modules", ".git", "test_venv"})

# === CONFIG — the only block to edit when stamping this template ============
PACKAGE_NAME = "calibrax"  # directory under src/, e.g. "datarax"
README_PATH = REPO_ROOT / "README.md"
# label -> regex with ONE capture group pulling the asserted value out of the
# README. Drop entries you do not assert; the derived value is still printed.
ASSERTIONS: dict[str, str] = {
    # The README's metrics headline: "(<n> registered Tier 0 metrics, <m> domains, ...)".
    "tier0_metrics": r"\((\d+) registered Tier 0 metrics, \d+ domains",
    "metric_domains": r"\(\d+ registered Tier 0 metrics, (\d+) domains",
}
# ===========================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class Metric:
    """A derived metric paired with whatever the docs assert for it."""

    label: str
    measured: str
    asserted: str | None

    @property
    def is_drifted(self) -> bool:
        """Whether an asserted value exists and disagrees with measurement."""
        return self.asserted is not None and self.asserted != self.measured


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
    "from calibrax.metrics import MetricRegistry, MetricTier\n"
    "registry = MetricRegistry()\n"
    "names = registry.list_names()\n"
    "print(len(registry.list_by_tier(MetricTier.PURE_FUNCTION)))\n"
    "print(len({registry.get(name).domain for name in names}))\n"
)


@functools.cache
def _registry_facts() -> tuple[str, str]:
    """Count the Tier 0 metrics and their domains in a fresh interpreter.

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
    tier0, domains = result.stdout.split()
    return tier0, domains


def measure_tier0_metrics(_root: Path, _package: str) -> str:
    """Count the Tier 0 (pure-function) metrics the registry holds after import."""
    return _registry_facts()[0]


def measure_metric_domains(_root: Path, _package: str) -> str:
    """Count the distinct domains of the registered metrics."""
    return _registry_facts()[1]


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


# label -> measure function. Extend per repo as the README asserts more.
MEASUREMENTS: dict[str, Callable[[Path, str], str]] = {
    "version": measure_version,
    "tests": measure_tests,
    "modules": measure_modules,
    "subpackages": measure_subpackages,
    "tier0_metrics": measure_tier0_metrics,
    "metric_domains": measure_metric_domains,
    "todos": measure_todos,
}


def _asserted_value(label: str, readme_text: str) -> str | None:
    """Extract the asserted value for ``label`` from the README, if declared."""
    pattern = ASSERTIONS.get(label)
    if pattern is None:
        return None
    matches = re.findall(pattern, readme_text, flags=re.MULTILINE)
    return ",".join(sorted(matches)) if matches else None


def collect_metrics(root: Path, package: str, readme: Path = README_PATH) -> list[Metric]:
    """Run every measurement and pair it with its asserted value (if any)."""
    readme_text = readme.read_text() if readme.is_file() else ""
    return [
        Metric(
            label=label,
            measured=measure(root, package),
            asserted=_asserted_value(label, readme_text),
        )
        for label, measure in MEASUREMENTS.items()
    ]


def render_table(metrics: list[Metric]) -> str:
    """Format the metrics as a fixed-width table for human reading."""
    header = f"{'metric':<12} {'measured':<16} {'asserted':<16} drift"
    rows = [
        f"{m.label:<12} {m.measured:<16} {(m.asserted or '—'):<16} "
        f"{'DRIFT' if m.is_drifted else 'ok'}"
        for m in metrics
    ]
    return "\n".join([header, "-" * len(header), *rows])


def main() -> int:
    """Print the derived table; with ``--check``, exit non-zero on drift."""
    parser = argparse.ArgumentParser(description="Derive and verify repo status.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any asserted value disagrees with measurement",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    metrics = collect_metrics(REPO_ROOT, PACKAGE_NAME)
    print(render_table(metrics))

    drifted = [m for m in metrics if m.is_drifted]
    if args.check and drifted:
        for metric in drifted:
            logger.error(
                "drift: %s asserted=%s measured=%s",
                metric.label,
                metric.asserted,
                metric.measured,
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
