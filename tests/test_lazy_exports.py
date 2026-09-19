"""Packages that load each export on first use, so their light modules import without JAX."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
from substrax.testing import run_python


_LAZY_PACKAGES = ("calibrax.core", "calibrax.profiling")
_NO_JAX_PROBE = (
    "import sys, {module}; "
    "print(sorted(name for name in ('jax', 'flax', 'jaxlib') if name in sys.modules))"
)


def _stub_exports(package: str) -> dict[str, str]:
    """Each name the package's stub re-exports, with the module it comes from."""
    stub = Path(importlib.import_module(package).__file__ or "").with_suffix(".pyi")
    exports: dict[str, str] = {}
    for node in ast.parse(stub.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            module = f"{package}.{node.module}" if node.level else node.module
            exports.update({alias.asname or alias.name: module for alias in node.names})
    return exports


@pytest.mark.parametrize(
    "module",
    [
        *_LAZY_PACKAGES,
        "calibrax.core.record_values",
        "calibrax.core.models",
        "calibrax.core.registry",
        "calibrax.core.result",
        "calibrax.profiling.timing_records",
        "calibrax.profiling.resources",
        "calibrax.profiling.energy",
        "calibrax.storage",
        "calibrax.ci",
        "calibrax.validation",
        "calibrax.analysis",
    ],
)
def test_light_modules_import_without_jax(module: str) -> None:
    result = run_python(_NO_JAX_PROBE.format(module=module), timeout=120)

    assert result.stdout.strip() == "[]"


@pytest.mark.parametrize("package", _LAZY_PACKAGES)
def test_every_stub_export_resolves_to_its_module(package: str) -> None:
    exports = _stub_exports(package)
    lazy = importlib.import_module(package)

    assert sorted(exports) == sorted(lazy.__all__)
    for name, module in exports.items():
        assert getattr(lazy, name) is getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("package", _LAZY_PACKAGES)
@pytest.mark.parametrize("name", ["NoSuchExport", "MetadataValue"])
def test_a_name_the_stub_does_not_export_is_an_attribute_error(package: str, name: str) -> None:
    with pytest.raises(AttributeError):
        getattr(importlib.import_module(package), name)
