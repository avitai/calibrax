"""calibrax.core loads each export on first use, so its light modules import without JAX."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
from substrax.testing import run_python

import calibrax.core


_STUB = Path(calibrax.core.__file__).with_suffix(".pyi")
_NO_JAX_PROBE = (
    "import sys, {module}; "
    "print(sorted(name for name in ('jax', 'flax', 'jaxlib') if name in sys.modules))"
)


def _stub_exports() -> dict[str, str]:
    """Each name the stub re-exports, with the module it comes from."""
    exports: dict[str, str] = {}
    for node in ast.parse(_STUB.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            module = f"calibrax.core.{node.module}" if node.level else node.module
            exports.update({alias.asname or alias.name: module for alias in node.names})
    return exports


@pytest.mark.parametrize(
    "module",
    [
        "calibrax.core.record_values",
        "calibrax.core.models",
        "calibrax.core.registry",
        "calibrax.storage",
        "calibrax.ci",
        "calibrax.validation",
        "calibrax.analysis",
    ],
)
def test_light_modules_import_without_jax(module: str) -> None:
    result = run_python(_NO_JAX_PROBE.format(module=module), timeout=120)

    assert result.stdout.strip() == "[]"


def test_importing_the_package_loads_no_export() -> None:
    result = run_python(_NO_JAX_PROBE.format(module="calibrax.core"), timeout=120)

    assert result.stdout.strip() == "[]"


def test_every_stub_export_resolves_to_its_module() -> None:
    exports = _stub_exports()

    assert sorted(exports) == sorted(calibrax.core.__all__)
    for name, module in exports.items():
        assert getattr(calibrax.core, name) is getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("name", ["NoSuchExport", "MetadataValue"])
def test_a_name_the_stub_does_not_export_is_an_attribute_error(name: str) -> None:
    with pytest.raises(AttributeError):
        getattr(calibrax.core, name)
