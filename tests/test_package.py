"""The package's version has one source, ``pyproject.toml``, which the installed package reports."""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

import calibrax


_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _declared_version() -> str:
    return tomllib.loads(_PYPROJECT.read_text())["project"]["version"]


def test_the_version_is_the_one_pyproject_declares() -> None:
    assert calibrax.__version__ == _declared_version()


def test_the_version_is_the_installed_distributions() -> None:
    assert calibrax.__version__ == importlib.metadata.version("calibrax")


def test_the_version_is_a_release_string() -> None:
    """MAJOR.MINOR.PATCH, the form the release tags ``vX.Y.Z`` and PyPI use."""
    parts = calibrax.__version__.split(".")

    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
