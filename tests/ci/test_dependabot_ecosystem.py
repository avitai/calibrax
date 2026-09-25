"""Dependabot must maintain the lock the workflows install from.

Every workflow installs with ``uv sync --locked``, which refuses a lock that does not match
``pyproject.toml``. Dependabot's ``pip`` ecosystem edits ``pyproject.toml`` and leaves
``uv.lock`` alone, so each of its pull requests failed at the install step before a single
check ran. The ``uv`` ecosystem updates both.
"""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEPENDABOT_CONFIG = ROOT / ".github" / "dependabot.yml"
WORKFLOWS = ROOT / ".github" / "workflows"
LOCKED_INSTALL = "--locked"


def _python_ecosystems() -> list[str]:
    config = yaml.safe_load(DEPENDABOT_CONFIG.read_text(encoding="utf-8"))
    return [
        update["package-ecosystem"]
        for update in config["updates"]
        if update["package-ecosystem"] != "github-actions"
    ]


def _workflows_installing_from_the_lock() -> list[str]:
    return sorted(
        path.name
        for path in WORKFLOWS.glob("*.yml")
        if LOCKED_INSTALL in path.read_text(encoding="utf-8")
    )


def test_dependabot_updates_python_through_uv() -> None:
    """The lock is part of the dependency state, so the updater has to own it."""
    assert _python_ecosystems() == ["uv"]


def test_the_workflows_install_from_the_lock() -> None:
    """The premise of the rule above: a stale lock is an install failure, not a warning."""
    assert _workflows_installing_from_the_lock(), "no workflow installs with --locked"
