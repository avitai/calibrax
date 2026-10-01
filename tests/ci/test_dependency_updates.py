"""Dependency updates arrive through pull requests the maintainers open, never from a bot.

A version-update bot opens pull requests against ``main`` on its own schedule. None is configured
in this repository. Dependabot alerts stay on: they report advisories and open nothing.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_no_dependabot_version_updates_are_configured() -> None:
    """No Dependabot configuration exists, so Dependabot opens no pull request."""
    assert sorted(path.name for path in (ROOT / ".github").glob("dependabot.y*ml")) == []
