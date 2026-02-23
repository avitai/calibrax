"""Policy checks for installation-command consistency."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
POLICY_PATHS = (
    REPO_ROOT / "src",
    REPO_ROOT / "docs",
    REPO_ROOT / "README.md",
)
TEXT_SUFFIXES = {".md", ".py", ".rst", ".toml", ".txt", ".yaml", ".yml"}


def _iter_policy_files() -> list[Path]:
    files: list[Path] = []
    for target in POLICY_PATHS:
        if target.is_file():
            files.append(target)
            continue

        for path in target.rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                files.append(path)
    return sorted(files)


def test_no_bare_pip_install_guidance() -> None:
    """User-facing install guidance should consistently use uv commands."""
    violations: list[str] = []
    for path in _iter_policy_files():
        rel_path = path.relative_to(REPO_ROOT)
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if "pip install" in line and "uv pip install" not in line:
                violations.append(f"{rel_path}:{lineno}: {line.strip()}")

    assert not violations, "Found non-uv install guidance:\n" + "\n".join(violations)
