"""CI fails when coverage drops below the floor, and uploads coverage nowhere."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
CAP_OVERRIDES = ("--no-cov", "-o addopts", "--override-ini")


def _load_workflow(path: Path) -> dict:
    # BaseLoader keeps every scalar a string, so the `on:` key stays `on` rather than True.
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)  # noqa: S506


def coverage_cap_violations(workflow: dict, pyproject: dict) -> list[str]:
    """Return why the test job would not fail below the coverage floor, if it would not."""
    job = workflow["jobs"]["test"]
    command = next(step["run"] for step in job["steps"] if "pytest" in step.get("run", ""))
    addopts = pyproject["tool"]["pytest"]["ini_options"].get("addopts", "")
    addopts = " ".join(addopts) if isinstance(addopts, list) else addopts
    # A cap on the command line overrides the one in addopts.
    caps = re.findall(r"--cov-fail-under[= ](\d+)", command) or re.findall(
        r"--cov-fail-under[= ](\d+)", addopts
    )

    problems = []
    if not caps or min(int(cap) for cap in caps) < 80:
        problems.append(f"the pytest coverage floor is {caps}, not at least 80")
    if not {"push", "pull_request"} <= set(workflow["on"]):
        problems.append(f"CI runs on {sorted(workflow['on'])}, not on both push and pull_request")
    if "if" in job:
        problems.append(f"the test job only runs when {job['if']}")
    problems += [
        f"the pytest command overrides the floor with {override}"
        for override in CAP_OVERRIDES
        if override in command
    ]
    return problems


def test_ci_test_job_fails_below_the_coverage_floor() -> None:
    """The test job runs pytest with a floor of at least 80% on every push and pull request."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert coverage_cap_violations(_load_workflow(WORKFLOWS / "ci.yml"), pyproject) == []


def test_ci_uploads_no_coverage_to_codecov() -> None:
    """coverage.py in the test job is the coverage gate; no workflow uploads to Codecov."""
    uses = [
        step.get("uses", "")
        for path in sorted(WORKFLOWS.glob("*.yml"))
        for job in _load_workflow(path).get("jobs", {}).values()
        for step in job.get("steps", [])
    ]

    assert any(action.startswith("actions/checkout@") for action in uses)
    assert [action for action in uses if action.startswith("codecov/")] == []
