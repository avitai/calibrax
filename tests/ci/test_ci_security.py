"""The Security workflow audits every extra calibrax's lock resolves, not one environment.

pip-audit run on an environment audits only the extras installed there: the job synced the dev
extra, so advisories in the test, docs, mlflow and wandb extras never reached it. The audit is
substrax's ``audit-lock`` action, pinned by commit: it exports every extra the lock resolves and
audits each export with a pinned pip-audit and a fresh advisory cache. Its ignore table lives in
``pyproject.toml`` as ``[tool.substrax.audit-lock.ignore]``; the action refuses an empty reason and
an entry no advisory matches, so those rules are tested in substrax, not here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
AUDIT_ACTION = re.compile(r"^avitai/substrax/\.github/actions/audit-lock@[0-9a-f]{40}$")


def _load(path: Path) -> dict[str, Any]:
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)  # noqa: S506  # BaseLoader keeps `on:` a string


def _security_steps() -> list[dict[str, Any]]:
    return _load(WORKFLOWS / "security.yml")["jobs"]["security"]["steps"]


def test_the_audit_is_the_shared_action_pinned_to_a_commit() -> None:
    audits = [step for step in _security_steps() if "audit-lock" in step.get("uses", "")]

    assert len(audits) == 1
    assert AUDIT_ACTION.match(audits[0]["uses"]), audits[0]["uses"]
    assert "continue-on-error" not in audits[0]


def test_uv_is_on_path_before_the_audit_runs() -> None:
    """The action runs ``uv export`` and ``uvx``."""
    uses = [step.get("uses", "") for step in _security_steps()]
    setup = next(i for i, use in enumerate(uses) if use.startswith("astral-sh/setup-uv@"))
    audit = next(i for i, use in enumerate(uses) if AUDIT_ACTION.match(use))

    assert setup < audit


def test_the_audit_runs_on_pull_requests_pushes_to_main_and_the_weekly_schedule() -> None:
    assert {"push", "pull_request", "schedule"} <= set(_load(WORKFLOWS / "security.yml")["on"])


def test_no_workflow_runs_pip_audit_on_an_environment() -> None:
    runs = [
        f"{path.name}:{name}"
        for path in sorted(WORKFLOWS.glob("*.yml"))
        for name, job in _load(path)["jobs"].items()
        for step in job.get("steps", [])
        if "pip-audit" in step.get("run", "") or "pip_audit" in step.get("run", "")
    ]

    assert runs == []


def test_bandit_runs_whatever_the_audit_found_and_its_failure_fails_the_job() -> None:
    """A failed audit skipped Bandit, so the job reported one scan's verdict, not two."""
    bandit = next(step for step in _security_steps() if "bandit " in step.get("run", ""))

    assert bandit.get("if") == "${{ !cancelled() }}"
    assert "continue-on-error" not in bandit
    assert "||" not in bandit["run"], "a captured exit status can hide bandit's failure"
