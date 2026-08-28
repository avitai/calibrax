"""Policy tests for the PyPI publishing workflow."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"
RELEASING_DOC = ROOT / "RELEASING.md"


def _load_publish_workflow() -> dict[str, object]:
    return yaml.load(PUBLISH_WORKFLOW.read_text(), Loader=yaml.BaseLoader)


def test_publish_workflow_does_not_create_releases_on_push() -> None:
    """No commit or tag push should create a GitHub release by itself."""
    workflow = _load_publish_workflow()

    assert "push" not in workflow["on"]  # type: ignore[operator]


def test_manual_release_target_generates_release_notes() -> None:
    """Only an explicit manual release target should generate release notes."""
    workflow = _load_publish_workflow()

    workflow_inputs = workflow["on"]["workflow_dispatch"]["inputs"]  # type: ignore[index]
    target_options = workflow_inputs["target"]["options"]
    release_job = workflow["jobs"]["create-github-release"]  # type: ignore[index]
    release_step = release_job["steps"][-1]

    assert "github-release" in target_options
    assert "version_tag" in workflow_inputs
    assert release_job["if"] == (
        "github.event_name == 'workflow_dispatch' && github.event.inputs.target == 'github-release'"
    )
    assert release_job["permissions"]["contents"] == "write"
    # Pinned to a major version on purpose: this is the action that publishes releases,
    # so a major bump should be a deliberate edit here rather than something that rides
    # in unnoticed. Updated to v3 with the dependabot bump that moved the workflow.
    assert release_step["uses"] == "softprops/action-gh-release@v3"
    assert release_step["with"]["generate_release_notes"] == "true"
    assert release_step["with"]["tag_name"] == "${{ github.event.inputs.version_tag }}"


def test_publish_pypi_waits_for_manual_release_creation() -> None:
    """Manual generated-release publishing must wait for release creation."""
    workflow = _load_publish_workflow()

    publish_job = workflow["jobs"]["publish-pypi"]  # type: ignore[index]
    condition = publish_job["if"]

    assert publish_job["needs"] == ["build", "create-github-release"]
    assert "github.event.inputs.target == 'github-release'" in condition
    assert "needs.create-github-release.result == 'success'" in condition


def test_releasing_docs_describe_generated_release_note_automation() -> None:
    """Human release docs should match the automated tag-push release path."""
    releasing = RELEASING_DOC.read_text()

    assert "No commit or tag push creates a release by itself" in releasing
    assert "softprops/action-gh-release@v3" in releasing
    assert "generate_release_notes: true" in releasing
    assert "target=github-release" in releasing
