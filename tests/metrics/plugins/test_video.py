"""Tests for optional video metric plugins."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from calibrax.metrics.plugins.video import vmaf_score


def _touch_video_pair(tmp_path: Path) -> tuple[Path, Path]:
    reference = tmp_path / "reference.mp4"
    distorted = tmp_path / "distorted.mp4"
    reference.write_bytes(b"reference")
    distorted.write_bytes(b"distorted")
    return reference, distorted


def _extract_log_path(command: list[str]) -> Path:
    filter_arg = command[command.index("-lavfi") + 1]
    for part in filter_arg.split(":"):
        if part.startswith("log_path="):
            return Path(part.removeprefix("log_path="))
    raise AssertionError(f"missing log_path in command: {command}")


def test_vmaf_score_parses_json_log_file(tmp_path: Path) -> None:
    """VMAF should parse the JSON file produced by FFmpeg/libvmaf."""
    reference, distorted = _touch_video_pair(tmp_path)

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        _extract_log_path(command).write_text('{"pooled_metrics":{"vmaf":{"mean":93.25}}}')
        return subprocess.CompletedProcess(command, 0, "", "")

    with patch("calibrax.metrics.plugins.video.subprocess.run", side_effect=fake_run) as run:
        assert vmaf_score(reference, distorted) == pytest.approx(93.25)

    command = run.call_args.args[0]
    filter_arg = command[command.index("-lavfi") + 1]
    assert "libvmaf" in filter_arg
    assert "log_fmt=json" in filter_arg
    assert "log_path=" in filter_arg


def test_vmaf_score_includes_model_argument(tmp_path: Path) -> None:
    """Optional model values should be passed through to libvmaf."""
    reference, distorted = _touch_video_pair(tmp_path)

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        _extract_log_path(command).write_text('{"pooled_metrics":{"vmaf":{"mean":88.0}}}')
        return subprocess.CompletedProcess(command, 0, "", "")

    with patch("calibrax.metrics.plugins.video.subprocess.run", side_effect=fake_run) as run:
        assert vmaf_score(reference, distorted, model="version=vmaf_v0.6.1") == pytest.approx(88.0)

    filter_arg = run.call_args.args[0][run.call_args.args[0].index("-lavfi") + 1]
    assert "model=version=vmaf_v0.6.1" in filter_arg


def test_vmaf_score_missing_reference_raises(tmp_path: Path) -> None:
    """Missing reference videos should fail before spawning FFmpeg."""
    distorted = tmp_path / "distorted.mp4"
    distorted.write_bytes(b"distorted")

    with pytest.raises(FileNotFoundError):
        vmaf_score(tmp_path / "missing-reference.mp4", distorted)


def test_vmaf_score_missing_distorted_raises(tmp_path: Path) -> None:
    """Missing distorted videos should fail before spawning FFmpeg."""
    reference = tmp_path / "reference.mp4"
    reference.write_bytes(b"reference")

    with pytest.raises(FileNotFoundError):
        vmaf_score(reference, tmp_path / "missing-distorted.mp4")


def test_vmaf_score_subprocess_failure_raises_runtime_error(tmp_path: Path) -> None:
    """FFmpeg failures should become clear runtime errors."""
    reference, distorted = _touch_video_pair(tmp_path)
    error = subprocess.CalledProcessError(1, ["ffmpeg"], stderr="libvmaf not found")

    with (
        patch("calibrax.metrics.plugins.video.subprocess.run", side_effect=error),
        pytest.raises(RuntimeError, match="FFmpeg/libvmaf failed"),
    ):
        vmaf_score(reference, distorted)


def test_vmaf_score_malformed_json_raises_value_error(tmp_path: Path) -> None:
    """Malformed libvmaf logs should raise ValueError."""
    reference, distorted = _touch_video_pair(tmp_path)

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        _extract_log_path(command).write_text("{not-json")
        return subprocess.CompletedProcess(command, 0, "", "")

    with (
        patch("calibrax.metrics.plugins.video.subprocess.run", side_effect=fake_run),
        pytest.raises(ValueError, match="Invalid VMAF JSON"),
    ):
        vmaf_score(reference, distorted)


def test_vmaf_score_missing_mean_raises_value_error(tmp_path: Path) -> None:
    """Missing VMAF mean values should raise ValueError."""
    reference, distorted = _touch_video_pair(tmp_path)

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        _extract_log_path(command).write_text('{"pooled_metrics":{"psnr":{"mean":40.0}}}')
        return subprocess.CompletedProcess(command, 0, "", "")

    with (
        patch("calibrax.metrics.plugins.video.subprocess.run", side_effect=fake_run),
        pytest.raises(ValueError, match="VMAF mean"),
    ):
        vmaf_score(reference, distorted)
