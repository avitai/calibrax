"""Tests for optional video metric plugins."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from calibrax.metrics.plugins.video import vmaf_score


@pytest.fixture(autouse=True)
def ffmpeg_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """subprocess.run is mocked in these tests; the executable lookup finds a fixed path."""
    monkeypatch.setattr(
        "calibrax.metrics.plugins.video.shutil.which", lambda _name: "/usr/bin/ffmpeg"
    )


def _touch_video_pair(tmp_path: Path) -> tuple[Path, Path]:
    reference = tmp_path / "reference.mp4"
    distorted = tmp_path / "distorted.mp4"
    reference.write_bytes(b"reference")
    distorted.write_bytes(b"distorted")
    return reference, distorted


def _split_unescaped(text: str, separators: str) -> list[str]:
    """Split on unescaped separators, keeping escapes for the next level (FFmpeg's rule)."""
    parts, current, index = [], "", 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            current += text[index : index + 2]
            index += 2
            continue
        if char in separators:
            parts.append(current)
            current = ""
        else:
            current += char
        index += 1
    parts.append(current)
    return parts


def _unescape(text: str) -> str:
    out, index = "", 0
    while index < len(text):
        if text[index] == "\\" and index + 1 < len(text):
            index += 1
        out += text[index]
        index += 1
    return out


def _libvmaf_options(command: list[str]) -> dict[str, str]:
    """The libvmaf options FFmpeg reads from -lavfi: filtergraph level, then option level.

    "Notes on filtergraph escaping" (ffmpeg-filters): ',' and ';' separate filters and '\\'
    escapes at the filtergraph level; ':' separates options and '\\' escapes inside a value.
    """
    graph = command[command.index("-lavfi") + 1]
    filters = _split_unescaped(graph, ",;")
    assert len(filters) == 1, f"the filtergraph holds {len(filters)} filters: {graph}"
    name, _, arguments = _unescape(filters[0]).partition("=")
    assert name == "libvmaf"
    options = {}
    for option in _split_unescaped(arguments, ":"):
        key, _, value = option.partition("=")
        options[key] = _unescape(value)
    return options


def _extract_log_path(command: list[str]) -> Path:
    return Path(_libvmaf_options(command)["log_path"])


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


def test_a_model_string_stays_one_option_value(tmp_path: Path) -> None:
    """Separators in the model are escaped, so they cannot add options or filters."""
    reference, distorted = _touch_video_pair(tmp_path)
    hostile = "version=vmaf_v0.6.1:name=x,movie=/etc/passwd;[out]'q"

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        _extract_log_path(command).write_text('{"pooled_metrics":{"vmaf":{"mean":80.0}}}')
        return subprocess.CompletedProcess(command, 0, "", "")

    with patch("calibrax.metrics.plugins.video.subprocess.run", side_effect=fake_run) as run:
        vmaf_score(reference, distorted, model=hostile)

    options = _libvmaf_options(run.call_args.args[0])
    assert set(options) == {"log_fmt", "log_path", "model"}
    assert options["model"] == hostile


def test_inputs_are_absolute_file_urls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``file:`` keeps a name holding ':' from being read as another FFmpeg protocol."""
    reference, distorted = _touch_video_pair(tmp_path)
    monkeypatch.chdir(tmp_path)

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        _extract_log_path(command).write_text('{"pooled_metrics":{"vmaf":{"mean":80.0}}}')
        return subprocess.CompletedProcess(command, 0, "", "")

    with patch("calibrax.metrics.plugins.video.subprocess.run", side_effect=fake_run) as run:
        vmaf_score(Path(reference.name), Path(distorted.name))

    command = run.call_args.args[0]
    inputs = [command[i + 1] for i, arg in enumerate(command) if arg == "-i"]
    assert inputs == [f"file:{distorted.resolve()}", f"file:{reference.resolve()}"]


def test_a_missing_ffmpeg_raises_runtime_error(tmp_path: Path) -> None:
    reference, distorted = _touch_video_pair(tmp_path)

    with (
        patch("calibrax.metrics.plugins.video.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="FFmpeg"),
    ):
        vmaf_score(reference, distorted)
