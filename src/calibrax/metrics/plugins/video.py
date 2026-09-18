"""Video quality metrics backed by optional external tools."""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404
import tempfile
from pathlib import Path


def _escape_option_value(value: str) -> str:
    r"""Escape a filter option value: ``\``, ``'`` and ``:`` (FFmpeg's first escaping level)."""
    return "".join(f"\\{char}" if char in "\\':" else char for char in value)


def _escape_filtergraph(description: str) -> str:
    r"""Escape a filter description: ``\``, ``'``, ``[``, ``]``, ``,`` and ``;`` (second level).

    FFmpeg reads a filtergraph in two passes ("Notes on filtergraph escaping", ffmpeg-filters):
    the graph splits filters on ``,`` and ``;``, then each filter splits its options on ``:``.
    Escaping a value at both levels keeps every character of it inside that one value.
    """
    return "".join(f"\\{char}" if char in "\\'[],;" else char for char in description)


def vmaf_score(reference: str | Path, distorted: str | Path, *, model: str | None = None) -> float:
    """Compute VMAF using FFmpeg with libvmaf JSON logging.

    Args:
        reference: Reference video path.
        distorted: Distorted video path.
        model: Optional libvmaf model expression, such as
            ``"version=vmaf_v0.6.1"``.

    Returns:
        Mean pooled VMAF score. Higher is better.

    Raises:
        FileNotFoundError: If either video path does not exist.
        RuntimeError: If FFmpeg/libvmaf cannot run successfully.
        ValueError: If FFmpeg does not produce a valid VMAF JSON log.
    """
    reference_path = Path(reference)
    distorted_path = Path(distorted)
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    if not distorted_path.exists():
        raise FileNotFoundError(distorted_path)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        msg = "FFmpeg executable not found; install FFmpeg with libvmaf support"
        raise RuntimeError(msg)

    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = Path(temp_dir) / "vmaf.json"
        options = {"log_fmt": "json", "log_path": str(log_path)}
        if model is not None:
            options["model"] = model
        filter_graph = _escape_filtergraph(
            "libvmaf=" + ":".join(f"{key}={_escape_option_value(v)}" for key, v in options.items())
        )
        # The file protocol keeps a name holding ':' from being read as another protocol.
        command = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-i",
            f"file:{distorted_path.resolve()}",
            "-i",
            f"file:{reference_path.resolve()}",
            "-lavfi",
            filter_graph,
            "-f",
            "null",
            "-",
        ]

        try:
            subprocess.run(  # noqa: S603  # nosec B603  # resolved ffmpeg, escaped filter, file: inputs
                command, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            msg = "FFmpeg/libvmaf failed while computing VMAF"
            raise RuntimeError(msg) from e

        try:
            payload = json.loads(log_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            msg = "Invalid VMAF JSON log produced by FFmpeg/libvmaf"
            raise ValueError(msg) from e

    try:
        return float(payload["pooled_metrics"]["vmaf"]["mean"])
    except (KeyError, TypeError, ValueError) as e:
        msg = "VMAF mean missing from FFmpeg/libvmaf JSON log"
        raise ValueError(msg) from e


__all__ = ["vmaf_score"]
