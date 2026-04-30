"""Video quality metrics backed by optional external tools."""

from __future__ import annotations

import json
import subprocess  # nosec B404
import tempfile
from pathlib import Path


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

    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = Path(temp_dir) / "vmaf.json"
        filter_args = f"libvmaf=log_fmt=json:log_path={log_path}"
        if model is not None:
            filter_args += f":model={model}"

        command = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(distorted_path),
            "-i",
            str(reference_path),
            "-lavfi",
            filter_args,
            "-f",
            "null",
            "-",
        ]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)  # nosec B603
        except FileNotFoundError as e:
            msg = "FFmpeg executable not found; install FFmpeg with libvmaf support"
            raise RuntimeError(msg) from e
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
