"""The profile and profile-gpu commands, run end to end on a small JAX function."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from calibrax.cli.main import main
from tests.factories import make_fake_nvml


_USER_MODULE = """
import jax.numpy as jnp


def step():
    return jnp.ones((8, 8)) @ jnp.ones((8, 8))


not_callable = 3
"""


@pytest.fixture
def user_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A module on ``sys.path`` defining a JAX workload, as a user's benchmark would."""
    (tmp_path / "profiled_workload.py").write_text(_USER_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    return "profiled_workload"


def _profile(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(main, list(args))
    return result.exit_code, result.output


class TestProfileCommand:
    """Tests for the 'profile' CLI command."""

    def test_nonexistent_module_shows_error(self) -> None:
        code, output = _profile("profile", "--module", "nonexistent_module_xyz", "--function", "f")

        assert code != 0
        assert "Cannot import module" in output

    def test_nonexistent_function_shows_error(self, user_module: str) -> None:
        code, output = _profile("profile", "--module", user_module, "--function", "missing")

        assert code != 0
        assert "not found" in output

    def test_an_attribute_that_is_not_callable_shows_error(self, user_module: str) -> None:
        code, output = _profile("profile", "--module", user_module, "--function", "not_callable")

        assert code != 0
        assert "is not callable" in output

    def test_missing_module_option(self) -> None:
        code, output = _profile("profile", "--function", "foo")

        assert code != 0
        assert "module" in output.lower()

    def test_missing_function_option(self) -> None:
        code, output = _profile("profile", "--module", "math")

        assert code != 0
        assert "function" in output.lower()

    def test_timing_results_are_printed(self, user_module: str) -> None:
        code, output = _profile(
            "profile", "--module", user_module, "--function", "step", "--iterations", "3"
        )

        assert code == 0, output
        assert f"Profiling {user_module}.step" in output
        assert "Batches: 4 (warmup excluded: 1)" in output
        assert "Mean batch time" in output
        assert "Profile complete" in output

    def test_flops_are_counted(self, user_module: str) -> None:
        code, output = _profile("profile", "--module", user_module, "--function", "step", "--flops")

        assert code == 0, output
        assert "FLOP Analysis" in output
        assert "Total FLOPs: 1,024" in output  # 2 * 8 * 8 * 8

    def test_energy_results_are_printed(self, user_module: str) -> None:
        code, output = _profile(
            "profile", "--module", user_module, "--function", "step", "--energy"
        )

        assert code == 0, output
        assert "Energy Results" in output
        assert "GPU energy" not in output  # profile measures the CPU only

    def test_the_run_is_saved_to_the_store(self, user_module: str, tmp_path: Path) -> None:
        data = tmp_path / "store"

        code, output = _profile(
            "profile",
            "--module",
            user_module,
            "--function",
            "step",
            "--flops",
            "--data",
            str(data),
        )

        assert code == 0, output
        (run_file,) = (data / "runs").glob("*.json")
        metrics = json.loads(run_file.read_text())["points"][0]["metrics"]
        assert {"wall_clock_sec", "mean_batch_time_sec", "total_flops"} <= set(metrics)

    def test_profile_help(self) -> None:
        code, output = _profile("profile", "--help")

        assert code == 0
        for option in ("--module", "--function", "--warmup", "--iterations", "--energy", "--flops"):
            assert option in output


class TestProfileGpuCommand:
    """Tests for the 'profile-gpu' CLI command, with NVML answered by a fake."""

    def test_gpu_energy_is_measured(self, user_module: str) -> None:
        fake_nvml = make_fake_nvml()

        with patch("calibrax.profiling.nvml.pynvml", fake_nvml):
            code, output = _profile(
                "profile-gpu",
                "--module",
                user_module,
                "--function",
                "step",
                "--iterations",
                "20",
                "--gpu-index",
                "1",
            )

        assert code == 0, output
        assert "Mean GPU power: 240.00 W" in output
        assert fake_nvml.calls == ["init", "shutdown"]

    def test_profile_gpu_help(self) -> None:
        code, output = _profile("profile-gpu", "--help")

        assert code == 0
        assert "--gpu-index" in output
        assert "--energy" not in output
