"""Tests for the setup environment helper scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_setup_env_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "setup_env.py"
    spec = importlib.util.spec_from_file_location("calibrax_setup_env", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_backend_sets_cpu_platform_without_cuda_paths(tmp_path: Path) -> None:
    """CPU backend writes deterministic managed env without CUDA path injection."""
    setup_env = _load_setup_env_module()

    contents = setup_env.build_env_contents(tmp_path, "cpu")

    assert "export CALIBRAX_BACKEND=cpu" in contents
    assert "export CALIBRAX_ENV_ROOT=" in contents
    assert "export JAX_PLATFORMS=cpu" in contents
    assert "LD_LIBRARY_PATH" not in contents
    assert "CUDA_HOME" not in contents
    assert "XLA_FLAGS" not in contents


def test_accelerator_backends_leave_platform_selection_to_jax(tmp_path: Path) -> None:
    """CUDA and Metal backends let JAX select the available accelerator."""
    setup_env = _load_setup_env_module()

    cuda_contents = setup_env.build_env_contents(tmp_path, "cuda12")
    metal_contents = setup_env.build_env_contents(tmp_path, "metal")

    assert "export CALIBRAX_BACKEND=cuda12" in cuda_contents
    assert "unset JAX_PLATFORMS" in cuda_contents
    assert "export CALIBRAX_BACKEND=metal" in metal_contents
    assert "unset JAX_PLATFORMS" in metal_contents


def test_show_status_reports_managed_and_user_env_layers(tmp_path: Path, capsys) -> None:
    """Status output reports generated and user-owned env files separately."""
    setup_env = _load_setup_env_module()
    managed_env = tmp_path / ".calibrax.env"
    managed_env.write_text("export CALIBRAX_BACKEND=cpu\n")
    (tmp_path / ".env.local").write_text("export EXAMPLE=1\n")

    result = setup_env.show_status(tmp_path, managed_env)

    assert result == 0
    output = capsys.readouterr().out
    assert ".calibrax.env present: True" in output
    assert ".env present: False" in output
    assert ".env.local present: True" in output
    assert "Configured backend: cpu" in output


def test_auto_backend_resolves_to_a_supported_concrete_backend() -> None:
    """`auto` never leaks through as a stored backend value."""
    setup_env = _load_setup_env_module()

    resolved = setup_env.resolve_backend("auto")

    assert resolved in {"cpu", "cuda12", "metal"}
    assert resolved != "auto"
