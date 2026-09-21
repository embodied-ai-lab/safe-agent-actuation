"""Regression tests for the CPU-only, lockfile-driven setup script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts" / "setup.sh"
DOCTOR = ROOT / "scripts" / "doctor.py"
BUILD_AUTH = ROOT / "sst" / "scripts" / "build_auth.sh"
GENERATE_RUNTIME = ROOT / "sst" / "scripts" / "generate_runtime.sh"


def test_setup_uses_only_locked_runtime_dependencies() -> None:
    text = SETUP.read_text(encoding="utf-8")
    assert "uv sync" in text
    assert "--locked" in text
    assert "--no-install-project" in text
    assert "uv pip install" not in text
    assert "third_party/openpi/pyproject.toml" not in text
    assert "git " not in text
    assert "safe_agent_actuation_course.pth" in text
    assert 'root / "src"' in text
    assert 'root / "third_party" / "iotauth" / "entity" / "python"' in text


def test_setup_refuses_login_node_before_writing_environment(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    hostname = fake_bin / "hostname"
    hostname.write_text("#!/usr/bin/env bash\necho login-test\n", encoding="utf-8")
    hostname.chmod(0o750)
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"

    result = subprocess.run(
        [str(SETUP)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Refusing environment setup on a login node" in result.stderr


def test_runtime_checks_and_sst_staging_do_not_invoke_git() -> None:
    """The validated workflow can run under an explicit no-Git constraint."""

    doctor_text = DOCTOR.read_text(encoding="utf-8")
    build_text = BUILD_AUTH.read_text(encoding="utf-8")
    generate_text = GENERATE_RUNTIME.read_text(encoding="utf-8")

    assert "subprocess" not in doctor_text
    assert '["git"' not in doctor_text
    assert "git -C" not in build_text
    assert "git -C" not in generate_text
    assert "*/target" in build_text
    assert "*/target" in generate_text
