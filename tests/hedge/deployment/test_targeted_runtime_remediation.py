from __future__ import annotations

from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[3]


def test_windows_runtime_alignment_uses_project_venv_and_requirements_pin() -> None:
    script = (ROOT / "scripts/Align-Freqtrade-Hedge-V17-Windows-Runtime-PS51.ps1").read_text(
        encoding="ascii"
    )
    assert ".venv\\Scripts\\python.exe" in script
    assert "requirements-freqai-rl.txt" in script
    assert "tqdm==" in script
    assert "pip check" in script
    assert "pip install -e" not in script


def test_docker_pytest_harness_is_ephemeral_and_runtime_venv_immutable() -> None:
    script = (ROOT / "scripts/Test-Freqtrade-Hedge-Docker-IsolatedPytest-PS51.ps1").read_text(
        encoding="ascii"
    )
    expected_test_venv = str(PurePosixPath("/", "tmp", "freqtrade-hedge-pytest-venv"))
    assert expected_test_venv in script
    assert "/opt/hedge-venv/bin/python" in script
    assert "pip freeze --all" in script
    assert "runtime venv contamination detected" in script
    assert "rm -rf $TestVenv" in script
    assert "docker build" not in script
    assert "docker run" not in script


def test_docker_pytest_harness_installs_only_test_tooling_into_temp_venv() -> None:
    script = (ROOT / "scripts/Test-Freqtrade-Hedge-Docker-IsolatedPytest-PS51.ps1").read_text(
        encoding="ascii"
    )
    for requirement in (
        "pytest==9.1.1",
        "pytest-asyncio==1.4.0",
        "pytest-cov==7.1.0",
        "pytest-mock==3.15.1",
        "pytest-random-order==1.2.0",
        "pytest-timeout==2.4.0",
        "pytest-xdist==3.8.0",
    ):
        assert requirement in script
    assert '$TestPython, "-m", "pip", "install"' in script
    assert '$RuntimePython, "-m", "pip", "install"' not in script
