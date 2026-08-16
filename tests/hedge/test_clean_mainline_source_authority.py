from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tools import validate_clean_mainline as validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_project_venv_cannot_contain_an_installed_freqtrade_copy(tmp_path: Path) -> None:
    site_packages = tmp_path / ".venv" / "Lib" / "site-packages"
    (site_packages / "freqtrade").mkdir(parents=True)
    (site_packages / "freqtrade-2026.8.dev0.dist-info").mkdir()

    result = validator.check_local_source_authority(tmp_path)

    assert result["status"] == "FAIL"
    assert result["detail"] == [
        ".venv/Lib/site-packages/freqtrade",
        ".venv/Lib/site-packages/freqtrade-2026.8.dev0.dist-info",
    ]


def test_user_data_versioned_runtime_paths_are_outside_source_layout_policy(
    tmp_path: Path,
) -> None:
    runtime_path = tmp_path / "user_data" / "r5" / "r5-mainnet-journal.sqlite"
    runtime_path.parent.mkdir(parents=True)
    runtime_path.write_bytes(b"runtime")

    result = validator.check_layout(tmp_path, workspace_mode=True)

    assert result["status"] == "FAIL"  # canonical files are absent in this fixture
    assert not any("versioned path: user_data/" in item for item in result["detail"])
    assert validator.should_ignore_workspace_path(tmp_path, runtime_path)


def test_external_cwd_reports_source_authority_identity(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import freqtrade; "
                "print(freqtrade.__file__); "
                "print(freqtrade.__source_root__); "
                "print(freqtrade.__source_authority_fingerprint__); "
                "print(freqtrade.__version__)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = probe.stdout.splitlines()

    assert Path(lines[0]).resolve() == PROJECT_ROOT / "freqtrade" / "__init__.py"
    assert Path(lines[1]).resolve() == PROJECT_ROOT
    assert len(lines[2]) == 12
    assert f"-src{lines[2]}" in lines[3]
