import sys

import pytest

from freqtrade.hedge.production.runtime_test_isolation import (
    create_isolated_test_environment,
    read_test_specs,
    source_tree_sha256,
)


def test_source_digest_detects_python_byte_change(tmp_path):
    root = tmp_path / "project"
    target = root / "freqtrade" / "sample.py"
    target.parent.mkdir(parents=True)
    target.write_text("value = 1\n", encoding="utf-8")
    before = source_tree_sha256(root)
    target.write_text("value = 2\n", encoding="utf-8")
    after = source_tree_sha256(root)
    assert before != after


def test_read_test_specs_selects_only_test_tooling(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "x"
version = "0"
dependencies = []
[project.optional-dependencies]
develop = ["pytest==9.0.0", "pytest-xdist", "time-machine", "ruff", "mypy"]
""".strip(),
        encoding="utf-8",
    )
    assert read_test_specs(tmp_path) == ("pytest==9.0.0", "pytest-xdist", "time-machine")


def test_isolated_environment_must_not_live_under_source(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "x"
version = "0"
dependencies = []
[project.optional-dependencies]
develop = ["pytest"]
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside the source tree"):
        create_isolated_test_environment(
            tmp_path,
            runtime_python=sys.executable,
            environment_root=tmp_path / ".testenv",
            install_missing=False,
        )


def test_priority5_operator_cli_bootstraps_source_root_when_run_by_absolute_path(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[3]
    tool = root / "tools" / "hprl_priority5_closure.py"
    output = tmp_path / "source-probe.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(tool),
            "source-probe",
            "--root",
            str(root),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=60,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert Path(payload["source_root"]) == root
