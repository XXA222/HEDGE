from pathlib import Path

from freqtrade.hedge.validate import _commands


def test_unified_validation_dispatcher_builds_workspace_commands() -> None:
    root = Path("D:/freqtrade-hedge")
    commands = _commands(root, "all")

    assert commands
    assert any("validate_clean_mainline.py" in item for command in commands for item in command)
    assert any("validate_hedge_ruff_staged.py" in item for command in commands for item in command)
    assert any(
        "benchmark_hedge_audit_h01_1500.py" in item for command in commands for item in command
    )
    assert all(command[0] for command in commands)
