from __future__ import annotations

import ast
from pathlib import Path


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "freqtrade/hedge/hprl").is_dir() and (parent / "tools").is_dir():
            return parent
    raise AssertionError("cannot resolve installed HEDGE project root")


def test_installed_core_files_parse_and_contracts_exist():
    root = project_root()
    paths = [
        root / "freqtrade/hedge/hprl/qualification.py",
        root / "freqtrade/hedge/hprl/diagnostics.py",
        root / "freqtrade/hedge/hprl/checkpoint_resume.py",
        root / "freqtrade/hedge/hprl/exchange_risk.py",
        root / "freqtrade/hedge/telemetry/training_health.py",
        root / "freqtrade/hedge/hprl/config.py",
        root / "freqtrade/hedge/hprl/networks.py",
        root / "freqtrade/hedge/hprl/algorithms/fast_td3.py",
        root / "freqtrade/hedge/hprl/algorithms/base.py",
        root / "freqtrade/hedge/hprl/replay.py",
        root / "freqtrade/hedge/hprl/trainer.py",
        root / "tools/train_hprl_eth_two_year.py",
    ]
    for path in paths:
        assert path.is_file(), path
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_installed_source_contains_learning_integrity_guards():
    root = project_root()
    config = (root / "freqtrade/hedge/hprl/config.py").read_text(encoding="utf-8")
    fast_td3 = (root / "freqtrade/hedge/hprl/algorithms/fast_td3.py").read_text(encoding="utf-8")
    trainer = (root / "freqtrade/hedge/hprl/trainer.py").read_text(encoding="utf-8")
    workflow = (root / "tools/train_hprl_eth_two_year.py").read_text(encoding="utf-8")
    assert "health_fail_mode" in config
    assert "fast_td3_actor_output_mode" in config
    assert "fast_td3_tier_exploration_epsilon" in fast_td3
    assert '"actor_updated"' in fast_td3
    assert "stop_requested" in trainer
    assert "executed_environment_steps" in trainer
    assert "qualify_candidate(" in workflow
    assert "holdout-role" in workflow
    assert "exchange-risk-evidence" in workflow
    assert "qualified = not final_collapsed" not in workflow
