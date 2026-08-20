#!/usr/bin/env python3
"""Dependency-light post-install validator for HPRL Learning Integrity V1."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "freqtrade/hedge/hprl/qualification.py",
        root / "freqtrade/hedge/hprl/diagnostics.py",
        root / "freqtrade/hedge/hprl/checkpoint_resume.py",
        root / "freqtrade/hedge/hprl/exchange_risk.py",
        root / "freqtrade/hedge/telemetry/training_health.py",
        root / "freqtrade/hedge/hprl/config.py",
        root / "freqtrade/hedge/hprl/trainer.py",
        root / "freqtrade/hedge/hprl/replay.py",
        root / "freqtrade/hedge/hprl/algorithms/base.py",
        root / "freqtrade/hedge/hprl/algorithms/fast_td3.py",
        root / "freqtrade/hedge/hprl/networks.py",
        root / "tools/train_hprl_eth_two_year.py",
    ]
    for path in required:
        check(path.is_file(), f"missing required file: {path}")
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    from freqtrade.hedge.hprl.qualification import (
        QualificationStatus,
        qualify_candidate,
        search_is_degenerate,
        select_winner,
    )
    from freqtrade.hedge.telemetry.training_health import CollapseThresholds, RollingCollapseDetector

    healthy = {
        "training_health_ready": 1.0,
        "training_health_collapsed": 0.0,
        "policy_collapse": 0.0,
        "gradient_collapse": 0.0,
    }
    flat_metrics = {
        "net_return": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "max_drawdown": 0.0,
        "cvar": 0.0,
        "turnover": 0.0,
        "fees": 0.0,
        "funding": 0.0,
        "liquidations": 0,
    }
    flat = qualify_candidate(metrics=flat_metrics, health=healthy, actions=[[0.0, 0.0]] * 3349)
    check(flat.status == QualificationStatus.REJECTED_INACTIVE, "flat ETH regression must reject")

    detector = RollingCollapseDetector(CollapseThresholds(window=3, patience=2))
    result = {}
    collapsed_metrics = {
        "actor_grad_norm": 1e-3,
        "critic_grad_norm": 1.0,
        "actor_update_ratio": 1e-4,
        "critic_update_ratio": 1e-4,
        "policy_entropy": 0.0,
        "action_saturation": 1.0,
        "policy_action_mean": 0.0,
        "policy_action_std": 0.0,
        "flat_saturation": 1.0,
        "heavy_saturation": 0.0,
        "advantage_std": 0.1,
    }
    for _ in range(4):
        result = detector.update(collapsed_metrics)
    check(result["policy_flat_collapse"] == 1.0, "flat collapse classification failed")
    check(result["training_health_collapsed"] == 1.0, "pure policy collapse must be fatal")
    check(result["gradient_collapse"] == 0.0, "regression expects gradients still alive")

    winner, status = select_winner([
        {"status": "PASS", "objective": 0.0, "algorithm": "a"},
        {"status": "PASS", "objective": 0.0, "algorithm": "b"},
    ])
    check(winner is None and status == QualificationStatus.NO_DISTINGUISHABLE_WINNER, "tie must fail closed")
    check(search_is_degenerate([{"status": "PASS", "objective": 0.0}] * 8), "flat search degeneracy missing")

    training_source = (root / "tools/train_hprl_eth_two_year.py").read_text(encoding="utf-8")
    for token in (
        "qualify_candidate(",
        "holdout-role",
        "walk_forward",
        "benchmark_suite",
        "funding_rate",
        "source_manifest",
        "RESEARCH_PASS",
    ):
        check(token in training_source, f"training workflow contract missing: {token}")
    check("qualified = not final_collapsed" not in training_source, "legacy final qualification leak remains")
    check("winner = max(successful" not in training_source, "legacy winner selection remains")

    marker = root / "HPRL-LEARNING-INTEGRITY-V1-R2.json"
    check(marker.is_file(), "installation marker missing")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    check(payload.get("release") == "hprl-learning-integrity-v1-r2-20260819", "release marker mismatch")

    print(json.dumps({"status": "PASS", "checks": 11, "python": sys.version.split()[0]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
