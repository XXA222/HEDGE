#!/usr/bin/env python3
"""Deterministic 400-round dual-runtime acceptance for Hedge Risk-Level RL.

Covers trading semantics, source-separated runtime projection, Windows launch contracts,
Docker source authority, CPU-only policy intent, and regression invariants inherited from
the V3 Action/Reward and V1.6 adaptive/memory lines.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from freqtrade.enums.hedge import PositionMode
from freqtrade.freqai.hedge_rl.risk_levels import HedgeRiskLevelAction, RiskLevelProfile
from freqtrade.freqai.hedge_rl.risk_planner_adapter import HedgeRiskLevelPlannerAdapter
from freqtrade.freqai.hedge_rl.risk_portfolio import RiskAccountState
from freqtrade.freqai.hedge_rl.risk_projection_adapter import HedgeRiskRuntimeContextProvider
from freqtrade.hedge.config import HedgeRuntimeConfig
from freqtrade.hedge.risk.models import AccountRiskSnapshot
from freqtrade.hedge.runtime import HedgeProjectionSource, HedgeRuntime


ROOT = Path(__file__).resolve().parents[1]
PAIR = "BTC/USDT:USDT"


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _risk(equity: Decimal) -> AccountRiskSnapshot:
    return AccountRiskSnapshot(
        account_id="acct",
        equity=equity,
        wallet_balance=equity,
        available_balance=equity,
        initial_margin=Decimal(0),
        maintenance_margin=Decimal(0),
        gross_long_notional=Decimal(0),
        gross_short_notional=Decimal(0),
        net_notional=Decimal(0),
        risk_data_valid=True,
        source_version=max(0, int(equity)),
    )


def _runtime(mode: str) -> tuple[HedgeRuntime, HedgeProjectionSource]:
    runtime = HedgeRuntime(
        HedgeRuntimeConfig(
            position_mode=PositionMode.HEDGE,
            enabled=True,
            managed_pair=PAIR,
            account_id="acct",
            operation_mode=mode,
        )
    )
    source = HedgeProjectionSource.PAPER if mode == "paper" else HedgeProjectionSource.EXCHANGE
    return runtime, source


def _publish(
    runtime: HedgeRuntime, source: HedgeProjectionSource, equity: Decimal, *, good: bool
) -> None:
    if source is HedgeProjectionSource.PAPER:
        checks = {
            "common.persistence_healthy": good,
            "paper.market_data_fresh": good,
            "paper.funding_source_healthy": good,
            "paper.account_events_durable": good,
            "paper.simulation_engine_healthy": good,
            "paper.ledger_durable": good,
            "paper.risk_snapshot_valid": good,
        }
        reconciliation = "NOT_APPLICABLE"
        stream = "NOT_APPLICABLE"
    else:
        checks = {
            "common.persistence_healthy": good,
            "exchange.readonly_service_bound": good,
            "exchange.rest_calibrated": good,
            "exchange.user_stream_fresh": good,
            "exchange.reconciliation_converged": good,
            "exchange.risk_snapshot_valid": good,
        }
        reconciliation = "HEALTHY" if good else "DRIFT"
        stream = "CONNECTED" if good else "STALE"
    runtime.publish(
        source=source,
        positions=(),
        risk=_risk(equity),
        reconciliation_status=reconciliation,
        reconciliation_at=datetime.now(UTC),
        stream_state=stream,
        stream_last_event_at=datetime.now(UTC),
        stream_reconnect_count=0,
        checks=checks,
        reasons=() if good else ("INJECTED_UNHEALTHY",),
        stale=not good,
    )


def _static_contracts() -> None:
    bot = (ROOT / "freqtrade/freqtradebot.py").read_text(encoding="utf-8")
    interface = (ROOT / "freqtrade/strategy/interface.py").read_text(encoding="utf-8")
    learner = (
        ROOT / "freqtrade/freqai/prediction_models/HedgeRiskLevelReinforcementLearner.py"
    ).read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    docker_ps = (ROOT / "scripts/Start-Hedge-Docker.ps1").read_text(encoding="ascii")
    broker_ps = (ROOT / "scripts/Start-Freqtrade-Hedge-Adaptive-Resource-Broker.ps1").read_text(
        encoding="ascii"
    )

    _check(
        "self.strategy.ft_bot_start(hedge_runtime=self.hedge_runtime)" in bot,
        "FreqtradeBot must pass the Hedge runtime into strategy startup",
    )
    _check(
        interface.index("self.load_freqAI_model()") < interface.index('"set_hedge_runtime"'),
        "FreqAI model loading must precede Hedge runtime registration",
    )
    _check(
        interface.index('"set_hedge_runtime"')
        < interface.index("strategy_safe_wrapper(self.bot_start)()"),
        "Hedge runtime registration must precede bot_start",
    )
    _check("HedgeRiskRuntimeContextProvider" in learner, "risk runtime provider is required")
    _check("_ensure_model_cpu(model)" in learner, "risk model CPU guard is required")
    _check('parameters["device"] = "cpu"' in learner, "risk model device must be CPU")
    _check('device="cpu"' in learner, "risk model load must request CPU")
    _check("INSTALL_HEDGE_RISKLEVEL_RL=true" in dockerfile, "CPU RL Docker flag is required")
    _check("TORCH_CPU_INDEX_URL" in dockerfile, "CPU Torch index is required")
    _check("requirements-hedge-mlrl.txt" in dockerfile, "CPU RL requirements are required")
    _check("assert not torch.cuda.is_available()" not in dockerfile, "Docker must not assert CUDA")
    _check("freqtradeorg/freqtrade:stable" not in compose, "compose must use the project image")
    _check("build:" in compose and "dockerfile: Dockerfile" in compose, "compose build is required")
    _check("/opt/freqtrade-hedge/user_data" in compose, "compose must mount user_data")
    _check("freqtrade-hedge:1.7-risklevel-rl-cpu" in compose, "compose image tag is required")
    _check(
        "freqtrade-hedge:1.7-risklevel-rl-cpu" in docker_ps,
        "Docker launcher image tag is required",
    )
    _check('"freqtrade", "trade"' in docker_ps, "Docker launcher must start trade mode")
    _check("/opt/freqtrade-hedge/user_data" in docker_ps, "Docker launcher must mount user_data")
    _check(
        "Start-Freqtrade-Hedge-Adaptive-Resource-Broker.ps1" in docker_ps,
        "Docker launcher must start the resource broker",
    )
    _check('UserDataRoot = ""' in broker_ps, "resource broker must use its configured data root")
    _check("host-resource-snapshot.json" in broker_ps, "resource broker snapshot is required")


def _scenario(round_id: int, rng: random.Random, profile: RiskLevelProfile) -> str:
    family = round_id % 10
    if family == 0:
        long = rng.randrange(5)
        short = rng.randrange(5)
        action = HedgeRiskLevelAction.from_value((long, short))
        _check(
            int(action.long_level) == long and int(action.short_level) == short,
            "joint action round-trip changed",
        )
        _check(0 <= action.joint_id < 25, "joint action id is outside the 5x5 space")
        return "action-25-state"

    if family == 1:
        account = RiskAccountState.initial(1000.0)
        planner = HedgeRiskLevelPlannerAdapter(profile)
        action = HedgeRiskLevelAction.from_value((rng.randrange(5), rng.randrange(5)))
        signal = planner.from_account_action(action, account=account, mark=100.0)
        _check(
            signal.allow_new_risk == (int(action.long_level) > 0 or int(action.short_level) > 0),
            "planner increase permission changed",
        )
        return "planner-explicit-increase"

    if family in {2, 3, 4, 5}:
        mode = "paper" if family in {2, 4} else "readonly"
        runtime, source = _runtime(mode)
        good = family in {2, 3}
        _publish(runtime, source, Decimal(1000), good=good)
        provider = HedgeRiskRuntimeContextProvider(runtime, profile=profile)
        context = provider(PAIR, 0, round_id)
        _check(context.projection_fresh is good, "projection freshness contract changed")
        _check(math.isfinite(context.account.equity), "risk account equity must be finite")
        if mode == "paper" and good:
            _check(
                dict(runtime.view().checks)["paper.risk_snapshot_valid"] is True,
                "paper risk snapshot must be valid",
            )
        return f"runtime-{mode}-{'fresh' if good else 'failclosed'}"

    if family == 6:
        runtime, source = _runtime("readonly")
        provider = HedgeRiskRuntimeContextProvider(runtime, profile=profile)
        _publish(runtime, source, Decimal(1000), good=True)
        first = provider(PAIR, 1, round_id)
        # Repeated rows against the same runtime sequence must not advance account history.
        duplicate = provider(PAIR, 2, round_id)
        _check(
            first.account.step == duplicate.account.step,
            "duplicate projection advanced history",
        )
        _check(duplicate.downside_semideviation == 0.0, "duplicate projection changed downside")
        _publish(runtime, source, Decimal(900), good=True)
        second = provider(PAIR, 3, round_id)
        _check(second.account.peak_equity == 1000.0, "peak equity history changed")
        _check(second.account.drawdown() > 0.09, "drawdown history was not recorded")
        _check(second.downside_semideviation > 0, "downside history was not recorded")
        return "runtime-history-no-doublecount"

    if family == 7:
        _static_contracts()
        return "windows-docker-static-contract"

    if family == 8:
        _check(profile.position_levels == (0.0, 0.05, 0.12, 0.25, 0.40), "risk levels changed")
        _check(
            profile.long_leverage > 0 and profile.short_leverage > 0,
            "leverage must be positive",
        )
        _check(profile.position_levels[-1] < 0.5, "risk levels must remain non-all-in")
        return "v3-risk-budget-semantics"

    _static_contracts()
    risk_sources = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in (ROOT / "freqtrade/freqai/hedge_rl").glob("risk_*.py")
    )
    _check("hprl" not in risk_sources, "legacy HPRL namespace leaked into risk sources")
    return "independence-and-cpu-contract"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = random.Random(0x170814)  # noqa: S311 - deterministic offline validation matrix
    profile = RiskLevelProfile()
    counts: dict[str, int] = {}
    failures: list[dict[str, object]] = []
    for round_id in range(1, 401):
        try:
            family = _scenario(round_id, rng, profile)
            counts[family] = counts.get(family, 0) + 1
        except Exception as exc:  # pragma: no cover - validator failure path
            failures.append({"round": round_id, "error": repr(exc)})
    payload = {
        "schema": "freqtrade-hedge-risklevel-rl-dual-runtime-400-v1",
        "rounds": 400,
        "passed": 400 - len(failures),
        "failed": len(failures),
        "families": counts,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
