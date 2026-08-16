from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np

from freqtrade.enums.hedge import PositionSide
from freqtrade.freqai.hedge_rl.actions import DEFAULT_ACTION_CATALOG
from freqtrade.freqai.hedge_rl.risk_bridge import HedgeRiskLevelPolicyBridge
from freqtrade.freqai.hedge_rl.risk_levels import HedgeRiskLevelAction, RiskLevelProfile
from freqtrade.freqai.hedge_rl.risk_planner_adapter import HedgeRiskLevelPlannerAdapter
from freqtrade.freqai.hedge_rl.risk_projection_adapter import context_from_central_projection
from freqtrade.freqai.hedge_rl.risk_runtime import (
    RiskRLAdaptiveCpuConfig,
    RiskRLAdaptiveCpuController,
)
from freqtrade.hedge.config_schema_extension import extend_config_schema
from freqtrade.hedge.integration.projection import CentralRuntimeProjection
from freqtrade.hedge.performance.resource_governor import ResourceSnapshot
from freqtrade.hedge.position_book import PositionRecord
from freqtrade.hedge.risk.models import AccountRiskSnapshot


def _risk() -> AccountRiskSnapshot:
    return AccountRiskSnapshot(
        account_id="acct",
        equity=Decimal(1000),
        wallet_balance=Decimal(1000),
        available_balance=Decimal(700),
        initial_margin=Decimal(300),
        maintenance_margin=Decimal(10),
        gross_long_notional=Decimal(150),
        gross_short_notional=Decimal(90),
        net_notional=Decimal(60),
        risk_data_valid=True,
        source_version=7,
    )


def _projection(*, stale: bool = False, healthy: bool = True) -> CentralRuntimeProjection:
    checks = {
        "exchange.rest_calibrated": healthy,
        "exchange.user_stream_fresh": healthy,
        "exchange.reconciliation_converged": healthy,
        "exchange.risk_snapshot_valid": healthy,
    }
    return CentralRuntimeProjection(
        positions=(
            PositionRecord(
                symbol="BTC/USDT:USDT",
                position_side=PositionSide.LONG,
                amount=Decimal("0.5"),
                entry_price=Decimal(100),
                mark_price=Decimal(100),
                leverage=Decimal(3),
                exchange="binance",
                account_id="acct",
            ),
            PositionRecord(
                symbol="BTC/USDT:USDT",
                position_side=PositionSide.SHORT,
                amount=Decimal("0.3"),
                entry_price=Decimal(100),
                mark_price=Decimal(100),
                leverage=Decimal(3),
                exchange="binance",
                account_id="acct",
            ),
        ),
        risk=_risk(),
        reconciliation_status="HEALTHY" if healthy else "DRIFT",
        reconciliation_at=datetime.now(UTC),
        reconciliation_details=(),
        stream_state="CONNECTED" if healthy else "STALE",
        stream_last_event_at=datetime.now(UTC),
        stream_reconnect_count=0,
        checks=checks,
        reasons=(),
        source_version="7",
        source_event_time=datetime.now(UTC),
        stale=stale,
    )


def test_existing_21_action_contract_is_preserved() -> None:
    assert len(DEFAULT_ACTION_CATALOG) == 21
    assert DEFAULT_ACTION_CATALOG.decode(0).action.name == "HOLD"


def test_risk_level_contract_is_parallel_5x5() -> None:
    profile = RiskLevelProfile()
    assert profile.position_levels == (0.0, 0.05, 0.12, 0.25, 0.40)
    assert len([HedgeRiskLevelAction.from_value((i, j)) for i in range(5) for j in range(5)]) == 25


def test_schema_exposes_action_reward_and_runtime_without_replacing_freqai_ref() -> None:
    schema = {
        "properties": {"freqai": {"$ref": "#/definitions/freqai"}},
        "definitions": {
            "freqai": {
                "type": "object",
                "properties": {"rl_config": {"type": "object", "properties": {}}},
            }
        },
    }
    extend_config_schema(schema)
    freqai = schema["definitions"]["freqai"]["properties"]
    assert "hedge_action_space" in freqai["rl_config"]["properties"]
    assert "hedge_reward" in freqai["rl_config"]["properties"]
    assert "hedge_rl_config" in freqai
    assert "adaptive_cpu" in freqai["hedge_rl_config"]["properties"]
    assert schema["properties"]["freqai"] == {"$ref": "#/definitions/freqai"}


def test_canonical_projection_builds_fresh_context_and_conservative_levels() -> None:
    profile = RiskLevelProfile(long_leverage=3.0, short_leverage=3.0)
    context = context_from_central_projection(
        _projection(), pair="BTC/USDT:USDT", mark=100.0, profile=profile
    )
    assert context.projection_fresh is True
    # LONG margin = 50/3/1000 ~= 1.67% -> ceiling level 1 (5%).
    # SHORT margin = 30/3/1000 = 1% -> ceiling level 1.
    assert context.account.long_level == 1
    assert context.account.short_level == 1
    assert context.account.long.quantity == 0.5
    assert context.account.short.quantity == 0.3


def test_stale_or_unhealthy_projection_fails_closed() -> None:
    profile = RiskLevelProfile()
    stale = context_from_central_projection(
        _projection(stale=True), pair="BTC/USDT:USDT", mark=100.0, profile=profile
    )
    unhealthy = context_from_central_projection(
        _projection(healthy=False), pair="BTC/USDT:USDT", mark=100.0, profile=profile
    )
    assert stale.projection_fresh is False
    assert unhealthy.projection_fresh is False


def test_policy_bridge_returns_flat_on_stale_projection() -> None:
    profile = RiskLevelProfile()
    bridge = HedgeRiskLevelPolicyBridge(
        feature_names=("f",), window_size=2, profile=profile, max_feature_age_steps=1
    )
    context = context_from_central_projection(
        _projection(stale=True), pair="BTC/USDT:USDT", mark=100.0, profile=profile
    )
    obs = bridge.observation(np.asarray([[0.1], [0.2]], dtype=np.float32), tick=1, context=context)

    class Model:
        def predict(self, *_args, **_kwargs):
            raise AssertionError("stale projection must not invoke the policy")

    action = bridge.predict_action(Model(), obs, context=context)
    assert (int(action.long_level), int(action.short_level)) == (0, 0)


def test_same_level_planner_signal_cannot_add_risk() -> None:
    profile = RiskLevelProfile(long_leverage=3.0, short_leverage=3.0)
    context = context_from_central_projection(
        _projection(), pair="BTC/USDT:USDT", mark=100.0, profile=profile
    )
    adapter = HedgeRiskLevelPlannerAdapter(profile)
    signal = adapter.from_account_action(
        HedgeRiskLevelAction.from_value((1, 1)), account=context.account, mark=100.0
    )
    assert signal.long_increase_allowed is False
    assert signal.short_increase_allowed is False
    assert signal.allow_new_risk is False
    assert signal.target_semantics == "RISK_CAP_NO_SAME_LEVEL_SCALE_IN"


def test_explicit_level_raise_can_add_only_that_leg() -> None:
    profile = RiskLevelProfile(long_leverage=3.0, short_leverage=3.0)
    context = context_from_central_projection(
        _projection(), pair="BTC/USDT:USDT", mark=100.0, profile=profile
    )
    signal = HedgeRiskLevelPlannerAdapter(profile).from_account_action(
        HedgeRiskLevelAction.from_value((2, 1)), account=context.account, mark=100.0
    )
    assert signal.long_increase_allowed is True
    assert signal.short_increase_allowed is False
    assert signal.allow_new_risk is True


class _FakeGovernor:
    def __init__(self, *, cpu: float, physical: int, suggested: int) -> None:
        self._cpu = cpu
        self._physical = physical
        self._suggested = suggested

    def snapshot(self, *, sample_seconds: float = 0.0) -> ResourceSnapshot:
        del sample_seconds
        return ResourceSnapshot(
            logical_cpus=32,
            physical_cpus=self._physical,
            affinity_cpus=32,
            system_cpu_percent=self._cpu,
            process_cpu_percent=0.0,
            cgroup_memory_limit_bytes=8 * 1024**3,
            cgroup_memory_current_bytes=1 * 1024**3,
            host_memory_available_bytes=12 * 1024**3,
            timestamp_monotonic=1.0,
            source="host-broker",
            host_snapshot_age_seconds=0.1,
        )

    def numeric_threads(self, *, concurrent_python_workers: int, snapshot: ResourceSnapshot) -> int:
        assert concurrent_python_workers == 1
        assert snapshot.source == "host-broker"
        return self._suggested


def test_adaptive_risk_rl_threads_respect_physical_and_config_caps() -> None:
    controller = RiskRLAdaptiveCpuController(
        RiskRLAdaptiveCpuConfig(max_torch_threads=12),
        governor=_FakeGovernor(cpu=10.0, physical=16, suggested=20),  # type: ignore[arg-type]
    )
    assert controller.recommended_threads() == 12
    assert controller.telemetry()["resource_source"] == "host-broker"


def test_adaptive_risk_rl_threads_can_contract_under_load() -> None:
    controller = RiskRLAdaptiveCpuController(
        RiskRLAdaptiveCpuConfig(max_torch_threads=16),
        governor=_FakeGovernor(cpu=95.0, physical=16, suggested=1),  # type: ignore[arg-type]
    )
    assert controller.recommended_threads() == 1


def test_risk_level_learner_source_is_cpu_only_and_adaptive() -> None:
    source = (
        Path(__file__).parents[3]
        / "freqtrade"
        / "freqai"
        / "prediction_models"
        / "HedgeRiskLevelReinforcementLearner.py"
    ).read_text(encoding="utf-8")
    assert 'parameters["device"] = "cpu"' in source
    assert 'device="cpu"' in source
    assert "RiskRLAdaptiveCpuController" in source
    assert "_AdaptiveRiskCpuCallback" in source


def test_risk_level_runtime_has_no_hprl_dependency() -> None:
    root = Path(__file__).parents[3] / "freqtrade" / "freqai" / "hedge_rl"
    risk_sources = "\n".join(p.read_text(encoding="utf-8") for p in root.glob("risk_*.py")).lower()
    assert "hprl" not in risk_sources


def _runtime_risk(*, equity: int = 1000, wallet: int | None = None) -> AccountRiskSnapshot:
    value = equity if wallet is None else wallet
    return AccountRiskSnapshot(
        account_id="acct",
        equity=Decimal(equity),
        wallet_balance=Decimal(value),
        available_balance=Decimal(max(0, value - 100)),
        initial_margin=Decimal(0),
        maintenance_margin=Decimal(0),
        gross_long_notional=Decimal(0),
        gross_short_notional=Decimal(0),
        net_notional=Decimal(0),
        risk_data_valid=True,
        source_version=max(0, equity),
    )


def _runtime_for_mode(mode: str):
    from freqtrade.enums.hedge import PositionMode
    from freqtrade.freqai.hedge_rl.risk_projection_adapter import HedgeRiskRuntimeContextProvider
    from freqtrade.hedge.config import HedgeRuntimeConfig
    from freqtrade.hedge.runtime import HedgeProjectionSource, HedgeRuntime

    runtime = HedgeRuntime(
        HedgeRuntimeConfig(
            position_mode=PositionMode.HEDGE,
            enabled=True,
            managed_pair="BTC/USDT:USDT",
            account_id="acct",
            operation_mode=mode,
        )
    )
    provider = HedgeRiskRuntimeContextProvider(runtime, profile=RiskLevelProfile())
    source = HedgeProjectionSource.PAPER if mode == "paper" else HedgeProjectionSource.EXCHANGE
    return runtime, provider, source


def test_runtime_context_provider_supports_paper_without_exchange_checks() -> None:
    runtime, provider, source = _runtime_for_mode("paper")
    runtime.publish(
        source=source,
        positions=(),
        risk=_runtime_risk(),
        reconciliation_status="NOT_APPLICABLE",
        reconciliation_at=None,
        stream_state="NOT_APPLICABLE",
        stream_last_event_at=None,
        stream_reconnect_count=0,
        checks={
            "common.persistence_healthy": True,
            "paper.market_data_fresh": True,
            "paper.funding_source_healthy": True,
            "paper.account_events_durable": True,
            "paper.simulation_engine_healthy": True,
            "paper.ledger_durable": True,
            "paper.risk_snapshot_valid": True,
        },
    )
    context = provider("BTC/USDT:USDT", 0, 0)
    assert context.projection_fresh is True
    assert context.account.equity == 1000.0


def test_runtime_context_provider_preserves_peak_equity_across_sequences() -> None:
    runtime, provider, source = _runtime_for_mode("readonly")
    checks = {
        "common.persistence_healthy": True,
        "exchange.readonly_service_bound": True,
        "exchange.rest_calibrated": True,
        "exchange.user_stream_fresh": True,
        "exchange.reconciliation_converged": True,
        "exchange.risk_snapshot_valid": True,
    }
    for equity in (1000, 900):
        runtime.publish(
            source=source,
            positions=(),
            risk=_runtime_risk(equity=equity, wallet=equity),
            reconciliation_status="HEALTHY",
            reconciliation_at=datetime.now(UTC),
            stream_state="CONNECTED",
            stream_last_event_at=datetime.now(UTC),
            stream_reconnect_count=0,
            checks=checks,
        )
        context = provider("BTC/USDT:USDT", 0, 0)
    assert context.projection_fresh is True
    assert context.account.peak_equity == 1000.0
    assert abs(context.account.drawdown() - 0.1) < 1e-12
    assert context.downside_semideviation > 0


def test_freqtradebot_passes_runtime_into_strategy_start_lifecycle() -> None:
    root = Path(__file__).parents[3] / "freqtrade"
    bot = (root / "freqtradebot.py").read_text(encoding="utf-8")
    interface = (root / "strategy" / "interface.py").read_text(encoding="utf-8")
    assert "self.strategy.ft_bot_start(hedge_runtime=self.hedge_runtime)" in bot
    load = interface.index("self.load_freqAI_model()")
    bind = interface.index('getattr(self.freqai, "set_hedge_runtime", None)')
    callback = interface.index("strategy_safe_wrapper(self.bot_start)()")
    assert load < bind < callback


def test_projection_leverage_mismatch_fails_closed_instead_of_misclassifying_margin() -> None:
    context = context_from_central_projection(
        _projection(),
        pair="BTC/USDT:USDT",
        mark=100.0,
        profile=RiskLevelProfile(long_leverage=1.0, short_leverage=1.0),
    )
    assert context.projection_fresh is False
