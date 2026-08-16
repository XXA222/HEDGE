from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from types import SimpleNamespace

import pytest
import torch

from freqtrade.freqai.hedge_rl.risk_bridge import HedgeRiskPolicyContext
from freqtrade.freqai.hedge_rl.risk_levels import RiskLevelProfile
from freqtrade.freqai.hedge_rl.risk_portfolio import LegSide, RiskAccountState, RiskLegState
from freqtrade.freqai.hedge_rl.risk_projection_adapter import _observed_margin_over_cap
from freqtrade.hedge.execution.binance_usdm_adapter import (
    BinanceUSDMExecutionAdapter,
    BinanceExecutionApiError,
    _timestamp,
)
from freqtrade.hedge.exchange.shared_rate_limit import SharedWeightDecision
from freqtrade.hedge.hprl.checkpoint import load_checkpoint
from freqtrade.hedge.hprl.config import HPRLActionConfig, HPRLCostConfig, HPRLRewardConfig
from freqtrade.hedge.hprl.contracts import PlannedExecutionIntent
from freqtrade.hedge.hprl.costs import ExecutionCostModel
from freqtrade.hedge.hprl.replay import TensorReplayBuffer
from freqtrade.hedge.hprl.reward import CompositeReward, RewardFactsTensor
from freqtrade.hedge.hprl.risk import HedgeActionProjector
from freqtrade.hedge.hprl.trainer import DiscountedReturnNormalizer
from freqtrade.hedge.integration.production_context import ReadonlyPlanningContextBuilder
from freqtrade.hedge.integration.main_loop_config import ProductionMainLoopConfig
from freqtrade.hedge.integration.production_main_loop import HedgeExecutionMode
from freqtrade.hedge.planning.context import utc_aware
from freqtrade.hedge.production.closed_loop import (
    ClosedLoopCycleRecord,
    ClosedLoopCycleStatus,
    ZERO_HASH,
)
from freqtrade.hedge.production.control import ControlMode
from freqtrade.hedge.production.hprl_hedge_adapter import (
    HprlHedgeAdapter,
    HprlHedgeAdapterPolicy,
)
from freqtrade.hedge.production.model_governance import (
    ApprovalRecord,
    InferenceHealth,
    ModelIdentity,
    ModelStatus,
)
from freqtrade.hedge.production.recovery_checkpoint import DurableRecoveryCheckpoint
from freqtrade.hedge.production.runtime_supervisor import RuntimeSafetyInput
from freqtrade.hedge.production.source_convergence import _is_live_critical


NOW = datetime(2026, 8, 16, tzinfo=UTC)
HASH = "a" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("projection_fresh", "false"),
        ("projection_fresh", 1),
        ("feature_age_steps", True),
        ("long_level", 1.0),
        ("mark", float("nan")),
        ("turnover", float("inf")),
    ],
)
def test_risk_bridge_mapping_rejects_coercive_or_nonfinite_values(field, value) -> None:
    values = {"mark": Decimal("100"), "cash_balance": Decimal("1000"), field: value}
    with pytest.raises((TypeError, ValueError)):
        HedgeRiskPolicyContext.from_mapping(values)


def test_hprl_mandatory_numeric_checks_cannot_be_disabled() -> None:
    projector = HedgeActionProjector(HPRLActionConfig(mode="continuous"), validate_inputs=False)
    with pytest.raises(ValueError, match="finite"):
        projector.project(torch.tensor([[[float("inf"), 0.0]]]), torch.zeros((1, 1, 2)))

    costs = ExecutionCostModel(HPRLCostConfig(), validate_inputs=False)
    with pytest.raises(ValueError, match="finite"):
        costs.evaluate(
            turnover_notional=torch.tensor([float("nan")]),
            equity=torch.tensor([1000.0]),
        )

    reward = CompositeReward(HPRLRewardConfig(), validate_inputs=False)
    zero = torch.zeros(1)
    facts = RewardFactsTensor(
        equity_return=torch.tensor([float("nan")]),
        drawdown_increase=zero,
        downside_return=zero,
        cvar_loss=zero,
        turnover_ratio=zero,
        fee_ratio=zero,
        slippage_ratio=zero,
        impact_ratio=zero,
        funding_ratio=zero,
    )
    with pytest.raises(ValueError, match="finite"):
        reward.evaluate_tensor(facts)

    replay = TensorReplayBuffer(4, 2, 1, device="cpu", pin_memory=False, validate_inputs=False)
    with pytest.raises(ValueError, match="finite"):
        replay.add(
            torch.tensor([[float("nan"), 0.0]]),
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            torch.zeros((1, 2)),
            torch.zeros((1, 1)),
        )

    normalizer = DiscountedReturnNormalizer(1, 0.99, device="cpu", validate_inputs=False)
    with pytest.raises(ValueError, match="finite"):
        normalizer.normalize(torch.tensor([float("inf")]), torch.tensor([0.0]))


def test_malformed_active_order_is_not_silently_dropped_or_repriced() -> None:
    order = SimpleNamespace(
        original_quantity=Decimal("1"),
        cumulative_filled_quantity=Decimal("0"),
        position_side="SIDEWAYS",
        side="BUY",
        exchange_order_id="1",
        client_order_id="c1",
        average_price=Decimal("0"),
        raw={},
        reduce_only=False,
        order_type="LIMIT",
    )
    market = SimpleNamespace(symbol="BTCUSDT", mark=Decimal("100"))
    leg = SimpleNamespace(quantity=Decimal("0"))
    with pytest.raises(ValueError, match="side identity"):
        ReadonlyPlanningContextBuilder._active_order(order, market=market, long=leg, short=leg)

    order.position_side = "LONG"
    order.raw = {"price": "not-a-number"}
    with pytest.raises(ValueError, match="price"):
        ReadonlyPlanningContextBuilder._active_order(order, market=market, long=leg, short=leg)


def _intent(*, symbol: str = "BTCUSDT", model_id: str = "model-a") -> PlannedExecutionIntent:
    return PlannedExecutionIntent(symbol, 0.1, 0.05, 1.0, model_id)


@pytest.mark.parametrize("changed", ["symbol", "model"])
def test_previous_hprl_projection_is_bound_to_symbol_and_model(changed: str) -> None:
    adapter = HprlHedgeAdapter(HprlHedgeAdapterPolicy())
    previous = adapter.adapt(_intent(), sequence=1, observed_at=NOW, now=NOW)
    current = (
        _intent(symbol="ETHUSDT") if changed == "symbol" else _intent(model_id="model-b")
    )
    with pytest.raises(ValueError, match="previous HPRL projection"):
        adapter.adapt(current, sequence=2, observed_at=NOW, now=NOW, previous=previous)


@pytest.mark.parametrize("value", [None, True, 0, -1, "12.3", 1.5])
def test_binance_timestamp_rejects_missing_or_coercive_values(value) -> None:
    with pytest.raises(BinanceExecutionApiError):
        _timestamp(value)


def test_checkpoint_rejects_unknown_future_schema(tmp_path) -> None:
    path = tmp_path / "future.pt"
    torch.save({"schema": 999, "metadata": {}, "agent_state": {}}, path)
    with pytest.raises(ValueError, match="unsupported checkpoint schema"):
        load_checkpoint(path, SimpleNamespace(device="cpu"))


def test_recovery_payload_rejects_boolean_integer_fields() -> None:
    checkpoint = DurableRecoveryCheckpoint(
        generation=1,
        created_at=NOW,
        source_release="release",
        model_id="model",
        evidence_digest=HASH,
        reconciliation_digest=HASH,
        projection_chain_sha256=HASH,
        last_market_sequence=1,
        last_user_sequence=1,
    )
    payload = checkpoint.payload()
    payload["generation"] = True
    with pytest.raises(TypeError, match="generation"):
        DurableRecoveryCheckpoint.from_payload(payload)


def _identity() -> ModelIdentity:
    return ModelIdentity("model", "HPRL", HASH, HASH, HASH, HASH, "torch")


def test_model_governance_rejects_future_approval_and_nonfinite_health() -> None:
    with pytest.raises(ValueError, match="future"):
        ApprovalRecord(
            _identity(),
            ModelStatus.APPROVED,
            datetime.now(UTC) + timedelta(days=1),
            "ops",
            True,
            True,
            True,
            "golden",
        )
    with pytest.raises(ValueError, match="finite"):
        InferenceHealth(float("nan"), True, HASH, HASH, 0.0)


def test_runtime_safety_input_rejects_truthy_non_booleans() -> None:
    reconciliation = SimpleNamespace()
    with pytest.raises(TypeError, match="market_data_fresh"):
        RuntimeSafetyInput(
            control_mode=ControlMode.RUN,
            reconciliation=reconciliation,
            incident_blocks_new_risk=False,
            incident_blocks_account=False,
            market_data_fresh="false",
            risk_data_fresh=True,
        )


def _closed_loop_record() -> ClosedLoopCycleRecord:
    digest = lambda value: sha256(value.encode()).hexdigest()
    return ClosedLoopCycleRecord(
        sequence=1,
        cycle_id="cycle",
        observed_at=NOW,
        source_release="release",
        model_id="model",
        symbol="BTCUSDT",
        projection_sequence=1,
        projection_observed_at=NOW,
        projection_source_sha256=digest("source"),
        projection_semantic_sha256=digest("projection"),
        long_margin_ratio=Decimal("0.1"),
        short_margin_ratio=Decimal("0.05"),
        long_notional_ratio=Decimal("0.1"),
        short_notional_ratio=Decimal("0.05"),
        confidence=Decimal("1"),
        projection_accepted=True,
        projection_reasons=(),
        projection_chain_sha256=digest("chain"),
        planner_profile_sha256=digest("planner"),
        input_state_sha256=digest("input"),
        planning_sha256=digest("planning"),
        execution_sha256=digest("execution"),
        reconciliation_digest=digest("reconciliation"),
        evidence_digest=digest("evidence"),
        safety_allows_reduce=True,
        safety_allows_new_risk=True,
        status=ClosedLoopCycleStatus.COMMITTED,
        writes_attempted=1,
        previous_record_sha256=ZERO_HASH,
    )


def test_closed_loop_payload_rejects_truthy_string_boolean() -> None:
    payload = _closed_loop_record().payload()
    payload["safety_allows_new_risk"] = "false"
    with pytest.raises(TypeError, match="safety_allows_new_risk"):
        ClosedLoopCycleRecord.from_payload(payload)


def test_freqai_hedge_rl_is_live_critical_source() -> None:
    assert _is_live_critical("freqtrade/freqai/hedge_rl/risk_bridge.py")


def test_observed_margin_beyond_highest_tier_blocks_fresh_projection() -> None:
    profile = RiskLevelProfile()
    account = replace(
        RiskAccountState.initial(1000.0),
        long=RiskLegState(LegSide.LONG, quantity=5.0, average_price=100.0),
        long_level=4,
    )
    assert _observed_margin_over_cap(account, mark=100.0, profile=profile)


def test_production_armed_rejects_memory_state_backend() -> None:
    with pytest.raises(Exception, match="durable"):
        ProductionMainLoopConfig(
            mode=HedgeExecutionMode.HEDGE_PRODUCTION_ARMED,
            authoritative_execution_enabled=True,
            state_backend="memory",
        )


def test_planner_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        utc_aware(datetime(2026, 8, 16))


class _NeverGrantBudget:
    def reserve_weight(self, _weight: int) -> SharedWeightDecision:
        return SharedWeightDecision(False, 1000, 60.0, 1)


def test_shared_rate_budget_wait_has_a_deadline() -> None:
    adapter = object.__new__(BinanceUSDMExecutionAdapter)
    adapter._shared_weight_budget = _NeverGrantBudget()
    adapter._shared_weight_wait_timeout = 1.0
    clock = iter((0.0, 2.0))
    adapter._monotonic = lambda: next(clock)
    adapter._sleep = lambda _seconds: None
    with pytest.raises(BinanceExecutionApiError, match="deadline"):
        adapter._reserve_shared_weight("/fapi/v1/order")
