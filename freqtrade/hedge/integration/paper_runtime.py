"""Runnable hedge paper loop joining planning, risk and fake execution.

This module is the operational composition missing from the five independent
feature directions.  It keeps planner state between cycles, rebuilds the
planner wallet from actual execution fills and uses the direction-three risk
engine before every direction-five submission.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from threading import RLock
from typing import Any, cast

from freqtrade.hedge.business_reconciliation import (
    business_reconciliation_log_payload,
    reconcile_business_state,
)
from freqtrade.hedge.contracts.ports import (
    EventPublisherPort,
    MarketRulesPort,
    PositionLockPort,
    ReadinessGatePort,
    SingleWriterPort,
)
from freqtrade.hedge.control.dryrun import DryRunControlState
from freqtrade.hedge.execution.action_group_store import ActionGroupRepository
from freqtrade.hedge.execution.event_publisher import InMemoryEventPublisher
from freqtrade.hedge.execution.idempotency import IdempotencyPort
from freqtrade.hedge.execution.integrated_fake import (
    IntegratedFakeRuntime,
    build_integrated_fake_runtime,
)
from freqtrade.hedge.execution.ledger import InMemoryExecutionLedger
from freqtrade.hedge.execution.planner_adapter import adapt_planner_intents
from freqtrade.hedge.execution.service import (
    ExecutionBlockedError,
    ExecutionResult,
    ExecutionStorePort,
    InMemoryExecutionStore,
)
from freqtrade.hedge.execution.service import (
    IntentAction as ExecutionAction,
)
from freqtrade.hedge.execution.service import (
    PositionSide as ExecutionSide,
)
from freqtrade.hedge.integration.business_identity import BusinessIdentityBinder
from freqtrade.hedge.integration.candle_cursor import bar_fingerprint
from freqtrade.hedge.integration.paper_events import (
    NullPaperAccountEventSink,
    NullPaperExecutionRecovery,
    PaperAccountEventSink,
    PaperExecutionRecoveryPort,
    fee_account_event,
)
from freqtrade.hedge.integration.paper_matching import PaperMatchingMixin
from freqtrade.hedge.integration.paper_projection import PaperStateProjectionMixin, _BucketState
from freqtrade.hedge.integration.paper_publisher import PaperRuntimePublisherMixin
from freqtrade.hedge.integration.paper_recovery import PaperRecoveryMixin
from freqtrade.hedge.integration.paper_risk_gate import apply_new_risk_gate
from freqtrade.hedge.integration.paper_serialization import PaperSerializationMixin
from freqtrade.hedge.integration.paper_state import NullPaperStateStore, PaperStateStore
from freqtrade.hedge.integration.risk_adapter import PortfolioRiskApprovalAdapter
from freqtrade.hedge.native.admission import (
    AdmissionProvider,
    CompositeAdmissionPolicy,
    apply_planning_admission_gate,
)
from freqtrade.hedge.native.exit_overlay import NativeExitOverlay, policies_from_config
from freqtrade.hedge.native.exits import HedgeExitPolicyEngine
from freqtrade.hedge.numeric import ZERO
from freqtrade.hedge.operations.config import operations_config
from freqtrade.hedge.operations.runtime import DryRunOperationsRuntime, OperationsCycleInput
from freqtrade.hedge.paper_config import PaperOhlcvSource, PaperSimulationConfig
from freqtrade.hedge.planning.context import (
    IntentAction,
    MarketSnapshot,
    PlannerConfig,
    PlanningContext,
    PlanningResult,
    PositionBucket,
    PositionSide,
    StrategyLegState,
    WalletSnapshot,
)
from freqtrade.hedge.planning.context import (
    OrderIntent as PlannerOrderIntent,
)
from freqtrade.hedge.planning.ideal_orders import PureHedgePlanner
from freqtrade.hedge.risk.engine import HedgeRiskEngine
from freqtrade.hedge.risk.limits import RiskLimits
from freqtrade.hedge.simulation.exchange import AccountEvent, BarEvent, FundingEvent
from freqtrade.hedge.simulation.matcher import ConservativeMatcher, MatchConfig
from freqtrade.hedge.strategies.contract import (
    StrategyDirective,
    planner_config_for_directive,
    target_net_quantity_for_directive,
)
from freqtrade.hedge.symbols import raw_symbol
from freqtrade.hedge.telemetry.dryrun import (
    DryRunCycleTelemetry,
    DryRunTelemetryStore,
    JsonlDryRunTelemetryStore,
    StrategyTelemetry,
)


logger = logging.getLogger(__name__)


ONE = Decimal(1)


def _decimal(value: object, default: str) -> Decimal:
    if value is None:
        return Decimal(default)
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite():
        raise ValueError("paper runtime decimal configuration must be finite")
    return result


def planner_config_from_mapping(values: Mapping[str, Any] | None) -> PlannerConfig:
    raw = dict(values or {})
    aliases = {
        "qty_scale": "grid_qty_growth",
        "grid_initial_distance": "trailing_trigger_distance",
    }
    for old_name, new_name in aliases.items():
        if old_name not in raw:
            continue
        if new_name in raw and raw[new_name] != raw[old_name]:
            raise ValueError(f"hedge.planner.{old_name} conflicts with hedge.planner.{new_name}")
        raw[new_name] = raw.pop(old_name)
    fields = PlannerConfig.__dataclass_fields__
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ValueError("unknown hedge.planner option(s): " + ", ".join(unknown))
    converted: dict[str, object] = {}
    for name, value in raw.items():
        default = fields[name].default
        if isinstance(default, Decimal):
            if isinstance(value, bool):
                raise TypeError(f"hedge.planner.{name} must be an exact decimal")
            converted[name] = Decimal(str(value))
        elif isinstance(default, int) and not isinstance(default, bool):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"hedge.planner.{name} must be an integer")
            converted[name] = value
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise TypeError(f"hedge.planner.{name} must be a boolean")
            converted[name] = value
        else:
            converted[name] = value
    return PlannerConfig(**cast(Any, converted))


def _risk_limits(values: Mapping[str, Any], initial_balance: Decimal) -> RiskLimits:
    max_gross = values.get("max_gross_notional")
    max_ratio = values.get("max_gross_exposure_ratio", "0.80")
    max_single = values.get("max_single_order_notional")
    if max_single is None:
        max_single = initial_balance * Decimal("0.25")
    return RiskLimits(
        max_margin_utilization=_decimal(values.get("max_margin_utilization"), "0.80"),
        min_liquidation_buffer_ratio=_decimal(values.get("min_liquidation_buffer_ratio"), "0.05"),
        max_gross_notional=(None if max_gross is None else _decimal(max_gross, "0")),
        max_gross_exposure_ratio=(None if max_ratio is None else _decimal(max_ratio, "0.80")),
        max_single_order_notional=_decimal(max_single, "0"),
    )


@dataclass(frozen=True, slots=True)
class PaperCycleResult:
    planning: PlanningResult
    executions: tuple[ExecutionResult, ...]
    fills: tuple[ExecutionResult, ...]
    cancellations: tuple[ExecutionResult, ...]
    wallet: WalletSnapshot
    account_events: tuple[AccountEvent, ...] = ()


class IntegratedPaperHedgeApplication(
    PaperStateProjectionMixin,
    PaperMatchingMixin,
    PaperSerializationMixin,
    PaperRecoveryMixin,
    PaperRuntimePublisherMixin,
):
    """Stateful paper application suitable for the real Freqtrade process loop."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        account_id: str,
        symbol: str,
        publisher: object | None = None,
        execution_runtime: IntegratedFakeRuntime | None = None,
        build_execution: bool = True,
        state_store: PaperStateStore | None = None,
        account_event_sink: PaperAccountEventSink | None = None,
        execution_recovery: PaperExecutionRecoveryPort | None = None,
    ) -> None:
        hedge_raw = config.get("hedge", {})
        hedge = dict(hedge_raw) if isinstance(hedge_raw, Mapping) else {}
        planner_values = hedge.get("planner", {})
        if not isinstance(planner_values, Mapping):
            planner_values = {}
        self.paper_config = PaperSimulationConfig.from_hedge_mapping(hedge)
        self.account_id = account_id
        self.symbol = symbol
        self.execution_symbol = raw_symbol(symbol)
        self.initial_balance = self.paper_config.initial_balance
        self.leverage = self.paper_config.leverage
        self.auto_fill = self.paper_config.auto_fill
        self.fill_model = self.paper_config.fill_model
        self.default_long_signal = self.paper_config.default_long_signal
        self.default_short_signal = self.paper_config.default_short_signal
        self.planner_config = planner_config_from_mapping(planner_values)
        dashboard_raw = hedge.get("dashboard", {})
        dashboard = dict(dashboard_raw) if isinstance(dashboard_raw, Mapping) else {}
        self.dashboard_enabled = bool(dashboard.get("enabled", False))
        telemetry_capacity = int(dashboard.get("telemetry_capacity", 2000))
        telemetry_backend = str(dashboard.get("telemetry_backend", "memory")).lower()
        self.telemetry: DryRunTelemetryStore | JsonlDryRunTelemetryStore
        if telemetry_backend == "jsonl":
            telemetry_path = str(
                dashboard.get("telemetry_path", "user_data/hedge/telemetry/dryrun_cycles.jsonl")
            )
            self.telemetry = JsonlDryRunTelemetryStore(telemetry_path, capacity=telemetry_capacity)
        else:
            self.telemetry = DryRunTelemetryStore(telemetry_capacity)
        control_path = dashboard.get("control_state_path")
        self.dryrun_control = DryRunControlState(
            None if control_path in {None, ""} else str(control_path)
        )
        operations_values = dict(operations_config(config))
        self.operations_error: str | None = None
        self.operations: DryRunOperationsRuntime | None = None
        if bool(operations_values.get("enabled", False)):
            state_path = operations_values.get(
                "state_path",
                "user_data/hedge/operations/runtime-state.json",
            )
            self.operations = DryRunOperationsRuntime(
                account_id=account_id,
                symbols=(symbol,),
                config={"hedge": {"operations": operations_values}},
                state_path=None if state_path in {None, ""} else str(state_path),
            )
        self._new_risk_enabled_providers: list[Callable[[], bool]] = []
        self._order_admission_policy = CompositeAdmissionPolicy()
        self._intent_transformers: list[Callable[[object], object]] = []
        self._business_identity_binder: BusinessIdentityBinder | None = None
        self._fill_observers: list[Callable[[object, object, object, object], None]] = []
        self.add_new_risk_provider(
            lambda: (
                self.dryrun_control.snapshot().new_risk_enabled
                and (
                    self.operations is None
                    or (
                        self.operations_error is None
                        and self.operations.latest is not None
                        and self.operations.latest.new_risk_enabled
                    )
                )
            )
        )
        self.planner = PureHedgePlanner()
        self._exit_overlay = NativeExitOverlay(HedgeExitPolicyEngine(policies_from_config(config)))
        self.long_state = StrategyLegState(PositionSide.LONG)
        self.short_state = StrategyLegState(PositionSide.SHORT)
        self._bucket = {
            PositionSide.LONG: _BucketState(),
            PositionSide.SHORT: _BucketState(),
        }
        self._planner_order_to_client: dict[str, str] = {}
        self._simulation_intents: dict[str, PlannerOrderIntent] = {}
        self._last_market: MarketSnapshot | None = None
        self._last_bar: BarEvent | None = None
        self._cycle_market: MarketSnapshot | None = None
        self._lock = RLock()
        self._state_store = state_store or NullPaperStateStore()
        self._state_loaded = False
        self._requires_restart = False
        self._state_durable = not isinstance(self._state_store, NullPaperStateStore)
        self._account_event_sink = account_event_sink or NullPaperAccountEventSink()
        self._execution_recovery = execution_recovery or NullPaperExecutionRecovery()
        self._applied_account_event_ids: set[str] = set()
        self._funding_balance_delta = ZERO
        self._last_funding_event_time: datetime | None = None
        self._paper_fee_rate = self.paper_config.taker_fee_rate
        self._bar_volume = self.paper_config.bar_volume
        self.matcher = ConservativeMatcher(
            MatchConfig(
                maker_fee_rate=self.paper_config.maker_fee_rate,
                taker_fee_rate=self.paper_config.taker_fee_rate,
                volume_participation=self.paper_config.volume_participation,
                market_slippage_bps=self.paper_config.market_slippage_bps,
                price_tick=self.paper_config.tick_size,
                qty_step=self.paper_config.qty_step,
                min_fill_qty=self.paper_config.min_qty,
                min_fill_notional=self.paper_config.min_notional,
                max_entry_layers_per_bar=self.paper_config.max_entry_layers_per_bar,
                max_reduce_layers_per_bar=self.paper_config.max_reduce_layers_per_bar,
                max_fill_ratio_per_order=self.paper_config.max_fill_ratio_per_order,
                max_fills_per_bar=self.paper_config.max_fills_per_bar,
            )
        )
        self.execution: IntegratedFakeRuntime | None = execution_runtime
        if self.execution is None and build_execution:
            risk_engine = HedgeRiskEngine(_risk_limits(hedge, self.initial_balance))
            risk = PortfolioRiskApprovalAdapter(
                engine=risk_engine,
                portfolio_provider=self.risk_portfolio,
            )
            self.execution = build_integrated_fake_runtime(
                risk=risk,
                publisher=cast(EventPublisherPort | None, publisher),
                fee_rate=self._paper_fee_rate,
            )
        if self.execution is not None:
            self._restore_state()

    def bind_execution(
        self,
        *,
        risk: object,
        readiness: ReadinessGatePort,
        single_writer: SingleWriterPort,
        position_lock: PositionLockPort,
        market_rules: MarketRulesPort,
        publisher: object | None = None,
        action_groups: ActionGroupRepository | None = None,
        transaction: object | None = None,
        store: ExecutionStorePort | None = None,
        idempotency: IdempotencyPort[ExecutionResult] | None = None,
        account_event_sink: PaperAccountEventSink | None = None,
        business_identity_binder: BusinessIdentityBinder | None = None,
    ) -> None:
        """Bind the authoritative direction-three/direction-five graph exactly once."""

        with self._lock:
            if self.execution is not None:
                raise RuntimeError("paper execution runtime is already bound")
            transaction_port = transaction or InMemoryExecutionLedger()
            event_publisher = publisher or InMemoryEventPublisher()
            self.execution = build_integrated_fake_runtime(
                risk=risk,  # type: ignore[arg-type]
                publisher=event_publisher,  # type: ignore[arg-type]
                readiness=readiness,
                single_writer=single_writer,
                position_lock=position_lock,
                market_rules=market_rules,
                transaction=transaction_port,  # type: ignore[arg-type]
                action_groups=action_groups,
                store=store,
                idempotency=idempotency,
                fee_rate=self._paper_fee_rate,
                strict_dependencies=True,
                require_business_identity=business_identity_binder is not None,
            )
            if account_event_sink is not None:
                self._account_event_sink = account_event_sink
            self._business_identity_binder = business_identity_binder
            self._restore_state()

    def _execution(self) -> IntegratedFakeRuntime:
        if self.execution is None:
            raise RuntimeError("paper execution runtime has not been bound")
        return self.execution

    def bind_new_risk_provider(self, provider: Callable[[], bool]) -> None:
        """Add a control-plane gate to Paper order submission.

        The runtime composes the built-in Dry-run operations gate with external
        control-plane providers instead of overwriting either authority. Providers
        are now composed with logical AND so official
        bot state, the Hedge control plane, readiness and the production-equivalent loop
        must all agree before new risk is allowed.
        """

        self.add_new_risk_provider(provider)

    def add_new_risk_provider(self, provider: Callable[[], bool]) -> None:
        if not callable(provider):
            raise TypeError("new-risk provider must be callable")
        self._new_risk_enabled_providers.append(provider)

    @property
    def new_risk_provider_count(self) -> int:
        return len(self._new_risk_enabled_providers)

    def bind_order_admission_provider(self, provider: AdmissionProvider) -> None:
        """Add a side-aware admission provider for every planned submission."""

        self._order_admission_policy.add(provider)

    @property
    def order_admission_provider_count(self) -> int:
        return self._order_admission_policy.provider_count

    def bind_order_transformer(self, transformer: Callable[[object], object]) -> None:
        if not callable(transformer):
            raise TypeError("order transformer must be callable")
        self._intent_transformers.append(transformer)

    def add_fill_observer(self, observer: Callable[[object, object, object, object], None]) -> None:
        if not callable(observer):
            raise TypeError("fill observer must be callable")
        self._fill_observers.append(observer)

    def _transform_planning_intents(self, planning: PlanningResult) -> PlanningResult:
        if not self._intent_transformers:
            return planning
        transformed: list[PlannerOrderIntent] = []
        for item in planning.submit_orders:
            current: object = item
            for transformer in tuple(self._intent_transformers):
                current = transformer(current)
            if not isinstance(current, PlannerOrderIntent):
                raise TypeError("Paper order transformer must return PlannerOrderIntent")
            transformed.append(current)
        return replace(planning, submit_orders=tuple(transformed))

    def _notify_fill_observers(
        self, planner_intent: object, price: object, quantity: object, at: object
    ) -> None:
        for observer in tuple(self._fill_observers):
            try:
                observer(planner_intent, price, quantity, at)
            except Exception as exc:
                # Execution/fill authority must never be rolled back by notifications.
                logger.warning("fill observer failed: %s", exc)
                continue

    def new_risk_enabled(self) -> bool:
        for provider in tuple(self._new_risk_enabled_providers):
            try:
                value = provider()
            except Exception as exc:
                raise RuntimeError("Paper new-risk provider failed closed") from exc
            if not isinstance(value, bool):
                raise TypeError("Paper new-risk provider must return bool")
            if not value:
                return False
        return True

    def cancel_managed_orders(self) -> tuple[tuple[ExecutionResult, ...], tuple[str, ...]]:
        """Cancel every Paper order owned by the Hedge planner.

        This is the Hedge equivalent of ``cancel_open_orders_on_exit``.  It never
        touches external exchange orders because the mapping contains only order IDs
        created by this Paper application's planner/execution graph.
        """

        canceled: list[ExecutionResult] = []
        errors: list[str] = []
        with self._lock:
            execution = self._execution()
            for planner_id, client_id in tuple(self._planner_order_to_client.items()):
                terminal = False
                try:
                    canceled.append(execution.engine.cancel(client_id))
                    terminal = True
                except (KeyError, ValueError):
                    # Already terminal or absent is idempotent success for shutdown.
                    terminal = True
                except Exception as exc:
                    errors.append(f"{client_id}:{type(exc).__name__}:{exc}")
                if terminal:
                    self._simulation_intents.pop(client_id, None)
                    self._planner_order_to_client.pop(planner_id, None)
        return tuple(canceled), tuple(errors)

    @property
    def last_funding_event_time(self) -> datetime | None:
        return self._last_funding_event_time

    @property
    def last_market(self) -> MarketSnapshot | None:
        with self._lock:
            return self._last_market

    @property
    def last_bar(self) -> BarEvent | None:
        with self._lock:
            return self._last_bar

    @property
    def last_bar_fingerprint(self) -> str | None:
        with self._lock:
            return None if self._last_bar is None else bar_fingerprint(self._last_bar)

    def _restore_authoritative_fills(self) -> bool:
        recover = getattr(self._execution_recovery, "recover_fills", None)
        if not callable(recover):
            return False
        fills = recover()
        if fills is None:
            return False
        execution = self._execution()
        execution.account.restore(())
        self._bucket = {
            PositionSide.LONG: _BucketState(),
            PositionSide.SHORT: _BucketState(),
        }
        remember = getattr(execution.exchange, "remember_fill_identity", None)
        for fill in fills:
            side = ExecutionSide(fill.position_side)
            action = ExecutionAction(fill.action)
            execution.account.apply_fill(
                trade_id=fill.trade_id,
                account_id=self.account_id,
                symbol=self.execution_symbol,
                position_side=side,
                action=action,
                quantity=fill.quantity,
                price=fill.price,
                fee_amount=fill.fee,
            )
            bucket = PositionBucket(fill.bucket)
            bucket_state = self._bucket[PositionSide(fill.position_side)]
            if action in {ExecutionAction.OPEN, ExecutionAction.INCREASE}:
                bucket_state.increase(
                    bucket,
                    fill.quantity,
                    fill.price,
                    fill.event_time,
                    business_identity=fill.business_identity,
                )
            else:
                bucket_state.reduce(
                    bucket, fill.quantity, business_identity=fill.business_identity
                )
            if callable(remember):
                remember(
                    trade_id=fill.trade_id,
                    client_order_id=fill.client_order_id,
                    quantity=fill.quantity,
                    price=fill.price,
                    fee=fill.fee,
                    fee_currency=fill.fee_currency,
                )
            # A process may terminate after the immutable SQL fill commits but
            # before its fee account-event/outbox transaction. Reconcile the
            # deterministic fee event from the authoritative fill fact during
            # recovery without applying the cash effect twice.
            if fill.fee > ZERO:
                self._record_account_event(
                    fee_account_event(
                        fill_event_id=fill.trade_id,
                        timestamp=fill.event_time,
                        symbol=self.symbol,
                        amount=fill.fee,
                        position_side=PositionSide(fill.position_side),
                    )
                )
        return True

    def _restore_state(self) -> None:
        if self._state_loaded:
            return
        payload = self._state_store.load()
        self._state_loaded = True
        if payload is None:
            self._restore_account_events()
            self._restore_authoritative_execution_orders()
            self._restore_authoritative_fills()
            return
        if payload.get("account_id") != self.account_id or payload.get("symbol") != self.symbol:
            raise ValueError("paper state identity does not match configured account/symbol")
        execution = self._execution()
        execution.account.restore(payload.get("account_legs", ()))
        self._restore_buckets(payload)
        self._restore_checkpoint(payload)
        self._restore_account_events()
        self._planner_order_to_client.clear()
        self._simulation_intents.clear()
        if not self._restore_authoritative_execution_orders():
            self._restore_active_execution_orders(payload.get("active_orders"))
        self._restore_authoritative_fills()

    def _persist_state(
        self,
        *,
        committed_market: MarketSnapshot | None = None,
        committed_bar: BarEvent | None = None,
    ) -> None:
        execution = self._execution()
        payload = {
            "account_id": self.account_id,
            "symbol": self.symbol,
            "account_legs": list(execution.account.snapshot()),
            "buckets": {
                side.value: {
                    "core_quantity": str(state.core_quantity),
                    "core_average": str(state.core_average),
                    "core_opened_at": (
                        None if state.core_opened_at is None else state.core_opened_at.isoformat()
                    ),
                    "tactical_quantity": str(state.tactical_quantity),
                    "tactical_average": str(state.tactical_average),
                    "tactical_opened_at": (
                        None
                        if state.tactical_opened_at is None
                        else state.tactical_opened_at.isoformat()
                    ),
                    "business_lots": state.encode_business_lots(),
                }
                for side, state in self._bucket.items()
            },
            "long_state": self._encode_leg_state(self.long_state),
            "short_state": self._encode_leg_state(self.short_state),
            "last_market": self._encode_market(committed_market or self._last_market),
            "last_bar": self._encode_bar(committed_bar or self._last_bar),
            "funding_balance_delta": str(self._funding_balance_delta),
            "last_funding_event_time": (
                None
                if self._last_funding_event_time is None
                else self._last_funding_event_time.isoformat()
            ),
            "applied_account_event_ids": sorted(self._applied_account_event_ids),
            "execution_state_authority": (
                "checkpoint" if isinstance(execution.store, InMemoryExecutionStore) else "sql"
            ),
            "active_orders": (
                self._encode_active_execution_orders()
                if isinstance(execution.store, InMemoryExecutionStore)
                else []
            ),
            "pending_orders_recovered": True,
        }
        try:
            self._state_store.save(payload)
        except Exception:
            # Business facts may already be committed to SQL, while planner
            # temporal state still relies on the checkpoint. Continuing in the
            # same process would mix pre- and post-failure memory. Force a clean
            # restart so recovery can converge from durable facts.
            self._requires_restart = True
            raise

    def run_market_cycle(  # noqa: C901 - transactional orchestration boundary
        self,
        market: MarketSnapshot,
        *,
        bar: BarEvent | None = None,
        funding_events: tuple[FundingEvent, ...] = (),
        long_signal: Decimal | None = None,
        short_signal: Decimal | None = None,
        target_net_quantity: Decimal | None = None,
        target_net_ratio: Decimal | None = None,
        confidence: Decimal = ONE,
        risk_scale: Decimal = ONE,
        long_exposure_scale: Decimal = ONE,
        short_exposure_scale: Decimal = ONE,
        allow_new_risk: bool = True,
        regime: str = "UNSPECIFIED",
        strategy_reason: str = "",
        model_version: str = "strategy",
        maker_fee_rate: Decimal | None = None,
        taker_fee_rate: Decimal | None = None,
    ) -> PaperCycleResult:
        with self._lock:
            if self._requires_restart:
                raise RuntimeError(
                    "Paper runtime requires restart after a failed durable checkpoint"
                )
            if bar is None:
                if self.paper_config.ohlcv_source is not PaperOhlcvSource.TICKER_COMPAT:
                    raise ValueError(
                        "production Paper cycles require the closed DataProvider BarEvent"
                    )
                previous = self._last_market
                open_price = market.mark if previous is None else previous.mark
                bar = BarEvent(
                    timestamp=market.timestamp,
                    symbol=market.symbol,
                    open=open_price,
                    high=max(open_price, market.bid, market.ask, market.mark),
                    low=min(open_price, market.bid, market.ask, market.mark),
                    close=market.mark,
                    volume=self._bar_volume,
                )
            if bar.symbol != market.symbol or bar.timestamp != market.timestamp:
                raise ValueError("Paper market and OHLCV bar identity must match")
            if bar.close != market.mark:
                raise ValueError("Paper planning mark must equal the analyzed candle close")
            previous_market = self._last_market
            if previous_market is not None and market.timestamp <= previous_market.timestamp:
                relation = (
                    "duplicate" if market.timestamp == previous_market.timestamp else "out-of-order"
                )
                raise ValueError(
                    f"Paper refused {relation} candle {market.timestamp.isoformat()} "
                    f"after {previous_market.timestamp.isoformat()}"
                )

            try:
                self._cycle_market = market
                self._update_market_rules(
                    market,
                    maker_fee_rate=maker_fee_rate,
                    taker_fee_rate=taker_fee_rate,
                )
                account_events: list[AccountEvent] = list(
                    self._apply_funding_events(funding_events)
                )
                # Match only orders accepted before this candle.  New orders are
                # created at the analyzed candle close and become eligible on the
                # next bar, eliminating same-candle look-ahead in live Paper.
                fills: list[ExecutionResult] = []
                cancellations: list[ExecutionResult] = []
                if self.auto_fill and self.fill_model != "instant":
                    matched, expired, matched_events = self._match_active_orders(market, bar)
                    fills.extend(matched)
                    cancellations.extend(expired)
                    account_events.extend(matched_events)

                wallet_before = self.wallet(market)
                directive = StrategyDirective(
                    long_score=self.default_long_signal if long_signal is None else long_signal,
                    short_score=self.default_short_signal if short_signal is None else short_signal,
                    target_net_quantity=(
                        None if target_net_ratio is not None else target_net_quantity
                    ),
                    target_net_ratio=target_net_ratio,
                    confidence=confidence,
                    risk_scale=risk_scale,
                    long_exposure_scale=long_exposure_scale,
                    short_exposure_scale=short_exposure_scale,
                    allow_new_risk=allow_new_risk,
                    regime=regime,
                    reason=strategy_reason,
                    model_version=model_version,
                )
                effective_config = planner_config_for_directive(
                    self.planner_config, cast(Any, directive)
                )
                effective_target = target_net_quantity_for_directive(
                    directive=cast(Any, directive),
                    base=self.planner_config,
                    equity=wallet_before.equity,
                    mark_price=market.mark,
                )
                context = PlanningContext(
                    market=market,
                    wallet=wallet_before,
                    config=effective_config,
                    long_state=self.long_state,
                    short_state=self.short_state,
                    long_signal=directive.long_score,
                    short_signal=directive.short_score,
                    target_net_quantity=effective_target,
                )
                planning = self.planner.plan(context)
                planning, _native_exit_diagnostics = self._exit_overlay.apply(
                    planning, app=self, market=market
                )
                planning = self._transform_planning_intents(planning)
                planning, _blocked_new_risk = apply_new_risk_gate(
                    planning,
                    enabled=self.new_risk_enabled() and directive.allow_new_risk,
                    current_long_state=self.long_state,
                    current_short_state=self.short_state,
                )
                planning_result, native_admission_blocks = apply_planning_admission_gate(
                    planning,
                    evaluate=self._order_admission_policy.evaluate,
                    current_long_state=self.long_state,
                    current_short_state=self.short_state,
                )
                planning = cast(PlanningResult, planning_result)
                _blocked_new_risk += len(native_admission_blocks)
                if self._business_identity_binder is not None:
                    planning = self._business_identity_binder.bind_planning_result(
                        planning,
                        active_orders=wallet_before.active_orders,
                    )
                for planner_order_id in planning.cancel_order_ids:
                    client_id = self._planner_order_to_client.pop(planner_order_id, None)
                    if client_id is None:
                        continue
                    try:
                        cancellations.append(self._execution().engine.cancel(client_id))
                    finally:
                        self._simulation_intents.pop(client_id, None)

                execution_intents = adapt_planner_intents(
                    planning.submit_orders,
                    account_id=self.account_id,
                    exchange="paper",
                    strategy_id="pure-hedge-planner",
                    cycle_id=market.timestamp.isoformat(),
                    require_business_identity=(
                        self._business_identity_binder is not None
                    ),
                )
                executions: list[ExecutionResult] = []
                for planner_intent, execution_intent in zip(
                    planning.submit_orders,
                    execution_intents,
                    strict=True,
                ):
                    try:
                        result = self._execution().engine.submit(execution_intent)
                    except ExecutionBlockedError:
                        continue
                    executions.append(result)
                    client_id = result.order.client_order_id
                    self._planner_order_to_client[planner_intent.intent_id] = client_id
                    self._simulation_intents[client_id] = planner_intent

                if self.auto_fill and self.fill_model == "instant":
                    for result in executions:
                        fill_price = result.order.intent.limit_price or market.mark
                        snapshot = self._execution().exchange.fill_order(
                            result.order.client_order_id,
                            quantity=result.order.approved_quantity,
                            price=fill_price,
                        )
                        applied = self._execution().engine.apply_exchange_event(snapshot)
                        fills.append(applied)
                        trade_id = snapshot.exchange_trade_id or result.order.client_order_id
                        fee = result.order.approved_quantity * fill_price * self._paper_fee_rate
                        fee_event = fee_account_event(
                            fill_event_id=trade_id,
                            timestamp=bar.timestamp,
                            symbol=market.symbol,
                            amount=fee,
                            position_side=PositionSide(result.order.intent.position_side.value),
                        )
                        if self._record_account_event(fee_event):
                            account_events.append(fee_event)
                        simulation_intent = self._simulation_intents.get(
                            result.order.client_order_id
                        )
                        if simulation_intent is not None:
                            self._simulation_intents.pop(result.order.client_order_id, None)
                        if simulation_intent is not None:
                            self._notify_fill_observers(
                                simulation_intent,
                                fill_price,
                                result.order.approved_quantity,
                                bar.timestamp,
                            )
                            state = self._bucket[simulation_intent.position_side]
                            if simulation_intent.action in {
                                IntentAction.OPEN,
                                IntentAction.INCREASE,
                            }:
                                state.increase(
                                    simulation_intent.bucket,
                                    result.order.approved_quantity,
                                    fill_price,
                                    bar.timestamp,
                                    business_identity=simulation_intent.business_identity,
                                )
                            else:
                                state.reduce(
                                    simulation_intent.bucket,
                                    result.order.approved_quantity,
                                    business_identity=simulation_intent.business_identity,
                                )

                self.long_state = planning.long_state
                self.short_state = planning.short_state
                self._prune_planner_order_map()
                cycle_result = PaperCycleResult(
                    planning=planning,
                    executions=tuple(executions),
                    fills=tuple(fills),
                    cancellations=tuple(cancellations),
                    wallet=self.wallet(market),
                    account_events=tuple(account_events),
                )
                # The durable candle cursor advances only after every business fact
                # and the auxiliary checkpoint have committed. A failed save leaves
                # the prior cursor intact so the same candle can be retried and
                # converged from SQL facts instead of being silently skipped.
                self._persist_state(committed_market=market, committed_bar=bar)
                self._last_market = market
                self._last_bar = bar
                wallet_after = cycle_result.wallet
                business_reconciliation = None
                if self._business_identity_binder is not None:
                    business_reconciliation = reconcile_business_state(
                        open_lots=(
                            *wallet_after.long.position_lots,
                            *wallet_after.short.position_lots,
                        ),
                        managed_orders=self._active_execution_orders(),
                        remote_long_quantity=self._fake_leg(PositionSide.LONG).quantity,
                        remote_short_quantity=self._fake_leg(PositionSide.SHORT).quantity,
                        amount_tolerance=market.qty_step,
                        account_id=self.account_id,
                        symbol=self.symbol,
                    )
                    if not business_reconciliation.consistent:
                        logger.error(
                            "Paper business identity reconciliation drift",
                            extra=business_reconciliation_log_payload(
                                business_reconciliation
                            ),
                        )
                realized = wallet_after.long.realized_pnl + wallet_after.short.realized_pnl
                fees = (
                    self._fake_leg(PositionSide.LONG).fees + self._fake_leg(PositionSide.SHORT).fees
                )
                cycle_telemetry = DryRunCycleTelemetry(
                    cycle_id=market.timestamp.isoformat(),
                    account_id=self.account_id,
                    symbol=self.symbol,
                    timestamp=market.timestamp,
                    mark_price=market.mark,
                    equity=wallet_after.equity,
                    available_balance=wallet_after.available_balance,
                    gross_notional=wallet_after.gross_notional(market.mark),
                    net_quantity=wallet_after.long.quantity - wallet_after.short.quantity,
                    target_net_quantity=planning.target_net_quantity,
                    net_gap_quantity=planning.net_gap_quantity,
                    long_quantity=wallet_after.long.quantity,
                    short_quantity=wallet_after.short.quantity,
                    long_target_quantity=planning.long_target_quantity,
                    short_target_quantity=planning.short_target_quantity,
                    long_average_price=wallet_after.long.average_price,
                    short_average_price=wallet_after.short.average_price,
                    unrealized_pnl=wallet_after.long.unrealized_pnl(market.mark)
                    + wallet_after.short.unrealized_pnl(market.mark),
                    realized_pnl=realized,
                    funding_pnl=self._funding_balance_delta,
                    fees=fees,
                    ideal_order_count=len(planning.ideal_orders),
                    submit_order_count=len(planning.submit_orders),
                    cancel_order_count=len(planning.cancel_order_ids),
                    fill_count=len(fills),
                    active_order_count=len(wallet_after.active_orders),
                    risk_blocked=bool(_blocked_new_risk),
                    diagnostics=planning.diagnostics,
                    strategy=StrategyTelemetry(
                        long_score=directive.long_score,
                        short_score=directive.short_score,
                        target_net_quantity=directive.target_net_quantity,
                        target_net_ratio=directive.target_net_ratio,
                        confidence=directive.confidence,
                        risk_scale=directive.risk_scale,
                        long_exposure_scale=directive.long_exposure_scale,
                        short_exposure_scale=directive.short_exposure_scale,
                        allow_new_risk=directive.allow_new_risk,
                        regime=directive.regime,
                        reason=directive.reason,
                        model_version=directive.model_version,
                    ),
                )
                self.telemetry.append(cycle_telemetry)
                if self.operations is not None:
                    try:
                        self.operations.observe(
                            OperationsCycleInput(
                                timestamp=market.timestamp,
                                symbol=self.symbol,
                                timeframe_seconds=60,
                                mark_price=market.mark,
                                index_price=market.mark,
                                equity=wallet_after.equity,
                                initial_equity=self.initial_balance,
                                long_notional=wallet_after.long.quantity * market.mark,
                                short_notional=wallet_after.short.quantity * market.mark,
                                margin_used=(
                                    wallet_after.gross_notional(market.mark)
                                    / max(self.leverage, ONE)
                                ),
                                realized_pnl=realized,
                                unrealized_pnl=cycle_telemetry.unrealized_pnl,
                                funding_pnl=self._funding_balance_delta,
                                fees=fees,
                                slippage_cost=ZERO,
                                base_candles=self.operations.session.cycle_sequence + 1,
                                informative_candles={},
                                observed_at=datetime.now(UTC),
                                order_count=len(executions),
                                fill_count=len(fills),
                                active_order_count=len(wallet_after.active_orders),
                                reconciliation_fresh=(
                                    self._state_durable and not self._requires_restart
                                ),
                                api_healthy=self.dashboard_enabled,
                                dashboard_healthy=self.dashboard_enabled,
                                business_reconciliation_consistent=(
                                    None
                                    if business_reconciliation is None
                                    else business_reconciliation.consistent
                                ),
                                managed_order_identity_coverage=(
                                    None
                                    if business_reconciliation is None
                                    else (
                                        business_reconciliation
                                        .managed_order_identity_coverage
                                    )
                                ),
                                business_trade_display_ids=(
                                    ()
                                    if business_reconciliation is None
                                    else business_reconciliation.display_ids
                                ),
                                business_reconciliation_issues=(
                                    ()
                                    if business_reconciliation is None
                                    else business_reconciliation.operation_details()
                                ),
                            )
                        )
                        self.operations_error = None
                    except Exception as exc:
                        self.operations_error = f"{type(exc).__name__}: {exc}"[:512]
                        logger.warning(
                            "Paper operations telemetry update failed",
                            extra={
                                "reason_code": "PAPER_OPERATIONS_TELEMETRY_FAILED",
                                "error_type": type(exc).__name__,
                            },
                        )
                self._cycle_market = None
                return cycle_result
            finally:
                self._cycle_market = None
