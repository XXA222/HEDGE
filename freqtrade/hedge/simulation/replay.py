from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from typing import Any, cast

from freqtrade.hedge.planning.context import (
    ZERO,
    IntentAction,
    MarketSnapshot,
    PlannerConfig,
    PlanningContext,
    PositionSide,
    StrategyLegState,
    StrategyPlanningPort,
    TrailingPhase,
)
from freqtrade.hedge.planning.ideal_orders import PureHedgePlanner
from freqtrade.hedge.planning.trailing import enter_trailing_cooldown
from freqtrade.hedge.planning.unstuck import unstuck_budget_keys
from freqtrade.hedge.strategies.contract import (
    StrategyDirective,
    planner_config_for_directive,
    target_net_quantity_for_directive,
)

from .cross_wallet import CrossWallet
from .exchange import (
    AccountEvent,
    AccountEventType,
    BarEvent,
    FillEvent,
    FundingEvent,
    LiquidationEvent,
    MarketRules,
    OrderAcceptedEvent,
    OrderCancelledEvent,
    SignalEvent,
    SimulationInputEvent,
    SimulationPort,
    SimulationResult,
    StandardEvent,
)
from .funding import FundingEngine
from .matcher import ConservativeMatcher, MatchConfig
from .reports import build_report
from .risk_gate import apply_new_risk_gate


@dataclass(frozen=True, slots=True)
class ReplayCheckpoint:
    config_fingerprint: str
    wallet: CrossWallet
    long_state: StrategyLegState
    short_state: StrategyLegState
    long_signal: Decimal
    short_signal: Decimal
    signal_directive: StrategyDirective
    counter: int
    symbol: str | None
    final_mark: Decimal
    last_event_key: tuple[datetime, int] | None
    processed_slots: frozenset[tuple[str, datetime, str]]
    target_net_quantity: Decimal
    net_gap_quantity: Decimal
    long_target_quantity: Decimal
    short_target_quantity: Decimal

    @property
    def signal_target_net_quantity(self) -> Decimal | None:
        """Backward-compatible alias for the signal directive exact target."""

        return self.signal_directive.target_net_quantity

    @property
    def signal_target_net_ratio(self) -> Decimal | None:
        """Backward-compatible alias for the signal directive target ratio."""

        return self.signal_directive.target_net_ratio


class EventReplayEngine(SimulationPort):
    def __init__(
        self,
        *,
        initial_balance: Decimal,
        planner_config: PlannerConfig | None = None,
        leverage: Decimal = Decimal(3),
        fee_rate: Decimal = Decimal("0.0004"),
        long_signal: Decimal = Decimal(1),
        short_signal: Decimal = Decimal(1),
        target_net_quantity: Decimal | None = None,
        market_rules: MarketRules | None = None,
        planner: StrategyPlanningPort | None = None,
        match_config: MatchConfig | None = None,
    ) -> None:
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.planner_config = planner_config or PlannerConfig()
        self.initial_long_signal = long_signal
        self.initial_short_signal = short_signal
        self.initial_target_net_quantity_override = target_net_quantity
        self.target_net_quantity_override = target_net_quantity
        self.market_rules = market_rules or MarketRules()
        self.planner = planner or PureHedgePlanner()
        if not isinstance(self.planner, StrategyPlanningPort):
            raise TypeError("planner must implement StrategyPlanningPort")
        if match_config is None:
            self.match_config = MatchConfig(
                fee_rate=fee_rate,
                price_tick=self.market_rules.tick_size,
                qty_step=self.market_rules.qty_step,
            )
        else:
            self.match_config = replace(
                match_config,
                price_tick=self.market_rules.tick_size,
                qty_step=self.market_rules.qty_step,
            )
        self._config_fingerprint = self._build_config_fingerprint()
        self.funding = FundingEngine()
        self.matcher = ConservativeMatcher(self.match_config)
        self._reset_runtime()

    def _build_config_fingerprint(self) -> str:
        payload = repr(
            (
                self.initial_balance,
                self.leverage,
                self.fee_rate,
                self.planner_config,
                self.initial_long_signal,
                self.initial_short_signal,
                self.initial_target_net_quantity_override,
                self.market_rules,
                self.match_config,
                type(self.planner).__module__,
                type(self.planner).__qualname__,
            )
        ).encode()
        return sha256(payload).hexdigest()

    def checkpoint(self) -> ReplayCheckpoint:
        import copy

        return ReplayCheckpoint(
            config_fingerprint=self._config_fingerprint,
            wallet=copy.deepcopy(self.wallet),
            long_state=self.long_state,
            short_state=self.short_state,
            long_signal=self.long_signal,
            short_signal=self.short_signal,
            signal_directive=self.strategy_directive,
            counter=self._counter,
            symbol=self._symbol,
            final_mark=self._final_mark,
            last_event_key=self._last_event_key,
            processed_slots=frozenset(self._processed_slots),
            target_net_quantity=self._last_target_net,
            net_gap_quantity=self._last_net_gap,
            long_target_quantity=self._last_long_target,
            short_target_quantity=self._last_short_target,
        )

    def restore(self, checkpoint: ReplayCheckpoint) -> None:
        import copy

        if checkpoint.config_fingerprint != self._config_fingerprint:
            raise ValueError("checkpoint configuration does not match replay engine")
        self.wallet = copy.deepcopy(checkpoint.wallet)
        self.long_state = checkpoint.long_state
        self.short_state = checkpoint.short_state
        self.long_signal = checkpoint.long_signal
        self.short_signal = checkpoint.short_signal
        self.strategy_directive = checkpoint.signal_directive
        self.target_net_quantity_override = checkpoint.signal_directive.target_net_quantity
        self._counter = checkpoint.counter
        self._symbol = checkpoint.symbol
        self._final_mark = checkpoint.final_mark
        self._last_event_key = checkpoint.last_event_key
        self._processed_slots = set(checkpoint.processed_slots)
        self._last_target_net = checkpoint.target_net_quantity
        self._last_net_gap = checkpoint.net_gap_quantity
        self._last_long_target = checkpoint.long_target_quantity
        self._last_short_target = checkpoint.short_target_quantity

    def _reset_runtime(self) -> None:
        self.wallet = CrossWallet(
            initial_balance=self.initial_balance,
            leverage=self.leverage,
            fee_rate=self.fee_rate,
            maintenance_margin_rate=self.planner_config.maintenance_margin_rate,
            liquidation_fee_rate=self.planner_config.liquidation_fee_rate,
            liquidation_buffer_warning_ratio=(self.planner_config.liquidation_buffer_warning_ratio),
        )
        self.long_state = StrategyLegState(side=PositionSide.LONG)
        self.short_state = StrategyLegState(side=PositionSide.SHORT)
        self.long_signal = self.initial_long_signal
        self.short_signal = self.initial_short_signal
        self.target_net_quantity_override = self.initial_target_net_quantity_override
        self.strategy_directive = StrategyDirective(
            long_score=self.initial_long_signal,
            short_score=self.initial_short_signal,
            target_net_quantity=self.initial_target_net_quantity_override,
        )
        self._counter = 0
        self._symbol = None
        self._final_mark = Decimal(1)
        self._last_event_key = None
        self._processed_slots = set()
        self._last_target_net = ZERO
        self._last_net_gap = ZERO
        self._last_long_target = ZERO
        self._last_short_target = ZERO
        self._effective_planner_config_key: tuple[Decimal, Decimal, Decimal, Decimal] | None = None
        self._effective_planner_config = self.planner_config

    def _order_id(self, intent_id: str) -> str:
        self._counter += 1
        digest = sha256(f"{intent_id}|{self._counter}".encode()).hexdigest()[:20]
        return f"sim-{digest}"

    @staticmethod
    def _account_event_id(prefix: str, source_id: str, side: PositionSide | None) -> str:
        raw = f"{prefix}|{source_id}|{side.value if side else ''}".encode()
        return "acct-" + sha256(raw).hexdigest()[:24]

    def _assert_symbol(self, symbol: str) -> None:
        if self._symbol is None:
            self._symbol = symbol
        elif symbol != self._symbol:
            raise ValueError(
                "EventReplayEngine is single-symbol; create one engine per symbol "
                f"({self._symbol!r} != {symbol!r})"
            )

    def _plan(
        self,
        bar: BarEvent,
        *,
        emit_events: bool = True,
        collect_diagnostics: bool = True,
    ) -> tuple[StandardEvent, ...]:
        if self.wallet.liquidated:
            return ()
        rules = self.market_rules
        market = MarketSnapshot(
            symbol=bar.symbol,
            timestamp=bar.timestamp,
            bid=bar.open,
            ask=bar.open,
            mark=bar.open,
            tick_size=rules.tick_size,
            qty_step=rules.qty_step,
            min_qty=rules.min_qty,
            min_notional=rules.min_notional,
        )
        wallet_snapshot = self.wallet.planner_snapshot(bar.open, bar.timestamp)
        directive = self.strategy_directive
        config_key = (
            directive.confidence,
            directive.risk_scale,
            directive.long_exposure_scale,
            directive.short_exposure_scale,
        )
        if config_key != self._effective_planner_config_key:
            self._effective_planner_config = planner_config_for_directive(
                self.planner_config,
                cast(Any, directive),
            )
            self._effective_planner_config_key = config_key
        effective_config = self._effective_planner_config
        effective_target = target_net_quantity_for_directive(
            directive=cast(Any, self.strategy_directive),
            base=self.planner_config,
            equity=wallet_snapshot.equity,
            mark_price=bar.open,
        )
        context = PlanningContext(
            market=market,
            wallet=wallet_snapshot,
            config=effective_config,
            long_state=self.long_state,
            short_state=self.short_state,
            long_signal=self.strategy_directive.long_score,
            short_signal=self.strategy_directive.short_score,
            target_net_quantity=effective_target,
            collect_diagnostics=collect_diagnostics,
        )
        result = self.planner.plan(context)
        result, _ = apply_new_risk_gate(
            result,
            enabled=self.strategy_directive.allow_new_risk,
            current_long_state=self.long_state,
            current_short_state=self.short_state,
        )
        self.long_state = result.long_state
        self.short_state = result.short_state
        self._last_target_net = result.target_net_quantity
        self._last_net_gap = result.net_gap_quantity
        self._last_long_target = result.long_target_quantity
        self._last_short_target = result.short_target_quantity
        emitted: list[StandardEvent] = []
        for order_id in result.cancel_order_ids:
            self.wallet.cancel_order(order_id)
            if emit_events:
                emitted.append(OrderCancelledEvent(timestamp=bar.timestamp, order_id=order_id))
        for intent in result.submit_orders:
            order_id = self._order_id(intent.intent_id)
            self.wallet.accept_order(order_id, intent, accepted_at=bar.timestamp)
            if emit_events:
                emitted.append(
                    OrderAcceptedEvent(
                        timestamp=bar.timestamp,
                        order_id=order_id,
                        intent=intent,
                    )
                )
        return tuple(emitted)

    def _advance_states_after_fills(self, bar: BarEvent, fills) -> None:
        # Without fills the position and active-layer state did not change.  The
        # planner advances trailing state later in the same bar.  Liquidation is
        # the only no-fill transition that still needs the flat-state reset below.
        if not fills and not self.wallet.liquidated:
            return
        touched_entries: dict[PositionSide, set[int]] = {
            PositionSide.LONG: set(),
            PositionSide.SHORT: set(),
        }
        touched_reductions: dict[PositionSide, set[int]] = {
            PositionSide.LONG: set(),
            PositionSide.SHORT: set(),
        }
        any_entry: set[PositionSide] = set()
        any_reduction: set[PositionSide] = set()
        unstuck_losses: dict[PositionSide, Decimal] = {
            PositionSide.LONG: ZERO,
            PositionSide.SHORT: ZERO,
        }
        unstuck_sides: set[PositionSide] = set()
        for fill in fills:
            if fill.reduce_only:
                touched_reductions[fill.position_side].add(fill.layer)
                any_reduction.add(fill.position_side)
            else:
                touched_entries[fill.position_side].add(fill.layer)
                any_entry.add(fill.position_side)
            if fill.action is IntentAction.UNSTUCK:
                unstuck_sides.add(fill.position_side)
                realized = self.wallet.realized_by_fill.get(fill.event_id, ZERO)
                unstuck_losses[fill.position_side] += max(-realized, ZERO) + fill.fee

        def layer_is_complete(side: PositionSide, layer: int, reduce_only: bool) -> bool:
            return not any(
                intent.position_side is side
                and intent.layer == layer
                and intent.reduce_only == reduce_only
                for intent, _ in self.wallet.active_orders.values()
            )

        def next_state(side: PositionSide, state: StrategyLegState) -> StrategyLegState:
            completed_entries = sum(
                1 for layer in touched_entries[side] if layer_is_complete(side, layer, False)
            )
            completed_reductions = sum(
                1 for layer in touched_reductions[side] if layer_is_complete(side, layer, True)
            )
            layer_count = min(
                state.grid_layers_filled + completed_entries,
                self.planner_config.max_grid_layers,
            )
            layer_count = max(layer_count - completed_reductions, 0)
            next_value = replace(state, grid_layers_filled=layer_count)
            if side in any_entry:
                next_value = enter_trailing_cooldown(
                    replace(next_value, last_entry_at=bar.timestamp),
                    timestamp=bar.timestamp,
                    cooldown_seconds=self.planner_config.cooldown_seconds,
                )
            if side in any_reduction:
                next_value = replace(next_value, last_reduce_at=bar.timestamp)
            if side in unstuck_sides:
                day_key, week_key = unstuck_budget_keys(bar.timestamp)
                daily = (
                    next_value.unstuck_daily_loss
                    if next_value.unstuck_budget_day == day_key
                    else ZERO
                )
                weekly = (
                    next_value.unstuck_weekly_loss
                    if next_value.unstuck_budget_week == week_key
                    else ZERO
                )
                next_value = replace(
                    next_value,
                    last_unstuck_at=bar.timestamp,
                    unstuck_budget_day=day_key,
                    unstuck_budget_week=week_key,
                    unstuck_daily_loss=daily + unstuck_losses[side],
                    unstuck_weekly_loss=weekly + unstuck_losses[side],
                )
            if self.wallet.leg(side).quantity == 0:
                next_value = replace(
                    next_value,
                    grid_layers_filled=0,
                    trailing_phase=TrailingPhase.IDLE,
                    trailing_trigger_price=None,
                    trailing_extreme=None,
                    trailing_started_at=None,
                    trailing_confirmed_at=None,
                    trailing_cooldown_until=None,
                    trailing_armed=False,
                )
            return next_value

        self.long_state = next_state(PositionSide.LONG, self.long_state)
        self.short_state = next_state(PositionSide.SHORT, self.short_state)

    @staticmethod
    def _event_priority(event: SimulationInputEvent) -> int:
        if isinstance(event, SignalEvent):
            return 0
        if isinstance(event, FundingEvent):
            return 1
        return 2

    @staticmethod
    def _input_slot(event: SimulationInputEvent) -> tuple[str, datetime, str]:
        if isinstance(event, SignalEvent):
            kind = "SIGNAL"
        elif isinstance(event, FundingEvent):
            kind = "FUNDING"
        else:
            kind = "BAR"
        return kind, event.timestamp, event.symbol

    def _report(self):
        return build_report(
            self.wallet,
            self._final_mark,
            target_net_quantity=self._last_target_net,
            net_gap_quantity=self._last_net_gap,
            long_target_quantity=self._last_long_target,
            short_target_quantity=self._last_short_target,
        )

    def _validate_input_batch(
        self,
        indexed_events: list[tuple[int, SimulationInputEvent]],
    ) -> None:
        prospective_symbol = self._symbol
        prospective_last_key = self._last_event_key
        prospective_slots = set(self._processed_slots)
        for _, event in indexed_events:
            event_key = (event.timestamp, self._event_priority(event))
            if prospective_last_key is not None and event_key < prospective_last_key:
                raise ValueError(
                    "incremental simulation events cannot move backwards in time or priority"
                )
            slot = self._input_slot(event)
            if slot in prospective_slots:
                raise ValueError(
                    f"duplicate standard event slot: {slot[0]} {event.symbol} "
                    f"at {event.timestamp.isoformat()}"
                )
            if prospective_symbol is None:
                prospective_symbol = event.symbol
            elif event.symbol != prospective_symbol:
                raise ValueError(
                    "EventReplayEngine is single-symbol; create one engine per symbol "
                    f"({prospective_symbol!r} != {event.symbol!r})"
                )
            prospective_slots.add(slot)
            prospective_last_key = event_key

    def _run(  # noqa: C901 - deterministic replay orchestration boundary
        self,
        events: Iterable[SimulationInputEvent],
        *,
        reset: bool,
    ) -> SimulationResult:
        if reset:
            self._reset_runtime()
        emitted: list[StandardEvent] = []
        snapshots = []
        indexed_events = list(enumerate(events))
        indexed_events.sort(
            key=lambda item: (
                item[1].timestamp,
                self._event_priority(item[1]),
                item[0],
            )
        )

        self._validate_input_batch(indexed_events)

        for _, event in indexed_events:
            event_key = (event.timestamp, self._event_priority(event))
            slot = self._input_slot(event)
            self._processed_slots.add(slot)
            self._last_event_key = event_key
            self._assert_symbol(event.symbol)
            if isinstance(event, SignalEvent):
                self.long_signal = event.long_signal
                self.short_signal = event.short_signal
                self.target_net_quantity_override = event.target_net
                self.strategy_directive = StrategyDirective(
                    long_score=event.long_signal,
                    short_score=event.short_signal,
                    target_net_quantity=(
                        None if event.target_net_ratio is not None else event.target_net
                    ),
                    target_net_ratio=event.target_net_ratio,
                    confidence=event.confidence,
                    risk_scale=event.risk_scale,
                    long_exposure_scale=event.long_exposure_scale,
                    short_exposure_scale=event.short_exposure_scale,
                    allow_new_risk=event.allow_new_risk,
                    regime=event.regime,
                    reason=event.reason,
                    model_version=event.model_version,
                )
                emitted.append(event)
                continue
            if isinstance(event, FundingEvent):
                before_long = self.wallet.long_funding
                before_short = self.wallet.short_funding
                self.funding.apply(self.wallet, event)
                emitted.append(event)
                for side, amount in (
                    (PositionSide.LONG, self.wallet.long_funding - before_long),
                    (PositionSide.SHORT, self.wallet.short_funding - before_short),
                ):
                    if amount != ZERO:
                        emitted.append(
                            AccountEvent(
                                event_id=self._account_event_id(
                                    "funding",
                                    event.timestamp.isoformat(),
                                    side,
                                ),
                                timestamp=event.timestamp,
                                symbol=event.symbol,
                                event_type=AccountEventType.FUNDING,
                                amount=amount,
                                position_side=side,
                                source_event_id=event.timestamp.isoformat(),
                                description="funding settlement",
                            )
                        )
                self._final_mark = event.mark_price
                snapshots.append(self.wallet.snapshot(event.timestamp, self._final_mark))
                continue

            emitted.append(event)
            # Existing orders are matched first.  The current analyzed candle
            # may produce a new signal and a new ideal order set at its close,
            # but those orders are not allowed to use the same candle's path.
            # This is the shared no-lookahead contract for Backtest and Paper.
            if not self.wallet.liquidated:
                outcome = self.matcher.match_outcome(event, self.wallet)
                for fill in outcome.fills:
                    self.wallet.apply_fill(fill)
                    emitted.append(fill)
                    if fill.fee != ZERO:
                        liquidity_role = getattr(
                            fill.liquidity_role,
                            "value",
                            fill.liquidity_role,
                        )
                        emitted.append(
                            AccountEvent(
                                event_id=self._account_event_id(
                                    "fee",
                                    fill.event_id,
                                    fill.position_side,
                                ),
                                timestamp=fill.timestamp,
                                symbol=fill.symbol,
                                event_type=AccountEventType.FEE,
                                amount=-fill.fee,
                                position_side=fill.position_side,
                                source_event_id=fill.event_id,
                                description=f"{liquidity_role.lower()} fee",
                            )
                        )
                if outcome.liquidation_event is not None:
                    liquidation_cancel_ids = tuple(sorted(self.wallet.active_orders))
                    self.wallet.apply_liquidation(outcome.liquidation_event)
                    for order_id in liquidation_cancel_ids:
                        emitted.append(
                            OrderCancelledEvent(
                                timestamp=outcome.liquidation_event.timestamp,
                                order_id=order_id,
                            )
                        )
                    emitted.append(outcome.liquidation_event)
                    emitted.append(
                        AccountEvent(
                            event_id=self._account_event_id(
                                "liquidation-pnl",
                                outcome.liquidation_event.event_id,
                                None,
                            ),
                            timestamp=outcome.liquidation_event.timestamp,
                            symbol=outcome.liquidation_event.symbol,
                            event_type=AccountEventType.LIQUIDATION,
                            amount=outcome.liquidation_event.realized_pnl,
                            source_event_id=outcome.liquidation_event.event_id,
                            description="cross account liquidation realized PnL",
                        )
                    )
                    if outcome.liquidation_event.fee != ZERO:
                        emitted.append(
                            AccountEvent(
                                event_id=self._account_event_id(
                                    "liquidation-fee",
                                    outcome.liquidation_event.event_id,
                                    None,
                                ),
                                timestamp=outcome.liquidation_event.timestamp,
                                symbol=outcome.liquidation_event.symbol,
                                event_type=AccountEventType.FEE,
                                amount=-outcome.liquidation_event.fee,
                                source_event_id=outcome.liquidation_event.event_id,
                                description="cross account liquidation fee",
                            )
                        )
                for order_id in outcome.expired_order_ids:
                    if self.wallet.remaining(order_id) > 0:
                        self.wallet.cancel_order(order_id)
                        emitted.append(
                            OrderCancelledEvent(timestamp=event.timestamp, order_id=order_id)
                        )
                self.wallet.merge_risk_metrics(
                    gross_peak=outcome.gross_peak,
                    equity_peak=outcome.equity_peak,
                    max_drawdown=outcome.max_drawdown,
                )
                self._advance_states_after_fills(event, outcome.fills)
            if not self.wallet.liquidated:
                emitted.extend(self._plan(event))
            self._final_mark = event.close
            self.wallet.observe_bar_state(event.timestamp, self._final_mark)
            snapshots.append(self.wallet.snapshot(event.timestamp, self._final_mark, observe=False))

        return SimulationResult(
            events=tuple(emitted),
            snapshots=tuple(snapshots),
            report=self._report(),
        )

    def replay_ordered_stream(  # noqa: C901
        self,
        events: Iterable[SimulationInputEvent],
        *,
        retain_material_events: bool = False,
        snapshot_every_bars: int = 1440,
        max_retained_snapshots: int = 2048,
        compact_wallet_history: bool = True,
    ) -> SimulationResult:
        """Replay an already ordered trusted input stream with O(1) history memory.

        This is the long-history/optimization fast path.  It intentionally avoids
        the generic batch path's ``list(enumerate(...))``, sort, historical slot
        set, per-input event ledger and per-bar snapshot allocation.  Matching,
        planning, funding and liquidation semantics remain the same.

        The producer must yield Signal -> Funding -> Bar order for identical
        timestamps.  Duplicate validation is bounded to the current timestamp.
        Durable/live replay continues to use ``replay``/``advance`` and therefore
        retains its stronger historical idempotency state.
        """
        if snapshot_every_bars < 1:
            raise ValueError("snapshot_every_bars must be positive")
        if max_retained_snapshots < 2:
            raise ValueError("max_retained_snapshots must be at least 2")

        self._reset_runtime()
        retained_events: list[StandardEvent] = []
        retained_snapshots = []
        input_event_count = 0
        input_bar_count = 0
        previous_key: tuple[datetime, int] | None = None
        current_timestamp: datetime | None = None
        current_slot_mask = 0
        flat_idle_matcher_bypass_count = 0
        sample_stride = snapshot_every_bars

        for event in events:
            input_event_count += 1
            priority = self._event_priority(event)
            key = (event.timestamp, priority)
            if previous_key is not None and key < previous_key:
                raise ValueError("ordered simulation stream moved backwards in time or priority")
            self._assert_symbol(event.symbol)
            if current_timestamp != event.timestamp:
                current_timestamp = event.timestamp
                current_slot_mask = 0
            slot_bit = 1 << priority
            if current_slot_mask & slot_bit:
                kind = "SIGNAL" if priority == 0 else "FUNDING" if priority == 1 else "BAR"
                raise ValueError(
                    f"duplicate standard event slot: {kind} {event.symbol} "
                    f"at {event.timestamp.isoformat()}"
                )
            current_slot_mask |= slot_bit
            previous_key = key
            self._last_event_key = key

            if isinstance(event, SignalEvent):
                self.long_signal = event.long_signal
                self.short_signal = event.short_signal
                self.target_net_quantity_override = event.target_net
                self.strategy_directive = StrategyDirective(
                    long_score=event.long_signal,
                    short_score=event.short_signal,
                    target_net_quantity=(
                        None if event.target_net_ratio is not None else event.target_net
                    ),
                    target_net_ratio=event.target_net_ratio,
                    confidence=event.confidence,
                    risk_scale=event.risk_scale,
                    long_exposure_scale=event.long_exposure_scale,
                    short_exposure_scale=event.short_exposure_scale,
                    allow_new_risk=event.allow_new_risk,
                    regime=event.regime,
                    reason=event.reason,
                    model_version=event.model_version,
                )
                continue

            if isinstance(event, FundingEvent):
                self.funding.apply(self.wallet, event)
                self._final_mark = event.mark_price
                # Match detailed replay risk/timeline observation without adding a bar return.
                self.wallet.observe_state(event.timestamp, self._final_mark)
                continue

            input_bar_count += 1
            if not self.wallet.liquidated:
                flat_wallet = (
                    self.wallet.long.quantity <= ZERO and self.wallet.short.quantity <= ZERO
                )
                skip_matcher = flat_wallet and not self.matcher._bar_may_touch_active_order(
                    event, self.wallet
                )
                if skip_matcher:
                    flat_idle_matcher_bypass_count += 1
                else:
                    outcome = self.matcher.match_outcome(event, self.wallet)
                    for fill in outcome.fills:
                        self.wallet.apply_fill(fill)
                        if retain_material_events:
                            retained_events.append(fill)
                    if outcome.liquidation_event is not None:
                        self.wallet.apply_liquidation(outcome.liquidation_event)
                        if retain_material_events:
                            retained_events.append(outcome.liquidation_event)
                    for order_id in outcome.expired_order_ids:
                        if self.wallet.remaining(order_id) > 0:
                            self.wallet.cancel_order(order_id)
                    self.wallet.merge_risk_metrics(
                        gross_peak=outcome.gross_peak,
                        equity_peak=outcome.equity_peak,
                        max_drawdown=outcome.max_drawdown,
                    )
                    self._advance_states_after_fills(event, outcome.fills)
            if not self.wallet.liquidated:
                self._plan(
                    event,
                    emit_events=False,
                    collect_diagnostics=False,
                )
            self._final_mark = event.close

            # Update exact risk/dual-leg timeline state without allocating a
            # SimulationSnapshot for every candle.
            self.wallet.observe_bar_state(event.timestamp, self._final_mark)
            if input_bar_count == 1 or input_bar_count % sample_stride == 0:
                retained_snapshots.append(
                    self.wallet.snapshot(event.timestamp, self._final_mark, observe=False)
                )
                if len(retained_snapshots) >= max_retained_snapshots:
                    retained_snapshots[:] = retained_snapshots[::2]
                    sample_stride *= 2

            if compact_wallet_history:
                self.wallet.release_transient_history(trusted_replay=True)

        if input_bar_count == 0:
            raise ValueError("ordered replay requires at least one bar event")

        # Always expose the final point without duplicating the latest sample.
        final_timestamp = self.wallet.last_timestamp or current_timestamp
        if final_timestamp is None:
            raise RuntimeError("replay snapshot timestamp missing")
        final_snapshot = self.wallet.snapshot(
            final_timestamp,
            self._final_mark,
            observe=False,
        )
        if retained_snapshots and retained_snapshots[-1].timestamp == final_snapshot.timestamp:
            retained_snapshots[-1] = final_snapshot
        else:
            retained_snapshots.append(final_snapshot)

        report = dict(self._report())
        report.update(
            {
                "replay_mode": "COMPACT_ORDERED_STREAM_V2",
                "processed_chunk_count": 0,
                "processed_input_event_count": input_event_count,
                "processed_bar_count": input_bar_count,
                "slot_validation_mode": "BITMASK_SLOT_VALIDATION_V1",
                "matcher_mode": "FLAT_IDLE_BYPASS_V1",
                "flat_idle_matcher_bypass_count": flat_idle_matcher_bypass_count,
                "retained_event_count": len(retained_events),
                "retained_snapshot_count": len(retained_snapshots),
                "snapshot_stride_bars": sample_stride,
                "max_chunk_input_events": 1,
                "material_event_count": self.wallet.fill_count + self.wallet.liquidation_count,
                "wallet_processed_fill_id_count": len(self.wallet.processed_fill_ids),
                "wallet_realized_by_fill_count": len(self.wallet.realized_by_fill),
                "wallet_tactical_lot_object_count": (
                    len(self.wallet.long.tactical_lots) + len(self.wallet.short.tactical_lots)
                ),
            }
        )
        return SimulationResult(
            events=tuple(retained_events),
            snapshots=tuple(retained_snapshots),
            report=report,
        )

    def replay_ordered_chunks(
        self,
        chunks: Iterable[Iterable[SimulationInputEvent]],
        *,
        retain_material_events: bool = True,
        retain_chunk_snapshots: bool = True,
    ) -> SimulationResult:
        """Replay already-ordered bounded chunks without retaining the full input stream.

        This path is intended for long historical backtests and optimization.  The
        canonical ``replay``/``advance`` methods deliberately keep their original
        all-events semantics for parity and recovery tests.  Each supplied chunk is
        processed by the same deterministic matcher/planner implementation, then the
        large per-candle input-event ledger is discarded before the next chunk.

        Only fills and liquidations are retained by default because aggregate fees,
        funding, PnL, drawdown, exposure and fill counters already live in the wallet
        report.  One final snapshot per chunk is sufficient for a compact equity
        series while the exact max drawdown remains tracked continuously by the
        wallet itself.
        """

        retained_events: list[StandardEvent] = []
        retained_snapshots = []
        first_chunk = True
        chunk_count = 0
        input_event_count = 0
        input_bar_count = 0
        max_chunk_events = 0

        for chunk in chunks:
            chunk_events = tuple(chunk)
            if not chunk_events:
                continue
            max_chunk_events = max(max_chunk_events, len(chunk_events))
            input_event_count += len(chunk_events)
            input_bar_count += sum(isinstance(event, BarEvent) for event in chunk_events)

            partial = self._run(chunk_events, reset=first_chunk)
            first_chunk = False
            chunk_count += 1

            if retain_material_events:
                retained_events.extend(
                    event
                    for event in partial.events
                    if isinstance(event, (FillEvent, LiquidationEvent))
                )
            if retain_chunk_snapshots and partial.snapshots:
                retained_snapshots.append(partial.snapshots[-1])

            # ``_run`` needs a slot set for duplicate detection inside a batch.
            # Long historical streams are validated and ordered by the upstream
            # chunk producer, so retaining every historical slot after the chunk
            # has committed only creates O(n) memory growth with no additional
            # protection.  Keep the monotonic ``_last_event_key`` instead.
            self._processed_slots.clear()

        if first_chunk:
            raise ValueError("ordered replay requires at least one non-empty chunk")

        report = dict(self._report())
        report.update(
            {
                "replay_mode": "COMPACT_ORDERED_CHUNKS_V1",
                "processed_chunk_count": chunk_count,
                "processed_input_event_count": input_event_count,
                "processed_bar_count": input_bar_count,
                "retained_event_count": len(retained_events),
                "retained_snapshot_count": len(retained_snapshots),
                "max_chunk_input_events": max_chunk_events,
            }
        )
        return SimulationResult(
            events=tuple(retained_events),
            snapshots=tuple(retained_snapshots),
            report=report,
        )

    def replay(self, events: Iterable[SimulationInputEvent]) -> SimulationResult:
        """Run a deterministic batch simulation from a clean initial state."""
        return self._run(events, reset=True)

    def advance(self, events: Iterable[SimulationInputEvent]) -> SimulationResult:
        """Advance atomically; any failure restores the pre-batch simulation state."""
        checkpoint = self.checkpoint()
        try:
            return self._run(events, reset=False)
        except Exception:
            self.restore(checkpoint)
            raise
