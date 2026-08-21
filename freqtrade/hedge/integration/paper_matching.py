"""Market-rule, funding, and fill matching helpers for Paper runtime."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from typing import Any

from freqtrade.hedge.contracts.ports import MarketRules as ExecutionMarketRules
from freqtrade.hedge.contracts.ports import PositionKey as ContractPositionKey
from freqtrade.hedge.execution.service import (
    ExecutionOrder,
    ExecutionResult,
)
from freqtrade.hedge.execution.service import (
    PositionSide as ExecutionSide,
)
from freqtrade.hedge.integration.paper_events import fee_account_event, funding_account_event
from freqtrade.hedge.numeric import ZERO
from freqtrade.hedge.planning.context import (
    IntentAction,
    OrderSide,
    PositionBucket,
    PositionSide,
    TimeInForce,
)
from freqtrade.hedge.planning.context import (
    OrderIntent as PlannerOrderIntent,
)
from freqtrade.hedge.planning.context import (
    OrderType as PlannerOrderType,
)
from freqtrade.hedge.simulation.cross_wallet import CrossWallet
from freqtrade.hedge.simulation.exchange import AccountEvent, BarEvent, FundingEvent


ONE = Decimal(1)
logger = logging.getLogger(__name__)


class PaperMatchingMixin:
    """Methods that translate market facts into deterministic Paper fills."""

    _last_funding_event_time: datetime | None

    def _record_account_event(self: Any, event: AccountEvent) -> bool:
        if event.event_id in self._applied_account_event_ids:
            return False
        created = True
        if self.paper_config.account_events_enabled:
            created = self._account_event_sink.record(event)
        self._applied_account_event_ids.add(event.event_id)
        return created

    def _apply_funding_events(
        self: Any,
        events: tuple[FundingEvent, ...],
    ) -> tuple[AccountEvent, ...]:
        applied: list[AccountEvent] = []
        for funding in sorted(events, key=lambda item: item.timestamp):
            long_leg = self._leg(PositionSide.LONG)
            short_leg = self._leg(PositionSide.SHORT)
            long_amount = -(long_leg.quantity * funding.mark_price * funding.rate)
            short_amount = short_leg.quantity * funding.mark_price * funding.rate
            total = long_amount + short_amount
            event = funding_account_event(funding=funding, amount=total)
            if self._record_account_event(event):
                self._funding_balance_delta += total
                if (
                    self._last_funding_event_time is None
                    or funding.timestamp > self._last_funding_event_time
                ):
                    self._last_funding_event_time = funding.timestamp
                applied.append(event)
        return tuple(applied)

    def _update_market_rules(
        self: Any,
        market: Any,
        *,
        maker_fee_rate: Decimal | None = None,
        taker_fee_rate: Decimal | None = None,
    ) -> None:
        execution = self._execution()
        setter = getattr(execution.market_rules, "set_rules", None)
        rules = ExecutionMarketRules(
            quantity_step=market.qty_step,
            price_tick=market.tick_size,
            minimum_quantity=max(market.min_qty, Decimal("0.00000001")),
            minimum_notional=max(market.min_notional, Decimal("0.00000001")),
        )
        if callable(setter):
            for side in (ExecutionSide.LONG, ExecutionSide.SHORT):
                setter(
                    ContractPositionKey(
                        exchange="paper",
                        account_id=self.account_id,
                        symbol=self.symbol,
                        position_side=side,
                    ),
                    rules,
                )

        maker = self.matcher.config.maker_fee_rate if maker_fee_rate is None else maker_fee_rate
        taker = self.matcher.config.taker_fee_rate if taker_fee_rate is None else taker_fee_rate
        for field, value in (("maker_fee_rate", maker), ("taker_fee_rate", taker)):
            if not value.is_finite() or value < ZERO or value > ONE:
                raise ValueError(f"{field} must be finite and within [0, 1]")
        # DataProvider OHLCV, exchange precision and current account/market fee
        # rates become one immutable matching snapshot for this cycle.
        self.matcher = type(self.matcher)(
            replace(
                self.matcher.config,
                fee_rate=None,
                maker_fee_rate=maker,
                taker_fee_rate=taker,
                price_tick=market.tick_size,
                qty_step=market.qty_step,
                min_fill_qty=market.min_qty,
                min_fill_notional=market.min_notional,
            )
        )
        self._paper_fee_rate = taker

    def _planner_intent_for_execution_order(
        self: Any,
        order: ExecutionOrder,
    ) -> PlannerOrderIntent | None:
        existing = self._simulation_intents.get(order.client_order_id)
        if existing is not None:
            return existing
        price = order.intent.limit_price
        if price is None:
            reference = order.intent.metadata.get("reference_price")
            if reference is None:
                return None
            price = Decimal(str(reference))
        side = PositionSide(order.intent.position_side.value)
        reduce_only = order.intent.reduces_risk
        order_side = (
            OrderSide.SELL
            if side is PositionSide.LONG and reduce_only
            else OrderSide.BUY
            if side is PositionSide.LONG
            else OrderSide.BUY
            if reduce_only
            else OrderSide.SELL
        )
        try:
            bucket = PositionBucket(str(order.intent.metadata.get("bucket", "TACTICAL")).upper())
        except ValueError:
            bucket = PositionBucket.TACTICAL
        planner_intent = PlannerOrderIntent(
            intent_id=str(order.intent.metadata.get("planner_intent_id", order.intent.intent_id)),
            symbol=self.symbol,
            position_side=side,
            order_side=order_side,
            action=IntentAction(order.intent.action.value),
            bucket=bucket,
            quantity=order.approved_quantity - order.lifecycle.filled_quantity,
            price=price,
            reduce_only=reduce_only,
            order_type=PlannerOrderType(order.intent.order_type.value),
            time_in_force=TimeInForce(
                str(order.intent.metadata.get("time_in_force", "GTC")).upper()
            ),
            layer=int(order.intent.metadata.get("layer", 0)),
            reason=str(
                order.intent.metadata.get(
                    "control_action",
                    order.intent.metadata.get("reason", "managed_execution_order"),
                )
            )[:256],
            tactical_lot_id=(
                None
                if order.intent.metadata.get("tactical_lot_id") is None
                else str(order.intent.metadata.get("tactical_lot_id"))
            ),
            business_identity=order.intent.business_identity,
            order_role=order.intent.order_role,
            target_business_lot_id=order.intent.business_lot_id
            if order.intent.reduces_risk
            else None,
            strategy_entry_key=str(
                order.intent.metadata.get("strategy_entry_key", "")
            ),
            order_revision=int(order.intent.order_revision),
            submission_generation=int(order.intent.submission_generation),
        )
        self._simulation_intents[order.client_order_id] = planner_intent
        return planner_intent

    def _matcher_wallet(self: Any, market: Any) -> CrossWallet:
        execution = self._execution()
        wallet = CrossWallet(
            initial_balance=self.initial_balance,
            leverage=self.leverage,
            fee_rate=self._paper_fee_rate,
            maintenance_margin_rate=self.planner_config.maintenance_margin_rate,
            liquidation_fee_rate=self.planner_config.liquidation_fee_rate,
            liquidation_buffer_warning_ratio=self.planner_config.liquidation_buffer_warning_ratio,
        )
        now = market.timestamp
        for side in (PositionSide.LONG, PositionSide.SHORT):
            state = self._bucket[side]
            leg = wallet.leg(side)
            exact_lots = state.position_lots()
            if exact_lots:
                core_qty = sum(
                    (lot.quantity for lot in exact_lots if lot.bucket is PositionBucket.CORE),
                    ZERO,
                )
                if core_qty > ZERO:
                    core_quote = sum(
                        (
                            lot.quantity * lot.average_price
                            for lot in exact_lots
                            if lot.bucket is PositionBucket.CORE
                        ),
                        ZERO,
                    )
                    leg.increase(
                        core_qty,
                        core_quote / core_qty,
                        PositionBucket.CORE,
                        tactical_lot_id=None,
                        opened_at=now,
                        layer=0,
                        fee=ZERO,
                    )
                for lot in exact_lots:
                    if lot.bucket is not PositionBucket.TACTICAL:
                        continue
                    leg.increase(
                        lot.quantity,
                        lot.average_price,
                        PositionBucket.TACTICAL,
                        tactical_lot_id=lot.lot_id,
                        opened_at=lot.opened_at,
                        layer=lot.layer,
                        fee=ZERO,
                    )
            else:
                if state.core_quantity > ZERO:
                    leg.increase(
                        state.core_quantity,
                        state.core_average,
                        PositionBucket.CORE,
                        tactical_lot_id=None,
                        opened_at=now,
                        layer=0,
                        fee=ZERO,
                    )
                if state.tactical_quantity > ZERO:
                    leg.increase(
                        state.tactical_quantity,
                        state.tactical_average,
                        PositionBucket.TACTICAL,
                        tactical_lot_id=f"paper-legacy-{side.value.lower()}",
                        opened_at=now,
                        layer=0,
                        fee=ZERO,
                    )
            fake = self._fake_leg(side)
            leg.realized_pnl = fake.realized_pnl
            leg.tactical_realized_pnl = fake.realized_pnl
        fees = self._fake_leg(PositionSide.LONG).fees + self._fake_leg(PositionSide.SHORT).fees
        realized = (
            self._fake_leg(PositionSide.LONG).realized_pnl
            + self._fake_leg(PositionSide.SHORT).realized_pnl
        )
        wallet.balance = self.initial_balance + realized - fees + self._funding_balance_delta
        for order in execution.core.list_orders(
            account_id=self.account_id,
            symbol=self.execution_symbol,
            include_terminal=False,
        ):
            planner_intent = self._planner_intent_for_execution_order(order)
            if planner_intent is None:
                continue
            remaining = order.approved_quantity - order.lifecycle.filled_quantity
            if remaining <= ZERO:
                continue
            wallet.accept_order(
                order.client_order_id,
                replace(planner_intent, quantity=remaining),
                accepted_at=order.created_at,
            )
        return wallet

    def _match_active_orders(
        self: Any,
        market: Any,
        bar: BarEvent,
    ) -> tuple[list[ExecutionResult], list[ExecutionResult], list[AccountEvent]]:
        execution = self._execution()
        if bar.symbol != market.symbol or bar.timestamp != market.timestamp:
            raise ValueError("Paper BarEvent must match the planning MarketSnapshot")
        outcome = self.matcher.match_outcome(bar, self._matcher_wallet(market))
        fills: list[ExecutionResult] = []
        expirations: list[ExecutionResult] = []
        account_events: list[AccountEvent] = []
        for fill in outcome.fills:
            try:
                snapshot = execution.exchange.fill_order(
                    fill.order_id,
                    quantity=fill.quantity,
                    price=fill.price,
                    exchange_trade_id=fill.event_id,
                    fee=fill.fee,
                )
                result = execution.engine.apply_exchange_event(snapshot)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "Rejected simulation fill during paper cycle",
                    extra={
                        "reason_code": "PAPER_FILL_REJECTED",
                        "error_type": type(exc).__name__,
                    },
                )
                continue
            fills.append(result)
            planner_for_callback = self._simulation_intents.get(fill.order_id)
            if planner_for_callback is not None:
                self._notify_fill_observers(
                    planner_for_callback, fill.price, fill.quantity, fill.timestamp
                )
            fee_event = fee_account_event(
                fill_event_id=fill.event_id,
                timestamp=fill.timestamp,
                symbol=fill.symbol,
                amount=fill.fee,
                position_side=fill.position_side,
            )
            if self._record_account_event(fee_event):
                account_events.append(fee_event)
            state = self._bucket[fill.position_side]
            if fill.action in {IntentAction.OPEN, IntentAction.INCREASE}:
                state.increase(
                    fill.bucket,
                    fill.quantity,
                    fill.price,
                    fill.timestamp,
                    business_identity=(
                        None
                        if planner_for_callback is None
                        else planner_for_callback.business_identity
                    ),
                )
            else:
                state.reduce(
                    fill.bucket,
                    fill.quantity,
                    business_identity=(
                        None
                        if planner_for_callback is None
                        else planner_for_callback.business_identity
                    ),
                )
            if result.order.lifecycle.filled_quantity >= result.order.approved_quantity:
                self._simulation_intents.pop(fill.order_id, None)
        for client_id in outcome.expired_order_ids:
            try:
                expirations.append(execution.engine.cancel(client_id))
            except Exception as exc:
                logger.warning("simulation order cancellation failed for %s: %s", client_id, exc)
                continue
            self._simulation_intents.pop(client_id, None)
        return fills, expirations, account_events
