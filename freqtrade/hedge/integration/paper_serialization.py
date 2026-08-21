"""Checkpoint codecs for the integrated Paper Hedge runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from freqtrade.hedge.contracts.business_identity import (
    BusinessIdentity,
    BusinessOrderRole,
)
from freqtrade.hedge.execution.service import (
    ApprovedOrderIntent,
    ExecutionOrder,
    ExecutionResult,
    ExternalOrderSnapshot,
)
from freqtrade.hedge.execution.service import (
    IntentAction as ExecutionAction,
)
from freqtrade.hedge.execution.service import (
    OrderIntent as ExecutionOrderIntent,
)
from freqtrade.hedge.execution.service import (
    OrderType as ExecutionOrderType,
)
from freqtrade.hedge.execution.service import (
    PositionSide as ExecutionSide,
)
from freqtrade.hedge.execution.state_machine import OrderLifecycle, OrderState
from freqtrade.hedge.integration.candle_cursor import bar_fingerprint
from freqtrade.hedge.planning.context import (
    IntentAction,
    MarketSnapshot,
    OrderSide,
    PositionBucket,
    PositionSide,
    StrategyLegState,
    TimeInForce,
)
from freqtrade.hedge.planning.context import (
    OrderIntent as PlannerOrderIntent,
)
from freqtrade.hedge.planning.context import (
    OrderType as PlannerOrderType,
)
from freqtrade.hedge.simulation.exchange import BarEvent as SimulationBarEvent


class PaperSerializationMixin:
    """Encode and restore state that crosses the Paper checkpoint boundary."""

    @staticmethod
    def _encode_leg_state(state: StrategyLegState) -> dict[str, object]:
        payload: dict[str, object] = {}
        for item in fields(StrategyLegState):
            value = getattr(state, item.name)
            if isinstance(value, Decimal):
                payload[item.name] = str(value)
            elif isinstance(value, datetime):
                payload[item.name] = value.isoformat()
            elif hasattr(value, "value"):
                payload[item.name] = value.value
            else:
                payload[item.name] = value
        return payload

    @staticmethod
    def _decode_leg_state(payload: object, side: PositionSide) -> StrategyLegState:
        if not isinstance(payload, Mapping):
            return StrategyLegState(side)
        values = dict(payload)
        values["side"] = side
        for name in (
            "trailing_extreme",
            "trailing_trigger_price",
            "unstuck_daily_loss",
            "unstuck_weekly_loss",
        ):
            if values.get(name) is not None:
                values[name] = Decimal(str(values[name]))
        for name in (
            "last_entry_at",
            "last_reduce_at",
            "trailing_started_at",
            "trailing_confirmed_at",
            "trailing_cooldown_until",
            "last_unstuck_at",
        ):
            if values.get(name):
                values[name] = datetime.fromisoformat(str(values[name]))
        allowed = {item.name for item in fields(StrategyLegState)}
        return StrategyLegState(**{key: value for key, value in values.items() if key in allowed})

    @staticmethod
    def _encode_market(market: MarketSnapshot | None) -> dict[str, object] | None:
        if market is None:
            return None
        return {
            "symbol": market.symbol,
            "timestamp": market.timestamp.isoformat(),
            "bid": str(market.bid),
            "ask": str(market.ask),
            "mark": str(market.mark),
            "tick_size": str(market.tick_size),
            "qty_step": str(market.qty_step),
            "min_qty": str(market.min_qty),
            "min_notional": str(market.min_notional),
        }

    @staticmethod
    def _decode_market(payload: object) -> MarketSnapshot | None:
        if not isinstance(payload, Mapping):
            return None
        return MarketSnapshot(
            symbol=str(payload["symbol"]),
            timestamp=datetime.fromisoformat(str(payload["timestamp"])),
            bid=Decimal(str(payload["bid"])),
            ask=Decimal(str(payload["ask"])),
            mark=Decimal(str(payload["mark"])),
            tick_size=Decimal(str(payload.get("tick_size", "0.01"))),
            qty_step=Decimal(str(payload.get("qty_step", "0.001"))),
            min_qty=Decimal(str(payload.get("min_qty", "0"))),
            min_notional=Decimal(str(payload.get("min_notional", "0"))),
        )

    @staticmethod
    def _encode_bar(bar: SimulationBarEvent | None) -> dict[str, object] | None:
        if bar is None:
            return None
        return {
            "timestamp": bar.timestamp.isoformat(),
            "symbol": bar.symbol,
            "open": str(bar.open),
            "high": str(bar.high),
            "low": str(bar.low),
            "close": str(bar.close),
            "volume": None if bar.volume is None else str(bar.volume),
            "fingerprint": bar_fingerprint(bar),
        }

    @staticmethod
    def _decode_bar(payload: object) -> SimulationBarEvent | None:
        if not isinstance(payload, Mapping):
            return None
        bar = SimulationBarEvent(
            timestamp=datetime.fromisoformat(str(payload["timestamp"])),
            symbol=str(payload["symbol"]),
            open=Decimal(str(payload["open"])),
            high=Decimal(str(payload["high"])),
            low=Decimal(str(payload["low"])),
            close=Decimal(str(payload["close"])),
            volume=(None if payload.get("volume") is None else Decimal(str(payload["volume"]))),
        )
        expected = payload.get("fingerprint")
        if expected is not None and str(expected) != bar_fingerprint(bar):
            raise ValueError("paper checkpoint bar fingerprint is invalid")
        return bar

    @staticmethod
    def _json_compatible(value: object) -> object:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Mapping):
            return {
                str(key): PaperSerializationMixin._json_compatible(item)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list, set, frozenset)):
            return [PaperSerializationMixin._json_compatible(item) for item in value]
        raise TypeError(f"paper state cannot serialize {type(value).__name__}")

    @staticmethod
    def _encode_planner_intent(intent: PlannerOrderIntent | None) -> dict[str, object] | None:
        if intent is None:
            return None
        return {
            "intent_id": intent.intent_id,
            "symbol": intent.symbol,
            "account_id": (
                intent.business_identity.account_id
                if intent.business_identity is not None
                else "paper"
            ),
            "position_side": intent.position_side.value,
            "order_side": intent.order_side.value,
            "action": intent.action.value,
            "bucket": intent.bucket.value,
            "quantity": str(intent.quantity),
            "price": str(intent.price),
            "reduce_only": intent.reduce_only,
            "order_type": intent.order_type.value,
            "time_in_force": intent.time_in_force.value,
            "layer": intent.layer,
            "reason": intent.reason,
            "tactical_lot_id": intent.tactical_lot_id,
            "business_trade_id": (
                None
                if intent.business_identity is None
                else str(intent.business_identity.business_trade_id)
            ),
            "business_trade_seq": (
                None
                if intent.business_identity is None
                else intent.business_identity.business_trade_seq
            ),
            "business_lot_id": (
                None
                if intent.business_identity is None
                else str(intent.business_identity.business_lot_id)
            ),
            "lot_index": (
                None if intent.business_identity is None else intent.business_identity.lot_index
            ),
            "order_role": None if intent.order_role is None else intent.order_role.value,
            "strategy_entry_key": intent.strategy_entry_key,
            "order_revision": intent.order_revision,
            "submission_generation": intent.submission_generation,
        }

    @staticmethod
    def _decode_planner_intent(payload: object) -> PlannerOrderIntent | None:
        if not isinstance(payload, Mapping):
            return None
        return PlannerOrderIntent(
            intent_id=str(payload["intent_id"]),
            symbol=str(payload["symbol"]),
            position_side=PositionSide(str(payload["position_side"])),
            order_side=OrderSide(str(payload["order_side"])),
            action=IntentAction(str(payload["action"])),
            bucket=PositionBucket(str(payload["bucket"])),
            quantity=Decimal(str(payload["quantity"])),
            price=Decimal(str(payload["price"])),
            reduce_only=bool(payload["reduce_only"]),
            order_type=PlannerOrderType(str(payload.get("order_type", "LIMIT"))),
            time_in_force=TimeInForce(str(payload.get("time_in_force", "GTC"))),
            layer=int(payload.get("layer", 0)),
            reason=str(payload.get("reason", "")),
            tactical_lot_id=(
                None
                if payload.get("tactical_lot_id") is None
                else str(payload.get("tactical_lot_id"))
            ),
            business_identity=(
                None
                if payload.get("business_trade_id") is None
                else BusinessIdentity(
                    business_trade_id=payload["business_trade_id"],
                    business_trade_seq=int(payload["business_trade_seq"]),
                    business_lot_id=payload["business_lot_id"],
                    lot_index=int(payload.get("lot_index", 1)),
                    account_id=str(payload.get("account_id", "paper")),
                    symbol=str(payload["symbol"]),
                    position_side=str(payload["position_side"]),
                )
            ),
            order_role=(
                None
                if payload.get("order_role") is None
                else BusinessOrderRole(str(payload["order_role"]))
            ),
            strategy_entry_key=str(payload.get("strategy_entry_key", "")),
            order_revision=int(payload.get("order_revision", 0)),
            submission_generation=int(payload.get("submission_generation", 0)),
        )

    def _encode_active_execution_orders(self: Any) -> list[dict[str, object]]:
        execution = self._execution()
        active_orders = execution.store.list_orders(
            account_id=self.account_id,
            symbol=self.execution_symbol,
            statuses=(OrderState.ACKNOWLEDGED, OrderState.PARTIAL, OrderState.UNKNOWN),
            include_terminal=False,
        )
        rows: list[dict[str, object]] = []
        for order in active_orders:
            snapshot = execution.exchange.query_order(client_order_id=order.client_order_id)
            intent = order.intent
            rows.append(
                {
                    "client_order_id": order.client_order_id,
                    "approved_quantity": str(order.approved_quantity),
                    "created_at": order.created_at.isoformat(),
                    "intent": {
                        "account_id": intent.account_id,
                        "symbol": intent.symbol,
                        "position_side": intent.position_side.value,
                        "action": intent.action.value,
                        "quantity": str(intent.quantity),
                        "idempotency_key": intent.idempotency_key,
                        "order_type": intent.order_type.value,
                        "limit_price": None
                        if intent.limit_price is None
                        else str(intent.limit_price),
                        "reduce_only": intent.reduce_only,
                        "intent_id": str(intent.intent_id),
                        "action_group_id": (
                            None
                            if intent.action_group_id is None
                            else str(intent.action_group_id)
                        ),
                        "business_trade_id": (
                            None
                            if intent.business_trade_id is None
                            else str(intent.business_trade_id)
                        ),
                        "business_trade_seq": intent.business_trade_seq,
                        "business_lot_id": (
                            None
                            if intent.business_lot_id is None
                            else str(intent.business_lot_id)
                        ),
                        "lot_index": intent.lot_index,
                        "order_role": (
                            None if intent.order_role is None else intent.order_role.value
                        ),
                        "order_revision": intent.order_revision,
                        "submission_generation": intent.submission_generation,
                        "metadata": self._json_compatible(intent.metadata),
                    },
                    "lifecycle": {
                        "status": order.lifecycle.status.value,
                        "filled_quantity": str(order.lifecycle.filled_quantity),
                        "average_price": (
                            None
                            if order.lifecycle.average_price is None
                            else str(order.lifecycle.average_price)
                        ),
                        "exchange_order_id": order.lifecycle.exchange_order_id,
                        "version": order.lifecycle.version,
                        "updated_at": order.lifecycle.updated_at.isoformat(),
                        "reason": order.lifecycle.reason,
                    },
                    "external": self._encode_external_snapshot(snapshot),
                    "planner_intent": self._encode_planner_intent(
                        self._simulation_intents.get(order.client_order_id)
                    ),
                }
            )
        return rows

    @staticmethod
    def _encode_external_snapshot(snapshot: Any) -> dict[str, object] | None:
        if snapshot is None:
            return None
        return {
            "status": snapshot.status.value,
            "filled_quantity": str(snapshot.filled_quantity),
            "average_price": None
            if snapshot.average_price is None
            else str(snapshot.average_price),
            "exchange_order_id": snapshot.exchange_order_id,
            "exchange_trade_id": snapshot.exchange_trade_id,
            "last_fill_fee": str(snapshot.last_fill_fee),
            "fee_currency": snapshot.fee_currency,
            "reason": snapshot.reason,
            "observed_at": snapshot.observed_at.isoformat(),
        }

    def _restore_active_execution_orders(self: Any, payload: object) -> None:
        if payload is None:
            return
        if not isinstance(payload, list):
            raise TypeError("paper active_orders must be a list")
        execution = self._execution()
        for raw in payload:
            if not isinstance(raw, Mapping):
                raise TypeError("paper active order row must be a mapping")
            intent_payload = raw.get("intent")
            lifecycle_payload = raw.get("lifecycle")
            if not isinstance(intent_payload, Mapping) or not isinstance(
                lifecycle_payload, Mapping
            ):
                raise TypeError("paper active order is missing intent/lifecycle")
            intent = ExecutionOrderIntent(
                account_id=str(intent_payload["account_id"]),
                symbol=str(intent_payload["symbol"]),
                position_side=ExecutionSide(str(intent_payload["position_side"])),
                action=ExecutionAction(str(intent_payload["action"])),
                quantity=Decimal(str(intent_payload["quantity"])),
                idempotency_key=str(intent_payload["idempotency_key"]),
                order_type=ExecutionOrderType(str(intent_payload.get("order_type", "LIMIT"))),
                limit_price=(
                    None
                    if intent_payload.get("limit_price") is None
                    else Decimal(str(intent_payload["limit_price"]))
                ),
                reduce_only=bool(intent_payload.get("reduce_only", False)),
                intent_id=UUID(str(intent_payload["intent_id"])),
                action_group_id=(
                    None
                    if intent_payload.get("action_group_id") is None
                    else UUID(str(intent_payload["action_group_id"]))
                ),
                business_trade_id=(
                    None
                    if intent_payload.get("business_trade_id") is None
                    else UUID(str(intent_payload["business_trade_id"]))
                ),
                business_trade_seq=(
                    None
                    if intent_payload.get("business_trade_seq") is None
                    else int(intent_payload["business_trade_seq"])
                ),
                business_lot_id=(
                    None
                    if intent_payload.get("business_lot_id") is None
                    else UUID(str(intent_payload["business_lot_id"]))
                ),
                lot_index=(
                    None
                    if intent_payload.get("lot_index") is None
                    else int(intent_payload["lot_index"])
                ),
                order_role=(
                    None
                    if intent_payload.get("order_role") is None
                    else BusinessOrderRole(str(intent_payload["order_role"]))
                ),
                order_revision=int(intent_payload.get("order_revision", 0)),
                submission_generation=int(
                    intent_payload.get("submission_generation", 0)
                ),
                metadata=(
                    dict(intent_payload.get("metadata", {}))
                    if isinstance(intent_payload.get("metadata", {}), Mapping)
                    else {}
                ),
            )
            lifecycle = OrderLifecycle(
                status=OrderState(str(lifecycle_payload["status"])),
                filled_quantity=Decimal(str(lifecycle_payload.get("filled_quantity", "0"))),
                average_price=(
                    None
                    if lifecycle_payload.get("average_price") is None
                    else Decimal(str(lifecycle_payload["average_price"]))
                ),
                exchange_order_id=(
                    None
                    if lifecycle_payload.get("exchange_order_id") is None
                    else str(lifecycle_payload["exchange_order_id"])
                ),
                version=int(lifecycle_payload.get("version", 0)),
                updated_at=datetime.fromisoformat(str(lifecycle_payload["updated_at"])),
                reason=(
                    None
                    if lifecycle_payload.get("reason") is None
                    else str(lifecycle_payload["reason"])
                ),
            )
            client_order_id = str(raw["client_order_id"])
            created_at = datetime.fromisoformat(str(raw["created_at"]))
            order = ExecutionOrder(
                intent=intent,
                client_order_id=client_order_id,
                approved_quantity=Decimal(str(raw["approved_quantity"])),
                lifecycle=lifecycle,
                created_at=created_at,
            )
            approved = ApprovedOrderIntent(
                intent=intent,
                approved_quantity=order.approved_quantity,
                client_order_id=client_order_id,
                approved_at=created_at,
                risk_reason_codes=("RECOVERED_DURABLE_PAPER",),
            )
            external_payload = raw.get("external")
            if isinstance(external_payload, Mapping):
                snapshot = ExternalOrderSnapshot(
                    client_order_id=client_order_id,
                    status=OrderState(str(external_payload["status"])),
                    filled_quantity=Decimal(str(external_payload.get("filled_quantity", "0"))),
                    average_price=(
                        None
                        if external_payload.get("average_price") is None
                        else Decimal(str(external_payload["average_price"]))
                    ),
                    exchange_order_id=(
                        None
                        if external_payload.get("exchange_order_id") is None
                        else str(external_payload["exchange_order_id"])
                    ),
                    exchange_trade_id=(
                        None
                        if external_payload.get("exchange_trade_id") is None
                        else str(external_payload["exchange_trade_id"])
                    ),
                    last_fill_fee=Decimal(str(external_payload.get("last_fill_fee", "0"))),
                    fee_currency=(
                        None
                        if external_payload.get("fee_currency") is None
                        else str(external_payload["fee_currency"])
                    ),
                    reason=(
                        None
                        if external_payload.get("reason") is None
                        else str(external_payload["reason"])
                    ),
                    observed_at=datetime.fromisoformat(str(external_payload["observed_at"])),
                )
            else:
                snapshot = ExternalOrderSnapshot(
                    client_order_id=client_order_id,
                    status=order.lifecycle.status,
                    filled_quantity=order.lifecycle.filled_quantity,
                    average_price=order.lifecycle.average_price,
                    exchange_order_id=order.lifecycle.exchange_order_id,
                    reason=order.lifecycle.reason,
                    observed_at=order.lifecycle.updated_at,
                )
            execution.store.put(order)
            execution.exchange.restore_order(approved, snapshot)
            reservation = execution.idempotency.reserve(intent.idempotency_key)
            if reservation.value is None:
                execution.idempotency.complete(
                    intent.idempotency_key,
                    ExecutionResult(order=order, message="RECOVERED_DURABLE_PAPER"),
                )
            execution.ledger.record(
                order=order,
                event_type="ORDER_RECOVERED",
                payload={
                    "account_id": self.account_id,
                    "symbol": self.symbol,
                    "client_order_id": client_order_id,
                    "status": order.lifecycle.status.value,
                },
            )
            planner_intent = self._decode_planner_intent(raw.get("planner_intent"))
            if planner_intent is not None:
                self._simulation_intents[client_order_id] = planner_intent
                self._planner_order_to_client[planner_intent.intent_id] = client_order_id
