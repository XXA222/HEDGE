"""Durable execution recovery helpers for the Paper runtime."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from freqtrade.hedge.execution.idempotency import ReservationState
from freqtrade.hedge.execution.service import (
    ApprovedOrderIntent,
    ExecutionOrder,
    ExecutionResult,
    ExternalOrderSnapshot,
)
from freqtrade.hedge.execution.state_machine import OrderState
from freqtrade.hedge.integration.paper_projection import _BucketState
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


class PaperRecoveryMixin:
    """Restore authoritative SQL execution facts before matching resumes."""

    @staticmethod
    def _planner_intent_from_execution(order: ExecutionOrder) -> PlannerOrderIntent | None:
        metadata = order.intent.metadata
        planner_id = metadata.get("planner_intent_id")
        if planner_id is None or order.intent.limit_price is None:
            return None
        side = PositionSide(order.intent.position_side.value)
        action = IntentAction(order.intent.action.value)
        increases = action in {IntentAction.OPEN, IntentAction.INCREASE}
        order_side = OrderSide.BUY if (side is PositionSide.LONG) == increases else OrderSide.SELL
        try:
            bucket = PositionBucket(str(metadata.get("bucket", "TACTICAL")))
            time_in_force = TimeInForce(str(metadata.get("time_in_force", "GTC")))
        except ValueError:
            return None
        return PlannerOrderIntent(
            intent_id=str(planner_id),
            symbol=order.intent.symbol,
            position_side=side,
            order_side=order_side,
            action=action,
            bucket=bucket,
            quantity=order.approved_quantity,
            price=order.intent.limit_price,
            reduce_only=order.intent.reduce_only,
            order_type=PlannerOrderType(order.intent.order_type.value),
            time_in_force=time_in_force,
            layer=int(metadata.get("layer", 0)),
            reason=str(metadata.get("reason", "recovered from SQL execution state")),
            tactical_lot_id=(
                None
                if metadata.get("tactical_lot_id") is None
                else str(metadata.get("tactical_lot_id"))
            ),
        )

    def _restore_authoritative_execution_orders(self: Any) -> bool:
        execution = self._execution()
        restored_any = False
        for order in execution.store.list_orders(
            account_id=self.account_id,
            symbol=self.execution_symbol,
            statuses=(OrderState.ACKNOWLEDGED, OrderState.PARTIAL, OrderState.UNKNOWN),
            include_terminal=False,
        ):
            if str(order.intent.metadata.get("exchange", "")).lower() != "paper":
                continue
            approved = ApprovedOrderIntent(
                intent=order.intent,
                approved_quantity=order.approved_quantity,
                client_order_id=order.client_order_id,
                approved_at=order.created_at,
                risk_reason_codes=("RECOVERED_SQL_EXECUTION_STATE",),
            )
            snapshot = execution.exchange.query_order(client_order_id=order.client_order_id)
            if snapshot is None:
                snapshot = ExternalOrderSnapshot(
                    client_order_id=order.client_order_id,
                    status=order.lifecycle.status,
                    filled_quantity=order.lifecycle.filled_quantity,
                    average_price=order.lifecycle.average_price,
                    exchange_order_id=order.lifecycle.exchange_order_id,
                    reason=order.lifecycle.reason,
                    observed_at=order.lifecycle.updated_at,
                )
                execution.exchange.restore_order(approved, snapshot)
            elif (
                snapshot.status is not order.lifecycle.status
                or snapshot.filled_quantity != order.lifecycle.filled_quantity
                or snapshot.average_price != order.lifecycle.average_price
            ):
                raise RuntimeError(
                    "existing Paper exchange snapshot conflicts with authoritative order state"
                )
            result = ExecutionResult(order=order, message="RECOVERED_SQL_EXECUTION_STATE")
            recover_completed = getattr(execution.idempotency, "recover_completed", None)
            if callable(recover_completed):
                recover_completed(order.intent.idempotency_key, result)
            else:
                reservation = execution.idempotency.reserve(order.intent.idempotency_key)
                if reservation.state is ReservationState.NEW:
                    execution.idempotency.complete(order.intent.idempotency_key, result)
                elif reservation.state is ReservationState.IN_FLIGHT:
                    raise RuntimeError(
                        "cannot recover SQL order while idempotency is still in flight"
                    )
            planner_intent = self._planner_intent_from_execution(order)
            if planner_intent is not None:
                self._simulation_intents[order.client_order_id] = planner_intent
                self._planner_order_to_client[planner_intent.intent_id] = order.client_order_id
            restored_any = True
        return restored_any

    def _restore_account_events(self: Any) -> None:
        if not self.paper_config.account_events_enabled:
            return
        recover = getattr(self._account_event_sink, "recover", None)
        if not callable(recover):
            return
        recovered = recover()
        if recovered is None:
            return
        self._applied_account_event_ids.update(recovered.event_ids)
        self._funding_balance_delta = recovered.funding_balance_delta
        self._last_funding_event_time = recovered.last_funding_event_time

    def _restore_buckets(self: Any, payload: Mapping[str, Any]) -> None:
        buckets = payload.get("buckets", {})
        if not isinstance(buckets, Mapping):
            raise TypeError("paper bucket state must be a mapping")
        for side in (PositionSide.LONG, PositionSide.SHORT):
            row = buckets.get(side.value, {})
            if not isinstance(row, Mapping):
                raise TypeError("paper bucket row must be a mapping")
            self._bucket[side] = _BucketState(
                core_quantity=Decimal(str(row.get("core_quantity", "0"))),
                core_average=Decimal(str(row.get("core_average", "0"))),
                core_opened_at=(
                    None
                    if row.get("core_opened_at") in (None, "")
                    else datetime.fromisoformat(str(row["core_opened_at"]))
                ),
                tactical_quantity=Decimal(str(row.get("tactical_quantity", "0"))),
                tactical_average=Decimal(str(row.get("tactical_average", "0"))),
                tactical_opened_at=(
                    None
                    if row.get("tactical_opened_at") in (None, "")
                    else datetime.fromisoformat(str(row["tactical_opened_at"]))
                ),
            )

    def _restore_checkpoint(self: Any, payload: Mapping[str, Any]) -> None:
        self.long_state = self._decode_leg_state(payload.get("long_state"), PositionSide.LONG)
        self.short_state = self._decode_leg_state(payload.get("short_state"), PositionSide.SHORT)
        self._last_market = self._decode_market(payload.get("last_market"))
        self._last_bar = self._decode_bar(payload.get("last_bar"))
        if self._last_bar is not None:
            if self._last_market is None or self._last_bar.timestamp != self._last_market.timestamp:
                raise ValueError("paper checkpoint market/bar cursor mismatch")
            if (
                self._last_bar.symbol != self._last_market.symbol
                or self._last_bar.close != self._last_market.mark
            ):
                raise ValueError("paper checkpoint market/bar values mismatch")
        self._funding_balance_delta = Decimal(str(payload.get("funding_balance_delta", "0")))
        raw_funding_time = payload.get("last_funding_event_time")
        self._last_funding_event_time = (
            None
            if raw_funding_time in (None, "")
            else datetime.fromisoformat(str(raw_funding_time))
        )
        if (
            self._last_funding_event_time is not None
            and self._last_funding_event_time.tzinfo is None
        ):
            self._last_funding_event_time = self._last_funding_event_time.replace(tzinfo=UTC)
        event_ids = payload.get("applied_account_event_ids", ())
        if isinstance(event_ids, (str, bytes)):
            raise TypeError("paper account event ids must be a sequence")
        self._applied_account_event_ids = {str(item) for item in event_ids}
