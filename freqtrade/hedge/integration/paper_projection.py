"""Position and account projections for the integrated Paper runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from freqtrade.enums.hedge import PositionAction as RiskAction
from freqtrade.enums.hedge import PositionSide as RiskPositionSide
from freqtrade.hedge.contracts.business_identity import BusinessIdentity
from freqtrade.hedge.execution.service import PositionSide as ExecutionSide
from freqtrade.hedge.numeric import ZERO
from freqtrade.hedge.planning.context import (
    ActiveOrder,
    IntentAction,
    LegPosition,
    MarketSnapshot,
    OrderSide,
    PositionBucket,
    PositionSide,
    TacticalLot,
    WalletSnapshot,
)
from freqtrade.hedge.risk.models import PendingOrderRisk
from freqtrade.hedge.risk.portfolio import (
    PositionRiskLeg,
    RiskPortfolioSnapshot,
    build_risk_portfolio,
)


ONE = Decimal(1)


@dataclass(slots=True)
class _BusinessLotState:
    identity: BusinessIdentity
    bucket: PositionBucket
    quantity: Decimal
    average_price: Decimal
    opened_at: datetime


@dataclass(slots=True)
class _BucketState:
    """Core/tactical aggregate cache plus exact business-lot ownership."""

    core_quantity: Decimal = ZERO
    core_average: Decimal = ZERO
    core_opened_at: datetime | None = None
    tactical_quantity: Decimal = ZERO
    tactical_average: Decimal = ZERO
    tactical_opened_at: datetime | None = None
    business_lots: dict[str, _BusinessLotState] = field(default_factory=dict)

    @staticmethod
    def _aware(value: datetime | None) -> datetime:
        observed = datetime.now(UTC) if value is None else value
        if observed.tzinfo is None:
            return observed.replace(tzinfo=UTC)
        return observed.astimezone(UTC)

    def _recompute(self) -> None:
        core = [item for item in self.business_lots.values() if item.bucket is PositionBucket.CORE]
        tactical = [
            item for item in self.business_lots.values() if item.bucket is PositionBucket.TACTICAL
        ]
        if core:
            self.core_quantity = sum((item.quantity for item in core), ZERO)
            self.core_average = (
                sum((item.quantity * item.average_price for item in core), ZERO)
                / self.core_quantity
            )
            self.core_opened_at = min(item.opened_at for item in core)
        elif self.business_lots:
            self.core_quantity = ZERO
            self.core_average = ZERO
            self.core_opened_at = None
        if tactical:
            self.tactical_quantity = sum((item.quantity for item in tactical), ZERO)
            self.tactical_average = (
                sum((item.quantity * item.average_price for item in tactical), ZERO)
                / self.tactical_quantity
            )
            self.tactical_opened_at = min(item.opened_at for item in tactical)
        elif self.business_lots:
            self.tactical_quantity = ZERO
            self.tactical_average = ZERO
            self.tactical_opened_at = None

    def increase(
        self,
        bucket: PositionBucket,
        quantity: Decimal,
        price: Decimal,
        opened_at: datetime | None = None,
        *,
        business_identity: BusinessIdentity | None = None,
    ) -> None:
        observed = self._aware(opened_at)
        if business_identity is not None:
            key = str(business_identity.business_lot_id)
            current = self.business_lots.get(key)
            if current is None:
                self.business_lots[key] = _BusinessLotState(
                    identity=business_identity,
                    bucket=bucket,
                    quantity=quantity,
                    average_price=price,
                    opened_at=observed,
                )
            else:
                if current.identity != business_identity or current.bucket is not bucket:
                    raise ValueError("business lot identity/bucket changed during fill projection")
                total = current.quantity + quantity
                current.average_price = (
                    current.quantity * current.average_price + quantity * price
                ) / total
                current.quantity = total
            self._recompute()
            return

        # Legacy checkpoint compatibility. New managed execution always supplies identity.
        if bucket is PositionBucket.CORE:
            total = self.core_quantity + quantity
            self.core_average = (self.core_quantity * self.core_average + quantity * price) / total
            if self.core_quantity == ZERO:
                self.core_opened_at = observed
            self.core_quantity = total
        else:
            total = self.tactical_quantity + quantity
            self.tactical_average = (
                self.tactical_quantity * self.tactical_average + quantity * price
            ) / total
            if self.tactical_quantity == ZERO:
                self.tactical_opened_at = observed
            self.tactical_quantity = total

    def reduce(
        self,
        bucket: PositionBucket,
        quantity: Decimal,
        *,
        business_identity: BusinessIdentity | None = None,
    ) -> None:
        if business_identity is not None:
            key = str(business_identity.business_lot_id)
            current = self.business_lots.get(key)
            if current is None or current.bucket is not bucket:
                raise ValueError("target business lot is not open in the Paper projection")
            if quantity > current.quantity:
                raise ValueError("targeted business-lot reduction exceeds open quantity")
            current.quantity -= quantity
            if current.quantity == ZERO:
                self.business_lots.pop(key, None)
            self._recompute()
            return

        # Legacy aggregate behavior is retained only for pre-identity checkpoint recovery.
        remaining = quantity
        if bucket is PositionBucket.CORE:
            used = min(remaining, self.core_quantity)
            self.core_quantity -= used
            remaining -= used
            if self.core_quantity == ZERO:
                self.core_average = ZERO
                self.core_opened_at = None
        else:
            used = min(remaining, self.tactical_quantity)
            self.tactical_quantity -= used
            remaining -= used
            if self.tactical_quantity == ZERO:
                self.tactical_average = ZERO
                self.tactical_opened_at = None
        if remaining > ZERO:
            if bucket is PositionBucket.CORE:
                self.tactical_quantity = max(self.tactical_quantity - remaining, ZERO)
                if self.tactical_quantity == ZERO:
                    self.tactical_average = ZERO
                    self.tactical_opened_at = None
            else:
                self.core_quantity = max(self.core_quantity - remaining, ZERO)
                if self.core_quantity == ZERO:
                    self.core_average = ZERO
                    self.core_opened_at = None

    def position_lots(self) -> tuple[TacticalLot, ...]:
        return tuple(
            TacticalLot(
                lot_id=str(item.identity.business_lot_id),
                quantity=item.quantity,
                average_price=item.average_price,
                opened_at=item.opened_at,
                business_identity=item.identity,
                bucket=item.bucket,
            )
            for item in sorted(
                self.business_lots.values(),
                key=lambda value: (value.identity.business_trade_seq, value.identity.lot_index),
            )
            if item.quantity > ZERO
        )

    def restore_business_lots(self, payload: object) -> None:
        if payload in (None, ()):
            return
        if not isinstance(payload, list):
            raise TypeError("paper business_lots checkpoint must be a list")
        restored: dict[str, _BusinessLotState] = {}
        for raw in payload:
            if not isinstance(raw, dict):
                raise TypeError("paper business lot row must be a mapping")
            identity = BusinessIdentity(
                business_trade_id=raw["business_trade_id"],
                business_trade_seq=int(raw["business_trade_seq"]),
                business_lot_id=raw["business_lot_id"],
                lot_index=int(raw.get("lot_index", 1)),
                account_id=str(raw["account_id"]),
                symbol=str(raw["symbol"]),
                position_side=str(raw["position_side"]),
            )
            quantity = Decimal(str(raw["quantity"]))
            if quantity <= ZERO:
                continue
            state = _BusinessLotState(
                identity=identity,
                bucket=PositionBucket(str(raw["bucket"])),
                quantity=quantity,
                average_price=Decimal(str(raw["average_price"])),
                opened_at=self._aware(datetime.fromisoformat(str(raw["opened_at"]))),
            )
            restored[str(identity.business_lot_id)] = state
        self.business_lots = restored
        if restored:
            self._recompute()

    def encode_business_lots(self) -> list[dict[str, object]]:
        return [
            {
                "business_trade_id": str(item.identity.business_trade_id),
                "business_trade_seq": item.identity.business_trade_seq,
                "business_lot_id": str(item.identity.business_lot_id),
                "lot_index": item.identity.lot_index,
                "account_id": item.identity.account_id,
                "symbol": item.identity.symbol,
                "position_side": item.identity.position_side,
                "bucket": item.bucket.value,
                "quantity": str(item.quantity),
                "average_price": str(item.average_price),
                "opened_at": item.opened_at.isoformat(),
            }
            for item in self.business_lots.values()
        ]


class PaperStateProjectionMixin:
    """Position, order, wallet, and risk projections for Paper runtime."""

    def _fake_leg(self: Any, side: PositionSide) -> Any:
        return self._execution().account.leg(
            account_id=self.account_id,
            symbol=self.execution_symbol,
            position_side=ExecutionSide(side.value),
        )

    def _leg(self: Any, side: PositionSide) -> LegPosition:
        fake = self._fake_leg(side)
        bucket = self._bucket[side]
        exact_lots = bucket.position_lots()
        bucket_total = bucket.core_quantity + bucket.tactical_quantity
        if exact_lots and bucket_total != fake.quantity:
            raise RuntimeError(
                "Paper business-lot sum differs from authoritative execution quantity"
            )
        if not exact_lots and bucket_total != fake.quantity:
            delta = fake.quantity - bucket_total
            if delta > ZERO:
                combined = bucket.tactical_quantity + delta
                bucket.tactical_average = fake.average_price if combined > ZERO else ZERO
                bucket.tactical_quantity = combined
            elif delta < ZERO:
                bucket.reduce(PositionBucket.TACTICAL, -delta)
        return LegPosition(
            side=side,
            quantity=fake.quantity,
            average_price=fake.average_price,
            core_quantity=bucket.core_quantity,
            core_average_price=bucket.core_average,
            tactical_quantity=bucket.tactical_quantity,
            tactical_average_price=bucket.tactical_average,
            realized_pnl=fake.realized_pnl,
            tactical_realized_pnl=fake.realized_pnl,
            tactical_lots=tuple(
                item for item in exact_lots if item.bucket is PositionBucket.TACTICAL
            ),
            position_lots=exact_lots,
        )

    def _active_execution_orders(self: Any) -> tuple[Any, ...]:
        return self._execution().core.list_orders(
            account_id=self.account_id,
            symbol=self.execution_symbol,
            include_terminal=False,
        )

    def _prune_planner_order_map(self: Any) -> None:
        if not self._planner_order_to_client:
            return
        live_client_ids = {order.client_order_id for order in self._active_execution_orders()}
        for planner_id, client_id in tuple(self._planner_order_to_client.items()):
            if client_id not in live_client_ids:
                self._planner_order_to_client.pop(planner_id, None)

    def collection_gauges(self: Any) -> dict[str, int]:
        gauges = {
            "planner_order_map": len(self._planner_order_to_client),
            "simulation_intents": len(self._simulation_intents),
            "applied_account_events": len(self._applied_account_event_ids),
        }
        execution = self.execution
        if execution is not None:
            store_gauges = getattr(execution.store, "collection_gauges", None)
            if callable(store_gauges):
                gauges.update(store_gauges())
            exchange_gauges = getattr(execution.exchange, "collection_gauges", None)
            if callable(exchange_gauges):
                gauges.update(
                    {f"exchange_{key}": value for key, value in exchange_gauges().items()}
                )
        return gauges

    def _active_orders(self: Any) -> tuple[ActiveOrder, ...]:
        rows: list[ActiveOrder] = []
        for order in self._active_execution_orders():
            price = order.intent.limit_price
            if price is None:
                raw = order.intent.metadata.get("reference_price")
                if raw is None:
                    continue
                price = Decimal(str(raw))
            rows.append(
                ActiveOrder(
                    order_id=str(
                        order.intent.metadata.get("planner_intent_id", order.client_order_id)
                    ),
                    client_order_id=order.client_order_id,
                    symbol=self.symbol,
                    position_side=PositionSide(order.intent.position_side.value),
                    order_side=(
                        OrderSide.BUY
                        if (
                            order.intent.position_side is ExecutionSide.LONG
                            and not order.intent.reduces_risk
                        )
                        or (
                            order.intent.position_side is ExecutionSide.SHORT
                            and order.intent.reduces_risk
                        )
                        else OrderSide.SELL
                    ),
                    quantity=order.approved_quantity - order.lifecycle.filled_quantity,
                    price=price,
                    reduce_only=order.intent.reduce_only,
                    bucket=PositionBucket(str(order.intent.metadata.get("bucket", "TACTICAL"))),
                    action=IntentAction(
                        str(order.intent.metadata.get("strategy_action", order.intent.action.value))
                    ),
                    created_at=order.created_at,
                    layer=int(order.intent.metadata.get("layer", 0)),
                    business_identity=order.intent.business_identity,
                    order_role=order.intent.order_role,
                    order_revision=int(order.intent.order_revision),
                    strategy_entry_key=str(
                        order.intent.metadata.get("strategy_entry_key", "")
                    ),
                )
            )
        return tuple(rows)

    def wallet(self: Any, market: MarketSnapshot | None = None) -> WalletSnapshot:
        current_market = market or self.last_market
        mark = Decimal(1) if current_market is None else current_market.mark
        long = self._leg(PositionSide.LONG)
        short = self._leg(PositionSide.SHORT)
        fake_long = self._fake_leg(PositionSide.LONG)
        fake_short = self._fake_leg(PositionSide.SHORT)
        active_orders = self._active_orders()
        realized = long.realized_pnl + short.realized_pnl
        fees = fake_long.fees + fake_short.fees
        unrealized = long.unrealized_pnl(mark) + short.unrealized_pnl(mark)
        balance = self.initial_balance + realized - fees + self._funding_balance_delta
        equity = balance + unrealized
        initial_margin = (long.quantity + short.quantity) * mark / self.leverage
        pending_margin = sum(
            (item.notional / self.leverage for item in active_orders if not item.reduce_only),
            ZERO,
        )
        available = max(equity - initial_margin - pending_margin, ZERO)
        return WalletSnapshot(
            balance=balance,
            equity=max(equity, Decimal("0.00000001")),
            available_balance=available,
            long=long,
            short=short,
            active_orders=active_orders,
            leverage=self.leverage,
        )

    def risk_portfolio(self: Any) -> RiskPortfolioSnapshot:
        market = self._cycle_market or self.last_market
        mark = Decimal(1) if market is None else market.mark
        wallet = self.wallet(market)
        positions: list[PositionRiskLeg] = []
        for side, leg in (
            (RiskPositionSide.LONG, wallet.long),
            (RiskPositionSide.SHORT, wallet.short),
        ):
            if leg.quantity <= ZERO:
                continue
            maintenance = leg.quantity * mark * self.planner_config.maintenance_margin_rate
            liquidation = (
                max(
                    leg.average_price
                    * (ONE - ONE / self.leverage + self.planner_config.maintenance_margin_rate),
                    Decimal("0.00000001"),
                )
                if side is RiskPositionSide.LONG
                else leg.average_price
                * (ONE + ONE / self.leverage - self.planner_config.maintenance_margin_rate)
            )
            positions.append(
                PositionRiskLeg(
                    account_id=self.account_id,
                    symbol=self.symbol,
                    position_side=side,
                    quantity=leg.quantity,
                    mark_price=mark,
                    leverage=self.leverage,
                    maintenance_margin=maintenance,
                    liquidation_price=liquidation,
                )
            )
        pending: list[PendingOrderRisk] = []
        for order in self._execution().core.list_orders(
            account_id=self.account_id,
            symbol=self.execution_symbol,
            include_terminal=False,
        ):
            pending.append(
                PendingOrderRisk(
                    account_id=self.account_id,
                    symbol=self.symbol,
                    position_side=RiskPositionSide(order.intent.position_side.value),
                    action=RiskAction(order.intent.action.value),
                    remaining_quantity=order.approved_quantity - order.lifecycle.filled_quantity,
                    reference_price=order.intent.limit_price or mark,
                    leverage=self.leverage,
                    maintenance_margin_rate=self.planner_config.maintenance_margin_rate,
                )
            )
        return build_risk_portfolio(
            account_id=self.account_id,
            equity=max(wallet.equity, Decimal("0.00000001")),
            wallet_balance=max(wallet.balance, ZERO),
            available_balance=wallet.available_balance,
            positions=positions,
            pending_orders=pending,
            strict_completeness=True,
        )
