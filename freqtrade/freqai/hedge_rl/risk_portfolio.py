"""Deterministic cross-wallet simulator for target risk-level Hedge actions.

V3 preserves independent LONG/SHORT books while adding side-specific transition
accounting, next-open sizing equity, and a same-level rebalance deadband.  The
additional fields are diagnostic/reward facts only; the primary account equity remains
the single source of PnL truth.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import IntEnum

from .risk_levels import HedgeRiskLevelAction, RiskLevelMapper, RiskLevelProfile, TargetExposure


class LegSide(IntEnum):
    LONG = 1
    SHORT = -1


@dataclass(frozen=True, slots=True)
class RiskLegState:
    side: LegSide
    quantity: float = 0.0
    average_price: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    funding_paid: float = 0.0

    def notional(self, mark: float) -> float:
        return abs(self.quantity) * float(mark)

    def unrealized_pnl(self, mark: float) -> float:
        if self.quantity <= 0:
            return 0.0
        return (float(mark) - self.average_price) * self.quantity * int(self.side)

    def net_pnl(self, mark: float) -> float:
        """Cumulative side PnL after fees and funding, including unrealized PnL."""

        return self.realized_pnl + self.unrealized_pnl(mark) - self.fees_paid - self.funding_paid


@dataclass(frozen=True, slots=True)
class RiskAccountState:
    cash_balance: float
    equity: float
    peak_equity: float
    long: RiskLegState
    short: RiskLegState
    long_level: int = 0
    short_level: int = 0
    step: int = 0
    turnover: float = 0.0

    @classmethod
    def initial(cls, equity: float) -> RiskAccountState:
        value = float(equity)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("starting equity must be finite and positive")
        return cls(
            cash_balance=value,
            equity=value,
            peak_equity=value,
            long=RiskLegState(LegSide.LONG),
            short=RiskLegState(LegSide.SHORT),
        )

    def drawdown(self) -> float:
        return max(0.0, (self.peak_equity - self.equity) / max(self.peak_equity, 1e-12))

    def gross_notional_ratio(self, mark: float) -> float:
        return (self.long.notional(mark) + self.short.notional(mark)) / max(abs(self.equity), 1e-12)

    def net_notional_ratio(self, mark: float) -> float:
        return (self.long.notional(mark) - self.short.notional(mark)) / max(abs(self.equity), 1e-12)

    def long_margin_fraction(self, mark: float, profile: RiskLevelProfile) -> float:
        return self.long.notional(mark) / profile.long_leverage / max(abs(self.equity), 1e-12)

    def short_margin_fraction(self, mark: float, profile: RiskLevelProfile) -> float:
        return self.short.notional(mark) / profile.short_leverage / max(abs(self.equity), 1e-12)

    def used_margin_fraction(self, mark: float, profile: RiskLevelProfile) -> float:
        return self.long_margin_fraction(mark, profile) + self.short_margin_fraction(mark, profile)

    def reserve_margin_fraction(self, mark: float, profile: RiskLevelProfile) -> float:
        return max(0.0, 1.0 - self.used_margin_fraction(mark, profile))


@dataclass(frozen=True, slots=True)
class RiskPortfolioTransition:
    previous_equity: float
    sizing_equity: float
    equity: float
    previous_drawdown: float
    drawdown: float
    realized_pnl: float
    long_realized_pnl: float
    short_realized_pnl: float
    previous_long_unrealized: float
    previous_short_unrealized: float
    previous_long_net_pnl: float
    previous_short_net_pnl: float
    long_unrealized: float
    short_unrealized: float
    fees: float
    long_fee: float
    short_fee: float
    slippage_cost: float
    long_slippage_cost: float
    short_slippage_cost: float
    funding_cashflow: float
    long_funding_cashflow: float
    short_funding_cashflow: float
    traded_notional: float
    long_traded_notional: float
    short_traded_notional: float
    long_quantity_delta: float
    short_quantity_delta: float
    previous_long_level: int
    previous_short_level: int
    long_level: int
    short_level: int
    previous_long_margin_fraction: float
    previous_short_margin_fraction: float
    long_margin_fraction: float
    short_margin_fraction: float
    previous_used_margin_fraction: float
    used_margin_fraction: float
    reference_price: float
    previous_mark_price: float
    mark_price: float
    market_return: float
    target: TargetExposure

    @property
    def level_distance(self) -> int:
        return abs(self.long_level - self.previous_long_level) + abs(
            self.short_level - self.previous_short_level
        )

    @property
    def gross_margin_delta(self) -> float:
        return self.used_margin_fraction - self.previous_used_margin_fraction

    @property
    def long_step_net_pnl(self) -> float:
        return (
            self.long_realized_pnl
            + self.long_unrealized
            - self.previous_long_unrealized
            - self.long_fee
            + self.long_funding_cashflow
        )

    @property
    def short_step_net_pnl(self) -> float:
        return (
            self.short_realized_pnl
            + self.short_unrealized
            - self.previous_short_unrealized
            - self.short_fee
            + self.short_funding_cashflow
        )

    @property
    def accounting_cost(self) -> float:
        # Slippage is already embedded in fill-price PnL.  Expose it diagnostically but
        # do not add it to fee/funding here, avoiding a second accounting path.
        return self.fees + max(0.0, -self.funding_cashflow)


@dataclass(frozen=True, slots=True)
class _LegFill:
    leg: RiskLegState
    realized_pnl: float
    fee: float
    slippage_cost: float
    turnover: float
    quantity_delta: float


class TargetLevelPortfolioSimulator:
    """Apply target risk caps without implicit same-level scale-in.

    Raising a level may add risk and lowering a level reduces toward the new target.
    Keeping the same level never increases quantity merely because price/equity moved;
    it may only trim an over-budget leg outside the configured deadband.
    """

    def __init__(
        self,
        starting_equity: float,
        *,
        profile: RiskLevelProfile,
        fee_rate: float = 0.0004,
        slippage_bps: float = 1.0,
    ) -> None:
        if not 0 <= float(fee_rate) < 0.1:
            raise ValueError("fee_rate must be within [0, 0.1)")
        if not math.isfinite(float(slippage_bps)) or not 0 <= float(slippage_bps) < 10_000:
            raise ValueError("slippage_bps must be finite and within [0, 10000)")
        self.profile = profile
        self.mapper = RiskLevelMapper(profile)
        self.fee_rate = float(fee_rate)
        self.slippage_bps = float(slippage_bps)
        self.state = RiskAccountState.initial(starting_equity)
        self._last_mark_price: float | None = None

    def reset(self, equity: float | None = None) -> RiskAccountState:
        self.state = RiskAccountState.initial(self.state.peak_equity if equity is None else equity)
        self._last_mark_price = None
        return self.state

    def _fill_price(self, reference_price: float, *, is_buy: bool) -> float:
        slip = self.slippage_bps / 10_000.0
        return reference_price * (1.0 + slip if is_buy else 1.0 - slip)

    def _adjust_leg(
        self,
        leg: RiskLegState,
        *,
        target_notional: float,
        reference_price: float,
        deadband_notional: float = 0.0,
        allow_increase: bool = True,
    ) -> _LegFill:
        target_quantity = max(0.0, target_notional / reference_price)
        delta = target_quantity - leg.quantity
        delta_notional = abs(delta) * reference_price
        if abs(delta) <= 1e-15 or delta_notional <= max(0.0, deadband_notional):
            return _LegFill(leg, 0.0, 0.0, 0.0, 0.0, 0.0)
        increasing = delta > 0
        if increasing and not allow_increase:
            return _LegFill(leg, 0.0, 0.0, 0.0, 0.0, 0.0)
        quantity = abs(delta)
        # LONG increase = buy, LONG reduce = sell; SHORT increase = sell, SHORT reduce = buy.
        is_buy = increasing if leg.side is LegSide.LONG else not increasing
        fill_price = self._fill_price(reference_price, is_buy=is_buy)
        notional = quantity * fill_price
        fee = notional * self.fee_rate
        slippage = quantity * abs(fill_price - reference_price)
        realized = 0.0
        if increasing:
            new_quantity = leg.quantity + quantity
            if leg.quantity <= 1e-15:
                average = fill_price
            else:
                average = (leg.average_price * leg.quantity + fill_price * quantity) / new_quantity
        else:
            quantity = min(quantity, leg.quantity)
            new_quantity = max(0.0, leg.quantity - quantity)
            realized = (fill_price - leg.average_price) * quantity * int(leg.side)
            average = leg.average_price if new_quantity > 1e-15 else 0.0
        updated = replace(
            leg,
            quantity=new_quantity,
            average_price=average,
            realized_pnl=leg.realized_pnl + realized,
            fees_paid=leg.fees_paid + fee,
        )
        signed_delta = quantity if increasing else -quantity
        return _LegFill(updated, realized, fee, slippage, notional, signed_delta)

    @staticmethod
    def _funding_cashflow(side: LegSide, notional: float, funding_rate: float) -> float:
        # Positive Binance-style funding: LONG pays, SHORT receives.
        return -float(side) * float(notional) * float(funding_rate)

    def apply_target(
        self,
        action: Sequence[int] | HedgeRiskLevelAction,
        *,
        reference_price: float,
        mark_price: float | None = None,
        funding_rate: float = 0.0,
    ) -> RiskPortfolioTransition:
        ref = float(reference_price)
        mark = ref if mark_price is None else float(mark_price)
        if not all(math.isfinite(item) and item > 0 for item in (ref, mark)):
            raise ValueError("reference and mark prices must be finite and positive")
        if not math.isfinite(float(funding_rate)):
            raise ValueError("funding_rate must be finite")

        previous = self.state
        selected = HedgeRiskLevelAction.from_value(action)
        previous_mark = ref if self._last_mark_price is None else self._last_mark_price
        previous_long_u = previous.long.unrealized_pnl(previous_mark)
        previous_short_u = previous.short.unrealized_pnl(previous_mark)
        previous_long_net = previous.long.net_pnl(previous_mark)
        previous_short_net = previous.short.net_pnl(previous_mark)
        previous_drawdown = previous.drawdown()
        previous_long_margin = previous.long_margin_fraction(previous_mark, self.profile)
        previous_short_margin = previous.short_margin_fraction(previous_mark, self.profile)
        previous_used_margin = previous_long_margin + previous_short_margin

        # Mark the existing book to next-bar open before sizing the new target.  This avoids
        # using stale prior-close equity after a large gap while reward still captures the gap
        # from previous_equity to the final next-bar-close equity.
        open_equity = (
            previous.cash_balance
            + previous.long.unrealized_pnl(ref)
            + previous.short.unrealized_pnl(ref)
        )
        sizing_equity = max(open_equity, 1e-12)
        target = self.mapper.map(selected, equity=sizing_equity)
        deadband_notional = sizing_equity * self.profile.rebalance_deadband_fraction
        long_deadband = (
            deadband_notional if int(selected.long_level) == previous.long_level else 0.0
        )
        short_deadband = (
            deadband_notional if int(selected.short_level) == previous.short_level else 0.0
        )

        long_fill = self._adjust_leg(
            previous.long,
            target_notional=target.long_target_notional,
            reference_price=ref,
            deadband_notional=long_deadband,
            allow_increase=int(selected.long_level) > previous.long_level,
        )
        short_fill = self._adjust_leg(
            previous.short,
            target_notional=target.short_target_notional,
            reference_price=ref,
            deadband_notional=short_deadband,
            allow_increase=int(selected.short_level) > previous.short_level,
        )
        realized = long_fill.realized_pnl + short_fill.realized_pnl
        fees = long_fill.fee + short_fill.fee
        cash = previous.cash_balance + realized - fees

        long_funding = self._funding_cashflow(
            LegSide.LONG, long_fill.leg.notional(mark), funding_rate
        )
        short_funding = self._funding_cashflow(
            LegSide.SHORT, short_fill.leg.notional(mark), funding_rate
        )
        funding_cashflow = long_funding + short_funding
        cash += funding_cashflow
        long_leg = replace(long_fill.leg, funding_paid=long_fill.leg.funding_paid - long_funding)
        short_leg = replace(
            short_fill.leg,
            funding_paid=short_fill.leg.funding_paid - short_funding,
        )
        long_u = long_leg.unrealized_pnl(mark)
        short_u = short_leg.unrealized_pnl(mark)
        equity = cash + long_u + short_u
        peak = max(previous.peak_equity, equity)
        turnover = long_fill.turnover + short_fill.turnover
        self.state = RiskAccountState(
            cash_balance=cash,
            equity=equity,
            peak_equity=peak,
            long=long_leg,
            short=short_leg,
            long_level=int(selected.long_level),
            short_level=int(selected.short_level),
            step=previous.step + 1,
            turnover=previous.turnover + turnover,
        )
        self._last_mark_price = mark
        current_long_margin = self.state.long_margin_fraction(mark, self.profile)
        current_short_margin = self.state.short_margin_fraction(mark, self.profile)
        current_used_margin = current_long_margin + current_short_margin
        return RiskPortfolioTransition(
            previous_equity=previous.equity,
            sizing_equity=sizing_equity,
            equity=equity,
            previous_drawdown=previous_drawdown,
            drawdown=self.state.drawdown(),
            realized_pnl=realized,
            long_realized_pnl=long_fill.realized_pnl,
            short_realized_pnl=short_fill.realized_pnl,
            previous_long_unrealized=previous_long_u,
            previous_short_unrealized=previous_short_u,
            previous_long_net_pnl=previous_long_net,
            previous_short_net_pnl=previous_short_net,
            long_unrealized=long_u,
            short_unrealized=short_u,
            fees=fees,
            long_fee=long_fill.fee,
            short_fee=short_fill.fee,
            slippage_cost=long_fill.slippage_cost + short_fill.slippage_cost,
            long_slippage_cost=long_fill.slippage_cost,
            short_slippage_cost=short_fill.slippage_cost,
            funding_cashflow=funding_cashflow,
            long_funding_cashflow=long_funding,
            short_funding_cashflow=short_funding,
            traded_notional=turnover,
            long_traded_notional=long_fill.turnover,
            short_traded_notional=short_fill.turnover,
            long_quantity_delta=long_fill.quantity_delta,
            short_quantity_delta=short_fill.quantity_delta,
            previous_long_level=previous.long_level,
            previous_short_level=previous.short_level,
            long_level=int(selected.long_level),
            short_level=int(selected.short_level),
            previous_long_margin_fraction=previous_long_margin,
            previous_short_margin_fraction=previous_short_margin,
            long_margin_fraction=current_long_margin,
            short_margin_fraction=current_short_margin,
            previous_used_margin_fraction=previous_used_margin,
            used_margin_fraction=current_used_margin,
            reference_price=ref,
            previous_mark_price=previous_mark,
            mark_price=mark,
            market_return=mark / previous_mark - 1.0,
            target=target,
        )

    def mark_to_market(self, mark_price: float) -> RiskAccountState:
        mark = float(mark_price)
        if not math.isfinite(mark) or mark <= 0:
            raise ValueError("mark_price must be finite and positive")
        current = self.state
        equity = (
            current.cash_balance
            + current.long.unrealized_pnl(mark)
            + current.short.unrealized_pnl(mark)
        )
        self.state = replace(current, equity=equity, peak_equity=max(current.peak_equity, equity))
        self._last_mark_price = mark
        return self.state
