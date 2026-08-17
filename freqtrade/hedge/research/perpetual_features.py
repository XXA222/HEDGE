"""Point-in-time perpetual-market features for research model observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from itertools import pairwise

from freqtrade.hedge.contracts import finite_decimal


def _aware(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _optional_decimal(value: Decimal | None, *, field_name: str) -> Decimal | None:
    return None if value is None else finite_decimal(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class PerpetualMarketObservation:
    event_time: datetime
    available_at: datetime
    close: Decimal
    funding_rate: Decimal | None = None
    mark_price: Decimal | None = None
    index_price: Decimal | None = None
    open_interest: Decimal | None = None
    bid: Decimal | None = None
    ask: Decimal | None = None
    aggressive_buy_notional: Decimal | None = None
    aggressive_sell_notional: Decimal | None = None
    long_liquidation_notional: Decimal | None = None
    short_liquidation_notional: Decimal | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_time", _aware(self.event_time, field_name="event_time"))
        object.__setattr__(
            self,
            "available_at",
            _aware(self.available_at, field_name="available_at"),
        )
        if self.available_at < self.event_time:
            raise ValueError("available_at cannot precede event_time")
        object.__setattr__(self, "close", finite_decimal(self.close, field_name="close"))
        if self.close <= 0:
            raise ValueError("close must be positive")
        nonnegative = {
            "mark_price",
            "index_price",
            "open_interest",
            "bid",
            "ask",
            "aggressive_buy_notional",
            "aggressive_sell_notional",
            "long_liquidation_notional",
            "short_liquidation_notional",
        }
        for field_name in tuple(self.__dataclass_fields__)[3:]:
            value = _optional_decimal(getattr(self, field_name), field_name=field_name)
            if field_name in nonnegative and value is not None and value < 0:
                raise ValueError(f"{field_name} must be nonnegative")
            object.__setattr__(self, field_name, value)
        if self.bid is not None and self.ask is not None and self.ask < self.bid:
            raise ValueError("ask cannot be below bid")


@dataclass(frozen=True, slots=True)
class PerpetualFeatureSnapshot:
    decision_time: datetime
    source_observation_count: int
    funding_rate: Decimal | None
    funding_zscore: Decimal | None
    funding_acceleration: Decimal | None
    mark_index_basis: Decimal | None
    basis_zscore: Decimal | None
    basis_volatility: Decimal | None
    open_interest: Decimal | None
    open_interest_delta: Decimal | None
    price_open_interest_divergence: Decimal | None
    realized_volatility: Decimal | None
    spread_proxy: Decimal | None
    agg_trade_imbalance: Decimal | None
    trade_intensity: Decimal | None
    aggressive_buy_pressure: Decimal | None
    aggressive_sell_pressure: Decimal | None
    liquidation_pressure: Decimal | None
    missing_features: tuple[str, ...]


def _mean(values: tuple[Decimal, ...]) -> Decimal:
    return sum(values, Decimal(0)) / Decimal(len(values))


def _std(values: tuple[Decimal, ...]) -> Decimal:
    if len(values) < 2:
        return Decimal(0)
    average = _mean(values)
    return _mean(tuple((value - average) ** 2 for value in values)).sqrt()


def _zscore(values: tuple[Decimal, ...]) -> Decimal | None:
    if len(values) < 2:
        return None
    deviation = _std(values)
    return Decimal(0) if deviation == 0 else (values[-1] - _mean(values)) / deviation


def _ratio_delta(current: Decimal, previous: Decimal) -> Decimal | None:
    return None if previous == 0 else current / previous - Decimal(1)


def _basis(row: PerpetualMarketObservation) -> Decimal | None:
    if row.mark_price is None or row.index_price in {None, Decimal(0)}:
        return None
    return (row.mark_price - row.index_price) / row.index_price


def _pressure(buy: Decimal | None, sell: Decimal | None) -> tuple[Decimal | None, ...]:
    if buy is None or sell is None or buy + sell == 0:
        return None, None, None, None
    total = buy + sell
    return (buy - sell) / total, total, buy / total, sell / total


def build_perpetual_feature_snapshot(
    observations: tuple[PerpetualMarketObservation, ...],
    *,
    decision_time: datetime,
    window: int = 20,
) -> PerpetualFeatureSnapshot:
    decision = _aware(decision_time, field_name="decision_time")
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be an int of at least 2")
    usable = tuple(sorted(
        (row for row in observations if row.available_at <= decision),
        key=lambda row: (row.event_time, row.available_at),
    ))
    if not usable:
        raise ValueError("no observations are available at decision_time")
    rows = usable[-window:]
    latest = rows[-1]
    funding = tuple(row.funding_rate for row in rows if row.funding_rate is not None)
    bases = tuple(value for row in rows if (value := _basis(row)) is not None)
    closes = tuple(row.close for row in rows)
    returns = tuple(
        value
        for previous, current in pairwise(closes)
        if (value := _ratio_delta(current, previous)) is not None
    )
    oi = tuple(row.open_interest for row in rows if row.open_interest is not None)
    price_delta = _ratio_delta(closes[-1], closes[-2]) if len(closes) >= 2 else None
    oi_delta = _ratio_delta(oi[-1], oi[-2]) if len(oi) >= 2 else None
    spread = None
    if latest.bid is not None and latest.ask is not None and latest.bid + latest.ask > 0:
        spread = (latest.ask - latest.bid) / ((latest.ask + latest.bid) / Decimal(2))
    imbalance, intensity, buy_pressure, sell_pressure = _pressure(
        latest.aggressive_buy_notional,
        latest.aggressive_sell_notional,
    )
    liquidation, _, _, _ = _pressure(
        latest.long_liquidation_notional,
        latest.short_liquidation_notional,
    )
    values: dict[str, Decimal | None] = {
        "funding_rate": latest.funding_rate,
        "funding_zscore": _zscore(funding),
        "funding_acceleration": funding[-1] - funding[-2] if len(funding) >= 2 else None,
        "mark_index_basis": _basis(latest),
        "basis_zscore": _zscore(bases),
        "basis_volatility": _std(bases) if len(bases) >= 2 else None,
        "open_interest": latest.open_interest,
        "open_interest_delta": oi_delta,
        "price_open_interest_divergence": (
            price_delta - oi_delta if price_delta is not None and oi_delta is not None else None
        ),
        "realized_volatility": _std(returns) if len(returns) >= 2 else None,
        "spread_proxy": spread,
        "agg_trade_imbalance": imbalance,
        "trade_intensity": intensity,
        "aggressive_buy_pressure": buy_pressure,
        "aggressive_sell_pressure": sell_pressure,
        "liquidation_pressure": liquidation,
    }
    return PerpetualFeatureSnapshot(
        decision_time=decision,
        source_observation_count=len(rows),
        **values,
        missing_features=tuple(name for name, value in values.items() if value is None),
    )
