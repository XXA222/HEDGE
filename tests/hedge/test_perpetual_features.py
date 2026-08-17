from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from freqtrade.hedge.research.perpetual_features import (
    PerpetualMarketObservation,
    build_perpetual_feature_snapshot,
)


NOW = datetime(2026, 8, 17, tzinfo=UTC)


def _row(index: int, *, available_delay: int = 0, funding: str | None = None):
    event = NOW + timedelta(minutes=index)
    price = Decimal(100 + index)
    return PerpetualMarketObservation(
        event_time=event,
        available_at=event + timedelta(minutes=available_delay),
        close=price,
        funding_rate=None if funding is None else Decimal(funding),
        mark_price=price + Decimal("0.1"),
        index_price=price,
        open_interest=Decimal(1000 + index * 20),
        bid=price - Decimal("0.05"),
        ask=price + Decimal("0.05"),
        aggressive_buy_notional=Decimal(60 + index),
        aggressive_sell_notional=Decimal(40),
        long_liquidation_notional=Decimal(30),
        short_liquidation_notional=Decimal(10),
    )


def test_perpetual_features_cover_high_roi_and_flow_inputs() -> None:
    rows = tuple(_row(i, funding=str(Decimal("0.0001") * i)) for i in range(4))
    result = build_perpetual_feature_snapshot(rows, decision_time=NOW + timedelta(minutes=3))
    assert result.source_observation_count == 4
    assert result.funding_rate == Decimal("0.0003")
    assert result.funding_acceleration == Decimal("0.0001")
    assert result.funding_zscore is not None
    assert result.mark_index_basis is not None
    assert result.basis_volatility is not None
    assert result.open_interest_delta is not None
    assert result.price_open_interest_divergence is not None
    assert result.realized_volatility is not None
    assert result.spread_proxy is not None
    assert result.agg_trade_imbalance is not None
    assert result.trade_intensity == Decimal(103)
    assert result.liquidation_pressure == Decimal("0.5")
    assert result.missing_features == ()


def test_future_available_observation_is_excluded() -> None:
    rows = (_row(0, funding="0.001"), _row(1, available_delay=10, funding="9"))
    result = build_perpetual_feature_snapshot(rows, decision_time=NOW + timedelta(minutes=2))
    assert result.source_observation_count == 1
    assert result.funding_rate == Decimal("0.001")
    assert "funding_zscore" in result.missing_features


def test_missing_inputs_are_explicit_and_invalid_book_fails() -> None:
    bare = PerpetualMarketObservation(NOW, NOW, Decimal(100))
    result = build_perpetual_feature_snapshot((bare,), decision_time=NOW)
    assert "mark_index_basis" in result.missing_features
    assert "agg_trade_imbalance" in result.missing_features
    with pytest.raises(ValueError, match="ask"):
        PerpetualMarketObservation(
            NOW,
            NOW,
            Decimal(100),
            bid=Decimal(101),
            ask=Decimal(100),
        )
