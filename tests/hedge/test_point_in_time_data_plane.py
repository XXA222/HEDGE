from __future__ import annotations

from datetime import UTC, datetime, timedelta

from freqtrade.hedge.research.data_plane import (
    FeatureSet, FeatureSnapshot, FeatureSpec, MarketDataCatalog, MarketDataKind, MarketDataPartition,
)


HASH = "f" * 64
NOW = datetime(2026, 8, 17, tzinfo=UTC)


def _partition(*, available_at: datetime) -> MarketDataPartition:
    return MarketDataPartition(
        dataset_id="btc-candles", exchange="binance", market_type="perpetual",
        symbol="BTC/USDT:USDT", data_kind=MarketDataKind.CANDLES, timeframe="5m",
        event_start=NOW, event_end=NOW, row_count=1, schema_version="v1", source="recorded",
        sha256=HASH, ingested_at=NOW, available_at=available_at,
    )


def test_catalog_hides_data_until_its_availability_time() -> None:
    catalog = MarketDataCatalog((_partition(available_at=NOW + timedelta(minutes=1)),))
    assert not catalog.available_for(decision_time=NOW)
    assert len(catalog.available_for(decision_time=NOW + timedelta(minutes=1))) == 1


def test_feature_snapshot_enforces_three_time_semantics_and_set_lineage() -> None:
    feature_set = FeatureSet((FeatureSpec(
        "funding_z", "v1", "float", timedelta(minutes=5), ("btc-candles",), timedelta(seconds=5),
        10, "zscore", "reject", HASH,
    ),))
    snapshot = FeatureSnapshot(feature_set.sha256, NOW, NOW + timedelta(seconds=1), NOW + timedelta(seconds=5), {"funding_z": 1.0})
    assert not snapshot.usable_at(NOW)
    assert snapshot.usable_at(NOW + timedelta(seconds=5))
