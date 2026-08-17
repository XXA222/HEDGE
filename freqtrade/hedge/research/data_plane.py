"""Point-in-time market-data catalogue and feature lineage contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
import json

from freqtrade.hedge.contracts.types import canonical_symbol, required_text


class MarketDataKind(StrEnum):
    CANDLES = "CANDLES"
    TRADES = "TRADES"
    AGG_TRADES = "AGG_TRADES"
    FUNDING = "FUNDING"
    MARK = "MARK"
    INDEX = "INDEX"
    OPEN_INTEREST = "OPEN_INTEREST"
    BASIS = "BASIS"
    LIQUIDATION = "LIQUIDATION"
    ORDERBOOK = "ORDERBOOK"
    DERIVED = "DERIVED"


def _aware(value: object, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware datetime")
    return value.astimezone(UTC)


def _hash(value: object, *, name: str) -> str:
    raw = required_text(value, field_name=name, max_length=64).lower()
    if len(raw) != 64 or any(char not in "0123456789abcdef" for char in raw):
        raise ValueError(f"{name} must be sha256")
    return raw


@dataclass(frozen=True, slots=True)
class MarketDataPartition:
    dataset_id: str
    exchange: str
    market_type: str
    symbol: str
    data_kind: MarketDataKind
    timeframe: str | None
    event_start: datetime
    event_end: datetime
    row_count: int
    schema_version: str
    source: str
    sha256: str
    ingested_at: datetime
    available_at: datetime

    def __post_init__(self) -> None:
        for name, limit in (("dataset_id", 128), ("exchange", 64), ("market_type", 64), ("schema_version", 64), ("source", 256)):
            object.__setattr__(self, name, required_text(getattr(self, name), field_name=name, max_length=limit))
        object.__setattr__(self, "symbol", canonical_symbol(self.symbol))
        if not isinstance(self.data_kind, MarketDataKind):
            raise TypeError("data_kind must be MarketDataKind")
        if self.timeframe is not None:
            object.__setattr__(self, "timeframe", required_text(self.timeframe, field_name="timeframe", max_length=32))
        for name in ("event_start", "event_end", "ingested_at", "available_at"):
            object.__setattr__(self, name, _aware(getattr(self, name), name=name))
        if self.event_end < self.event_start:
            raise ValueError("event_end must not precede event_start")
        if self.available_at < self.ingested_at:
            raise ValueError("available_at cannot precede ingested_at")
        if isinstance(self.row_count, bool) or not isinstance(self.row_count, int) or self.row_count <= 0:
            raise ValueError("row_count must be positive int")
        object.__setattr__(self, "sha256", _hash(self.sha256, name="sha256"))


class MarketDataCatalog:
    def __init__(self, partitions: tuple[MarketDataPartition, ...] = ()) -> None:
        self._partitions: dict[str, MarketDataPartition] = {}
        for partition in partitions:
            self.register(partition)

    def register(self, partition: MarketDataPartition) -> None:
        if not isinstance(partition, MarketDataPartition):
            raise TypeError("partition must be MarketDataPartition")
        existing = self._partitions.get(partition.dataset_id)
        if existing is not None and existing != partition:
            raise ValueError("dataset_id is immutable and already maps to another partition")
        self._partitions[partition.dataset_id] = partition

    def available_for(self, *, decision_time: datetime, symbol: str | None = None, kind: MarketDataKind | None = None) -> tuple[MarketDataPartition, ...]:
        decision_time = _aware(decision_time, name="decision_time")
        normalized_symbol = canonical_symbol(symbol) if symbol is not None else None
        return tuple(sorted(
            (partition for partition in self._partitions.values()
             if partition.available_at <= decision_time
             and (normalized_symbol is None or partition.symbol == normalized_symbol)
             and (kind is None or partition.data_kind is kind)),
            key=lambda partition: (partition.available_at, partition.dataset_id),
        ))


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    version: str
    dtype: str
    lookback: timedelta
    input_dataset_ids: tuple[str, ...]
    availability_delay: timedelta
    warmup: int
    normalization: str
    missing_policy: str
    code_sha256: str

    def __post_init__(self) -> None:
        for name in ("name", "version", "dtype", "normalization", "missing_policy"):
            object.__setattr__(self, name, required_text(getattr(self, name), field_name=name, max_length=128))
        if self.lookback < timedelta(0) or self.availability_delay < timedelta(0):
            raise ValueError("feature time windows must be nonnegative")
        if isinstance(self.warmup, bool) or not isinstance(self.warmup, int) or self.warmup < 0:
            raise ValueError("warmup must be a nonnegative int")
        datasets = tuple(required_text(item, field_name="input_dataset_id", max_length=128) for item in self.input_dataset_ids)
        if not datasets or len(set(datasets)) != len(datasets):
            raise ValueError("input_dataset_ids must be unique and nonempty")
        object.__setattr__(self, "input_dataset_ids", datasets)
        object.__setattr__(self, "code_sha256", _hash(self.code_sha256, name="code_sha256"))


@dataclass(frozen=True, slots=True)
class FeatureSet:
    specs: tuple[FeatureSpec, ...]

    def __post_init__(self) -> None:
        specs = tuple(self.specs)
        if not specs or len({(spec.name, spec.version) for spec in specs}) != len(specs):
            raise ValueError("FeatureSet requires unique nonempty specs")
        object.__setattr__(self, "specs", specs)

    @property
    def sha256(self) -> str:
        payload = [
            {name: str(getattr(spec, name)) for name in spec.__dataclass_fields__}
            for spec in sorted(self.specs, key=lambda item: (item.name, item.version))
        ]
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    feature_set_sha256: str
    event_time: datetime
    observed_time: datetime
    available_at: datetime
    values: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_set_sha256", _hash(self.feature_set_sha256, name="feature_set_sha256"))
        for name in ("event_time", "observed_time", "available_at"):
            object.__setattr__(self, name, _aware(getattr(self, name), name=name))
        if self.observed_time < self.event_time or self.available_at < self.observed_time:
            raise ValueError("feature times must satisfy event <= observed <= available")
        if not isinstance(self.values, Mapping) or not self.values:
            raise ValueError("values must be a nonempty mapping")

    def usable_at(self, decision_time: datetime) -> bool:
        return self.available_at <= _aware(decision_time, name="decision_time")
