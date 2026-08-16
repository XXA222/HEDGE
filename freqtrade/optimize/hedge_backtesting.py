"""Standard Freqtrade data/strategy adapter for deterministic Hedge backtesting.

The low-level :class:`HedgeBacktesting` facade remains useful for tests and
programmatic event replays.  ``run_freqtrade_hedge_backtest`` adds the missing
production-facing path: normal Freqtrade configuration, downloaded OHLCV,
strategy analysis, futures funding data, the shared Hedge planner/matcher, and a
reproducible JSON result artifact.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import (
    asdict,
    dataclass,
    fields,
    is_dataclass,
    replace,
)
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Any

from freqtrade.exceptions import OperationalException
from freqtrade.hedge.backtesting.memory import (
    DEFAULT_HEDGE_BACKTEST_MEMORY_POLICY,
    release_phase_memory,
)
from freqtrade.hedge.paper_config import PaperFundingSource, PaperSimulationConfig
from freqtrade.hedge.planning.context import PlannerConfig, StrategyPlanningPort
from freqtrade.hedge.simulation.exchange import (
    BarEvent,
    FundingEvent,
    MarketRules,
    SignalEvent,
    SimulationInputEvent,
    SimulationResult,
)
from freqtrade.hedge.simulation.matcher import MatchConfig
from freqtrade.hedge.simulation.replay import EventReplayEngine


DEFAULT_FUNDING_RATE_MULTIPLIER = Decimal(1)
DEFAULT_LEVERAGE = Decimal(3)
DEFAULT_FEE_RATE = Decimal("0.0004")
DEFAULT_LONG_SIGNAL = Decimal(1)
DEFAULT_SHORT_SIGNAL = Decimal(1)
DEFAULT_STREAM_CHUNK_BARS = 2048
_STREAM_FINGERPRINT_VERSION = b"hedge-event-stream-v3\0"


@dataclass(frozen=True, slots=True)
class HedgeBacktestDataset:
    events: tuple[SimulationInputEvent, ...]
    pair: str
    timeframe: str
    start: datetime
    end: datetime
    bar_count: int
    signal_count: int
    funding_count: int
    missing_candle_count: int = 0
    data_fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class HedgeBacktestRun:
    result: SimulationResult
    dataset: HedgeBacktestDataset
    export_path: Path
    strategy: str
    market_rule_source: str
    market_rule_version: str
    artifact_sha256: str = ""
    result_fingerprint: str = ""
    native_artifact: object | None = None


class _ArrayRowView:
    """Reusable get-only row facade using array references plus one row index.

    The previous compact path still allocated one tuple per bar through ``zip``.
    This version retains the narrow arrays once and only mutates an integer row index.
    """

    __slots__ = ("_arrays", "_index", "_row_index")

    def __init__(self, names: tuple[str, ...], arrays: tuple[Any, ...]) -> None:
        if len(names) != len(arrays):
            raise ValueError("row-view names/arrays must have identical width")
        self._index = {name: index for index, name in enumerate(names)}
        self._arrays = arrays
        self._row_index = 0

    def bind_index(self, row_index: int) -> _ArrayRowView:
        self._row_index = int(row_index)
        return self

    def get(self, name: str, default: object = None) -> object:
        index = self._index.get(name)
        if index is None:
            return default
        return self._arrays[index][self._row_index]


def _compact_signal_bar_from_row(
    *,
    pair: str,
    bar_delta: timedelta,
    row: _ArrayRowView,
    columns: set[str],
    open_time: datetime,
    strategy_version: object = None,
) -> tuple[SignalEvent, BarEvent]:
    """Build canonical replay events without intermediate signal/candle dataclasses.

    Live Paper keeps ``signal_from_analyzed_row`` and its rich SignalSnapshot.  The
    historical compact path only consumes SignalEvent/BarEvent, so allocating an
    AnalyzedCandle, a validation BarEvent, a SignalSnapshot and then a second
    BarEvent for every minute is avoidable.  Directive parsing remains shared via
    ``directive_from_values`` and the final simulation events retain their normal
    invariant validation.
    """
    from freqtrade.hedge.strategies.contract import directive_from_values

    close_time = open_time + bar_delta
    directive = directive_from_values(row)  # type: ignore[arg-type]
    volume = (
        _decimal(row.get("volume"), field="volume")
        if "volume" in columns and row.get("volume") is not None
        else None
    )
    bar = BarEvent(
        timestamp=close_time,
        symbol=pair,
        open=_decimal(row.get("open"), field="open"),
        high=_decimal(row.get("high"), field="high"),
        low=_decimal(row.get("low"), field="low"),
        close=_decimal(row.get("close"), field="close"),
        volume=volume,
    )
    fallback_reason = (
        "HEDGE_TARGET_COLUMNS"
        if {"hedge_long_score", "hedge_short_score"} & columns
        else "ENTER_SIGNAL_COMPATIBILITY"
    )
    if "hedge_model_version" in columns:
        model_version = directive.model_version
    else:
        model_version = str(strategy_version or "strategy")[:128]
    signal_event = SignalEvent(
        timestamp=close_time,
        symbol=pair,
        long_signal=directive.long_score,
        short_signal=directive.short_score,
        target_net=directive.target_net_quantity,
        model_version=model_version,
        reason=directive.reason or fallback_reason,
        target_net_ratio=directive.target_net_ratio,
        confidence=directive.confidence,
        risk_scale=directive.risk_scale,
        long_exposure_scale=directive.long_exposure_scale,
        short_exposure_scale=directive.short_exposure_scale,
        allow_new_risk=directive.allow_new_risk,
        regime=directive.regime,
    )
    return signal_event, bar


def _missing_candle_count_seconds(
    previous: datetime,
    current: datetime,
    seconds: int,
) -> int:
    """Cached-timeframe equivalent of candle_cursor.missing_candle_count()."""
    delta = (current - previous).total_seconds()
    if delta <= 0:
        raise ValueError("candle timestamps must move forward")
    integer_delta = int(delta)
    slots, remainder = divmod(integer_delta, int(seconds))
    if remainder != 0 or delta != integer_delta:
        raise ValueError("candle timestamps are not aligned to the configured timeframe")
    return max(slots - 1, 0)


class HedgeBacktestEventChunks:
    """Single-use, bounded-memory event producer for analyzed OHLCV.

    The producer keeps the same Signal -> Funding -> Bar priority contract as
    ``EventReplayEngine`` but never materializes the full multi-year event list.
    It also computes the dataset fingerprint incrementally and validates candle
    chronology/missing slots while rows are consumed.
    """

    def __init__(
        self,
        *,
        pair: str,
        timeframe: str,
        frame: Any,
        funding_frame: Any | None = None,
        strategy_version: object = None,
        require_funding_data: bool = False,
        max_missing_candles: int = 0,
        funding_rate_multiplier: Decimal = DEFAULT_FUNDING_RATE_MULTIPLIER,
        chunk_bars: int = DEFAULT_STREAM_CHUNK_BARS,
        copy_arrays: bool = True,
    ) -> None:
        if chunk_bars < 1:
            raise ValueError("chunk_bars must be positive")
        if frame is None or frame.empty:
            raise OperationalException("Hedge backtest analyzed dataframe is empty")
        required = {"date", "open", "high", "low", "close"}
        columns = set(frame.columns)
        missing = sorted(required - columns)
        if missing:
            raise OperationalException(
                "Hedge backtest analyzed dataframe is missing: " + ", ".join(missing)
            )
        signal_columns = {
            "hedge_long_score",
            "hedge_short_score",
            "hedge_target_net",
            "enter_long",
            "enter_short",
        }
        if not signal_columns.intersection(columns):
            raise OperationalException(
                "Hedge backtest strategy produced no hedge_* or enter_long/enter_short columns"
            )

        multiplier = _decimal(funding_rate_multiplier, field="funding_rate_multiplier")
        if multiplier < 0:
            raise OperationalException("funding_rate_multiplier cannot be negative")

        funding_missing = funding_frame is None or funding_frame.empty
        if require_funding_data and funding_missing:
            raise OperationalException(
                "Hedge futures backtest requires downloaded funding/mark data; "
                "run download-data with the futures configuration first"
            )
        if funding_frame is not None and not funding_frame.empty:
            funding_columns = set(funding_frame.columns)
            required_funding = {"date", "open_fund", "open_mark"}
            if not required_funding.issubset(funding_columns):
                raise OperationalException(
                    "Futures funding data must contain date/open_fund/open_mark"
                )

        from freqtrade.exchange import timeframe_to_seconds

        seconds = timeframe_to_seconds(timeframe)
        first_open = _aware(frame["date"].iloc[0], field="date")
        last_open = _aware(frame["date"].iloc[-1], field="date")
        from freqtrade.hedge.strategies.contract import HEDGE_SIGNAL_COLUMNS

        preferred = (
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            *HEDGE_SIGNAL_COLUMNS,
            "enter_long",
            "enter_short",
        )
        names = tuple(name for name in preferred if name in columns)
        # Copy only the narrow canonical replay surface.  This intentionally
        # detaches the multi-year stream from the often much wider analyzed
        # strategy dataframe so the caller can release indicator/intermediate
        # columns before event replay begins.  ExtensionArray.copy() preserves
        # compact timezone-aware datetime storage instead of creating Timestamp
        # object arrays.
        row_arrays = tuple(
            frame[name].array.copy() if copy_arrays else frame[name].array for name in names
        )
        funding_arrays = (
            None
            if funding_frame is None or funding_frame.empty
            else tuple(
                funding_frame[name].array.copy() if copy_arrays else funding_frame[name].array
                for name in ("date", "open_fund", "open_mark")
            )
        )

        self.pair = pair
        self.timeframe = timeframe
        self.strategy_version = strategy_version
        self._names = names
        self._row_arrays = row_arrays
        self._funding_arrays = funding_arrays
        self.require_funding_data = require_funding_data
        self.max_missing_candles = max_missing_candles
        self.funding_rate_multiplier = multiplier
        self.chunk_bars = chunk_bars
        self.start = first_open + timedelta(seconds=seconds)
        self.end = last_open + timedelta(seconds=seconds)
        self.bar_count = 0
        self.funding_count = 0
        self.missing_candle_count = 0
        self.max_chunk_input_events = 0
        self._columns = columns
        self._seconds = seconds
        self._bar_delta = timedelta(seconds=seconds)
        self._row_count = len(frame)
        self._funding_row_count = (
            len(funding_frame) if funding_frame is not None and not funding_frame.empty else 0
        )
        self.row_view_mode = "INDEXED_ARRAY_VIEW_V2"
        self.chronology_mode = "CACHED_TIMEFRAME_SECONDS_V2"
        self._consumed = False
        self._complete = False
        self._hasher = sha256(_STREAM_FINGERPRINT_VERSION)
        self.data_fingerprint = ""

    @staticmethod
    def _row_values(arrays: tuple[Any, ...]):
        return zip(*arrays, strict=True)

    def _funding_events(self):
        if self._funding_arrays is None:
            return
        previous: datetime | None = None
        date_array, rate_array, mark_array = self._funding_arrays
        for row_index in range(self._funding_row_count):
            timestamp = _aware(date_array[row_index], field="funding.date")
            if previous is not None and timestamp <= previous:
                raise OperationalException(
                    "Hedge funding dataframe must be strictly chronological and unique"
                )
            previous = timestamp
            if timestamp < self.start:
                continue
            if timestamp > self.end:
                break
            rate = (
                _decimal(rate_array[row_index], field="funding.open_fund")
                * self.funding_rate_multiplier
            )
            mark = _decimal(mark_array[row_index], field="funding.open_mark")
            if mark <= 0:
                raise OperationalException("funding.open_mark must be positive")
            yield FundingEvent(
                timestamp=timestamp,
                symbol=self.pair,
                rate=rate,
                mark_price=mark,
            )

    def _hash_event(self, event: SimulationInputEvent) -> None:
        _update_event_hash(self._hasher, event)

    def _iter_events(self):
        if self._consumed:
            raise RuntimeError("HedgeBacktestEventChunks is single-use")
        self._consumed = True

        names = self._names
        signal_columns = set(names)
        row_view = _ArrayRowView(names, self._row_arrays)
        funding_iter = iter(self._funding_events() or ())
        next_funding = next(funding_iter, None)
        previous_open: datetime | None = None

        for row_index in range(self._row_count):
            row = row_view.bind_index(row_index)
            open_time = _aware(row.get("date"), field="date")
            if previous_open is not None:
                if open_time <= previous_open:
                    raise OperationalException(
                        "Hedge backtest candles must be strictly chronological and unique"
                    )
                self.missing_candle_count += _missing_candle_count_seconds(
                    previous_open, open_time, self._seconds
                )
                if self.missing_candle_count > self.max_missing_candles:
                    raise OperationalException(
                        f"Hedge backtest has {self.missing_candle_count} missing candle slots; "
                        f"limit={self.max_missing_candles}"
                    )
            previous_open = open_time

            signal_event, bar = _compact_signal_bar_from_row(
                pair=self.pair,
                bar_delta=self._bar_delta,
                row=row,
                columns=signal_columns,
                open_time=open_time,
                strategy_version=self.strategy_version,
            )

            while next_funding is not None and next_funding.timestamp < bar.timestamp:
                self._hash_event(next_funding)
                yield next_funding
                self.funding_count += 1
                next_funding = next(funding_iter, None)

            self._hash_event(signal_event)
            yield signal_event
            while next_funding is not None and next_funding.timestamp == bar.timestamp:
                self._hash_event(next_funding)
                yield next_funding
                self.funding_count += 1
                next_funding = next(funding_iter, None)
            self._hash_event(bar)
            yield bar

            self.bar_count += 1

        self.max_chunk_input_events = 1
        self.data_fingerprint = self._hasher.hexdigest()
        self._complete = True

    def events(self):
        """Yield the canonical ordered input stream one event at a time."""
        yield from self._iter_events()

    def __iter__(self):
        """Compatibility chunk view used by tests and detailed diagnostics."""
        chunk: list[SimulationInputEvent] = []
        bars = 0
        for event in self._iter_events():
            chunk.append(event)
            if isinstance(event, BarEvent):
                bars += 1
            if bars >= self.chunk_bars:
                yield tuple(chunk)
                chunk.clear()
                bars = 0
        if chunk:
            yield tuple(chunk)

    def dataset(self) -> HedgeBacktestDataset:
        if not self._complete:
            raise RuntimeError("event stream must be fully consumed before dataset()")
        return HedgeBacktestDataset(
            events=(),
            pair=self.pair,
            timeframe=self.timeframe,
            start=self.start,
            end=self.end,
            bar_count=self.bar_count,
            signal_count=self.bar_count,
            funding_count=self.funding_count,
            missing_candle_count=self.missing_candle_count,
            data_fingerprint=self.data_fingerprint,
        )


def _aware(value: object, *, field: str) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        raise OperationalException(f"{field} must contain datetime values")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _decimal(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise OperationalException(f"{field} must be numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise OperationalException(f"{field} must be a valid decimal") from exc
    if not result.is_finite():
        raise OperationalException(f"{field} must be finite")
    return result


def _strict_dates(frame: Any, *, field: str = "date") -> tuple[datetime, ...]:
    if field not in frame.columns:
        raise OperationalException(f"Hedge backtest dataframe is missing {field!r}")
    values = tuple(_aware(value, field=field) for value in frame[field].tolist())
    if not values:
        raise OperationalException("Hedge backtest has no analyzed candles")
    if any(right <= left for left, right in pairwise(values)):
        raise OperationalException(
            "Hedge backtest candles must be strictly chronological and unique"
        )
    return values


def events_from_analyzed_dataframe(  # noqa: C901
    *,
    pair: str,
    timeframe: str,
    frame: Any,
    funding_frame: Any | None = None,
    strategy_version: object = None,
    require_funding_data: bool = False,
    max_missing_candles: int = 0,
    funding_rate_multiplier: Decimal = DEFAULT_FUNDING_RATE_MULTIPLIER,
) -> HedgeBacktestDataset:
    """Build the shared event stream from one analyzed Freqtrade dataframe.

    A signal and bar share the candle close timestamp.  The engine first matches
    orders accepted on earlier bars and only then creates orders from the current
    signal, so the current OHLC path can never fill an order derived from its own
    close or indicators.
    """

    from freqtrade.hedge.integration.candle_cursor import missing_candle_count
    from freqtrade.hedge.integration.signal_provider import signal_from_analyzed_row

    funding_rate_multiplier = _decimal(funding_rate_multiplier, field="funding_rate_multiplier")
    if funding_rate_multiplier < 0:
        raise OperationalException("funding_rate_multiplier cannot be negative")

    if frame is None or frame.empty:
        raise OperationalException("Hedge backtest analyzed dataframe is empty")
    required = {"date", "open", "high", "low", "close"}
    columns = set(frame.columns)
    missing = sorted(required - columns)
    if missing:
        raise OperationalException(
            "Hedge backtest analyzed dataframe is missing: " + ", ".join(missing)
        )
    signal_columns = {
        "hedge_long_score",
        "hedge_short_score",
        "hedge_target_net",
        "enter_long",
        "enter_short",
    }
    if not signal_columns.intersection(columns):
        raise OperationalException(
            "Hedge backtest strategy produced no hedge_* or enter_long/enter_short columns"
        )
    dates = _strict_dates(frame)
    missing_slots = sum(
        missing_candle_count(left, right, timeframe) for left, right in pairwise(dates)
    )
    if missing_slots > max_missing_candles:
        raise OperationalException(
            f"Hedge backtest has {missing_slots} missing candle slots; limit={max_missing_candles}"
        )

    events: list[SimulationInputEvent] = []
    bars: list[BarEvent] = []
    for _, row in frame.iterrows():
        signal = signal_from_analyzed_row(
            pair=pair,
            timeframe=timeframe,
            row=row,
            columns=columns,
            feature_timestamp=_aware(row.get("date"), field="date"),
            strategy_version=strategy_version,
        )
        candle = signal.candle
        if candle is None:  # guarded by the required columns above
            raise OperationalException("Hedge backtest could not build an OHLCV candle")
        events.append(
            SignalEvent(
                timestamp=signal.candle_close_time,
                symbol=pair,
                long_signal=signal.long_score,
                short_signal=signal.short_score,
                target_net=signal.target_net,
                model_version=signal.model_version,
                reason=signal.strategy_reason or signal.reason,
                target_net_ratio=signal.target_net_ratio,
                confidence=signal.confidence,
                risk_scale=signal.risk_scale,
                long_exposure_scale=signal.long_exposure_scale,
                short_exposure_scale=signal.short_exposure_scale,
                allow_new_risk=signal.allow_new_risk,
                regime=signal.regime,
            )
        )
        bar = candle.to_bar_event()
        bars.append(bar)
        events.append(bar)

    start = bars[0].timestamp
    end = bars[-1].timestamp
    funding_count = 0
    funding_missing = funding_frame is None or funding_frame.empty
    if require_funding_data and funding_missing:
        raise OperationalException(
            "Hedge futures backtest requires downloaded funding/mark data; "
            "run download-data with the futures configuration first"
        )
    if funding_frame is not None and not funding_frame.empty:
        funding_columns = set(funding_frame.columns)
        required_funding = {"date", "open_fund", "open_mark"}
        if not required_funding.issubset(funding_columns):
            raise OperationalException("Futures funding data must contain date/open_fund/open_mark")
        funding_dates = _strict_dates(funding_frame)
        for (_, row), timestamp in zip(funding_frame.iterrows(), funding_dates, strict=True):
            if timestamp < start or timestamp > end:
                continue
            rate = (
                _decimal(row.get("open_fund"), field="funding.open_fund") * funding_rate_multiplier
            )
            mark = _decimal(row.get("open_mark"), field="funding.open_mark")
            if mark <= 0:
                raise OperationalException("funding.open_mark must be positive")
            events.append(
                FundingEvent(
                    timestamp=timestamp,
                    symbol=pair,
                    rate=rate,
                    mark_price=mark,
                )
            )
            funding_count += 1

    # Keep detailed and compact mode on one canonical replay order, then hash
    # incrementally so the detailed path does not create a second full JSON tree.
    events.sort(key=lambda event: (event.timestamp, _input_event_priority(event)))
    data_fingerprint = _fingerprint_events(events)
    return HedgeBacktestDataset(
        events=tuple(events),
        pair=pair,
        timeframe=timeframe,
        start=start,
        end=end,
        bar_count=len(bars),
        signal_count=len(bars),
        funding_count=funding_count,
        missing_candle_count=missing_slots,
        data_fingerprint=data_fingerprint,
    )


class HedgeBacktesting:
    """Freqtrade-facing adapter over the shared deterministic event engine."""

    def __init__(
        self,
        *,
        initial_balance: Decimal,
        planner_config: PlannerConfig | None = None,
        leverage: Decimal = DEFAULT_LEVERAGE,
        fee_rate: Decimal = DEFAULT_FEE_RATE,
        long_signal: Decimal = DEFAULT_LONG_SIGNAL,
        short_signal: Decimal = DEFAULT_SHORT_SIGNAL,
        target_net_quantity: Decimal | None = None,
        market_rules: MarketRules | None = None,
        planner: StrategyPlanningPort | None = None,
        match_config: MatchConfig | None = None,
    ) -> None:
        self.engine = EventReplayEngine(
            initial_balance=initial_balance,
            planner_config=planner_config,
            leverage=leverage,
            fee_rate=fee_rate,
            long_signal=long_signal,
            short_signal=short_signal,
            target_net_quantity=target_net_quantity,
            market_rules=market_rules,
            planner=planner,
            match_config=match_config,
        )

    def run(self, events: Iterable[SimulationInputEvent]) -> SimulationResult:
        return self.engine.replay(events)

    def run_compact(
        self,
        stream: HedgeBacktestEventChunks,
    ) -> SimulationResult:
        policy = DEFAULT_HEDGE_BACKTEST_MEMORY_POLICY
        result = self.engine.replay_ordered_stream(
            stream.events(),
            retain_material_events=policy.retain_material_events,
            snapshot_every_bars=policy.snapshot_every_bars(stream._seconds),
            max_retained_snapshots=policy.max_retained_snapshots,
            compact_wallet_history=policy.compact_wallet_history,
        )
        report = dict(result.report)
        report.update(
            {
                "stream_row_mode": stream.row_view_mode,
                "chronology_mode": stream.chronology_mode,
            }
        )
        return replace(result, report=report)


def _paper_match_config(config: PaperSimulationConfig) -> MatchConfig:
    return MatchConfig(
        maker_fee_rate=config.maker_fee_rate,
        taker_fee_rate=config.taker_fee_rate,
        volume_participation=config.volume_participation,
        market_slippage_bps=config.market_slippage_bps,
        price_tick=config.tick_size,
        qty_step=config.qty_step,
        min_fill_qty=config.min_qty,
        min_fill_notional=config.min_notional,
        max_entry_layers_per_bar=config.max_entry_layers_per_bar,
        max_reduce_layers_per_bar=config.max_reduce_layers_per_bar,
        max_fill_ratio_per_order=config.max_fill_ratio_per_order,
        max_fills_per_bar=config.max_fills_per_bar,
    )


def _input_event_priority(event: SimulationInputEvent) -> int:
    if isinstance(event, SignalEvent):
        return 0
    if isinstance(event, FundingEvent):
        return 1
    return 2


def _hash_scalar(hasher, value: object) -> None:
    """Hash a scalar with explicit type/length framing and no JSON allocation."""
    if value is None:
        hasher.update(b"N")
        return
    if isinstance(value, bool):
        hasher.update(b"B1" if value else b"B0")
        return
    if isinstance(value, datetime):
        payload = value.isoformat().encode("ascii")
        tag = b"T"
    elif isinstance(value, Decimal):
        payload = str(value).encode("ascii")
        tag = b"D"
    elif isinstance(value, Enum):
        payload = str(value.value).encode("utf-8")
        tag = b"E"
    elif isinstance(value, (int, float)):
        payload = repr(value).encode("ascii")
        tag = b"Q"
    else:
        payload = str(value).encode("utf-8")
        tag = b"S"
    hasher.update(tag)
    hasher.update(len(payload).to_bytes(4, "big"))
    hasher.update(payload)


def _update_event_hash(hasher, event: SimulationInputEvent) -> None:
    name = type(event).__qualname__.encode("ascii")
    hasher.update(len(name).to_bytes(2, "big"))
    hasher.update(name)
    for item in fields(event):
        field_name = item.name.encode("ascii")
        hasher.update(len(field_name).to_bytes(2, "big"))
        hasher.update(field_name)
        _hash_scalar(hasher, getattr(event, item.name))


def _fingerprint_events(events: Iterable[SimulationInputEvent]) -> str:
    """Hash canonical events incrementally with O(1) temporary memory."""
    hasher = sha256(_STREAM_FINGERPRINT_VERSION)
    for event in events:
        _update_event_hash(hasher, event)
    return hasher.hexdigest()


def _json_value(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


@dataclass(slots=True)
class PreparedHedgeBacktest:
    """Reusable analyzed input for planner/paper-only optimization epochs.

    Freqtrade's native Hyperopt calculates indicators once and reuses a simplified
    execution surface for each epoch.  This object applies the same lifecycle to
    Hedge: the expensive OHLCV/informative analysis is detached once, upstream
    caches are released, and every trial replays the same narrow immutable columns.
    """

    pair: str
    timeframe: str
    frame: Any
    funding_frame: Any | None
    strategy_name: str
    strategy_version: object
    market_rules: MarketRules
    market_rule_source: str
    market_rule_version: str
    maker_fee_rate: Decimal
    taker_fee_rate: Decimal

    def run(
        self,
        config: dict[str, Any],
        *,
        export_path: Path | None = None,
        row_slice: slice | None = None,
        persist_artifact: bool = True,
    ) -> HedgeBacktestRun:
        from freqtrade.hedge.config import validate_hedge_config
        from freqtrade.hedge.integration.paper_runtime import planner_config_from_mapping

        runtime = validate_hedge_config(config)
        paper = runtime.paper
        if not runtime.enabled or paper is None:
            raise OperationalException("prepared Hedge backtest requires hedge.paper")
        if runtime.managed_pair != self.pair:
            raise OperationalException("prepared Hedge backtest pair cannot change between trials")

        optimization_runtime = config.get("hedge_optimization_runtime", {})
        runtime_mapping = optimization_runtime if isinstance(optimization_runtime, Mapping) else {}
        funding_multiplier = _decimal(
            runtime_mapping.get("funding_rate_multiplier", "1"),
            field="hedge_optimization_runtime.funding_rate_multiplier",
        )
        maker_multiplier = _decimal(
            runtime_mapping.get("maker_fee_multiplier", "1"),
            field="hedge_optimization_runtime.maker_fee_multiplier",
        )
        taker_multiplier = _decimal(
            runtime_mapping.get("taker_fee_multiplier", "1"),
            field="hedge_optimization_runtime.taker_fee_multiplier",
        )
        if min(funding_multiplier, maker_multiplier, taker_multiplier) < 0:
            raise OperationalException("prepared Hedge backtest multipliers cannot be negative")

        raw_hedge = config.get("hedge", {})
        hedge_mapping = raw_hedge if isinstance(raw_hedge, Mapping) else {}
        planner_raw = hedge_mapping.get("planner", {})
        planner_mapping = planner_raw if isinstance(planner_raw, Mapping) else {}

        match_config = _paper_match_config(paper)
        match_config = MatchConfig(
            **{
                **asdict(match_config),
                "maker_fee_rate": self.maker_fee_rate * maker_multiplier,
                "taker_fee_rate": self.taker_fee_rate * taker_multiplier,
                "price_tick": self.market_rules.tick_size,
                "qty_step": self.market_rules.qty_step,
                "min_fill_qty": self.market_rules.min_qty,
                "min_fill_notional": self.market_rules.min_notional,
            }
        )
        replay_frame = self.frame if row_slice is None else self.frame.iloc[row_slice]
        if replay_frame is None or replay_frame.empty:
            raise OperationalException("prepared Hedge backtest slice is empty")
        stream = HedgeBacktestEventChunks(
            pair=self.pair,
            timeframe=self.timeframe,
            frame=replay_frame,
            funding_frame=self.funding_frame,
            strategy_version=self.strategy_version,
            require_funding_data=paper.funding_source is PaperFundingSource.EXCHANGE,
            max_missing_candles=paper.max_missing_candles,
            funding_rate_multiplier=funding_multiplier,
            copy_arrays=False,
        )
        runner = HedgeBacktesting(
            initial_balance=paper.initial_balance,
            planner_config=planner_config_from_mapping(planner_mapping),
            leverage=paper.leverage,
            fee_rate=self.taker_fee_rate * taker_multiplier,
            long_signal=paper.default_long_signal,
            short_signal=paper.default_short_signal,
            market_rules=self.market_rules,
            match_config=match_config,
        )
        result = runner.run_compact(stream)
        dataset = stream.dataset()
        # The result contains only bounded compact evidence.  The stream owns the
        # replay column arrays and the runner owns the now-finished wallet/matcher;
        # neither is needed during result serialization.
        del stream, runner, replay_frame
        release_phase_memory()
        output = (export_path or _default_export_path(config)).expanduser().resolve()
        if persist_artifact:
            artifact_sha256, result_fingerprint, native_artifact = _write_result(
                path=output,
                result=result,
                dataset=dataset,
                strategy=self.strategy_name,
                market_rule_source=self.market_rule_source,
                market_rule_version=self.market_rule_version,
                export_events=False,
            )
        else:
            from freqtrade.hedge.native.backtest import HedgeBacktestResultAdapter

            native_artifact = HedgeBacktestResultAdapter().build(
                result,
                strategy_name=self.strategy_name,
                pairs=(dataset.pair,),
                timeframe=dataset.timeframe,
                timerange=f"{dataset.start.isoformat()}-{dataset.end.isoformat()}",
                metadata={
                    "start": dataset.start.isoformat(),
                    "end": dataset.end.isoformat(),
                    "bar_count": dataset.bar_count,
                    "signal_count": dataset.signal_count,
                    "funding_count": dataset.funding_count,
                    "data_fingerprint": dataset.data_fingerprint,
                    "market_rule_source": self.market_rule_source,
                    "market_rule_version": self.market_rule_version,
                    "execution_timing": "NEXT_BAR_NO_LOOKAHEAD",
                    "artifact_persisted": False,
                },
            )
            result_fingerprint = str(native_artifact.to_dict()["result_sha256"])
            artifact_sha256 = ""
        return HedgeBacktestRun(
            result=result,
            dataset=dataset,
            export_path=output,
            strategy=self.strategy_name,
            market_rule_source=self.market_rule_source,
            market_rule_version=self.market_rule_version,
            artifact_sha256=artifact_sha256,
            result_fingerprint=result_fingerprint,
            native_artifact=native_artifact,
        )


def prepare_freqtrade_hedge_backtest(config: dict[str, Any]) -> PreparedHedgeBacktest:
    """Analyze Freqtrade data once, detach a narrow replay surface, release caches."""
    from freqtrade.data.converter import trim_dataframe
    from freqtrade.hedge.config import validate_hedge_config
    from freqtrade.hedge.integration.market_data import exchange_market_rules
    from freqtrade.hedge.strategies.contract import HEDGE_SIGNAL_COLUMNS
    from freqtrade.optimize.backtesting import Backtesting

    config.setdefault(
        "reduce_df_footprint", DEFAULT_HEDGE_BACKTEST_MEMORY_POLICY.reduce_dataframe_footprint
    )
    runtime = validate_hedge_config(config)
    paper = runtime.paper
    if not runtime.enabled or paper is None:
        raise OperationalException("prepare Hedge backtest requires hedge.paper")

    backend = Backtesting(config)
    with backend.progress or nullcontext():
        data, timerange = backend.load_bt_data()
        if len(backend.strategylist) != 1:
            raise OperationalException("prepared Hedge backtest requires exactly one strategy")
        strategy = backend.strategylist[0]
        backend._set_strategy(strategy)
        pair = runtime.managed_pair
        if pair is None or pair not in data:
            raise OperationalException(f"managed_pair {pair!r} is not available in backtest data")
        from freqtrade.hedge.performance.resource_governor import (
            AdaptiveResourceGovernor,
            numeric_execution_context,
        )

        numeric_threads = AdaptiveResourceGovernor().numeric_threads(concurrent_python_workers=1)
        with numeric_execution_context(numeric_threads):
            analyzed = strategy.advise_all_indicators(data)
            frame = strategy.ft_advise_signals(analyzed[pair], {"pair": pair})
        frame = trim_dataframe(frame, timerange, startup_candles=backend.required_startup)
        if len(frame) < 2:
            raise OperationalException("prepared Hedge backtest requires at least two candles")
        preferred = (
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            *HEDGE_SIGNAL_COLUMNS,
            "enter_long",
            "enter_short",
        )
        narrow_names = [name for name in preferred if name in frame.columns]
        frame = frame.loc[:, narrow_names].copy()
        funding = backend.futures_data.get(pair)
        funding_frame = (
            None
            if funding is None or funding.empty
            else funding.loc[:, ["date", "open_fund", "open_mark"]].copy()
        )
        version_attr = strategy.version if hasattr(strategy, "version") else None
        strategy_version = version_attr() if callable(version_attr) else version_attr
        strategy_name = strategy.get_strategy_name()
        timeframe = backend.timeframe
        raw_hedge = config.get("hedge", {})
        hedge_mapping = raw_hedge if isinstance(raw_hedge, Mapping) else {}
        paper_raw = hedge_mapping.get("paper", {})
        paper_mapping = paper_raw if isinstance(paper_raw, Mapping) else {}
        rule_snapshot = exchange_market_rules(
            exchange=backend.exchange, pair=pair, fallback=paper_mapping
        )

        del analyzed, data, funding

    market_rules = MarketRules(
        tick_size=rule_snapshot.tick_size,
        qty_step=rule_snapshot.qty_step,
        min_qty=rule_snapshot.min_qty,
        min_notional=rule_snapshot.min_notional,
    )
    backend.dataprovider.clear_cache(include_backtesting=True)
    backend.futures_data.clear()
    backend.price_pair_prec.clear()
    backend.detail_data.clear()
    backend.available_pairs.clear()
    backend.strategylist.clear()
    backend.all_bt_content.clear()
    for cache in backend.analysis_results.values():
        cache.clear()
    backend.rejected_dict.clear()
    backend.wallet_captures.clear()
    strategy_dp_attribute = "dp"
    setattr(strategy, strategy_dp_attribute, None)
    strategy.wallets = None
    del backend, strategy
    release_phase_memory()

    return PreparedHedgeBacktest(
        pair=pair,
        timeframe=timeframe,
        frame=frame,
        funding_frame=funding_frame,
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        market_rules=market_rules,
        market_rule_source=rule_snapshot.source,
        market_rule_version=rule_snapshot.version,
        maker_fee_rate=rule_snapshot.maker_fee_rate,
        taker_fee_rate=rule_snapshot.taker_fee_rate,
    )


def _default_export_path(config: Mapping[str, Any]) -> Path:
    user_data = Path(str(config.get("user_data_dir", "user_data")))
    directory = user_data / "backtest_results"
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    return directory / f"hedge-backtest-{stamp}.json"


def _write_result(
    *,
    path: Path,
    result: SimulationResult,
    dataset: HedgeBacktestDataset,
    strategy: str,
    market_rule_source: str,
    market_rule_version: str,
    export_events: bool,
) -> tuple[str, str, object]:
    from freqtrade.hedge.native.backtest import HedgeBacktestResultAdapter

    native_artifact = HedgeBacktestResultAdapter().build(
        result,
        strategy_name=strategy,
        pairs=(dataset.pair,),
        timeframe=dataset.timeframe,
        timerange=f"{dataset.start.isoformat()}-{dataset.end.isoformat()}",
        metadata={
            "start": dataset.start.isoformat(),
            "end": dataset.end.isoformat(),
            "bar_count": dataset.bar_count,
            "signal_count": dataset.signal_count,
            "funding_count": dataset.funding_count,
            "data_fingerprint": dataset.data_fingerprint,
            "market_rule_source": market_rule_source,
            "market_rule_version": market_rule_version,
            "execution_timing": "NEXT_BAR_NO_LOOKAHEAD",
            "replay_mode": result.report.get("replay_mode", "FULL_MATERIALIZED"),
            "retained_snapshot_count": result.report.get(
                "retained_snapshot_count", len(result.snapshots)
            ),
            "retained_event_count": result.report.get("retained_event_count", len(result.events)),
        },
    )
    export_artifact = native_artifact if export_events else replace(native_artifact, events=())
    deterministic_payload: dict[str, object] = {
        "schema_version": "hedge-backtest-result-v4",
        "execution_timing": "NEXT_BAR_NO_LOOKAHEAD",
        "pair": dataset.pair,
        "timeframe": dataset.timeframe,
        "start": dataset.start,
        "end": dataset.end,
        "strategy": strategy,
        "market_rule_source": market_rule_source,
        "market_rule_version": market_rule_version,
        "bar_count": dataset.bar_count,
        "signal_count": dataset.signal_count,
        "funding_count": dataset.funding_count,
        "missing_candle_count": dataset.missing_candle_count,
        "data_fingerprint": dataset.data_fingerprint,
        "report": result.report,
        "snapshots": result.snapshots,
        "hedge_native": export_artifact.to_dict(),
        "freqtrade_projection": export_artifact.frequi_projection(),
    }
    if export_events:
        deterministic_payload["events"] = result.events
    canonical = json.dumps(
        _json_value(deterministic_payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    result_fingerprint = sha256(canonical).hexdigest()
    payload = {
        **deterministic_payload,
        "created_at": datetime.now(UTC),
        "result_fingerprint": result_fingerprint,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp.replace(path)
    digest = sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_temp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_temp.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    sidecar_temp.replace(sidecar)
    return digest, result_fingerprint, native_artifact


def run_freqtrade_hedge_backtest(
    config: dict[str, Any],
    *,
    export_path: Path | None = None,
    export_events: bool = False,
) -> HedgeBacktestRun:
    """Run a single-pair Hedge backtest through Freqtrade's normal data stack."""

    from freqtrade.data.converter import trim_dataframe
    from freqtrade.hedge.config import validate_hedge_config
    from freqtrade.hedge.integration.market_data import exchange_market_rules
    from freqtrade.hedge.integration.paper_runtime import planner_config_from_mapping
    from freqtrade.optimize.backtesting import Backtesting

    # Reuse Freqtrade's own backtest memory switch.  The upstream implementation
    # downcasts non-OHLCV float64/int64 indicator columns after populate_indicators;
    # Hedge enables it by default unless the caller explicitly opted out.
    config.setdefault(
        "reduce_df_footprint", DEFAULT_HEDGE_BACKTEST_MEMORY_POLICY.reduce_dataframe_footprint
    )

    hedge_runtime = validate_hedge_config(config)
    paper = hedge_runtime.paper
    if not hedge_runtime.enabled or paper is None:
        raise OperationalException("hedge-backtesting requires Hedge mode and hedge.paper")
    if hedge_runtime.operation_mode not in {"paper", "shadow"}:
        raise OperationalException(
            "hedge-backtesting requires hedge.operation_mode=paper or shadow"
        )

    backend = Backtesting(config)
    with backend.progress or nullcontext():
        data, timerange = backend.load_bt_data()
        if len(backend.strategylist) != 1:
            raise OperationalException(
                "hedge-backtesting currently supports exactly one strategy per run"
            )
        strategy = backend.strategylist[0]
        backend._set_strategy(strategy)
        pair = hedge_runtime.managed_pair
        if pair is None or pair not in data:
            raise OperationalException(
                f"managed_pair {pair!r} is not available in the loaded backtest data"
            )
        from freqtrade.hedge.performance.resource_governor import (
            AdaptiveResourceGovernor,
            numeric_execution_context,
        )

        numeric_threads = AdaptiveResourceGovernor().numeric_threads(concurrent_python_workers=1)
        with numeric_execution_context(numeric_threads):
            analyzed = strategy.advise_all_indicators(data)
            frame = strategy.ft_advise_signals(analyzed[pair], {"pair": pair})
        frame = trim_dataframe(
            frame,
            timerange,
            startup_candles=backend.required_startup,
        )
        del analyzed
        del data
        backtest_timeframe = backend.timeframe

    if len(frame) < 2:
        raise OperationalException(
            "hedge-backtesting requires at least two analyzed candles for next-bar execution"
        )
    version_attr = strategy.version if hasattr(strategy, "version") else None
    strategy_version = version_attr() if callable(version_attr) else version_attr
    strategy_name = strategy.get_strategy_name()
    optimization_runtime = config.get("hedge_optimization_runtime", {})
    runtime_mapping = optimization_runtime if isinstance(optimization_runtime, Mapping) else {}
    funding_rate_multiplier = _decimal(
        runtime_mapping.get("funding_rate_multiplier", "1"),
        field="hedge_optimization_runtime.funding_rate_multiplier",
    )
    if funding_rate_multiplier < 0:
        raise OperationalException(
            "hedge_optimization_runtime.funding_rate_multiplier cannot be negative"
        )

    detailed_dataset: HedgeBacktestDataset | None = None
    compact_stream: HedgeBacktestEventChunks | None = None
    if export_events:
        detailed_dataset = events_from_analyzed_dataframe(
            pair=pair,
            timeframe=backtest_timeframe,
            frame=frame,
            funding_frame=backend.futures_data.get(pair),
            strategy_version=strategy_version,
            require_funding_data=paper.funding_source is PaperFundingSource.EXCHANGE,
            max_missing_candles=paper.max_missing_candles,
            funding_rate_multiplier=funding_rate_multiplier,
        )
    else:
        compact_stream = HedgeBacktestEventChunks(
            pair=pair,
            timeframe=backtest_timeframe,
            frame=frame,
            funding_frame=backend.futures_data.get(pair),
            strategy_version=strategy_version,
            require_funding_data=paper.funding_source is PaperFundingSource.EXCHANGE,
            max_missing_candles=paper.max_missing_candles,
            funding_rate_multiplier=funding_rate_multiplier,
        )
        # The compact stream owns narrow copied arrays now.  Release the wide
        # analyzed frame and informative DataProvider cache before the million-bar
        # replay starts.  Funding arrays are detached by the stream as well.
        del frame
        backend.dataprovider.clear_cache(include_backtesting=True)
        # Backtesting prepares several large helper structures for the upstream
        # trade engine (funding dataframe, tick-size-over-time series and optional
        # detail candles).  Compact Hedge replay has already copied the narrow
        # funding inputs and does not use those structures, so release them before
        # the million-bar event loop starts.
        backend.futures_data.clear()
        backend.price_pair_prec.clear()
        backend.detail_data.clear()
        strategy_dp_attribute = "dp"
        setattr(strategy, strategy_dp_attribute, None)
        release_phase_memory()
    raw_hedge = config.get("hedge", {})
    hedge_mapping = raw_hedge if isinstance(raw_hedge, Mapping) else {}
    planner_raw = hedge_mapping.get("planner", {})
    planner_mapping = planner_raw if isinstance(planner_raw, Mapping) else {}
    paper_raw = hedge_mapping.get("paper", {})
    paper_mapping = paper_raw if isinstance(paper_raw, Mapping) else {}
    rule_snapshot = exchange_market_rules(
        exchange=backend.exchange,
        pair=pair,
        fallback=paper_mapping,
    )

    # One-shot Hedge replay has detached every input it needs.  Unlike upstream
    # Hyperopt, this Backtesting instance will never serve another epoch, so keeping
    # exchange markets, informative history, analyzed caches and Wallets alive only
    # increases the replay RSS.  Release the complete upstream object graph now.
    backend.dataprovider.clear_cache(include_backtesting=True)
    backend.futures_data.clear()
    backend.price_pair_prec.clear()
    backend.detail_data.clear()
    backend.available_pairs.clear()
    backend.strategylist.clear()
    backend.all_bt_content.clear()
    for cache in backend.analysis_results.values():
        cache.clear()
    backend.rejected_dict.clear()
    backend.wallet_captures.clear()
    strategy_dp_attribute = "dp"
    setattr(strategy, strategy_dp_attribute, None)
    strategy.wallets = None
    del backend, strategy
    release_phase_memory()

    market_rules = MarketRules(
        tick_size=rule_snapshot.tick_size,
        qty_step=rule_snapshot.qty_step,
        min_qty=rule_snapshot.min_qty,
        min_notional=rule_snapshot.min_notional,
    )
    maker_fee_multiplier = _decimal(
        runtime_mapping.get("maker_fee_multiplier", "1"),
        field="hedge_optimization_runtime.maker_fee_multiplier",
    )
    taker_fee_multiplier = _decimal(
        runtime_mapping.get("taker_fee_multiplier", "1"),
        field="hedge_optimization_runtime.taker_fee_multiplier",
    )
    if maker_fee_multiplier < 0 or taker_fee_multiplier < 0:
        raise OperationalException("hedge_optimization_runtime fee multipliers cannot be negative")
    match_config = _paper_match_config(paper)
    match_config = MatchConfig(
        **{
            **asdict(match_config),
            "maker_fee_rate": rule_snapshot.maker_fee_rate * maker_fee_multiplier,
            "taker_fee_rate": rule_snapshot.taker_fee_rate * taker_fee_multiplier,
            "price_tick": rule_snapshot.tick_size,
            "qty_step": rule_snapshot.qty_step,
            "min_fill_qty": rule_snapshot.min_qty,
            "min_fill_notional": rule_snapshot.min_notional,
        }
    )
    runner = HedgeBacktesting(
        initial_balance=paper.initial_balance,
        planner_config=planner_config_from_mapping(planner_mapping),
        leverage=paper.leverage,
        fee_rate=rule_snapshot.taker_fee_rate * taker_fee_multiplier,
        long_signal=paper.default_long_signal,
        short_signal=paper.default_short_signal,
        market_rules=market_rules,
        match_config=match_config,
    )
    if detailed_dataset is not None:
        dataset = detailed_dataset
        result = runner.run(dataset.events)
    else:
        if compact_stream is None:  # pragma: no cover - defensive invariant
            raise OperationalException("compact Hedge backtest stream was not initialized")
        result = runner.run_compact(compact_stream)
        dataset = compact_stream.dataset()
        del compact_stream, runner
        release_phase_memory()

    output = (export_path or _default_export_path(config)).expanduser().resolve()
    artifact_sha256, result_fingerprint, native_artifact = _write_result(
        path=output,
        result=result,
        dataset=dataset,
        strategy=strategy_name,
        market_rule_source=rule_snapshot.source,
        market_rule_version=rule_snapshot.version,
        export_events=export_events,
    )
    return HedgeBacktestRun(
        result=result,
        dataset=dataset,
        export_path=output,
        strategy=strategy_name,
        market_rule_source=rule_snapshot.source,
        market_rule_version=rule_snapshot.version,
        artifact_sha256=artifact_sha256,
        result_fingerprint=result_fingerprint,
        native_artifact=native_artifact,
    )
