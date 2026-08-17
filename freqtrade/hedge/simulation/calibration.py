"""Versioned differential calibration between recorded Binance and simulator traces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path

from freqtrade.hedge.contracts import IntentAction, PositionSide, finite_decimal
from freqtrade.hedge.contracts.types import required_text


def _hash(value: object, *, name: str) -> str:
    value = required_text(value, field_name=name, max_length=64).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be sha256")
    return value


def _nonnegative(value: object, *, name: str) -> Decimal:
    result = finite_decimal(value, field_name=name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class ExecutionTraceState:
    """One post-intent state, shared by a recorded exchange and simulation trace."""

    sequence: int
    client_order_id: str
    observed_at: datetime
    position_side: PositionSide
    action: IntentAction
    accepted: bool
    fill_quantity: Decimal
    fill_price: Decimal | None
    fee: Decimal
    funding: Decimal
    wallet_balance: Decimal
    equity: Decimal
    gross_notional: Decimal
    net_notional: Decimal
    pending_order_ids: tuple[str, ...]
    liquidation_buffer: Decimal
    long_quantity: Decimal
    short_quantity: Decimal

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("sequence must be a nonnegative int")
        object.__setattr__(self, "client_order_id", required_text(self.client_order_id, field_name="client_order_id", max_length=128))
        if not isinstance(self.observed_at, datetime) or self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(UTC))
        if not isinstance(self.position_side, PositionSide) or not isinstance(self.action, IntentAction):
            raise TypeError("position_side/action must use canonical enums")
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be bool")
        for name in (
            "fill_quantity", "fee", "wallet_balance", "equity", "gross_notional",
            "liquidation_buffer", "long_quantity", "short_quantity",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name=name))
        object.__setattr__(self, "funding", finite_decimal(self.funding, field_name="funding"))
        object.__setattr__(self, "net_notional", finite_decimal(self.net_notional, field_name="net_notional"))
        if self.fill_price is not None:
            price = _nonnegative(self.fill_price, name="fill_price")
            if price <= 0:
                raise ValueError("fill_price must be positive when supplied")
            object.__setattr__(self, "fill_price", price)
        if self.fill_quantity and self.fill_price is None:
            raise ValueError("positive fill_quantity requires fill_price")
        pending = tuple(required_text(item, field_name="pending_order_id", max_length=128) for item in self.pending_order_ids)
        if len(set(pending)) != len(pending):
            raise ValueError("pending_order_ids must be unique")
        object.__setattr__(self, "pending_order_ids", pending)


@dataclass(frozen=True, slots=True)
class TraceCorpus:
    schema_version: str
    corpus_sha256: str
    states: tuple[ExecutionTraceState, ...]

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("schema_version is required")
        object.__setattr__(self, "corpus_sha256", _hash(self.corpus_sha256, name="corpus_sha256"))
        states = tuple(self.states)
        if not states:
            raise ValueError("trace corpus cannot be empty")
        if tuple(state.sequence for state in states) != tuple(range(len(states))):
            raise ValueError("trace sequence must be contiguous from zero")
        object.__setattr__(self, "states", states)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "TraceCorpus":
        """Load a versioned, content-addressed recorded trace without network access."""
        raw = Path(path).read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("states"), list):
            raise ValueError("trace corpus must contain a states list")
        digest = sha256(raw).hexdigest()
        rows = []
        for row in payload["states"]:
            if not isinstance(row, dict):
                raise ValueError("trace state must be object")
            item = dict(row)
            item["observed_at"] = datetime.fromisoformat(str(item["observed_at"]))
            for name in ("fill_quantity", "fill_price", "fee", "funding", "wallet_balance", "equity", "gross_notional", "net_notional", "liquidation_buffer", "long_quantity", "short_quantity"):
                if item.get(name) is not None:
                    item[name] = Decimal(str(item[name]))
            item["position_side"] = PositionSide(item["position_side"])
            item["action"] = IntentAction(item["action"])
            item["pending_order_ids"] = tuple(item.get("pending_order_ids", ()))
            rows.append(ExecutionTraceState(**item))
        return cls(str(payload.get("schema_version", "")), digest, tuple(rows))


@dataclass(frozen=True, slots=True)
class SimulatorCalibrationArtifact:
    source_authority_sha256: str
    simulator_schema: str
    matcher_sha256: str
    cost_model_sha256: str
    funding_model_sha256: str
    latency_model_sha256: str
    trace_corpus_sha256: str
    case_count: int
    state_match_rate: Decimal
    max_pnl_error: Decimal
    max_fill_error: Decimal
    passed: bool

    def __post_init__(self) -> None:
        for name in (
            "source_authority_sha256", "matcher_sha256", "cost_model_sha256",
            "funding_model_sha256", "latency_model_sha256", "trace_corpus_sha256",
        ):
            object.__setattr__(self, name, _hash(getattr(self, name), name=name))
        if not self.simulator_schema.strip() or self.case_count <= 0:
            raise ValueError("simulator_schema and positive case_count are required")
        for name in ("state_match_rate", "max_pnl_error", "max_fill_error"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name=name))
        if self.state_match_rate > 1:
            raise ValueError("state_match_rate must be within [0, 1]")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")

    @property
    def fingerprint(self) -> str:
        payload = {name: str(getattr(self, name)) for name in self.__dataclass_fields__}
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def calibrate_simulation_trace(
    *,
    recorded: TraceCorpus,
    simulated: TraceCorpus,
    source_authority_sha256: str,
    simulator_schema: str,
    matcher_sha256: str,
    cost_model_sha256: str,
    funding_model_sha256: str,
    latency_model_sha256: str,
    max_pnl_error: Decimal = Decimal(0),
    max_fill_error: Decimal = Decimal(0),
) -> SimulatorCalibrationArtifact:
    """Compare every canonical state; qualification fails closed on any divergence."""

    if recorded.schema_version != simulated.schema_version:
        raise ValueError("recorded and simulated traces use different schemas")
    expected_pnl = _nonnegative(max_pnl_error, name="max_pnl_error")
    expected_fill = _nonnegative(max_fill_error, name="max_fill_error")
    matched = 0
    pnl_error = Decimal(0)
    fill_error = Decimal(0)
    for observed, replayed in zip(recorded.states, simulated.states, strict=False):
        pnl_error = max(pnl_error, abs(observed.equity - replayed.equity))
        fill_error = max(fill_error, abs(observed.fill_quantity - replayed.fill_quantity))
        if observed == replayed:
            matched += 1
    count = max(len(recorded.states), len(simulated.states))
    match_rate = Decimal(matched) / Decimal(count)
    passed = (
        len(recorded.states) == len(simulated.states)
        and match_rate == Decimal(1)
        and pnl_error <= expected_pnl
        and fill_error <= expected_fill
    )
    return SimulatorCalibrationArtifact(
        source_authority_sha256=source_authority_sha256,
        simulator_schema=simulator_schema,
        matcher_sha256=matcher_sha256,
        cost_model_sha256=cost_model_sha256,
        funding_model_sha256=funding_model_sha256,
        latency_model_sha256=latency_model_sha256,
        trace_corpus_sha256=recorded.corpus_sha256,
        case_count=count,
        state_match_rate=match_rate,
        max_pnl_error=pnl_error,
        max_fill_error=fill_error,
        passed=passed,
    )
