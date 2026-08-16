"""In-memory read-only repository owned by the read-only infrastructure layer.

The in-memory implementation is intentionally complete enough for the real
Binance read-only runtime, local smoke tests, Dry-run and integration tests.  It
can be replaced by the direction-one SQLAlchemy repository without changing the
exchange service.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from threading import RLock
from uuid import uuid4

from freqtrade.hedge.exchange.base import (
    AccountEventFact,
    AccountSnapshotFact,
    AtomicReadonlyFactRepository,
    BalanceFact,
    CalibrationKind,
    ExchangeFactBatch,
    FillFact,
    OrderFact,
    PositionFact,
    ReadonlyHistoryCursorRepository,
    ReconciliationDiffFact,
)


class InMemoryReadonlyRepository(AtomicReadonlyFactRepository, ReadonlyHistoryCursorRepository):
    """Deterministic latest-fact repository with append-only event evidence."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._positions: dict[tuple[str, str, str], PositionFact] = {}
        self._orders: dict[tuple[str, str, str], OrderFact] = {}
        self._fills: dict[tuple[str, str, str], FillFact] = {}
        self._account_events: dict[tuple[str, str], AccountEventFact] = {}
        self._account_snapshots: dict[str, AccountSnapshotFact] = {}
        self._balances: dict[tuple[str, str], BalanceFact] = {}
        self._runs: dict[str, dict[str, object]] = {}
        self._diffs: dict[str, list[ReconciliationDiffFact]] = {}
        self._history_cursors: dict[tuple[str, str], int] = {}
        self._batches: list[ExchangeFactBatch] = []

    @staticmethod
    def _newer(incoming: object, current: object | None) -> bool:
        if current is None:
            return True
        incoming_ms = int(
            getattr(incoming, "update_time_ms", getattr(incoming, "event_time_ms", 0))
        )
        current_ms = int(getattr(current, "update_time_ms", getattr(current, "event_time_ms", 0)))
        incoming_observed = getattr(incoming, "observed_at", datetime.min)
        current_observed = getattr(current, "observed_at", datetime.min)
        return (incoming_ms, incoming_observed) >= (current_ms, current_observed)

    async def append_exchange_fact_batch(self, batch: ExchangeFactBatch) -> None:
        with self._lock:
            self._batches.append(batch)
        if batch.account_snapshot is not None:
            await self.append_account_snapshot(
                batch.account_snapshot,
                reconciliation_run_id=batch.reconciliation_run_id,
            )
        await self.append_balance_snapshots(
            batch.balances,
            reconciliation_run_id=batch.reconciliation_run_id,
        )
        await self.append_position_snapshots(
            batch.positions,
            reconciliation_run_id=batch.reconciliation_run_id,
        )
        await self.append_order_snapshots(
            batch.orders,
            reconciliation_run_id=batch.reconciliation_run_id,
        )
        await self.append_fill_events(
            batch.fills,
            reconciliation_run_id=batch.reconciliation_run_id,
        )
        await self.append_account_events(batch.account_events)
        if batch.reconciliation_run_id is not None:
            await self.append_reconciliation_diffs(
                batch.reconciliation_run_id,
                batch.reconciliation_diffs,
            )

    async def append_position_snapshots(
        self,
        facts: Sequence[PositionFact],
        *,
        reconciliation_run_id: str | None = None,
    ) -> None:
        del reconciliation_run_id
        with self._lock:
            for fact in facts:
                current = self._positions.get(fact.key)
                if self._newer(fact, current):
                    if fact.quantity == 0:
                        self._positions.pop(fact.key, None)
                    else:
                        self._positions[fact.key] = fact

    async def append_order_snapshots(
        self,
        facts: Sequence[OrderFact],
        *,
        reconciliation_run_id: str | None = None,
    ) -> None:
        del reconciliation_run_id
        with self._lock:
            for fact in facts:
                current = self._orders.get(fact.key)
                if self._newer(fact, current):
                    self._orders[fact.key] = fact

    async def append_fill_events(
        self,
        facts: Sequence[FillFact],
        *,
        reconciliation_run_id: str | None = None,
    ) -> None:
        del reconciliation_run_id
        with self._lock:
            for fact in facts:
                self._fills.setdefault(fact.key, fact)

    async def append_account_snapshot(
        self,
        fact: AccountSnapshotFact,
        *,
        reconciliation_run_id: str | None = None,
    ) -> None:
        del reconciliation_run_id
        with self._lock:
            current = self._account_snapshots.get(fact.account_id)
            if current is None or fact.observed_at >= current.observed_at:
                self._account_snapshots[fact.account_id] = fact

    async def append_balance_snapshots(
        self,
        facts: Sequence[BalanceFact],
        *,
        reconciliation_run_id: str | None = None,
    ) -> None:
        del reconciliation_run_id
        with self._lock:
            for fact in facts:
                key = (fact.account_id, fact.asset)
                current = self._balances.get(key)
                if current is None or fact.observed_at >= current.observed_at:
                    self._balances[key] = fact

    async def append_account_events(self, facts: Sequence[AccountEventFact]) -> None:
        with self._lock:
            for fact in facts:
                self._account_events.setdefault((fact.account_id, fact.identity), fact)

    async def begin_reconciliation(
        self,
        *,
        account_id: str,
        kind: CalibrationKind,
        started_at: datetime,
    ) -> str:
        run_id = uuid4().hex
        with self._lock:
            self._runs[run_id] = {
                "account_id": account_id,
                "kind": kind.value,
                "started_at": started_at,
                "completed_at": None,
                "status": "RUNNING",
                "reason": "",
            }
            self._diffs[run_id] = []
        return run_id

    async def append_reconciliation_diffs(
        self,
        run_id: str,
        diffs: Sequence[ReconciliationDiffFact],
    ) -> None:
        with self._lock:
            self._diffs.setdefault(run_id, []).extend(diffs)

    async def complete_reconciliation(
        self,
        run_id: str,
        *,
        completed_at: datetime,
        status: str,
        reason: str,
    ) -> None:
        with self._lock:
            run = self._runs.setdefault(run_id, {})
            run.update(completed_at=completed_at, status=status, reason=reason)

    async def load_active_positions(self, account_id: str) -> tuple[PositionFact, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        fact
                        for fact in self._positions.values()
                        if fact.account_id == account_id and fact.quantity != 0
                    ),
                    key=lambda item: (item.symbol, item.position_side),
                )
            )

    async def load_active_orders(self, account_id: str) -> tuple[OrderFact, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        fact
                        for fact in self._orders.values()
                        if fact.account_id == account_id and fact.active
                    ),
                    key=lambda item: (item.symbol, item.position_side, item.exchange_order_id),
                )
            )

    async def has_fill(self, account_id: str, symbol: str, exchange_trade_id: str) -> bool:
        with self._lock:
            return (account_id, symbol, exchange_trade_id) in self._fills

    async def load_history_cursor(self, account_id: str, cursor_name: str) -> int | None:
        with self._lock:
            return self._history_cursors.get((account_id, cursor_name))

    async def save_history_cursor(self, account_id: str, cursor_name: str, cursor_ms: int) -> None:
        with self._lock:
            self._history_cursors[(account_id, cursor_name)] = int(cursor_ms)

    def account_snapshot(self, account_id: str) -> AccountSnapshotFact | None:
        with self._lock:
            return self._account_snapshots.get(account_id)

    def reconciliation_runs(self, account_id: str) -> tuple[dict[str, object], ...]:
        with self._lock:
            rows = [
                dict(value, run_id=key)
                for key, value in self._runs.items()
                if value.get("account_id") == account_id
            ]
            return tuple(sorted(rows, key=lambda item: str(item.get("started_at"))))

    def reconciliation_diffs(self, run_id: str) -> tuple[ReconciliationDiffFact, ...]:
        with self._lock:
            return tuple(self._diffs.get(run_id, ()))

    @property
    def batches(self) -> tuple[ExchangeFactBatch, ...]:
        with self._lock:
            return tuple(self._batches)

    @property
    def account_events(self) -> tuple[AccountEventFact, ...]:
        """Return an immutable event-evidence snapshot for diagnostics/acceptance."""
        with self._lock:
            return tuple(
                sorted(
                    self._account_events.values(),
                    key=lambda item: (item.event_time_ms, item.event_type, item.identity),
                )
            )
