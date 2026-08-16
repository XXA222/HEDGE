"""Reliable dispatcher for transactionally recorded execution outbox events."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from freqtrade.hedge.contracts.events import OutboxEvent
from freqtrade.hedge.contracts.ports import ClockPort, EventPublisherPort, SystemClock
from freqtrade.hedge.errors import is_definitive_error, is_retryable_error


class OutboxStorePort(Protocol):
    def outbox(self, *, unpublished_only: bool = False) -> Sequence[OutboxEvent]: ...

    def mark_published(
        self,
        event_id: str,
        *,
        published_at: datetime | None = None,
    ) -> None: ...

    def mark_publish_attempt(self, event_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class OutboxDispatchReport:
    attempted: int
    published: int
    failed: tuple[str, ...] = ()
    retryable: tuple[str, ...] = ()
    definitive: tuple[str, ...] = ()


class OutboxDispatcher:
    def __init__(
        self,
        store: OutboxStorePort,
        publisher: EventPublisherPort,
        *,
        clock: ClockPort | None = None,
    ) -> None:
        self._store = store
        self._publisher = publisher
        self._clock = clock or SystemClock()

    def dispatch(self, *, limit: int = 100) -> OutboxDispatchReport:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        events = tuple(self._store.outbox(unpublished_only=True))[:limit]
        published = 0
        failures: list[str] = []
        retryable_failures: list[str] = []
        definitive_failures: list[str] = []
        for event in events:
            try:
                self._publisher.publish(event)
            except Exception as exc:
                self._store.mark_publish_attempt(str(event.event_id))
                failure = f"{event.event_id}:{type(exc).__name__}"
                failures.append(failure)
                if is_definitive_error(exc):
                    definitive_failures.append(failure)
                elif is_retryable_error(exc):
                    retryable_failures.append(failure)
                else:
                    # Preserve the historic outbox behavior for unclassified
                    # publisher errors: an unpublished event remains retryable.
                    retryable_failures.append(failure)
                continue
            self._store.mark_published(
                str(event.event_id),
                published_at=self._clock.now(),
            )
            published += 1
        return OutboxDispatchReport(
            len(events),
            published,
            tuple(failures),
            tuple(retryable_failures),
            tuple(definitive_failures),
        )
