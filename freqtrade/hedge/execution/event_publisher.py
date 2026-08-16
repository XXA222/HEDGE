"""Synchronous outbox publisher adapters for telemetry and tests."""

from __future__ import annotations

import asyncio
import logging
import sys
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future as ConcurrentFuture
from threading import RLock
from typing import Any

from freqtrade.hedge.contracts.events import OutboxEvent
from freqtrade.hedge.telemetry.events import HedgeEventType, HedgeTelemetryEvent


logger = logging.getLogger(__name__)


class InMemoryEventPublisher:
    def __init__(
        self,
        callback: Callable[[OutboxEvent], None] | None = None,
        *,
        capacity: int = 10_000,
    ) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._events: deque[OutboxEvent] = deque(maxlen=capacity)
        self._callbacks: list[Callable[[OutboxEvent], None]] = []
        if callback is not None:
            self._callbacks.append(callback)
        self._lock = RLock()

    def publish(self, event: OutboxEvent) -> None:
        if not isinstance(event, OutboxEvent):
            raise TypeError("event must be an OutboxEvent")
        with self._lock:
            self._events.append(event)
            callbacks = tuple(self._callbacks)
        for callback in callbacks:
            callback(event)

    def events(self) -> tuple[OutboxEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def set_callback(self, callback: Callable[[OutboxEvent], None] | None) -> None:
        """Replace callbacks for backward compatibility."""
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None")
        with self._lock:
            self._callbacks = [] if callback is None else [callback]

    def add_callback(self, callback: Callable[[OutboxEvent], None]) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[OutboxEvent], None]) -> None:
        with self._lock:
            self._callbacks = [item for item in self._callbacks if item != callback]

    @property
    def callback_count(self) -> int:
        with self._lock:
            return len(self._callbacks)

    def collection_gauges(self) -> dict[str, int]:
        """Expose bounded buffer sizes for runtime monitoring."""
        with self._lock:
            return {
                "publisher_events": len(self._events),
                "publisher_callbacks": len(self._callbacks),
            }


class HedgeEventHubPublisher:
    """Publish execution outbox events to one owning async loop without blocking."""

    def __init__(
        self,
        hub: object,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        publish = getattr(hub, "publish", None)
        if not callable(publish):
            raise TypeError("hub must expose async publish(event)")
        self._hub: Any = hub
        self._loop = loop
        self._background_tasks: set[asyncio.Task[None]] = set()

    @staticmethod
    def _log_threadsafe_failure(future: ConcurrentFuture[object]) -> None:
        try:
            future.result()
        except Exception:
            logger.exception("Hedge event hub threadsafe publish failed")

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._background_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            logger.debug("Hedge event hub publish task was cancelled")
        except Exception:
            logger.exception("Hedge event hub async publish failed")

    def publish(self, event: OutboxEvent) -> None:
        category = event.event_type.split("_", 1)[0]
        event_type = HedgeEventType.ORDER
        if category == "INTENT":
            event_type = HedgeEventType.INTENT
        elif category == "FILL":
            event_type = HedgeEventType.FILL
        elif category == "HALT":
            event_type = HedgeEventType.HALT
        raw_symbol = event.payload.get("symbol")
        symbol = None if raw_symbol is None else sys.intern(str(raw_symbol))
        event_name = sys.intern(event.event_type)
        telemetry = HedgeTelemetryEvent(
            event_type=event_type,
            payload={"event_type": event_name, **dict(event.payload)},
            account_id=sys.intern(str(event.payload.get("account_id", "default"))),
            symbol=symbol,
            correlation_id=event.correlation_id,
            occurred_at=event.occurred_at,
        )
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        target = self._loop
        if running_loop is not None and (target is None or target is running_loop):
            if target is None:
                self._loop = running_loop
            task = running_loop.create_task(self._hub.publish(telemetry))
            self._background_tasks.add(task)
            task.add_done_callback(self._task_done)
            return

        if target is None or target.is_closed() or not target.is_running():
            logger.warning(
                "Hedge event hub loop unavailable; dropping event_type=%s",
                event.event_type,
            )
            return

        future = asyncio.run_coroutine_threadsafe(self._hub.publish(telemetry), target)
        future.add_done_callback(self._log_threadsafe_failure)
