"""Bounded-memory policy for historical Hedge simulation.

The policy follows the same lifecycle principle used by Freqtrade's native
backtester: calculate indicators once, convert/detach the narrow execution
surface, then discard large DataFrames/caches before the candle loop.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HedgeBacktestMemoryPolicy:
    """Internal defaults for one-shot historical replay and optimization."""

    reduce_dataframe_footprint: bool = True
    release_backtesting_cache: bool = True
    retain_material_events: bool = False
    compact_wallet_history: bool = True
    snapshot_target_seconds: int = 24 * 60 * 60
    max_retained_snapshots: int = 2048
    phase_boundary_gc: bool = True

    def snapshot_every_bars(self, timeframe_seconds: int) -> int:
        if timeframe_seconds <= 0:
            raise ValueError("timeframe_seconds must be positive")
        return max(1, self.snapshot_target_seconds // timeframe_seconds)


DEFAULT_HEDGE_BACKTEST_MEMORY_POLICY = HedgeBacktestMemoryPolicy()


_LAST_MEMORY_RELEASE_MONOTONIC = 0.0


@dataclass(frozen=True, slots=True)
class RegularTimestampSequence(Sequence[datetime]):
    """O(1)-memory timestamp sequence for gap-free historical candles."""

    start: datetime
    step_seconds: int
    length: int

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.start.utcoffset() is None:
            raise ValueError("start must be timezone-aware")
        if self.step_seconds <= 0 or self.length < 0:
            raise ValueError("invalid regular timestamp sequence")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.length)
            return tuple(self[item] for item in range(start, stop, step))
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self.start + timedelta(seconds=self.step_seconds * index)


def _read_cgroup_int(path: str) -> int | None:
    try:
        value = Path(path).read_text(encoding="ascii").strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _memory_pressure_ratio() -> float | None:
    current = _read_cgroup_int("/sys/fs/cgroup/memory.current")
    maximum = _read_cgroup_int("/sys/fs/cgroup/memory.max")
    if current is None or maximum is None:
        current = _read_cgroup_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        maximum = _read_cgroup_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if maximum is not None and maximum > (1 << 62):
            maximum = None
    if current is not None and maximum is not None and maximum > 0:
        return current / maximum
    try:
        import psutil

        vm = psutil.virtual_memory()
        if vm.total > 0:
            return 1.0 - (vm.available / vm.total)
    except Exception:
        logger.debug("Unable to read host memory pressure", exc_info=True)
    return None


def release_phase_memory(
    *,
    trim_allocator: bool | None = None,
    force: bool = False,
) -> bool:
    """Release cyclic/transient memory only when pressure justifies the cost.

    Reference-counted pandas/numpy objects usually release immediately.  Full
    generation-2 collection and ``malloc_trim`` are therefore reserved for hard
    pressure.  Soft pressure uses a cheaper generation-1 collection.  A cooldown
    prevents repeated optimizer trials from spending seconds in GC.
    """
    global _LAST_MEMORY_RELEASE_MONOTONIC

    mode = os.environ.get("HEDGE_MEMORY_RELEASE_MODE", "adaptive").strip().lower()
    if mode in {"off", "0", "false", "never"} and not force:
        return False

    rss = _rss_bytes()
    pressure = _memory_pressure_ratio()
    try:
        rss_threshold = int(float(os.environ.get("HEDGE_MEMORY_GC_RSS_MIB", "768")) * 1024 * 1024)
    except ValueError:
        rss_threshold = 768 * 1024 * 1024
    try:
        soft_pressure = float(os.environ.get("HEDGE_MEMORY_GC_PRESSURE_RATIO", "0.55"))
    except ValueError:
        soft_pressure = 0.55
    try:
        hard_pressure = float(os.environ.get("HEDGE_MEMORY_GC_HARD_PRESSURE_RATIO", "0.80"))
    except ValueError:
        hard_pressure = 0.80
    try:
        cooldown = max(0.0, float(os.environ.get("HEDGE_MEMORY_GC_COOLDOWN_SECONDS", "2")))
    except ValueError:
        cooldown = 2.0

    aggressive = force or mode in {"always", "1", "true", "aggressive"}
    hard = aggressive or (pressure is not None and pressure >= hard_pressure)
    soft = (rss is not None and rss >= rss_threshold) or (
        pressure is not None and pressure >= soft_pressure
    )
    if not hard and not soft:
        return False

    now = time.monotonic()
    if not force and cooldown and now - _LAST_MEMORY_RELEASE_MONOTONIC < cooldown:
        return False

    gc.collect(2 if hard else 1)
    _LAST_MEMORY_RELEASE_MONOTONIC = now

    if trim_allocator is None:
        trim_allocator = os.environ.get("HEDGE_MEMORY_TRIM", "1") not in {
            "0",
            "false",
            "False",
        }
    # malloc_trim is intentionally a hard-pressure operation.
    if not hard or not trim_allocator or not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        trim = getattr(libc, "malloc_trim", None)
        if trim is None:
            return False
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return bool(trim(0))
    except (OSError, AttributeError):
        return False
