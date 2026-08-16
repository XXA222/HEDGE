from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)

_HOST_SNAPSHOT_DEFAULT = "/opt/freqtrade-hedge/user_data/runtime/host-resource-snapshot.json"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _read_cgroup_int(path: str) -> int | None:
    try:
        value = Path(path).read_text(encoding="ascii").strip()
    except OSError:
        return None
    if value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _cgroup_memory_values() -> tuple[int | None, int | None]:
    maximum = _read_cgroup_int("/sys/fs/cgroup/memory.max")
    current = _read_cgroup_int("/sys/fs/cgroup/memory.current")
    if maximum is None or current is None:
        maximum = _read_cgroup_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        current = _read_cgroup_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        if maximum is not None and maximum > (1 << 62):
            maximum = None
    return maximum, current


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _safe_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


@dataclass(frozen=True, slots=True)
class ResourcePolicy:
    """Policy for opportunistic process-level parallelism.

    One symbol's chronological replay remains serial for deterministic wallet/order
    state.  Independent candidates, pairs and folds may use processes to bypass the
    CPython GIL.  The governor leaves CPU and memory headroom for interactive desktop
    work and can consume a host-side Windows resource snapshot when running in Docker.
    """

    enabled: bool = True
    target_system_cpu_percent: float = 88.0
    hard_system_cpu_percent: float = 96.0
    reserve_logical_cpus: int = 2
    hard_max_workers: int = 28
    min_workers: int = 1
    worker_memory_mib: int = 256
    memory_reserve_mib: int = 1024
    sample_seconds: float = 0.05
    use_smt: bool = True
    host_snapshot_max_age_seconds: float = 5.0
    scale_up_step: int = 4

    @classmethod
    def from_env(cls) -> ResourcePolicy:
        target = min(100.0, max(10.0, _env_float("HEDGE_CPU_TARGET_PERCENT", 88.0)))
        hard = min(100.0, max(target, _env_float("HEDGE_CPU_HARD_PERCENT", 96.0)))
        return cls(
            enabled=_env_bool("HEDGE_AUTO_RESOURCES", True),
            target_system_cpu_percent=target,
            hard_system_cpu_percent=hard,
            reserve_logical_cpus=max(0, _env_int("HEDGE_CPU_RESERVE_LOGICAL", 2)),
            hard_max_workers=max(1, _env_int("HEDGE_MAX_WORKERS", 28)),
            min_workers=max(1, _env_int("HEDGE_MIN_WORKERS", 1)),
            worker_memory_mib=max(64, _env_int("HEDGE_WORKER_MEMORY_MIB", 256)),
            memory_reserve_mib=max(128, _env_int("HEDGE_MEMORY_RESERVE_MIB", 1024)),
            sample_seconds=max(0.0, _env_float("HEDGE_RESOURCE_SAMPLE_SECONDS", 0.05)),
            use_smt=_env_bool("HEDGE_USE_SMT", True),
            host_snapshot_max_age_seconds=max(
                0.5, _env_float("HEDGE_HOST_SNAPSHOT_MAX_AGE_SECONDS", 5.0)
            ),
            scale_up_step=max(1, _env_int("HEDGE_RESOURCE_SCALE_UP_STEP", 4)),
        )


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    logical_cpus: int
    physical_cpus: int
    affinity_cpus: int
    system_cpu_percent: float
    process_cpu_percent: float
    cgroup_memory_limit_bytes: int | None
    cgroup_memory_current_bytes: int | None
    host_memory_available_bytes: int | None
    timestamp_monotonic: float
    source: str = "container"
    host_snapshot_age_seconds: float | None = None

    @property
    def cgroup_memory_available_bytes(self) -> int | None:
        if self.cgroup_memory_limit_bytes is None or self.cgroup_memory_current_bytes is None:
            return None
        return max(0, self.cgroup_memory_limit_bytes - self.cgroup_memory_current_bytes)

    @property
    def effective_available_memory_bytes(self) -> int | None:
        values = [
            value
            for value in (
                self.cgroup_memory_available_bytes,
                self.host_memory_available_bytes,
            )
            if value is not None
        ]
        return min(values) if values else None


class AdaptiveResourceGovernor:
    """Sample host/container pressure and choose safe independent worker counts."""

    def __init__(self, policy: ResourcePolicy | None = None) -> None:
        self.policy = policy or ResourcePolicy.from_env()

    @staticmethod
    def _affinity_count(logical: int) -> int:
        try:
            sched_getaffinity = getattr(os, "sched_getaffinity", None)
            if callable(sched_getaffinity):
                return max(1, len(sched_getaffinity(0)))
            return max(1, logical)
        except (AttributeError, OSError):
            return max(1, logical)

    @staticmethod
    def _host_snapshot_path() -> Path:
        configured = os.environ.get("HEDGE_HOST_RESOURCE_SNAPSHOT")
        if configured:
            return Path(configured).expanduser()
        return Path(_HOST_SNAPSHOT_DEFAULT)

    def _read_host_snapshot(self) -> dict[str, object] | None:
        path = self._host_snapshot_path()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(raw, dict):
            return None
        written = _safe_float(raw.get("timestamp_epoch"))
        if written is None:
            return None
        age = max(0.0, time.time() - written)
        if age > self.policy.host_snapshot_max_age_seconds:
            return None
        return {**raw, "_age_seconds": age}

    def snapshot(self, *, sample_seconds: float | None = None) -> ResourceSnapshot:
        try:
            import psutil

            logical = int(psutil.cpu_count(logical=True) or os.cpu_count() or 1)
            physical = int(psutil.cpu_count(logical=False) or logical)
            interval = self.policy.sample_seconds if sample_seconds is None else sample_seconds
            system_cpu = float(psutil.cpu_percent(interval=max(0.0, interval)))
            process = psutil.Process()
            process_cpu = float(process.cpu_percent(interval=None))
            host_available = int(psutil.virtual_memory().available)
        except Exception:
            logical = int(os.cpu_count() or 1)
            physical = logical
            system_cpu = 0.0
            process_cpu = 0.0
            host_available = None

        source = "container"
        age: float | None = None
        host = self._read_host_snapshot()
        if host is not None:
            host_cpu = _safe_float(host.get("cpu_percent"))
            host_available_value = _safe_int(host.get("memory_available_bytes"))
            host_logical = _safe_int(host.get("logical_cpus"))
            host_physical = _safe_int(host.get("physical_cpus"))
            if host_cpu is not None:
                system_cpu = host_cpu
            if host_available_value is not None:
                host_available = host_available_value
            if host_logical:
                logical = min(logical, host_logical)
            if host_physical:
                physical = min(physical, host_physical)
            source = "host-broker"
            age = _safe_float(host.get("_age_seconds"))

        cgroup_limit, cgroup_current = _cgroup_memory_values()
        return ResourceSnapshot(
            logical_cpus=max(1, logical),
            physical_cpus=max(1, min(physical, logical)),
            affinity_cpus=self._affinity_count(logical),
            system_cpu_percent=max(0.0, min(100.0, system_cpu)),
            process_cpu_percent=max(0.0, process_cpu),
            cgroup_memory_limit_bytes=cgroup_limit,
            cgroup_memory_current_bytes=cgroup_current,
            host_memory_available_bytes=host_available,
            timestamp_monotonic=time.monotonic(),
            source=source,
            host_snapshot_age_seconds=age,
        )

    def _cpu_cap(self, snapshot: ResourceSnapshot) -> int:
        logical = min(snapshot.logical_cpus, snapshot.affinity_cpus)
        if not self.policy.use_smt:
            logical = min(logical, snapshot.physical_cpus)
        base = max(1, logical - self.policy.reserve_logical_cpus)
        return min(base, self.policy.hard_max_workers)

    def _memory_cap(self, snapshot: ResourceSnapshot) -> int:
        available = snapshot.effective_available_memory_bytes
        if available is None:
            return self.policy.hard_max_workers
        reserve = self.policy.memory_reserve_mib * 1024 * 1024
        worker = self.policy.worker_memory_mib * 1024 * 1024
        usable = max(0, available - reserve)
        return max(1, int(usable // worker))

    def recommended_workers(
        self,
        *,
        tasks: int,
        requested: int = 0,
        current_workers: int = 0,
        snapshot: ResourceSnapshot | None = None,
    ) -> int:
        """Return a process worker target.

        ``requested`` semantics:
        - ``1``: force serial execution.
        - ``>1``: explicit upper bound.
        - ``0``: adaptive desktop-aware execution.
        - ``-1``: resource-aware maximum without CPU-load throttling.
        """
        if tasks < 1:
            return 0
        if requested == 1 or not self.policy.enabled:
            return 1
        if requested < -1:
            raise ValueError("workers must be -1, 0, or a positive integer")

        snap = snapshot or self.snapshot()
        cap = min(tasks, self._cpu_cap(snap), self._memory_cap(snap))
        if requested > 1:
            cap = min(cap, requested)
        if requested == -1:
            return max(1, cap)

        logical = max(1, min(snap.logical_cpus, snap.affinity_cpus))
        # Total CPU is averaged across logical CPUs.  Approximate the share occupied
        # by already-running single-core replay workers so native desktop load is not
        # mistaken for project load when a fresh host-broker snapshot is available.
        own_share = min(100.0, max(0, current_workers) * 100.0 / logical)
        other_cpu = max(0.0, snap.system_cpu_percent - own_share)
        if other_cpu >= self.policy.hard_system_cpu_percent:
            return min(cap, self.policy.min_workers)

        target_budget = max(0.0, self.policy.target_system_cpu_percent - other_cpu)
        by_load = max(
            self.policy.min_workers,
            math.floor(target_budget * logical / 100.0),
        )
        target = max(1, min(cap, by_load))

        # Back off immediately when the desktop becomes busy, but scale up in bounded
        # steps to avoid a burst of process creation and copy-on-write page faults.
        if current_workers > 0 and target > current_workers:
            target = min(target, current_workers + self.policy.scale_up_step)
        return target

    def numeric_threads(
        self,
        *,
        concurrent_python_workers: int = 1,
        snapshot: ResourceSnapshot | None = None,
    ) -> int:
        """Choose BLAS/NumExpr threads without oversubscribing process workers."""
        snap = snapshot or self.snapshot(sample_seconds=0.0)
        logical = max(1, min(snap.logical_cpus, snap.affinity_cpus))
        if concurrent_python_workers > 1:
            return 1
        cap = max(1, logical - self.policy.reserve_logical_cpus)
        if not self.policy.use_smt:
            cap = min(cap, snap.physical_cpus)
        cap = min(cap, self.policy.hard_max_workers)
        if not self.policy.enabled:
            return cap
        if snap.system_cpu_percent >= self.policy.hard_system_cpu_percent:
            return 1
        # For a fresh vectorized phase, treat current total CPU as external load.
        # Idle machines receive most cores; busy desktops receive proportionally
        # fewer numeric threads without requiring a restart.
        remaining = max(0.0, self.policy.target_system_cpu_percent - snap.system_cpu_percent)
        by_load = max(1, math.floor(remaining * logical / 100.0))
        return min(cap, by_load)


def numeric_thread_budget(
    *,
    concurrent_python_workers: int = 1,
    policy: ResourcePolicy | None = None,
) -> int:
    return AdaptiveResourceGovernor(policy).numeric_threads(
        concurrent_python_workers=concurrent_python_workers
    )


def limit_numeric_threads(threads: int):
    """Return a threadpoolctl context manager, or a null context when unavailable."""
    from contextlib import nullcontext

    threads = max(1, int(threads))
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=threads)
    except Exception:
        return nullcontext()


_WORKER_NUMERIC_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextmanager
def worker_numeric_environment():
    """Set one-thread BLAS env before spawn workers import numeric libraries.

    Spawned workers import the target module before their initializer executes.
    Holding these environment values while a process pool is alive guarantees
    that every initial or replacement worker inherits the one-thread budget.
    The parent environment is restored when the pool closes.
    """
    previous = {name: os.environ.get(name) for name in _WORKER_NUMERIC_ENV}
    for name in _WORKER_NUMERIC_ENV:
        os.environ[name] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def configure_worker_numeric_threads() -> None:
    """Prevent oversubscription and make background workers desktop-friendly.

    A positive Unix nice value does not cap throughput while the machine is idle;
    it simply lets interactive/native applications win scheduling contention.
    """
    try:
        nice_increment = max(0, _env_int("HEDGE_WORKER_NICE", 5))
        if nice_increment and hasattr(os, "nice"):
            os.nice(nice_increment)
    except OSError:
        logger.debug("Unable to adjust Hedge worker nice value", exc_info=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except Exception:
        logger.debug("threadpoolctl worker limit unavailable", exc_info=True)


def multiprocessing_context():
    """Prefer copy-on-write ``fork`` in Linux containers, spawn elsewhere."""
    import multiprocessing as mp

    if sys.platform.startswith("linux"):
        try:
            return mp.get_context("fork")
        except ValueError:
            logger.debug("fork multiprocessing context unavailable", exc_info=True)
    return mp.get_context("spawn")


def numeric_execution_context(threads: int):
    """Limit BLAS and NumExpr to one coordinated thread budget for a phase."""
    from contextlib import contextmanager

    @contextmanager
    def _context():
        previous_numexpr: int | None = None
        try:
            import numexpr

            requested = max(1, int(threads))
            max_numexpr = max(1, _env_int("NUMEXPR_MAX_THREADS", requested))
            previous_numexpr = int(numexpr.set_num_threads(min(requested, max_numexpr)))
        except Exception:
            previous_numexpr = None
        try:
            with limit_numeric_threads(threads):
                yield
        finally:
            if previous_numexpr is not None:
                try:
                    import numexpr

                    numexpr.set_num_threads(previous_numexpr)
                except Exception:
                    logger.debug("Unable to restore NumExpr thread count", exc_info=True)

    return _context()
