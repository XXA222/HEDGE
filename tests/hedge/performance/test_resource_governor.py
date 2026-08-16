from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from freqtrade.hedge.performance.resource_governor import (
    AdaptiveResourceGovernor,
    ResourcePolicy,
    ResourceSnapshot,
)


_GIB = 1024 * 1024 * 1024


def snapshot(
    *,
    cpu: float = 10.0,
    current: int = 1 * _GIB,
    maximum: int = 8 * _GIB,
    host_available: int = 16 * _GIB,
) -> ResourceSnapshot:
    return ResourceSnapshot(
        logical_cpus=32,
        physical_cpus=16,
        affinity_cpus=32,
        system_cpu_percent=cpu,
        process_cpu_percent=0.0,
        cgroup_memory_limit_bytes=maximum,
        cgroup_memory_current_bytes=current,
        host_memory_available_bytes=host_available,
        timestamp_monotonic=time.monotonic(),
    )


class ResourceGovernorTest(unittest.TestCase):
    def test_idle_machine_uses_cpu_and_memory_budget(self) -> None:
        governor = AdaptiveResourceGovernor(ResourcePolicy(sample_seconds=0))
        self.assertEqual(
            governor.recommended_workers(tasks=100, requested=0, snapshot=snapshot()),
            24,
        )

    def test_hard_desktop_load_backs_off_to_one(self) -> None:
        governor = AdaptiveResourceGovernor(ResourcePolicy(sample_seconds=0))
        self.assertEqual(
            governor.recommended_workers(tasks=100, requested=0, snapshot=snapshot(cpu=99)),
            1,
        )

    def test_scale_up_is_bounded_after_work_has_started(self) -> None:
        governor = AdaptiveResourceGovernor(ResourcePolicy(sample_seconds=0, scale_up_step=4))
        self.assertEqual(
            governor.recommended_workers(
                tasks=100,
                requested=0,
                current_workers=8,
                snapshot=snapshot(cpu=40),
            ),
            12,
        )

    def test_explicit_upper_bound_is_respected(self) -> None:
        governor = AdaptiveResourceGovernor(ResourcePolicy(sample_seconds=0))
        self.assertEqual(
            governor.recommended_workers(tasks=100, requested=6, snapshot=snapshot(cpu=0)),
            6,
        )

    def test_minus_one_uses_resource_maximum_without_load_throttle(self) -> None:
        governor = AdaptiveResourceGovernor(ResourcePolicy(sample_seconds=0))
        self.assertEqual(
            governor.recommended_workers(tasks=100, requested=-1, snapshot=snapshot(cpu=99)),
            24,
        )

    def test_memory_pressure_caps_worker_count(self) -> None:
        governor = AdaptiveResourceGovernor(ResourcePolicy(sample_seconds=0))
        limited = snapshot(current=7 * _GIB, maximum=8 * _GIB)
        self.assertEqual(
            governor.recommended_workers(tasks=100, requested=-1, snapshot=limited),
            1,
        )

    def test_disabling_smt_reserves_physical_cpu_capacity(self) -> None:
        governor = AdaptiveResourceGovernor(
            ResourcePolicy(sample_seconds=0, use_smt=False, reserve_logical_cpus=2)
        )
        generous = snapshot(current=0, maximum=32 * _GIB, host_available=32 * _GIB)
        self.assertEqual(
            governor.recommended_workers(tasks=100, requested=-1, snapshot=generous),
            14,
        )

    def test_parallel_python_workers_force_single_numeric_thread(self) -> None:
        governor = AdaptiveResourceGovernor(ResourcePolicy(sample_seconds=0))
        self.assertEqual(
            governor.numeric_threads(concurrent_python_workers=8, snapshot=snapshot()),
            1,
        )

    def test_single_numeric_phase_can_use_many_threads(self) -> None:
        governor = AdaptiveResourceGovernor(ResourcePolicy(sample_seconds=0))
        self.assertEqual(
            governor.numeric_threads(concurrent_python_workers=1, snapshot=snapshot()),
            24,
        )

    def test_fresh_host_broker_snapshot_overrides_container_cpu(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "host.json"
            path.write_text(
                json.dumps(
                    {
                        "timestamp_epoch": time.time(),
                        "cpu_percent": 77.5,
                        "memory_available_bytes": 9 * _GIB,
                        "logical_cpus": 32,
                        "physical_cpus": 16,
                    }
                ),
                encoding="utf-8",
            )
            old = os.environ.get("HEDGE_HOST_RESOURCE_SNAPSHOT")
            os.environ["HEDGE_HOST_RESOURCE_SNAPSHOT"] = str(path)
            try:
                snap = AdaptiveResourceGovernor(ResourcePolicy(sample_seconds=0)).snapshot(
                    sample_seconds=0
                )
            finally:
                if old is None:
                    os.environ.pop("HEDGE_HOST_RESOURCE_SNAPSHOT", None)
                else:
                    os.environ["HEDGE_HOST_RESOURCE_SNAPSHOT"] = old
            self.assertEqual(snap.source, "host-broker")
            self.assertEqual(snap.system_cpu_percent, 77.5)
            self.assertEqual(snap.host_memory_available_bytes, 9 * _GIB)

    def test_stale_host_snapshot_is_ignored(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "host.json"
            path.write_text(
                json.dumps({"timestamp_epoch": time.time() - 60, "cpu_percent": 99}),
                encoding="utf-8",
            )
            old = os.environ.get("HEDGE_HOST_RESOURCE_SNAPSHOT")
            os.environ["HEDGE_HOST_RESOURCE_SNAPSHOT"] = str(path)
            try:
                snap = AdaptiveResourceGovernor(
                    ResourcePolicy(sample_seconds=0, host_snapshot_max_age_seconds=1)
                ).snapshot(sample_seconds=0)
            finally:
                if old is None:
                    os.environ.pop("HEDGE_HOST_RESOURCE_SNAPSHOT", None)
                else:
                    os.environ["HEDGE_HOST_RESOURCE_SNAPSHOT"] = old
            self.assertEqual(snap.source, "container")


if __name__ == "__main__":
    unittest.main()
