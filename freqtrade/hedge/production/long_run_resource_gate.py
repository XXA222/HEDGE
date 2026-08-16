"""Measured memory/throughput stability gate for long HPRL replay/backtest runs."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite, sqrt
from statistics import mean

MIB = 1024 * 1024
HOUR = 3600.0


@dataclass(frozen=True, slots=True)
class ResourceSample:
    elapsed_seconds: float
    rss_bytes: int
    cpu_percent: float
    bars_completed: int
    events_completed: int

    def __post_init__(self) -> None:
        if not isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        if self.rss_bytes <= 0 or self.bars_completed < 0 or self.events_completed < 0:
            raise ValueError("resource counters are invalid")
        if not isfinite(self.cpu_percent) or self.cpu_percent < 0:
            raise ValueError("cpu_percent must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class ResourceStabilityPolicy:
    minimum_samples: int = 20
    minimum_duration_seconds: float = 600.0
    maximum_peak_rss_bytes: int = 12 * 1024**3
    maximum_rss_slope_bytes_per_hour: float = 128 * MIB
    maximum_tail_growth_ratio: float = 0.10
    maximum_throughput_cv: float = 0.35
    minimum_progress_bars: int = 10_000

    def __post_init__(self) -> None:
        if self.minimum_samples < 3 or self.minimum_duration_seconds <= 0:
            raise ValueError("resource sampling minima are invalid")
        if self.maximum_peak_rss_bytes <= 0 or self.maximum_rss_slope_bytes_per_hour < 0:
            raise ValueError("memory ceilings are invalid")
        if self.maximum_tail_growth_ratio < 0 or self.maximum_throughput_cv < 0:
            raise ValueError("stability ratios cannot be negative")
        if self.minimum_progress_bars <= 0:
            raise ValueError("minimum_progress_bars must be positive")


@dataclass(frozen=True, slots=True)
class ResourceStabilityReport:
    passed: bool
    samples: int
    duration_seconds: float
    peak_rss_bytes: int
    rss_slope_bytes_per_hour: float
    head_rss_bytes: float
    tail_rss_bytes: float
    tail_growth_ratio: float
    mean_bars_per_second: float
    throughput_cv: float
    bars_progress: int
    events_progress: int
    evidence_sha256: str
    reasons: tuple[str, ...]


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_bar = mean(xs)
    y_bar = mean(ys)
    denominator = sum((x - x_bar) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys, strict=True)) / denominator


def _throughput(rows: tuple[ResourceSample, ...]) -> list[float]:
    values: list[float] = []
    for previous, current in zip(rows, rows[1:], strict=False):
        seconds = current.elapsed_seconds - previous.elapsed_seconds
        bars = current.bars_completed - previous.bars_completed
        if seconds > 0 and bars >= 0:
            values.append(bars / seconds)
    return values



def _sequence_reasons(
    rows: tuple[ResourceSample, ...],
    policy: ResourceStabilityPolicy,
) -> list[str]:
    reasons: list[str] = []
    checks = (
        (len(rows) < policy.minimum_samples, "RESOURCE_SAMPLE_COUNT_INSUFFICIENT"),
        (
            any(
                current.elapsed_seconds <= previous.elapsed_seconds
                for previous, current in zip(rows, rows[1:], strict=False)
            ),
            "RESOURCE_SAMPLE_TIME_NOT_STRICTLY_MONOTONIC",
        ),
        (
            any(
                current.bars_completed < previous.bars_completed
                for previous, current in zip(rows, rows[1:], strict=False)
            ),
            "RESOURCE_BAR_PROGRESS_REGRESSED",
        ),
        (
            any(
                current.events_completed < previous.events_completed
                for previous, current in zip(rows, rows[1:], strict=False)
            ),
            "RESOURCE_EVENT_PROGRESS_REGRESSED",
        ),
    )
    reasons.extend(reason for failed, reason in checks if failed)
    return reasons


def _memory_metrics(
    rows: tuple[ResourceSample, ...],
    policy: ResourceStabilityPolicy,
) -> tuple[float, int, float, float, float, float, list[str]]:
    reasons: list[str] = []
    duration = rows[-1].elapsed_seconds - rows[0].elapsed_seconds
    peak = max(item.rss_bytes for item in rows)
    slope_per_second = _linear_slope(
        [item.elapsed_seconds for item in rows],
        [float(item.rss_bytes) for item in rows],
    )
    slope_per_hour = slope_per_second * HOUR
    window = max(1, len(rows) // 5)
    head = mean(item.rss_bytes for item in rows[:window])
    tail = mean(item.rss_bytes for item in rows[-window:])
    tail_growth = max(0.0, (tail - head) / head) if head > 0 else 0.0
    checks = (
        (duration < policy.minimum_duration_seconds, "RESOURCE_DURATION_INSUFFICIENT"),
        (peak > policy.maximum_peak_rss_bytes, "RESOURCE_PEAK_RSS_EXCEEDED"),
        (
            slope_per_hour > policy.maximum_rss_slope_bytes_per_hour,
            "RESOURCE_RSS_SLOPE_EXCEEDED",
        ),
        (
            tail_growth > policy.maximum_tail_growth_ratio,
            "RESOURCE_TAIL_MEMORY_GROWTH_EXCEEDED",
        ),
    )
    reasons.extend(reason for failed, reason in checks if failed)
    return duration, peak, slope_per_hour, head, tail, tail_growth, reasons


def _throughput_metrics(
    rows: tuple[ResourceSample, ...],
    policy: ResourceStabilityPolicy,
) -> tuple[float, float, int, int, list[str]]:
    reasons: list[str] = []
    throughput = _throughput(rows)
    throughput_mean = mean(throughput) if throughput else 0.0
    throughput_cv = float("inf")
    if throughput_mean <= 0:
        reasons.append("RESOURCE_NO_FORWARD_THROUGHPUT")
    else:
        variance = mean((value - throughput_mean) ** 2 for value in throughput)
        throughput_cv = sqrt(variance) / throughput_mean
        if throughput_cv > policy.maximum_throughput_cv:
            reasons.append("RESOURCE_THROUGHPUT_VARIANCE_EXCEEDED")
    bars_progress = rows[-1].bars_completed - rows[0].bars_completed
    events_progress = rows[-1].events_completed - rows[0].events_completed
    if bars_progress < policy.minimum_progress_bars:
        reasons.append("RESOURCE_BAR_PROGRESS_INSUFFICIENT")
    return throughput_mean, throughput_cv, bars_progress, events_progress, reasons

def evaluate_resource_stability(
    samples: Iterable[ResourceSample],
    *,
    policy: ResourceStabilityPolicy | None = None,
) -> ResourceStabilityReport:
    p = policy or ResourceStabilityPolicy()
    rows = tuple(sorted(samples, key=lambda item: item.elapsed_seconds))
    if not rows:
        reasons = ["RESOURCE_SAMPLE_COUNT_INSUFFICIENT"]
        digest = sha256(b"[]").hexdigest()
        return ResourceStabilityReport(
            False, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, digest, tuple(reasons)
        )
    reasons = _sequence_reasons(rows, p)
    duration, peak, slope_per_hour, head, tail, tail_growth, memory_reasons = (
        _memory_metrics(rows, p)
    )
    reasons.extend(memory_reasons)
    throughput_mean, throughput_cv, bars_progress, events_progress, throughput_reasons = (
        _throughput_metrics(rows, p)
    )
    reasons.extend(throughput_reasons)
    payload = {
        "policy": asdict(p),
        "samples": [asdict(item) for item in rows],
        "derived": {
            "duration": duration,
            "peak": peak,
            "slope_per_hour": slope_per_hour,
            "head": head,
            "tail": tail,
            "tail_growth": tail_growth,
            "throughput_mean": throughput_mean,
            "throughput_cv": throughput_cv,
            "bars_progress": bars_progress,
            "events_progress": events_progress,
        },
    }
    return ResourceStabilityReport(
        passed=not reasons,
        samples=len(rows),
        duration_seconds=duration,
        peak_rss_bytes=peak,
        rss_slope_bytes_per_hour=slope_per_hour,
        head_rss_bytes=head,
        tail_rss_bytes=tail,
        tail_growth_ratio=tail_growth,
        mean_bars_per_second=throughput_mean,
        throughput_cv=throughput_cv,
        bars_progress=bars_progress,
        events_progress=events_progress,
        evidence_sha256=sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        reasons=tuple(dict.fromkeys(reasons)),
    )


@dataclass(frozen=True, slots=True)
class LongRunClosurePolicy:
    maximum_peak_rss_repeat_ratio: float = 1.20
    maximum_elapsed_repeat_ratio: float = 1.25

    def __post_init__(self) -> None:
        if self.maximum_peak_rss_repeat_ratio < 1 or self.maximum_elapsed_repeat_ratio < 1:
            raise ValueError("repeat consistency ratios must be >= 1")


@dataclass(frozen=True, slots=True)
class LongRunClosureReport:
    passed: bool
    two_year_passed: bool
    primary_resource_passed: bool
    repeat_resource_passed: bool
    peak_rss_repeat_ratio: float
    elapsed_repeat_ratio: float
    evidence_sha256: str
    reasons: tuple[str, ...]


def _ratio(left: float, right: float) -> float:
    smaller = min(left, right)
    larger = max(left, right)
    return float("inf") if smaller <= 0 else larger / smaller


def evaluate_long_run_closure(
    two_year_report: object,
    primary_resource: ResourceStabilityReport,
    repeat_resource: ResourceStabilityReport,
    *,
    policy: LongRunClosurePolicy | None = None,
) -> LongRunClosureReport:
    p = policy or LongRunClosurePolicy()
    reasons: list[str] = []
    two_year_passed = bool(getattr(two_year_report, "passed", False))
    if not two_year_passed:
        reasons.append("LONG_RUN_TWO_YEAR_BACKTEST_NOT_PASSED")
    if not primary_resource.passed:
        reasons.append("LONG_RUN_PRIMARY_RESOURCE_NOT_PASSED")
    if not repeat_resource.passed:
        reasons.append("LONG_RUN_REPEAT_RESOURCE_NOT_PASSED")
    peak_ratio = _ratio(
        float(primary_resource.peak_rss_bytes),
        float(repeat_resource.peak_rss_bytes),
    )
    elapsed_ratio = _ratio(primary_resource.duration_seconds, repeat_resource.duration_seconds)
    if peak_ratio > p.maximum_peak_rss_repeat_ratio:
        reasons.append("LONG_RUN_REPEAT_PEAK_RSS_INCONSISTENT")
    if elapsed_ratio > p.maximum_elapsed_repeat_ratio:
        reasons.append("LONG_RUN_REPEAT_ELAPSED_INCONSISTENT")
    payload = {
        "two_year_passed": two_year_passed,
        "two_year_digest": str(getattr(two_year_report, "aggregate_sha256", "")),
        "primary": primary_resource.evidence_sha256,
        "repeat": repeat_resource.evidence_sha256,
        "peak_ratio": peak_ratio,
        "elapsed_ratio": elapsed_ratio,
        "policy": asdict(p),
    }
    return LongRunClosureReport(
        passed=not reasons,
        two_year_passed=two_year_passed,
        primary_resource_passed=primary_resource.passed,
        repeat_resource_passed=repeat_resource.passed,
        peak_rss_repeat_ratio=peak_ratio,
        elapsed_repeat_ratio=elapsed_ratio,
        evidence_sha256=sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        reasons=tuple(reasons),
    )
