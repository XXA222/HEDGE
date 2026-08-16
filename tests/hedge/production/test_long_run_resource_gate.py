from dataclasses import dataclass

from freqtrade.hedge.production.long_run_resource_gate import (
    LongRunClosurePolicy,
    ResourceSample,
    ResourceStabilityPolicy,
    evaluate_long_run_closure,
    evaluate_resource_stability,
)


def _samples(*, leak_per_step=0, seconds=60, count=21):
    rows = []
    rss = 1_000_000_000
    for index in range(count):
        rows.append(
            ResourceSample(
                index * seconds,
                rss + index * leak_per_step,
                50.0,
                index * 1000,
                index * 1200,
            )
        )
    return rows


def test_stable_memory_and_throughput_pass():
    policy = ResourceStabilityPolicy(
        minimum_duration_seconds=60,
        minimum_progress_bars=1000,
    )
    report = evaluate_resource_stability(_samples(), policy=policy)
    assert report.passed
    assert report.rss_slope_bytes_per_hour == 0


def test_memory_leak_slope_fails_before_oom():
    policy = ResourceStabilityPolicy(
        minimum_duration_seconds=60,
        minimum_progress_bars=1000,
        maximum_rss_slope_bytes_per_hour=10_000_000,
    )
    report = evaluate_resource_stability(_samples(leak_per_step=20_000_000), policy=policy)
    assert not report.passed
    assert "RESOURCE_RSS_SLOPE_EXCEEDED" in report.reasons


@dataclass
class _TwoYear:
    passed: bool
    aggregate_sha256: str = "a" * 64


def test_long_run_requires_two_year_and_repeat_resource_evidence():
    policy = ResourceStabilityPolicy(minimum_duration_seconds=60, minimum_progress_bars=1000)
    primary = evaluate_resource_stability(_samples(), policy=policy)
    repeat = evaluate_resource_stability(_samples(), policy=policy)
    report = evaluate_long_run_closure(
        _TwoYear(True),
        primary,
        repeat,
        policy=LongRunClosurePolicy(),
    )
    assert report.passed


def test_two_year_failure_cannot_be_hidden_by_good_memory():
    policy = ResourceStabilityPolicy(minimum_duration_seconds=60, minimum_progress_bars=1000)
    primary = evaluate_resource_stability(_samples(), policy=policy)
    report = evaluate_long_run_closure(_TwoYear(False), primary, primary)
    assert not report.passed
    assert "LONG_RUN_TWO_YEAR_BACKTEST_NOT_PASSED" in report.reasons
