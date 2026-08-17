from decimal import Decimal

from freqtrade.hedge.research.benchmark_tower import BenchmarkFamily, BenchmarkResult, qualify_benchmark_tower


def test_full_same_protocol_benchmark_tower_qualifies() -> None:
    rows = tuple(BenchmarkResult(family, "a" * 64, (1, 2, 3), Decimal(1), Decimal("0.1"), Decimal("0.2"), True) for family in BenchmarkFamily)
    assert qualify_benchmark_tower(rows) == (True, ())


def test_missing_family_fails_closed() -> None:
    row = BenchmarkResult(BenchmarkFamily.DETERMINISTIC, "a" * 64, (1,), Decimal(1), Decimal(0), Decimal(0), True)
    passed, reasons = qualify_benchmark_tower((row,))
    assert not passed
    assert any(reason.startswith("MISSING:") for reason in reasons)
