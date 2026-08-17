"""Unified benchmark-tower qualification across deterministic, ML and RL families."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from freqtrade.hedge.contracts import finite_decimal


class BenchmarkFamily(StrEnum):
    DETERMINISTIC = "B0_DETERMINISTIC"
    SUPERVISED_ML = "B1_SUPERVISED_ML"
    HEDGE_RL_21 = "B2_HEDGE_RL_21"
    RISK_LEVEL_RL = "B3_RISK_LEVEL_RL"
    HPRL_FAST_TD3 = "B4_HPRL_FAST_TD3"
    HPRL_CHALLENGER = "B5_HPRL_CHALLENGER"


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    family: BenchmarkFamily
    protocol_sha256: str
    seeds: tuple[int, ...]
    net_return: Decimal
    max_drawdown: Decimal
    expected_shortfall: Decimal
    passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.family, BenchmarkFamily) or not isinstance(self.passed, bool):
            raise TypeError("family/passed have invalid types")
        digest = self.protocol_sha256.lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("protocol_sha256 must be sha256")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("unique nonempty seeds required")
        for name in ("net_return", "max_drawdown", "expected_shortfall"):
            object.__setattr__(self, name, finite_decimal(getattr(self, name), field_name=name))


def qualify_benchmark_tower(results: tuple[BenchmarkResult, ...]) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    families = {row.family for row in results}
    missing = set(BenchmarkFamily) - families
    reasons.extend(f"MISSING:{family.value}" for family in sorted(missing, key=lambda item: item.value))
    if len(families) != len(results):
        reasons.append("DUPLICATE_BENCHMARK_FAMILY")
    protocols = {row.protocol_sha256 for row in results}
    seeds = {row.seeds for row in results}
    if len(protocols) > 1:
        reasons.append("PROTOCOL_MISMATCH")
    if len(seeds) > 1:
        reasons.append("SEED_SET_MISMATCH")
    reasons.extend(f"FAILED:{row.family.value}" for row in results if not row.passed)
    return not reasons, tuple(reasons)
