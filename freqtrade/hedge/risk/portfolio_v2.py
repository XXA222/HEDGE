"""Deterministic multi-asset portfolio risk aggregation."""

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json

from freqtrade.hedge.contracts import finite_decimal


@dataclass(frozen=True, slots=True)
class AssetRiskExposure:
    symbol: str
    net_notional: Decimal
    gross_notional: Decimal
    margin: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol is required")
        for name in ("net_notional", "gross_notional", "margin"):
            value = finite_decimal(getattr(self, name), field_name=name)
            if name != "net_notional" and value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if abs(self.net_notional) > self.gross_notional:
            raise ValueError("absolute net cannot exceed gross")


@dataclass(frozen=True, slots=True)
class PortfolioRiskV2:
    gross_notional: Decimal
    net_notional: Decimal
    total_margin: Decimal
    concentration_ratio: Decimal
    correlated_variance: Decimal
    fingerprint: str


def aggregate_portfolio_risk(exposures: tuple[AssetRiskExposure, ...], correlations: dict[tuple[str, str], Decimal]) -> PortfolioRiskV2:
    if not exposures or len({item.symbol for item in exposures}) != len(exposures):
        raise ValueError("unique nonempty exposures are required")
    gross = sum((item.gross_notional for item in exposures), Decimal(0))
    net = sum((item.net_notional for item in exposures), Decimal(0))
    margin = sum((item.margin for item in exposures), Decimal(0))
    concentration = max(item.gross_notional for item in exposures) / gross if gross else Decimal(0)
    variance = Decimal(0)
    for left in exposures:
        for right in exposures:
            key = (left.symbol, right.symbol)
            reverse = (right.symbol, left.symbol)
            corr = Decimal(1) if left.symbol == right.symbol else finite_decimal(correlations.get(key, correlations.get(reverse, Decimal(0))), field_name="correlation")
            if not Decimal(-1) <= corr <= Decimal(1):
                raise ValueError("correlation must be within [-1,1]")
            variance += left.net_notional * right.net_notional * corr
    payload = [(item.symbol, str(item.net_notional), str(item.gross_notional), str(item.margin)) for item in exposures]
    fingerprint = sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
    return PortfolioRiskV2(gross, net, margin, concentration, max(variance, Decimal(0)), fingerprint)
