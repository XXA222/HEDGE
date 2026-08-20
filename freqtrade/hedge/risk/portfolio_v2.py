"""Deterministic multi-asset portfolio risk aggregation."""

from dataclasses import dataclass
from decimal import Decimal

from freqtrade.hedge.contracts import finite_decimal
from freqtrade.hedge.risk.policy_identity import canonical_sha256


@dataclass(frozen=True, slots=True)
class AssetRiskExposure:
    symbol: str
    net_notional: Decimal
    gross_notional: Decimal
    margin: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol is required")
        object.__setattr__(self, "symbol", self.symbol.strip())
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


def _decimal_identity(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _canonical_correlation_matrix(
    exposures: tuple[AssetRiskExposure, ...],
    correlations: dict[tuple[str, str], Decimal],
) -> dict[tuple[str, str], Decimal]:
    symbols = tuple(sorted(item.symbol for item in exposures))
    result: dict[tuple[str, str], Decimal] = {}
    for left_index, left in enumerate(symbols):
        for right in symbols[left_index + 1 :]:
            direct = correlations.get((left, right))
            reverse = correlations.get((right, left))
            if direct is None and reverse is None:
                value = Decimal(0)
            elif direct is None:
                value = finite_decimal(reverse, field_name="correlation")
            elif reverse is None:
                value = finite_decimal(direct, field_name="correlation")
            else:
                direct_value = finite_decimal(direct, field_name="correlation")
                reverse_value = finite_decimal(reverse, field_name="correlation")
                if direct_value != reverse_value:
                    raise ValueError(
                        f"conflicting correlations for {left}/{right}: "
                        f"{direct_value} != {reverse_value}"
                    )
                value = direct_value
            if not Decimal(-1) <= value <= Decimal(1):
                raise ValueError("correlation must be within [-1,1]")
            result[(left, right)] = value
    return result


def aggregate_portfolio_risk(
    exposures: tuple[AssetRiskExposure, ...],
    correlations: dict[tuple[str, str], Decimal],
) -> PortfolioRiskV2:
    if not exposures or len({item.symbol for item in exposures}) != len(exposures):
        raise ValueError("unique nonempty exposures are required")

    ordered = tuple(sorted(exposures, key=lambda item: item.symbol))
    canonical_correlations = _canonical_correlation_matrix(ordered, correlations)
    gross = sum((item.gross_notional for item in ordered), Decimal(0))
    net = sum((item.net_notional for item in ordered), Decimal(0))
    margin = sum((item.margin for item in ordered), Decimal(0))
    concentration = (
        max(item.gross_notional for item in ordered) / gross if gross else Decimal(0)
    )

    variance = Decimal(0)
    for left in ordered:
        for right in ordered:
            if left.symbol == right.symbol:
                correlation = Decimal(1)
            else:
                key = tuple(sorted((left.symbol, right.symbol)))
                correlation = canonical_correlations[key]
            variance += left.net_notional * right.net_notional * correlation

    payload = {
        "schema": "portfolio-risk-v2",
        "exposures": [
            (
                item.symbol,
                _decimal_identity(item.net_notional),
                _decimal_identity(item.gross_notional),
                _decimal_identity(item.margin),
            )
            for item in ordered
        ],
        "correlations": [
            (left, right, _decimal_identity(value))
            for (left, right), value in sorted(canonical_correlations.items())
        ],
    }
    fingerprint = canonical_sha256(payload)
    return PortfolioRiskV2(
        gross,
        net,
        margin,
        concentration,
        max(variance, Decimal(0)),
        fingerprint,
    )
