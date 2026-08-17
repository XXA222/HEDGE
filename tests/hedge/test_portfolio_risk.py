from decimal import Decimal

from freqtrade.hedge.risk.portfolio_v2 import AssetRiskExposure, aggregate_portfolio_risk


def test_multi_asset_risk_aggregates_correlation_and_concentration() -> None:
    rows = (AssetRiskExposure("BTC", Decimal(100), Decimal(120), Decimal(20)), AssetRiskExposure("ETH", Decimal(-50), Decimal(80), Decimal(10)))
    result = aggregate_portfolio_risk(rows, {("BTC", "ETH"): Decimal("0.5")})
    assert result.gross_notional == 200
    assert result.concentration_ratio == Decimal("0.6")
    assert result.correlated_variance == Decimal(7500)
