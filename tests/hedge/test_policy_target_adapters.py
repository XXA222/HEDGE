from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from freqtrade.hedge.integration.policy_target_adapter import hprl_projection_to_policy_target
from freqtrade.hedge.production.hprl_hedge_adapter import HprlTargetProjection


HASH = "d" * 64
NOW = datetime(2026, 8, 17, tzinfo=UTC)


def test_hprl_adapter_preserves_explicit_margin_to_notional_semantics() -> None:
    projection = HprlTargetProjection(
        sequence=3, observed_at=NOW, symbol="BTC/USDT:USDT", model_id="hprl-v3",
        long_margin_ratio=Decimal("0.1"), short_margin_ratio=Decimal("0.05"),
        long_notional_ratio=Decimal("0.3"), short_notional_ratio=Decimal("0.15"),
        confidence=Decimal("0.8"), accepted=True, reasons=("MODEL_OK",), source_sha256=HASH,
    )
    target = hprl_projection_to_policy_target(
        projection, account_id="acct", equity=Decimal("1000"), leverage=Decimal("3"),
        expires_at=NOW + timedelta(seconds=5), source_authority_sha256=HASH,
        risk_policy_sha256=HASH, feature_fingerprint_sha256=HASH,
    )
    assert target.long_target_notional == Decimal("300")
    assert target.short_target_notional == Decimal("150")
