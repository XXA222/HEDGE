from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from freqtrade.hedge.contracts import PolicySourceKind, PolicyTarget


HASH = "a" * 64


def _target(**changes: object) -> PolicyTarget:
    values: dict[str, object] = {
        "account_id": "acct", "symbol": "btc/usdt:usdt", "decision_id": "decision-1",
        "observed_at": datetime(2026, 8, 17, tzinfo=UTC),
        "expires_at": datetime(2026, 8, 17, 0, 0, 5, tzinfo=UTC),
        "source_kind": PolicySourceKind.RISK_LEVEL_RL, "source_id": "risk-level-v3",
        "model_id": "model-1", "source_authority_sha256": HASH,
        "risk_policy_sha256": HASH, "feature_fingerprint_sha256": HASH,
        "equity": Decimal("1000"), "long_margin_fraction": Decimal("0.10"),
        "short_margin_fraction": Decimal("0.05"), "long_target_notional": Decimal("300"),
        "short_target_notional": Decimal("100"), "long_leverage": Decimal("3"),
        "short_leverage": Decimal("2"), "confidence": Decimal("0.8"),
        "uncertainty": Decimal("0.1"), "risk_budget_multiplier": Decimal("1"),
        "allow_new_risk": True, "pause_entry": False, "reason": "unit-test",
    }
    values.update(changes)
    return PolicyTarget(**values)  # type: ignore[arg-type]


def test_policy_target_is_exact_directional_narrow_waist() -> None:
    target = _target()
    assert target.symbol == "BTC/USDT:USDT"
    assert target.gross_margin_fraction == Decimal("0.15")
    assert target.net_target_notional == Decimal("200")
    assert len(target.fingerprint) == 64


def test_policy_target_rejects_notional_unit_mismatch() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        _target(long_target_notional=Decimal("301"))


def test_policy_target_requires_expiry_and_hash_provenance() -> None:
    with pytest.raises(ValueError, match="expires_at"):
        _target(expires_at=datetime(2026, 8, 17, tzinfo=UTC))
    with pytest.raises(ValueError, match="sha256"):
        _target(risk_policy_sha256="bad")
