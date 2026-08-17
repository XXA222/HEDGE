"""Explicit semantic translators into the canonical :class:`PolicyTarget`.

These translators deliberately consume policy outputs, not orders.  They are the
only supported bridge from HPRL/Risk-Level targets into planning's shared target
contract and make unit conversion visible at the boundary.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from freqtrade.hedge.contracts import PolicySourceKind, PolicyTarget


def _decimal(value: object) -> Decimal:
    return Decimal(str(value))


def hprl_projection_to_policy_target(
    projection: object,
    *,
    account_id: str,
    equity: Decimal,
    leverage: Decimal,
    expires_at: datetime,
    source_authority_sha256: str,
    risk_policy_sha256: str,
    feature_fingerprint_sha256: str,
    uncertainty: Decimal = Decimal(0),
) -> PolicyTarget:
    """Translate an HPRL projection without guessing margin/notional units."""

    accepted = getattr(projection, "accepted")
    if not isinstance(accepted, bool):
        raise TypeError("HPRL projection accepted flag must be bool")
    long_margin = _decimal(getattr(projection, "long_margin_ratio")) if accepted else Decimal(0)
    short_margin = _decimal(getattr(projection, "short_margin_ratio")) if accepted else Decimal(0)
    long_notional_ratio = _decimal(getattr(projection, "long_notional_ratio"))
    short_notional_ratio = _decimal(getattr(projection, "short_notional_ratio"))
    if accepted and (
        long_notional_ratio != long_margin * leverage
        or short_notional_ratio != short_margin * leverage
    ):
        raise ValueError("HPRL projection margin/notional semantics do not match leverage")
    sequence = getattr(projection, "sequence")
    return PolicyTarget(
        account_id=account_id,
        symbol=getattr(projection, "symbol"),
        decision_id=f"hprl:{sequence}",
        observed_at=getattr(projection, "observed_at"),
        expires_at=expires_at,
        source_kind=PolicySourceKind.HPRL,
        source_id=getattr(projection, "source_sha256"),
        model_id=getattr(projection, "model_id"),
        source_authority_sha256=source_authority_sha256,
        risk_policy_sha256=risk_policy_sha256,
        feature_fingerprint_sha256=feature_fingerprint_sha256,
        equity=equity,
        long_margin_fraction=long_margin,
        short_margin_fraction=short_margin,
        long_target_notional=equity * long_margin * leverage,
        short_target_notional=equity * short_margin * leverage,
        long_leverage=leverage,
        short_leverage=leverage,
        confidence=_decimal(getattr(projection, "confidence")),
        uncertainty=uncertainty,
        risk_budget_multiplier=Decimal(1),
        allow_new_risk=accepted,
        pause_entry=not accepted,
        reason=";".join(getattr(projection, "reasons")) or "HPRL",
    )


def risk_level_signal_to_policy_target(
    signal: object,
    *,
    account_id: str,
    symbol: str,
    observed_at: datetime,
    expires_at: datetime,
    model_id: str,
    source_authority_sha256: str,
    risk_policy_sha256: str,
    feature_fingerprint_sha256: str,
    long_leverage: Decimal,
    short_leverage: Decimal,
    confidence: Decimal = Decimal(1),
    uncertainty: Decimal = Decimal(0),
) -> PolicyTarget:
    """Translate a masked Risk-Level signal using margin budgets as its source unit."""

    allow_new_risk = getattr(signal, "allow_new_risk")
    if not isinstance(allow_new_risk, bool):
        raise TypeError("Risk-Level signal allow_new_risk must be bool")
    long_margin = _decimal(getattr(signal, "long_margin_fraction")) if allow_new_risk else Decimal(0)
    short_margin = _decimal(getattr(signal, "short_margin_fraction")) if allow_new_risk else Decimal(0)
    equity = _decimal(getattr(signal, "target_equity"))
    return PolicyTarget(
        account_id=account_id,
        symbol=symbol,
        decision_id=f"risk-level:{getattr(signal, 'requested_joint_id')}",
        observed_at=observed_at,
        expires_at=expires_at,
        source_kind=PolicySourceKind.RISK_LEVEL_RL,
        source_id=getattr(signal, "action_signature"),
        model_id=model_id,
        source_authority_sha256=source_authority_sha256,
        risk_policy_sha256=risk_policy_sha256,
        feature_fingerprint_sha256=feature_fingerprint_sha256,
        equity=equity,
        long_margin_fraction=long_margin,
        short_margin_fraction=short_margin,
        long_target_notional=equity * long_margin * long_leverage,
        short_target_notional=equity * short_margin * short_leverage,
        long_leverage=long_leverage,
        short_leverage=short_leverage,
        confidence=confidence,
        uncertainty=uncertainty,
        risk_budget_multiplier=Decimal(1),
        allow_new_risk=allow_new_risk,
        pause_entry=not allow_new_risk,
        reason=getattr(signal, "reason"),
    )
