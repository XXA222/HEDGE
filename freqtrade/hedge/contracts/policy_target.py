"""Canonical, order-free target-exposure contract for every Hedge policy.

Policies may disagree on how they select a target; they must not disagree on what a
target means.  This narrow waist carries exact directional margin/notional targets to
the planner while the Hedge risk engine remains the only live hard authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
import json

from .types import canonical_symbol, finite_decimal, required_text


class PolicySourceKind(StrEnum):
    RISK_LEVEL_RL = "RISK_LEVEL_RL"
    HPRL = "HPRL"
    HEDGE_RL = "HEDGE_RL"
    SUPERVISED = "SUPERVISED"
    DETERMINISTIC = "DETERMINISTIC"


def _nonnegative(value: object, *, name: str) -> Decimal:
    result = finite_decimal(value, field_name=name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _ratio(value: object, *, name: str) -> Decimal:
    result = _nonnegative(value, name=name)
    if result > Decimal(1):
        raise ValueError(f"{name} must be within [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class PolicyTarget:
    """Final policy target before planner conversion, never an order instruction."""

    account_id: str
    symbol: str
    decision_id: str
    observed_at: datetime
    expires_at: datetime
    source_kind: PolicySourceKind
    source_id: str
    model_id: str
    source_authority_sha256: str
    risk_policy_sha256: str
    feature_fingerprint_sha256: str
    equity: Decimal
    long_margin_fraction: Decimal
    short_margin_fraction: Decimal
    long_target_notional: Decimal
    short_target_notional: Decimal
    long_leverage: Decimal
    short_leverage: Decimal
    confidence: Decimal
    uncertainty: Decimal
    risk_budget_multiplier: Decimal
    allow_new_risk: bool
    pause_entry: bool
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "account_id", required_text(self.account_id, field_name="account_id", max_length=128))
        object.__setattr__(self, "symbol", canonical_symbol(self.symbol))
        for name, maximum in (("decision_id", 128), ("source_id", 128), ("model_id", 128), ("reason", 512)):
            object.__setattr__(self, name, required_text(getattr(self, name), field_name=name, max_length=maximum))
        if not isinstance(self.source_kind, PolicySourceKind):
            raise TypeError("source_kind must be PolicySourceKind")
        for name in ("source_authority_sha256", "risk_policy_sha256", "feature_fingerprint_sha256"):
            value = required_text(getattr(self, name), field_name=name, max_length=64).lower()
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be sha256")
            object.__setattr__(self, name, value)
        for name in ("observed_at", "expires_at"):
            value = getattr(self, name)
            if not isinstance(value, datetime):
                raise TypeError(f"{name} must be datetime")
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
            object.__setattr__(self, name, value.astimezone(UTC))
        if self.expires_at <= self.observed_at:
            raise ValueError("expires_at must be after observed_at")
        for name in ("equity", "long_target_notional", "short_target_notional", "long_leverage", "short_leverage"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name=name))
        if self.equity <= 0 or self.long_leverage <= 0 or self.short_leverage <= 0:
            raise ValueError("equity and leverages must be positive")
        for name in ("long_margin_fraction", "short_margin_fraction", "confidence", "uncertainty"):
            object.__setattr__(self, name, _ratio(getattr(self, name), name=name))
        object.__setattr__(self, "risk_budget_multiplier", _nonnegative(self.risk_budget_multiplier, name="risk_budget_multiplier"))
        if not isinstance(self.allow_new_risk, bool) or not isinstance(self.pause_entry, bool):
            raise TypeError("allow_new_risk and pause_entry must be bool")
        expected_long = self.equity * self.long_margin_fraction * self.long_leverage
        expected_short = self.equity * self.short_margin_fraction * self.short_leverage
        if self.long_target_notional != expected_long or self.short_target_notional != expected_short:
            raise ValueError("target notionals must exactly match equity × margin fraction × leverage")
        if (self.pause_entry or not self.allow_new_risk) and (self.long_margin_fraction or self.short_margin_fraction):
            # A reduce-only target may retain current risk only; it cannot request a
            # positive target because current holdings are intentionally absent here.
            raise ValueError("paused or no-new-risk PolicyTarget must be flat")

    @property
    def gross_margin_fraction(self) -> Decimal:
        return self.long_margin_fraction + self.short_margin_fraction

    @property
    def net_target_notional(self) -> Decimal:
        return self.long_target_notional - self.short_target_notional

    @property
    def fingerprint(self) -> str:
        payload = {
            name: (getattr(self, name).isoformat() if isinstance(getattr(self, name), datetime) else str(getattr(self, name)))
            for name in self.__dataclass_fields__
        }
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
