"""Stable identity for the unique live Hedge hard-risk policy."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str")
    result = value.strip().lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be sha256")
    return result


def canonical_sha256(value: object) -> str:
    """Hash a policy component using a deterministic, schema-visible encoding."""
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class RiskPolicyIdentity:
    schema_version: str
    policy_name: str
    limits_sha256: str
    risk_level_profile_sha256: str
    reservation_policy_sha256: str
    unknown_order_policy_sha256: str
    kill_switch_policy_sha256: str

    def __post_init__(self) -> None:
        for name in ("schema_version", "policy_name"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be str")
            if not value.strip():
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value.strip())
        for name in (
            "limits_sha256",
            "risk_level_profile_sha256",
            "reservation_policy_sha256",
            "unknown_order_policy_sha256",
            "kill_switch_policy_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256({name: getattr(self, name) for name in self.__dataclass_fields__})

    @classmethod
    def from_components(
        cls,
        *,
        schema_version: str,
        policy_name: str,
        limits: object,
        risk_level_profile: object,
        reservation_policy: object,
        unknown_order_policy: object,
        kill_switch_policy: object,
    ) -> RiskPolicyIdentity:
        return cls(
            schema_version=schema_version,
            policy_name=policy_name,
            limits_sha256=canonical_sha256(limits),
            risk_level_profile_sha256=canonical_sha256(risk_level_profile),
            reservation_policy_sha256=canonical_sha256(reservation_policy),
            unknown_order_policy_sha256=canonical_sha256(unknown_order_policy),
            kill_switch_policy_sha256=canonical_sha256(kill_switch_policy),
        )
